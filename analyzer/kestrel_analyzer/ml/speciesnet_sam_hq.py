"""SpeciesNet (MegaDetector + classifier + ensemble) + SAM-HQ segmentation for the analysis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from ..config import SAM_HQ_MODEL_KEY, SAM_HQ_WEIGHTS_PATH
from .speciesnet_taxonomy import route_prediction

_DEFAULT_MAX_BIRD_CROPS = 5
_MIN_MAX_BIRD_CROPS = 1
_MAX_MAX_BIRD_CROPS = 20


def _coerce_max_bird_crops(value) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = _DEFAULT_MAX_BIRD_CROPS
    return max(_MIN_MAX_BIRD_CROPS, min(_MAX_MAX_BIRD_CROPS, n))


def filter_overlapping_detections(masks, pred_boxes, pred_class, pred_score, iou_threshold=0.5):
    """Remove lower-confidence detections that overlap significantly with higher-confidence ones."""
    if masks is None or len(masks) == 0:
        return masks, pred_boxes, pred_class, pred_score

    n = len(pred_score)
    keep = [True] * n
    sorted_indices = sorted(range(n), key=lambda i: pred_score[i], reverse=True)

    for i_idx, i in enumerate(sorted_indices):
        if not keep[i]:
            continue
        for j in sorted_indices[i_idx + 1 :]:
            if not keep[j]:
                continue
            intersection = np.logical_and(masks[i], masks[j]).sum()
            union = np.logical_or(masks[i], masks[j]).sum()
            if union > 0 and intersection / union > iou_threshold:
                keep[j] = False

    indices = [i for i in range(n) if keep[i]]
    if not indices:
        return masks, pred_boxes, pred_class, pred_score

    return (
        masks[indices],
        [pred_boxes[i] for i in indices],
        [pred_class[i] for i in indices],
        [pred_score[i] for i in indices],
    )


def _md_bbox_to_pixel_box(md_bbox: list, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """MegaDetector normalized xywh -> pixel xyxy corners."""
    x_min_n, y_min_n, bw_n, bh_n = [float(v) for v in md_bbox]
    x1 = np.clip(x_min_n * img_w, 0, img_w)
    y1 = np.clip(y_min_n * img_h, 0, img_h)
    x2 = np.clip((x_min_n + bw_n) * img_w, 0, img_w)
    y2 = np.clip((y_min_n + bh_n) * img_h, 0, img_h)
    return x1, y1, x2, y2


def _pixel_box_to_pipeline_box(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Pipeline expects ``pred_boxes[i]`` as ``((xmin, ymin), (xmax, ymax))`` in pixels."""
    return (float(x1), float(y1)), (float(x2), float(y2))


def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1i = int(max(0, min(w - 1, x1)))
    y1i = int(max(0, min(h - 1, y1)))
    x2i = int(max(0, min(w, x2)))
    y2i = int(max(0, min(h, y2)))
    if x2i <= x1i:
        x2i = min(w, x1i + 1)
    if y2i <= y1i:
        y2i = min(h, y1i + 1)
    return x1i, y1i, x2i, y2i


class SpeciesNetSAMHQWrapper:
    """Detector/classifier/ensemble from SpeciesNet; masks from SAM-HQ (default ViT-Tiny) box prompts."""

    def __init__(self, max_bird_crops: int = _DEFAULT_MAX_BIRD_CROPS, use_gpu: bool = True):
        self.max_bird_crops = _coerce_max_bird_crops(max_bird_crops)
        self.use_gpu = bool(use_gpu)
        self._sam_force_cpu: Optional[bool] = None
        self.predictor = None
        self.detector = None
        self.classifier = None
        self.ensemble = None
        self.model_name: Optional[str] = None
        self._sam_checkpoint = Path(SAM_HQ_WEIGHTS_PATH)

    def _ensure_speciesnet(self) -> None:
        from speciesnet import (
            DEFAULT_MODEL,
            SpeciesNetClassifier,
            SpeciesNetDetector,
            SpeciesNetEnsemble,
        )

        if self.detector is None or self.classifier is None:
            self.model_name = DEFAULT_MODEL
            self.detector = SpeciesNetDetector(self.model_name)
            device = "cpu" if not self.use_gpu else self.detector.device
            self.classifier = SpeciesNetClassifier(self.model_name, device=device)
        if self.ensemble is None:
            self.ensemble = SpeciesNetEnsemble(self.model_name, geofence=False)

    def _ensure_sam(self) -> None:
        from segment_anything_hq import SamPredictor, sam_model_registry

        force_cpu = not self.use_gpu
        need_new = self.predictor is None or self._sam_force_cpu != force_cpu
        if not need_new:
            return
        if not self._sam_checkpoint.exists():
            raise FileNotFoundError(
                f"SAM-HQ weights not found at: {self._sam_checkpoint}\n"
                "Place sam_hq_vit_tiny.pth under analyzer/models/ (see config.SAM_HQ_WEIGHTS_PATH)."
            )
        original_torch_load = torch.load

        def _safe_torch_load(*args, **kwargs):
            if force_cpu:
                kwargs.setdefault("map_location", "cpu")
            return original_torch_load(*args, **kwargs)

        torch.load = _safe_torch_load
        try:
            sam = sam_model_registry[SAM_HQ_MODEL_KEY](checkpoint=str(self._sam_checkpoint))
        finally:
            torch.load = original_torch_load
        if force_cpu and hasattr(sam, "to"):
            sam.to("cpu")
        sam.eval()
        self.predictor = SamPredictor(sam)
        self._sam_force_cpu = force_cpu

    def _run_ensemble_for_item(
        self,
        filepath: str,
        classifications: dict[str, Any],
        detections: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.ensemble is None:
            raise RuntimeError("SpeciesNet ensemble is not loaded.")
        results = self.ensemble.combine(
            filepaths=[filepath],
            classifier_results={filepath: {"classifications": classifications}},
            detector_results={filepath: {"detections": detections}},
            geolocation_results={filepath: {}},
            partial_predictions={},
        )
        return results[0] if results else {}

    def get_prediction(
        self,
        image_data: np.ndarray,
        image_path: str | Path,
        *,
        wildlife_enabled: bool = True,
        threshold: float = 0.75,
        mask_threshold: float = 0.5,
    ):
        """Run SpeciesNet + SAM-HQ.

        Args:
            image_data: RGB uint8 image.
            image_path: Path passed to SpeciesNet (must exist on disk).
            wildlife_enabled: When False, non-aves animals are omitted.
            threshold: MegaDetector minimum confidence for an ``animal`` detection.
            mask_threshold: Unused (legacy Mask R-CNN pixel threshold); retained for API compatibility.

        Returns:
            (masks, pred_boxes, pred_class, pred_score) — same contract as ``MaskRCNNWrapper``.
        """
        _ = mask_threshold  # SAM-HQ path does not use Mask R-CNN mask pixel threshold; UI keeps knob for compatibility.

        self._ensure_speciesnet()
        self._ensure_sam()

        from speciesnet.utils import BBox

        fp = str(Path(image_path).resolve())
        h, w = image_data.shape[:2]
        img_pil = Image.fromarray(image_data)

        detector_input = self.detector.preprocess(img_pil)
        det_result = self.detector.predict(fp, detector_input)
        detections = det_result.get("detections", []) or []

        print("[SpeciesNet] detector raw:", det_result)
        print("[SpeciesNet] image:", fp, "detections count:", len(detections))

        animal_dets: list[dict[str, Any]] = []
        for det in detections:
            label = str(det.get("label", ""))
            conf = float(det.get("conf", 0.0))
            if label != "animal":
                continue
            if conf < float(threshold):
                continue
            animal_dets.append(det)

        animal_dets.sort(key=lambda d: float(d.get("conf", 0.0)), reverse=True)

        bird_rows: list[dict[str, Any]] = []
        wildlife_rows: list[dict[str, Any]] = []

        if self.predictor is None:
            return [], [], [], []

        self.predictor.set_image(image_data)

        for det_idx, det in enumerate(animal_dets):
            md_bbox = det.get("bbox", [0.0, 0.0, 0.0, 0.0])
            label = str(det.get("label", "animal"))
            conf = float(det.get("conf", 0.0))

            cls_input = self.classifier.preprocess(img_pil, bboxes=[BBox(*md_bbox)])
            cls_pred = self.classifier.predict(fp, cls_input)
            cls_info = cls_pred.get("classifications", {})

            fp_det = f"{fp}#det{det_idx}"
            try:
                ensemble_det = self._run_ensemble_for_item(
                    filepath=fp_det,
                    classifications=cls_info,
                    detections=[{"label": label, "conf": conf}],
                )
                pred_raw = str(ensemble_det.get("prediction", ""))
                pred_score = float(ensemble_det.get("prediction_score", conf))
                pred_source = str(ensemble_det.get("prediction_source", ""))
            except Exception as e:
                print("[SpeciesNet] ensemble error, fallback to classifier top-1:", e)
                classes = cls_info.get("classes", [])
                scores = cls_info.get("scores", [])
                pred_raw = str(classes[0]) if classes else "unknown"
                pred_score = float(scores[0]) if scores else conf
                pred_source = "classifier_fallback"

            print(
                "[SpeciesNet] det",
                det_idx,
                "md_bbox",
                md_bbox,
                "ensemble pred:",
                pred_raw,
                "score",
                pred_score,
                "source",
                pred_source,
            )

            route, pred_label = route_prediction(pred_raw, wildlife_enabled=wildlife_enabled)
            print("[SpeciesNet] route:", route, "pred_label:", pred_label)

            if route == "ignore" or pred_label is None:
                continue

            x1, y1, x2, y2 = _md_bbox_to_pixel_box(md_bbox, w, h)
            xi1, yi1, xi2, yi2 = _clip_xyxy(x1, y1, x2, y2, w, h)

            try:
                masks_out, _mask_scores, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.array([xi1, yi1, xi2, yi2], dtype=np.float32),
                    multimask_output=False,
                    hq_token_only=True,
                )
                mask = masks_out[0].astype(bool)
            except Exception as e:
                print("[SAM-HQ] mask failed:", e)
                continue

            row = {
                "mask": mask,
                "pred_boxes": _pixel_box_to_pipeline_box(x1, y1, x2, y2),
                "pred_class": pred_label if route == "wildlife" else "bird",
                "pred_score": pred_score,
            }
            if route == "bird":
                bird_rows.append(row)
            else:
                wildlife_rows.append(row)

        bird_rows = bird_rows[: self.max_bird_crops]
        wildlife_rows = wildlife_rows[: self.max_bird_crops]

        combined = bird_rows + wildlife_rows
        if not combined:
            return [], [], [], []

        masks_list = [r["mask"] for r in combined]
        pred_boxes = [r["pred_boxes"] for r in combined]
        pred_class = [r["pred_class"] for r in combined]
        pred_score = [r["pred_score"] for r in combined]

        masks_arr = np.stack(masks_list, axis=0)
        return filter_overlapping_detections(
            masks_arr, pred_boxes, pred_class, pred_score, iou_threshold=0.5
        )

    # --- Geometry (aligned with MaskRCNNWrapper) for square crops and species bbox ---

    @staticmethod
    def _center_of_mass(mask):
        y, x = np.where(mask > 0)
        return (int(np.mean(x)), int(np.mean(y)))

    @staticmethod
    def _fsolve(func, xmin, xmax):
        x_min, x_max = xmin, xmax
        while x_max - x_min > 10:
            x_mid = (x_min + x_max) / 2
            if func(x_mid) < 0:
                x_min = x_mid
            else:
                x_max = x_mid
        return (x_min + x_max) / 2

    def _get_bounding_box(self, mask):
        center = self._center_of_mass(mask)

        def fraction_inside(center_of_mass, S):
            x_min = int(center_of_mass[0] - S / 2)
            x_max = int(center_of_mass[0] + S / 2)
            y_min = int(center_of_mass[1] - S / 2)
            y_max = int(center_of_mass[1] + S / 2)
            x_min2 = max(0, x_min)
            x_max2 = min(mask.shape[1], x_max)
            y_min2 = max(0, y_min)
            y_max2 = min(mask.shape[0], y_max)
            return np.sum(mask[y_min2:y_max2, x_min2:x_max2]) / np.sum(mask)

        S = self._fsolve(lambda S: fraction_inside(center, S) - 0.8, 10, 3000)
        S = int(S * 1 / 0.5)
        x_min = int(center[0] - S / 2)
        x_max = int(center[0] + S / 2)
        y_min = int(center[1] - S / 2)
        y_max = int(center[1] + S / 2)
        x_min = max(0, x_min)
        x_max = min(mask.shape[1], x_max)
        y_min = max(0, y_min)
        y_max = min(mask.shape[0], y_max)
        slx = x_max - x_min
        sly = y_max - y_min
        if slx > sly:
            center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
            s_new = sly
        else:
            center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
            s_new = slx
        x_min = int(center[0] - s_new / 2)
        x_max = int(center[0] + s_new / 2)
        y_min = int(center[1] - s_new / 2)
        y_max = int(center[1] + s_new / 2)
        return x_min, x_max, y_min, y_max

    def get_square_crop(self, mask, img, resize=True):
        bbox = self.get_square_crop_box(mask)
        x_min = bbox["x_min"]
        x_max = bbox["x_max"]
        y_min = bbox["y_min"]
        y_max = bbox["y_max"]
        crop = img[y_min:y_max, x_min:x_max]
        mask_crop = mask[y_min:y_max, x_min:x_max]
        if resize:
            crop = cv2.resize(crop, (1024, 1024))
            mask_crop = cv2.resize(mask_crop.astype(np.uint8), (1024, 1024))
        return crop, mask_crop

    def get_square_crop_box(self, mask):
        x_min, x_max, y_min, y_max = self._get_bounding_box(mask)
        h, w = mask.shape[:2]
        x_min = max(0, min(int(x_min), max(0, w - 1)))
        y_min = max(0, min(int(y_min), max(0, h - 1)))
        x_max = max(x_min + 1, min(int(x_max), w))
        y_max = max(y_min + 1, min(int(y_max), h))

        width = x_max - x_min
        height = y_max - y_min
        w_denom = float(max(1, w))
        h_denom = float(max(1, h))
        x_center = x_min + (width / 2.0)
        y_center = y_min + (height / 2.0)

        return {
            "x_min": int(x_min),
            "x_max": int(x_max),
            "y_min": int(y_min),
            "y_max": int(y_max),
            "width": int(width),
            "height": int(height),
            "x_min_norm": float(x_min / w_denom),
            "x_max_norm": float(x_max / w_denom),
            "y_min_norm": float(y_min / h_denom),
            "y_max_norm": float(y_max / h_denom),
            "x_center_norm": float(x_center / w_denom),
            "y_center_norm": float(y_center / h_denom),
        }

    @staticmethod
    def get_species_crop(box, img):
        xmin = int(box[0][0])
        ymin = int(box[0][1])
        xmax = int(box[1][0])
        ymax = int(box[1][1])
        return img[ymin:ymax, xmin:xmax]
