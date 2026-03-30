import json
import os
import time
import warnings
from typing import Callable, Dict, Optional

import cv2
import numpy as np
import pandas as pd

from .config import (
    JPEG_EXTENSIONS,
    RAW_EXTENSIONS,
    SPECIESCLASSIFIER_LABELS,
    SPECIESCLASSIFIER_PATH,
    QUALITYCLASSIFIER_PATH,
    QUALITY_NORMALIZATION_DATA_PATH,
    WILDLIFE_CATEGORIES,
    MODELS_DIR,
    KESTREL_DIR_NAME,
    METADATA_FILENAME,
    VERSION,
)
from .database import (
    load_database,
    save_database,
    load_scenedata,
    save_scenedata,
    build_scenedata_from_database,
    update_scenedata_with_database,
)
from .image_utils import read_image, read_image_for_pipeline
from .ratings import quality_to_rating, get_profile_thresholds
from .similarity import compute_image_similarity_akaze, compute_similarity_timestamp
from .raw_exif import get_capture_time
from .logging_utils import get_log_path, log_event, log_exception, log_warning

try:
    from ..settings_utils import load_persisted_settings
except ImportError:
    try:
        from analyzer.settings_utils import load_persisted_settings
    except ImportError:
        load_persisted_settings = None
from .ml.mask_rcnn import MaskRCNNWrapper
from .ml.bird_species import BirdSpeciesClassifier
from .ml.quality import QualityClassifier


class AnalysisPipeline:
    def __init__(self, use_gpu: bool):
        self.use_gpu = use_gpu
        self.mask_rcnn: Optional[MaskRCNNWrapper] = None
        self.species_clf: Optional[BirdSpeciesClassifier] = None
        self.quality_clf: Optional[QualityClassifier] = None
        self._log_path: Optional[str] = None

    @staticmethod
    def _create_mask_overlay(
        thumbnail: np.ndarray,
        masks: Optional[np.ndarray],
        indices: Optional[list],
        color=(255, 64, 64),
        alpha: float = 0.45,
    ) -> Optional[np.ndarray]:
        if thumbnail is None:
            return None
        overlay = thumbnail.copy()
        if masks is None or not indices:
            return overlay
        h, w = overlay.shape[:2]
        for i in indices:
            mask = masks[i].astype(np.uint8)
            mask_small = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask_small.astype(bool)
            if not np.any(mask_bool):
                continue
            overlay[mask_bool] = (
                overlay[mask_bool] * (1.0 - alpha) + np.array(color, dtype=np.uint8) * alpha
            ).astype(np.uint8)
        return overlay

    @staticmethod
    def _compute_exposure_stops(img: np.ndarray, mask: np.ndarray, profile: str = "normal") -> float:
        """Estimate exposure correction in stops for the masked subject region.

        This is histogram-aware (percentile based), not mean-only.

        The solver first finds the maximum exposure lift that still respects
        highlight and clipping guardrails, then applies profile-specific
        aggressiveness.

        Profile can be "lenient", "normal", or "aggressive".

        Returns a value in [-2.0, +3.0] where positive lifts exposure and
        negative pulls it down.
        """
        EPS = 1e-3
        profile_name = str(profile or "normal").strip().lower()
        if profile_name not in {"lenient", "normal", "aggressive"}:
            profile_name = "normal"

        profile_cfg = {
            "lenient": {
                "TARGET_HI_P95": 0.90,
                "TARGET_HI_P98": 0.97,
                "TARGET_SHADOW_P10": 0.12,
                "CLIP_THRESH": 0.99,
                "MAX_CLIP_RATIO": 0.025,
                "BRIGHTEN_STRENGTH": 0.74,
                "DARKEN_STRENGTH": 0.7,
                "MAX_DARKEN": -1.8,
                "MAX_BRIGHTEN": 1.85,
                "MAX_ERODE_ITERS": 1,
            },
            "normal": {
                "TARGET_HI_P95": 0.86,
                "TARGET_HI_P98": 0.945,
                "TARGET_SHADOW_P10": 0.15,
                "CLIP_THRESH": 0.985,
                "MAX_CLIP_RATIO": 0.012,
                "BRIGHTEN_STRENGTH": 0.88,
                "DARKEN_STRENGTH": 0.85,
                "MAX_DARKEN": -1.9,
                "MAX_BRIGHTEN": 2.45,
                "MAX_ERODE_ITERS": 3,
            },
            "aggressive": {
                "TARGET_HI_P95": 0.76,
                "TARGET_HI_P98": 0.84,
                "TARGET_SHADOW_P10": 0.19,
                "CLIP_THRESH": 0.965,
                "MAX_CLIP_RATIO": 0.0015,
                "BRIGHTEN_STRENGTH": 1.0,
                "DARKEN_STRENGTH": 1.0,
                "MAX_DARKEN": -2.0,
                "MAX_BRIGHTEN": 3.0,
                "MAX_ERODE_ITERS": 5,
            },
        }[profile_name]

        TARGET_HI_P95 = profile_cfg["TARGET_HI_P95"]
        TARGET_HI_P98 = profile_cfg["TARGET_HI_P98"]
        TARGET_SHADOW_P10 = profile_cfg["TARGET_SHADOW_P10"]
        CLIP_THRESH = profile_cfg["CLIP_THRESH"]
        MAX_CLIP_RATIO = profile_cfg["MAX_CLIP_RATIO"]
        BRIGHTEN_STRENGTH = profile_cfg["BRIGHTEN_STRENGTH"]
        DARKEN_STRENGTH = profile_cfg["DARKEN_STRENGTH"]
        MAX_DARKEN = profile_cfg["MAX_DARKEN"]
        MAX_BRIGHTEN = profile_cfg["MAX_BRIGHTEN"]
        MAX_ERODE_ITERS = int(profile_cfg["MAX_ERODE_ITERS"])
        try:
            img_f = img.astype(np.float32) / 255.0
            # Exposure compensation uses a detached mask copy so any erosion
            # done here cannot alter masks used by overlays/crops/classifier.
            mask_bool = np.array(mask, dtype=bool, copy=True) if mask is not None else None
            if (
                mask_bool is not None
                and mask_bool.any()
                and img_f.ndim == 3
                and img_f.shape[:2] == mask_bool.shape
            ):
                # Exclude edge pixels from the exposure histogram because mask
                # boundaries often contain mixed foreground/background values.
                mask_u8 = np.array(mask_bool, dtype=np.uint8, copy=True)
                area = int(np.count_nonzero(mask_u8))
                if area >= 256:
                    erode_iters = 1
                    if area >= 2000:
                        erode_iters += 1
                    if area >= 6000:
                        erode_iters += 1
                    if area >= 12000:
                        erode_iters += 1
                    if area >= 22000:
                        erode_iters += 1
                    erode_iters = min(erode_iters, MAX_ERODE_ITERS)
                    eroded = cv2.erode(mask_u8, np.ones((3, 3), dtype=np.uint8), iterations=erode_iters).astype(bool)
                    if eroded.any():
                        mask_bool = eroded
                pixels = img_f[mask_bool]
            else:
                pixels = img_f.reshape(-1, 3)
            # Perceptual luminance weights (Rec. 709)
            lum = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
            lum = lum[np.isfinite(lum)]
            if lum.size == 0:
                return 0.0
            p10, p95, p98 = np.percentile(lum, [10, 95, 98])

            # Find maximum stop value that still satisfies clipped-pixel ratio.
            def _clip_ratio_after(stops_val: float) -> float:
                scale = 2.0 ** float(stops_val)
                return float(np.mean((lum * scale) >= CLIP_THRESH))

            if _clip_ratio_after(MAX_DARKEN) > MAX_CLIP_RATIO:
                clip_ceiling = MAX_DARKEN
            elif _clip_ratio_after(MAX_BRIGHTEN) <= MAX_CLIP_RATIO:
                clip_ceiling = MAX_BRIGHTEN
            else:
                lo = MAX_DARKEN
                hi = MAX_BRIGHTEN
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    if _clip_ratio_after(mid) <= MAX_CLIP_RATIO:
                        lo = mid
                    else:
                        hi = mid
                clip_ceiling = lo

            hi95_stop = float(np.log2(TARGET_HI_P95 / max(float(p95), EPS)))
            hi98_stop = float(np.log2(TARGET_HI_P98 / max(float(p98), EPS)))
            highlight_ceiling = min(hi95_stop, hi98_stop, clip_ceiling, MAX_BRIGHTEN)

            # Ensure dark subjects are still lifted when highlight headroom allows.
            shadow_push_stop = float(np.log2(TARGET_SHADOW_P10 / max(float(p10), EPS)))

            if highlight_ceiling >= 0.0:
                stops = highlight_ceiling * BRIGHTEN_STRENGTH
                if shadow_push_stop > 0.0:
                    stops = max(stops, min(shadow_push_stop, highlight_ceiling))
            else:
                # Negative stops darken bright subjects; aggressive mode uses full
                # darken strength while other profiles back off.
                stops = highlight_ceiling * DARKEN_STRENGTH

            stops = float(np.clip(stops, MAX_DARKEN, MAX_BRIGHTEN))

            if not np.isfinite(stops):
                return 0.0
            return stops
        except Exception:
            return 0.0

    @staticmethod
    def _refine_exposure_stops(
        img: np.ndarray,
        mask: np.ndarray,
        initial_stops: float,
        profile: str,
        raw_obj=None,
    ) -> float:
        """Iteratively refine exposure stops for residual clipping/highlights.

        This is used as a safety pass for aggressive profile darkening where
        one-shot estimation can under-correct very hot highlights.
        """
        total = float(initial_stops)
        profile_name = str(profile or "normal").strip().lower()
        if profile_name != "aggressive":
            return total
        # Refinement is only needed when we're already in highlight-recovery mode.
        if total >= -0.05:
            return total

        for _ in range(2):
            corrected = AnalysisPipeline._apply_exposure_correction(img, total, raw_obj)
            residual = AnalysisPipeline._compute_exposure_stops(corrected, mask, profile_name)
            # Only accumulate additional darkening from residual highlight pressure.
            if not np.isfinite(residual) or residual >= -0.03:
                break
            total += residual * 0.85
            total = float(np.clip(total, -2.0, 3.0))
            if total <= -1.95:
                break
        return total

    @staticmethod
    def _apply_exposure_correction(
        img: np.ndarray, stops: float, raw_obj=None
    ) -> np.ndarray:
        """Return a copy of *img* with exposure shifted by *stops* stops.

        When *raw_obj* is a live rawpy.RawPy instance (RAW files), the
        correction is applied via rawpy.postprocess(exp_shift=…) in the linear
        domain before demosaicing, which correctly recovers highlight detail
        on overexposed subjects rather than clipping.

        For non-RAW sources (JPEG/PNG, or if rawpy re-process fails) the
        method falls back to pixel-level multiplication.

        Positive stops lifts (brightens), negative stops pulls (darkens).
        Returns the original array unchanged when stops == 0.0.
        """
        if stops == 0.0:
            return img
        if raw_obj is not None:
            try:
                # exp_shift is in linear scale: 2^stops.
                # rawpy usable range: 0.25 (−2 stops) … 8.0 (+3 stops).
                linear_scale = float(np.clip(2.0 ** stops, 0.25, 8.0))
                # Preserve highlights when brightening to avoid blowing out
                # any remaining highlight detail.
                if stops > 1.0:
                    preserve = 0.95
                elif stops > 0.4:
                    preserve = 0.9
                elif stops > 0.0:
                    preserve = 0.85
                else:
                    preserve = 0.0
                return raw_obj.postprocess(
                    exp_shift=linear_scale,
                    exp_preserve_highlights=preserve,
                )
            except Exception:
                pass  # fall through to pixel-level fallback below
        factor = 2.0 ** stops
        return (img.astype(np.float32) * factor).clip(0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _get_image_orientation(img: np.ndarray) -> str:
        if img is None or img.ndim < 2:
            return "unknown"
        h, w = img.shape[:2]
        if h > w:
            return "portrait"
        if w > h:
            return "landscape"
        return "square"

    def load_models(self, status_cb: Optional[Callable[[str], None]] = None) -> None:
        if self.mask_rcnn and self.species_clf and self.quality_clf:
            return
        if status_cb:
            status_cb("Loading models... This may take a while on first run.")
        self.mask_rcnn = MaskRCNNWrapper()
        self.species_clf = BirdSpeciesClassifier(
            str(SPECIESCLASSIFIER_PATH),
            str(SPECIESCLASSIFIER_LABELS),
            self.use_gpu,
            models_dir=str(MODELS_DIR),
        )
        self.quality_clf = QualityClassifier(
            str(QUALITYCLASSIFIER_PATH),
            normalization_data_path=str(QUALITY_NORMALIZATION_DATA_PATH),
        )
        if status_cb:
            status_cb("Models loaded. Processing started.")

    def process_folder(
        self,
        folder: str,
        pause_event=None,
        cancel_event=None,
        callbacks: Optional[Dict[str, Callable]] = None,
        analyzer_name: str = "pipeline",
        wildlife_enabled: bool = True,
        detection_threshold: float = 0.75,
        scene_time_threshold: float = 1.0,
        mask_threshold: float = 0.5,
    ) -> None:
        callbacks = callbacks or {}
        status_cb = callbacks.get("on_status")
        progress_cb = callbacks.get("on_progress")
        image_cb = callbacks.get("on_image")
        thumbnail_cb = callbacks.get("on_thumbnail")
        detection_cb = callbacks.get("on_detection")
        crops_cb = callbacks.get("on_crops")
        quality_cb = callbacks.get("on_quality")
        species_cb = callbacks.get("on_species")
        error_cb = callbacks.get("on_error")

        rating_thresholds = None
        exposure_profile = "normal"
        if callable(load_persisted_settings):
            try:
                sett = load_persisted_settings() or {}
                profile = sett.get('rating_profile', 'balanced')
                rating_thresholds = get_profile_thresholds(profile)
                raw_exp_profile = str(sett.get('exposure_compensation_profile', 'normal') or 'normal').strip().lower()
                if raw_exp_profile in {'lenient', 'normal', 'aggressive'}:
                    exposure_profile = raw_exp_profile
            except Exception:
                rating_thresholds = None

        active_wildlife_categories = WILDLIFE_CATEGORIES if wildlife_enabled else []

        self._log_path = get_log_path(folder)
        stage_ctx = {"stage": "startup", "file": None}

        original_showwarning = warnings.showwarning

        def _showwarning(message, category, filename, lineno, file=None, line=None):
            log_warning(
                self._log_path,
                message,
                category=category,
                filename=filename,
                lineno=lineno,
                stage=stage_ctx["stage"],
                context={"file": stage_ctx["file"], "folder": folder},
            )
            if original_showwarning:
                original_showwarning(message, category, filename, lineno, file=file, line=line)

        warnings.showwarning = _showwarning

        try:
            stage_ctx["stage"] = "list_files"
            files = [
                f
                for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f))
                and os.path.splitext(f)[1].lower() in RAW_EXTENSIONS
            ]
            if not files:
                files = [
                    f
                    for f in os.listdir(folder)
                    if os.path.isfile(os.path.join(folder, f))
                    and os.path.splitext(f)[1].lower() in JPEG_EXTENSIONS
                ]
            files.sort()
            if not files:
                if status_cb:
                    status_cb("No supported image files found.")
                log_event(
                    self._log_path,
                    {
                        "level": "warning",
                        "event": "no_supported_files",
                        "analyzer": analyzer_name,
                        "folder": folder,
                    },
                )
                return

            log_event(
                self._log_path,
                {
                    "level": "info",
                    "event": "analysis_start",
                    "analyzer": analyzer_name,
                    "folder": folder,
                    "file_count": len(files),
                },
            )

            stage_ctx["stage"] = "create_kestrel_dirs"
            kestrel_dir = os.path.join(folder, KESTREL_DIR_NAME)
            export_dir = os.path.join(kestrel_dir, "export")
            crop_dir = os.path.join(kestrel_dir, "crop")
            os.makedirs(export_dir, exist_ok=True)
            os.makedirs(crop_dir, exist_ok=True)

            stage_ctx["stage"] = "load_database"
            database, db_path = load_database(kestrel_dir, analyzer_name, log_path=self._log_path)

            processed_set = set(database["filename"].values)
            new_files = [f for f in files if f not in processed_set]
            processed_count = len(files) - len(new_files)
            total = len(files)
            if progress_cb:
                progress_cb(processed_count, total)
            if processed_count > 0 and status_cb:
                status_cb("Picking up where Kestrel left off...")
            if not new_files:
                if status_cb:
                    status_cb("No new files to process.")
                if progress_cb:
                    progress_cb(total, total)
                return

            stage_ctx["stage"] = "load_models"
            self.load_models(status_cb=status_cb)

            previous_image = None
            previous_image_path = None
            previous_orientation = None
            if not database.empty:
                last_row = database.iloc[-1]
                last_filename = last_row["filename"]
                last_image_path = os.path.join(folder, last_filename)
                if os.path.exists(last_image_path):
                    img = read_image(last_image_path)
                    if img is not None:
                        previous_image = img
                        previous_image_path = last_image_path
                        previous_orientation = self._get_image_orientation(img)
            scene_count = database["scene_count"].max() if not database.empty else 0

            for idx, raw_file in enumerate(new_files, start=1):
                # Pause: wait until resume or until cancel_event is set.
                if pause_event is not None:
                    while not pause_event.is_set():
                        if cancel_event is not None and cancel_event.is_set():
                            if status_cb:
                                status_cb('Cancelled')
                            return
                        # Wait with timeout to be interruptible
                        pause_event.wait(timeout=0.5)
                if cancel_event is not None and cancel_event.is_set():
                    if status_cb:
                        status_cb('Cancelled')
                    return

                entry = {
                    "filename": raw_file,
                    "species": "Unknown",
                    "species_confidence": 0.0,
                    "family": "Unknown",
                    "family_confidence": 0.0,
                    "quality": -1.0,
                    "export_path": "N/A",
                    "crop_path": "N/A",
                    "crops_json": "[]",
                    "primary_crop_index": 0,
                    "scene_count": scene_count,
                    "feature_similarity": -1.0,
                    "feature_confidence": -1.0,
                    "color_similarity": -1.0,
                    "color_confidence": -1.0,
                    "similar": False,
                    "secondary_species_list": [],
                    "secondary_species_scores": [],
                    "secondary_family_list": [],
                    "secondary_family_scores": [],
                    "exposure_correction": 0.0,
                    "detection_scores": [],
                    "capture_time": "",
                    "orientation": "unknown",
                }

                image_path = None
                raw_obj = None
                try:
                    stage_ctx["stage"] = "read_image"
                    stage_ctx["file"] = raw_file
                    image_path = os.path.join(folder, raw_file)
                    img, raw_obj = read_image_for_pipeline(image_path)
                    if img is None:
                        raise RuntimeError("Image read returned None")

                    current_orientation = self._get_image_orientation(img)
                    entry["orientation"] = current_orientation

                    try:
                        ct = get_capture_time(image_path)
                        entry["capture_time"] = ct.isoformat() if ct is not None else ""
                    except Exception:
                        pass

                    stage_ctx["stage"] = "compute_similarity"
                    timestamp_similar = None
                    try:
                        timestamp_similar = compute_similarity_timestamp(
                            previous_image_path, image_path,
                            threshold_seconds=scene_time_threshold
                        ) if previous_image_path else None
                    except Exception as e:
                        log_warning(
                            self._log_path,
                            f"Timestamp similarity check failed: {e}",
                            stage=stage_ctx["stage"],
                            context={"file": raw_file, "folder": folder},
                        )

                    orientation_changed = (
                        previous_orientation is not None
                        and current_orientation != "unknown"
                        and previous_orientation != "unknown"
                        and current_orientation != previous_orientation
                    )

                    if orientation_changed:
                        scene_count += 1
                        entry.update(
                            {
                                "feature_similarity": -1.0,
                                "feature_confidence": -1.0,
                                "color_similarity": -1.0,
                                "color_confidence": -1.0,
                                "scene_count": scene_count,
                                "similar": False,
                            }
                        )
                    elif timestamp_similar is True:
                        # Images captured within the same second — treat as similar, skip AKAZE
                        entry.update(
                            {
                                "feature_similarity": -1.0,
                                "feature_confidence": -1.0,
                                "color_similarity": -1.0,
                                "color_confidence": -1.0,
                                "scene_count": scene_count,
                                "similar": True,
                            }
                        )
                    else:
                        similarity = compute_image_similarity_akaze(previous_image, img)
                        if not similarity["similar"]:
                            scene_count += 1
                        entry.update(
                            {
                                "feature_similarity": similarity["feature_similarity"],
                                "feature_confidence": similarity["feature_confidence"],
                                "color_similarity": similarity["color_similarity"],
                                "color_confidence": similarity["color_confidence"],
                                "scene_count": scene_count,
                                "similar": similarity["similar"],
                            }
                        )
                    previous_image = img.copy()
                    previous_image_path = image_path
                    previous_orientation = current_orientation

                    stage_ctx["stage"] = "export_image"
                    export_path = os.path.join(export_dir, f"{os.path.splitext(raw_file)[0]}_export.jpg")
                    img_small = cv2.resize(img, (1200, int(1200 * img.shape[0] / img.shape[1])))
                    cv2.imwrite(
                        export_path,
                        cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 70],
                    )
                    # Store relative path for cross-platform compatibility
                    export_path_rel = os.path.relpath(export_path, folder)
                    entry.update({"export_path": export_path_rel})
                    if thumbnail_cb:
                        thumbnail_cb({"filename": raw_file, "thumbnail": img_small, "export_path": export_path_rel})

                    stage_ctx["stage"] = "mask_rcnn_prediction"
                    # MaskRCNN inference can take many seconds. Pause semantics are
                    # handled at the start of each image loop so we do not check
                    # repeatedly inside the image processing path.
                    masks, pred_boxes, pred_class, pred_score = self.mask_rcnn.get_prediction(img, threshold=detection_threshold, mask_threshold=mask_threshold)
                    if masks is None or len(masks) == 0:
                        if detection_cb:
                            detection_cb(
                                {
                                    "filename": raw_file,
                                    "overlay": self._create_mask_overlay(img_small, None, None),
                                    "bird_count": 0,
                                }
                            )
                        if crops_cb:
                            crops_cb({"filename": raw_file, "crops": [], "confidences": []})
                        if quality_cb:
                            quality_cb({"filename": raw_file, "results": []})
                        if species_cb:
                            species_cb({"filename": raw_file, "results": []})
                        if status_cb:
                            status_cb(f"No detections in {raw_file}")
                        stage_ctx["stage"] = "write_crop"
                        crop_path = os.path.join(crop_dir, f"{os.path.splitext(raw_file)[0]}_crop_0.jpg")
                        cv2.imwrite(
                            crop_path,
                            cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 85],
                        )
                        # Store relative path for cross-platform compatibility
                        crop_path_rel = os.path.relpath(crop_path, folder)
                        h, w = img.shape[:2]
                        fallback_crop = {
                            "crop_index": 0,
                            "crop_path": crop_path_rel,
                            "detection_index": -1,
                            "detection_confidence": 0.0,
                            "species": "Unknown",
                            "species_confidence": 0.0,
                            "family": "Unknown",
                            "family_confidence": 0.0,
                            "quality": -1.0,
                            "rating": 0,
                            "exposure_correction": 0.0,
                            "bbox": {
                                "x_min": 0,
                                "x_max": int(w),
                                "y_min": 0,
                                "y_max": int(h),
                                "width": int(w),
                                "height": int(h),
                                "x_min_norm": 0.0,
                                "x_max_norm": 1.0,
                                "y_min_norm": 0.0,
                                "y_max_norm": 1.0,
                                "x_center_norm": 0.5,
                                "y_center_norm": 0.5,
                            },
                        }
                        entry.update(
                            {
                                "crop_path": crop_path_rel,
                                "crops_json": json.dumps([fallback_crop]),
                                "primary_crop_index": 0,
                            }
                        )
                        stage_ctx["stage"] = "save_database"
                        database = pd.concat([database, pd.DataFrame([entry])], ignore_index=True)
                        save_database(database, db_path)
                        if image_cb:
                            image_cb(entry)
                        if progress_cb:
                            progress_cb(idx + processed_count, total)
                        continue

                    # Store top-5 raw MaskRCNN detection confidence scores
                    entry["detection_scores"] = json.dumps([float(s) for s in sorted(pred_score, reverse=True)[:5]])

                    wildlife_indices = [i for i, c in enumerate(pred_class) if c in active_wildlife_categories]
                    bird_indices = [i for i, c in enumerate(pred_class) if c == "bird"]
                    bird_indices = sorted(bird_indices, key=lambda i: pred_score[i], reverse=True)[:5]

                    overlay_indices = bird_indices if bird_indices else wildlife_indices[:1]
                    if detection_cb:
                        detection_cb(
                            {
                                "filename": raw_file,
                                "overlay": self._create_mask_overlay(img_small, masks, overlay_indices),
                                "bird_count": len(bird_indices),
                            }
                        )

                    def process_nonbird(primary_mask_i):
                        stage_ctx["stage"] = "process_nonbird"
                        stops = self._compute_exposure_stops(img, masks[primary_mask_i], exposure_profile)
                        stops = self._refine_exposure_stops(
                            img,
                            masks[primary_mask_i],
                            stops,
                            exposure_profile,
                            raw_obj=raw_obj,
                        )
                        img_src = self._apply_exposure_correction(img, stops, raw_obj)
                        crop_bbox = self.mask_rcnn.get_square_crop_box(masks[primary_mask_i])
                        quality_crop, quality_mask = self.mask_rcnn.get_square_crop(
                            masks[primary_mask_i], img_src, resize=True
                        )
                        quality_score = self.quality_clf.classify(quality_crop, quality_mask)
                        return {
                            "index": int(primary_mask_i),
                            "confidence": float(pred_score[primary_mask_i]),
                            "species": pred_class[primary_mask_i],
                            "species_confidence": float(pred_score[primary_mask_i]),
                            "family": "N/A",
                            "family_confidence": 0.0,
                            "quality": quality_score,
                            "rating": quality_to_rating(quality_score, rating_thresholds),
                            "quality_crop": quality_crop,
                            "exposure_correction": round(stops, 4),
                            "crop_bbox": crop_bbox,
                        }

                    def process_bird_items(indices):
                        stage_ctx["stage"] = "process_bird"
                        items = []
                        for i in indices:
                            # Process per-crop results. Pause is checked at the
                            # top of the image loop so we avoid pausing mid-image.
                            stops = self._compute_exposure_stops(img, masks[i], exposure_profile)
                            stops = self._refine_exposure_stops(
                                img,
                                masks[i],
                                stops,
                                exposure_profile,
                                raw_obj=raw_obj,
                            )
                            img_src = self._apply_exposure_correction(img, stops, raw_obj)
                            species_crop = self.mask_rcnn.get_species_crop(pred_boxes[i], img_src)
                            crop_bbox = self.mask_rcnn.get_square_crop_box(masks[i])
                            quality_crop, quality_mask = self.mask_rcnn.get_square_crop(masks[i], img_src, resize=True)
                            items.append(
                                {
                                    "index": i,
                                    "confidence": float(pred_score[i]),
                                    "species_crop": species_crop,
                                    "quality_crop": quality_crop,
                                    "quality_mask": quality_mask,
                                    "stops": stops,
                                    "crop_bbox": crop_bbox,
                                }
                            )
                        if crops_cb:
                            crops_cb(
                                {
                                    "filename": raw_file,
                                    "crops": [i["quality_crop"] for i in items],
                                    "confidences": [i["confidence"] for i in items],
                                }
                            )
                        for item in items:
                            i = item["index"]
                            if pred_class[i] == "bird":
                                species_result = self.species_clf.classify(item["species_crop"])
                                item["species"] = (
                                    species_result["top_species_labels"][0]
                                    if len(species_result["top_species_labels"])
                                    else "Unknown"
                                )
                                item["species_confidence"] = (
                                    float(species_result["top_species_scores"][0])
                                    if len(species_result["top_species_scores"])
                                    else 0.0
                                )
                                item["family"] = (
                                    species_result["top_family_labels"][0]
                                    if len(species_result["top_family_labels"])
                                    else "Unknown"
                                )
                                item["family_confidence"] = (
                                    float(species_result["top_family_scores"][0])
                                    if len(species_result["top_family_scores"])
                                    else 0.0
                                )
                            else:
                                item["species"] = pred_class[i]
                                item["species_confidence"] = float(pred_score[i])
                                item["family"] = "N/A"
                                item["family_confidence"] = 0.0
                            item["exposure_correction"] = round(item["stops"], 4)
                            stage_ctx["stage"] = "quality_score"
                            quality_score = self.quality_clf.classify(item["quality_crop"], item["quality_mask"])
                            item["quality"] = quality_score
                            item["rating"] = quality_to_rating(quality_score, rating_thresholds)
                        if quality_cb:
                            quality_cb(
                                {
                                    "filename": raw_file,
                                    "results": [
                                        {"quality": i["quality"], "rating": i["rating"]} for i in items
                                    ],
                                }
                            )
                        if species_cb:
                            species_cb(
                                {
                                    "filename": raw_file,
                                    "results": [
                                        {
                                            "species": i["species"],
                                            "species_confidence": i["species_confidence"],
                                            "family": i["family"],
                                            "family_confidence": i["family_confidence"],
                                        }
                                        for i in items
                                    ],
                                }
                            )
                        return items

                    crop_items_for_write = []
                    primary_crop_index = 0
                    img_h, img_w = img.shape[:2]

                    if bird_indices:
                        bird_items = process_bird_items(bird_indices)
                        bird_data = [
                            {
                                "index": i["index"],
                                "confidence": i["confidence"],
                                "species": i["species"],
                                "species_confidence": i["species_confidence"],
                                "family": i["family"],
                                "family_confidence": i["family_confidence"],
                                "quality": i["quality"],
                                "rating": i["rating"],
                                "quality_crop": i["quality_crop"],
                                "exposure_correction": i.get("exposure_correction", 0.0),
                                "crop_bbox": i.get("crop_bbox"),
                            }
                            for i in bird_items
                        ]
                        primary_crop_index = int(np.argmax([b["quality"] for b in bird_data]))
                        primary_bird = bird_data[primary_crop_index]
                        entry.update(
                            {
                                "species": primary_bird["species"],
                                "species_confidence": primary_bird["species_confidence"],
                                "family": primary_bird["family"],
                                "family_confidence": primary_bird["family_confidence"],
                                "quality": primary_bird["quality"],
                                "exposure_correction": primary_bird["exposure_correction"],
                            }
                        )
                        all_species = [b["species"] for b in bird_data]
                        all_species_conf = [float(b["species_confidence"]) for b in bird_data]
                        all_families = [b["family"] for b in bird_data]
                        all_family_conf = [float(b["family_confidence"]) for b in bird_data]
                        entry.update(
                            {
                                "secondary_species_list": json.dumps(all_species),
                                "secondary_species_scores": json.dumps(all_species_conf),
                                "secondary_family_list": json.dumps(all_families),
                                "secondary_family_scores": json.dumps(all_family_conf),
                            }
                        )
                        crop_items_for_write = bird_data
                    else:
                        if wildlife_indices:
                            primary_index = wildlife_indices[np.argmax([pred_score[i] for i in wildlife_indices])]
                            result = process_nonbird(primary_index)
                            if crops_cb:
                                crops_cb(
                                    {
                                        "filename": raw_file,
                                        "crops": [result["quality_crop"]],
                                        "confidences": [float(pred_score[primary_index])],
                                    }
                                )
                            if quality_cb:
                                quality_cb(
                                    {
                                        "filename": raw_file,
                                        "results": [{"quality": result["quality"], "rating": result["rating"]}],
                                    }
                                )
                            if species_cb:
                                species_cb(
                                    {
                                        "filename": raw_file,
                                        "results": [
                                            {
                                                "species": result["species"],
                                                "species_confidence": result["species_confidence"],
                                                "family": result["family"],
                                                "family_confidence": result["family_confidence"],
                                            }
                                        ],
                                    }
                                )
                            entry.update(
                                {
                                    "species": result["species"],
                                    "species_confidence": result["species_confidence"],
                                    "family": result["family"],
                                    "family_confidence": result["family_confidence"],
                                    "quality": result["quality"],
                                    "exposure_correction": result["exposure_correction"],
                                }
                            )
                            crop_items_for_write = [result]
                            primary_crop_index = 0
                        else:
                            if crops_cb:
                                crops_cb({"filename": raw_file, "crops": [], "confidences": []})
                            if quality_cb:
                                quality_cb({"filename": raw_file, "results": []})
                            if species_cb:
                                species_cb({"filename": raw_file, "results": []})
                            crop_items_for_write = [
                                {
                                    "index": -1,
                                    "confidence": 0.0,
                                    "species": entry.get("species", "Unknown"),
                                    "species_confidence": entry.get("species_confidence", 0.0),
                                    "family": entry.get("family", "Unknown"),
                                    "family_confidence": entry.get("family_confidence", 0.0),
                                    "quality": entry.get("quality", -1.0),
                                    "rating": 0,
                                    "quality_crop": img_small,
                                    "exposure_correction": entry.get("exposure_correction", 0.0),
                                    "crop_bbox": {
                                        "x_min": 0,
                                        "x_max": int(img_w),
                                        "y_min": 0,
                                        "y_max": int(img_h),
                                        "width": int(img_w),
                                        "height": int(img_h),
                                        "x_min_norm": 0.0,
                                        "x_max_norm": 1.0,
                                        "y_min_norm": 0.0,
                                        "y_max_norm": 1.0,
                                        "x_center_norm": 0.5,
                                        "y_center_norm": 0.5,
                                    },
                                }
                            ]
                            primary_crop_index = 0

                    stage_ctx["stage"] = "write_crop"
                    serialized_crops = []
                    base_name = os.path.splitext(raw_file)[0]
                    for crop_idx, crop_item in enumerate(crop_items_for_write):
                        crop_img = crop_item.get("quality_crop")
                        if crop_img is None:
                            continue
                        crop_path = os.path.join(crop_dir, f"{base_name}_crop_{crop_idx}.jpg")
                        cv2.imwrite(
                            crop_path,
                            cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 85],
                        )
                        crop_path_rel = os.path.relpath(crop_path, folder)
                        bbox = crop_item.get("crop_bbox") or {
                            "x_min": 0,
                            "x_max": int(img_w),
                            "y_min": 0,
                            "y_max": int(img_h),
                            "width": int(img_w),
                            "height": int(img_h),
                            "x_min_norm": 0.0,
                            "x_max_norm": 1.0,
                            "y_min_norm": 0.0,
                            "y_max_norm": 1.0,
                            "x_center_norm": 0.5,
                            "y_center_norm": 0.5,
                        }
                        serialized_crops.append(
                            {
                                "crop_index": int(crop_idx),
                                "crop_path": crop_path_rel,
                                "detection_index": int(crop_item.get("index", -1)),
                                "detection_confidence": float(crop_item.get("confidence", 0.0)),
                                "species": str(crop_item.get("species", "Unknown") or "Unknown"),
                                "species_confidence": float(crop_item.get("species_confidence", 0.0)),
                                "family": str(crop_item.get("family", "Unknown") or "Unknown"),
                                "family_confidence": float(crop_item.get("family_confidence", 0.0)),
                                "quality": float(crop_item.get("quality", -1.0)),
                                "rating": int(crop_item.get("rating", 0)),
                                "exposure_correction": float(crop_item.get("exposure_correction", 0.0)),
                                "bbox": {
                                    "x_min": int(bbox.get("x_min", 0)),
                                    "x_max": int(bbox.get("x_max", img_w)),
                                    "y_min": int(bbox.get("y_min", 0)),
                                    "y_max": int(bbox.get("y_max", img_h)),
                                    "width": int(bbox.get("width", img_w)),
                                    "height": int(bbox.get("height", img_h)),
                                    "x_min_norm": float(bbox.get("x_min_norm", 0.0)),
                                    "x_max_norm": float(bbox.get("x_max_norm", 1.0)),
                                    "y_min_norm": float(bbox.get("y_min_norm", 0.0)),
                                    "y_max_norm": float(bbox.get("y_max_norm", 1.0)),
                                    "x_center_norm": float(bbox.get("x_center_norm", 0.5)),
                                    "y_center_norm": float(bbox.get("y_center_norm", 0.5)),
                                },
                            }
                        )

                    if not serialized_crops:
                        raise RuntimeError("No crop records generated for image")

                    primary_crop_index = int(np.clip(primary_crop_index, 0, len(serialized_crops) - 1))
                    primary_crop = serialized_crops[primary_crop_index]
                    entry.update(
                        {
                            "species": primary_crop["species"],
                            "species_confidence": primary_crop["species_confidence"],
                            "family": primary_crop["family"],
                            "family_confidence": primary_crop["family_confidence"],
                            "quality": primary_crop["quality"],
                            "exposure_correction": primary_crop["exposure_correction"],
                            "crop_path": primary_crop["crop_path"],
                            "crops_json": json.dumps(serialized_crops),
                            "primary_crop_index": primary_crop_index,
                        }
                    )

                    stage_ctx["stage"] = "save_database"
                    database = pd.concat([database, pd.DataFrame([entry])], ignore_index=True)
                    save_database(database, db_path)

                    if image_cb:
                        image_cb(entry)

                    if status_cb:
                        _q = entry.get('quality', -1)
                        _display_q = f"{float(_q):.3f}" if _q not in (None, 'N/A', -1) else '—'
                        status_cb(
                            f"Processed {raw_file}: {entry['species']} Q={_display_q}"
                            f" ({idx + processed_count}/{total})"
                        )
                except Exception as e:
                    log_exception(
                        self._log_path,
                        e,
                        stage=stage_ctx["stage"],
                        context={
                            "file": raw_file,
                            "folder": folder,
                            "image_path": image_path,
                            "analyzer": analyzer_name,
                        },
                    )
                    if error_cb:
                        error_cb(raw_file, e)
                    if status_cb:
                        status_cb(f"Error {raw_file}: {e}")
                    entry["scene_count"] = scene_count
                    entry["species"] = "Error"
                    entry["similar"] = False
                    database = pd.concat([database, pd.DataFrame([entry])], ignore_index=True)
                    save_database(database, db_path)
                    time.sleep(2)

                if progress_cb:
                    progress_cb(idx + processed_count, total)

                # Explicitly clear large temporary variables after each image
                # so that pausing between images doesn't retain large buffers.
                try:
                    # Close the rawpy object first to release the RAW file buffer.
                    try:
                        if raw_obj is not None:
                            raw_obj.close()
                        del raw_obj
                    except Exception: pass
                    try: del masks
                    except Exception: pass
                    try: del pred_boxes
                    except Exception: pass
                    try: del pred_class
                    except Exception: pass
                    try: del pred_score
                    except Exception: pass
                    try: del img
                    except Exception: pass
                    try: del crop_img
                    except Exception: pass
                    try: del items
                    except Exception: pass
                    try: del bird_items
                    except Exception: pass
                except Exception:
                    pass

            # === Post-analysis: compute quality distribution and normalized ratings ===
            stage_ctx["stage"] = "post_analysis_normalization"
            try:
                from .ratings import compute_quality_distribution
                if not database.empty and "quality" in database.columns:
                    quality_scores = database["quality"].tolist()
                    distribution = compute_quality_distribution(quality_scores)

                    # Save analysis results (no normalized_rating; computed at runtime)
                    save_database(database, db_path)

                    # Cache quality distribution in kestrel_metadata.json for runtime normalization
                    metadata_path = os.path.join(kestrel_dir, METADATA_FILENAME)
                    try:
                        import json as _json
                        if os.path.exists(metadata_path):
                            with open(metadata_path, "r", encoding="utf-8") as mf:
                                _meta = _json.load(mf)
                        else:
                            _meta = {"version": VERSION, "analyzer": analyzer_name}
                        _meta["quality_distribution"] = distribution
                        _meta["quality_distribution_stored"] = True
                        with open(metadata_path, "w", encoding="utf-8") as mf:
                            _json.dump(_meta, mf, indent=2)
                    except Exception as _meta_e:
                        log_warning(
                            self._log_path,
                            f"Failed to write quality distribution to metadata: {_meta_e}",
                            stage="post_analysis_normalization",
                        )

                    # Create or update kestrel_scenedata.json
                    try:
                        existing_scenedata = load_scenedata(kestrel_dir)
                        if not existing_scenedata.get("scenes"):
                            new_scenedata = build_scenedata_from_database(database)
                            save_scenedata(new_scenedata, kestrel_dir)
                        else:
                            update_scenedata_with_database(existing_scenedata, database)
                            save_scenedata(existing_scenedata, kestrel_dir)
                    except Exception as _sd_e:
                        log_warning(
                            self._log_path,
                            f"Failed to create/update kestrel_scenedata.json: {_sd_e}",
                            stage="post_analysis_normalization",
                        )

                log_event(
                    self._log_path,
                    {
                        "level": "info",
                        "event": "analysis_complete",
                        "folder": folder,
                        "total_files": len(database),
                    },
                )
                if status_cb:
                    status_cb("Analysis complete.")
            except Exception as _post_e:
                log_warning(
                    self._log_path,
                    f"Post-analysis normalization failed: {_post_e}",
                    stage="post_analysis_normalization",
                )

        except Exception as e:
            log_exception(
                self._log_path,
                e,
                stage=stage_ctx["stage"],
                context={"folder": folder, "analyzer": analyzer_name},
            )
            if status_cb:
                status_cb(f"Fatal error: {e}")
            if error_cb:
                error_cb("fatal", e)
        finally:
            warnings.showwarning = original_showwarning
