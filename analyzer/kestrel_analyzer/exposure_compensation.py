from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


_ALLOWED_PROFILES = {"lenient", "normal", "aggressive"}
_ALLOWED_SOLVERS = {
    "legacy_iterative",
    "convergent_two_pass",
    "lifted_two_pass",
    "two_pass",
    "single_pass",
    "predictive_fast",
    "adaptive_fast",
}


_METER_PROFILE_CFG: Dict[str, Dict[str, float]] = {
    "lenient": {
        "target_p50": 0.30,
        "target_p90": 0.68,
        "target_p98": 0.92,
        "meter_min": 0.30,
        "meter_max": 3.30,
    },
    "normal": {
        "target_p50": 0.33,
        "target_p90": 0.72,
        "target_p98": 0.90,
        "meter_min": 0.35,
        "meter_max": 3.50,
    },
    "aggressive": {
        "target_p50": 0.36,
        "target_p90": 0.74,
        "target_p98": 0.88,
        "meter_min": 0.35,
        "meter_max": 3.60,
    },
}


_PRESERVE_HIGHLIGHTS_CFG: Dict[str, Dict[str, float]] = {
    "lenient": {
        "gt_1_0": 0.90,
        "gt_0_4": 0.85,
        "gt_0_0": 0.75,
    },
    "normal": {
        "gt_1_0": 0.95,
        "gt_0_4": 0.90,
        "gt_0_0": 0.85,
    },
    "aggressive": {
        "gt_1_0": 0.98,
        "gt_0_4": 0.94,
        "gt_0_0": 0.90,
    },
}


def normalize_profile_name(profile: str) -> str:
    profile_name = str(profile or "aggressive").strip().lower()
    if profile_name not in _ALLOWED_PROFILES:
        profile_name = "aggressive"
    return profile_name


def normalize_solver_name(solver: str) -> str:
    solver_name = str(solver or "convergent_two_pass").strip().lower()
    if solver_name not in _ALLOWED_SOLVERS:
        solver_name = "convergent_two_pass"
    return solver_name


def preserve_highlights_for_stops(stops: float, profile: str = "aggressive") -> float:
    stops = float(stops)
    profile_name = normalize_profile_name(profile)
    cfg = _PRESERVE_HIGHLIGHTS_CFG.get(profile_name, _PRESERVE_HIGHLIGHTS_CFG["aggressive"])
    if stops > 1.0:
        return float(cfg["gt_1_0"])
    if stops > 0.4:
        return float(cfg["gt_0_4"])
    if stops > 0.0:
        return float(cfg["gt_0_0"])
    return 0.0


def _preview_from_linear(norm_rgb: np.ndarray) -> np.ndarray:
    v = np.clip(norm_rgb, 0.0, 1.0)
    return (np.power(v, 1.0 / 2.2) * 255.0 + 0.5).astype(np.uint8)


def compute_global_meter_scale(
    noauto_linear_rgb: np.ndarray,
    profile: str = "aggressive",
) -> Tuple[float, Dict[str, float]]:
    profile_name = normalize_profile_name(profile)
    meter_cfg = _METER_PROFILE_CFG.get(profile_name, _METER_PROFILE_CFG["aggressive"])
    lum = (
        0.2126 * noauto_linear_rgb[..., 0]
        + 0.7152 * noauto_linear_rgb[..., 1]
        + 0.0722 * noauto_linear_rgb[..., 2]
    )
    p50, p90, p98 = [float(x) for x in np.percentile(lum, [50, 90, 98])]
    eps = 1e-4
    target_p50 = float(meter_cfg["target_p50"])
    target_p90 = float(meter_cfg["target_p90"])
    target_p98 = float(meter_cfg["target_p98"])
    g50 = target_p50 / max(p50, eps)
    g90 = target_p90 / max(p90, eps)
    g98 = target_p98 / max(p98, eps)
    meter_scale = min(g50, g90, g98)
    meter_scale = float(np.clip(meter_scale, float(meter_cfg["meter_min"]), float(meter_cfg["meter_max"])))
    meter_debug = {
        "profile": profile_name,
        "p50": p50,
        "p90": p90,
        "p98": p98,
        "target_p50": target_p50,
        "target_p90": target_p90,
        "target_p98": target_p98,
        "g50": float(g50),
        "g90": float(g90),
        "g98": float(g98),
        "meter_min": float(meter_cfg["meter_min"]),
        "meter_max": float(meter_cfg["meter_max"]),
        "meter_scale": meter_scale,
    }
    return meter_scale, meter_debug


def build_metered_detection_image(
    raw_obj,
    profile: str = "aggressive",
) -> Tuple[Optional[np.ndarray], float, Dict[str, Any]]:
    """Build the detection image from RAW using no_auto_bright + global metering."""
    try:
        noauto16 = raw_obj.postprocess(no_auto_bright=True, output_bps=16)
        noauto_norm = np.clip(noauto16.astype(np.float32) / 65535.0, 0.0, 1.0)
        meter_scale, meter_debug = compute_global_meter_scale(noauto_norm, profile=profile)
        metered_norm = np.clip(noauto_norm * meter_scale, 0.0, 1.0)
        metered8 = _preview_from_linear(metered_norm)
        return metered8, meter_scale, meter_debug
    except Exception as exc:
        return None, 1.0, {"error": str(exc), "meter_scale": 1.0}


def compose_total_stops(subject_stops: float, meter_scale: float) -> float:
    meter_scale = float(np.clip(float(meter_scale), 0.25, 8.0))
    return float(np.log2(meter_scale) + float(subject_stops))


def compute_exposure_stops(img: np.ndarray, mask: np.ndarray, profile: str = "aggressive") -> float:
    """Estimate exposure correction in stops for the masked subject region."""
    EPS = 1e-3
    profile_name = normalize_profile_name(profile)

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
        mask_bool = np.array(mask, dtype=bool, copy=True) if mask is not None else None
        if (
            mask_bool is not None
            and mask_bool.any()
            and img_f.ndim == 3
            and img_f.shape[:2] == mask_bool.shape
        ):
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
                eroded = cv2.erode(
                    mask_u8,
                    np.ones((3, 3), dtype=np.uint8),
                    iterations=erode_iters,
                ).astype(bool)
                if eroded.any():
                    mask_bool = eroded
            pixels = img_f[mask_bool]
        else:
            pixels = img_f.reshape(-1, 3)

        lum = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
        lum = lum[np.isfinite(lum)]
        if lum.size == 0:
            return 0.0
        p10, p95, p98 = np.percentile(lum, [10, 95, 98])

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

        shadow_push_stop = float(np.log2(TARGET_SHADOW_P10 / max(float(p10), EPS)))

        if highlight_ceiling >= 0.0:
            stops = highlight_ceiling * BRIGHTEN_STRENGTH
            if shadow_push_stop > 0.0:
                stops = max(stops, min(shadow_push_stop, highlight_ceiling))
        else:
            stops = highlight_ceiling * DARKEN_STRENGTH

        stops = float(np.clip(stops, MAX_DARKEN, MAX_BRIGHTEN))
        if not np.isfinite(stops):
            return 0.0
        return stops
    except Exception:
        return 0.0


def refine_exposure_stops(
    img: np.ndarray,
    mask: np.ndarray,
    initial_stops: float,
    profile: str,
    solver: str = "convergent_two_pass",
    raw_obj=None,
    *,
    base_scale: float = 1.0,
    no_auto_bright: bool = False,
) -> float:
    """Refine aggressive exposure stops using one of several solver strategies.

    Solver modes:
    - legacy_iterative: 3-pass residual refinement (highest quality, slowest).
    - convergent_two_pass: 2-pass refinement with convergence-tail extrapolation.
    - lifted_two_pass: 2-pass refinement with bounded positive-seed lift.
    - two_pass: up to 2 passes.
    - single_pass: up to 1 pass.
    - predictive_fast: no refinement postprocess calls; applies a bounded analytic bias.
    - adaptive_fast: chooses one- or two-pass refinement based on stop magnitude.
    """

    def _refine_loop(
        start_stops: float,
        max_iters: int,
        residual_tolerance: float,
        gain_positive: float,
        gain_negative: float,
    ) -> float:
        total_local = float(start_stops)
        for _ in range(int(max(0, max_iters))):
            corrected = apply_exposure_correction(
                img,
                total_local,
                raw_obj,
                base_scale=base_scale,
                no_auto_bright=no_auto_bright,
                profile=profile_name,
            )
            residual = compute_exposure_stops(corrected, mask, profile_name)
            if not np.isfinite(residual):
                break
            residual = float(residual)
            if abs(residual) <= residual_tolerance:
                break

            gain = float(gain_positive if residual > 0.0 else gain_negative)
            updated = float(np.clip(total_local + residual * gain, -2.0, 3.0))
            if abs(updated - total_local) <= 0.01:
                total_local = updated
                break
            total_local = updated
            if total_local <= -1.95 or total_local >= 2.95:
                break
        return total_local

    def _predictive_fast(stops_val: float) -> float:
        # Fast path with no extra RAW postprocess calls. The mapping applies a
        # bounded profile bias so results remain close to iterative refinement.
        s = float(stops_val)
        mag = abs(s)
        if mag <= 0.10:
            return float(np.clip(s, -2.0, 3.0))
        if s > 0.0:
            boost = min(0.28, 0.10 + 0.18 * min(1.0, mag / 2.0))
            s = s + boost * min(1.0, mag / 2.2)
        else:
            boost = min(0.22, 0.06 + 0.16 * min(1.0, mag / 2.0))
            s = s - boost * min(1.0, mag / 2.0)
        return float(np.clip(s, -2.0, 3.0))

    def _convergent_two_pass(start_stops: float) -> float:
        """Run two RAW refinement passes, then estimate the remaining tail.

        The tail estimate is only used for positive residuals (underexposure)
        to avoid pushing highlight clipping in already-bright crops.
        """
        total_local = float(start_stops)
        residuals: list[float] = []
        last_corrected: Optional[np.ndarray] = None

        for _ in range(2):
            corrected = apply_exposure_correction(
                img,
                total_local,
                raw_obj,
                base_scale=base_scale,
                no_auto_bright=no_auto_bright,
                profile=profile_name,
            )
            last_corrected = corrected
            residual = compute_exposure_stops(corrected, mask, profile_name)
            if not np.isfinite(residual):
                break
            residual = float(residual)
            residuals.append(residual)
            if abs(residual) <= residual_tolerance:
                break

            gain = float(0.75 if residual > 0.0 else 0.90)
            updated = float(np.clip(total_local + residual * gain, -2.0, 3.0))
            if abs(updated - total_local) <= 0.01:
                total_local = updated
                break
            total_local = updated
            if total_local <= -1.95 or total_local >= 2.95:
                break

        if len(residuals) < 2:
            return total_local

        prev_residual = float(residuals[-2])
        last_residual = float(residuals[-1])
        if prev_residual <= 0.0 or last_residual <= 0.0:
            return total_local
        if abs(last_residual) <= residual_tolerance:
            return total_local
        if abs(prev_residual) <= 1e-4:
            return total_local

        contraction = float(np.clip(abs(last_residual / prev_residual), 0.05, 0.92))
        if contraction < 0.20:
            return total_local

        # Predict one additional refinement step (legacy's 3rd pass) without
        # paying another RAW postprocess call.
        tail = float(0.75 * last_residual * contraction)
        tail_cap = float(min(0.28, max(0.08, 3.0 - total_local)))

        if last_corrected is not None:
            try:
                img_f = last_corrected.astype(np.float32) / 255.0
                mask_bool = np.asarray(mask, dtype=bool)
                if (
                    mask_bool.ndim == 2
                    and img_f.ndim == 3
                    and mask_bool.shape == img_f.shape[:2]
                    and np.any(mask_bool)
                ):
                    pixels = img_f[mask_bool]
                else:
                    pixels = img_f.reshape(-1, 3)

                if pixels.shape[0] > 0:
                    lum = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
                    p98 = float(np.percentile(lum[np.isfinite(lum)], 98)) if np.any(np.isfinite(lum)) else 0.0
                    if p98 > 0.0:
                        p98_headroom_stops = float(np.log2(0.84 / max(p98, 1e-3)))
                        if np.isfinite(p98_headroom_stops):
                            tail_cap = min(tail_cap, float(np.clip(0.85 * p98_headroom_stops, 0.04, 0.24)))
            except Exception:
                pass

        tail = float(np.clip(tail, 0.0, tail_cap))
        if tail <= 0.01:
            return total_local

        return float(np.clip(total_local + tail, -2.0, 3.0))

    total = float(initial_stops)
    profile_name = normalize_profile_name(profile)
    solver_name = normalize_solver_name(solver)
    if profile_name != "aggressive":
        return total

    residual_tolerance = 0.02
    if abs(total) <= residual_tolerance:
        return total

    if solver_name == "predictive_fast":
        return _predictive_fast(total)

    if solver_name == "legacy_iterative":
        return _refine_loop(total, max_iters=3, residual_tolerance=residual_tolerance, gain_positive=0.75, gain_negative=0.90)

    if solver_name == "convergent_two_pass":
        seed = float(total)
        if seed > 0.25:
            seed += min(0.16, 0.06 + 0.08 * min(1.0, seed / 2.0))
        elif seed < -0.6:
            seed -= min(0.10, 0.03 + 0.05 * min(1.0, abs(seed) / 1.8))
        return _convergent_two_pass(seed)

    if solver_name == "lifted_two_pass":
        seed = float(total)
        if seed > 0.25:
            seed += min(0.20, 0.08 + 0.10 * min(1.0, seed / 2.0))
        elif seed < -0.6:
            seed -= min(0.10, 0.03 + 0.05 * min(1.0, abs(seed) / 1.8))
        return _refine_loop(seed, max_iters=2, residual_tolerance=residual_tolerance, gain_positive=0.74, gain_negative=0.89)

    if solver_name == "two_pass":
        return _refine_loop(total, max_iters=2, residual_tolerance=residual_tolerance, gain_positive=0.74, gain_negative=0.88)

    if solver_name == "single_pass":
        return _refine_loop(total, max_iters=1, residual_tolerance=residual_tolerance, gain_positive=0.72, gain_negative=0.86)

    # adaptive_fast (default): aggressively reduce postprocess calls while
    # preserving behavior on hard cases with a bounded second pass.
    if abs(total) < 0.45:
        return total
    if raw_obj is None:
        return _refine_loop(total, max_iters=1, residual_tolerance=residual_tolerance, gain_positive=0.70, gain_negative=0.84)
    if abs(total) < 1.20:
        return _refine_loop(total, max_iters=1, residual_tolerance=residual_tolerance, gain_positive=0.72, gain_negative=0.86)
    return _refine_loop(total, max_iters=2, residual_tolerance=residual_tolerance, gain_positive=0.74, gain_negative=0.88)


def apply_exposure_correction(
    img: np.ndarray,
    stops: float,
    raw_obj=None,
    *,
    base_scale: float = 1.0,
    no_auto_bright: bool = False,
    profile: str = "aggressive",
) -> np.ndarray:
    """Return a copy of *img* with exposure shifted by *stops* stops."""
    if raw_obj is not None:
        try:
            total_scale = float(np.clip(float(base_scale) * (2.0 ** float(stops)), 0.25, 8.0))
            if abs(total_scale - 1.0) <= 1e-6 and not no_auto_bright:
                return img
            pp_kwargs = {
                "exp_shift": total_scale,
                "exp_preserve_highlights": preserve_highlights_for_stops(stops, profile=profile),
            }
            if no_auto_bright:
                pp_kwargs["no_auto_bright"] = True
            return raw_obj.postprocess(**pp_kwargs)
        except Exception:
            pass

    if stops == 0.0:
        return img
    factor = 2.0 ** float(stops)
    return (img.astype(np.float32) * factor).clip(0.0, 255.0).astype(np.uint8)
