"""BBox opposite-view rotation-axis calibration for OrbitTo3D canvas padding.

Consumes an existing CPU, file-backed RGBA VSR mmap batch. It scans one
alpha plane at a time and never materializes another image batch. No pixels,
angles, cameras, seeds, or Pixal3D conditioning are modified here.
"""
from __future__ import annotations

import json
import math
import statistics

import torch


def _parse_azimuths(text: str) -> list[float]:
    s = str(text).strip()
    if not s:
        raise ValueError("azimuths must not be empty")
    try:
        values = json.loads(s)
    except (ValueError, TypeError):
        values = [part.strip() for part in s.strip("[]").split(",") if part.strip()]
    if not isinstance(values, (list, tuple)):
        values = [values]
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid azimuths: {s[:100]!r}") from e
    if not out or not all(math.isfinite(x) for x in out):
        raise ValueError("azimuths must contain finite angles")
    return out


def _circular_difference(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _find_opposite_pairs(azimuths: list[float], tolerance_deg: float) -> list[tuple[int, int, float]]:
    angles = [a % 360.0 for a in azimuths]
    candidate_pairs = []
    for i, ai in enumerate(angles):
        for j in range(i + 1, len(angles)):
            error = _circular_difference(angles[j], (ai + 180.0) % 360.0)
            if error <= tolerance_deg:
                candidate_pairs.append((error, i, j))
    # One-to-one matching. A duplicate/irregular angle list must not silently
    # count one opposite frame multiple times.
    used = set()
    pairs = []
    for error, i, j in sorted(candidate_pairs):
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        pairs.append((i, j, error))
    return sorted(pairs)


def _canvas_coordinates(width: int, height: int, axis_x: float, align: int, legacy: bool):
    if legacy:
        side = max(width, height)
        return side, (side - width) // 2, (side - height) // 2

    if not (0 <= axis_x <= width):
        raise ValueError(f"Axis x={axis_x} is outside input width {width}")

    # The axis, not the image center, must lie at the center of the square.
    # Besides x >= 0, require the rightmost pixel to fit inside the canvas.
    side = max(width, height, math.ceil(2 * axis_x), math.ceil(2 * (width - axis_x)))
    if side > max(width, height) and align > 1:
        side = math.ceil(side / align) * align

    x = round(side / 2.0 - axis_x)
    while x < 0 or x + width > side:
        side += max(align, 1)
        x = round(side / 2.0 - axis_x)
    y = (side - height) // 2
    assert x >= 0 and y >= 0 and x + width <= side and y + height <= side
    return side, int(x), int(y)


class Orbit3DAxisBBoxCanvas:
    """Find the horizontal rotation axis from opposite alpha-mask bbox pairs.

    Important: alpha bbox centers from a single view are NOT the axis. For an
    approximately rigid turntable, opposite views' projected silhouettes
    reverse their lateral extent, so their bbox-center midpoint estimates it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "azimuths": ("STRING", {"default": "0, 90, 180, 270", "forceInput": True}),
            "fov_deg": ("FLOAT", {"default": 40.0, "min": 0.1, "max": 179.0, "forceInput": True}),
            "mode": (["bbox_opposite_pairs", "legacy_center"], {"default": "bbox_opposite_pairs"}),
            "alpha_threshold": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.05}),
            "opposite_tolerance_deg": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 20.0, "step": 0.1}),
            "canvas_alignment": (["32", "16", "8", "1"], {"default": "32"}),
            "correct_canvas_fov": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "FLOAT", "STRING", "FLOAT")
    RETURN_NAMES = ("canvas_size", "x_offset", "y_offset", "axis_x_px", "axis_delta_px", "report", "canvas_fov_deg")
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Views"

    @torch.inference_mode()
    def run(self, images, azimuths, fov_deg, mode="bbox_opposite_pairs", alpha_threshold=0.5,
            opposite_tolerance_deg=1.0, canvas_alignment="32", correct_canvas_fov=True):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise TypeError("Expected IMAGE tensor [V,H,W,C] from Orbit3DVSRBatchCache.mmap_batch")
        if images.device.type != "cpu" or images.dtype != torch.float32:
            raise ValueError("Axis BBox Canvas requires CPU float32 VSR mmap batch")
        views, height, width, channels = (int(x) for x in images.shape)
        if views < 1 or width <= 0 or height <= 0:
            raise ValueError("Empty or invalid VSR batch")
        if mode not in ("bbox_opposite_pairs", "legacy_center"):
            raise ValueError(f"Unknown axis calibration mode: {mode}")
        angles = _parse_azimuths(azimuths)
        if len(angles) != views:
            raise ValueError(f"Azimuth count {len(angles)} differs from VSR view count {views}")
        align = int(canvas_alignment)
        legacy = mode == "legacy_center"
        axis_x = width / 2.0
        bboxes = []
        pairs = []
        pair_centers = []

        if not legacy:
            if channels != 4:
                raise ValueError(
                    f"Expected 4-channel RGBA VSR (RMBG alpha mask), got channels={channels}. "
                    "Do not infer an axis from RGB/background pixels."
                )
            for i in range(views):
                # Only one alpha plane is accessed at any time; this is a view
                # into the file-backed tensor, NOT a batch-sized allocation.
                alpha = images[i, :, :, 3]
                # A single short-lived 2-D bool array is the largest temporary.
                foreground = alpha > float(alpha_threshold)
                cols = foreground.any(dim=0).nonzero(as_tuple=True)[0]
                if cols.numel() == 0:
                    raise RuntimeError(f"Empty alpha mask for VSR view {i}")
                left, right = int(cols[0]), int(cols[-1])
                bboxes.append((left, right))
                del alpha, foreground, cols

            pairs = _find_opposite_pairs(angles, float(opposite_tolerance_deg))
            if len(pairs) < 2:
                raise RuntimeError(
                    f"Only {len(pairs)} opposite-view mask pairs found for {views} views. "
                    "Check azimuths and opposite_tolerance_deg, or select legacy_center."
                )
            for i, j, error in pairs:
                li, ri = bboxes[i]
                lj, rj = bboxes[j]
                pair_centers.append(((li + ri) / 2.0 + (lj + rj) / 2.0) / 2.0)
            axis_x = float(statistics.median(pair_centers))

        side, x, y = _canvas_coordinates(width, height, axis_x, align, legacy)
        delta = float(axis_x - width / 2.0)
        fov_original = float(fov_deg)
        if not math.isfinite(fov_original) or not (0.1 <= fov_original < 179.0):
            raise ValueError(f"Invalid fov_deg={fov_deg}")
        base_side = max(width, height)
        # Preserve pixel focal length when the square canvas becomes larger.
        # The original max(fov_x,fov_y) describes the pre-expansion long side.
        canvas_fov = fov_original
        if not legacy and correct_canvas_fov and side > base_side:
            canvas_fov = math.degrees(2.0 * math.atan(
                (side / base_side) * math.tan(math.radians(fov_original) / 2.0)
            ))
        pair_mad = (float(statistics.median(abs(z - axis_x) for z in pair_centers))
                    if pair_centers else None)
        report = {
            "method": mode,
            "view_count": views,
            "input_size_wh": [width, height],
            "alpha_threshold": float(alpha_threshold),
            "opposite_pair_count": len(pairs),
            "opposite_pair_error_max_deg": max((p[2] for p in pairs), default=None),
            "axis_x_px": axis_x,
            "axis_delta_px": delta,
            "pair_median_absolute_deviation_px": pair_mad,
            "pair_min_max_px": [min(pair_centers), max(pair_centers)] if pair_centers else None,
            "canvas_size": side,
            "fov_input_deg": fov_original,
            "fov_canvas_deg": canvas_fov,
            "fov_corrected": bool(not legacy and correct_canvas_fov and side > base_side),
            "x_offset": x,
            "y_offset": y,
            "axis_residual_to_canvas_center_px": x + axis_x - side / 2.0,
            "read_policy": "one RGBA alpha view at a time from existing CPU mmap; no image-batch copy",
            "warning": (
                "BBox opposite-pair symmetry approximates the physical rotation axis; "
                "perspective, nonrigid pose, mask errors or mismatched view angles may bias it."
            ),
        }
        print("[Orbit3D Axis BBox Canvas] " + json.dumps(report, ensure_ascii=False), flush=True)
        return (int(side), int(x), int(y), float(axis_x), float(delta),
                json.dumps(report, ensure_ascii=False, indent=2), float(canvas_fov))
