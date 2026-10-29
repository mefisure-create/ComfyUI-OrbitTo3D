import gc
import json
import math
import sys
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TVF

import comfy.model_management as model_management
import comfy.model_patcher as model_patcher
import comfy.utils


DEFAULT_TARGET_ANGLES = "0, 90, 180, 270"
_GIB = 1024 ** 3


def _parse_target_angles(value):
    """Parse a user supplied angle list while preserving order and duplicates."""
    text = str(value).strip()
    if not text:
        raise ValueError("target_angles must not be empty")

    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Also accept a plain comma separated string: 0, 90, 180, 270
        body = text[1:-1] if text.startswith("[") and text.endswith("]") else text
        parsed = [part.strip() for part in body.split(",") if part.strip()]

    if isinstance(parsed, (int, float, str)):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise ValueError(
            "target_angles must be a non-empty comma-separated list, e.g. 0, 90, 180, 270"
        )

    out = []
    for i, item in enumerate(parsed):
        if isinstance(item, bool):
            raise ValueError(f"target_angles[{i}] must be a number, got bool")
        try:
            angle = float(item)
        except (TypeError, ValueError) as e:
            raise ValueError(f"target_angles[{i}]={item!r} is not a number") from e
        if not math.isfinite(angle):
            raise ValueError(f"target_angles[{i}] must be finite, got {angle}")
        if not 0.0 <= angle <= 360.0:
            raise ValueError(
                f"target_angles[{i}]={angle} is outside the supported 0..360 degree range"
            )
        out.append(angle)

    return tuple(out)


def _angle_text(angle):
    value = float(angle)
    return str(int(value)) if value.is_integer() else f"{value:g}"


class _ManagedVGGT(torch.nn.Module):
    """Thin wrapper that gives ComfyUI a normal nn.Module with a dtype query."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, images):
        return self.inner(images)

    def get_dtype(self):
        try:
            return next(self.inner.parameters()).dtype
        except StopIteration:
            return torch.float32


class _VGGTCache:
    lock = threading.RLock()
    key = None
    patcher = None
    encoding_to_camera = None

    @classmethod
    def _unload_old(cls):
        if cls.patcher is None:
            return
        try:
            fn = getattr(model_management, "unload_model_and_clones", None)
            if callable(fn):
                fn(cls.patcher)
        except Exception as e:
            print(f"[VGGT Turntable Native] Warning while unloading old cached model: {e}")
        cls.key = None
        cls.patcher = None
        cls.encoding_to_camera = None
        gc.collect()
        try:
            model_management.soft_empty_cache()
        except Exception:
            pass

    @classmethod
    def get(cls, repo: Path, checkpoint: Path):
        key = (str(repo.resolve()), str(checkpoint.resolve()))
        with cls.lock:
            if cls.key == key and cls.patcher is not None:
                return cls.patcher, cls.encoding_to_camera

            cls._unload_old()

            repo_s = str(repo.resolve())
            if repo_s not in sys.path:
                sys.path.insert(0, repo_s)

            try:
                from vggt_omega.models import VGGTOmega
                from vggt_omega.utils.pose_enc import encoding_to_camera
            except Exception as e:
                raise RuntimeError(
                    "Could not import VGGT-Omega from the configured repository. "
                    f"repo={repo}\n{type(e).__name__}: {e}"
                ) from e

            print(f"[VGGT Turntable Native] Creating VGGT-Omega on CPU")
            inner = VGGTOmega().eval()
            print(f"[VGGT Turntable Native] Loading checkpoint on CPU: {checkpoint}")
            state = torch.load(str(checkpoint), map_location="cpu")
            try:
                result = inner.load_state_dict(state, assign=True)
            except TypeError:
                result = inner.load_state_dict(state)
            del state

            if getattr(result, "missing_keys", None) or getattr(result, "unexpected_keys", None):
                raise RuntimeError(
                    "VGGT checkpoint did not load cleanly. "
                    f"missing={list(result.missing_keys)[:20]}, "
                    f"unexpected={list(result.unexpected_keys)[:20]}"
                )

            wrapper = _ManagedVGGT(inner).eval()
            load_device = model_management.get_torch_device()
            offload_device = torch.device("cpu")
            patcher = model_patcher.ModelPatcher(
                wrapper,
                load_device=load_device,
                offload_device=offload_device,
            )

            # Keep the patcher strongly referenced so ComfyUI's loaded-model registry
            # can manage it rather than seeing a dead weakref after this call returns.
            cls.key = key
            cls.patcher = patcher
            cls.encoding_to_camera = encoding_to_camera
            print(
                "[VGGT Turntable Native] Model registered with ComfyUI: "
                f"weights={patcher.model_size() / _GIB:.2f} GiB, "
                f"load_device={load_device}, offload_device={offload_device}"
            )
            return patcher, encoding_to_camera


# ---------- preprocessing, intentionally equivalent to VGGT-Omega load_fn ----------

def _round_half_to_even(x: float) -> int:
    # Python round and np.round both use bankers rounding for the values encountered here.
    return int(round(float(x)))


def _balanced_target_shape(aspect_ratio, image_resolution, patch_size=16):
    token_number = (image_resolution // patch_size) ** 2
    w_patches = math.sqrt(token_number / aspect_ratio)
    h_patches = token_number / w_patches
    w_patches = max(1, _round_half_to_even(w_patches))
    h_patches = max(1, _round_half_to_even(h_patches))
    return h_patches * patch_size, w_patches * patch_size


def _max_size_target_shape(aspect_ratio, image_resolution, patch_size=16):
    def rpm(v):
        return max(patch_size, _round_half_to_even(float(v) / patch_size) * patch_size)

    if aspect_ratio >= 1.0:
        return image_resolution, rpm(image_resolution / aspect_ratio)
    return rpm(image_resolution * aspect_ratio), image_resolution


def _tensor_to_rgb_pil(t: torch.Tensor) -> Image.Image:
    arr = t.detach().cpu().float().clamp(0, 1).numpy()
    if arr.ndim != 3:
        raise RuntimeError(f"Expected HWC image tensor, got {arr.shape}")
    c = arr.shape[-1]
    if c == 1:
        arr = np.repeat(arr, 3, axis=-1)
        c = 3
    if c not in (3, 4):
        if c > 4:
            arr = arr[..., :4]
            c = 4
        else:
            raise RuntimeError(f"Unsupported image channel count: {c}")
    arr = np.rint(arr * 255.0).astype(np.uint8)
    if c == 4:
        rgba = Image.fromarray(arr, mode="RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return Image.fromarray(arr, mode="RGB")


def _crop_supported_aspect(image: Image.Image, min_ar=0.5, max_ar=2.0):
    width, height = image.size
    ar = height / max(width, 1)
    if ar < min_ar:
        crop_width = min(width, max(1, int(round(height / min_ar))))
        left = max((width - crop_width) // 2, 0)
        return image.crop((left, 0, left + crop_width, height))
    if ar > max_ar:
        crop_height = min(height, max(1, int(round(width * max_ar))))
        top = max((height - crop_height) // 2, 0)
        return image.crop((0, top, width, top + crop_height))
    return image


def _preprocess_selected(images, indices, image_resolution, resize_mode, interrupt_cb):
    processed = []
    shapes = set()
    for j, idx in enumerate(indices):
        if j % 8 == 0:
            interrupt_cb()
        image = _crop_supported_aspect(_tensor_to_rgb_pil(images[int(idx)]))
        width, height = image.size
        ar = height / max(width, 1)
        if resize_mode == "balanced":
            target_h, target_w = _balanced_target_shape(ar, image_resolution, 16)
        else:
            target_h, target_w = _max_size_target_shape(ar, image_resolution, 16)
        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        tensor = TVF.to_tensor(image)
        processed.append(tensor)
        shapes.add((target_h, target_w))

    if len(shapes) > 1:
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)
        padded = []
        for x in processed:
            hp = max_h - x.shape[1]
            wp = max_w - x.shape[2]
            if hp or wp:
                top = hp // 2
                bottom = hp - top
                left = wp // 2
                right = wp - left
                x = torch.nn.functional.pad(x, (left, right, top, bottom), value=1.0)
            padded.append(x)
        processed = padded
    return torch.stack(processed, dim=0)


# ---------- camera FOV ----------

def _fov_stats_from_intrinsics(intrinsics, image_hw):
    """
    Convert VGGT pixel-space intrinsics to horizontal/vertical FOV in degrees.

    intrinsics: [S,3,3] or [1,S,3,3]
    image_hw:   (H, W) used by encoding_to_camera()
    """
    k = np.asarray(intrinsics, dtype=np.float64)
    if k.ndim == 4 and k.shape[0] == 1:
        k = k[0]
    if k.ndim != 3 or k.shape[-2:] != (3, 3):
        raise RuntimeError(f"Unexpected VGGT intrinsics shape: {k.shape}")

    h, w = int(image_hw[0]), int(image_hw[1])
    fx = k[:, 0, 0]
    fy = k[:, 1, 1]

    valid_x = np.isfinite(fx) & (fx > 0)
    valid_y = np.isfinite(fy) & (fy > 0)
    if not np.any(valid_x):
        raise RuntimeError("VGGT returned no valid positive fx values")
    if not np.any(valid_y):
        raise RuntimeError("VGGT returned no valid positive fy values")

    fov_x = np.full_like(fx, np.nan, dtype=np.float64)
    fov_y = np.full_like(fy, np.nan, dtype=np.float64)

    fov_x[valid_x] = np.degrees(2.0 * np.arctan(float(w) / (2.0 * fx[valid_x])))
    fov_y[valid_y] = np.degrees(2.0 * np.arctan(float(h) / (2.0 * fy[valid_y])))

    def stats(v):
        vv = v[np.isfinite(v)]
        return {
            "count": int(vv.size),
            "median_deg": float(np.median(vv)),
            "mean_deg": float(np.mean(vv)),
            "std_deg": float(np.std(vv)),
            "min_deg": float(np.min(vv)),
            "p10_deg": float(np.percentile(vv, 10)),
            "p90_deg": float(np.percentile(vv, 90)),
            "max_deg": float(np.max(vv)),
        }

    return {
        "image_hw": [h, w],
        "fov_x": stats(fov_x),
        "fov_y": stats(fov_y),
        "per_view_fov_x_deg": [None if not np.isfinite(x) else float(x) for x in fov_x],
        "per_view_fov_y_deg": [None if not np.isfinite(y) else float(y) for y in fov_y],
    }


# ---------- orbit math ----------

def _isotonic_increasing(y):
    y = np.asarray(y, dtype=np.float64)
    means, weights, starts, ends = [], [], [], []
    for i, val in enumerate(y):
        means.append(float(val)); weights.append(1.0); starts.append(i); ends.append(i)
        while len(means) >= 2 and means[-2] > means[-1]:
            w = weights[-2] + weights[-1]
            m = (means[-2] * weights[-2] + means[-1] * weights[-1]) / w
            s, e = starts[-2], ends[-1]
            means[-2:] = [m]; weights[-2:] = [w]; starts[-2:] = [s]; ends[-2:] = [e]
    out = np.empty_like(y)
    for m, s, e in zip(means, starts, ends):
        out[s:e + 1] = m
    return out


def _camera_centers(extrinsics):
    e = np.asarray(extrinsics)
    if e.ndim == 4 and e.shape[0] == 1:
        e = e[0]
    R = e[:, :3, :3]
    t = e[:, :3, 3]
    return -np.einsum("sji,sj->si", R, t)


def _fit_orbit(centers, fit_mask=None):
    c = np.asarray(centers, dtype=np.float64)
    if fit_mask is None:
        fit_mask = np.ones(len(c), dtype=bool)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    cf = c[fit_mask]
    if len(cf) < 4:
        raise RuntimeError("Need at least 4 anchor cameras to fit the orbit")

    mean = cf.mean(axis=0)
    _, svals, vt = np.linalg.svd(cf - mean, full_matrices=False)
    basis = vt[:2]
    xy = (c - mean) @ basis.T
    xy_fit = xy[fit_mask]
    x, y = xy_fit[:, 0], xy_fit[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    cx, cy, c0 = np.linalg.lstsq(A, b, rcond=None)[0]
    radius = float(np.sqrt(max(c0 + cx * cx + cy * cy, 0.0)))

    rr = np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
    theta = np.unwrap(np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx))
    if np.median(np.diff(theta)) < 0:
        theta = -theta
    rel = np.rad2deg(theta - theta[0])
    rel -= rel[0]
    iso = _isotonic_increasing(rel)
    iso -= iso[0]

    rr_fit = rr[fit_mask]
    return {
        "rel_deg": rel,
        "iso_deg": iso,
        "span_raw_deg": float(rel[-1] - rel[0]),
        "span_iso_deg": float(iso[-1] - iso[0]),
        "monotonic_ratio": float(np.mean(np.diff(rel) >= -2.0)) if len(rel) > 1 else 1.0,
        "plane_flatness": float(svals[2] / max(svals[1], 1e-12)) if len(svals) >= 3 else float("nan"),
        "radius": radius,
        "radius_cv": float(rr_fit.std() / max(rr_fit.mean(), 1e-12)),
    }


def _nearest(source_frames, angles, target):
    idx = int(np.argmin(np.abs(np.asarray(angles) - target)))
    return {
        "source_frame": int(source_frames[idx]),
        "sample_index": idx,
        "angle_deg": float(angles[idx]),
        "error_deg": float(angles[idx] - target),
    }


def _orbit_summary(frames, orbit, targets):
    return {
        "input_views": int(len(frames)),
        "span_raw_deg": orbit["span_raw_deg"],
        "span_isotonic_deg": orbit["span_iso_deg"],
        "monotonic_ratio_tolerance_2deg": orbit["monotonic_ratio"],
        "plane_flatness_s3_over_s2": orbit["plane_flatness"],
        "orbit_radius": orbit["radius"],
        "orbit_radius_cv": orbit["radius_cv"],
        "targets_raw": [
            {"target_deg": float(t), **_nearest(frames, orbit["rel_deg"], t)}
            for t in targets
        ],
        "targets_isotonic": [
            {"target_deg": float(t), **_nearest(frames, orbit["iso_deg"], t)}
            for t in targets
        ],
    }


class VGGTTurntableFourViewsNative:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "target_angles": ("STRING", {
                    "default": DEFAULT_TARGET_ANGLES,
                    "multiline": False,
                }),
                "clockwise": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1}),
                "checkpoint": ("STRING", {
                    "default": "~/vggt-omega/checkpoints/VGGT-Omega-1B-512/vggt_omega_1b_512.pt",
                    "multiline": False,
                }),
                "vggt_repo": ("STRING", {
                    "default": "~/vggt-omega",
                    "multiline": False,
                }),
                "coarse_step": ("INT", {"default": 5, "min": 2, "max": 20, "step": 1}),
                "anchor_step": ("INT", {"default": 9, "min": 3, "max": 30, "step": 1}),
                "refine_radius": ("INT", {"default": 12, "min": 3, "max": 40, "step": 1}),
                "resolution": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 16}),
                "resize_mode": (["balanced", "max_size"], {"default": "balanced"}),
                "inference_headroom_gb": ("FLOAT", {"default": 8.0, "min": 2.0, "max": 20.0, "step": 0.5}),
                "strict_validation": ("BOOLEAN", {"default": True}),
                "offload_after_run": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = (
        "views", "frames", "report", "fov_x_deg", "fov_y_deg",
    )
    FUNCTION = "run"
    CATEGORY = "Frame Tools/Angular Search"

    @staticmethod
    def _interrupt():
        fn = getattr(model_management, "throw_exception_if_processing_interrupted", None)
        if callable(fn):
            fn()

    @staticmethod
    def _ensure_model_loaded(patcher, headroom_gb):
        headroom = int(float(headroom_gb) * _GIB)
        model_management.load_models_gpu(
            [patcher],
            memory_required=headroom,
            minimum_memory_required=headroom,
            force_full_load=True,
        )

    def _infer(self, patcher, encoding_to_camera, images, indices, resolution, resize_mode, headroom_gb, label):
        self._interrupt()
        cpu_batch = _preprocess_selected(images, indices, resolution, resize_mode, self._interrupt)
        print(
            f"[VGGT Turntable Native] {label}: {len(indices)} views, "
            f"preprocessed={tuple(cpu_batch.shape)}"
        )

        # Ask the central ComfyUI model manager for model VRAM + activation headroom.
        self._ensure_model_loaded(patcher, headroom_gb)
        device = patcher.load_device
        batch = cpu_batch.to(device=device, non_blocking=False)
        del cpu_batch

        self._interrupt()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        with torch.inference_mode():
            predictions = patcher.model(batch)
            image_hw = tuple(int(x) for x in predictions["images"].shape[-2:])
            extrinsics, intrinsics = encoding_to_camera(
                predictions["pose_enc"], image_hw
            )

        peak_gib = None
        if device.type == "cuda":
            peak_gib = float(torch.cuda.max_memory_allocated(device) / _GIB)

        extr = extrinsics.detach().float().cpu().numpy()
        if extr.ndim == 4 and extr.shape[0] == 1:
            extr = extr[0]

        intr = intrinsics.detach().float().cpu().numpy()
        fov_stats = _fov_stats_from_intrinsics(intr, image_hw)

        print(
            f"[VGGT Turntable Native] {label}: "
            f"FOVx median={fov_stats['fov_x']['median_deg']:.3f} deg, "
            f"p10={fov_stats['fov_x']['p10_deg']:.3f}, "
            f"p90={fov_stats['fov_x']['p90_deg']:.3f}; "
            f"FOVy median={fov_stats['fov_y']['median_deg']:.3f} deg"
        )

        del intr, intrinsics, extrinsics, predictions, batch
        try:
            model_management.soft_empty_cache()
        except Exception:
            pass
        self._interrupt()
        return extr, peak_gib, fov_stats

    def run(
        self,
        images,
        target_angles,
        clockwise,
        checkpoint,
        vggt_repo,
        coarse_step,
        anchor_step,
        refine_radius,
        resolution,
        resize_mode,
        inference_headroom_gb,
        strict_validation,
        offload_after_run,
    ):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise RuntimeError("images must be a ComfyUI IMAGE batch [B,H,W,C]")
        n = int(images.shape[0])
        if n < 20:
            raise RuntimeError(f"Expected a full turntable sequence; got only {n} frames")
        if int(coarse_step) <= 0 or int(anchor_step) <= 0:
            raise RuntimeError("coarse_step and anchor_step must be positive")

        targets = _parse_target_angles(target_angles)
        clockwise = int(clockwise)
        if clockwise not in (0, 1):
            raise ValueError(f"clockwise must be 0 or 1, got {clockwise}")

        # target_angles are always expressed in the objective object convention:
        # positive angles mean the OBJECT rotates clockwise when viewed from above.
        # The recovered VGGT orbit is normalized only by temporal progression, so
        # counter-clockwise source rotation must mirror the requested angle.
        lookup_targets = tuple(
            float(t) if clockwise == 1 else (360.0 - float(t)) % 360.0
            for t in targets
        )

        repo = Path(vggt_repo).expanduser().resolve()
        ckpt = Path(checkpoint).expanduser().resolve()
        if not repo.is_dir():
            raise RuntimeError(f"VGGT repository not found: {repo}")
        if not ckpt.is_file():
            raise RuntimeError(f"VGGT checkpoint not found: {ckpt}")

        patcher, encoding_to_camera = _VGGTCache.get(repo, ckpt)

        # Keep each refinement inference at roughly the same target-window load as
        # the original 4-view node (three searched targets: 90/180/270). 0 degrees
        # is exact by definition because the first source frame is the reference.
        refine_target_indices = [
            i for i, t in enumerate(lookup_targets) if abs(float(t)) > 1e-9
        ]
        refine_groups = [
            refine_target_indices[i:i + 3]
            for i in range(0, len(refine_target_indices), 3)
        ]
        progress = comfy.utils.ProgressBar(1 + len(refine_groups))

        # PASS 1: uniform sparse global orbit.
        coarse_indices = list(range(0, n, int(coarse_step)))
        if coarse_indices[-1] != n - 1:
            coarse_indices.append(n - 1)
        c_extr, c_peak, c_fov = self._infer(
            patcher, encoding_to_camera, images, coarse_indices,
            int(resolution), str(resize_mode), float(inference_headroom_gb),
            "PASS 1 / coarse global orbit",
        )
        c_frames = np.asarray(coarse_indices, dtype=np.int64)
        c_orbit = _fit_orbit(_camera_centers(c_extr))
        c_result = _orbit_summary(c_frames, c_orbit, lookup_targets)
        progress.update(1)

        coarse_matches = [
            _nearest(c_frames, c_orbit["iso_deg"], t) for t in lookup_targets
        ]
        coarse_target_frames = [int(match["source_frame"]) for match in coarse_matches]
        print(
            "[VGGT Turntable Native] coarse: "
            + ", ".join(
                f"{_angle_text(req)}°->lookup {_angle_text(lookup)}°={frame}"
                for req, lookup, frame in zip(targets, lookup_targets, coarse_target_frames)
            )
        )

        # Build dense windows for searched targets. 0° does not need a window: it
        # is always source frame 0 because all angles are defined relative to it.
        anchors = set(range(0, n, int(anchor_step)))
        anchors.add(n - 1)
        windows = []
        window_by_index = {}
        for i, (requested_t, lookup_t, cf) in enumerate(
            zip(targets, lookup_targets, coarse_target_frames)
        ):
            if abs(float(lookup_t)) <= 1e-9:
                info = {
                    "target_index": i,
                    "target_deg": float(requested_t),
                    "lookup_deg": float(lookup_t),
                    "coarse_frame": 0,
                    "window": [0, 0],
                    "exact_reference_frame": True,
                }
            else:
                lo = max(0, cf - int(refine_radius))
                hi = min(n - 1, cf + int(refine_radius))
                info = {
                    "target_index": i,
                    "target_deg": float(requested_t),
                    "lookup_deg": float(lookup_t),
                    "coarse_frame": int(cf),
                    "window": [lo, hi],
                    "exact_reference_frame": False,
                }
                window_by_index[i] = (lo, hi)
            windows.append(info)

        # Fill exact reference-angle matches now; all other targets are refined in
        # groups of at most three so large angle lists do not create a huge batch.
        final_matches = [None] * len(targets)
        for i, lookup_t in enumerate(lookup_targets):
            if abs(float(lookup_t)) <= 1e-9:
                final_matches[i] = {
                    "source_frame": 0,
                    "sample_index": 0,
                    "angle_deg": 0.0,
                    "error_deg": 0.0,
                }

        refined_groups_report = []
        refined_orbits = []
        refined_fovs = []
        refined_peaks = []

        for group_number, group_indices in enumerate(refine_groups, start=1):
            refined_set = set(anchors)
            for target_index in group_indices:
                lo, hi = window_by_index[target_index]
                refined_set.update(range(lo, hi + 1))
            refined_indices = sorted(refined_set)

            group_targets = [targets[i] for i in group_indices]
            group_lookup_targets = [lookup_targets[i] for i in group_indices]
            r_extr, r_peak, r_fov = self._infer(
                patcher, encoding_to_camera, images, refined_indices,
                int(resolution), str(resize_mode), float(inference_headroom_gb),
                f"PASS 2.{group_number}/{len(refine_groups)} / anchors + dense windows",
            )
            r_frames = np.asarray(refined_indices, dtype=np.int64)
            fit_mask = np.asarray([int(x) in anchors for x in r_frames], dtype=bool)
            r_orbit = _fit_orbit(_camera_centers(r_extr), fit_mask=fit_mask)
            r_result = _orbit_summary(r_frames, r_orbit, group_lookup_targets)

            group_matches = [
                _nearest(r_frames, r_orbit["iso_deg"], t)
                for t in group_lookup_targets
            ]
            for target_index, match in zip(group_indices, group_matches):
                final_matches[target_index] = match

            refined_orbits.append(r_orbit)
            refined_fovs.append(r_fov)
            refined_peaks.append(r_peak)
            refined_groups_report.append({
                "group": group_number,
                "target_indices": group_indices,
                "target_angles_deg": [float(t) for t in group_targets],
                "lookup_angles_deg": [float(t) for t in group_lookup_targets],
                "input_views": len(refined_indices),
                "orbit": r_result,
                "camera_fov": r_fov,
                "peak_cuda_allocated_gib": r_peak,
            })
            progress.update(1)

        if any(match is None for match in final_matches):
            raise RuntimeError("Internal error: not all requested target angles were resolved")

        selected_frames = [int(match["source_frame"]) for match in final_matches]
        warnings = []
        if c_orbit["monotonic_ratio"] < 0.95:
            warnings.append("coarse trajectory is not sufficiently monotonic")
        if not (330.0 <= c_orbit["span_raw_deg"] <= 390.0):
            warnings.append("coarse raw span is not close to 360 degrees")
        if c_orbit["plane_flatness"] > 0.10:
            warnings.append("coarse camera orbit is not sufficiently planar")
        if c_orbit["radius_cv"] > 0.10:
            warnings.append("coarse camera orbit radius varies too much")

        for group_number, r_orbit in enumerate(refined_orbits, start=1):
            prefix = f"refinement group {group_number}"
            if r_orbit["monotonic_ratio"] < 0.95:
                warnings.append(f"{prefix} trajectory is not sufficiently monotonic")
            if not (330.0 <= r_orbit["span_raw_deg"] <= 390.0):
                warnings.append(f"{prefix} raw span is not close to 360 degrees")
            if r_orbit["plane_flatness"] > 0.10:
                warnings.append(f"{prefix} camera orbit is not sufficiently planar")
            if r_orbit["radius_cv"] > 0.10:
                warnings.append(f"{prefix} camera orbit radius varies too much")
        cross_pass = []
        for requested_t, lookup_t, cf, match in zip(
            targets, lookup_targets, coarse_target_frames, final_matches
        ):
            rf = int(match["source_frame"])
            delta = rf - cf
            cross_pass.append({
                "target_deg": float(requested_t),
                "lookup_deg": float(lookup_t),
                "coarse_frame": int(cf),
                "refined_frame": rf,
                "delta_frames": delta,
            })
            if abs(delta) > max(int(refine_radius), int(coarse_step) * 2):
                warnings.append(
                    f"target {_angle_text(requested_t)} (lookup {_angle_text(lookup_t)}) "
                    f"moved unexpectedly far between passes ({cf} -> {rf})"
                )

        if refined_fovs:
            selected_fov_x = float(np.median([f["fov_x"]["median_deg"] for f in refined_fovs]))
            selected_fov_y = float(np.median([f["fov_y"]["median_deg"] for f in refined_fovs]))
            fov_selection = "median_of_refinement_group_medians"
        else:
            selected_fov_x = float(c_fov["fov_x"]["median_deg"])
            selected_fov_y = float(c_fov["fov_y"]["median_deg"])
            fov_selection = "coarse_median_no_refinement_needed"

        report_obj = {
            "implementation": "native_comfyui_modelpatcher_v4_dynamic_targets_clockwise",
            "source_frames": n,
            "clockwise": clockwise,
            "rotation_direction": "clockwise" if clockwise == 1 else "counterclockwise",
            "angle_convention": "object_clockwise_positive",
            "target_angles_deg": [float(t) for t in targets],
            "lookup_angles_deg": [float(t) for t in lookup_targets],
            "selected_frames": selected_frames,
            "selected": [
                {
                    "target_deg": float(requested_t),
                    "lookup_deg": float(lookup_t),
                    **match,
                }
                for requested_t, lookup_t, match in zip(
                    targets, lookup_targets, final_matches
                )
            ],
            "coarse_step": int(coarse_step),
            "anchor_step": int(anchor_step),
            "refine_radius": int(refine_radius),
            "dense_windows": windows,
            "coarse": c_result,
            "refined_groups": refined_groups_report,
            "cross_pass": cross_pass,
            "peak_cuda_allocated_gib": {
                "coarse": c_peak,
                "refined_groups": refined_peaks,
            },
            "camera_fov": {
                "selected_fov_x_deg": selected_fov_x,
                "selected_fov_y_deg": selected_fov_y,
                "selection": fov_selection,
                "coarse": c_fov,
                "refined_groups": refined_fovs,
            },
            "model_weight_gib": float(patcher.model_size() / _GIB),
            "inference_headroom_gb": float(inference_headroom_gb),
            "warnings": warnings,
        }
        report = json.dumps(report_obj, indent=2, ensure_ascii=False)

        selected_text = ", ".join(
            f"{_angle_text(t)}°={frame}"
            for t, frame in zip(targets, selected_frames)
        )
        print(
            f"[VGGT Turntable Native] FINAL: {selected_text}; "
            f"coarse_span={c_orbit['span_raw_deg']:.2f}°, "
            f"coarse_monotonic={c_orbit['monotonic_ratio']:.3f}, "
            f"refine_groups={len(refine_groups)}; "
            f"FOVx={selected_fov_x:.3f}°, FOVy={selected_fov_y:.3f}°"
        )

        if warnings:
            print("[VGGT Turntable Native] Validation warnings:")
            for w in warnings:
                print(f"  - {w}")
            if strict_validation:
                if offload_after_run:
                    try:
                        model_management.unload_model_and_clones(patcher)
                        model_management.soft_empty_cache()
                    except Exception:
                        pass
                raise RuntimeError("VGGT reconstruction failed strict validation:\n- " + "\n- ".join(warnings))

        views = images[selected_frames]
        frames = json.dumps(selected_frames, ensure_ascii=False)

        if offload_after_run:
            print("[VGGT Turntable Native] Offloading VGGT after run by request")
            model_management.unload_model_and_clones(patcher)
            model_management.soft_empty_cache()

        return (
            views, frames, report,
            selected_fov_x, selected_fov_y,
        )


NODE_CLASS_MAPPINGS = {
    "VGGTTurntableFourViewsNative": VGGTTurntableFourViewsNative,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VGGTTurntableFourViewsNative": "VGGT Turntable Views (Native)",
}
