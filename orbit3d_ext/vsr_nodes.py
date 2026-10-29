from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F
import folder_paths

from .realbasicvsr_backend import RealBasicVSRBackend


def _to_nchw_rgb(x):
    if x.ndim != 4 or x.shape[-1] < 3:
        raise ValueError(f"Expected IMAGE [B,H,W,C>=3], got {tuple(x.shape)}")
    return x[..., :3].permute(0, 3, 1, 2).contiguous()


def _to_nchw_alpha(x):
    if x.shape[-1] >= 4:
        return x[..., 3:4].permute(0, 3, 1, 2).contiguous()
    b, h, w, _ = x.shape
    return torch.ones((b, 1, h, w), dtype=x.dtype, device=x.device)


def _to_comfy_rgba(rgb, alpha):
    return torch.cat((rgb, alpha), dim=1).permute(0, 2, 3, 1).contiguous()


def _resize(x, size, mode):
    if mode in ("bilinear", "bicubic"):
        return F.interpolate(x, size=size, mode=mode, align_corners=False)
    return F.interpolate(x, size=size, mode=mode)


def _parse_frame_list(value):
    """Parse VGGT frame list, preserving order and duplicates."""
    text = str(value).strip()
    if not text:
        raise ValueError("frames must not be empty")

    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        body = text[1:-1] if text.startswith("[") and text.endswith("]") else text
        parsed = [part.strip() for part in body.split(",") if part.strip()]

    if isinstance(parsed, (int, float, str)):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise ValueError("frames must be a non-empty list, e.g. [0, 72, 144, 216]")

    out = []
    for i, item in enumerate(parsed):
        if isinstance(item, bool):
            raise ValueError(f"frames[{i}] must be an integer frame index, got bool")
        try:
            value_f = float(item)
        except (TypeError, ValueError) as e:
            raise ValueError(f"frames[{i}]={item!r} is not a number") from e
        if not value_f.is_integer():
            raise ValueError(f"frames[{i}]={item!r} is not an integer frame index")
        out.append(int(value_f))
    return tuple(out)


def _source_frame_index(frame_number, count, label):
    """VGGT Turntable returns direct 0-based source batch indices."""
    idx = int(frame_number)
    if not 0 <= idx < count:
        raise ValueError(
            f"{label}={frame_number} is not a valid 0-based batch index for {count} frames"
        )
    return idx


class TurntableCyclicRealBasicVSR:
    MODEL_NAME = "realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frames": ("STRING", {
                    "default": "[0, 1, 2, 3]",
                    "multiline": False,
                }),
                "temporal_frames": (
                    ["5", "7", "9", "11", "15", "21", "29"],
                    {"default": "29"},
                ),
                "scale": (["2", "4"], {"default": "2"}),
                "tile_size": (
                    "INT",
                    {"default": 384, "min": 64, "max": 2048, "step": 32},
                ),
                "tile_overlap": (
                    "INT",
                    {"default": 48, "min": 0, "max": 512, "step": 8},
                ),
                "cleaning_threshold": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 255.0, "step": 0.5},
                ),
                "alpha_upscale_mode": (
                    ["nearest-exact", "bilinear"],
                    {"default": "nearest-exact"},
                ),
                "debug_print": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("views_sr",)
    FUNCTION = "run"
    CATEGORY = "Turntable/SR"

    def _model_path(self):
        path = os.path.join(
            folder_paths.models_dir,
            "video_sr",
            "realbasicvsr",
            self.MODEL_NAME,
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "RealBasicVSR checkpoint not found:\n"
                f"  {path}\n\n"
                "Put the official checkpoint in:\n"
                "  ComfyUI/models/video_sr/realbasicvsr/"
            )
        return path

    def _process_one(
        self,
        rgb,
        alpha,
        center,
        temporal_frames,
        backend,
        alpha_mode,
        tag,
        debug,
    ):
        n = rgb.shape[0]
        radius = temporal_frames // 2
        indices = tuple(
            (center + offset) % n
            for offset in range(-radius, radius + 1)
        )
        if debug:
            print(
                f"[Turntable RealBasicVSR] {tag}: source center={center} (0-based), "
                f"T={temporal_frames}, indices={indices}"
            )

        # Cyclic context: negative/overflow offsets wrap around the turntable batch.
        sequence = torch.stack([rgb[i] for i in indices], dim=0)
        sr_rgb = backend.enhance_center(sequence).float().cpu().clamp_(0.0, 1.0)
        h2, w2 = sr_rgb.shape[-2:]

        # Alpha stays tied to the exact selected source frame.
        a = alpha[center:center + 1].float().cpu()
        mode = "nearest-exact" if alpha_mode == "nearest-exact" else "bilinear"
        sr_alpha = _resize(a, (h2, w2), mode).clamp_(0.0, 1.0)
        return _to_comfy_rgba(sr_rgb, sr_alpha)

    @torch.inference_mode()
    def run(
        self,
        images,
        frames,
        temporal_frames,
        scale,
        tile_size,
        tile_overlap,
        cleaning_threshold,
        alpha_upscale_mode,
        debug_print,
    ):
        if images.ndim != 4:
            raise ValueError(f"Expected IMAGE batch [B,H,W,C], got {tuple(images.shape)}")

        n = int(images.shape[0])
        t = int(temporal_frames)
        if n < 3:
            raise ValueError(f"Need >=3 video frames, got {n}")
        if t < 3 or t % 2 == 0:
            raise ValueError(f"temporal_frames must be odd and >=3, got {t}")
        if t > n:
            raise ValueError(f"temporal_frames={t} exceeds video batch size {n}")

        # VGGT returns direct 0-based source batch indices as a JSON-style string.
        # Preserve exactly the order supplied by VGGT; do NOT subtract 1.
        requested_frames = _parse_frame_list(frames)
        centers = tuple(
            _source_frame_index(frame, n, f"frames[{i}]")
            for i, frame in enumerate(requested_frames)
        )

        rgb = _to_nchw_rgb(images).float().cpu()
        alpha = _to_nchw_alpha(images).float().cpu()

        if debug_print:
            h, w = rgb.shape[-2:]
            print(
                f"[Turntable RealBasicVSR] batch={n}, input={w}x{h}, "
                f"centers={centers} [DIRECT 0-BASED VGGT INDICES, ORDER PRESERVED], "
                f"T={t}, scale=x{scale}, tile={tile_size}, overlap={tile_overlap}, "
                f"clean_threshold={cleaning_threshold}"
            )
            print("[Turntable RealBasicVSR] CYCLIC wrap-around is ON")

        backend = RealBasicVSRBackend(
            checkpoint_path=self._model_path(),
            device="cuda" if torch.cuda.is_available() else "cpu",
            output_scale=int(scale),
            tile_size=int(tile_size),
            tile_overlap=int(tile_overlap),
            dynamic_refine_thres=float(cleaning_threshold),
        )

        try:
            outputs = [
                self._process_one(
                    rgb,
                    alpha,
                    center,
                    t,
                    backend,
                    alpha_upscale_mode,
                    f"item_{i}:frame_{center}",
                    debug_print,
                )
                for i, center in enumerate(centers)
            ]
            output_batch = torch.cat(outputs, dim=0)
        finally:
            backend.close()
            del backend
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return (output_batch,)


NODE_CLASS_MAPPINGS = {
    "TurntableCyclicRealBasicVSR": TurntableCyclicRealBasicVSR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TurntableCyclicRealBasicVSR": "Turntable Cyclic RealBasicVSR",
}
