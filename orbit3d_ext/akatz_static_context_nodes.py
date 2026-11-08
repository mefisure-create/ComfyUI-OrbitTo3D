
"""Orbit3D patch 3.6.13: Akatz static loop context.

Purpose:
  * carry all large/expensive loop-invariant objects through Akatz value4;
  * remove recursive body dependencies on VSR/VGGT/Prepare Canvas/model loaders/
    CacheSession, so RAM-pressure eviction cannot force those upstream branches
    to execute again mid-loop;
  * persist only the tiny FOV scalar in the cache-session directory so post-loop
    MultiView does not need to keep the full static context alive.

No reconstruction formulas, camera math, model inference, seeds, visibility
weighting, or VSR/VGGT mechanics are changed here.  The context is a reference
bundle: tensors/models are not cloned, concatenated, moved, or materialized.
"""
from __future__ import annotations

from pathlib import Path
import json
import os

_META_NAME = "loop_static_context.json"
_KIND = "loop_static_context"
_META_KIND = "loop_static_context_meta"


def _validated_session(cache_session):
    if not isinstance(cache_session, dict):
        raise TypeError("cache_session must be an ORBIT3D cache-session dict")
    if cache_session.get("orbit3d_cache_type") != "orbit3d_session":
        raise ValueError(
            "Invalid Orbit3D cache session type: "
            f"{cache_session.get('orbit3d_cache_type')!r}"
        )
    session_id = str(cache_session.get("session_id", "")).strip()
    root_text = str(cache_session.get("root", "")).strip()
    if not session_id or not root_text:
        raise ValueError("Orbit3D cache session is missing session_id/root")
    root = Path(root_text).resolve()
    if not root.is_dir():
        raise ValueError(f"Orbit3D cache session root does not exist: {root}")
    session_file = root / "session.json"
    if not session_file.is_file():
        raise ValueError(f"Orbit3D cache session marker missing: {session_file}")
    try:
        marker = json.loads(session_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid Orbit3D session marker: {session_file}: {exc}") from exc
    if str(marker.get("session_id", "")) != session_id:
        raise ValueError("Orbit3D cache session marker/session_id mismatch")
    marker_root = str(marker.get("root", "")).strip()
    if marker_root and Path(marker_root).resolve() != root:
        raise ValueError("Orbit3D cache session marker/root mismatch")
    return root, session_id


def _persist_fov(cache_session, fov):
    root, session_id = _validated_session(cache_session)
    fov_value = float(fov)
    meta = {
        "orbit3d_cache_type": _META_KIND,
        "version": 1,
        "session_id": session_id,
        "fov": fov_value,
    }
    path = root / _META_NAME
    tmp = root / f".{_META_NAME}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return fov_value


class Orbit3DStaticLoopContextPack:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "background": ("IMAGE",),
            "canvas": ("IMAGE",),
            "x": ("INT", {"forceInput": True}),
            "y": ("INT", {"forceInput": True}),
            "diffuse": ("IMAGE",),
            "whole_mask": ("MASK",),
            "width": ("INT", {"forceInput": True}),
            "height": ("INT", {"forceInput": True}),
            "black": ("IMAGE",),
            "moge_model": ("MOGE_MODEL",),
            "fov": ("FLOAT", {"forceInput": True}),
            "normal_model": ("MARIGOLDV2_MODEL",),
            "normal_vae": ("VAE",),
            "depth_model": ("MARIGOLDV2_MODEL",),
            "depth_vae": ("VAE",),
            "cache_session": ("ORBIT3D_CACHE_SESSION",),
        }}

    RETURN_TYPES = ("ORBIT3D_LOOP_STATIC_CONTEXT",)
    RETURN_NAMES = ("context",)
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"

    def run(
        self, background, canvas, x, y, diffuse, whole_mask, width, height,
        black, moge_model, fov, normal_model, normal_vae, depth_model,
        depth_vae, cache_session,
    ):
        fov_value = _persist_fov(cache_session, fov)
        context = {
            "orbit3d_cache_type": _KIND,
            "version": 1,
            # Reference-preserving payload: DO NOT clone/copy tensors or models.
            "background": background,
            "canvas": canvas,
            "x": int(x),
            "y": int(y),
            "diffuse": diffuse,
            "whole_mask": whole_mask,
            "width": int(width),
            "height": int(height),
            "black": black,
            "moge_model": moge_model,
            "fov": fov_value,
            "normal_model": normal_model,
            "normal_vae": normal_vae,
            "depth_model": depth_model,
            "depth_vae": depth_vae,
            "cache_session": cache_session,
        }
        print(
            "[Orbit3D Static Context] packed once; "
            f"fov={fov_value:.6f}; tensors/models copied=0; session={cache_session.get('session_id')}",
            flush=True,
        )
        return (context,)


class Orbit3DStaticLoopContextUnpack:
    @classmethod
    def INPUT_TYPES(cls):
        # Akatz value4 is a wildcard output; validate the actual dict at runtime.
        return {"required": {"context": ("*",)}}

    RETURN_TYPES = (
        "IMAGE", "IMAGE", "INT", "INT", "IMAGE", "MASK", "INT", "INT",
        "IMAGE", "MOGE_MODEL", "FLOAT", "MARIGOLDV2_MODEL", "VAE",
        "MARIGOLDV2_MODEL", "VAE", "ORBIT3D_CACHE_SESSION",
    )
    RETURN_NAMES = (
        "background", "canvas", "x", "y", "diffuse", "whole_mask",
        "width", "height", "black", "moge_model", "fov",
        "normal_model", "normal_vae", "depth_model", "depth_vae",
        "cache_session",
    )
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"

    def run(self, context):
        if not isinstance(context, dict) or context.get("orbit3d_cache_type") != _KIND:
            raise TypeError("context must come from Orbit3D - Pack Static Loop Context")
        required = (
            "background", "canvas", "x", "y", "diffuse", "whole_mask",
            "width", "height", "black", "moge_model", "fov",
            "normal_model", "normal_vae", "depth_model", "depth_vae",
            "cache_session",
        )
        missing = [k for k in required if k not in context]
        if missing:
            raise ValueError(f"Orbit3D static loop context missing fields: {missing}")
        return tuple(context[k] for k in required)


class Orbit3DStaticLoopFOVFromSession:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cache_session": ("ORBIT3D_CACHE_SESSION",)}}

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("fov",)
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"

    def run(self, cache_session):
        root, session_id = _validated_session(cache_session)
        path = root / _META_NAME
        if not path.is_file():
            raise RuntimeError(f"Orbit3D static FOV metadata missing: {path}")
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Invalid Orbit3D static FOV metadata: {path}: {exc}") from exc
        if meta.get("orbit3d_cache_type") != _META_KIND or int(meta.get("version", -1)) != 1:
            raise RuntimeError(f"Invalid Orbit3D static FOV metadata schema: {path}")
        if str(meta.get("session_id", "")) != session_id:
            raise RuntimeError("Orbit3D static FOV metadata/session mismatch")
        fov = float(meta["fov"])
        print(f"[Orbit3D Static Context] restored FOV={fov:.6f} from session metadata", flush=True)
        return (fov,)


NODE_CLASS_MAPPINGS = {
    "Orbit3DStaticLoopContextPack": Orbit3DStaticLoopContextPack,
    "Orbit3DStaticLoopContextUnpack": Orbit3DStaticLoopContextUnpack,
    "Orbit3DStaticLoopFOVFromSession": Orbit3DStaticLoopFOVFromSession,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Orbit3DStaticLoopContextPack": "Orbit3D - Pack Static Loop Context",
    "Orbit3DStaticLoopContextUnpack": "Orbit3D - Unpack Static Loop Context",
    "Orbit3DStaticLoopFOVFromSession": "Orbit3D - Load Static FOV From Session",
}
