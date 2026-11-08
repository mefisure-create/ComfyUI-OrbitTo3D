from __future__ import annotations

from pathlib import Path
import os
import re
import time

import torch
import folder_paths

from ..trellis2.representations import MeshWithVoxel


_ORBIT3D_CHECKPOINT_FORMAT = "orbit3d_meshwithvoxel"
_ORBIT3D_CHECKPOINT_VERSION = 1
_ORBIT3D_CHECKPOINT_SUBDIR = "orbit3d_mesh_checkpoints"
_ORBIT3D_EMPTY = "<no Orbit3D MESHWITHVOXEL checkpoints found>"


def _checkpoint_dir() -> Path:
    path = Path(folder_paths.get_output_directory()) / _ORBIT3D_CHECKPOINT_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkpoint_names():
    root = _checkpoint_dir()
    return sorted(
        (p.name for p in root.glob("*.pt") if p.is_file()),
        reverse=True,
    )


def _safe_prefix(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._-")
    return value or "orbit3d_mesh"


def _cpu_tensor_or_none(value, name: str):
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"mesh.{name} must be torch.Tensor or None, got {type(value).__name__}")
    out = value.detach()
    if out.device.type != "cpu":
        out = out.to("cpu")
    if not out.is_contiguous():
        out = out.contiguous()
    return out


def _shape_or_none(value):
    if value is None:
        return None
    return tuple(int(x) for x in value)


def _validate_mesh(mesh):
    if not isinstance(mesh, MeshWithVoxel):
        raise TypeError(
            "Expected Trellis2 MeshWithVoxel / MESHWITHVOXEL, got "
            f"{type(mesh).__module__}.{type(mesh).__name__}"
        )
    if not isinstance(mesh.vertices, torch.Tensor):
        raise TypeError("mesh.vertices must be torch.Tensor")
    if not isinstance(mesh.faces, torch.Tensor):
        raise TypeError("mesh.faces must be torch.Tensor")
    if mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices [V,3], got {tuple(mesh.vertices.shape)}")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError(f"Expected faces [F,3], got {tuple(mesh.faces.shape)}")


def _payload_from_mesh(mesh):
    _validate_mesh(mesh)

    origin = mesh.origin
    if isinstance(origin, torch.Tensor):
        origin = origin.detach().cpu().tolist()
    else:
        origin = list(origin)

    payload = {
        "format": _ORBIT3D_CHECKPOINT_FORMAT,
        "version": _ORBIT3D_CHECKPOINT_VERSION,
        "saved_unix_time": time.time(),
        "source_class": f"{type(mesh).__module__}.{type(mesh).__name__}",
        "source_vertex_device": str(mesh.vertices.device),
        "vertices": _cpu_tensor_or_none(mesh.vertices, "vertices"),
        "faces": _cpu_tensor_or_none(mesh.faces, "faces"),
        "origin": origin,
        "voxel_size": float(mesh.voxel_size),
        "coords": _cpu_tensor_or_none(mesh.coords, "coords"),
        "attrs": _cpu_tensor_or_none(mesh.attrs, "attrs"),
        "voxel_shape": _shape_or_none(mesh.voxel_shape),
        "layout": mesh.layout,
    }
    return payload


def _validate_payload(payload, path: Path):
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint is not a dict: {path}")
    if payload.get("format") != _ORBIT3D_CHECKPOINT_FORMAT:
        raise ValueError(
            f"Unsupported checkpoint format {payload.get('format')!r}: {path}"
        )
    if int(payload.get("version", -1)) != _ORBIT3D_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {payload.get('version')!r}: {path}"
        )

    required = (
        "vertices", "faces", "origin", "voxel_size",
        "coords", "attrs", "voxel_shape", "layout",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"Checkpoint missing fields {missing}: {path}")

    vertices = payload["vertices"]
    faces = payload["faces"]
    if not isinstance(vertices, torch.Tensor) or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Checkpoint vertices are not [V,3] torch.Tensor")
    if not isinstance(faces, torch.Tensor) or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Checkpoint faces are not [F,3] torch.Tensor")


def _to_device_or_none(value, device):
    if value is None:
        return None
    return value.to(device)


class Orbit3DSaveMeshWithVoxelCheckpoint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "filename_prefix": (
                    "STRING",
                    {"default": "orbit3d_mesh", "multiline": False},
                ),
            }
        }

    RETURN_TYPES = ("MESHWITHVOXEL", "STRING")
    RETURN_NAMES = ("mesh", "checkpoint_path")
    FUNCTION = "save"
    CATEGORY = "Orbit3D/Cache"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Lossless native MESHWITHVOXEL checkpoint. Stores MeshWithVoxel constructor "
        "fields directly; no trimesh conversion, voxelization, remesh, merge or repair."
    )

    def save(self, mesh, filename_prefix):
        payload = _payload_from_mesh(mesh)
        root = _checkpoint_dir()
        prefix = _safe_prefix(filename_prefix)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = root / f"{prefix}_{stamp}.pt"
        counter = 1
        while path.exists():
            path = root / f"{prefix}_{stamp}_{counter:02d}.pt"
            counter += 1

        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

        vertices = payload["vertices"]
        faces = payload["faces"]
        size_gib = path.stat().st_size / (1024.0 ** 3)
        print(
            "[Orbit3D Mesh Checkpoint] SAVED "
            f"{vertices.shape[0]:,} vertices / {faces.shape[0]:,} faces; "
            f"{size_gib:.3f} GiB; path={path}",
            flush=True,
        )

        # Important: return the ORIGINAL object, untouched, for passthrough use.
        return (mesh, str(path))


class Orbit3DLoadMeshWithVoxelCheckpoint:
    @classmethod
    def INPUT_TYPES(cls):
        names = _checkpoint_names()
        return {
            "required": {
                "checkpoint": ((names if names else [_ORBIT3D_EMPTY]),),
                "load_device": (["cpu", "cuda"], {"default": "cpu"}),
            }
        }

    RETURN_TYPES = ("MESHWITHVOXEL",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "load"
    CATEGORY = "Orbit3D/Cache"
    DESCRIPTION = (
        "Loads an Orbit3D native MESHWITHVOXEL checkpoint directly into "
        "MeshWithVoxel. Default CPU avoids consuming VRAM before repair."
    )

    def load(self, checkpoint, load_device):
        if checkpoint == _ORBIT3D_EMPTY:
            raise FileNotFoundError(
                f"No checkpoints found in {_checkpoint_dir()}. "
                "Run Orbit3D - Save MESHWITHVOXEL Checkpoint first and refresh ComfyUI."
            )

        root = _checkpoint_dir().resolve()
        path = (root / checkpoint).resolve()
        if path.parent != root:
            raise ValueError("Checkpoint path escapes Orbit3D checkpoint directory")
        if not path.is_file():
            raise FileNotFoundError(path)

        payload = torch.load(path, map_location="cpu", weights_only=False)
        _validate_payload(payload, path)

        if load_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("load_device=cuda requested, but CUDA is unavailable")
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        vertices = payload["vertices"].to(device)
        faces = payload["faces"].to(device)
        coords = _to_device_or_none(payload["coords"], device)
        attrs = _to_device_or_none(payload["attrs"], device)
        voxel_shape = payload["voxel_shape"]
        if voxel_shape is not None:
            voxel_shape = torch.Size(voxel_shape)

        mesh = MeshWithVoxel(
            vertices,
            faces,
            origin=list(payload["origin"]),
            voxel_size=float(payload["voxel_size"]),
            coords=coords,
            attrs=attrs,
            voxel_shape=voxel_shape,
            layout=payload["layout"],
        )

        # Constructor intentionally normalizes vertices->float32 and faces->int32,
        # exactly as the native Trellis2 MeshWithVoxel constructor does.
        if mesh.vertices.shape != payload["vertices"].shape:
            raise RuntimeError("Loaded vertex shape differs from checkpoint")
        if mesh.faces.shape != payload["faces"].shape:
            raise RuntimeError("Loaded face shape differs from checkpoint")

        print(
            "[Orbit3D Mesh Checkpoint] LOADED "
            f"{mesh.vertices.shape[0]:,} vertices / {mesh.faces.shape[0]:,} faces; "
            f"device={mesh.vertices.device}; path={path}",
            flush=True,
        )
        return (mesh,)
