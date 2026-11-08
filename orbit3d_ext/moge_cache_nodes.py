"""Orbit3D Patch 3.1.1: per-view lossless MoGe disk cache.

V1 node contract is used deliberately here because this package registers nodes
through NODE_CLASS_MAPPINGS.  The actual MoGe inference still delegates to the
existing Orbit3DMoGeInference / native ComfyUI execute path.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
import shutil
import uuid

import torch
import folder_paths
from safetensors import safe_open
from safetensors.torch import save_file

from .moge_nodes import Orbit3DMoGeInference


_CACHE_VERSION = 1
_FORMAT = "safetensors"


def _default_cache_root() -> Path:
    return Path(folder_paths.get_temp_directory()) / "orbit3d_moge_cache"


def _resolve_cache_root(value: str) -> Path:
    value = str(value or "").strip()
    root = Path(os.path.expanduser(value)) if value else _default_cache_root()
    return root.resolve()


def _jsonable_metadata(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable_metadata(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable_metadata(v) for k, v in value.items()}
    raise TypeError(
        "Orbit3D MoGe cache encountered unsupported non-tensor metadata "
        f"type {type(value).__name__}; refusing to silently discard it."
    )


def _tensor_schema(t: torch.Tensor) -> dict:
    return {"shape": [int(x) for x in t.shape], "dtype": str(t.dtype)}


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.numel() == 0:
        return True
    aa = a.detach().cpu().contiguous()
    bb = b.detach().cpu().contiguous()
    try:
        return bool(torch.equal(aa.view(torch.uint8), bb.view(torch.uint8)))
    except Exception:
        if aa.dtype.is_floating_point:
            return bool(torch.allclose(aa, bb, rtol=0.0, atol=0.0, equal_nan=True))
        return bool(torch.equal(aa, bb))


def _verify_file_exact(path: Path, originals: dict[str, torch.Tensor]) -> None:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        disk_keys = set(f.keys())
        src_keys = set(originals)
        if disk_keys != src_keys:
            raise RuntimeError(
                "Orbit3D MoGe cache round-trip key mismatch: "
                f"saved={sorted(src_keys)} loaded={sorted(disk_keys)}"
            )
        for key in sorted(src_keys):
            loaded = f.get_tensor(key)
            if not _bitwise_equal(originals[key], loaded):
                raise RuntimeError(
                    "Orbit3D MoGe cache round-trip mismatch for "
                    f"{key!r}: source={tuple(originals[key].shape)} "
                    f"{originals[key].dtype}, loaded={tuple(loaded.shape)} "
                    f"{loaded.dtype}"
                )



# --- ORBIT3D PATCH 3.2 LAZY CACHE CONSUMER BEGIN ---
class _MoGeCacheReader:
    """One-view-at-a-time safetensors reader shared by lazy field proxies."""

    def __init__(self, manifest: dict):
        self.manifest = manifest
        self.items = list(manifest["items"])
        self._current_index = None
        self._current_fields = {}

    def get(self, index: int, field: str) -> torch.Tensor:
        index = int(index)
        if index < 0:
            index += len(self.items)
        if not (0 <= index < len(self.items)):
            raise IndexError(index)

        if self._current_index != index:
            self._current_fields.clear()
            self._current_index = index

        if field not in self._current_fields:
            item = self.items[index]
            path = Path(item["path"])
            if not path.is_file():
                raise FileNotFoundError(f"Missing cached MoGe view: {path}")
            tensor_key = (item.get("tensor_key_map") or {}).get(field, field)
            with safe_open(str(path), framework="pt", device="cpu") as f:
                if tensor_key not in f.keys():
                    raise KeyError(
                        f"Cached MoGe view {index} has no tensor key "
                        f"{tensor_key!r} for field {field!r}: {path}"
                    )
                self._current_fields[field] = f.get_tensor(tensor_key)

        return self._current_fields[field]

    def clear(self):
        self._current_fields.clear()
        self._current_index = None


def _cache_manifest_schema(moge_cache: dict):
    if not isinstance(moge_cache, dict):
        raise TypeError("moge_cache must come from Orbit3D - Finalize MoGe Cache")
    if moge_cache.get("orbit3d_cache_type") != "moge_cache":
        raise ValueError("Invalid ORBIT3D_MOGE_CACHE handle")
    if int(moge_cache.get("version", -1)) != _CACHE_VERSION:
        raise ValueError(
            f"Unsupported ORBIT3D_MOGE_CACHE version {moge_cache.get('version')!r}"
        )

    items = list(moge_cache.get("items") or [])
    if not items:
        raise ValueError("ORBIT3D_MOGE_CACHE contains no views")

    required = ("points", "depth", "normal", "mask", "intrinsics")
    first_schema = dict(items[0].get("tensor_schema") or {})
    missing = [k for k in required if k not in first_schema]
    if missing:
        raise ValueError(f"MoGe cache missing required fields in view 0: {missing}")

    for index, item in enumerate(items):
        schema = dict(item.get("tensor_schema") or {})
        miss = [k for k in required if k not in schema]
        if miss:
            raise ValueError(f"MoGe cache view {index} missing fields: {miss}")
        for key in required:
            if list(schema[key].get("shape") or []) != list(first_schema[key].get("shape") or []):
                raise ValueError(
                    f"MoGe cache field {key!r} changes shape at view {index}: "
                    f"{schema[key].get('shape')} != {first_schema[key].get('shape')}"
                )

    return items, first_schema


class _LazyMoGeField:
    """Tensor-like [B=1,V,...] view over one cached native MoGe field.

    It intentionally supports only single-view indexing.  Existing Orbit SS/S4
    code accesses MoGe as points[b,v,...] / mask[b,v,...], so a request that
    would materialize multiple views is treated as a programming error instead
    of silently rebuilding the large batch we are removing.
    """

    def __init__(self, reader, field, per_view_shape, cast_dtype=None):
        shape = tuple(int(x) for x in per_view_shape)
        if not shape or shape[0] != 1:
            raise ValueError(
                f"Cached MoGe field {field!r} must have per-view leading batch 1, got {shape}"
            )
        self._reader = reader
        self._field = str(field)
        self._per_view_shape = shape
        self._cast_dtype = cast_dtype
        self._shape = (1, len(reader.items), *shape[1:])

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return self._cast_dtype

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return _LazyMoGeField(
            self._reader, self._field, self._per_view_shape, torch.float32
        )

    def bool(self):
        return _LazyMoGeField(
            self._reader, self._field, self._per_view_shape, torch.bool
        )

    @staticmethod
    def _norm_tuple(key):
        if not isinstance(key, tuple):
            key = (key,)
        if Ellipsis in key:
            pos = key.index(Ellipsis)
            missing = max(0, 2 - (len(key) - 1))
            key = key[:pos] + (slice(None),) * missing + key[pos + 1:]
        return key

    def __getitem__(self, key):
        key = self._norm_tuple(key)
        if len(key) < 2:
            raise RuntimeError(
                "Lazy MoGe cache fields require explicit [batch, view] indexing; "
                "refusing to materialize multiple views."
            )

        batch_key, view_key = key[0], key[1]
        if not isinstance(view_key, int):
            raise RuntimeError(
                "Lazy MoGe cache fields support one view at a time; "
                f"got view index {view_key!r}."
            )
        view = view_key if view_key >= 0 else len(self._reader.items) + view_key

        t = self._reader.get(view, self._field)
        if self._cast_dtype is not None and t.dtype != self._cast_dtype:
            t = t.to(dtype=self._cast_dtype)

        rest = key[2:]
        if isinstance(batch_key, int):
            b = batch_key if batch_key >= 0 else 1 + batch_key
            if b != 0:
                raise IndexError(batch_key)
            out = t[0]
            if rest:
                out = out[rest]
            return out

        if isinstance(batch_key, slice):
            start, stop, step = batch_key.indices(1)
            if (start, stop, step) != (0, 1, 1):
                raise RuntimeError(
                    f"Lazy MoGe cache has B=1; unsupported batch slice {batch_key!r}"
                )
            out = t
            if rest:
                out = out[(slice(None),) + rest]
            return out

        raise TypeError(f"Unsupported lazy MoGe batch index {batch_key!r}")

    def __repr__(self):
        return (
            f"_LazyMoGeField(field={self._field!r}, shape={self.shape}, "
            f"cast_dtype={self._cast_dtype})"
        )


class _LazyMoGeShapeSource:
    """Lazy [1,V,1,H,W] Shape source derived exactly from cached MoGe."""

    def __init__(self, reader, kind, points_shape):
        self._reader = reader
        self._kind = str(kind)
        ps = tuple(int(x) for x in points_shape)
        if len(ps) != 4 or ps[0] != 1 or ps[-1] != 3:
            raise ValueError(f"Unexpected cached MoGe points shape {ps}")
        self._shape = (1, len(reader.items), 1, ps[1], ps[2])

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    def _view(self, view):
        if self._kind == "depth":
            points = self._reader.get(view, "points").float()
            return points[..., 2].unsqueeze(1)

        mask = self._reader.get(view, "mask").bool()
        alpha = mask.float()

        if self._kind == "alpha":
            return alpha.unsqueeze(1)

        if self._kind == "normal_confidence":
            normal = self._reader.get(view, "normal").float()
            n = normal / torch.linalg.vector_norm(
                normal, dim=-1, keepdim=True
            ).clamp_min(1e-6)
            # Identical to mv_camera.attach_moge_shape512_geometry:
            # OpenCV +Z is into scene, so front-facing normal is -Z.
            facing = (-n[..., 2]).clamp(0.0, 1.0)
            conf = (0.50 + 0.50 * alpha * facing).clamp(0.50, 1.00)
            return conf.unsqueeze(1)

        raise ValueError(self._kind)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) < 2:
            raise RuntimeError(
                "Lazy MoGe Shape source requires explicit [batch, view] indexing."
            )
        batch_key, view_key = key[0], key[1]
        if not isinstance(view_key, int):
            raise RuntimeError(
                "Lazy MoGe Shape source supports one view at a time; "
                f"got {view_key!r}."
            )
        view = view_key if view_key >= 0 else len(self._reader.items) + view_key
        t = self._view(view)  # [1,1,H,W]
        rest = key[2:]

        if isinstance(batch_key, int):
            b = batch_key if batch_key >= 0 else 1 + batch_key
            if b != 0:
                raise IndexError(batch_key)
            out = t[0]
            if rest:
                out = out[rest]
            return out
        if isinstance(batch_key, slice):
            start, stop, step = batch_key.indices(1)
            if (start, stop, step) != (0, 1, 1):
                raise RuntimeError(f"Unsupported B=1 slice {batch_key!r}")
            out = t
            if rest:
                out = out[(slice(None),) + rest]
            return out
        raise TypeError(batch_key)

    def __repr__(self):
        return f"_LazyMoGeShapeSource(kind={self._kind!r}, shape={self.shape})"


def attach_moge_cache_to_views(views: dict, moge_cache: dict) -> dict:
    """Attach a cache-backed MoGe contract without materializing V native maps."""
    items, schema = _cache_manifest_schema(moge_cache)

    transform = views.get("transform_matrix")
    if not torch.is_tensor(transform) or transform.ndim < 2:
        raise ValueError("PIXAL3D_MV_VIEWS lacks transform_matrix for cache validation")
    expected_v = int(transform.shape[1])
    if len(items) != expected_v:
        raise ValueError(
            f"MoGe cache has V={len(items)} but PIXAL3D_MV_VIEWS has V={expected_v}"
        )

    reader = _MoGeCacheReader(moge_cache)
    points_shape = schema["points"]["shape"]
    depth_shape = schema["depth"]["shape"]
    normal_shape = schema["normal"]["shape"]
    mask_shape = schema["mask"]["shape"]
    k_shape = schema["intrinsics"]["shape"]

    # Validate the same geometry relationships expected by the old batch path.
    ps = tuple(points_shape)
    ns = tuple(normal_shape)
    ds = tuple(depth_shape)
    ms = tuple(mask_shape)
    ks = tuple(k_shape)
    if len(ps) != 4 or ps[0] != 1 or ps[-1] != 3:
        raise ValueError(f"Cached points must be [1,H,W,3], got {ps}")
    if ns != ps:
        raise ValueError(f"Cached normal shape {ns} != points shape {ps}")
    if ds != ps[:3]:
        raise ValueError(f"Cached depth shape {ds} != expected {ps[:3]}")
    if ms != ps[:3]:
        raise ValueError(f"Cached mask shape {ms} != expected {ps[:3]}")
    if ks != (1, 3, 3):
        raise ValueError(f"Cached intrinsics must be [1,3,3], got {ks}")

    # Intrinsics are tiny; materialize them once so existing K diagnostics can
    # perform vectorized operations unchanged.
    intrinsics = torch.cat(
        [reader.get(v, "intrinsics").float() for v in range(expected_v)], dim=0
    ).unsqueeze(0)
    reader.clear()

    views["ss_geometry"] = {
        "points": _LazyMoGeField(reader, "points", ps, torch.float32),
        "depth": _LazyMoGeField(reader, "depth", ds, torch.float32),
        "normal": _LazyMoGeField(reader, "normal", ns, torch.float32),
        "mask": _LazyMoGeField(reader, "mask", ms, torch.bool),
        "intrinsics": intrinsics,
    }

    # Shape512 / ShapeHR keep their existing code and exact equations.  These
    # proxies only postpone native map construction until [:, view] is requested.
    views["moge_geometry_confidence"] = _LazyMoGeShapeSource(
        reader, "normal_confidence", ps
    )
    views["moge_depth_native"] = _LazyMoGeShapeSource(reader, "depth", ps)
    views["moge_alpha_native"] = _LazyMoGeShapeSource(reader, "alpha", ps)
    views["orbit3d_moge_cache"] = moge_cache
    views["orbit3d_moge_cache_reader"] = reader

    print(
        f"[Orbit3D MoGe Cache] attached lazy V={expected_v}, "
        f"native={ps[2]}x{ps[1]}; no full MOGE_GEOMETRY batch materialized",
        flush=True,
    )
    return views
# --- ORBIT3D PATCH 3.2 LAZY CACHE CONSUMER END ---


class Orbit3DCacheSession:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_root": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Empty = ComfyUI temp/orbit3d_moge_cache. "
                            "A unique session subdirectory is created per run."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("ORBIT3D_CACHE_SESSION", "STRING")
    RETURN_NAMES = ("cache_session", "session_path")
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, cache_root=""):
        root = _resolve_cache_root(cache_root)
        root.mkdir(parents=True, exist_ok=True)

        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        session_id = f"moge_{stamp}_{uuid.uuid4().hex[:8]}"
        session_dir = root / session_id
        items_dir = session_dir / "items"
        items_dir.mkdir(parents=True, exist_ok=False)

        session = {
            "orbit3d_cache_type": "moge_session",
            "version": _CACHE_VERSION,
            "format": _FORMAT,
            "session_id": session_id,
            "root": str(session_dir),
            "items_dir": str(items_dir),
        }

        (session_dir / "session.json").write_text(
            json.dumps(session, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        usage = shutil.disk_usage(session_dir)
        print(
            "[Orbit3D MoGe Cache] session "
            f"{session_id} at {session_dir}; free={usage.free / (1024**3):.2f} GiB",
            flush=True,
        )
        return session, str(session_dir)


class Orbit3DMoGeInferenceToCache:
    """Run the unchanged native Orbit/ComfyUI MoGe inference and cache one view."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "moge_model": ("MOGE_MODEL",),
                "image": ("IMAGE",),
                "resolution_level": (
                    "INT",
                    {
                        "default": 9,
                        "min": 0,
                        "max": 30,
                        "tooltip": (
                            "Native ComfyUI MoGe resolution level; Orbit permits 0..30."
                        ),
                    },
                ),
                "fov_x_degrees": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 170.0,
                        "step": 0.1,
                        "advanced": True,
                    },
                ),
                "batch_size": (
                    "INT",
                    {"default": 4, "min": 1, "max": 64},
                ),
                "force_projection": (
                    "BOOLEAN",
                    {"default": True, "advanced": True},
                ),
                "apply_mask": (
                    "BOOLEAN",
                    {"default": True, "advanced": True},
                ),
                "refine_steps": (
                    "INT",
                    {
                        "default": 3,
                        "min": 0,
                        "max": 8,
                        "advanced": True,
                    },
                ),
                "cache_session": ("ORBIT3D_CACHE_SESSION",),
                "verify_roundtrip": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "advanced": True,
                        "tooltip": (
                            "Reload every cached tensor and compare raw bytes. "
                            "Keep enabled for the first validation run."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("ORBIT3D_MOGE_CACHE_ITEM",)
    RETURN_NAMES = ("moge_cache_item",)
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"

    def run(
        self,
        moge_model,
        image,
        resolution_level,
        fov_x_degrees,
        batch_size,
        force_projection,
        apply_mask,
        refine_steps,
        cache_session,
        verify_roundtrip=True,
    ):
        if not isinstance(cache_session, dict):
            raise TypeError("cache_session must come from Orbit3D - Cache Session.")
        if cache_session.get("orbit3d_cache_type") != "moge_session":
            raise ValueError("Invalid Orbit3D MoGe cache session handle.")
        if int(cache_session.get("version", -1)) != _CACHE_VERSION:
            raise ValueError(
                "Unsupported Orbit3D MoGe cache session version: "
                f"{cache_session.get('version')!r}"
            )

        if not torch.is_tensor(image):
            raise TypeError(f"image must be torch.Tensor, got {type(image).__name__}")
        if int(image.shape[0]) != 1:
            raise ValueError(
                "Orbit3D - MoGe Inference → Cache is a per-view mapped node; "
                f"expected image batch size 1, got {int(image.shape[0])}."
            )

        # Exact same inference path as the existing working Orbit MoGe node.
        native_output = Orbit3DMoGeInference.execute(
            moge_model,
            image,
            resolution_level,
            fov_x_degrees,
            batch_size,
            force_projection,
            apply_mask,
            refine_steps,
        )

        try:
            geometry = native_output[0]
        except Exception as exc:
            raise RuntimeError(
                "Native Orbit/ComfyUI MoGe execute did not return the expected "
                "single geometry packet."
            ) from exc

        if not isinstance(geometry, dict):
            raise TypeError(
                "Native MoGe output must be a MOGE_GEOMETRY dict, "
                f"got {type(geometry).__name__}."
            )

        tensor_fields: dict[str, torch.Tensor] = {}
        metadata_fields = {}
        for key, value in geometry.items():
            key = str(key)
            if torch.is_tensor(value):
                # Preserve dtype and values. Only residence changes: CUDA -> CPU -> disk.
                tensor_fields[key] = value.detach().to("cpu").contiguous()
            else:
                metadata_fields[key] = _jsonable_metadata(value)

        if not tensor_fields:
            raise ValueError("Native MoGe geometry contained no tensor fields.")

        items_dir = Path(cache_session["items_dir"])
        if not items_dir.is_dir():
            raise FileNotFoundError(
                f"Orbit3D MoGe cache session directory is missing: {items_dir}"
            )

        item_id = uuid.uuid4().hex
        tmp_path = items_dir / f".{item_id}.tmp.safetensors"
        final_path = items_dir / f"{item_id}.safetensors"

        save_file(
            tensor_fields,
            str(tmp_path),
            metadata={
                "orbit3d_cache_version": str(_CACHE_VERSION),
                "orbit3d_cache_kind": "moge_view",
                "session_id": str(cache_session["session_id"]),
            },
        )
        os.replace(tmp_path, final_path)

        if verify_roundtrip:
            _verify_file_exact(final_path, tensor_fields)

        schemas = {k: _tensor_schema(v) for k, v in tensor_fields.items()}
        file_size = int(final_path.stat().st_size)
        usage = shutil.disk_usage(final_path.parent)

        item = {
            "orbit3d_cache_type": "moge_item",
            "version": _CACHE_VERSION,
            "format": _FORMAT,
            "session_id": str(cache_session["session_id"]),
            "session_root": str(cache_session["root"]),
            "path": str(final_path),
            "file_size": file_size,
            "tensor_schema": schemas,
            "metadata": metadata_fields,
        }

        print(
            "[Orbit3D MoGe Cache] wrote "
            f"{final_path.name} size={file_size / (1024**2):.1f} MiB "
            f"fields={','.join(sorted(tensor_fields))} "
            f"free={usage.free / (1024**3):.2f} GiB "
            f"verify={'OK' if verify_roundtrip else 'off'}",
            flush=True,
        )

        # Do not return geometry tensors. No gc/empty_cache is forced.
        del tensor_fields
        del geometry
        del native_output
        return (item,)


class Orbit3DMoGeCacheFinalize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_item": ("ORBIT3D_MOGE_CACHE_ITEM",),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("ORBIT3D_MOGE_CACHE", "STRING")
    RETURN_NAMES = ("moge_cache", "report")
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"
    OUTPUT_NODE = True

    def run(self, cache_item):
        items = [x for x in cache_item if x is not None]
        if not items:
            raise ValueError("No Orbit3D MoGe cache items received.")

        first = items[0]
        if not isinstance(first, dict) or first.get("orbit3d_cache_type") != "moge_item":
            raise TypeError("Invalid Orbit3D MoGe cache item.")

        session_id = first["session_id"]
        session_root = Path(first["session_root"])
        ordered = []
        total_size = 0

        for index, item in enumerate(items):
            if not isinstance(item, dict) or item.get("orbit3d_cache_type") != "moge_item":
                raise TypeError(f"Invalid cache item at mapped index {index}.")
            if item.get("session_id") != session_id:
                raise ValueError(
                    "Mapped MoGe cache items came from different cache sessions."
                )

            path = Path(item["path"])
            if not path.is_file():
                raise FileNotFoundError(f"Missing cached MoGe view: {path}")

            size = int(item.get("file_size") or path.stat().st_size)
            total_size += size
            ordered.append(
                {
                    "index": index,
                    "path": str(path),
                    "file_size": size,
                    "tensor_schema": item.get("tensor_schema", {}),
                    "metadata": item.get("metadata", {}),
                }
            )

        manifest = {
            "orbit3d_cache_type": "moge_cache",
            "version": _CACHE_VERSION,
            "format": _FORMAT,
            "session_id": session_id,
            "root": str(session_root),
            "view_count": len(ordered),
            "total_file_size": total_size,
            "items": ordered,
        }

        manifest_path = session_root / "manifest.json"
        tmp_manifest = session_root / ".manifest.tmp.json"
        tmp_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp_manifest, manifest_path)
        manifest["manifest_path"] = str(manifest_path)

        usage = shutil.disk_usage(session_root)
        report = (
            f"Orbit3D MoGe cache finalized: {len(ordered)} views; "
            f"{total_size / (1024**3):.2f} GiB on disk; "
            f"free {usage.free / (1024**3):.2f} GiB; "
            f"manifest={manifest_path}"
        )
        print(f"[Orbit3D MoGe Cache] {report}", flush=True)
        return (manifest, report)


class Orbit3DMoGeCacheCleanup:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "moge_cache": ("ORBIT3D_MOGE_CACHE",),
                "after_export": ("STRING", {"forceInput": True}),
                "delete_cache": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"
    OUTPUT_NODE = True

    def run(self, moge_cache, after_export, delete_cache=True):
        _cache_manifest_schema(moge_cache)
        export_marker = str(after_export or "").strip()
        if not export_marker:
            raise ValueError(
                "Orbit3D cache cleanup requires the completed export path as a dependency."
            )

        root = Path(str(moge_cache.get("root", ""))).resolve()
        session_id = str(moge_cache.get("session_id", ""))
        if not session_id or not root.name.startswith("moge_"):
            raise ValueError(f"Refusing unsafe cache cleanup target: {root}")

        session_json = root / "session.json"
        manifest_json = root / "manifest.json"
        if not session_json.is_file() or not manifest_json.is_file():
            raise ValueError(
                f"Refusing cache cleanup without session.json + manifest.json: {root}"
            )

        session = json.loads(session_json.read_text(encoding="utf-8"))
        manifest_disk = json.loads(manifest_json.read_text(encoding="utf-8"))
        if session.get("session_id") != session_id:
            raise ValueError("Cache cleanup session_id mismatch in session.json")
        if manifest_disk.get("session_id") != session_id:
            raise ValueError("Cache cleanup session_id mismatch in manifest.json")
        if Path(str(manifest_disk.get("root", ""))).resolve() != root:
            raise ValueError("Cache cleanup root mismatch in manifest.json")

        if not bool(delete_cache):
            report = f"Orbit3D MoGe cache kept: {root}; export={export_marker}"
            print(f"[Orbit3D MoGe Cache] {report}", flush=True)
            return (report,)

        total_size = int(moge_cache.get("total_file_size") or 0)
        shutil.rmtree(root)
        report = (
            f"Orbit3D MoGe cache deleted after export: {root}; "
            f"freed~{total_size / (1024**3):.2f} GiB; export={export_marker}"
        )
        print(f"[Orbit3D MoGe Cache] {report}", flush=True)
        return (report,)


NODE_CLASS_MAPPINGS = {
    "Orbit3DCacheSession": Orbit3DCacheSession,
    "Orbit3DMoGeInferenceToCache": Orbit3DMoGeInferenceToCache,
    "Orbit3DMoGeCacheFinalize": Orbit3DMoGeCacheFinalize,
    "Orbit3DMoGeCacheCleanup": Orbit3DMoGeCacheCleanup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Orbit3DCacheSession": "Orbit3D - Cache Session",
    "Orbit3DMoGeInferenceToCache": "Orbit3D - MoGe Inference → Cache",
    "Orbit3DMoGeCacheFinalize": "Orbit3D - Finalize MoGe Cache",
    "Orbit3DMoGeCacheCleanup": "Orbit3D - Cleanup MoGe Cache",
}
