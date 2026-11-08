"""Orbit3D Patch 3.3 PoC: generic per-iteration disk spill.

This module deliberately knows nothing about Marigold, MoGe, or any other
producer node. Put it inside ComfyUI's native Start Loop / End Loop body and
feed arbitrary supported producer outputs into value_1..value_4.

Only representation/lifetime/storage changes:
  producer output -> lossless disk representation -> lightweight cache ref.

No ComfyUI executor internals are patched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import uuid

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

_CACHE_VERSION = 1


class _AnyType(str):
    def __ne__(self, other):
        return False


ANY = _AnyType("*")
_MISSING = object()


def _safe_stream(value: str) -> str:
    value = str(value or "generic").strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "generic"


def _validate_session(cache_session: dict) -> tuple[Path, str]:
    if not isinstance(cache_session, dict):
        raise TypeError("cache_session must come from Orbit3D - Cache Session")

    kind = cache_session.get("orbit3d_cache_type")
    if kind not in {"moge_session", "orbit3d_session"}:
        raise ValueError(f"Unsupported Orbit3D cache session type: {kind!r}")

    session_id = str(cache_session.get("session_id", "")).strip()
    root_raw = str(cache_session.get("root", "")).strip()
    if not session_id or not root_raw:
        raise ValueError("Orbit3D cache session is missing session_id/root")

    root = Path(root_raw).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Orbit3D cache session directory is missing: {root}")

    session_json = root / "session.json"
    if not session_json.is_file():
        raise FileNotFoundError(f"Orbit3D cache session.json is missing: {session_json}")

    disk = json.loads(session_json.read_text(encoding="utf-8"))
    if str(disk.get("session_id", "")) != session_id:
        raise ValueError("Orbit3D cache session_id mismatch")

    return root, session_id


def _tensor_schema(t: torch.Tensor) -> dict:
    return {"shape": [int(x) for x in t.shape], "dtype": str(t.dtype)}


def _numpy_schema(a: np.ndarray) -> dict:
    return {"shape": [int(x) for x in a.shape], "dtype": str(a.dtype)}


class _Encoder:
    def __init__(self, item_dir: Path):
        self.item_dir = item_dir
        self.arrays_dir = item_dir / "arrays"
        self.tensors: dict[str, torch.Tensor] = {}
        self.tensor_originals: dict[str, torch.Tensor] = {}
        self.numpy_originals: dict[str, np.ndarray] = {}
        self._t = 0
        self._n = 0
        self._seen: set[int] = set()

    def _enter(self, obj):
        if isinstance(obj, (dict, list, tuple)):
            oid = id(obj)
            if oid in self._seen:
                raise ValueError("Generic spill does not support cyclic containers")
            self._seen.add(oid)
            return oid
        return None

    def _leave(self, oid):
        if oid is not None:
            self._seen.remove(oid)

    def encode(self, value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return {"kind": "scalar", "value": value}

        if isinstance(value, np.generic):
            return {
                "kind": "numpy_scalar",
                "dtype": str(value.dtype),
                "value": value.item(),
            }

        if torch.is_tensor(value):
            key = f"t{self._t:05d}"
            self._t += 1
            cpu = value.detach().to("cpu").contiguous()
            self.tensors[key] = cpu
            self.tensor_originals[key] = cpu
            return {"kind": "torch", "key": key, **_tensor_schema(cpu)}

        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError(
                    f"Generic spill refuses object-dtype numpy arrays: {value.dtype}"
                )
            self.arrays_dir.mkdir(parents=True, exist_ok=True)
            key = f"n{self._n:05d}"
            self._n += 1
            rel = f"arrays/{key}.npy"
            arr = np.ascontiguousarray(value)
            np.save(self.item_dir / rel, arr, allow_pickle=False)
            self.numpy_originals[key] = arr
            return {"kind": "numpy", "key": key, "file": rel, **_numpy_schema(arr)}

        if isinstance(value, Image.Image):
            arr = np.ascontiguousarray(np.asarray(value))
            return {"kind": "pil", "mode": value.mode, "array": self.encode(arr)}

        if isinstance(value, Path):
            return {"kind": "path", "value": str(value)}

        if isinstance(value, dict):
            oid = self._enter(value)
            try:
                return {
                    "kind": "dict",
                    "items": [[self.encode(k), self.encode(v)] for k, v in value.items()],
                }
            finally:
                self._leave(oid)

        if isinstance(value, list):
            oid = self._enter(value)
            try:
                return {"kind": "list", "items": [self.encode(v) for v in value]}
            finally:
                self._leave(oid)

        if isinstance(value, tuple):
            oid = self._enter(value)
            try:
                return {"kind": "tuple", "items": [self.encode(v) for v in value]}
            finally:
                self._leave(oid)

        raise TypeError(
            "Orbit3D Generic Spill encountered unsupported value type "
            f"{type(value).__module__}.{type(value).__qualname__}. "
            "Supported: torch.Tensor, numpy arrays/scalars, PIL.Image, "
            "dict/list/tuple, Path, strings/numbers/bools/None."
        )

    def save_tensors(self):
        if not self.tensors:
            return
        save_file(
            self.tensors,
            str(self.item_dir / "tensors.safetensors"),
            metadata={
                "orbit3d_cache_version": str(_CACHE_VERSION),
                "orbit3d_cache_kind": "generic_spill",
            },
        )

    def verify(self):
        tensor_path = self.item_dir / "tensors.safetensors"
        if self.tensor_originals:
            if not tensor_path.is_file():
                raise RuntimeError("Generic spill verification: tensors file missing")
            with safe_open(str(tensor_path), framework="pt", device="cpu") as f:
                if set(f.keys()) != set(self.tensor_originals):
                    raise RuntimeError("Generic spill verification: tensor keys differ")
                for key, src in self.tensor_originals.items():
                    got = f.get_tensor(key)
                    if got.shape != src.shape or got.dtype != src.dtype:
                        raise RuntimeError(f"Generic spill tensor schema mismatch for {key}")
                    if not torch.equal(got, src):
                        raise RuntimeError(f"Generic spill tensor round-trip mismatch for {key}")

        for key, src in self.numpy_originals.items():
            p = self.item_dir / "arrays" / f"{key}.npy"
            got = np.load(p, allow_pickle=False)
            if got.shape != src.shape or got.dtype != src.dtype:
                raise RuntimeError(f"Generic spill numpy schema mismatch for {key}")
            if got.tobytes(order="C") != src.tobytes(order="C"):
                raise RuntimeError(f"Generic spill numpy round-trip mismatch for {key}")


class Orbit3DGenericCacheSpill:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_session": ("ORBIT3D_CACHE_SESSION",),
                "iteration_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0x7FFFFFFFFFFFFFFF, "forceInput": True},
                ),
                "stream": (
                    "STRING",
                    {"default": "generic", "multiline": False},
                ),
                "value_1": (ANY,),
                "verify_roundtrip": (
                    "BOOLEAN",
                    {"default": True, "advanced": True},
                ),
            },
            "optional": {
                "value_2": (ANY,),
                "value_3": (ANY,),
                "value_4": (ANY,),
            },
        }

    RETURN_TYPES = ("ORBIT3D_CACHE_REF", "STRING")
    RETURN_NAMES = ("cache_ref", "report")
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"

    def run(
        self,
        cache_session,
        iteration_index,
        stream,
        value_1,
        verify_roundtrip=True,
        value_2=_MISSING,
        value_3=_MISSING,
        value_4=_MISSING,
    ):
        root, session_id = _validate_session(cache_session)
        stream = _safe_stream(stream)
        index = int(iteration_index)

        stream_dir = root / "spill" / stream
        stream_dir.mkdir(parents=True, exist_ok=True)

        item_id = uuid.uuid4().hex
        final_dir = stream_dir / f"{index:06d}_{item_id}"
        tmp_dir = stream_dir / f".{index:06d}_{item_id}.tmp"
        tmp_dir.mkdir(parents=True, exist_ok=False)

        values = [value_1]
        for value in (value_2, value_3, value_4):
            if value is not _MISSING:
                values.append(value)

        enc = _Encoder(tmp_dir)
        try:
            tree = {"kind": "bundle", "values": [enc.encode(v) for v in values]}
            enc.save_tensors()

            manifest = {
                "orbit3d_cache_type": "generic_spill_item",
                "version": _CACHE_VERSION,
                "session_id": session_id,
                "stream": stream,
                "iteration_index": index,
                "value_count": len(values),
                "tree": tree,
            }
            (tmp_dir / "item.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            if bool(verify_roundtrip):
                enc.verify()

            os.replace(tmp_dir, final_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        total_size = sum(p.stat().st_size for p in final_dir.rglob("*") if p.is_file())
        usage = shutil.disk_usage(final_dir)

        ref = {
            "orbit3d_cache_type": "generic_spill_ref",
            "version": _CACHE_VERSION,
            "session_id": session_id,
            "session_root": str(root),
            "stream": stream,
            "iteration_index": index,
            "path": str(final_dir),
            "file_size": int(total_size),
            "value_count": len(values),
        }

        report = (
            f"Orbit3D spill {stream}[{index}] -> {final_dir.name}; "
            f"{total_size / (1024**2):.1f} MiB; "
            f"free={usage.free / (1024**3):.2f} GiB; "
            f"verify={'OK' if verify_roundtrip else 'off'}"
        )
        print(f"[Orbit3D Generic Spill] {report}", flush=True)
        return (ref, report)


def _flatten_refs(value):
    if isinstance(value, dict) and value.get("orbit3d_cache_type") == "generic_spill_ref":
        yield value
        return
    if isinstance(value, (list, tuple)):
        for x in value:
            yield from _flatten_refs(x)


class Orbit3DGenericCacheRefReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cache_ref": ("ORBIT3D_CACHE_REF",)}}

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "ORBIT3D_GENERIC_CACHE_MANIFEST")
    RETURN_NAMES = ("report", "cache_manifest")
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"
    OUTPUT_NODE = True

    def run(self, cache_ref):
        refs = list(_flatten_refs(cache_ref))
        if not refs:
            raise ValueError("No Orbit3D generic spill refs received")

        session_ids = {str(r.get("session_id", "")) for r in refs}
        roots = {str(r.get("session_root", "")) for r in refs}
        streams = {str(r.get("stream", "")) for r in refs}
        if len(session_ids) != 1 or len(roots) != 1:
            raise ValueError("Generic spill refs came from different cache sessions")
        if len(streams) != 1:
            raise ValueError(f"Generic spill report expects one stream, got {sorted(streams)}")

        root = Path(next(iter(roots))).resolve()
        stream = next(iter(streams))
        ordered = sorted(refs, key=lambda r: int(r["iteration_index"]))
        total = sum(int(r.get("file_size") or 0) for r in ordered)

        manifest = {
            "orbit3d_cache_type": "generic_spill_manifest",
            "version": _CACHE_VERSION,
            "session_id": next(iter(session_ids)),
            "root": str(root),
            "stream": stream,
            "item_count": len(ordered),
            "total_file_size": total,
            "items": ordered,
        }

        out = root / f"spill_{_safe_stream(stream)}_manifest.json"
        tmp = root / f".spill_{_safe_stream(stream)}_manifest.tmp.json"
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, out)
        manifest["manifest_path"] = str(out)

        report = (
            f"Orbit3D generic spill finalized: stream={stream}; "
            f"{len(ordered)} items; {total / (1024**3):.2f} GiB; manifest={out}"
        )
        print(f"[Orbit3D Generic Spill] {report}", flush=True)
        return (report, manifest)


NODE_CLASS_MAPPINGS = {
    "Orbit3DGenericCacheSpill": Orbit3DGenericCacheSpill,
    "Orbit3DGenericCacheRefReport": Orbit3DGenericCacheRefReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Orbit3DGenericCacheSpill": "Orbit3D - Generic Spill To Cache",
    "Orbit3DGenericCacheRefReport": "Orbit3D - Generic Cache Ref Report",
}
