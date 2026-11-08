"""Orbit3D Patch 3.6.8: disk-backed VSR batch spill.

The VSR node remains completely unchanged. This module consumes its ordinary
CPU IMAGE batch, stores every view losslessly in the existing Orbit3D cache
session, evicts only the upstream VSR cache entry, destroys the now-consumed
batch storage, and feeds StartLoop lightweight ordered refs.

No reconstruction math, camera math, model inference, seeds, or view order are
changed.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_GIB = 1024 ** 3
_CACHE_VERSION = 1
_REF_KIND = "orbit3d_image_ref_v1"
_EXPECTED_UPSTREAM_CLASS = "Orbit3DRealBasicVSR"


def _read_kb_file(path: str) -> dict[str, int]:
    out = {}
    try:
        for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            parts = rest.strip().split()
            if parts and parts[0].isdigit():
                out[key] = int(parts[0]) * 1024
    except Exception:
        pass
    return out


def _gib(v):
    return "n/a" if v is None else f"{v / _GIB:.2f} GiB"


def _malloc_trim():
    try:
        libc = ctypes.CDLL(None)
        fn = getattr(libc, "malloc_trim", None)
        if fn is None:
            return False, "malloc_trim unavailable"
        fn.argtypes = [ctypes.c_size_t]
        fn.restype = ctypes.c_int
        rc = int(fn(0))
        return bool(rc), f"malloc_trim rc={rc}"
    except Exception as exc:
        return False, f"malloc_trim error: {type(exc).__name__}: {exc}"


def _validate_session(cache_session: dict) -> tuple[Path, str]:
    if not isinstance(cache_session, dict):
        raise TypeError("cache_session must come from Orbit3D - Cache Session")
    if cache_session.get("orbit3d_cache_type") != "orbit3d_session":
        raise ValueError("Expected a unified Orbit3D cache session")
    session_id = str(cache_session.get("session_id", "")).strip()
    root = Path(str(cache_session.get("root", ""))).resolve()
    session_json = root / "session.json"
    if not session_id or not session_json.is_file():
        raise ValueError("Invalid Orbit3D cache session")
    disk = json.loads(session_json.read_text(encoding="utf-8"))
    if disk.get("orbit3d_cache_type") != "orbit3d_session":
        raise ValueError("session.json is not a unified Orbit3D session")
    if str(disk.get("session_id", "")) != session_id:
        raise ValueError("Orbit3D cache session_id mismatch")
    if Path(str(disk.get("root", ""))).resolve() != root:
        raise ValueError("Orbit3D cache root mismatch")
    return root, session_id


def _node_key(mapping, wanted):
    if mapping is None:
        return None
    if wanted in mapping:
        return wanted
    sw = str(wanted)
    for k in mapping.keys():
        if str(k) == sw:
            return k
    return None


def _prompt_node(prompt, node_id):
    if not isinstance(prompt, dict):
        return None
    k = _node_key(prompt, node_id)
    return None if k is None else prompt[k]


def _parse_link(value):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        a, b = value[0], value[1]
        if isinstance(a, (str, int)) and isinstance(b, int):
            return str(a), int(b)
    return None


def _single_cached_tensor(entry, socket_index: int):
    if entry is None or not hasattr(entry, "outputs"):
        return None
    outputs = entry.outputs
    if not isinstance(outputs, (list, tuple)) or socket_index >= len(outputs):
        return None
    value = outputs[socket_index]
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1 and torch.is_tensor(value[0]):
        return value[0]
    return None


def _resolve_upstream_and_validate_graph(prompt, unique_id):
    me = _prompt_node(prompt, unique_id)
    if not isinstance(me, dict):
        raise RuntimeError("Spill Image Batch could not locate itself in PROMPT")
    inputs = me.get("inputs") or {}
    link = _parse_link(inputs.get("images"))
    if link is None:
        raise RuntimeError("Spill Image Batch images input is not a direct node link")
    upstream_id, upstream_socket = link
    if upstream_socket != 0:
        raise RuntimeError(
            f"Spill Image Batch expects VSR output socket 0, got socket {upstream_socket}"
        )

    upstream = _prompt_node(prompt, upstream_id)
    if not isinstance(upstream, dict):
        raise RuntimeError(f"Upstream node {upstream_id!r} is missing from PROMPT")
    upstream_class = str(upstream.get("class_type", ""))
    if upstream_class != _EXPECTED_UPSTREAM_CLASS:
        raise RuntimeError(
            "Spill Image Batch destructive release is allowed only for "
            f"{_EXPECTED_UPSTREAM_CLASS}, got {upstream_class!r}"
        )

    consumers = []
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        for input_name, input_value in (node.get("inputs") or {}).items():
            parsed = _parse_link(input_value)
            if parsed == (str(upstream_id), upstream_socket):
                consumers.append((str(nid), str(input_name)))

    expected = [(str(unique_id), "images")]
    if consumers != expected:
        raise RuntimeError(
            "Refusing destructive VSR release: upstream VSR output must have exactly "
            f"one consumer (this Spill.images). Found: {consumers}"
        )
    return str(upstream_id), upstream_socket


def _get_concrete_cache(output_cache, node_id):
    cache = output_cache
    if hasattr(output_cache, "_get_cache_for"):
        cache = output_cache._get_cache_for(node_id)
    return cache


def _evict_upstream_cache(execution_list, upstream_id, socket_index, images, unique_id):
    if execution_list is None or not hasattr(execution_list, "output_cache"):
        raise RuntimeError(
            "EXECUTION_LIST/output_cache is unavailable; refusing destructive release"
        )

    output_cache = execution_list.output_cache
    if not hasattr(output_cache, "get_local"):
        raise RuntimeError("output_cache.get_local is unavailable")

    entry = output_cache.get_local(upstream_id)
    if entry is None:
        raise RuntimeError(
            f"Upstream VSR cache entry {upstream_id!r} is missing; "
            "do not use this Spill node with --cache-none"
        )

    cached_tensor = _single_cached_tensor(entry, socket_index)
    if cached_tensor is not images:
        raise RuntimeError(
            "Upstream cache identity check failed; refusing to destroy IMAGE storage"
        )

    concrete = _get_concrete_cache(output_cache, upstream_id)
    if concrete is None:
        raise RuntimeError("Could not resolve concrete Comfy output cache")
    if not hasattr(concrete, "cache") or not hasattr(concrete, "cache_key_set"):
        raise RuntimeError(
            "Comfy output cache internals differ from the validated 0.42.x contract"
        )

    key = concrete.cache_key_set.get_data_key(upstream_id)
    if key not in concrete.cache:
        raise RuntimeError("Resolved VSR cache key is not present in concrete cache")
    stored = concrete.cache[key]
    if _single_cached_tensor(stored, socket_index) is not images:
        raise RuntimeError(
            "Concrete cache identity check failed; refusing destructive release"
        )

    # Remove the persistent output entry first.
    del concrete.cache[key]

    # Keep alternate cache implementations internally consistent.
    for attr in ("used_generation", "timestamps", "children"):
        obj = getattr(concrete, attr, None)
        if isinstance(obj, dict):
            obj.pop(key, None)

    # Remove the strong per-execution reference from VSR -> this Spill, if present.
    execution_cache = getattr(execution_list, "execution_cache", None)
    if isinstance(execution_cache, dict):
        spill_key = _node_key(execution_cache, unique_id)
        if spill_key is not None:
            bucket = execution_cache.get(spill_key)
            if isinstance(bucket, dict):
                upstream_key = _node_key(bucket, upstream_id)
                if upstream_key is not None:
                    linked_entry = bucket[upstream_key]
                    linked_tensor = _single_cached_tensor(linked_entry, socket_index)
                    if linked_tensor is not images:
                        raise RuntimeError(
                            "Execution-cache identity check failed after main cache eviction"
                        )
                    del bucket[upstream_key]
                if not bucket:
                    execution_cache.pop(spill_key, None)

    # Listener metadata does not hold the tensor, but remove this resolved edge so
    # the current execution cannot write the consumed entry back accidentally.
    listeners_map = getattr(execution_list, "execution_cache_listeners", None)
    if isinstance(listeners_map, dict):
        lk = _node_key(listeners_map, upstream_id)
        if lk is not None:
            listeners = listeners_map.get(lk)
            if isinstance(listeners, set):
                filtered = set()
                for item in listeners:
                    target = item[0] if isinstance(item, tuple) and item else item
                    if str(target) != str(unique_id):
                        filtered.add(item)
                if filtered:
                    listeners_map[lk] = filtered
                else:
                    listeners_map.pop(lk, None)

    return key


class Orbit3DImageBatchSpill:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "cache_session": ("ORBIT3D_CACHE_SESSION",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "execution_list": "EXECUTION_LIST",
            },
        }

    RETURN_TYPES = ("IMAGE", "ORBIT3D_IMAGE_REF", "STRING")
    RETURN_NAMES = ("first_image", "image_refs", "report")
    OUTPUT_IS_LIST = (False, True, False)
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"

    def run(
        self,
        images,
        cache_session,
        unique_id=None,
        prompt=None,
        execution_list=None,
    ):
        if not torch.is_tensor(images):
            raise TypeError(f"images must be torch.Tensor, got {type(images).__name__}")
        if images.ndim != 4:
            raise ValueError(f"Expected IMAGE batch [V,H,W,C], got {tuple(images.shape)}")
        if images.device.type != "cpu":
            raise RuntimeError(
                f"Spill Image Batch expects CPU VSR output, got device={images.device}"
            )
        if images.dtype != torch.float32:
            raise RuntimeError(
                f"Spill Image Batch expects VSR float32 output, got dtype={images.dtype}"
            )
        if int(images.shape[0]) < 1:
            raise ValueError("VSR IMAGE batch is empty")

        # Validate graph ownership BEFORE any disk I/O or destructive action.
        upstream_id, upstream_socket = _resolve_upstream_and_validate_graph(
            prompt, unique_id
        )
        root, session_id = _validate_session(cache_session)

        view_count = int(images.shape[0])
        batch_shape = [int(x) for x in images.shape]
        batch_bytes = int(images.untyped_storage().nbytes())
        first_image = images[0:1].clone(memory_format=torch.contiguous_format)

        spill_dir = root / "vsr_views"
        if spill_dir.exists():
            raise RuntimeError(
                f"VSR spill directory already exists in this cache session: {spill_dir}"
            )
        spill_dir.mkdir(parents=True, exist_ok=False)

        refs = []
        total_file_size = 0
        try:
            for i in range(view_count):
                final_path = spill_dir / f"view_{i:06d}.safetensors"
                tmp_path = spill_dir / f".view_{i:06d}.tmp.safetensors"

                # One-frame private storage only; never retain all frame copies.
                frame = images[i:i + 1].clone(memory_format=torch.contiguous_format)
                save_file(
                    {"image": frame},
                    str(tmp_path),
                    metadata={
                        "orbit3d_cache_version": str(_CACHE_VERSION),
                        "orbit3d_cache_kind": "vsr_image_view",
                        "session_id": session_id,
                        "index": str(i),
                        "view_count": str(view_count),
                    },
                )
                del frame
                os.replace(tmp_path, final_path)
                file_size = int(final_path.stat().st_size)
                total_file_size += file_size
                refs.append({
                    "orbit3d_cache_type": _REF_KIND,
                    "version": _CACHE_VERSION,
                    "session_id": session_id,
                    "session_root": str(root),
                    "index": i,
                    "view_count": view_count,
                    "path": str(final_path),
                    "tensor_key": "image",
                    "shape": [1] + batch_shape[1:],
                    "dtype": str(images.dtype),
                    "file_size": file_size,
                })

            manifest = {
                "orbit3d_cache_type": "orbit3d_vsr_view_manifest",
                "version": _CACHE_VERSION,
                "session_id": session_id,
                "root": str(root),
                "view_count": view_count,
                "source_shape": batch_shape,
                "dtype": str(images.dtype),
                "total_file_size": total_file_size,
                "items": refs,
            }
            manifest_tmp = spill_dir / ".manifest.tmp.json"
            manifest_path = spill_dir / "manifest.json"
            manifest_tmp.write_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(manifest_tmp, manifest_path)
            for ref in refs:
                ref["manifest_path"] = str(manifest_path)

            # Verify first/last disk entries before touching the upstream cache.
            check_indices = sorted(set((0, view_count - 1)))
            for i in check_indices:
                with safe_open(refs[i]["path"], framework="pt", device="cpu") as f:
                    got = f.get_tensor("image")
                expected = images[i:i + 1]
                if got.shape != expected.shape or got.dtype != expected.dtype:
                    raise RuntimeError(f"VSR spill round-trip schema mismatch at index {i}")
                if not torch.equal(got, expected):
                    raise RuntimeError(f"VSR spill round-trip value mismatch at index {i}")
                del got

        except Exception:
            shutil.rmtree(spill_dir, ignore_errors=True)
            raise

        status0 = _read_kb_file("/proc/self/status")
        mem0 = _read_kb_file("/proc/meminfo")
        storage_before = int(images.untyped_storage().nbytes())

        cache_key = _evict_upstream_cache(
            execution_list,
            upstream_id,
            upstream_socket,
            images,
            unique_id,
        )

        # The persistent cache entry and current strong edge are gone. Mutate the
        # exact Tensor object that Comfy passed to this node so its multi-GiB
        # storage is no longer resident.
        images.set_(torch.empty(0, dtype=images.dtype, device=images.device))
        storage_after = int(images.untyped_storage().nbytes())

        status1 = _read_kb_file("/proc/self/status")
        _, trim_status = _malloc_trim()
        status2 = _read_kb_file("/proc/self/status")
        mem2 = _read_kb_file("/proc/meminfo")

        report = (
            "Orbit3D VSR batch spilled and evicted: "
            f"V={view_count}; source_shape={tuple(batch_shape)}; "
            f"batch_storage={batch_bytes / _GIB:.3f} GiB; "
            f"disk={total_file_size / _GIB:.3f} GiB; "
            f"upstream={upstream_id}:0; cache_key={cache_key!r}; "
            f"storage={storage_before / _GIB:.3f}->{storage_after / _GIB:.3f} GiB; "
            f"VmRSS={_gib(status0.get('VmRSS'))}->{_gib(status1.get('VmRSS'))}"
            f"->{_gib(status2.get('VmRSS'))}; "
            f"MemAvailable={_gib(mem0.get('MemAvailable'))}"
            f"->{_gib(mem2.get('MemAvailable'))}; "
            f"{trim_status}; manifest={spill_dir / 'manifest.json'}"
        )
        print(f"[Orbit3D VSR Spill] {report}", flush=True)
        return first_image, refs, report


class Orbit3DImageRefLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("ORBIT3D_IMAGE_REF",),
                "iteration_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0x7FFFFFFFFFFFFFFF, "forceInput": True},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"

    def run(self, image_ref, iteration_index):
        if not isinstance(image_ref, dict):
            raise TypeError("image_ref must come from Orbit3D - Spill Image Batch")
        if image_ref.get("orbit3d_cache_type") != _REF_KIND:
            raise ValueError(
                f"Unsupported image ref kind: {image_ref.get('orbit3d_cache_type')!r}"
            )

        index = int(image_ref.get("index", -1))
        expected_index = int(iteration_index)
        if index != expected_index:
            raise RuntimeError(
                "Orbit3D view-order violation: "
                f"StartLoop iteration={expected_index}, disk ref index={index}. "
                "Refusing to feed a wrong angle/frame into reconstruction."
            )

        view_count = int(image_ref.get("view_count", 0))
        if not (0 <= index < view_count):
            raise ValueError(f"Invalid image ref index {index} for V={view_count}")

        root = Path(str(image_ref.get("session_root", ""))).resolve()
        path = Path(str(image_ref.get("path", ""))).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Image ref path escaped cache session: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"VSR cached view is missing: {path}")

        key = str(image_ref.get("tensor_key", "image"))
        with safe_open(str(path), framework="pt", device="cpu") as f:
            if key not in f.keys():
                raise KeyError(f"Tensor key {key!r} is missing in {path}")
            image = f.get_tensor(key)

        expected_shape = tuple(int(x) for x in image_ref.get("shape") or ())
        expected_dtype = str(image_ref.get("dtype", ""))
        if expected_shape and tuple(image.shape) != expected_shape:
            raise RuntimeError(
                f"Cached view {index} shape mismatch: {tuple(image.shape)} != {expected_shape}"
            )
        if expected_dtype and str(image.dtype) != expected_dtype:
            raise RuntimeError(
                f"Cached view {index} dtype mismatch: {image.dtype} != {expected_dtype}"
            )
        if image.ndim != 4 or image.shape[0] != 1:
            raise RuntimeError(
                f"Cached view {index} must be [1,H,W,C], got {tuple(image.shape)}"
            )

        return (image,)


NODE_CLASS_MAPPINGS = {
    "Orbit3DImageBatchSpill": Orbit3DImageBatchSpill,
    "Orbit3DImageRefLoad": Orbit3DImageRefLoad,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Orbit3DImageBatchSpill": "Orbit3D - Spill Image Batch",
    "Orbit3DImageRefLoad": "Orbit3D - Load Image Ref",
}
