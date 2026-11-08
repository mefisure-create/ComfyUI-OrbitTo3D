"""Orbit3D patch 3.6.10: file-backed VSR IMAGE batch.

This is an alternative to the 3.6.8 per-frame Image Ref / Load Image Ref path.

The VSR node remains unchanged and still returns its ordinary CPU float32 IMAGE
batch. This node writes that batch losslessly to one raw float32 file inside the
existing Orbit3D cache session, maps the file back with torch.from_file(), evicts
the original VSR output cache entry, and destroys the original anonymous storage.

The mapped IMAGE batch is then split into ordered tensor views by
Orbit3DMMapBatchToImageList. All list items share ONE file-backed storage.

No reconstruction math, camera math, model inference, seeds, or view order are
changed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import torch

from .image_spill_nodes import (
    _GIB,
    _evict_upstream_cache,
    _gib,
    _malloc_trim,
    _read_kb_file,
    _resolve_upstream_and_validate_graph,
    _validate_session,
)

_CACHE_VERSION = 1


def _rss_snapshot():
    status = _read_kb_file("/proc/self/status")
    return {
        "VmRSS": status.get("VmRSS"),
        "RssAnon": status.get("RssAnon"),
        "RssFile": status.get("RssFile"),
    }


def _drop_file_cache_hint(fd: int) -> str:
    """Best-effort hint only; correctness never depends on it."""
    fn = getattr(os, "posix_fadvise", None)
    flag = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fn is None or flag is None:
        return "posix_fadvise unavailable"
    try:
        fn(fd, 0, 0, flag)
        return "posix_fadvise(DONTNEED)=ok"
    except Exception as exc:
        return f"posix_fadvise(DONTNEED)={type(exc).__name__}: {exc}"


class Orbit3DImageBatchSpillMMap:
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

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("first_image", "mmap_batch", "report")
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
                f"MMap Spill expects CPU VSR output, got device={images.device}"
            )
        if images.dtype != torch.float32:
            raise RuntimeError(
                f"MMap Spill expects VSR float32 output, got dtype={images.dtype}"
            )
        if int(images.shape[0]) < 1:
            raise ValueError("VSR IMAGE batch is empty")
        if not hasattr(torch, "from_file"):
            raise RuntimeError("This PyTorch build does not provide torch.from_file")

        # Same fail-closed ownership checks as 3.6.8.
        upstream_id, upstream_socket = _resolve_upstream_and_validate_graph(
            prompt, unique_id
        )
        root, session_id = _validate_session(cache_session)

        view_count = int(images.shape[0])
        batch_shape = tuple(int(x) for x in images.shape)
        numel = int(images.numel())
        expected_bytes = int(numel * images.element_size())
        original_storage_bytes = int(images.untyped_storage().nbytes())

        # 784 needs only the first view. Keep one small ordinary CPU tensor.
        first_image = images[0:1].clone(memory_format=torch.contiguous_format)

        spill_dir = root / "vsr_mmap"
        if spill_dir.exists():
            raise RuntimeError(
                f"VSR mmap spill directory already exists in this session: {spill_dir}"
            )
        spill_dir.mkdir(parents=True, exist_ok=False)

        raw_path = spill_dir / "views_float32.raw"
        tmp_path = spill_dir / ".views_float32.tmp"
        manifest_path = spill_dir / "manifest.json"

        try:
            # Stream one view at a time. There is never a second anonymous full batch.
            # VSR torch.cat output is normally contiguous; the fallback creates at most
            # one private frame at a time.
            with tmp_path.open("wb") as f:
                for i in range(view_count):
                    frame = images[i:i + 1]
                    private = None
                    if not frame.is_contiguous():
                        private = frame.contiguous()
                        frame = private
                    frame.detach().numpy().tofile(f)
                    if private is not None:
                        del private
                f.flush()
                os.fsync(f.fileno())
                fadvise_status = _drop_file_cache_hint(f.fileno())

            os.replace(tmp_path, raw_path)

            actual_bytes = int(raw_path.stat().st_size)
            if actual_bytes != expected_bytes:
                raise RuntimeError(
                    f"VSR mmap raw size mismatch: {actual_bytes} != {expected_bytes}"
                )

            # PyTorch 2.8: storage is backed by mmap(2), not an anonymous allocation.
            flat = torch.from_file(
                str(raw_path),
                shared=False,
                size=numel,
                dtype=images.dtype,
                device=torch.device("cpu"),
            )
            mmap_batch = flat.reshape(batch_shape)
            del flat

            if mmap_batch.device.type != "cpu":
                raise RuntimeError(f"Mapped batch unexpectedly on {mmap_batch.device}")
            if mmap_batch.dtype != images.dtype:
                raise RuntimeError(
                    f"Mapped dtype mismatch: {mmap_batch.dtype} != {images.dtype}"
                )
            if tuple(mmap_batch.shape) != batch_shape:
                raise RuntimeError(
                    f"Mapped shape mismatch: {tuple(mmap_batch.shape)} != {batch_shape}"
                )
            if not mmap_batch.is_contiguous():
                raise RuntimeError("Mapped VSR batch must be contiguous")

            # Exact first/last round-trip guard before destructive release.
            for i in sorted(set((0, view_count - 1))):
                if not torch.equal(mmap_batch[i:i + 1], images[i:i + 1]):
                    raise RuntimeError(
                        f"VSR mmap round-trip value mismatch at index {i}"
                    )

            manifest = {
                "orbit3d_cache_type": "orbit3d_vsr_mmap_batch",
                "version": _CACHE_VERSION,
                "session_id": session_id,
                "root": str(root),
                "path": str(raw_path),
                "view_count": view_count,
                "shape": list(batch_shape),
                "dtype": str(images.dtype),
                "numel": numel,
                "file_size": actual_bytes,
                "order": "batch index 0..V-1 exactly",
            }
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        except Exception:
            shutil.rmtree(spill_dir, ignore_errors=True)
            raise

        rss0 = _rss_snapshot()
        mem0 = _read_kb_file("/proc/meminfo")

        cache_key = _evict_upstream_cache(
            execution_list,
            upstream_id,
            upstream_socket,
            images,
            unique_id,
        )

        # Remove the exact original VSR anonymous storage after its persistent
        # Comfy cache entry and current execution edge have been removed.
        images.set_(torch.empty(0, dtype=images.dtype, device=images.device))
        original_storage_after = int(images.untyped_storage().nbytes())

        rss1 = _rss_snapshot()
        _, trim_status = _malloc_trim()
        rss2 = _rss_snapshot()
        mem2 = _read_kb_file("/proc/meminfo")

        report = (
            "Orbit3D VSR mmap spill: "
            f"V={view_count}; shape={batch_shape}; "
            f"original_storage={original_storage_bytes / _GIB:.3f}"
            f"->{original_storage_after / _GIB:.3f} GiB; "
            f"mmap_file={expected_bytes / _GIB:.3f} GiB; "
            f"mapped_storage={mmap_batch.untyped_storage().nbytes() / _GIB:.3f} GiB; "
            f"upstream={upstream_id}:0; cache_key={cache_key!r}; "
            f"VmRSS={_gib(rss0['VmRSS'])}->{_gib(rss1['VmRSS'])}"
            f"->{_gib(rss2['VmRSS'])}; "
            f"RssAnon={_gib(rss0['RssAnon'])}->{_gib(rss2['RssAnon'])}; "
            f"RssFile={_gib(rss0['RssFile'])}->{_gib(rss2['RssFile'])}; "
            f"MemAvailable={_gib(mem0.get('MemAvailable'))}"
            f"->{_gib(mem2.get('MemAvailable'))}; "
            f"{fadvise_status}; {trim_status}; "
            f"manifest={manifest_path}"
        )
        print(f"[Orbit3D VSR MMap Spill] {report}", flush=True)
        return first_image, mmap_batch, report


class Orbit3DMMapBatchToImageList:
    """Ordered zero-copy split: every list item is a view of one mmap storage."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"

    def run(self, image):
        if not torch.is_tensor(image) or image.ndim != 4:
            raise TypeError("image must be an IMAGE batch [V,H,W,C]")
        if image.device.type != "cpu":
            raise RuntimeError(
                f"MMap Batch to Image List expects CPU batch, got {image.device}"
            )
        if int(image.shape[0]) < 1:
            raise ValueError("IMAGE batch is empty")

        storage = image.untyped_storage()
        storage_ptr = int(storage.data_ptr())
        frame_stride = int(image.stride(0))

        items = [image[i:i + 1, ...] for i in range(int(image.shape[0]))]

        # Fail rather than silently clone/reorder.
        for i, item in enumerate(items):
            if int(item.untyped_storage().data_ptr()) != storage_ptr:
                raise RuntimeError(
                    f"MMap list item {i} does not share the batch storage"
                )
            expected_offset = int(image.storage_offset()) + i * frame_stride
            if int(item.storage_offset()) != expected_offset:
                raise RuntimeError(
                    "MMap list order/storage-offset violation: "
                    f"index={i}, offset={item.storage_offset()}, "
                    f"expected={expected_offset}"
                )

        print(
            "[Orbit3D VSR MMap List] "
            f"V={len(items)}; one_storage={storage.nbytes() / _GIB:.3f} GiB; "
            "order=0..V-1; copies=0",
            flush=True,
        )
        return (items,)


NODE_CLASS_MAPPINGS = {
    "Orbit3DImageBatchSpillMMap": Orbit3DImageBatchSpillMMap,
    "Orbit3DMMapBatchToImageList": Orbit3DMMapBatchToImageList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Orbit3DImageBatchSpillMMap": "Orbit3D - Spill Image Batch (MMap)",
    "Orbit3DMMapBatchToImageList": "Orbit3D - MMap Batch to Image List",
}
