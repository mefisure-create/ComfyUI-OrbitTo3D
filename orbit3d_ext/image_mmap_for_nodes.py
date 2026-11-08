"""Orbit3D patch 3.6.11: VSR mmap + native StartLoop For-mode experiment.

This module intentionally does NOT change reconstruction/model mechanics.
It adds:
  * one pre-loop node that performs the existing 3.6.10 VSR mmap spill and
    also exposes view_count, so the old Batch->ImageList conversion is no
    longer needed in the workflow;
  * one zero-copy per-iteration selector for For-mode StartLoop.

The selector returns image[index:index+1], i.e. a tensor view sharing the
same file-backed storage. No frame load, clone, reorder, or model inference
is performed here.
"""
from __future__ import annotations

import torch

from .image_mmap_nodes import Orbit3DImageBatchSpillMMap
from .image_spill_nodes import _GIB


class Orbit3DImageBatchSpillMMapIndexed(Orbit3DImageBatchSpillMMap):
    """3.6.10 spill + mmap, with view_count for StartLoop For mode."""

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("first_image", "mmap_batch", "view_count", "report")
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
        first_image, mmap_batch, report = super().run(
            images,
            cache_session,
            unique_id=unique_id,
            prompt=prompt,
            execution_list=execution_list,
        )
        view_count = int(mmap_batch.shape[0])
        if view_count < 1:
            raise RuntimeError("Mapped VSR batch unexpectedly has zero views")
        report = (
            report
            + f"; for_mode_view_count={view_count}; "
              "preloop_image_list=removed"
        )
        print(
            "[Orbit3D VSR MMap Indexed] "
            f"V={view_count}; mmap_storage="
            f"{mmap_batch.untyped_storage().nbytes() / _GIB:.3f} GiB; "
            "preloop_list_items=0",
            flush=True,
        )
        return first_image, mmap_batch, view_count, report


class Orbit3DMMapImageAtIndex:
    """Select one IMAGE view from a mapped batch without copying it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "index": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"

    def run(self, image, index):
        if not torch.is_tensor(image):
            raise TypeError(f"image must be torch.Tensor, got {type(image).__name__}")
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE batch [V,H,W,C], got {tuple(image.shape)}")
        if image.device.type != "cpu":
            raise RuntimeError(
                f"MMap Image At Index expects CPU file-backed batch, got {image.device}"
            )
        if image.dtype != torch.float32:
            raise RuntimeError(
                f"MMap Image At Index expects float32 VSR batch, got {image.dtype}"
            )

        i = int(index)
        view_count = int(image.shape[0])
        if i < 0 or i >= view_count:
            raise IndexError(f"MMap view index {i} out of range 0..{view_count - 1}")

        storage = image.untyped_storage()
        storage_ptr = int(storage.data_ptr())
        frame_stride = int(image.stride(0))
        expected_offset = int(image.storage_offset()) + i * frame_stride

        item = image[i:i + 1, ...]

        # Fail closed: this node exists specifically to avoid a per-view copy.
        if int(item.untyped_storage().data_ptr()) != storage_ptr:
            raise RuntimeError(
                f"MMap indexed view {i} does not share the source storage"
            )
        if int(item.storage_offset()) != expected_offset:
            raise RuntimeError(
                "MMap indexed view order/storage-offset violation: "
                f"index={i}, offset={item.storage_offset()}, "
                f"expected={expected_offset}"
            )
        if tuple(item.shape) != (1, *tuple(image.shape[1:])):
            raise RuntimeError(
                f"MMap indexed view has unexpected shape {tuple(item.shape)}"
            )

        # Keep normal logs quiet; first/last are enough to prove the contract
        # without producing 120 lines on every run.
        if i == 0 or i == view_count - 1:
            print(
                "[Orbit3D VSR MMap Index] "
                f"index={i}/{view_count - 1}; storage_offset={item.storage_offset()}; "
                "shared_storage=yes; copies=0",
                flush=True,
            )

        return (item,)


NODE_CLASS_MAPPINGS = {
    "Orbit3DImageBatchSpillMMapIndexed": Orbit3DImageBatchSpillMMapIndexed,
    "Orbit3DMMapImageAtIndex": Orbit3DMMapImageAtIndex,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Orbit3DImageBatchSpillMMapIndexed": "Orbit3D - Spill Image Batch (MMap / For)",
    "Orbit3DMMapImageAtIndex": "Orbit3D - MMap Image At Index",
}
