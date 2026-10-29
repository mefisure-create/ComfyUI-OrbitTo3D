# ORBIT3D_STAGE4_MOGE_API_V1
"""
Small list/batch adapters for ComfyUI's native MOGE_GEOMETRY packet.

They deliberately keep the public socket type MOGE_GEOMETRY in both
directions.  Orbit3DMultiViewConfig consumes the same native type directly.

The list->batch adapter offloads tensor fields to CPU before concatenation.
This preserves the previous Orbit adapter's VRAM-friendly behavior when a
mapped MoGe branch produces one geometry packet per view.
"""

from __future__ import annotations

import torch


# Current native ComfyUI MoGe fields.  The model may omit some of them
# (for example older MoGe revisions may not provide normals).
_FIELD_NDIMS = {
    "points": (4, 3),
    "depth": (3, 2),
    "intrinsics": (3, 2),
    "mask": (3, 2),
    "normal": (4, 3),
    "image": (4, 3),
}


def _as_batched_tensor(key, value, label):
    if not torch.is_tensor(value):
        raise TypeError(
            f"{label}[{key!r}] must be torch.Tensor, "
            f"got {type(value).__name__}"
        )

    batched_ndim, single_ndim = _FIELD_NDIMS[key]
    if value.ndim == single_ndim:
        value = value.unsqueeze(0)
    if value.ndim != batched_ndim:
        raise ValueError(
            f"{label}[{key!r}] has shape {tuple(value.shape)}; "
            f"expected {batched_ndim}D batched tensor"
        )

    if key in ("points", "normal") and value.shape[-1] != 3:
        raise ValueError(
            f"{label}[{key!r}] has shape {tuple(value.shape)}; "
            "last dimension must be 3"
        )
    if key == "intrinsics" and tuple(value.shape[-2:]) != (3, 3):
        raise ValueError(
            f"{label}['intrinsics'] has shape {tuple(value.shape)}; "
            "expected [...,3,3]"
        )

    # Match the old adapter's important memory behavior: mapped per-view
    # geometry is moved off the GPU before one large batch is constructed.
    value = value.detach().to("cpu")
    if key == "mask":
        value = value.bool()
    return value


def _normalize_geometry(g, label):
    if not isinstance(g, dict):
        raise TypeError(
            f"{label} must be a MOGE_GEOMETRY dictionary, "
            f"got {type(g).__name__}"
        )

    out = {}
    batch_size = None

    for key, value in g.items():
        if key in _FIELD_NDIMS:
            value = _as_batched_tensor(key, value, label)
            b = int(value.shape[0])
            if batch_size is None:
                batch_size = b
            elif b != batch_size:
                raise ValueError(
                    f"{label}: field {key!r} has batch {b}, "
                    f"expected {batch_size}"
                )
            out[key] = value
        else:
            # Preserve future/nonstandard metadata without interpreting it.
            out[key] = value

    if batch_size is None:
        raise ValueError(
            f"{label} has none of the recognized batched MoGe fields: "
            f"{', '.join(_FIELD_NDIMS)}"
        )

    return out, batch_size


def _same_metadata(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return False
    try:
        result = a == b
    except Exception:
        return a is b
    return bool(result) if isinstance(result, bool) else False


def _concat_geometries(values):
    if not values:
        raise ValueError("No connected MOGE_GEOMETRY inputs.")

    prepared = []
    batch_sizes = []
    for i, g in enumerate(values):
        item, b = _normalize_geometry(g, f"moge_geometry_{i + 1}")
        prepared.append(item)
        batch_sizes.append(b)

    all_keys = set().union(*(g.keys() for g in prepared))
    result = {}

    for key in sorted(all_keys):
        present = [key in g for g in prepared]
        if not all(present):
            raise ValueError(
                f"MOGE_GEOMETRY field {key!r} is present only in some list "
                "items; batching would misalign views."
            )

        vals = [g[key] for g in prepared]

        if key in _FIELD_NDIMS:
            # Spatial sizes and remaining dimensions must agree; torch.cat will
            # report the exact mismatch, but this message is easier to diagnose.
            ref_tail = tuple(vals[0].shape[1:])
            for i, v in enumerate(vals[1:], start=2):
                if tuple(v.shape[1:]) != ref_tail:
                    raise ValueError(
                        f"moge_geometry_{i}[{key!r}] tail shape "
                        f"{tuple(v.shape[1:])} does not match {ref_tail}"
                    )
            result[key] = torch.cat(vals, dim=0)
            continue

        # Unknown metadata is retained only when every item agrees.  This avoids
        # silently attaching one view's metadata to a multi-view batch.
        first = vals[0]
        if all(_same_metadata(first, v) for v in vals[1:]):
            result[key] = first
        else:
            raise ValueError(
                f"Cannot safely batch nonstandard MOGE_GEOMETRY field {key!r}; "
                "values differ between list items."
            )

    return result


def _geometry_batch_size(g):
    normalized, batch_size = _normalize_geometry(g, "moge_geometry")
    return normalized, batch_size


class Orbit3DMoGeGeometryListToBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"moge_geometry": ("MOGE_GEOMETRY",)}}

    # Comfy list semantics: collect all mapped upstream packets in one call.
    INPUT_IS_LIST = True
    RETURN_TYPES = ("MOGE_GEOMETRY",)
    RETURN_NAMES = ("moge_geometry_batch",)
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Geometry"

    def run(self, moge_geometry):
        if not isinstance(moge_geometry, (list, tuple)):
            moge_geometry = [moge_geometry]
        values = [g for g in moge_geometry if g is not None]
        return (_concat_geometries(values),)


class Orbit3DMoGeGeometryBatchToList:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"moge_geometry": ("MOGE_GEOMETRY",)}}

    RETURN_TYPES = ("MOGE_GEOMETRY",)
    RETURN_NAMES = ("moge_geometry_list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Geometry"

    def run(self, moge_geometry):
        normalized, batch_size = _geometry_batch_size(moge_geometry)

        out = []
        for i in range(batch_size):
            item = {}
            for key, value in normalized.items():
                if key in _FIELD_NDIMS:
                    item[key] = value[i:i + 1]
                else:
                    item[key] = value
            out.append(item)

        return (out,)
