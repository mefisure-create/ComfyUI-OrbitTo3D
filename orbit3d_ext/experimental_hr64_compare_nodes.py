"""Isolated diagnostics: extract the pre-Shape1024 Cascade HR64 occupancy and
compare it with the native SS64 occupancy in the same normalized 64**3 grid.

Does not modify Shape sampling, conditioning, VAE weights, or production nodes.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import torch

from ..orbit_nodes import _require_pixal3d_mv


GRID_RESOLUTION = 64


def _validate_coords(coords, label):
    if not isinstance(coords, torch.Tensor) or coords.ndim != 2 or coords.shape[1] != 4:
        raise TypeError(f"{label}: expected COORDS tensor [N,4]")
    if coords.is_floating_point() or coords.is_complex():
        raise TypeError(f"{label}: coordinates must be integers")
    c = coords.detach().to(device="cpu", dtype=torch.int64).contiguous()
    if c.numel() == 0:
        raise ValueError(f"{label}: empty coordinate array")
    if bool((c[:, 0] != 0).any()):
        raise ValueError(f"{label}: only batch 0 is supported")
    if bool((c[:, 1:] < 0).any()) or bool((c[:, 1:] >= GRID_RESOLUTION).any()):
        raise ValueError(f"{label}: spatial coordinates must be inside [0,63]^3")
    if torch.unique(c, dim=0).shape[0] != c.shape[0]:
        raise ValueError(f"{label}: duplicated 3D coordinates detected")
    return c.numpy()


class Orbit3DExtractCascadeHR64CoordsExperimental:
    """Export exact Cascade R64 tokens from a normal SS32->Shape512 SLat.

    Computes only the stock VAE.upample(..., upsample_times=4) coordinate
    proposal and its stock integer quantization for LR512, SS32 -> HR1024.
    Shape1024 is NOT loaded or sampled.
    """

    CATEGORY = "Orbit to 3D/Experimental"
    FUNCTION = "process"
    RETURN_TYPES = ("COORDS", "INT", "TRELLIS2PIPELINE", "INT")
    RETURN_NAMES = ("hr64_coords", "hr_grid_resolution", "pipeline", "num_tokens")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "shape_slat": ("SHAPE_SLAT",),
                "shape512_resolution": ("INT", {"forceInput": True}),
                "sparse_structure_resolution": ("INT", {"forceInput": True}),
            },
        }

    @torch.no_grad()
    def process(self, pipeline, shape_slat, shape512_resolution, sparse_structure_resolution):
        _require_pixal3d_mv(pipeline)
        if int(shape512_resolution) != 512:
            raise ValueError(
                "Extract Cascade HR64 requires the *regular* Shape512 SLat "
                f"(resolution=512), got {shape512_resolution}."
            )
        if int(sparse_structure_resolution) != 32:
            raise ValueError(
                "Extract Cascade HR64 requires regular SS32->Shape512, "
                f"not SS{sparse_structure_resolution}->Shape512."
            )
        if not hasattr(shape_slat, "coords") or not hasattr(shape_slat, "feats"):
            raise TypeError("shape_slat must be the regular SHAPE_SLAT SparseTensor")
        if shape_slat.coords.ndim != 2 or shape_slat.coords.shape[1] != 4:
            raise ValueError("Shape512 SLat coordinates must have shape [N,4]")
        if int(shape_slat.coords[:, 1:].min()) < 0 or int(shape_slat.coords[:, 1:].max()) >= 32:
            raise ValueError("Shape512 SLat is not on its expected 32^3 grid")
        if int(shape_slat.coords.shape[0]) != int(shape_slat.feats.shape[0]):
            raise ValueError("Shape512 coords/features count mismatch")

        pipeline.load_shape_slat_decoder()
        decoder = pipeline.models["shape_slat_decoder"]
        try:
            if pipeline.low_vram:
                decoder.to(pipeline.device)
                decoder.low_vram = True
            raw = decoder.upsample(shape_slat, upsample_times=4)
        finally:
            if pipeline.low_vram:
                decoder.cpu()
                decoder.low_vram = False
            if not pipeline.keep_models_loaded:
                pipeline.unload_shape_slat_decoder()

        # Verbatim Cascade.sample() formula at to_resolution=1024:
        lr_resolution = 512
        ss_resolution = 32
        hr_resolution = 1024
        ratio = ss_resolution / 32
        quant = torch.cat([
            raw[:, :1],
            ((raw[:, 1:] + 0.5) / (lr_resolution * ratio)
             * (hr_resolution // 16)).int(),
        ], dim=1)
        coords = quant.unique(dim=0)
        n = int(coords.shape[0])
        _validate_coords(coords, "Cascade HR64")
        print(
            f"[Orbit3D HR64 Extract] Shape512 tokens={int(shape_slat.coords.shape[0])}; "
            f"raw VAE upsample coords={int(raw.shape[0])}; "
            f"quantized HR64 coords={n}; no Shape1024 generation",
            flush=True,
        )
        return coords.cpu(), 64, pipeline, n


def _mask_from_coords(coords_np):
    mask = np.zeros((64, 64, 64), dtype=np.bool_)
    coords_xyz = coords_np[:, 1:]
    mask[coords_xyz[:, 0], coords_xyz[:, 1], coords_xyz[:, 2]] = True
    return mask


def _stats(mask):
    from scipy.ndimage import generate_binary_structure, label

    vox = np.argwhere(mask)
    bbox_min = vox.min(axis=0).tolist()
    bbox_max = vox.max(axis=0).tolist()
    comps = {}
    for connectivity, st in ((6, generate_binary_structure(3, 1)),
                             (26, np.ones((3, 3, 3), dtype=bool))):
        labeled, count = label(mask, structure=st)
        sizes = np.bincount(labeled[mask])
        comps[str(connectivity)] = {
            "count": int(count),
            "largest": int(sizes[1:].max()) if count else 0,
            "singletons": int(np.count_nonzero(sizes[1:] == 1)),
            "under_10": int(np.count_nonzero(sizes[1:] < 10)),
        }
    return {
        "voxels": int(vox.shape[0]),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "components": comps,
        "occupancy_per_axis": {
            "x": mask.sum(axis=(1, 2)).astype(int).tolist(),
            "y": mask.sum(axis=(0, 2)).astype(int).tolist(),
            "z": mask.sum(axis=(0, 1)).astype(int).tolist(),
        },
    }


def _write_colored_point_cloud(path, ss_only, hr_only, shared):
    # Geometry is a voxel-center *point cloud*, NOT a reconstructed mesh.
    groups = ((ss_only, (245, 91, 91)),
              (hr_only, (75, 137, 240)),
              (shared, (61, 190, 123)))
    total = sum(int(np.count_nonzero(m)) for m, _ in groups)
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {total}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for mask, rgb in groups:
            for x, y, z in np.argwhere(mask):
                stream.write(
                    f"{(x+0.5)/64:.8f} {(y+0.5)/64:.8f} {(z+0.5)/64:.8f} "
                    f"{rgb[0]} {rgb[1]} {rgb[2]}\n"
                )


class Orbit3DCompareSS64CascadeHR64Experimental:
    """Compare exact SS64 and Cascade HR64 occupancy, optionally persist report."""

    CATEGORY = "Orbit to 3D/Experimental"
    FUNCTION = "process"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("summary", "report_directory")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "ss64_coords": ("COORDS",),
            "ss64_resolution": ("INT", {"forceInput": True}),
            "hr64_coords": ("COORDS",),
            "hr64_resolution": ("INT", {"forceInput": True}),
            "report_parent_directory": (
                "STRING",
                {"default": "/home/ai/orbit3d-cache/experiments/ss64_hr64_compare", "multiline": False},
            ),
        }}

    def process(self, ss64_coords, ss64_resolution, hr64_coords, hr64_resolution,
                report_parent_directory):
        if int(ss64_resolution) != GRID_RESOLUTION or int(hr64_resolution) != GRID_RESOLUTION:
            raise ValueError("Comparison requires BOTH coordinate grids to have resolution 64")
        a = _mask_from_coords(_validate_coords(ss64_coords, "SS64"))
        b = _mask_from_coords(_validate_coords(hr64_coords, "HR64"))
        shared = a & b
        a_only = a & ~b
        b_only = b & ~a
        union = a | b
        n_a, n_b = int(a.sum()), int(b.sum())
        n_shared = int(shared.sum())
        from scipy.ndimage import binary_dilation
        # Chebyshev <= 1 voxel spatial proximity (same R64 space).
        near_b = binary_dilation(b, structure=np.ones((3, 3, 3), dtype=bool))
        near_a = binary_dilation(a, structure=np.ones((3, 3, 3), dtype=bool))
        stats = {
            "grid_resolution": 64,
            "ss64": _stats(a),
            "cascade_hr64": _stats(b),
            "comparison": {
                "shared": n_shared,
                "ss64_only": int(a_only.sum()),
                "cascade_hr64_only": int(b_only.sum()),
                "union": int(union.sum()),
                "intersection_over_union": n_shared / int(union.sum()),
                "ss64_overlap_fraction": n_shared / n_a,
                "cascade_hr64_overlap_fraction": n_shared / n_b,
                "ss64_within_1_voxel_of_hr64_fraction": float((a & near_b).sum()) / n_a,
                "hr64_within_1_voxel_of_ss64_fraction": float((b & near_a).sum()) / n_b,
            },
        }
        base = Path(str(report_parent_directory).strip()).expanduser().resolve()
        if not str(report_parent_directory).strip():
            raise ValueError("report_parent_directory must not be empty")
        base.mkdir(parents=True, exist_ok=True)
        outdir = base / f"comparison_{uuid.uuid4().hex[:12]}"
        outdir.mkdir(exist_ok=False)
        (outdir / "report.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.savez_compressed(
            outdir / "coordinates.npz",
            ss64_coords=np.argwhere(a).astype(np.int16),
            cascade_hr64_coords=np.argwhere(b).astype(np.int16),
        )
        _write_colored_point_cloud(outdir / "comparison_points.ply", a_only, b_only, shared)
        summary = (
            f"SS64={n_a}; Cascade HR64={n_b}; shared={n_shared}; "
            f"IoU={stats['comparison']['intersection_over_union']:.4f}; "
            f"SS64 within 1 voxel of HR64={stats['comparison']['ss64_within_1_voxel_of_hr64_fraction']:.2%}; "
            f"HR64 within 1 voxel of SS64={stats['comparison']['hr64_within_1_voxel_of_ss64_fraction']:.2%}; "
            f"6-components SS64/HR64={stats['ss64']['components']['6']['count']}/"
            f"{stats['cascade_hr64']['components']['6']['count']}"
        )
        print(f"[Orbit3D SS64 vs HR64] {summary}\nResults: {outdir}", flush=True)
        return summary, str(outdir)
