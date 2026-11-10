"""Isolated SS64 -> Shape1024 Direct experiment.

The Shape1024 denoiser and its native 1024px/R64 DINO conditioner are used
WITHOUT a preceding Shape512/Cascade reconstruction. This is intentionally
out-of-distribution for Shape1024's *input occupancy*, even though the grid
resolution matches its normal conditioning scale.

No production classes, sampler math, pretrained weights, or geometry functions
are patched here.
"""
from __future__ import annotations

import torch

from .. import nodes as _trellis
from ..orbit_nodes import (
    _normalize_mode,
    _normalize_source,
    _require_geometry_sources,
    _require_pixal3d_mv,
)


class Orbit3DSS64Shape1024DirectExperimental:
    """Sample Shape1024 weights directly on a validated SS64 occupancy grid."""

    CATEGORY = "Orbit to 3D/Experimental"
    FUNCTION = "process"
    RETURN_TYPES = ("SHAPE_SLAT", "INT", "TRELLIS2PIPELINE", "INT")
    RETURN_NAMES = ("shape_slat", "decode_resolution", "pipeline", "num_tokens")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        # Reuse only the standard Shape sampler controls, not the Cascade
        # coordinate upsample/quantization or any previous Shape512 SLat.
        base = _trellis.Trellis2ShapeGenerator.INPUT_TYPES()
        required = dict(base["required"])
        required.pop("resolution", None)
        required.update({
            "sparse_structure_resolution": (
                "INT",
                {
                    "forceInput": True,
                    "tooltip": "Connect the actual resolution from Sparse Generator / Load Sparse Stage. Exactly 64 is required.",
                },
            ),
            "max_num_tokens": (
                "INT",
                {
                    "default": 12000, "min": 1, "max": 999999, "step": 100,
                    "tooltip": "Abort if token count exceeds this limit; does NOT remove any voxels. Raise cautiously (full attention).",
                },
            ),
            "shape_hr_mode": (
                ["off", "normal", "depth", "normal+depth"],
                {"default": "normal+depth", "tooltip": "Uses the stock ShapeHR geometry stage; no custom geometry weighting."},
            ),
            "shape_hr_normals_source": (
                ["Marigold", "MoGe"],
                {"default": "Marigold"},
            ),
            "shape_hr_depth_source": (
                ["Marigold", "MoGe"],
                {"default": "MoGe"},
            ),
        })
        return {
            "required": required,
            "optional": {"pixal3d_mv_views": ("PIXAL3D_MV_VIEWS",)},
        }

    def process(
        self,
        pipeline,
        coords,
        sparse_structure_resolution,
        shape_steps,
        shape_guidance_strength,
        shape_guidance_rescale,
        shape_rescale_t,
        shape_sampler,
        shape_guidance_interval_start,
        shape_guidance_interval_end,
        verbose,
        dino_lock,
        dino_substeps,
        dino_foundation_cap,
        shape_seed_mode="inherit",
        shape_seed=0,
        max_num_tokens=12000,
        shape_hr_mode="normal+depth",
        shape_hr_normals_source="Marigold",
        shape_hr_depth_source="MoGe",
        pixal3d_mv_views=None,
    ):
        _require_pixal3d_mv(pipeline)
        if pixal3d_mv_views is None:
            raise ValueError("Shape1024 Direct requires PIXAL3D_MV_VIEWS")

        grid = int(sparse_structure_resolution)
        if grid != 64:
            raise ValueError(
                f"Shape1024 Direct expects SS64 (sparse_structure_resolution=64); got {grid}"
            )
        if not isinstance(coords, torch.Tensor) or coords.ndim != 2 or coords.shape[1] != 4:
            raise TypeError(
                f"Expected COORDS tensor [N,4], got {type(coords)} {getattr(coords, 'shape', None)}"
            )
        if coords.is_floating_point() or coords.is_complex():
            raise TypeError("COORDS must contain integer [batch,x,y,z] values")
        count = int(coords.shape[0])
        if count == 0:
            raise ValueError("SS64 contains no occupied cells")
        if count > int(max_num_tokens):
            raise RuntimeError(
                f"SS64 contains {count} tokens, exceeding max_num_tokens={int(max_num_tokens)}. "
                "No tokens were removed. Increase the limit only if sufficient VRAM is available."
            )

        # Fail before loading DINO or Shape weights if the coordinate contract
        # does not match R64. No recentering, sorting, or silent deduplication.
        coords_cpu = coords.detach().cpu()
        if int(coords_cpu[:, 0].min()) != 0 or int(coords_cpu[:, 0].max()) != 0:
            raise ValueError("This experiment supports only batch index 0")
        if int(coords_cpu[:, 1:].min()) < 0 or int(coords_cpu[:, 1:].max()) >= 64:
            raise ValueError("Spatial coordinates must lie within [0,63]^3")
        if int(torch.unique(coords_cpu, dim=0).shape[0]) != count:
            raise ValueError("Duplicate SS64 coordinates found; refusing implicit deduplication")
        del coords_cpu

        mode = _normalize_mode(shape_hr_mode, "shape_hr_mode")
        normal = _normalize_source(shape_hr_normals_source, "shape_hr_normals_source")
        depth = _normalize_source(shape_hr_depth_source, "shape_hr_depth_source")
        _require_geometry_sources(
            pixal3d_mv_views, mode, normal, depth, "Shape1024 Direct",
        )

        seed_mode = str(shape_seed_mode).strip().lower()
        if seed_mode not in ("inherit", "fixed"):
            raise ValueError(f"invalid shape_seed_mode={shape_seed_mode!r}")

        # Make a stage-local copy, just as the production Cascade does.
        views = dict(pixal3d_mv_views)
        views["shape_hr_mode"] = mode
        views["shape_hr_normals_source"] = normal
        views["shape_hr_depth_source"] = depth

        images = views.get("images", {})
        if 1024 not in images:
            raise ValueError(
                "Shape1024 Direct needs 1024px images in PIXAL3D_MV_VIEWS. "
                "Connect the regular MultiView Config output containing the 1024 stage."
            )

        params = {
            "steps": int(shape_steps),
            "guidance_strength": shape_guidance_strength,
            "guidance_rescale": shape_guidance_rescale,
            "guidance_interval": [
                shape_guidance_interval_start, shape_guidance_interval_end,
            ],
            "rescale_t": shape_rescale_t,
        }
        args = pipeline._pretrained_args
        sampler_prefix = pipeline.GetSamplerName(shape_sampler)
        pipeline.shape_slat_sampler = getattr(
            _trellis.samplers,
            f"Flow{sampler_prefix}GuidanceIntervalSampler",
        )(**args["shape_slat_sampler"]["args"])

        decode_resolution = 1024
        print(
            f"[Orbit3D SS64 -> Shape1024 Direct EXP] occupied={count}, "
            f"weights=shape_slat_flow_model_1024, "
            f"DINO=1024px, proj_grid=64^3, decode=1024; "
            f"geometry={mode}, normal={normal}, depth={depth}; "
            f"seed_mode={seed_mode}; "
            "no Shape512, no Cascade, no token changes",
            flush=True,
        )

        # Native Shape1024 image conditioner: 1024px DINO/NAF, R64 projector,
        # original ShapeHR visibility, normal and depth processing.
        image_cond_model = pipeline.load_pixal3d_mv_image_cond_shape_1024()
        cond = None
        try:
            if int(image_cond_model.grid_resolution) != 64 or int(image_cond_model.image_size) != 1024:
                raise RuntimeError(
                    "Unexpected Shape1024 conditioning config; expected image_size=1024, grid_resolution=64"
                )
            cond = pipeline.get_proj_cond_shape_mv(
                image_cond_model, views, coords,
                geometry_stage="shape_hr",
            )
        finally:
            if cond is None or not pipeline.keep_models_loaded:
                pipeline.unload_pixal3d_mv_image_cond_shape_1024()

        # Match the standard cascade's Shape1024 flow weights and sampler.
        # Crucially, do not run its SLat decoder / coordinate upsample first.
        pipeline.unload_shape_slat_flow_model_512()
        pipeline.load_shape_slat_flow_model_1024()
        try:
            with _trellis._orbit3d_seed_scope(seed_mode, shape_seed):
                slat = pipeline.sample_shape_slat(
                    cond,
                    pipeline.models["shape_slat_flow_model_1024"],
                    coords,
                    params,
                    verbose=verbose,
                    dino_lock=dino_lock,
                    dino_substeps=dino_substeps,
                    dino_foundation_cap=dino_foundation_cap,
                )
        finally:
            if not pipeline.keep_models_loaded:
                pipeline.unload_shape_slat_flow_model_1024()

        if int(slat.coords.shape[0]) != count:
            raise RuntimeError(
                f"Shape1024 returned {slat.coords.shape[0]} tokens; SS64 had {count}"
            )

        print(
            f"[Orbit3D SS64 -> Shape1024 Direct EXP] completed: "
            f"tokens={count}, decode_resolution={decode_resolution}",
            flush=True,
        )
        return slat, decode_resolution, pipeline, count
