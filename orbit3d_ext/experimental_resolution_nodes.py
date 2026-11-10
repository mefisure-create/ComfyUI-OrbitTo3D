"""Isolated SS64 -> Shape512-weight resolution experiment.

Only this node changes the projection-grid size passed to Shape512 conditioning.
It does not patch production generators, the Pixal3D pipeline, or model weights.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

from ..trellis2.modules.sparse import SparseTensor
from ..trellis2.modules.sparse.attention.rope import SparseRotaryPositionEmbedder

from .. import nodes as _trellis
from ..orbit_nodes import (
    _normalize_mode,
    _normalize_source,
    _require_geometry_sources,
    _require_pixal3d_mv,
)


@contextmanager
def _temporary_rope_coordinate_scale(flow_model, scale: float):
    """Change ONLY the RoPE physical coordinate scale in this model instance.

    Geometry, sparse-coordinate indexing and DINO projection retain their
    original integer 64^3 coordinates. Restore instance methods even on OOM.
    New SparseTensor noise is constructed each run, so RoPE spatial phase
    caches do not carry between comparison runs.
    """
    if abs(float(scale) - 1.0) < 1e-9:
        yield 0
        return

    if getattr(flow_model, "pe_mode", None) != "rope":
        raise RuntimeError(
            "RoPE normalization requested but Shape512 flow_model.pe_mode is not 'rope'"
        )

    modules = [
        module for module in flow_model.modules()
        if isinstance(module, SparseRotaryPositionEmbedder)
    ]
    if not modules:
        raise RuntimeError(
            "RoPE normalization requested but no SparseRotaryPositionEmbedder modules were found"
        )

    old_methods = []
    try:
        for module in modules:
            old_get_phases = module._get_phases

            def _scaled_phases(indices, original=old_get_phases, factor=float(scale)):
                return original(indices.to(dtype=torch.float32) * factor)

            module._get_phases = _scaled_phases
            old_methods.append((module, old_get_phases))
        yield len(modules)
    finally:
        for module, original in old_methods:
            module._get_phases = original


class Orbit3DSS64Shape512Experimental:
    """Run *Shape512 weights* on native 64^3 SS coords (OOD experiment).

    R=32 is also allowed as a baseline sanity check. The returned resolution
    is the voxel/mesh decode scale (R*16), NOT the weights used for generation.
    This is deliberately not a Cascade or a local-ROI repair node.
    """

    CATEGORY = "Orbit to 3D/Experimental"
    FUNCTION = "process"
    RETURN_TYPES = ("SHAPE_SLAT", "INT", "TRELLIS2PIPELINE", "INT")
    RETURN_NAMES = ("shape_slat", "decode_resolution", "pipeline", "num_tokens")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        base = _trellis.Trellis2ShapeGenerator.INPUT_TYPES()
        required = dict(base["required"])
        required.pop("resolution", None)
        required.update({
            "sparse_structure_resolution": (
                "INT", {"forceInput": True,
                        "tooltip": "Connect the resolution output from Sparse Generator / Load Sparse Stage. Must be 64 for the SS64 experiment."},
            ),
            "shape512_mode": (
                ["off", "normal", "depth", "normal+depth"],
                {"default": "normal+depth"},
            ),
            "shape512_normals_source": (
                ["Marigold", "MoGe"], {"default": "Marigold"},
            ),
            "shape512_depth_source": (
                ["Marigold", "MoGe"], {"default": "MoGe"},
            ),
            "detail_tile_grid": (
                "INT", {"default": 3, "min": 3, "max": 5, "step": 1,
                        "tooltip": "3=the accepted DINO baseline; 4/5 require the previously installed tile-grid patch."},
            ),
            "detail_cond_mode": (
                ["blend", "full_only", "tiles_only"],
                {"default": "blend",
                 "tooltip": "blend=baseline; other modes require the earlier DINO ablation patch."},
            ),
            "max_num_tokens": (
                "INT", {"default": 12000, "min": 1, "max": 999999, "step": 100,
                        "tooltip": "OOM safety guard only: fail before conditioning if SS tokens exceed the limit. No token filtering."},
            ),
            "geometry_world_scale_mode": (
                ["legacy", "visibility_only", "depth_only", "world_matched"],
                {"default": "world_matched", "tooltip": (
                    "legacy=original R64 behavior; visibility_only rescales sparse visibility; "
                    "depth_only rescales depth agreement; world_matched rescales both "
                    "to preserve R32 world-space tolerances."
                )},
            ),
            "rope_coordinate_mode": (
                ["native", "world_matched_32"],
                {"default": "native", "tooltip": (
                    "native=exact previous experiment, RoPE uses integer SS64 coords; "
                    "world_matched_32=scale ONLY the RoPE phase positions by 31/(R-1), "
                    "preserving their physical spacing as at Shape512 R32. "
                    "SS coords, DINO, geometry and decoder remain unchanged."
                )},
            ),
        })
        return {
            "required": required,
            "optional": {
                "pixal3d_mv_views": ("PIXAL3D_MV_VIEWS",),
            },
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
        shape512_mode="normal+depth",
        shape512_normals_source="Marigold",
        shape512_depth_source="MoGe",
        detail_tile_grid=3,
        detail_cond_mode="blend",
        max_num_tokens=12000,
        geometry_world_scale_mode="world_matched",
        rope_coordinate_mode="native",
        pixal3d_mv_views=None,
    ):
        _require_pixal3d_mv(pipeline)
        if pixal3d_mv_views is None:
            raise ValueError("SS64 Shape512 experiment requires pixal3d_mv_views")

        grid = int(sparse_structure_resolution)
        if grid not in (32, 64):
            raise ValueError(
                f"SS64 Shape512 experiment accepts SS resolutions 32 or 64, got {grid}. "
                "Other grids require separate validation."
            )
        if not isinstance(coords, torch.Tensor) or coords.ndim != 2 or coords.shape[1] != 4:
            raise TypeError("Expected COORDS torch.Tensor [N,4] from Sparse Generator")
        if coords.is_floating_point() or coords.is_complex():
            raise TypeError("Sparse COORDS must be integer coordinates")
        count = int(coords.shape[0])
        if count == 0:
            raise ValueError("Empty Sparse Structure: no Shape tokens to generate")
        if count > int(max_num_tokens):
            raise RuntimeError(
                f"SS64 Shape512 experiment: {count} sparse tokens exceed "
                f"max_num_tokens={int(max_num_tokens)}. This guard does NOT "
                "truncate the model. Increase it only if sufficient VRAM is available."
            )
        bounds = coords.detach().to("cpu")
        if int(bounds[:, 0].min()) != 0 or int(bounds[:, 0].max()) != 0:
            raise ValueError("The experimental node currently supports batch index 0 only")
        if int(bounds[:, 1:].min()) < 0 or int(bounds[:, 1:].max()) >= grid:
            raise ValueError(
                f"COORDS are not in the expected [0,{grid - 1}] spatial grid; "
                "check that sparse_structure_resolution matches the connected SS"
            )
        if torch.unique(bounds, dim=0).shape[0] != count:
            raise ValueError("SS contains duplicate coordinates; refusing silent deduplication")
        del bounds

        mode = _normalize_mode(shape512_mode, "shape512_mode")
        normal_source = _normalize_source(shape512_normals_source, "shape512_normals_source")
        depth_source = _normalize_source(shape512_depth_source, "shape512_depth_source")
        _require_geometry_sources(pixal3d_mv_views, mode, normal_source, depth_source,
                                  "Experimental SS64 Shape512")
        if int(detail_tile_grid) not in (3, 4, 5):
            raise ValueError("detail_tile_grid must be 3, 4, or 5")
        dino_mode = str(detail_cond_mode).strip().lower()
        if dino_mode not in ("blend", "full_only", "tiles_only"):
            raise ValueError(f"invalid detail_cond_mode={dino_mode!r}")
        if str(shape_seed_mode).strip().lower() not in ("inherit", "fixed"):
            raise ValueError(f"invalid shape_seed_mode={shape_seed_mode!r}")

        geo_scale_mode = str(geometry_world_scale_mode).strip().lower()
        if geo_scale_mode not in ("legacy", "visibility_only", "depth_only", "world_matched"):
            raise ValueError(f"invalid geometry_world_scale_mode={geo_scale_mode!r}")
        rope_mode = str(rope_coordinate_mode).strip().lower()
        if rope_mode not in ("native", "world_matched_32"):
            raise ValueError(f"invalid rope_coordinate_mode={rope_mode!r}")
        rope_scale = (31.0 / float(grid - 1)) if rope_mode == "world_matched_32" else 1.0

        # A shallow views copy isolates the stage settings from all other nodes.
        views = dict(pixal3d_mv_views)
        views["marigold_shape512_mode"] = mode  # upstream-compatible legacy key
        views["shape512_normals_source"] = normal_source
        views["shape512_depth_source"] = depth_source
        views["shape512_detail_tile_grid"] = int(detail_tile_grid)
        views["shape512_detail_cond_mode"] = dino_mode

        decode_resolution = grid * 16
        print(
            f"[Orbit3D SS64->Shape512 EXP] SS grid={grid}^3; "
            f"coords={count}; weights=shape_slat_flow_model_512; "
            f"DINO image_size=512, projection={grid}^3; "
            f"decoder_resolution={decode_resolution}; "
            f"geometry={mode} normal={normal_source} depth={depth_source}; "
            f"geometry_world_scale_mode={geo_scale_mode}; "
            f"rope_coordinate_mode={rope_mode}; rope_scale={rope_scale:.7f}; "
            f"detail={int(detail_tile_grid)}x{int(detail_tile_grid)}/{dino_mode}; "
            f"seed_mode={shape_seed_mode}; "
            "no resampling, no added/removed tokens; "
            "model used outside its trained spatial scale if grid=64",
            flush=True,
        )

        # Match the existing Shape512 sampler selection and parameters exactly.
        params = {
            "steps": int(shape_steps),
            "guidance_strength": shape_guidance_strength,
            "guidance_rescale": shape_guidance_rescale,
            "guidance_interval": [shape_guidance_interval_start, shape_guidance_interval_end],
            "rescale_t": shape_rescale_t,
        }
        args = pipeline._pretrained_args
        sampler_prefix = pipeline.GetSamplerName(shape_sampler)
        pipeline.shape_slat_sampler = getattr(
            _trellis.samplers,
            f"Flow{sampler_prefix}GuidanceIntervalSampler",
        )(**args["shape_slat_sampler"]["args"])

        # Using the production method's already-supported grid override avoids
        # changing image_size, DINO/NAF mechanics, or the production Shape node.
        image_cond_model = pipeline.load_pixal3d_mv_image_cond_shape_512()
        projected = None
        try:
            if geo_scale_mode == "legacy":
                # Exact production conditioning path used by the original
                # SS64 experiment, kept for controlled A/B comparisons.
                projected = pipeline.get_proj_cond_shape_mv(
                    image_cond_model,
                    views,
                    coords,
                    grid_resolution_override=grid,
                    geometry_stage="shape512",
                )
            else:
                projected = self._project_with_world_scale_controls(
                    pipeline, image_cond_model, views, coords, grid,
                    geo_scale_mode,
                )
        finally:
            # On failure, its internal projection-grid restoration is not
            # guaranteed. Unload the experimental instance in that case.
            if projected is None or not pipeline.keep_models_loaded:
                pipeline.unload_pixal3d_mv_image_cond_shape_512()

        # Production code unloads the other Shape model before loading 512.
        pipeline.unload_shape_slat_flow_model_1024()
        pipeline.load_shape_slat_flow_model_512()
        try:
            with _trellis._orbit3d_seed_scope(shape_seed_mode, shape_seed):
                with _temporary_rope_coordinate_scale(
                    pipeline.models["shape_slat_flow_model_512"], rope_scale
                ) as n_rope_modules:
                    print(
                        f"[Orbit3D SS64->Shape512 EXP] RoPE mode={rope_mode}, "
                        f"phase_scale={rope_scale:.7f}, patched_modules={n_rope_modules}",
                        flush=True,
                    )
                    slat = pipeline.sample_shape_slat(
                        projected,
                        pipeline.models["shape_slat_flow_model_512"],
                        coords,
                        params,
                        verbose=verbose,
                        dino_lock=dino_lock,
                        dino_substeps=dino_substeps,
                        dino_foundation_cap=dino_foundation_cap,
                    )
        finally:
            if not pipeline.keep_models_loaded:
                pipeline.unload_shape_slat_flow_model_512()

        if int(slat.coords.shape[0]) != count:
            raise RuntimeError(
                f"Unexpected output token count: {slat.coords.shape[0]} != {count}"
            )
        print(
            f"[Orbit3D SS64->Shape512 EXP] completed; "
            f"SLat tokens={count}; decode_resolution={decode_resolution}",
            flush=True,
        )
        return (slat, decode_resolution, pipeline, count)


    @staticmethod
    def _project_with_world_scale_controls(
        pipeline, image_cond_model, views, coords, grid, scale_mode
    ):
        """Isolated copy of get_proj_cond_shape_mv with two tolerance controls.

        Uses the same visibility builder, geometry diagnostics, DINO extractor,
        weighted fusion, and SparseTensor creation as the production method.
        Only depth_tolerance_voxels and depth_tau_voxels can differ.
        """
        device = pipeline.device
        image_cond_model.to(device)
        original_grid = int(image_cond_model.grid_resolution)
        override = grid != original_grid
        if override:
            image_cond_model.grid_resolution = grid
            image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
                grid_resolution=grid,
                image_resolution=image_cond_model.proj_grid.image_resolution,
            ).to(device)

        vis_adjust = scale_mode in ("visibility_only", "world_matched")
        depth_adjust = scale_mode in ("depth_only", "world_matched")
        factor = max(grid - 1, 1) / 31.0
        vis_tol = 1.5 * (factor if vis_adjust else 1.0)
        depth_tau = 2.0 * (factor if depth_adjust else 1.0)
        print(
            f"[SS64 Shape512 geometry] grid={grid}, mode={scale_mode}, "
            f"visibility={vis_tol:.6f} vox, depth_tau={depth_tau:.6f} vox, "
            f"world_scale_factor={factor:.6f}",
            flush=True,
        )

        try:
            visibility_weights = None
            if getattr(image_cond_model, "multiview_fusion", None) == "weighted_average":
                visibility_weights = pipeline._build_sparse_mv_visibility(
                    image_cond_model, views, coords, depth_tolerance_voxels=vis_tol
                )

            stage_cfg = pipeline._shape_geometry_stage_config(
                image_cond_model, views, geometry_stage="shape512"
            )
            depth_weights = None
            if stage_cfg["stage"] != "off" and visibility_weights is not None:
                source_info = stage_cfg["source_info"]
                mode = stage_cfg["mode"]
                label = stage_cfg["label"]
                normal_source = source_info["normal_source"]
                depth_source = source_info["depth_source"]
                has_normal = views.get(source_info["normal_key"]) is not None
                has_depth = (
                    views.get(source_info["depth_key"]) is not None
                    and views.get(source_info["alpha_key"]) is not None
                )
                if mode in ("normal", "normal+depth") and not has_normal:
                    raise RuntimeError(
                        f"{label} normal mode requires source={normal_source}, "
                        f"but {source_info['normal_key']!r} is missing"
                    )

                need_diag = (
                    stage_cfg["stage"] == "shape512"
                    or mode in ("depth", "normal+depth")
                )
                if need_diag and (has_normal or has_depth):
                    depth_weights = pipeline._shape512_marigold_diagnostics(
                        image_cond_model,
                        views,
                        coords,
                        visibility_weights,
                        depth_floor=stage_cfg["depth_floor"],
                        depth_tau_voxels=depth_tau,
                        geometry_stage=stage_cfg["stage"],
                        stage_label=f"{label}-EXP-{scale_mode}-R{grid}",
                        stage_slug=f"{stage_cfg['slug']}_exp_{scale_mode}_r{grid}",
                        selected_mode=mode,
                    )

                if mode in ("depth", "normal+depth"):
                    if not has_depth:
                        raise RuntimeError(
                            f"{label} depth mode requires source={depth_source}, "
                            "but native depth/alpha is missing"
                        )
                    if depth_weights is None:
                        raise RuntimeError(
                            f"{label} depth mode requested but diagnostic returned no weights"
                        )
                    visibility_weights.mul_(depth_weights)
                    print(f"[{label} EXP] depth=True, tau={depth_tau:.6f}", flush=True)
                else:
                    print(f"[{label} EXP] depth=False", flush=True)
            elif stage_cfg["stage"] != "off":
                print(f"[{stage_cfg['label']} EXP] skipped: visibility weights unavailable")

            z_global, z_proj = pipeline._run_mv_extractor(
                image_cond_model,
                views,
                visibility_weights=visibility_weights,
                geometry_stage="shape512",
            )
            del visibility_weights, depth_weights

            B = z_global.shape[0]
            actual_grid = int(image_cond_model.grid_resolution)
            coord_indices = coords.long()
            z_proj_grid = z_proj.reshape(B, actual_grid, actual_grid, actual_grid, -1)
            z_proj_sparse = z_proj_grid[
                coord_indices[:, 0], coord_indices[:, 1],
                coord_indices[:, 2], coord_indices[:, 3],
            ]
            z_proj_st = SparseTensor(feats=z_proj_sparse, coords=coords)
            projected = {
                "cond": {"global": z_global, "proj": z_proj_st},
                "neg_cond": {
                    "global": torch.zeros_like(z_global),
                    "proj": SparseTensor(
                        feats=torch.zeros_like(z_proj_sparse), coords=coords
                    ),
                },
            }
        finally:
            # Return the cached 512px conditioning model to its original grid
            # even on an exception, before the standard low-VRAM offload.
            if override:
                image_cond_model.grid_resolution = original_grid
                image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
                    grid_resolution=original_grid,
                    image_resolution=image_cond_model.proj_grid.image_resolution,
                ).to(device)
        if pipeline.low_vram:
            image_cond_model.cpu()
            pipeline._cleanup_cuda()
        return projected
