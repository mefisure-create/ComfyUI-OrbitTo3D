# ORBIT3D_PUBLIC_MV_ONLY_V1
"""
Public Pixal3D multi-view API wrappers for ComfyUI-OrbitTo3D.

This module deliberately does NOT modify the copied Trellis2/Pixal3D generation
implementation. It only:
  * restores the upstream modelname selector with Pixal3D as the safe default,
  * moves SS visibility controls to Sparse,
  * moves Shape512 geometry controls to Shape,
  * keeps ShapeHR geometry controls on Cascade,
  * validates that Orbit reconstruction nodes receive a Pixal3D MV pipeline,
  * exposes only the Pixal3D multi-view generator path (no legacy single-image or camera-config generator sockets);
  * uses native MOGE_GEOMETRY on MultiView;
  * standardizes public angle-string defaults as comma+space lists;
  * preserves all remaining Trellis2-compatible socket type names.
"""

from __future__ import annotations

from . import nodes as _trellis


_PUBLIC_API = "ORBIT3D_PUBLIC_MV_ONLY_V1"


def _base_required(cls):
    return dict(cls.INPUT_TYPES().get("required", {}))


def _base_optional(cls):
    return dict(cls.INPUT_TYPES().get("optional", {}))


def _require_pixal3d_mv(pipeline):
    if not getattr(pipeline, "isPixal3D", False):
        raise ValueError(
            "Orbit3D requires a TencentARC/Pixal3D TRELLIS2PIPELINE; "
            "the connected pipeline is not Pixal3D."
        )
    if not getattr(pipeline, "isPixal3DMV", False):
        raise ValueError(
            "Orbit3D requires Pixal3D multi-view weights "
            "(pipeline_mv.json / *_mv checkpoints). "
            "Load TencentARC/Pixal3D with pixal3d_multiview=True, "
            "or use Orbit3D - Load Model with TencentARC/Pixal3D."
        )


def _normalize_mode(value, label):
    value = str(value).strip().lower()
    if value not in ("off", "normal", "depth", "normal+depth"):
        raise ValueError(f"invalid {label}={value!r}")
    return value


def _normalize_source(value, label):
    value = str(value).strip().lower()
    if value not in ("marigold", "moge"):
        raise ValueError(f"invalid {label}={value!r}")
    return value


def _require_geometry_sources(views, mode, normal_source, depth_source, label):
    if views is None:
        raise ValueError(f"{label} requires pixal3d_mv_views")

    if mode in ("normal", "normal+depth"):
        key = (
            "geometry_confidence"
            if normal_source == "marigold"
            else "moge_geometry_confidence"
        )
        if views.get(key) is None:
            raise ValueError(
                f"{label} normal source {normal_source!r} requires "
                f"{key!r} in PIXAL3D_MV_VIEWS. "
                "Connect the corresponding Marigold/MoGe geometry to "
                "Orbit3D - MultiView Config."
            )

    if mode in ("depth", "normal+depth"):
        key = (
            "marigold_depth_native"
            if depth_source == "marigold"
            else "moge_depth_native"
        )
        if views.get(key) is None:
            raise ValueError(
                f"{label} depth source {depth_source!r} requires "
                f"{key!r} in PIXAL3D_MV_VIEWS. "
                "Connect the corresponding Marigold/MoGe geometry to "
                "Orbit3D - MultiView Config."
            )


class Orbit3DLoadModel(_trellis.Trellis2LoadModel):
    @classmethod
    def INPUT_TYPES(cls):
        required = _base_required(_trellis.Trellis2LoadModel)

        # Preserve the exact upstream model choices, but make the tested
        # TencentARC/Pixal3D MV path the safe/default Orbit choice.
        model_choices, model_options = required["modelname"]
        model_options = dict(model_options)
        model_options["default"] = "TencentARC/Pixal3D"
        required["modelname"] = (list(model_choices), model_options)

        keep = (
            "modelname",
            "backend",
            "device",
            "low_vram",
            "keep_models_loaded",
            "conv_backend",
            "sparse_backend",
        )
        return {"required": {name: required[name] for name in keep}}

    CATEGORY = "Orbit to 3D/Pixal3D"

    def process(
        self,
        modelname,
        backend,
        device,
        low_vram,
        keep_models_loaded,
        conv_backend,
        sparse_backend,
    ):
        # The current TencentARC/Pixal3D model is loaded with its multi-view
        # denoisers; other upstream choices
        # are allowed to load normally and will be rejected by Orbit generators
        # unless they expose a compatible isPixal3D/isPixal3DMV pipeline.
        return super().process(
            modelname=modelname,
            backend=backend,
            device=device,
            low_vram=low_vram,
            keep_models_loaded=keep_models_loaded,
            conv_backend=conv_backend,
            sparse_backend=sparse_backend,
            pixal3d_multiview=(modelname == "TencentARC/Pixal3D"),
        )


class Orbit3DMultiViewConfig(_trellis.Trellis2Pixal3DMultiViewConfig):
    @classmethod
    def INPUT_TYPES(cls):
        base = _trellis.Trellis2Pixal3DMultiViewConfig.INPUT_TYPES()
        required = dict(base["required"])
        optional = dict(base.get("optional", {}))

        for name in (
            "marigold_shape512_mode",
            "shape512_normals_source",
            "shape512_depth_source",
            "ss_visibility_mode",
            "ss_occlusion_floor",
            "ss_occlusion_margin_voxels",
            "ss_occlusion_tau_voxels",
            "ss_anchor_band_voxels",
            "ss_anchored_floor",
        ):
            required.pop(name, None)

        # Keep the angle-entry convention aligned with VGGT: plain
        # comma-separated values with a space after each comma.
        required["azimuths"] = ("STRING", {"default": "0, 90, 180, 270"})
        required["elevations"] = ("STRING", {"default": "0, 0, 0, 0"})

        # Base runtime core already exposes standard MOGE_GEOMETRY.
        return {"required": required, "optional": optional}

    CATEGORY = "Orbit to 3D/Pixal3D"

    def process(
        self,
        images,
        azimuths,
        elevations,
        fov,
        fov_unit,
        mesh_scale,
        distance,
        remove_background,
        invert_mask,
        framing="auto",
        masks=None,
        normal_maps=None,
        depth_maps=None,
        moge_geometry=None,
        cleanup_before_pixal3d=True,
    ):
        # Geometry maps are attached here through the standard MOGE_GEOMETRY
        # packet. Sparse/Shape consumption policy stays unchanged.
        return super().process(
            images=images,
            azimuths=azimuths,
            elevations=elevations,
            fov=fov,
            fov_unit=fov_unit,
            mesh_scale=mesh_scale,
            distance=distance,
            remove_background=remove_background,
            invert_mask=invert_mask,
            framing=framing,
            marigold_shape512_mode="off",
            shape512_normals_source="Marigold",
            shape512_depth_source="Marigold",
            ss_visibility_mode="all_views",
            ss_occlusion_floor=1.0,
            ss_occlusion_margin_voxels=1.50,
            ss_occlusion_tau_voxels=1.50,
            ss_anchor_band_voxels=1.50,
            ss_anchored_floor=1.0,
            masks=masks,
            normal_maps=normal_maps,
            depth_maps=depth_maps,
            moge_geometry=moge_geometry,
            cleanup_before_pixal3d=cleanup_before_pixal3d,
        )


class Orbit3DSparseGenerator(_trellis.Trellis2SparseGenerator):
    @classmethod
    def INPUT_TYPES(cls):
        base = _trellis.Trellis2SparseGenerator.INPUT_TYPES()
        required = dict(base["required"])
        optional = {
            "pixal3d_mv_views": base.get("optional", {})["pixal3d_mv_views"],
        }
        required.update(
            {
                "ss_visibility_mode": (
                    ["all_views", "soft_occlusion", "anchored_soft_occlusion"],
                    {"default": "anchored_soft_occlusion"},
                ),
                "ss_occlusion_floor": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.05, "max": 1.00, "step": 0.05},
                ),
                "ss_occlusion_margin_voxels": (
                    "FLOAT",
                    {"default": 1.50, "min": 0.00, "max": 6.00, "step": 0.25},
                ),
                "ss_occlusion_tau_voxels": (
                    "FLOAT",
                    {"default": 1.50, "min": 0.25, "max": 8.00, "step": 0.25},
                ),
                "ss_anchor_band_voxels": (
                    "FLOAT",
                    {"default": 1.50, "min": 0.25, "max": 4.00, "step": 0.25},
                ),
                "ss_anchored_floor": (
                    "FLOAT",
                    {"default": 0.50, "min": 0.10, "max": 1.00, "step": 0.05},
                ),
            }
        )
        return {"required": required, "optional": optional}

    CATEGORY = "Orbit to 3D/Pixal3D"

    def process(
        self,
        pipeline,
        seed,
        sparse_structure_steps,
        sparse_structure_guidance_strength,
        sparse_structure_guidance_rescale,
        sparse_structure_rescale_t,
        sparse_structure_sampler,
        sparse_structure_resolution,
        sparse_structure_guidance_interval_start,
        sparse_structure_guidance_interval_end,
        fill_holes,
        hole_iterations,
        verbose,
        dino_lock,
        dino_substeps,
        hole_fill_algorithm,
        dino_foundation_cap,
        keep_only_shell,
        ss_visibility_mode="anchored_soft_occlusion",
        ss_occlusion_floor=0.35,
        ss_occlusion_margin_voxels=1.50,
        ss_occlusion_tau_voxels=1.50,
        ss_anchor_band_voxels=1.50,
        ss_anchored_floor=0.50,
        pixal3d_mv_views=None,
    ):
        _require_pixal3d_mv(pipeline)
        if pixal3d_mv_views is None:
            raise ValueError("Orbit3D Sparse Generator requires pixal3d_mv_views")

        mode = str(ss_visibility_mode).strip().lower()
        if mode not in ("all_views", "soft_occlusion", "anchored_soft_occlusion"):
            raise ValueError(f"invalid ss_visibility_mode={mode!r}")

        floor = float(ss_occlusion_floor)
        margin = float(ss_occlusion_margin_voxels)
        tau = float(ss_occlusion_tau_voxels)
        anchor_band = float(ss_anchor_band_voxels)
        anchored_floor = float(ss_anchored_floor)

        if not (0.0 < floor <= 1.0):
            raise ValueError("ss_occlusion_floor must be in (0,1]")
        if margin < 0.0:
            raise ValueError("ss_occlusion_margin_voxels must be >= 0")
        if tau <= 0.0:
            raise ValueError("ss_occlusion_tau_voxels must be > 0")
        if anchor_band <= 0.0:
            raise ValueError("ss_anchor_band_voxels must be > 0")
        if not (0.0 < anchored_floor <= 1.0):
            raise ValueError("ss_anchored_floor must be in (0,1]")

        if mode != "all_views" and pixal3d_mv_views.get("ss_geometry") is None:
            raise ValueError(
                f"ss_visibility_mode={mode!r} requires MoGe geometry in "
                "PIXAL3D_MV_VIEWS"
            )

        views = dict(pixal3d_mv_views)
        views["ss_visibility_mode"] = mode
        views["ss_soft_occlusion_params"] = {
            "floor": floor,
            "margin_voxels": margin,
            "tau_voxels": tau,
            "anchor_band_voxels": anchor_band,
            "anchored_floor": anchored_floor,
        }

        print(
            f"[Orbit3D Sparse config] mode={mode}; "
            f"soft_floor={floor:.3f}, anchored_floor={anchored_floor:.3f}, "
            f"anchor_band={anchor_band:.2f} vox, margin={margin:.2f} vox, "
            f"tau={tau:.2f} vox"
        )

        return super().process(
            pipeline=pipeline,
            seed=seed,
            sparse_structure_steps=sparse_structure_steps,
            sparse_structure_guidance_strength=sparse_structure_guidance_strength,
            sparse_structure_guidance_rescale=sparse_structure_guidance_rescale,
            sparse_structure_rescale_t=sparse_structure_rescale_t,
            sparse_structure_sampler=sparse_structure_sampler,
            sparse_structure_resolution=sparse_structure_resolution,
            sparse_structure_guidance_interval_start=sparse_structure_guidance_interval_start,
            sparse_structure_guidance_interval_end=sparse_structure_guidance_interval_end,
            fill_holes=fill_holes,
            hole_iterations=hole_iterations,
            verbose=verbose,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            hole_fill_algorithm=hole_fill_algorithm,
            dino_foundation_cap=dino_foundation_cap,
            keep_only_shell=keep_only_shell,
            pixal3d_mv_views=views,
        )


class Orbit3DShapeGenerator(_trellis.Trellis2ShapeGenerator):
    @classmethod
    def INPUT_TYPES(cls):
        base = _trellis.Trellis2ShapeGenerator.INPUT_TYPES()
        required = dict(base["required"])
        optional = {
            "pixal3d_mv_views": base.get("optional", {})["pixal3d_mv_views"],
        }
        required.update(
            {
                "shape512_mode": (
                    ["off", "normal", "depth", "normal+depth"],
                    {"default": "normal"},
                ),
                "shape512_normals_source": (
                    ["Marigold", "MoGe"],
                    {"default": "Marigold"},
                ),
                "shape512_depth_source": (
                    ["Marigold", "MoGe"],
                    {"default": "Marigold"},
                ),
            }
        )
        return {"required": required, "optional": optional}

    CATEGORY = "Orbit to 3D/Pixal3D"

    def process(
        self,
        pipeline,
        coords,
        resolution,
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
        shape512_mode="normal",
        shape512_normals_source="Marigold",
        shape512_depth_source="Marigold",
        pixal3d_mv_views=None,
    ):
        _require_pixal3d_mv(pipeline)
        if int(resolution) != 512:
            raise ValueError(
                "Orbit3D Shape Generator is the Shape512 stage and requires resolution=512"
            )

        mode = _normalize_mode(shape512_mode, "shape512_mode")
        normal_source = _normalize_source(
            shape512_normals_source, "shape512_normals_source"
        )
        depth_source = _normalize_source(
            shape512_depth_source, "shape512_depth_source"
        )
        _require_geometry_sources(
            pixal3d_mv_views, mode, normal_source, depth_source, "Shape512"
        )

        views = dict(pixal3d_mv_views)
        # Keep the internal legacy key untouched: the generation pipeline already
        # consumes it and changing it here would alter tested math.
        views["marigold_shape512_mode"] = mode
        views["shape512_normals_source"] = normal_source
        views["shape512_depth_source"] = depth_source

        print(
            f"[Orbit3D Shape512 config] mode={mode}; "
            f"normal={normal_source}; depth={depth_source}"
        )

        return super().process(
            pipeline=pipeline,
            coords=coords,
            resolution=resolution,
            shape_steps=shape_steps,
            shape_guidance_strength=shape_guidance_strength,
            shape_guidance_rescale=shape_guidance_rescale,
            shape_rescale_t=shape_rescale_t,
            shape_sampler=shape_sampler,
            shape_guidance_interval_start=shape_guidance_interval_start,
            shape_guidance_interval_end=shape_guidance_interval_end,
            verbose=verbose,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
            pixal3d_mv_views=views,
        )


class Orbit3DShapeCascadeGenerator(_trellis.Trellis2ShapeCascadeGenerator):
    @classmethod
    def INPUT_TYPES(cls):
        base = _trellis.Trellis2ShapeCascadeGenerator.INPUT_TYPES()
        required = dict(base["required"])
        optional = {
            "pixal3d_mv_views": base.get("optional", {})["pixal3d_mv_views"],
        }
        return {"required": required, "optional": optional}

    CATEGORY = "Orbit to 3D/Pixal3D"

    def process(
        self,
        pipeline,
        shape_slat,
        from_resolution,
        to_resolution,
        sparse_structure_resolution,
        max_num_tokens,
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
        shape_hr_mode="off",
        shape_hr_normals_source="Marigold",
        shape_hr_depth_source="Marigold",
        pixal3d_mv_views=None,
    ):
        _require_pixal3d_mv(pipeline)
        mode = _normalize_mode(shape_hr_mode, "shape_hr_mode")
        normal_source = _normalize_source(
            shape_hr_normals_source, "shape_hr_normals_source"
        )
        depth_source = _normalize_source(
            shape_hr_depth_source, "shape_hr_depth_source"
        )
        _require_geometry_sources(
            pixal3d_mv_views, mode, normal_source, depth_source, "ShapeHR"
        )

        return super().process(
            pipeline=pipeline,
            shape_slat=shape_slat,
            from_resolution=from_resolution,
            to_resolution=to_resolution,
            sparse_structure_resolution=sparse_structure_resolution,
            max_num_tokens=max_num_tokens,
            shape_steps=shape_steps,
            shape_guidance_strength=shape_guidance_strength,
            shape_guidance_rescale=shape_guidance_rescale,
            shape_rescale_t=shape_rescale_t,
            shape_sampler=shape_sampler,
            shape_guidance_interval_start=shape_guidance_interval_start,
            shape_guidance_interval_end=shape_guidance_interval_end,
            verbose=verbose,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
            shape_hr_mode=mode,
            shape_hr_normals_source=normal_source,
            shape_hr_depth_source=depth_source,
            pixal3d_mv_views=pixal3d_mv_views,
        )
