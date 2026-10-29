# SS_ANCHORED_SOFT_OCCLUSION_V2
from typing import *
import torch
import torch.nn as nn
import numpy as np
from .base import Pipeline
from . import samplers
from ..modules.sparse import SparseTensor
from ..representations import Mesh, MeshWithVoxel

from .. import models

import gc
import os
import folder_paths
import trimesh

script_directory = os.path.dirname(os.path.abspath(__file__))


def build_pixal3d_mv_image_cond_model(config: dict):
    from ..trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjMultiViewFeatureExtractor
    model = DinoV3ProjMultiViewFeatureExtractor(**config)
    model.eval()
    return model



class Trellis2ImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    # model_names_to_load = [
        # 'sparse_structure_flow_model',
        # 'sparse_structure_decoder',
        # 'shape_slat_flow_model_512',
        # 'shape_slat_flow_model_1024',
        # 'shape_slat_decoder',
    # ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self._low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self._device = 'cpu'
        
        self.PIXAL3D_IMAGE_COND_CONFIGS = {
            "ss": {
                "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "image_size": 512,
                "grid_resolution": 16,
            },
            "shape_512": {
                "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "image_size": 512,
                "grid_resolution": 32,
                "use_naf_upsample": True,
                "naf_target_size": 512,
            },
            "shape_1024": {
                "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "image_size": 1024,
                "grid_resolution": 64,
                "use_naf_upsample": True,
                "naf_target_size": 512,
            },
        }

        # Multi-view DINOv3 conditioning configs for the retained mesh-only runtime.
        self.PIXAL3D_MV_IMAGE_COND_CONFIGS = {
            key: {
                **cfg,
                "multiview_fusion": (
                    "weighted_average"
                    if key in ("ss", "shape_512", "shape_1024")
                    else "average"
                ),
            }
            for key, cfg in self.PIXAL3D_IMAGE_COND_CONFIGS.items()
        }
        self.PIXAL3D_MV_IMAGE_COND_CONFIGS["shape_512"].update({
            "detail_tiles": True,
            "detail_tile_fraction": 0.41,
            "detail_full_strength": 0.35,
            "detail_tile_strength": 1.0,
            "detail_feather": 0.20,
        })
        self.PIXAL3D_MV_IMAGE_COND_CONFIGS["shape_1024"].update({
            "detail_tiles": True,
            "detail_tile_fraction": 0.40,
            "detail_full_strength": 0.35,
            "detail_tile_strength": 1.0,
            "detail_feather": 0.20,
        })

    @property
    def low_vram(self) -> bool:
        return self._low_vram

    @low_vram.setter
    def low_vram(self, value: bool):
        self._low_vram = value
        for m in self.models.values():
            if hasattr(m, 'low_vram'):
                m.low_vram = value

    def _cond_to(self, cond: dict, device: torch.device) -> dict:
        # Move only tensors; keep other items unchanged
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cond.items()}

    def _cond_cpu(self, cond: dict) -> dict:
        return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in cond.items()}

    def _cleanup_cuda(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json", keep_models_loaded = True, use_fp8 = False, isPixal3D = False, isPixal3DMV = False) -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        if use_fp8:
            config_file = "pipeline_fp8.json"
        elif isPixal3DMV:
            # The multi-view denoisers live next to the single-view ones under
            # ckpts/*_mv in the same repo; this config file is what points at them.
            config_file = "pipeline_mv.json"
            
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']

        pipeline.low_vram = args.get('low_vram', True)
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        pipeline._device = 'cpu'
        pipeline.path = path
        pipeline.keep_models_loaded = keep_models_loaded
        pipeline.last_processing = ''
        pipeline.use_fp8 = use_fp8
        pipeline.isPixal3D = isPixal3D
        pipeline.isPixal3DMV = isPixal3DMV

        if not isPixal3D:
            pipeline._pretrained_args['models']['sparse_structure_decoder'] = os.path.join(folder_paths.models_dir,"microsoft","TRELLIS-image-large","ckpts","ss_dec_conv3d_16l8_fp16")
            
        facebook_model_path = os.path.join(folder_paths.models_dir,"facebook","dinov3-vitl16-pretrain-lvd1689m")
        for _cfg in pipeline.PIXAL3D_MV_IMAGE_COND_CONFIGS.values():
            _cfg["model_name"] = facebook_model_path

        try:
            from mmgp import safetensors2 as _mmgp_st2
            if 'C64' not in _mmgp_st2._map_to_dtype:
                _mmgp_st2._map_to_dtype['C64'] = torch.complex64
                _mmgp_st2._map_to_dtype['C128'] = torch.complex128
                print("[Pixal3D] Patched mmgp.safetensors2._map_to_dtype with C64/C128")
        except (ImportError, AttributeError):
            print('mmgp not installed')
            # mmgp not installed (comfy_env worker / non-Desktop ComfyUI) — no patch needed.
            pass

        return pipeline
            

        
        
            
        
            
        

        

    # ---- Pixal3D multi-view image cond models (DinoV3ProjMultiViewFeatureExtractor) ----

    def _load_pixal3d_mv_image_cond(self, stage: str):
        attr = f'pixal3d_mv_image_cond_{stage}'
        model = getattr(self, attr, None)
        if model is not None:
            return model

        print(f'Loading Pixal3D MultiView Image Cond {stage} Model ...')
        model = build_pixal3d_mv_image_cond_model(self.PIXAL3D_MV_IMAGE_COND_CONFIGS[stage])
        setattr(self, attr, model)
        return model

    def _unload_pixal3d_mv_image_cond(self, stage: str):
        attr = f'pixal3d_mv_image_cond_{stage}'
        if getattr(self, attr, None) is not None:
            setattr(self, attr, None)
            self._cleanup_cuda()

    def load_pixal3d_mv_image_cond_ss(self):
        return self._load_pixal3d_mv_image_cond('ss')

    def unload_pixal3d_mv_image_cond_ss(self):
        self._unload_pixal3d_mv_image_cond('ss')

    def load_pixal3d_mv_image_cond_shape_512(self):
        return self._load_pixal3d_mv_image_cond('shape_512')

    def unload_pixal3d_mv_image_cond_shape_512(self):
        self._unload_pixal3d_mv_image_cond('shape_512')

    def load_pixal3d_mv_image_cond_shape_1024(self):
        return self._load_pixal3d_mv_image_cond('shape_1024')

    def unload_pixal3d_mv_image_cond_shape_1024(self):
        self._unload_pixal3d_mv_image_cond('shape_1024')


        
    def load_sparse_structure_model(self):        
        if self.models['sparse_structure_flow_model'] is None:
            print('Loading Sparse Structure model ...')
            self.models['sparse_structure_flow_model'] = models.from_pretrained(f"{self.path}/{self._pretrained_args['models']['sparse_structure_flow_model']}")
            self.models['sparse_structure_flow_model'].eval()
            self.models['sparse_structure_flow_model'].to(self._device)
        
        if self.models['sparse_structure_decoder'] is None:  
            if self.isPixal3D:
                self.models['sparse_structure_decoder'] = models.from_pretrained(f"{self.path}/{self._pretrained_args['models']['sparse_structure_decoder']}")
            else:
                self.models['sparse_structure_decoder'] = models.from_pretrained(self._pretrained_args['models']['sparse_structure_decoder'])
            self.models['sparse_structure_decoder'].eval()        
            self.models['sparse_structure_decoder'].to(self._device)
            if hasattr(self.models['sparse_structure_decoder'], 'low_vram'):
                self.models['sparse_structure_decoder'].low_vram = self.low_vram
                
                
                

    
    def unload_sparse_structure_model(self):
        if self.models['sparse_structure_flow_model']:
            del self.models['sparse_structure_flow_model']
            self.models['sparse_structure_flow_model'] = None            
            
        if self.models['sparse_structure_decoder']:
            del self.models['sparse_structure_decoder']
            self.models['sparse_structure_decoder'] = None
        
        self._cleanup_cuda()
            
            
            
    def load_shape_slat_flow_model_512(self):        
        if self.models['shape_slat_flow_model_512'] is None:
            print('Loading Shape Slat Flow 512 model ...')
            self.models['shape_slat_flow_model_512'] = models.from_pretrained(f"{self.path}/{self._pretrained_args['models']['shape_slat_flow_model_512']}")
            self.models['shape_slat_flow_model_512'].eval()
            self.models['shape_slat_flow_model_512'].to(self._device)
            
    def unload_shape_slat_flow_model_512(self):
        if self.models['shape_slat_flow_model_512'] is not None:
            del self.models['shape_slat_flow_model_512']
            self.models['shape_slat_flow_model_512'] = None
            self._cleanup_cuda()
            



            
    def load_shape_slat_decoder(self):        
        if self.models['shape_slat_decoder'] is None:
            print('Loading Shape Slat decoder model ...')
            self.models['shape_slat_decoder'] = models.from_pretrained(f"{self.path}/{self._pretrained_args['models']['shape_slat_decoder']}")
            self.models['shape_slat_decoder'].eval()
            self.models['shape_slat_decoder'].to(self._device)
            if hasattr(self.models['shape_slat_decoder'], 'low_vram'):
                self.models['shape_slat_decoder'].low_vram = self.low_vram

    def unload_shape_slat_decoder(self):
        if self.models['shape_slat_decoder'] is not None:
            del self.models['shape_slat_decoder']
            self.models['shape_slat_decoder'] = None
            self._cleanup_cuda()

    def load_shape_slat_flow_model_1024(self):        
        if self.models['shape_slat_flow_model_1024'] is None:
            print('Loading Shape Slat Flow 1024 model ...')
            self.models['shape_slat_flow_model_1024'] = models.from_pretrained(f"{self.path}/{self._pretrained_args['models']['shape_slat_flow_model_1024']}")
            self.models['shape_slat_flow_model_1024'].eval()
            self.models['shape_slat_flow_model_1024'].to(self._device)           

    def unload_shape_slat_flow_model_1024(self):
        if self.models['shape_slat_flow_model_1024'] is not None:
            del self.models['shape_slat_flow_model_1024']
            self.models['shape_slat_flow_model_1024'] = None
            self._cleanup_cuda()






    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)

    # =========================================================================
    # Proj mode condition building
    # =========================================================================
    

        
    # ------------------------------------------------------------------
    # Pixal3D multi-view proj conditioning
    #
    # Retained Orbit cascade is SS -> Shape 512 -> Shape 1024. Instead of one
    # image at the canonical front view it takes V views with explicit c2w
    # matrices and lets DinoV3ProjMultiViewFeatureExtractor project all of them
    # into the shared 3D grid, averaging the per-view features.
    #
    # Because the fusion is "average", the fused z_proj has exactly the
    # single-view shape [B, R^3, C], so every per-block proj_linear / proj
    # cross-attn weight of the *_mv denoisers stays compatible. View 0 is the
    # main view and its calc_mat_0 == F by construction.
    #
    # `views` is the bundle built by trellis2.utils.mv_camera.build_views().
    # ------------------------------------------------------------------

    @staticmethod
    def _mv_image_for(views: dict, image_cond_model: nn.Module) -> torch.Tensor:
        """
        Pick the pre-decoded [1, V, 3, H, W] batch matching this stage input size.

        Unlike the single-view extractor (which takes PIL and resizes internally), the
        multi-view one takes tensors, so the caller has to supply the right resolution:
        the SS / shape-512 stages run at 512 and the 1024 stages at 1024.
        """
        size = int(image_cond_model.image_size)
        if size not in views['images']:
            raise KeyError(
                f"No view images at resolution {size}; got {sorted(views['images'])}. "
                f"Build them with mv_camera.build_views(image_sizes=...).")
        return views['images'][size]
    @torch.no_grad()
    def _build_sparse_mv_visibility(
        self,
        image_cond_model,
        views: dict,
        coords: torch.Tensor,
        depth_tolerance_voxels: float = 1.5,
    ):
        from ..trainers.flow_matching.mixins.image_conditioned_proj import (
            compute_relative_calc_mat,
            project_points_to_image_batch,
        )

        device = self.device
        R = int(image_cond_model.grid_resolution)
        R3 = R ** 3

        transform_matrix = views["transform_matrix"].to(
            device=device, dtype=torch.float32
        )
        camera_angle = views["camera_angle_x"].to(
            device=device, dtype=torch.float32
        )
        distance = views["camera_distance"].to(
            device=device, dtype=torch.float32
        )
        mesh_scale = torch.as_tensor(
            views["mesh_scale"],
            device=device,
            dtype=torch.float32,
        ).reshape(-1)

        B, V = transform_matrix.shape[:2]

        calc_mat = compute_relative_calc_mat(
            transform_matrix,
            distance,
            image_cond_model.proj_grid.front_view_transform_matrix,
        )

    # Dense weight tensor is tiny compared with projected DINO features.
    # Default = normal averaging. Sparse occupied tokens are overwritten below.
        weights = torch.ones(
            (B, V, R3, 1),
            device=device,
            dtype=torch.float16,
        )

        base_grid = image_cond_model.proj_grid.grid_points.to(
            device=device,
            dtype=torch.float32,
        )

        vis_res = R

        coords_long = coords.long()

        for b in range(B):
            sel = coords_long[:, 0] == b
            c = coords_long[sel]

            if c.numel() == 0:
                continue

            x = c[:, 1]
            y = c[:, 2]
            z = c[:, 3]

            flat = x * (R * R) + y * R + z

            # Exact 3D positions used by ProjGrid.
            pts3d = base_grid[flat]
            pts3d = pts3d / mesh_scale[b] / 2.0
            pts3d = pts3d.unsqueeze(0)

            visible = torch.zeros(
                (V, flat.numel()),
                device=device,
                dtype=torch.bool,
            )

            # One voxel-and-a-half front-surface band.
            voxel_depth = 1.0 / max(R - 1, 1) / float(mesh_scale[b].item())
            depth_eps = depth_tolerance_voxels * voxel_depth

            for v in range(V):
                points_2d, depth, valid = project_points_to_image_batch(
                    pts3d,
                    calc_mat[b:b + 1, v],
                    camera_angle[b:b + 1, v],
                    vis_res,
                )

                p = points_2d[0]
                d = depth[0].float()
                valid = valid[0]

                px = p[:, 0].round().long().clamp_(0, vis_res - 1)
                py = p[:, 1].round().long().clamp_(0, vis_res - 1)
                pixel_id = py * vis_res + px

                zbuf = torch.full(
                    (vis_res * vis_res,),
                    float("inf"),
                    device=device,
                    dtype=torch.float32,
                )

                if valid.any():
                    zbuf.scatter_reduce_(
                        0,
                        pixel_id[valid],
                        d[valid],
                        reduce="amin",
                        include_self=True,
                    )

                nearest = zbuf[pixel_id]

                visible[v] = valid & (d <= nearest + depth_eps)

                del zbuf, points_2d, depth

            # Interior / ambiguous token: preserve stock behaviour rather than
            # starving it of conditioning completely.
            unseen = ~visible.any(dim=0)
            num_unseen = unseen.sum().item()

            if unseen.any():
                visible[:, unseen] = True

            # Replace weights only for occupied sparse tokens.
            weights[b, :, flat, 0] = visible.to(weights.dtype)

            counts = visible.sum(dim=0)

            # PIXAL_MV_ARBITRARY_V_V1
            count_stats = {
                n: int((counts == n).sum().item())
                for n in range(1, V + 1)
            }

            print(
                f"[Pixal3D visibility] R={R} batch={b}: "
                f"V={V}, tokens={flat.numel()}, "
                f"unseen_fallback={num_unseen}, "
                f"view_counts={count_stats}"
            )

        return weights



    # SS_ANCHORED_SOFT_OCCLUSION_V2
    @torch.no_grad()
    def _build_ss_soft_occlusion_weights(
        self,
        image_cond_model,
        views: dict,
        state: dict,
        mode: str = "soft_occlusion",
        floor: float = 0.35,
        anchored_floor: float = 0.50,
        anchor_band_voxels: float = 1.50,
        margin_voxels: float = 1.50,
        tau_voxels: float = 1.50,
    ):
        """
        SS-only MoGe view re-ranking.

        soft_occlusion:
            Original V1 behaviour. Any usable candidate that lies behind
            the visible MoGe surface may downweight that view.

        anchored_soft_occlusion:
            First require that at least one view independently confirms
            the candidate as a plausible visible surface:

                abs(candidate_depth - surface_depth)
                    <= anchor_band_voxels

            Only such anchored cells are eligible for occlusion
            re-ranking in the other views.

        Neither mode decides occupancy, adds voxels, removes voxels,
        or modifies S4. S4 remains strictly post-SS.
        """

        import torch.nn.functional as F

        from ..trainers.flow_matching.mixins.image_conditioned_proj import (
            project_points_to_image_batch,
        )

        if state is None:
            return None

        mode = str(mode).strip().lower()

        if mode not in (
            "soft_occlusion",
            "anchored_soft_occlusion",
        ):
            raise ValueError(
                f"invalid SS occlusion mode {mode!r}"
            )

        required = (
            "points",
            "mask",
            "alphas",
            "calc_mat",
            "camera_angle_x",
            "mesh_scale",
            "view_names",
            "cond_res",
        )

        missing = [
            k for k in required
            if state.get(k, None) is None
        ]

        if missing:
            print(
                f"[SS-SoftOcc] unavailable; "
                f"missing state fields={missing}; "
                "falling back to all-view average"
            )
            return None

        floor = float(floor)
        anchored_floor = float(anchored_floor)
        anchor_band_voxels = float(anchor_band_voxels)
        margin_voxels = float(margin_voxels)
        tau_voxels = float(tau_voxels)

        points = state["points"]
        moge_valid = state["mask"]
        alphas = state["alphas"]

        calc_mat_cpu = state["calc_mat"]
        camera_angle_cpu = state["camera_angle_x"]
        mesh_scale_cpu = state["mesh_scale"]

        view_names = list(state["view_names"])
        cond_res = int(state["cond_res"])

        B, V = points.shape[:2]
        H, W = points.shape[2:4]

        if len(view_names) != V:
            raise ValueError(
                f"[SS-SoftOcc] V={V} "
                f"but view_names={len(view_names)}"
            )

        device = self.device

        calc_mat = calc_mat_cpu.to(
            device=device,
            dtype=torch.float32,
        )

        camera_angle = camera_angle_cpu.to(
            device=device,
            dtype=torch.float32,
        )

        mesh_scale = torch.as_tensor(
            mesh_scale_cpu,
            device=device,
            dtype=torch.float32,
        ).reshape(-1)

        if mesh_scale.numel() == 1 and B > 1:
            mesh_scale = mesh_scale.expand(B)

        # Same foreground definition as S4:
        # Pixal alpha AND MoGe validity.
        alpha_native = F.interpolate(
            alphas.reshape(
                alphas.shape[0] * alphas.shape[1],
                1,
                alphas.shape[-2],
                alphas.shape[-1],
            ).float(),
            size=(H, W),
            mode="nearest",
        ).reshape(B, V, H, W)

        foreground = alpha_native > 0.5

        R = int(image_cond_model.grid_resolution)
        R3 = R ** 3

        base_grid = (
            image_cond_model
            .proj_grid
            .grid_points
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        weights = torch.ones(
            (B, V, R3, 1),
            device=device,
            dtype=torch.float16,
        )

        stride = max(
            1,
            int(max(H, W) // 512),
        )

        origin_world_cpu = torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            dtype=torch.float32,
        )

        output_dir = os.path.join(
            folder_paths.get_output_directory(),
            "debug",
            "pixal3d",
        )

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        stamp = __import__("time").strftime(
            "%Y%m%d_%H%M%S"
        )

        for b in range(B):

            pts_world = (
                base_grid
                / mesh_scale[b]
                / 2.0
            )

            pts_h = torch.cat(
                [
                    pts_world,
                    torch.ones(
                        (R3, 1),
                        device=device,
                        dtype=torch.float32,
                    ),
                ],
                dim=-1,
            )

            voxel_size = (
                1.0
                / max(R - 1, 1)
                / float(mesh_scale[b].item())
            )

            per_view_scale = []

            # ----------------------------------------------------
            # Same independent radial gauge as S4.
            # ----------------------------------------------------

            for v in range(V):

                p = points[
                    b,
                    v,
                    ::stride,
                    ::stride,
                ].float()

                vm = moge_valid[
                    b,
                    v,
                    ::stride,
                    ::stride,
                ].bool()

                fg = foreground[
                    b,
                    v,
                    ::stride,
                    ::stride,
                ]

                z = p[..., 2]

                valid = (
                    fg
                    & vm
                    & torch.isfinite(p).all(dim=-1)
                    & torch.isfinite(z)
                    & (z > 0)
                )

                zv = z[valid]

                if zv.numel() < 64:
                    raise RuntimeError(
                        f"[SS-SoftOcc] b={b} v={v} "
                        f"{view_names[v]}: "
                        "too few valid foreground "
                        "MoGe samples"
                    )

                z01 = torch.quantile(zv, 0.01)
                z99 = torch.quantile(zv, 0.99)

                moge_center_z = (
                    z01 + z99
                ) * 0.5

                w2c_cpu = torch.linalg.inv(
                    calc_mat_cpu[
                        b,
                        v,
                    ].float()
                )

                origin_cam = (
                    origin_world_cpu
                    @ w2c_cpu.T
                )[:3]

                target_z = -origin_cam[2]

                scale_v = float(
                    (
                        target_z
                        / moge_center_z
                    ).item()
                )

                per_view_scale.append(
                    scale_v
                )

                print(
                    f"[SS-SoftOcc scale] "
                    f"b={b} v={v} "
                    f"{view_names[v]}: "
                    f"moge_center_z="
                    f"{float(moge_center_z):.6f}, "
                    f"target_z="
                    f"{float(target_z):.6f}, "
                    f"scale={scale_v:.8f}"
                )

            # ----------------------------------------------------
            # PASS 1.
            #
            # Measure signed candidate-vs-surface depth residual
            # independently in every view.
            #
            # positive delta:
            #     candidate is BEHIND visible surface
            #
            # ~zero delta:
            #     candidate agrees with visible surface
            #
            # negative delta:
            #     candidate lies in FRONT of visible surface
            # ----------------------------------------------------

            delta_all = torch.full(
                (V, R3),
                float("nan"),
                device=device,
                dtype=torch.float32,
            )

            usable_all = torch.zeros(
                (V, R3),
                device=device,
                dtype=torch.bool,
            )

            anchor_support = torch.zeros(
                (V, R3),
                device=device,
                dtype=torch.bool,
            )

            for v in range(V):

                (
                    points_2d,
                    _,
                    proj_valid,
                ) = project_points_to_image_batch(
                    pts_world.unsqueeze(0),
                    calc_mat[
                        b:b + 1,
                        v,
                    ],
                    camera_angle[
                        b:b + 1,
                        v,
                    ],
                    cond_res,
                )

                pix = points_2d[0]
                proj_valid = proj_valid[0]

                gx = (
                    (pix[:, 0] + 0.5)
                    * (W / float(cond_res))
                    - 0.5
                )

                gy = (
                    (pix[:, 1] + 0.5)
                    * (H / float(cond_res))
                    - 0.5
                )

                ix = (
                    gx.round()
                    .long()
                    .clamp(0, W - 1)
                )

                iy = (
                    gy.round()
                    .long()
                    .clamp(0, H - 1)
                )

                ix_cpu = ix.detach().cpu()
                iy_cpu = iy.detach().cpu()

                q_cpu = points[
                    b,
                    v,
                ][iy_cpu, ix_cpu]

                vm_cpu = moge_valid[
                    b,
                    v,
                ][iy_cpu, ix_cpu]

                fg_cpu = foreground[
                    b,
                    v,
                ][iy_cpu, ix_cpu]

                qz = (
                    q_cpu[:, 2]
                    .to(
                        device=device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                    * float(
                        per_view_scale[v]
                    )
                )

                vm = vm_cpu.to(
                    device=device,
                    dtype=torch.bool,
                    non_blocking=True,
                )

                fg = fg_cpu.to(
                    device=device,
                    dtype=torch.bool,
                    non_blocking=True,
                )

                with torch.amp.autocast(
                    "cuda",
                    enabled=False,
                ):
                    w2c = torch.linalg.inv(
                        calc_mat[
                            b,
                            v,
                        ].float()
                    )

                p_blender = (
                    pts_h
                    @ w2c.T
                )[:, :3]

                candidate_z = (
                    -p_blender[:, 2]
                )

                usable = (
                    proj_valid
                    & vm
                    & fg
                    & torch.isfinite(qz)
                    & torch.isfinite(
                        candidate_z
                    )
                    & (qz > 0)
                    & (candidate_z > 0)
                )

                delta_vox = (
                    candidate_z - qz
                ) / max(
                    voxel_size,
                    1e-12,
                )

                delta_all[
                    v,
                    usable,
                ] = delta_vox[
                    usable
                ]

                usable_all[
                    v
                ] = usable

                anchor_support[
                    v
                ] = (
                    usable
                    & (
                        delta_vox.abs()
                        <= anchor_band_voxels
                    )
                )

                del (
                    points_2d,
                    pix,
                    ix,
                    iy,
                    ix_cpu,
                    iy_cpu,
                    q_cpu,
                    vm_cpu,
                    fg_cpu,
                    qz,
                    vm,
                    fg,
                    p_blender,
                    candidate_z,
                    usable,
                    delta_vox,
                )

            anchor_count = (
                anchor_support
                .sum(dim=0)
            )

            anchored = (
                anchor_count > 0
            )

            anchor_hist = {
                int(n):
                    int(
                        (
                            anchor_count == n
                        ).sum().item()
                    )
                for n in range(V + 1)
                if int(
                    (
                        anchor_count == n
                    ).sum().item()
                ) > 0
            }

            # ----------------------------------------------------
            # PASS 2.
            #
            # V1 soft_occlusion:
            #     every usable candidate is eligible.
            #
            # V2 anchored_soft_occlusion:
            #     only candidates with >=1 positive surface anchor
            #     are eligible.
            # ----------------------------------------------------

            floor_use = (
                anchored_floor
                if mode
                == "anchored_soft_occlusion"
                else floor
            )

            per_view_stats = []

            for v in range(V):

                usable = usable_all[v]
                delta_vox = delta_all[v]

                if (
                    mode
                    == "anchored_soft_occlusion"
                ):
                    eligible = (
                        usable
                        & anchored
                    )
                else:
                    eligible = usable

                finite_delta = torch.where(
                    torch.isfinite(
                        delta_vox
                    ),
                    delta_vox,
                    torch.zeros_like(
                        delta_vox
                    ),
                )

                excess = (
                    finite_delta
                    - margin_voxels
                ).clamp_min(0.0)

                soft = (
                    floor_use
                    + (
                        1.0
                        - floor_use
                    )
                    * torch.exp(
                        -0.5
                        * (
                            excess
                            / tau_voxels
                        ).square()
                    )
                )

                wv = torch.ones(
                    (R3,),
                    device=device,
                    dtype=torch.float32,
                )

                wv[
                    eligible
                ] = soft[
                    eligible
                ]

                weights[
                    b,
                    v,
                    :,
                    0,
                ] = wv.to(
                    torch.float16
                )

                occ = (
                    eligible
                    & (
                        finite_delta
                        > margin_voxels
                    )
                )

                usable_n = int(
                    usable.sum().item()
                )

                anchor_n = int(
                    anchor_support[
                        v
                    ].sum().item()
                )

                eligible_n = int(
                    eligible.sum().item()
                )

                occ_n = int(
                    occ.sum().item()
                )

                if eligible_n:

                    w_use = wv[
                        eligible
                    ]

                    d_use = finite_delta[
                        eligible
                    ]

                    wp10 = float(
                        torch.quantile(
                            w_use,
                            0.10,
                        ).item()
                    )

                    wp50 = float(
                        torch.quantile(
                            w_use,
                            0.50,
                        ).item()
                    )

                    wp90 = float(
                        torch.quantile(
                            w_use,
                            0.90,
                        ).item()
                    )

                    dp50 = float(
                        torch.quantile(
                            d_use,
                            0.50,
                        ).item()
                    )

                    dp90 = float(
                        torch.quantile(
                            d_use,
                            0.90,
                        ).item()
                    )

                else:
                    wp10 = wp50 = wp90 = float("nan")
                    dp50 = dp90 = float("nan")

                lt95 = int(
                    (
                        wv[eligible]
                        < 0.95
                    ).sum().item()
                ) if eligible_n else 0

                lt75 = int(
                    (
                        wv[eligible]
                        < 0.75
                    ).sum().item()
                ) if eligible_n else 0

                lt50 = int(
                    (
                        wv[eligible]
                        < 0.50
                    ).sum().item()
                ) if eligible_n else 0

                per_view_stats.append(
                    {
                        "usable":
                            usable_n,
                        "anchors":
                            anchor_n,
                        "eligible":
                            eligible_n,
                        "occluded":
                            occ_n,
                        "lt95":
                            lt95,
                        "lt75":
                            lt75,
                        "lt50":
                            lt50,
                    }
                )

                prefix = (
                    "SS-AnchoredOcc"
                    if mode
                    == "anchored_soft_occlusion"
                    else "SS-SoftOcc"
                )

                print(
                    f"[{prefix} view] "
                    f"b={b} v={v} "
                    f"{view_names[v]}: "
                    f"usable={usable_n}/{R3}, "
                    f"anchors={anchor_n}, "
                    f"eligible={eligible_n}, "
                    f"occluded={occ_n}, "
                    f"w<.95/.75/.50="
                    f"{lt95}/{lt75}/{lt50}, "
                    f"weight p10/50/90="
                    f"{wp10:.3f}/"
                    f"{wp50:.3f}/"
                    f"{wp90:.3f}, "
                    f"delta_vox p50/90="
                    f"{dp50:.2f}/"
                    f"{dp90:.2f}"
                )

            wb = (
                weights[
                    b,
                    :,
                    :,
                    0,
                ].float()
            )

            probs = (
                wb
                / wb.sum(
                    dim=0,
                    keepdim=True,
                ).clamp_min(1e-8)
            )

            uniform = (
                1.0 / float(V)
            )

            tv = (
                0.5
                * (
                    probs
                    - uniform
                ).abs().sum(
                    dim=0
                )
            )

            tv50 = float(
                torch.quantile(
                    tv,
                    0.50,
                ).item()
            )

            tv90 = float(
                torch.quantile(
                    tv,
                    0.90,
                ).item()
            )

            tv99 = float(
                torch.quantile(
                    tv,
                    0.99,
                ).item()
            )

            tv05 = int(
                (tv >= 0.05)
                .sum().item()
            )

            tv10 = int(
                (tv >= 0.10)
                .sum().item()
            )

            tv20 = int(
                (tv >= 0.20)
                .sum().item()
            )

            down_counts = (
                (wb < 0.95)
                .sum(dim=0)
            )

            down_hist = {
                int(n):
                    int(
                        (
                            down_counts == n
                        ).sum().item()
                    )
                for n in range(V + 1)
                if int(
                    (
                        down_counts == n
                    ).sum().item()
                ) > 0
            }

            mean_share = [
                float(x)
                for x in
                probs.mean(
                    dim=1
                ).detach().cpu().tolist()
            ]

            if (
                mode
                == "anchored_soft_occlusion"
            ):

                print(
                    f"[SS-AnchoredOcc] b={b}: "
                    f"V={V}, R={R}, cells={R3}, "
                    f"anchored="
                    f"{int(anchored.sum().item())}/{R3}, "
                    f"anchor_count={anchor_hist}, "
                    f"TV p50/90/99="
                    f"{tv50:.4f}/"
                    f"{tv90:.4f}/"
                    f"{tv99:.4f}, "
                    f"TV>=.05/.10/.20="
                    f"{tv05}/{tv10}/{tv20}, "
                    f"downweighted_views_per_cell="
                    f"{down_hist}, "
                    f"mean_normalized_share="
                    f"{[round(x,4) for x in mean_share]}"
                )

            else:

                print(
                    f"[SS-SoftOcc] b={b}: "
                    f"V={V}, R={R}, cells={R3}, "
                    f"TV p50/90/99="
                    f"{tv50:.4f}/"
                    f"{tv90:.4f}/"
                    f"{tv99:.4f}, "
                    f"TV>=.05/.10/.20="
                    f"{tv05}/{tv10}/{tv20}, "
                    f"downweighted_views_per_cell="
                    f"{down_hist}, "
                    f"mean_normalized_share="
                    f"{[round(x,4) for x in mean_share]}"
                )

            # ----------------------------------------------------
            # Spatial diagnostics.
            # ----------------------------------------------------

            tv_cpu = (
                tv.detach()
                .cpu()
                .float()
            )

            if (
                mode
                == "anchored_soft_occlusion"
            ):
                spatial_mask = (
                    anchored.detach()
                    .cpu()
                )
                file_prefix = (
                    "ss_anchored_occ_impact"
                )
            else:
                spatial_mask = torch.ones(
                    R3,
                    dtype=torch.bool,
                )
                file_prefix = (
                    "ss_soft_occ_impact"
                )

            if spatial_mask.any():

                tv_sel = tv_cpu[
                    spatial_mask
                ]

                scale99 = max(
                    float(
                        torch.quantile(
                            tv_sel,
                            0.99,
                        ).item()
                    ),
                    1e-6,
                )

                heat = (
                    tv_sel
                    / scale99
                ).clamp(
                    0.0,
                    1.0,
                )

                red = (
                    255.0 * heat
                ).byte()

                blue = (
                    255.0
                    * (
                        1.0 - heat
                    )
                ).byte()

                green = torch.zeros_like(
                    red
                )

                alpha_c = torch.full_like(
                    red,
                    255,
                )

                colors = torch.stack(
                    [
                        red,
                        green,
                        blue,
                        alpha_c,
                    ],
                    dim=-1,
                ).numpy()

                point_sel = (
                    pts_world
                    .detach()
                    .cpu()[
                        spatial_mask
                    ]
                    .numpy()
                )

                cloud = (
                    trimesh.points
                    .PointCloud(
                        point_sel,
                        colors=colors,
                    )
                )

                ply_path = os.path.join(
                    output_dir,
                    f"{file_prefix}_"
                    f"b{b}_{stamp}.ply",
                )

                cloud.export(
                    ply_path
                )

                print(
                    f"[SS-SoftOcc] "
                    f"impact PLY: "
                    f"{ply_path}"
                )

            # Anchor-count PLY:
            # 1 anchor = dark, multiple anchors = brighter green.
            if (
                mode
                == "anchored_soft_occlusion"
                and anchored.any()
            ):

                ac = (
                    anchor_count[
                        anchored
                    ]
                    .detach()
                    .cpu()
                    .float()
                )

                intensity = (
                    ac
                    / max(
                        float(V),
                        1.0,
                    )
                ).clamp(
                    0.0,
                    1.0,
                )

                r = torch.zeros_like(
                    intensity
                ).byte()

                g = (
                    80.0
                    + 175.0
                    * intensity
                ).byte()

                bl = torch.zeros_like(
                    g
                )

                a = torch.full_like(
                    g,
                    255,
                )

                anchor_colors = torch.stack(
                    [r, g, bl, a],
                    dim=-1,
                ).numpy()

                anchor_cloud = (
                    trimesh.points
                    .PointCloud(
                        pts_world
                        .detach()
                        .cpu()[
                            anchored.detach().cpu()
                        ].numpy(),
                        colors=anchor_colors,
                    )
                )

                anchor_path = os.path.join(
                    output_dir,
                    f"ss_anchored_occ_anchors_"
                    f"b{b}_{stamp}.ply",
                )

                anchor_cloud.export(
                    anchor_path
                )

                print(
                    f"[SS-AnchoredOcc] "
                    f"anchor PLY: "
                    f"{anchor_path}"
                )

        print(
            f"[SS-SoftOcc] mode={mode}; "
            "conditioning weights built. "
            "S4-Repair remains strictly post-SS "
            "and reuses the preserved MoGe state unchanged."
        )

        return weights


    @torch.no_grad()
    def _prepare_ss_moge_diag_state(
        self,
        image_cond_model,
        views: dict,
    ):
        """
        Snapshot MoGe + Pixal camera state for post-SS diagnostics.

        IMPORTANT:
        this method does NOT change conditioning.
        """

        from ..trainers.flow_matching.mixins.image_conditioned_proj import (
            compute_relative_calc_mat,
        )

        geo = views.get("ss_geometry", None)

        if geo is None:
            self._ss_moge_diag_state = None
            print("[SS-MoGe DIAG] no ss_geometry")
            return

        required = (
            "points",
            "depth",
            "normal",
            "mask",
            "intrinsics",
        )

        missing = [k for k in required if k not in geo]

        if missing:
            raise ValueError(
                f"[SS-MoGe DIAG] missing fields: {missing}"
            )

        device = self.device

        transform_matrix = views["transform_matrix"].to(
            device=device,
            dtype=torch.float32,
        )

        camera_distance = views["camera_distance"].to(
            device=device,
            dtype=torch.float32,
        )

        calc_mat = compute_relative_calc_mat(
            transform_matrix,
            camera_distance,
            image_cond_model.proj_grid.front_view_transform_matrix,
        ).detach().cpu().float()

        camera_angle = (
            views["camera_angle_x"]
            .detach()
            .cpu()
            .float()
        )

        mesh_scale = torch.as_tensor(
            views["mesh_scale"],
            dtype=torch.float32,
        ).reshape(-1).cpu()

        points = geo["points"].detach().cpu().float()
        depth = geo["depth"].detach().cpu().float()
        normal = geo["normal"].detach().cpu().float()
        mask = geo["mask"].detach().cpu().bool()
        K = geo["intrinsics"].detach().cpu().float()

        # Real foreground mask supplied to Pixal #33.
        # build_views() stores it as alpha, independently from
        # MoGe's own validity mask.
        alphas = views.get("alphas", None)

        if alphas is not None:
            alphas = alphas.detach().cpu().float()

            print(
                f"[MoGe-Views DIAG] foreground alpha available: "
                f"{tuple(alphas.shape)}"
            )
        else:
            print(
                "[MoGe-Views DIAG] WARNING: views['alphas'] missing; "
                "MoGe-vs-MoGe diagnostic will be skipped"
            )

        B, V = calc_mat.shape[:2]

        if points.shape[:2] != (B, V):
            raise ValueError(
                f"[SS-MoGe DIAG] points start with "
                f"{tuple(points.shape[:2])}, expected {(B, V)}"
            )

        H = int(points.shape[2])
        W = int(points.shape[3])

        view_names = list(
            views.get(
                "view_names",
                [f"view{v:02d}" for v in range(V)],
            )
        )

        # --------------------------------------------------------
        # K diagnostics
        # --------------------------------------------------------

        fx = K[..., 0, 0].clamp_min(1e-12)
        fy = K[..., 1, 1].clamp_min(1e-12)
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]

        fovx = torch.rad2deg(
            2.0 * torch.atan(0.5 / fx)
        )

        fovy = torch.rad2deg(
            2.0 * torch.atan(0.5 / fy)
        )

        pixal_fov = torch.rad2deg(camera_angle)

        print(
            f"[SS-MoGe DIAG] armed: "
            f"B={B}, V={V}, MoGe={W}x{H}, "
            f"PixalCond={int(image_cond_model.image_size)}"
        )

        # SS_SOFT_OCCLUSION_V1
        ss_mode = str(views.get("ss_visibility_mode", "all_views"))
        print(
            f"[SS-MoGe DIAG] state armed for diagnostics/S4; "
            f"SS visibility mode={ss_mode}"
        )

        for b in range(B):
            for v in range(V):
                print(
                    f"[SS-MoGe DIAG K] "
                    f"b={b} v={v} {view_names[v]}: "
                    f"fx={float(fx[b,v]):.6f} "
                    f"fy={float(fy[b,v]):.6f} "
                    f"cx={float(cx[b,v]):.6f} "
                    f"cy={float(cy[b,v]):.6f}; "
                    f"MoGe FOV="
                    f"{float(fovx[b,v]):.3f}x"
                    f"{float(fovy[b,v]):.3f}; "
                    f"Pixal FOVx="
                    f"{float(pixal_fov[b,v]):.3f}"
                )

        # --------------------------------------------------------
        # Reproduce OLD S2 span-scale only for comparison.
        #
        # NOTE: this deliberately uses MoGe's validity mask,
        # exactly like the current S2 code.
        # --------------------------------------------------------

        old_scales = []
        old_spans = []

        stride = max(
            1,
            int(max(H, W) // 512),
        )

        for b in range(B):

            spans = []

            for v in range(V):

                p = points[
                    b, v,
                    ::stride,
                    ::stride,
                ]

                m = mask[
                    b, v,
                    ::stride,
                    ::stride,
                ]

                valid = (
                    m
                    & torch.isfinite(p).all(dim=-1)
                )

                pv = p[valid]

                if pv.shape[0] < 64:
                    spans.append(float("nan"))
                    continue

                xy = pv[:, :2]

                lo = torch.quantile(
                    xy,
                    0.005,
                    dim=0,
                )

                hi = torch.quantile(
                    xy,
                    0.995,
                    dim=0,
                )

                spans.append(
                    float(
                        (hi - lo)
                        .max()
                        .item()
                    )
                )

            finite = [
                s for s in spans
                if np.isfinite(s) and s > 1e-12
            ]

            if finite:
                max_span = max(finite)
                target_span = (
                    1.0
                    / float(mesh_scale[b].item())
                )
                old_scale = target_span / max_span
            else:
                old_scale = float("nan")

            old_spans.append(spans)
            old_scales.append(old_scale)

            print(
                f"[SS-MoGe DIAG OLD-S2] "
                f"b={b}: spans="
                f"{[round(x,6) if np.isfinite(x) else None for x in spans]}, "
                f"global_scale={old_scale:.8f}"
            )

        self._ss_moge_diag_state = {
            "points": points,
            "depth": depth,
            "normal": normal,
            "mask": mask,
            "intrinsics": K,
            "alphas": alphas,

            "calc_mat": calc_mat,
            "camera_angle_x": camera_angle,
            "camera_distance": (
                views["camera_distance"]
                .detach()
                .cpu()
                .float()
            ),

            "mesh_scale": mesh_scale,
            "view_names": view_names,

            "cond_res": int(
                image_cond_model.image_size
            ),

            "H": H,
            "W": W,

            "old_s2_spans": old_spans,
            "old_s2_scale": old_scales,
        }


    @staticmethod
    def _moge_views_quantiles(x: torch.Tensor):
        x = x[torch.isfinite(x)]

        if x.numel() == 0:
            return [
                float("nan"),
                float("nan"),
                float("nan"),
            ]

        return [
            float(torch.quantile(x, 0.10).item()),
            float(torch.quantile(x, 0.50).item()),
            float(torch.quantile(x, 0.90).item()),
        ]


    @staticmethod
    def _moge_views_sample_points(
        points: torch.Tensor,
        max_points: int = 4000,
    ):
        """
        Deterministic subsampling so repeated runs remain comparable.
        """

        n = int(points.shape[0])

        if n <= max_points:
            return points

        idx = torch.linspace(
            0,
            n - 1,
            steps=max_points,
            dtype=torch.float32,
        ).round().long()

        return points[idx]


    @staticmethod
    def _moge_views_nn(
        src: torch.Tensor,
        dst: torch.Tensor,
        chunk: int = 256,
    ):
        """
        Directed nearest-neighbour distances + indices.

        CPU implementation on purpose: this diagnostic must not consume
        Pixal VRAM.
        """

        distances = []
        indices = []

        for i in range(0, src.shape[0], chunk):

            s = src[i:i + chunk]

            d = torch.cdist(
                s.float(),
                dst.float(),
            )

            val, idx = d.min(dim=1)

            distances.append(val.cpu())
            indices.append(idx.cpu())

            del d

        return (
            torch.cat(distances, dim=0),
            torch.cat(indices, dim=0),
        )


    @torch.no_grad()
    def _moge_views_fit_similarity(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        iterations: int = 6,
        trim_fraction: float = 0.60,
    ):
        """
        Robust ICP with FIXED rotation.

        Both point clouds have already been rotated into Pixal's canonical
        frame.  The only fitted freedoms are:

            P' = s * P + t

        This is intentionally NOT a full ICP rotation fit: if the camera
        rotations themselves are wrong, we want that error to remain visible.

        Only the closest trim_fraction of src->dst correspondences is used
        to fit each iteration because two different camera views do not
        observe the exact same surface set.
        """

        src = src.float().cpu()
        dst = dst.float().cpu()

        if src.shape[0] < 32 or dst.shape[0] < 32:
            raise ValueError(
                "not enough points for similarity fit"
            )

        # Start from robust centre alignment.
        src_center = torch.median(
            src,
            dim=0,
        ).values

        dst_center = torch.median(
            dst,
            dim=0,
        ).values

        scale = torch.tensor(
            1.0,
            dtype=torch.float32,
        )

        translation = (
            dst_center - src_center
        )

        for _ in range(iterations):

            current = (
                scale * src
                + translation
            )

            nn_dist, nn_idx = (
                self._moge_views_nn(
                    current,
                    dst,
                )
            )

            if nn_dist.numel() < 16:
                break

            cutoff = torch.quantile(
                nn_dist,
                trim_fraction,
            )

            keep = (
                nn_dist <= cutoff
            )

            if int(keep.sum()) < 16:
                break

            x = src[keep]
            y = dst[nn_idx[keep]]

            mx = x.mean(dim=0)
            my = y.mean(dim=0)

            xc = x - mx
            yc = y - my

            denom = (
                xc * xc
            ).sum().clamp_min(1e-12)

            new_scale = (
                (xc * yc).sum()
                / denom
            )

            # Uniform negative scale is physically meaningless here.
            if (
                not torch.isfinite(new_scale)
                or float(new_scale) <= 0
            ):
                break

            scale = new_scale

            translation = (
                my - scale * mx
            )

        fitted = (
            scale * src
            + translation
        )

        # Directed residuals both ways.
        raw_s2d, _ = self._moge_views_nn(
            src,
            dst,
        )

        raw_d2s, _ = self._moge_views_nn(
            dst,
            src,
        )

        fit_s2d, _ = self._moge_views_nn(
            fitted,
            dst,
        )

        fit_d2s, _ = self._moge_views_nn(
            dst,
            fitted,
        )

        return {
            "scale":
                float(scale.item()),

            "translation": [
                float(v)
                for v in translation.tolist()
            ],

            "raw_src_to_ref":
                self._moge_views_quantiles(
                    raw_s2d
                ),

            "raw_ref_to_src":
                self._moge_views_quantiles(
                    raw_d2s
                ),

            "fit_src_to_ref":
                self._moge_views_quantiles(
                    fit_s2d
                ),

            "fit_ref_to_src":
                self._moge_views_quantiles(
                    fit_d2s
                ),

            "fitted_points":
                fitted,

            "trim_fraction":
                float(trim_fraction),
        }


    @torch.no_grad()
    def _diagnose_moge_views_alignment(self):
        """
        Direct MoGe-vs-MoGe diagnostic.

        No SS voxels participate here.

        Generates three overlays:

          1. oldS2:
             Exactly how the old S2 global-scale interpretation places
             the V MoGe point clouds in Pixal canonical space.

          2. normalized:
             Each view is independently:
                 - foreground masked,
                 - robust-centred,
                 - normalized to unit object height,
                 - rotated into Pixal canonical orientation.

             This deliberately removes global scale and translation so
             local shape disagreement becomes visible.

          3. fit_ref0:
             normalized view 1/2/3 fitted to normalized view 0 by
             uniform scale + translation ONLY. Rotation remains fixed.

        Foreground comes from views['alphas'].
        MoGe mask is used only as a validity mask.
        """

        import json
        import time
        import torch.nn.functional as F

        state = getattr(
            self,
            "_ss_moge_diag_state",
            None,
        )

        if state is None:
            return

        alphas = state.get(
            "alphas",
            None,
        )

        if alphas is None:

            print(
                "[MoGe-Views DIAG] no foreground alpha; "
                "skipping direct view consistency test"
            )

            return

        points = state["points"]
        moge_valid = state["mask"]
        calc_mat = state["calc_mat"]

        view_names = state[
            "view_names"
        ]

        old_scales = state[
            "old_s2_scale"
        ]

        B, V, H, W, _ = (
            points.shape
        )

        if alphas.shape[:2] != (B, V):

            raise ValueError(
                f"[MoGe-Views DIAG] alpha begins with "
                f"{tuple(alphas.shape[:2])}, expected {(B, V)}"
            )

        # --------------------------------------------------------
        # Resize real foreground masks to MoGe native resolution.
        # --------------------------------------------------------

        alpha_native = F.interpolate(
            alphas.reshape(
                B * V,
                1,
                alphas.shape[-2],
                alphas.shape[-1],
            ),
            size=(H, W),
            mode="nearest",
        ).reshape(
            B,
            V,
            H,
            W,
        )

        foreground = (
            alpha_native > 0.5
        )

        # OpenCV camera -> Blender camera.
        cv_to_blender = torch.tensor(
            [
                1.0,
                -1.0,
                -1.0,
            ],
            dtype=torch.float32,
        )

        view_colors = [
            np.array(
                [255, 60, 60, 255],
                dtype=np.uint8,
            ),  # 0 red

            np.array(
                [60, 220, 60, 255],
                dtype=np.uint8,
            ),  # 90 green

            np.array(
                [70, 120, 255, 255],
                dtype=np.uint8,
            ),  # 180 blue

            np.array(
                [255, 220, 60, 255],
                dtype=np.uint8,
            ),  # 270 yellow

            np.array(
                [220, 70, 220, 255],
                dtype=np.uint8,
            ),

            np.array(
                [60, 220, 220, 255],
                dtype=np.uint8,
            ),
        ]

        ref_color = np.array(
            [190, 190, 190, 255],
            dtype=np.uint8,
        )

        output_dir = os.path.join(
            folder_paths.get_output_directory(),
            "debug",
            "pixal3d",
        )

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        stamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        report = {
            "description":
                "Direct MoGe view consistency diagnostic",

            "foreground":
                "Pixal views['alphas'] AND MoGe validity mask",

            "oldS2":
                "one global XY-span scale, same as previous S2",

            "normalized":
                "robust centre removed + unit foreground height + "
                "fixed camera rotation",

            "fit":
                "trimmed fixed-rotation similarity fit to normalized view0",

            "batches": [],
        }

        for b in range(B):

            batch_report = {
                "batch": b,
                "old_s2_global_scale":
                    float(old_scales[b]),

                "views": [],
                "fits_to_view0": [],
            }

            old_s2_clouds = []
            normalized_clouds = []

            # ====================================================
            # Construct all V clouds.
            # ====================================================

            for v in range(V):

                q = points[
                    b,
                    v,
                ].float()

                valid = (
                    foreground[
                        b,
                        v,
                    ]
                    & moge_valid[
                        b,
                        v,
                    ].bool()
                    & torch.isfinite(q)
                    .all(dim=-1)
                )

                q = q[valid]

                count_full = int(
                    q.shape[0]
                )

                if count_full < 64:
                    raise ValueError(
                        f"[MoGe-Views DIAG] "
                        f"view {v} only has "
                        f"{count_full} foreground points"
                    )

                # -----------------------------------------------
                # Raw metric geometry statistics in OpenCV
                # camera coordinates.
                # -----------------------------------------------

                q01 = torch.quantile(
                    q,
                    0.01,
                    dim=0,
                )

                q50 = torch.quantile(
                    q,
                    0.50,
                    dim=0,
                )

                q99 = torch.quantile(
                    q,
                    0.99,
                    dim=0,
                )

                robust_span = (
                    q99 - q01
                )

                robust_center = (
                    q01 + q99
                ) * 0.5

                # OpenCV Y is down; magnitude is still object height.
                robust_height = float(
                    robust_span[1].item()
                )

                if (
                    not np.isfinite(
                        robust_height
                    )
                    or robust_height <= 1e-8
                ):
                    raise ValueError(
                        f"[MoGe-Views DIAG] invalid robust height "
                        f"for view {v}: {robust_height}"
                    )

                # Use a deterministic subset for exports/fitting.
                qs = (
                    self._moge_views_sample_points(
                        q,
                        max_points=4000,
                    )
                )

                # -----------------------------------------------
                # A) old-S2 interpretation.
                #
                # This reproduces:
                #     q *= global_moge_scale
                # and then puts camera-space q into Pixal's
                # canonical c2w camera.
                # -----------------------------------------------

                old_scale = float(
                    old_scales[b]
                )

                q_old_cv = (
                    qs * old_scale
                )

                q_old_blender = (
                    q_old_cv
                    * cv_to_blender
                )

                ones = torch.ones(
                    (
                        q_old_blender.shape[0],
                        1,
                    ),
                    dtype=torch.float32,
                )

                q_old_h = torch.cat(
                    [
                        q_old_blender,
                        ones,
                    ],
                    dim=-1,
                )

                q_old_world = (
                    q_old_h
                    @ calc_mat[b, v].T
                )[:, :3]

                old_s2_clouds.append(
                    q_old_world
                )

                # -----------------------------------------------
                # B) shape-normalized cloud.
                #
                # Remove global translation and scale in MoGe
                # camera space, then apply only Pixal's known
                # camera ROTATION into canonical coordinates.
                # -----------------------------------------------

                q_norm_cv = (
                    qs - robust_center
                ) / robust_height

                q_norm_blender = (
                    q_norm_cv
                    * cv_to_blender
                )

                R_cam_to_canonical = (
                    calc_mat[
                        b,
                        v,
                        :3,
                        :3,
                    ].float()
                )

                q_norm_world = (
                    q_norm_blender
                    @ R_cam_to_canonical.T
                )

                normalized_clouds.append(
                    q_norm_world
                )

                batch_report[
                    "views"
                ].append(
                    {
                        "view": v,

                        "name":
                            view_names[v],

                        "foreground_points":
                            count_full,

                        "metric_q01_xyz":
                            [
                                float(x)
                                for x in q01.tolist()
                            ],

                        "metric_median_xyz":
                            [
                                float(x)
                                for x in q50.tolist()
                            ],

                        "metric_q99_xyz":
                            [
                                float(x)
                                for x in q99.tolist()
                            ],

                        "metric_robust_span_xyz":
                            [
                                float(x)
                                for x in robust_span.tolist()
                            ],

                        "metric_robust_height_y":
                            robust_height,

                        "metric_robust_center_xyz":
                            [
                                float(x)
                                for x in robust_center.tolist()
                            ],
                    }
                )

                print(
                    f"[MoGe-Views DIAG] "
                    f"b={b} v={v} {view_names[v]}: "
                    f"fg={count_full}, "
                    f"span="
                    f"{[round(float(x),5) for x in robust_span.tolist()]}, "
                    f"height={robust_height:.6f}, "
                    f"medianZ={float(q50[2]):.6f}"
                )

            # ====================================================
            # Export OLD S2 interpretation.
            # ====================================================

            old_vertices = []
            old_colors = []

            for v, cloud in enumerate(
                old_s2_clouds
            ):

                old_vertices.append(
                    cloud.numpy()
                )

                old_colors.append(
                    np.tile(
                        view_colors[
                            v % len(view_colors)
                        ][None, :],
                        (
                            cloud.shape[0],
                            1,
                        ),
                    )
                )

            old_vertices = (
                np.concatenate(
                    old_vertices,
                    axis=0,
                ).astype(np.float32)
            )

            old_colors = (
                np.concatenate(
                    old_colors,
                    axis=0,
                ).astype(np.uint8)
            )

            old_cloud = (
                trimesh.points.PointCloud(
                    old_vertices,
                    colors=old_colors,
                )
            )

            old_path = os.path.join(
                output_dir,
                f"moge_views_oldS2_"
                f"{stamp}_b{b}.ply",
            )

            old_cloud.export(
                old_path
            )

            print(
                f"[MoGe-Views DIAG] "
                f"old-S2 overlay: {old_path}"
            )

            # ====================================================
            # Export normalized overlay.
            # ====================================================

            norm_vertices = []
            norm_colors = []

            for v, cloud in enumerate(
                normalized_clouds
            ):

                norm_vertices.append(
                    cloud.numpy()
                )

                norm_colors.append(
                    np.tile(
                        view_colors[
                            v % len(view_colors)
                        ][None, :],
                        (
                            cloud.shape[0],
                            1,
                        ),
                    )
                )

            norm_vertices = (
                np.concatenate(
                    norm_vertices,
                    axis=0,
                ).astype(np.float32)
            )

            norm_colors = (
                np.concatenate(
                    norm_colors,
                    axis=0,
                ).astype(np.uint8)
            )

            norm_cloud = (
                trimesh.points.PointCloud(
                    norm_vertices,
                    colors=norm_colors,
                )
            )

            norm_path = os.path.join(
                output_dir,
                f"moge_views_normalized_"
                f"{stamp}_b{b}.ply",
            )

            norm_cloud.export(
                norm_path
            )

            print(
                f"[MoGe-Views DIAG] "
                f"normalized overlay: {norm_path}"
            )

            # ====================================================
            # Fit normalized view1.. to normalized view0.
            #
            # Rotation stays FIXED.
            # ====================================================

            reference = (
                normalized_clouds[0]
            )

            fit_vertices = [
                reference.numpy()
            ]

            fit_colors = [
                np.tile(
                    ref_color[None, :],
                    (
                        reference.shape[0],
                        1,
                    ),
                )
            ]

            for v in range(
                1,
                V,
            ):

                fit = (
                    self._moge_views_fit_similarity(
                        normalized_clouds[v],
                        reference,
                        iterations=6,
                        trim_fraction=0.60,
                    )
                )

                fitted = fit[
                    "fitted_points"
                ]

                fit_vertices.append(
                    fitted.numpy()
                )

                fit_colors.append(
                    np.tile(
                        view_colors[
                            v % len(view_colors)
                        ][None, :],
                        (
                            fitted.shape[0],
                            1,
                        ),
                    )
                )

                fit_report = {
                    k: value
                    for k, value in fit.items()
                    if k != "fitted_points"
                }

                fit_report.update(
                    {
                        "view": v,
                        "name":
                            view_names[v],
                    }
                )

                batch_report[
                    "fits_to_view0"
                ].append(
                    fit_report
                )

                print(
                    f"[MoGe-Views FIT] "
                    f"{view_names[v]} -> {view_names[0]}: "
                    f"s={fit['scale']:.6f}, "
                    f"t="
                    f"{[round(x,5) for x in fit['translation']]}"
                )

                print(
                    f"[MoGe-Views FIT] "
                    f"  raw src->ref p10/med/p90="
                    f"{fit['raw_src_to_ref'][0]:.5f}/"
                    f"{fit['raw_src_to_ref'][1]:.5f}/"
                    f"{fit['raw_src_to_ref'][2]:.5f}; "
                    f"fit="
                    f"{fit['fit_src_to_ref'][0]:.5f}/"
                    f"{fit['fit_src_to_ref'][1]:.5f}/"
                    f"{fit['fit_src_to_ref'][2]:.5f}"
                )

                print(
                    f"[MoGe-Views FIT] "
                    f"  raw ref->src p10/med/p90="
                    f"{fit['raw_ref_to_src'][0]:.5f}/"
                    f"{fit['raw_ref_to_src'][1]:.5f}/"
                    f"{fit['raw_ref_to_src'][2]:.5f}; "
                    f"fit="
                    f"{fit['fit_ref_to_src'][0]:.5f}/"
                    f"{fit['fit_ref_to_src'][1]:.5f}/"
                    f"{fit['fit_ref_to_src'][2]:.5f}"
                )

            fit_vertices = (
                np.concatenate(
                    fit_vertices,
                    axis=0,
                ).astype(np.float32)
            )

            fit_colors = (
                np.concatenate(
                    fit_colors,
                    axis=0,
                ).astype(np.uint8)
            )

            fitted_cloud = (
                trimesh.points.PointCloud(
                    fit_vertices,
                    colors=fit_colors,
                )
            )

            fit_path = os.path.join(
                output_dir,
                f"moge_views_fit_ref0_"
                f"{stamp}_b{b}.ply",
            )

            fitted_cloud.export(
                fit_path
            )

            print(
                f"[MoGe-Views DIAG] "
                f"fit overlay: {fit_path}"
            )

            batch_report[
                "old_s2_overlay"
            ] = old_path

            batch_report[
                "normalized_overlay"
            ] = norm_path

            batch_report[
                "fit_overlay"
            ] = fit_path

            report[
                "batches"
            ].append(
                batch_report
            )

        json_path = os.path.join(
            output_dir,
            f"moge_views_alignment_"
            f"{stamp}.json",
        )

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                report,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"[MoGe-Views DIAG] "
            f"JSON report: {json_path}"
        )


    @torch.no_grad()
    def _diagnose_ss_moge_against_coords(
        self,
        coords: torch.Tensor,
        resolution: int,
    ):
        """
        Compare baseline SS occupancy to MoGe AFTER sampling.

        MoGe is not allowed to influence this SS run.

        Reports per view:
          * ray-angle error;
          * best scale-only fit;
          * best scale + camera-Z-offset fit;
          * residuals in SS voxel units.

        Also exports:
          * JSON report;
          * one PLY overlay:
              gray   = baseline SS voxels
              red    = view 0 aligned MoGe samples
              green  = view 1
              blue   = view 2
              yellow = view 3
        """

        state = getattr(
            self,
            "_ss_moge_diag_state",
            None,
        )

        if state is None:
            return

        # Independent of SS geometry itself:
        # compare the four foreground MoGe point clouds directly.
        try:
            self._diagnose_moge_views_alignment()

        except Exception as e:
            import traceback

            print(
                f"[MoGe-Views DIAG] FAILED: {e}"
            )

            traceback.print_exc()

        if coords is None or coords.numel() == 0:
            print(
                "[SS-MoGe DIAG] empty coords, "
                "post-SS analysis skipped"
            )
            return

        from ..trainers.flow_matching.mixins.image_conditioned_proj import (
            project_points_to_image_batch,
        )

        import json
        import time

        points = state["points"]
        mask = state["mask"]
        K = state["intrinsics"]

        calc_mat = state["calc_mat"]
        camera_angle = state["camera_angle_x"]
        mesh_scale = state["mesh_scale"]

        view_names = state["view_names"]

        cond_res = int(state["cond_res"])
        H = int(state["H"])
        W = int(state["W"])

        coords = (
            coords
            .detach()
            .cpu()
            .long()
        )

        B, V = calc_mat.shape[:2]
        R = int(resolution)

        # Same coordinate rotation used by ProjGrid.
        rot = torch.tensor(
            [
                [1.0, 0.0,  0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0,  0.0],
            ],
            dtype=torch.float32,
        )

        def quantiles(x):

            x = x[torch.isfinite(x)]

            if x.numel() == 0:
                return [
                    float("nan"),
                    float("nan"),
                    float("nan"),
                ]

            return [
                float(
                    torch.quantile(x, 0.10).item()
                ),
                float(
                    torch.quantile(x, 0.50).item()
                ),
                float(
                    torch.quantile(x, 0.90).item()
                ),
            ]

        def fit_scale(p, q):
            """
            Robust scale-only fit:
                P ~= s Q
            """

            pn = torch.linalg.vector_norm(
                p,
                dim=-1,
            )

            qn = torch.linalg.vector_norm(
                q,
                dim=-1,
            ).clamp_min(1e-12)

            s0 = torch.median(
                pn / qn
            )

            r0 = torch.linalg.vector_norm(
                p - s0 * q,
                dim=-1,
            )

            if r0.numel() > 8:
                cutoff = torch.quantile(
                    r0,
                    0.90,
                )
                keep = r0 <= cutoff
            else:
                keep = torch.ones_like(
                    r0,
                    dtype=torch.bool,
                )

            denom = (
                q[keep] * q[keep]
            ).sum().clamp_min(1e-12)

            s = (
                p[keep] * q[keep]
            ).sum() / denom

            residual = torch.linalg.vector_norm(
                p - s * q,
                dim=-1,
            )

            return (
                float(s.item()),
                residual,
            )

        def fit_scale_z(p, q):
            """
            Robust fit:
                P ~= s Q + [0,0,tz]
            """

            def solve(pp, qq):

                n = pp.shape[0]

                A = torch.zeros(
                    (n * 3, 2),
                    dtype=torch.float32,
                )

                y = (
                    pp
                    .reshape(-1)
                    .float()
                )

                A[0::3, 0] = qq[:, 0]
                A[1::3, 0] = qq[:, 1]
                A[2::3, 0] = qq[:, 2]

                A[2::3, 1] = 1.0

                sol = torch.linalg.lstsq(
                    A,
                    y[:, None],
                ).solution[:, 0]

                return sol[0], sol[1]

            s0, tz0 = solve(p, q)

            shift0 = torch.tensor(
                [0.0, 0.0, float(tz0)],
                dtype=torch.float32,
            )

            r0 = torch.linalg.vector_norm(
                p - (s0 * q + shift0),
                dim=-1,
            )

            if r0.numel() > 8:
                cutoff = torch.quantile(
                    r0,
                    0.90,
                )
                keep = r0 <= cutoff
            else:
                keep = torch.ones_like(
                    r0,
                    dtype=torch.bool,
                )

            s, tz = solve(
                p[keep],
                q[keep],
            )

            shift = torch.tensor(
                [0.0, 0.0, float(tz)],
                dtype=torch.float32,
            )

            residual = torch.linalg.vector_norm(
                p - (s * q + shift),
                dim=-1,
            )

            return (
                float(s.item()),
                float(tz.item()),
                residual,
            )

        report = {
            "resolution": R,
            "cond_res": cond_res,
            "moge_hw": [H, W],

            "old_s2_scale":
                state["old_s2_scale"],

            "old_s2_spans":
                state["old_s2_spans"],

            "views": [],
        }

        ply_vertices = []
        ply_colors = []

        gray = np.array(
            [170, 170, 170, 255],
            dtype=np.uint8,
        )

        colors = [
            np.array(
                [255, 60, 60, 255],
                dtype=np.uint8,
            ),
            np.array(
                [60, 220, 60, 255],
                dtype=np.uint8,
            ),
            np.array(
                [70, 120, 255, 255],
                dtype=np.uint8,
            ),
            np.array(
                [255, 220, 60, 255],
                dtype=np.uint8,
            ),
        ]

        for b in range(B):

            selected = (
                coords[:, 0] == b
            )

            c = coords[selected]

            if c.numel() == 0:
                continue

            idx = c[:, 1:4].float()

            lattice = (
                -1.0
                + 2.0 * idx
                / float(R - 1)
            )

            pts_world = (
                lattice
                @ rot.T
            )

            pts_world = (
                pts_world
                / float(mesh_scale[b].item())
                / 2.0
            )

            ply_vertices.append(
                pts_world.numpy()
            )

            ply_colors.append(
                np.tile(
                    gray[None, :],
                    (pts_world.shape[0], 1),
                )
            )

            voxel_size = (
                1.0
                / max(R - 1, 1)
                / float(mesh_scale[b].item())
            )

            pts_h = torch.cat(
                [
                    pts_world,
                    torch.ones(
                        (
                            pts_world.shape[0],
                            1,
                        ),
                        dtype=torch.float32,
                    ),
                ],
                dim=-1,
            )

            for v in range(V):

                points_2d, proj_depth, proj_valid = (
                    project_points_to_image_batch(
                        pts_world.unsqueeze(0),
                        calc_mat[b:b + 1, v],
                        camera_angle[b:b + 1, v],
                        cond_res,
                    )
                )

                pix = points_2d[0]
                d = proj_depth[0].float()
                valid = proj_valid[0].bool()

                px = (
                    pix[:, 0]
                    .round()
                    .long()
                    .clamp(
                        0,
                        cond_res - 1,
                    )
                )

                py = (
                    pix[:, 1]
                    .round()
                    .long()
                    .clamp(
                        0,
                        cond_res - 1,
                    )
                )

                pixel_id = (
                    py * cond_res + px
                )

                # ---------------------------------------------
                # Only front-most OCCUPIED SS voxels.
                # ---------------------------------------------

                zbuf = torch.full(
                    (
                        cond_res
                        * cond_res,
                    ),
                    float("inf"),
                    dtype=torch.float32,
                )

                if valid.any():

                    zbuf.scatter_reduce_(
                        0,
                        pixel_id[valid],
                        d[valid],
                        reduce="amin",
                        include_self=True,
                    )

                nearest = zbuf[pixel_id]

                front_eps = (
                    0.35 * voxel_size
                )

                front = (
                    valid
                    & (
                        d
                        <= nearest
                        + front_eps
                    )
                )

                # ---------------------------------------------
                # Same Pixal pixel -> native MoGe pixel.
                # ---------------------------------------------

                gx = (
                    (pix[:, 0] + 0.5)
                    * (
                        W
                        / float(cond_res)
                    )
                    - 0.5
                )

                gy = (
                    (pix[:, 1] + 0.5)
                    * (
                        H
                        / float(cond_res)
                    )
                    - 0.5
                )

                ix = (
                    gx.round()
                    .long()
                    .clamp(
                        0,
                        W - 1,
                    )
                )

                iy = (
                    gy.round()
                    .long()
                    .clamp(
                        0,
                        H - 1,
                    )
                )

                q = (
                    points[
                        b,
                        v,
                    ][iy, ix]
                    .float()
                )

                # IMPORTANT:
                # only MoGe's own validity flag is used.
                # There is NO Pixal silhouette/mask gating here.
                q_valid = (
                    mask[
                        b,
                        v,
                    ][iy, ix]
                    .bool()
                    & torch.isfinite(q)
                    .all(dim=-1)
                )

                # ---------------------------------------------
                # Pixal world -> camera -> OpenCV convention.
                # ---------------------------------------------

                world_to_camera = (
                    torch.linalg.inv(
                        calc_mat[b, v]
                        .float()
                    )
                )

                p_blender = (
                    pts_h
                    @ world_to_camera.T
                )[:, :3]

                p_cv = (
                    p_blender
                    * torch.tensor(
                        [
                            1.0,
                            -1.0,
                            -1.0,
                        ],
                        dtype=torch.float32,
                    )
                )

                corr = (
                    front
                    & q_valid
                    & torch.isfinite(p_cv)
                    .all(dim=-1)
                )

                p = p_cv[corr]
                qq = q[corr]

                if p.shape[0] < 16:

                    print(
                        f"[SS-MoGe DIAG] "
                        f"b={b} v={v} "
                        f"{view_names[v]}: "
                        f"only {p.shape[0]} "
                        f"correspondences"
                    )

                    continue

                # ---------------------------------------------
                # 1. Ray alignment — independent of metric scale.
                # ---------------------------------------------

                p_dir = (
                    p
                    / torch.linalg.vector_norm(
                        p,
                        dim=-1,
                        keepdim=True,
                    ).clamp_min(1e-12)
                )

                q_dir = (
                    qq
                    / torch.linalg.vector_norm(
                        qq,
                        dim=-1,
                        keepdim=True,
                    ).clamp_min(1e-12)
                )

                cosang = (
                    p_dir * q_dir
                ).sum(
                    dim=-1
                ).clamp(
                    -1.0,
                    1.0,
                )

                ray_deg = torch.rad2deg(
                    torch.acos(cosang)
                )

                ray_q = quantiles(
                    ray_deg
                )

                # ---------------------------------------------
                # 2. Scale-only.
                # ---------------------------------------------

                scale_only, res_scale = (
                    fit_scale(
                        p,
                        qq,
                    )
                )

                res_scale_q = quantiles(
                    res_scale
                )

                res_scale_vox = [
                    x / voxel_size
                    for x in res_scale_q
                ]

                # ---------------------------------------------
                # 3. Scale + camera-Z offset.
                # ---------------------------------------------

                scale_z, tz, res_scale_z = (
                    fit_scale_z(
                        p,
                        qq,
                    )
                )

                res_scale_z_q = quantiles(
                    res_scale_z
                )

                res_scale_z_vox = [
                    x / voxel_size
                    for x in res_scale_z_q
                ]

                metrics = {
                    "batch": b,
                    "view": v,
                    "name": view_names[v],

                    "front_ss_voxels":
                        int(
                            front.sum().item()
                        ),

                    "correspondences":
                        int(p.shape[0]),

                    "ray_deg_p10_med_p90":
                        ray_q,

                    "scale_only":
                        scale_only,

                    "scale_only_residual_vox_p10_med_p90":
                        res_scale_vox,

                    "scale_z":
                        scale_z,

                    "z_offset":
                        tz,

                    "scale_z_residual_vox_p10_med_p90":
                        res_scale_z_vox,

                    "K": {
                        "fx":
                            float(
                                K[b,v,0,0]
                            ),
                        "fy":
                            float(
                                K[b,v,1,1]
                            ),
                        "cx":
                            float(
                                K[b,v,0,2]
                            ),
                        "cy":
                            float(
                                K[b,v,1,2]
                            ),
                    },
                }

                report["views"].append(
                    metrics
                )

                print(
                    f"[SS-MoGe DIAG FIT] "
                    f"b={b} v={v} "
                    f"{view_names[v]}: "
                    f"corr={p.shape[0]}, "
                    f"front="
                    f"{int(front.sum())}; "
                    f"ray deg "
                    f"p10/med/p90="
                    f"{ray_q[0]:.4f}/"
                    f"{ray_q[1]:.4f}/"
                    f"{ray_q[2]:.4f}"
                )

                print(
                    f"[SS-MoGe DIAG FIT] "
                    f"  scale-only: "
                    f"s={scale_only:.8f}; "
                    f"residual vox "
                    f"p10/med/p90="
                    f"{res_scale_vox[0]:.3f}/"
                    f"{res_scale_vox[1]:.3f}/"
                    f"{res_scale_vox[2]:.3f}"
                )

                print(
                    f"[SS-MoGe DIAG FIT] "
                    f"  scale+Z: "
                    f"s={scale_z:.8f}, "
                    f"tz={tz:.8f}; "
                    f"residual vox "
                    f"p10/med/p90="
                    f"{res_scale_z_vox[0]:.3f}/"
                    f"{res_scale_z_vox[1]:.3f}/"
                    f"{res_scale_z_vox[2]:.3f}"
                )

                # ---------------------------------------------
                # Overlay PLY:
                # fitted MoGe samples -> Pixal canonical world.
                # ---------------------------------------------

                q_cv = (
                    scale_z * qq
                    + torch.tensor(
                        [
                            0.0,
                            0.0,
                            tz,
                        ],
                        dtype=torch.float32,
                    )
                )

                q_blender = (
                    q_cv
                    * torch.tensor(
                        [
                            1.0,
                            -1.0,
                            -1.0,
                        ],
                        dtype=torch.float32,
                    )
                )

                q_h = torch.cat(
                    [
                        q_blender,
                        torch.ones(
                            (
                                q_blender.shape[0],
                                1,
                            ),
                            dtype=torch.float32,
                        ),
                    ],
                    dim=-1,
                )

                q_world = (
                    q_h
                    @ calc_mat[b, v].T
                )[:, :3]

                color = colors[
                    v % len(colors)
                ]

                ply_vertices.append(
                    q_world.numpy()
                )

                ply_colors.append(
                    np.tile(
                        color[None, :],
                        (
                            q_world.shape[0],
                            1,
                        ),
                    )
                )

        if not report["views"]:

            print(
                "[SS-MoGe DIAG] "
                "no usable fits"
            )

            return

        output_dir = os.path.join(
            folder_paths.get_output_directory(),
            "debug",
            "pixal3d",
        )

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        stamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        json_path = os.path.join(
            output_dir,
            f"moge_ss_alignment_diag_"
            f"{stamp}.json",
        )

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                report,
                f,
                indent=2,
                ensure_ascii=False,
            )

        if ply_vertices:

            vertices = np.concatenate(
                ply_vertices,
                axis=0,
            ).astype(
                np.float32
            )

            colors_np = np.concatenate(
                ply_colors,
                axis=0,
            ).astype(
                np.uint8
            )

            cloud = trimesh.points.PointCloud(
                vertices,
                colors=colors_np,
            )

            ply_path = os.path.join(
                output_dir,
                f"moge_ss_alignment_diag_"
                f"{stamp}.ply",
            )

            cloud.export(
                ply_path
            )

            print(
                f"[SS-MoGe DIAG] "
                f"overlay PLY: {ply_path}"
            )

        print(
            f"[SS-MoGe DIAG] "
            f"JSON report: {json_path}"
        )


    def _shape512_selected_sources(self, views: dict, stage: str = "shape512"):
        stage = str(stage).strip().lower()

        if stage == "shape512":
            normal_source = str(
                views.get("shape512_normals_source", "marigold")
            ).strip().lower()
            depth_source = str(
                views.get("shape512_depth_source", "marigold")
            ).strip().lower()
            label = "Shape512"
        elif stage == "shape_hr":
            normal_source = str(
                views.get("shape_hr_normals_source", "marigold")
            ).strip().lower()
            depth_source = str(
                views.get("shape_hr_depth_source", "marigold")
            ).strip().lower()
            label = "ShapeHR"
        else:
            raise ValueError(f"invalid geometry source stage {stage!r}")

        if normal_source not in ("marigold", "moge"):
            raise ValueError(f"invalid {label} normal source {normal_source!r}")
        if depth_source not in ("marigold", "moge"):
            raise ValueError(f"invalid {label} depth source {depth_source!r}")

        normal_key = (
            "geometry_confidence"
            if normal_source == "marigold"
            else "moge_geometry_confidence"
        )
        depth_key = (
            "marigold_depth_native"
            if depth_source == "marigold"
            else "moge_depth_native"
        )
        alpha_key = (
            "marigold_alpha_native"
            if depth_source == "marigold"
            else "moge_alpha_native"
        )

        return {
            "normal_source": normal_source,
            "depth_source": depth_source,
            "normal_key": normal_key,
            "depth_key": depth_key,
            "alpha_key": alpha_key,
        }

    def _shape_geometry_stage_config(
        self,
        image_cond_model,
        views: dict,
        geometry_stage: str = "auto",
    ):
        stage_image_size = int(image_cond_model.image_size)
        grid_res = int(image_cond_model.grid_resolution)
        stage = str(geometry_stage).strip().lower()

        if stage == "auto":
            stage = "shape512" if stage_image_size == 512 else "off"

        if stage in ("off", "none"):
            return {
                "stage": "off",
                "label": "ShapeOff",
                "slug": "shapeoff",
                "mode": "off",
                "source_info": None,
                "depth_floor": 0.70,
                "depth_tau_voxels": 2.0,
            }

        if stage in ("shape512", "shape_512"):
            mode = str(
                views.get("marigold_shape512_mode", "normal")
            ).strip().lower()
            source_stage = "shape512"
            label = "Shape512"
            slug = "shape512"
            depth_floor = 0.70
            depth_tau_voxels = 2.0

        elif stage in ("shape_hr", "hr", "cascade"):
            mode = str(
                views.get("shape_hr_mode", "off")
            ).strip().lower()
            source_stage = "shape_hr"
            label = f"ShapeHR-R{grid_res}"
            slug = f"shapehr_r{grid_res}"
            depth_floor = 0.70

            # Preserve approximately the same WORLD-SPACE depth tolerance
            # as Shape512's 2.0 voxels at R=32.  A fixed 2.0 voxels at R=96
            # would be about 3x tighter physically and would over-penalize
            # small camera/depth disagreement.
            depth_tau_voxels = 2.0 * max(grid_res - 1, 1) / 31.0

        else:
            raise ValueError(f"invalid geometry_stage={geometry_stage!r}")

        if mode not in ("off", "normal", "depth", "normal+depth"):
            raise ValueError(f"invalid {label} mode {mode!r}")

        source_info = self._shape512_selected_sources(
            views,
            stage=source_stage,
        )

        return {
            "stage": source_stage,
            "label": label,
            "slug": slug,
            "mode": mode,
            "source_info": source_info,
            "depth_floor": float(depth_floor),
            "depth_tau_voxels": float(depth_tau_voxels),
        }


    # MARIGOLD512_DIAG_V2
    @torch.no_grad()
    def _shape512_marigold_diagnostics(
        self,
        image_cond_model,
        views: dict,
        coords: torch.Tensor,
        visibility_weights: torch.Tensor,
        depth_floor: float = 0.70,
        depth_tau_voxels: float = 2.0,
        geometry_stage: str = "shape512",
        stage_label: str = "Shape512",
        stage_slug: str = "shape512",
        selected_mode: str = None,
    ):
        """
        Evaluate how Marigold normals and relative depth would redistribute the
        existing Shape512 multi-view weights on the occupied sparse tokens.

        NORMAL is the existing B1.2N confidence.  DEPTH is a new conservative
        signal: per view, fit z_shape ~= a * depth_marigold + b on currently
        visible SS/S4 surface tokens, then turn residual-in-voxels into a soft
        0.70..1.00 weight.  No token/view is hard-rejected by Marigold depth.

        Returns a full [B,V,R^3,1] DEPTH weight tensor (ones away from occupied
        tokens) and writes JSON + three colored point clouds for NORMAL, DEPTH,
        and BOTH counterfactual redistribution.
        """
        from ..trainers.flow_matching.mixins.image_conditioned_proj import (
            compute_relative_calc_mat,
            project_points_to_image_batch,
        )
        import json
        import time

        source_info = self._shape512_selected_sources(
            views,
            stage=geometry_stage,
        )
        normal_source = source_info["normal_source"]
        depth_source = source_info["depth_source"]
        normal_native = views.get(source_info["normal_key"], None)
        depth_native = views.get(source_info["depth_key"], None)
        alpha_native = views.get(source_info["alpha_key"], None)

        normal_available = normal_native is not None
        depth_available = (depth_native is not None and alpha_native is not None)

        if not normal_available and not depth_available:
            print(
                f"[{stage_label} DIAG] selected native maps unavailable: "
                f"normal={normal_source} ({source_info['normal_key']}), "
                f"depth={depth_source} ({source_info['depth_key']})"
            )
            return None

        device = self.device
        R = int(image_cond_model.grid_resolution)
        R3 = R ** 3

        transform_matrix = views["transform_matrix"].to(
            device=device, dtype=torch.float32
        )
        camera_angle = views["camera_angle_x"].to(
            device=device, dtype=torch.float32
        )
        distance = views["camera_distance"].to(
            device=device, dtype=torch.float32
        )
        mesh_scale = torch.as_tensor(
            views["mesh_scale"], device=device, dtype=torch.float32
        ).reshape(-1)

        B, V = transform_matrix.shape[:2]
        calc_mat = compute_relative_calc_mat(
            transform_matrix,
            distance,
            image_cond_model.proj_grid.front_view_transform_matrix,
        )
        base_grid = image_cond_model.proj_grid.grid_points.to(
            device=device, dtype=torch.float32
        )
        coords_dev = coords.to(device=device).long()

        # Sample the native normal-confidence / depth / alpha maps through the
        # exact same ProjGridMV geometry used by the real extractor.
        normal_grid = torch.ones(
            (B, V, R3), device=device, dtype=torch.float32
        )
        depth_grid = torch.zeros_like(normal_grid)
        alpha_grid = torch.zeros_like(normal_grid)

        for v in range(V):
            if normal_available:
                n_v = normal_native[:, v].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )

                normal_grid[:, v] = image_cond_model.proj_grid(
                    n_v,
                    camera_angle[:, v],
                    distance[:, v],
                    mesh_scale,
                    calc_mat[:, v],
                    BHWC=False,
                )[..., 0].float().clamp_(0.50, 1.0)

                del n_v

            if depth_available:
                d_v = depth_native[:, v].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                a_v = alpha_native[:, v].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )

                depth_grid[:, v] = image_cond_model.proj_grid(
                    d_v,
                    camera_angle[:, v],
                    distance[:, v],
                    mesh_scale,
                    calc_mat[:, v],
                    BHWC=False,
                )[..., 0].float()

                alpha_grid[:, v] = image_cond_model.proj_grid(
                    a_v,
                    camera_angle[:, v],
                    distance[:, v],
                    mesh_scale,
                    calc_mat[:, v],
                    BHWC=False,
                )[..., 0].float().clamp_(0.0, 1.0)

                del d_v, a_v

        depth_weight_grid = torch.ones(
            (B, V, R3), device=device, dtype=torch.float32
        )

        def q(x, p):
            x = x[torch.isfinite(x)]
            if x.numel() == 0:
                return None
            return float(torch.quantile(x.float(), float(p)).item())

        def fmt(x, digits=3):
            return "NA" if x is None else f"{x:.{digits}f}"

        def affine_fit(x, y):
            if x.numel() < 2:
                return None
            xm = x.mean()
            ym = y.mean()
            xc = x - xm
            den = xc.square().sum()
            if float(den.item()) < 1e-12:
                return None
            a = (xc * (y - ym)).sum() / den
            b = ym - a * xm
            return a, b

        report = {
            "version": "SHAPE_GEOMETRY_DIAG_V4",
            "stage": str(geometry_stage),
            "stage_label": str(stage_label),
            "selected_mode": (
                str(selected_mode)
                if selected_mode is not None
                else str(views.get("marigold_shape512_mode", "normal"))
            ),
            "normal_source": normal_source,
            "depth_source": depth_source,
            "grid_resolution": R,
            "depth_floor": float(depth_floor),
            "depth_tau_voxels": float(depth_tau_voxels),
            "batches": [],
        }

        output_dir = os.path.join(
            folder_paths.get_output_directory(), "debug", "pixal3d"
        )
        os.makedirs(output_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")

        for b in range(B):
            sel = coords_dev[:, 0] == b
            c = coords_dev[sel]
            if c.numel() == 0:
                continue

            x, y, z = c[:, 1], c[:, 2], c[:, 3]
            flat = x * (R * R) + y * R + z
            pts_world = base_grid[flat] / mesh_scale[b] / 2.0
            pts3d = pts_world.unsqueeze(0)
            voxel_depth = (
                1.0 / max(R - 1, 1) / float(mesh_scale[b].item())
            )

            batch_report = {
                "batch": int(b),
                "tokens": int(flat.numel()),
                "views": [],
            }

            for v in range(V):
                _, token_depth, proj_valid = project_points_to_image_batch(
                    pts3d,
                    calc_mat[b:b + 1, v],
                    camera_angle[b:b + 1, v],
                    R,
                )
                td = token_depth[0].float()
                pv = proj_valid[0].bool()
                vis = visibility_weights[b, v, flat, 0].float() > 0
                md = depth_grid[b, v, flat]
                ma = alpha_grid[b, v, flat]
                mn = normal_grid[b, v, flat]

                fit_mask = (
                    vis & pv & (ma > 0.50)
                    & torch.isfinite(md) & torch.isfinite(td)
                )

                view_report = {
                    "view": int(v),
                    "visible_tokens": int(vis.sum().item()),
                    "fit_tokens": int(fit_mask.sum().item()),
                }
                if normal_available:
                    view_report.update({
                        "normal_p10": q(mn[vis], 0.10),
                        "normal_p50": q(mn[vis], 0.50),
                        "normal_p90": q(mn[vis], 0.90),
                    })

                if depth_available and int(fit_mask.sum().item()) >= 12:
                    xx = md[fit_mask]
                    yy = td[fit_mask]
                    fit = affine_fit(xx, yy)

                    if fit is not None:
                        a, bb = fit
                        signed = yy - (a * xx + bb)
                        med = signed.median()
                        mad = (signed - med).abs().median()
                        sigma = 1.4826 * mad
                        cut = max(
                            float((3.0 * sigma).item()),
                            0.50 * voxel_depth,
                        )
                        keep = (signed - med).abs() <= cut

                        if int(keep.sum().item()) >= 12:
                            fit2 = affine_fit(xx[keep], yy[keep])
                            if fit2 is not None:
                                a, bb = fit2

                        good = (
                            vis & pv & (ma > 0.50)
                            & torch.isfinite(md) & torch.isfinite(td)
                        )
                        residual = torch.full_like(td, float("nan"))
                        residual[good] = (
                            td[good] - (a * md[good] + bb)
                        ).abs() / max(voxel_depth, 1e-12)

                        dw = torch.ones_like(td)
                        rr = residual[good]
                        dw[good] = (
                            float(depth_floor)
                            + (1.0 - float(depth_floor))
                            * torch.exp(
                                -0.5 * (rr / float(depth_tau_voxels)).square()
                            )
                        )
                        depth_weight_grid[b, v, flat] = dw

                        rp50 = q(rr, 0.50)
                        rp90 = q(rr, 0.90)
                        rp95 = q(rr, 0.95)
                        wp10 = q(dw[good], 0.10)
                        wp50 = q(dw[good], 0.50)
                        wp90 = q(dw[good], 0.90)

                        view_report.update({
                            "affine_a": float(a.item()),
                            "affine_b": float(bb.item()),
                            "residual_vox_p50": rp50,
                            "residual_vox_p90": rp90,
                            "residual_vox_p95": rp95,
                            "depth_weight_p10": wp10,
                            "depth_weight_p50": wp50,
                            "depth_weight_p90": wp90,
                        })

                        print(
                            f"[{stage_label} DEPTH/{depth_source}] b={b} view={v}: "
                            f"fit={int(fit_mask.sum().item())}, "
                            f"a={a.item():.6f}, b={bb.item():.6f}, "
                            f"res_vox p50/90/95="
                            f"{fmt(rp50)}/{fmt(rp90)}/{fmt(rp95)}, "
                            f"weight p10/50/90="
                            f"{fmt(wp10)}/{fmt(wp50)}/{fmt(wp90)}"
                        )

                batch_report["views"].append(view_report)

            base_w = visibility_weights[b, :, flat, 0].float()
            normal_w = normal_grid[b, :, flat]
            depth_w = depth_weight_grid[b, :, flat]

            raw = {
                "base": base_w,
                "normal": base_w * normal_w,
                "depth": base_w * depth_w,
                "both": base_w * normal_w * depth_w,
            }
            probs = {
                name: w / w.sum(dim=0, keepdim=True).clamp_min(1e-8)
                for name, w in raw.items()
            }

            visible_count = (base_w > 0).sum(dim=0)
            multi = visible_count > 1
            batch_report["visibility_histogram"] = {
                str(k): int((visible_count == k).sum().item())
                for k in range(1, V + 1)
            }

            impacts = {}
            top_rows = {}
            p_base = probs["base"]

            for name in ("normal", "depth", "both"):
                p = probs[name]
                tv = 0.5 * (p - p_base).abs().sum(dim=0)
                impacts[name] = tv
                sample = tv[multi] if multi.any() else tv

                tv50 = q(sample, 0.50)
                tv90 = q(sample, 0.90)
                tv99 = q(sample, 0.99)
                frac05 = float((sample >= 0.05).float().mean().item()) if sample.numel() else 0.0
                frac10 = float((sample >= 0.10).float().mean().item()) if sample.numel() else 0.0
                frac20 = float((sample >= 0.20).float().mean().item()) if sample.numel() else 0.0

                entropy = -(
                    p.clamp_min(1e-8) * p.clamp_min(1e-8).log()
                ).sum(dim=0)
                eff = entropy.exp()
                eff_sample = eff[multi] if multi.any() else eff

                batch_report[name] = {
                    "tv_p50": tv50,
                    "tv_p90": tv90,
                    "tv_p99": tv99,
                    "fraction_tv_ge_0_05": frac05,
                    "fraction_tv_ge_0_10": frac10,
                    "fraction_tv_ge_0_20": frac20,
                    "effective_views_mean": float(eff_sample.mean().item()),
                    "mean_view_share": [float(vv) for vv in p.mean(dim=1).tolist()],
                }

                print(
                    f"[{stage_label} {name.upper()} N={normal_source} D={depth_source}] b={b}: "
                    f"multi={int(multi.sum().item())}/{flat.numel()}, "
                    f"TV p50/90/99={fmt(tv50,4)}/{fmt(tv90,4)}/{fmt(tv99,4)}, "
                    f"TV>=.05/.10/.20="
                    f"{100*frac05:.1f}%/{100*frac10:.1f}%/{100*frac20:.1f}%, "
                    f"effective_views={float(eff_sample.mean().item()):.3f}"
                )

                # Keep exact coordinates / shares for the strongest 32 tokens.
                kk = min(32, int(tv.numel()))
                vals, ids = torch.topk(tv, k=kk)
                rows = []
                for val, i in zip(vals.tolist(), ids.tolist()):
                    rows.append({
                        "idx": [int(t) for t in c[i, 1:4].tolist()],
                        "tv": float(val),
                        "visible_views": int(visible_count[i].item()),
                        "base_share": [float(t) for t in p_base[:, i].tolist()],
                        "variant_share": [float(t) for t in p[:, i].tolist()],
                    })
                top_rows[name] = rows

                # PLY impact map: blue = no redistribution, red = strongest.
                tv_cpu = tv.detach().cpu().float()
                scale = q(tv_cpu, 0.99)
                scale = max(scale if scale is not None else 0.0, 1e-6)
                heat = (tv_cpu / scale).clamp(0.0, 1.0).numpy()
                colors = np.zeros((len(heat), 4), dtype=np.uint8)
                colors[:, 0] = (255.0 * heat).astype(np.uint8)
                colors[:, 2] = (255.0 * (1.0 - heat)).astype(np.uint8)
                colors[:, 3] = 255

                ply_path = os.path.join(
                    output_dir,
                    f"{stage_slug}_{name}_impact_b{b}_{stamp}.ply",
                )
                trimesh.points.PointCloud(
                    pts_world.detach().cpu().numpy(), colors=colors
                ).export(ply_path)
                print(f"[{stage_label} DIAG] {name} impact PLY: {ply_path}")

            batch_report["top_tokens"] = top_rows

            if multi.any():
                normal_top = probs["normal"][:, multi].argmax(dim=0)
                depth_top = probs["depth"][:, multi].argmax(dim=0)
                agreement = float((normal_top == depth_top).float().mean().item())
                batch_report["normal_depth_topview_agreement"] = agreement
                print(
                    f"[{stage_label} N×D N={normal_source} D={depth_source}] b={b}: "
                    f"top-view agreement={100.0 * agreement:.1f}%"
                )

            report["batches"].append(batch_report)

        json_path = os.path.join(
            output_dir, f"{stage_slug}_diag_{stamp}.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[{stage_label} DIAG] JSON: {json_path}")

        return depth_weight_grid.unsqueeze(-1).to(
            dtype=visibility_weights.dtype
        )

    def _run_mv_extractor(
        self,
        image_cond_model: nn.Module,
        views: dict,
        visibility_weights=None,
        geometry_stage: str = "auto",
    ):
        device = self.device
        image = self._mv_image_for(views, image_cond_model).to(device)

        # PIXAL_MV_ARBITRARY_V_V1
        V = int(image.shape[1])
        view_names = list(
            views.get(
                "view_names",
                [f"view{v:02d}" for v in range(V)],
            )
        )
        if len(view_names) != V:
            raise ValueError(
                f"MV extractor got V={V} images but {len(view_names)} view_names"
            )
        print(
            f"[Pixal3D MV extractor] image_size={int(image_cond_model.image_size)}, "
            f"grid={int(image_cond_model.grid_resolution)}, V={V}, "
            f"views={view_names}"
        )

        detail_image = None

        if getattr(image_cond_model, "detail_tiles", False):
            detail_image = views.get("native_images", None)

            if detail_image is None:
                print(
                    "[Pixal3D tiles] WARNING: no native_images; "
                    "falling back to stage-resolution image"
                )
        cam_angle = views['camera_angle_x'].to(device=device, dtype=torch.float32)
        dist = views['camera_distance'].to(device=device, dtype=torch.float32)
        transform_matrix = views['transform_matrix'].to(device=device, dtype=torch.float32)
        # mesh_scale is per-object, i.e. [B]; the extractor expands it over views.
        scale = torch.as_tensor(views['mesh_scale'], dtype=torch.float32, device=device).reshape(-1)

        stage_cfg = self._shape_geometry_stage_config(
            image_cond_model,
            views,
            geometry_stage=geometry_stage,
        )

        geometry_confidence = None
        if stage_cfg["stage"] != "off":
            source_info = stage_cfg["source_info"]
            normal_source = source_info["normal_source"]
            mode = stage_cfg["mode"]
            stage_label = stage_cfg["label"]
            geometry_confidence = views.get(
                source_info["normal_key"],
                None,
            )

            if mode in ("off", "depth"):
                if geometry_confidence is not None:
                    print(
                        f"[{stage_label} APPLY] normal=False "
                        f"(mode={mode}); source={normal_source}"
                    )
                geometry_confidence = None

            elif geometry_confidence is None:
                raise RuntimeError(
                    f"{stage_label} normal mode requested source={normal_source}, "
                    f"but {source_info['normal_key']!r} is not attached"
                )

            else:
                print(
                    f"[{stage_label} APPLY] normal=True "
                    f"(mode={mode}); source={normal_source}; "
                    f"native={geometry_confidence.shape[-1]}x"
                    f"{geometry_confidence.shape[-2]}"
                )

        return image_cond_model(
            image,
            camera_angle_x=cam_angle,
            distance=dist,
            mesh_scale=scale,
            transform_matrix=transform_matrix,
            visibility_weights=visibility_weights,
            detail_image=detail_image,
            geometry_confidence=geometry_confidence,
        )

    @torch.no_grad()
    def get_proj_cond_ss_mv(
        self,
        views: dict,
        image_cond_model: nn.Module = None,
    ) -> dict:
        """
        Sparse Structure conditioning with optional soft occlusion weighting.

        MoGe never decides occupancy here. In soft_occlusion mode it may only
        downweight a view whose visible foreground surface lies clearly in front
        of a candidate SS grid point. The same MoGe snapshot is retained for
        diagnostics and the independent post-SS S4-Repair stage.
        """

        print(
            'Getting MultiView Proj Image Cond ...'
        )

        if image_cond_model is None:
            image_cond_model = (
                self.load_pixal3d_mv_image_cond_ss()
            )

        image_cond_model.to(
            self.device
        )

        if views.get(
            "ss_geometry",
            None,
        ) is not None:

            self._prepare_ss_moge_diag_state(
                image_cond_model,
                views,
            )

        else:
            self._ss_moge_diag_state = None

        visibility_weights = None

        if getattr(
            image_cond_model,
            "multiview_fusion",
            None,
        ) == "weighted_average":

            B, V = (
                views[
                    "transform_matrix"
                ].shape[:2]
            )

            R3 = (
                int(
                    image_cond_model
                    .grid_resolution
                )
                ** 3
            )

            # SS_SOFT_OCCLUSION_V1
            ss_visibility_mode = str(
                views.get("ss_visibility_mode", "all_views")
            ).strip().lower()
            occ_params = dict(
                views.get("ss_soft_occlusion_params", {})
            )

            if (
                ss_visibility_mode in (
                    "soft_occlusion",
                    "anchored_soft_occlusion",
                )
                and self._ss_moge_diag_state is not None
            ):
                visibility_weights = self._build_ss_soft_occlusion_weights(
                    image_cond_model,
                    views,
                    self._ss_moge_diag_state,
                    mode=ss_visibility_mode,
                    floor=float(
                        occ_params.get(
                            "floor",
                            0.35,
                        )
                    ),
                    anchored_floor=float(
                        occ_params.get(
                            "anchored_floor",
                            0.50,
                        )
                    ),
                    anchor_band_voxels=float(
                        occ_params.get(
                            "anchor_band_voxels",
                            1.50,
                        )
                    ),
                    margin_voxels=float(
                        occ_params.get(
                            "margin_voxels",
                            1.50,
                        )
                    ),
                    tau_voxels=float(
                        occ_params.get(
                            "tau_voxels",
                            1.50,
                        )
                    ),
                )

            if visibility_weights is None:
                # Exact pre-patch baseline: all-one weighted average.
                visibility_weights = torch.ones(
                    (B, V, R3, 1),
                    device=self.device,
                    dtype=torch.float16,
                )

                if ss_visibility_mode in (
                    "soft_occlusion",
                    "anchored_soft_occlusion",
                ):
                    print(
                        f"[SS-SoftOcc] mode={ss_visibility_mode} "
                        "requested but unavailable; "
                        "falling back to ALL ONES"
                    )
                else:
                    print(
                        "[SS-SoftOcc] mode=all_views; "
                        "using exact all-one baseline"
                    )

        # Never allow Marigold/B1 geometry into SS.
        ss_views = views

        if views.get(
            "geometry_confidence",
            None,
        ) is not None:

            ss_views = dict(
                views
            )

            ss_views[
                "geometry_confidence"
            ] = None

        z_global, z_proj = (
            self._run_mv_extractor(
                image_cond_model,
                ss_views,
                visibility_weights=
                    visibility_weights,
                geometry_stage="off",
            )
        )

        del visibility_weights

        if ss_views is not views:
            del ss_views

        if self.low_vram:
            image_cond_model.cpu()

        return {
            'cond': {
                'global': z_global,
                'proj': z_proj,
            },

            'neg_cond': {
                'global':
                    torch.zeros_like(
                        z_global
                    ),

                'proj':
                    torch.zeros_like(
                        z_proj
                    ),
            },
        }

    @torch.no_grad()
    def get_proj_cond_shape_mv(
        self,
        image_cond_model: nn.Module,
        views: dict,
        coords: torch.Tensor,
        grid_resolution_override: int = None,
        geometry_stage: str = "auto",
    ) -> dict:
        """Multi-view projected conditioning for sparse Shape tokens."""
        print('Getting MultiView Projected Image Cond ...')
        device = self.device
        # See get_proj_cond_ss_mv: these models are built after pipeline.to().
        image_cond_model.to(device)

        # The HR cascade grid resolution floats with the token budget (1536/16=96 and
        # down), so it can differ from what this stage was trained at; swap the grid.
        orig_grid_res = image_cond_model.grid_resolution
        override = (grid_resolution_override is not None
                    and grid_resolution_override != orig_grid_res)
        if override:
            image_cond_model.grid_resolution = grid_resolution_override
            image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
                grid_resolution=grid_resolution_override,
                image_resolution=image_cond_model.proj_grid.image_resolution,
            ).to(device)
        visibility_weights = None

        if getattr(image_cond_model, "multiview_fusion", None) == "weighted_average":
            visibility_weights = self._build_sparse_mv_visibility(
                image_cond_model,
                views,
                coords,
                depth_tolerance_voxels=1.5,
            )

        # SHAPE_HR_GEOMETRY_V1
        # Shape512 keeps the existing behavior.  The Cascade Generator may
        # explicitly request the same re-ranking on the HR shape pass.
        # Texture and SS calls leave geometry_stage off/auto and stay unchanged.
        depth_weights = None
        stage_cfg = self._shape_geometry_stage_config(
            image_cond_model,
            views,
            geometry_stage=geometry_stage,
        )

        if (
            stage_cfg["stage"] != "off"
            and visibility_weights is not None
        ):
            source_info = stage_cfg["source_info"]
            mode = stage_cfg["mode"]
            stage_label = stage_cfg["label"]
            normal_source = source_info["normal_source"]
            depth_source = source_info["depth_source"]

            has_normal = (
                views.get(source_info["normal_key"], None)
                is not None
            )
            has_depth = (
                views.get(source_info["depth_key"], None)
                is not None
                and views.get(source_info["alpha_key"], None)
                is not None
            )

            if mode in ("normal", "normal+depth") and not has_normal:
                raise RuntimeError(
                    f"{stage_label} normal mode requested source={normal_source}, "
                    f"but {source_info['normal_key']!r} is not attached"
                )

            need_depth_diag = (
                # Preserve the existing Shape512 counterfactual diagnostics.
                stage_cfg["stage"] == "shape512"
                or mode in ("depth", "normal+depth")
            )

            if need_depth_diag and (has_normal or has_depth):
                depth_weights = self._shape512_marigold_diagnostics(
                    image_cond_model,
                    views,
                    coords,
                    visibility_weights,
                    depth_floor=stage_cfg["depth_floor"],
                    depth_tau_voxels=stage_cfg["depth_tau_voxels"],
                    geometry_stage=stage_cfg["stage"],
                    stage_label=stage_label,
                    stage_slug=stage_cfg["slug"],
                    selected_mode=mode,
                )

            if mode in ("depth", "normal+depth"):
                if not has_depth:
                    raise RuntimeError(
                        f"{stage_label} depth mode requested source={depth_source}, "
                        f"but selected depth/alpha maps are not attached"
                    )
                if depth_weights is None:
                    raise RuntimeError(
                        f"{stage_label} depth mode requested, but diagnostics "
                        "could not build depth weights"
                    )

                visibility_weights.mul_(depth_weights)
                print(
                    f"[{stage_label} APPLY] depth=True "
                    f"(mode={mode}); source={depth_source}; "
                    f"tau={stage_cfg['depth_tau_voxels']:.3f} vox"
                )
            else:
                print(
                    f"[{stage_label} APPLY] depth=False "
                    f"(mode={mode}); source={depth_source}"
                )

        elif stage_cfg["stage"] != "off":
            print(
                f"[{stage_cfg['label']} DIAG] skipped: "
                "visibility weights unavailable"
            )

        z_global, z_proj = self._run_mv_extractor(
            image_cond_model,
            views,
            visibility_weights=visibility_weights,
            geometry_stage=geometry_stage,
        )

        if depth_weights is not None:
            del depth_weights
        del visibility_weights

        B = z_global.shape[0]
        grid_res = image_cond_model.grid_resolution
        b_idx = coords[:, 0].long()
        x_idx = coords[:, 1].long()
        y_idx = coords[:, 2].long()
        z_idx = coords[:, 3].long()
        z_proj_grid = z_proj.reshape(B, grid_res, grid_res, grid_res, -1)
        z_proj_sparse = z_proj_grid[b_idx, x_idx, y_idx, z_idx]
        z_proj_st = SparseTensor(feats=z_proj_sparse, coords=coords)

        if override:
            image_cond_model.grid_resolution = orig_grid_res
            image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
                grid_resolution=orig_grid_res,
                image_resolution=image_cond_model.proj_grid.image_resolution,
            ).to(device)

        if self.low_vram:
            image_cond_model.cpu()
            self._cleanup_cuda()
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(
                f"[VRAM] after MV cond offload: "
                f"{free / 1024**3:.2f} GiB free / "
                f"{total / 1024**3:.2f} GiB total"
            )
        return {
            'cond': {'global': z_global, 'proj': z_proj_st},
            'neg_cond': {
                'global': torch.zeros_like(z_global),
                'proj': SparseTensor(feats=torch.zeros_like(z_proj_sparse), coords=coords),
            },
        }





    def _ss_coords_signature(
        self,
        coords: torch.Tensor,
        label: str,
    ):
        import hashlib

        rows = [
            tuple(int(x) for x in row)
            for row in (
                coords
                .detach()
                .cpu()
                .int()
                .tolist()
            )
        ]

        rows.sort()

        payload = "\n".join(
            ",".join(
                str(x)
                for x in row
            )
            for row in rows
        ).encode("utf-8")

        digest = hashlib.sha256(
            payload
        ).hexdigest()[:20]

        print(
            f"[SS-Coords] {label}: "
            f"count={len(rows)}, "
            f"sha256={digest}"
        )

        return rows, digest


    @torch.no_grad()
    def _repair_ss_with_moge(
        self,
        coords: torch.Tensor,
        resolution: int,
        state: dict = None,
        support_threshold_voxels: float = 2.0,
        evidence_tau_voxels: float = 1.25,
        min_support_views: int = 2,
        min_evidence_sum: float = 0.90,
    ) -> torch.Tensor:
        """
        S4 conservative post-SS topology repair.

        Stock Pixal SS already exists.

        MoGe:
          - cannot delete voxels;
          - cannot move voxels;
          - cannot alter DINO fusion;
          - can only confirm a one-step missing local bridge.

        Exactly one repair iteration is performed.
        """

        import json
        import os
        import time

        import numpy as np
        import torch
        import torch.nn.functional as F
        import trimesh
        import folder_paths

        from scipy.ndimage import (
            binary_dilation,
            generate_binary_structure,
            label,
        )

        from ..trainers.flow_matching.mixins.image_conditioned_proj import (
            project_points_to_image_batch,
        )

        if state is None:
            state = getattr(
                self,
                "_ss_moge_diag_state",
                None,
            )

        print(
            "[SS-Repair S4 state] "
            f"explicit/local state present={state is not None}"
        )

        if state is None:
            print(
                "[SS-Repair S4] no MoGe diagnostic state; "
                "stock SS returned unchanged"
            )
            return coords

        points = state.get("points")
        moge_valid = state.get("mask")
        alphas = state.get("alphas")
        calc_mat = state.get("calc_mat")

        camera_angle = state.get(
            "camera_angle",
            state.get("camera_angle_x"),
        )

        mesh_scale_state = state.get(
            "mesh_scale",
            1.0,
        )

        if any(
            x is None
            for x in (
                points,
                moge_valid,
                alphas,
                calc_mat,
                camera_angle,
            )
        ):
            print(
                "[SS-Repair S4] incomplete diagnostic state; "
                "stock SS returned unchanged"
            )
            return coords

        coords = (
            coords
            .detach()
            .cpu()
            .int()
        )

        if coords.numel() == 0:
            return coords

        R = int(resolution)

        B = max(
            int(coords[:, 0].max()) + 1,
            int(points.shape[0]),
        )

        V = int(points.shape[1])
        H = int(points.shape[2])
        W = int(points.shape[3])

        # PIXAL_MV_ARBITRARY_V_V1
        view_names = list(
            state.get(
                "view_names",
                [f"view{v:02d}" for v in range(V)],
            )
        )
        if len(view_names) != V:
            raise ValueError(
                f"[SS-Repair S4] state has V={V} but {len(view_names)} view_names"
            )

        if H != W:
            raise ValueError(
                f"[SS-Repair S4] MoGe map is not square: "
                f"{H}x{W}"
            )

        calc_mat = (
            torch.as_tensor(
                calc_mat,
                dtype=torch.float32,
            )
            .cpu()
        )

        camera_angle = (
            torch.as_tensor(
                camera_angle,
                dtype=torch.float32,
            )
            .cpu()
        )

        if camera_angle.ndim == 1:
            camera_angle = (
                camera_angle.unsqueeze(0)
            )

        mesh_scale_t = (
            torch.as_tensor(
                mesh_scale_state,
                dtype=torch.float32,
            )
            .reshape(-1)
        )

        if mesh_scale_t.numel() == 1:
            mesh_scale_t = (
                mesh_scale_t.expand(B)
            )

        alphas = (
            torch.as_tensor(
                alphas,
                dtype=torch.float32,
            )
            .cpu()
        )

        # Foreground mask at native MoGe resolution.
        alpha_native = F.interpolate(
            alphas.reshape(
                alphas.shape[0] * alphas.shape[1],
                1,
                alphas.shape[-2],
                alphas.shape[-1],
            ),
            size=(H, W),
            mode="nearest",
        ).reshape(
            alphas.shape[0],
            alphas.shape[1],
            H,
            W,
        )

        foreground = (
            alpha_native > 0.5
        )

        # ========================================================
        # Per-view radial gauge calibration.
        #
        # Used ONLY to compare candidate depth with MoGe.
        # It does not alter stock SS.
        # ========================================================

        per_view_scale = torch.empty(
            (B, V),
            dtype=torch.float32,
        )

        stride = max(
            1,
            int(max(H, W) // 512),
        )

        origin_world = torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            dtype=torch.float32,
        )

        for b in range(B):
            for v in range(V):

                p = points[
                    b,
                    v,
                    ::stride,
                    ::stride,
                ].float()

                vm = moge_valid[
                    b,
                    v,
                    ::stride,
                    ::stride,
                ].bool()

                fg = foreground[
                    b,
                    v,
                    ::stride,
                    ::stride,
                ]

                z = p[..., 2]

                valid = (
                    fg
                    & vm
                    & torch.isfinite(p).all(dim=-1)
                    & torch.isfinite(z)
                    & (z > 0)
                )

                zv = z[valid]

                if zv.numel() < 64:
                    raise RuntimeError(
                        f"[SS-Repair S4] b={b} v={v}: "
                        "too few valid foreground MoGe samples"
                    )

                z01 = torch.quantile(
                    zv,
                    0.01,
                )

                z99 = torch.quantile(
                    zv,
                    0.99,
                )

                moge_center_z = (
                    z01 + z99
                ) * 0.5

                w2c = torch.linalg.inv(
                    calc_mat[b, v].float()
                )

                origin_cam = (
                    origin_world
                    @ w2c.T
                )[:3]

                # Blender camera -> OpenCV depth.
                target_z = -origin_cam[2]

                scale_v = (
                    target_z
                    / moge_center_z
                )

                per_view_scale[
                    b,
                    v,
                ] = scale_v

                print(
                    f"[SS-Repair S4 scale] "
                    f"b={b} v={v} {view_names[v]}: "
                    f"moge_center_z={float(moge_center_z):.6f}, "
                    f"target_z={float(target_z):.6f}, "
                    f"scale={float(scale_v):.8f}"
                )

        # Same lattice -> Blender alignment as ProjGrid.
        grid_rotation = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        output_dir = os.path.join(
            folder_paths.get_output_directory(),
            "debug",
            "pixal3d",
        )

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        stamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        report = {
            "mode":
                "S4 conservative post-SS bridge repair",


            "resolution":
                R,

            "support_threshold_voxels":
                float(
                    support_threshold_voxels
                ),

            "evidence_tau_voxels":
                float(
                    evidence_tau_voxels
                ),

            "min_support_views":
                int(
                    min_support_views
                ),

            "min_evidence_sum":
                float(
                    min_evidence_sum
                ),

            "batches": [],
        }

        repaired_batches = []

        for b in range(B):

            batch_coords = coords[
                coords[:, 0] == b
            ][:, 1:4].long()

            if batch_coords.numel() == 0:
                continue

            occ = np.zeros(
                (R, R, R),
                dtype=np.bool_,
            )

            idx_np = (
                batch_coords.numpy()
            )

            valid_idx = (
                (idx_np >= 0).all(axis=1)
                & (idx_np < R).all(axis=1)
            )

            idx_np = idx_np[
                valid_idx
            ]

            occ[
                idx_np[:, 0],
                idx_np[:, 1],
                idx_np[:, 2],
            ] = True

            # ====================================================
            # Step 1: topology candidates
            # ====================================================

            near_surface = (
                binary_dilation(
                    occ,
                    structure=np.ones(
                        (3, 3, 3),
                        dtype=bool,
                    ),
                    iterations=1,
                )
                & ~occ
            )

            # Ignore cube boundary.
            near_surface[0, :, :] = False
            near_surface[-1, :, :] = False
            near_surface[:, 0, :] = False
            near_surface[:, -1, :] = False
            near_surface[:, :, 0] = False
            near_surface[:, :, -1] = False

            raw_candidates = (
                np.argwhere(
                    near_surface
                )
            )

            bridge_candidates = []
            bridge_meta = []

            # Determine whether occupied neighbours around the
            # candidate form separate local pieces.
            local_structure = (
                generate_binary_structure(
                    3,
                    1,
                )
            )

            for c in raw_candidates:

                x, y, z = (
                    int(c[0]),
                    int(c[1]),
                    int(c[2]),
                )

                local = occ[
                    x-1:x+2,
                    y-1:y+2,
                    z-1:z+2,
                ].copy()

                local[1, 1, 1] = False

                neighbour_pos = (
                    np.argwhere(local)
                    - np.array(
                        [1, 1, 1]
                    )
                )

                n_neigh = int(
                    neighbour_pos.shape[0]
                )

                # Avoid both trivial contact and dense cavities.
                if (
                    n_neigh < 2
                    or n_neigh > 10
                ):
                    continue

                _, n_components = label(
                    local,
                    structure=local_structure,
                )

                # Ordinary outer surface normally gives one
                # connected local patch. A missing bridge gives
                # two or more.
                if int(n_components) < 2:
                    continue

                vec = (
                    neighbour_pos.astype(
                        np.float32
                    )
                )

                norms = np.linalg.norm(
                    vec,
                    axis=1,
                    keepdims=True,
                )

                vec = (
                    vec
                    / np.maximum(
                        norms,
                        1e-8,
                    )
                )

                dots = (
                    vec @ vec.T
                )

                np.fill_diagonal(
                    dots,
                    1.0,
                )

                min_dot = float(
                    dots.min()
                )

                # There must be occupied matter on genuinely
                # different/opposing sides of the empty cell.
                if min_dot > -0.25:
                    continue

                bridge_candidates.append(
                    [x, y, z]
                )

                bridge_meta.append(
                    {
                        "idx":
                            [x, y, z],

                        "neighbours":
                            n_neigh,

                        "local_components":
                            int(
                                n_components
                            ),

                        "min_direction_dot":
                            min_dot,
                    }
                )

            if bridge_candidates:

                cand_idx = torch.tensor(
                    bridge_candidates,
                    dtype=torch.float32,
                )

            else:

                cand_idx = torch.empty(
                    (0, 3),
                    dtype=torch.float32,
                )

            print(
                f"[SS-Repair S4] b={b}: "
                f"stock={int(occ.sum())}, "
                f"near_surface_empty={int(near_surface.sum())}, "
                f"bridge_candidates={int(cand_idx.shape[0])}"
            )

            if cand_idx.shape[0] == 0:

                repaired_batches.append(
                    torch.cat(
                        [
                            torch.full(
                                (
                                    batch_coords.shape[0],
                                    1,
                                ),
                                b,
                                dtype=torch.int32,
                            ),
                            batch_coords.int(),
                        ],
                        dim=1,
                    )
                )

                report["batches"].append(
                    {
                        "batch": b,
                        "stock_voxels":
                            int(occ.sum()),
                        "bridge_candidates":
                            0,
                        "accepted":
                            0,
                        "candidates":
                            [],
                    }
                )

                continue

            mesh_scale = float(
                mesh_scale_t[
                    b
                ].item()
            )

            voxel_size = (
                1.0
                / float(R - 1)
                / mesh_scale
            )

            # Decoder indices -> canonical Pixal world.
            lattice = (
                -1.0
                + 2.0
                * cand_idx
                / float(R - 1)
            )

            cand_world = (
                lattice
                @ grid_rotation.T
                / mesh_scale
                / 2.0
            )

            N = int(
                cand_world.shape[0]
            )

            residual_vox = torch.full(
                (N, V),
                float("nan"),
                dtype=torch.float32,
            )

            support = torch.zeros(
                (N, V),
                dtype=torch.bool,
            )

            evidence = torch.zeros(
                (N, V),
                dtype=torch.float32,
            )

            cand_world_dev = (
                cand_world.to(
                    self.device,
                    dtype=torch.float32,
                )
            )

            cand_h_dev = torch.cat(
                [
                    cand_world_dev,

                    torch.ones(
                        (N, 1),
                        device=self.device,
                        dtype=torch.float32,
                    ),
                ],
                dim=-1,
            )

            # ====================================================
            # Step 2: independent MoGe evidence per view
            # ====================================================

            for v in range(V):

                calc_v = calc_mat[
                    b:b+1,
                    v,
                ].to(
                    self.device,
                    dtype=torch.float32,
                )

                angle_v = camera_angle[
                    b:b+1,
                    v,
                ].to(
                    self.device,
                    dtype=torch.float32,
                )

                # Project candidates directly into native MoGe map.
                pix, _, proj_valid = (
                    project_points_to_image_batch(
                        cand_world_dev.unsqueeze(0),
                        calc_v,
                        angle_v,
                        W,
                    )
                )

                pix = pix[0]
                proj_valid = (
                    proj_valid[0]
                    .cpu()
                )

                ix = (
                    pix[:, 0]
                    .round()
                    .long()
                    .clamp(
                        0,
                        W - 1,
                    )
                    .cpu()
                )

                iy = (
                    pix[:, 1]
                    .round()
                    .long()
                    .clamp(
                        0,
                        H - 1,
                    )
                    .cpu()
                )

                q = points[
                    b,
                    v,
                ][
                    iy,
                    ix,
                ].float()

                vm = moge_valid[
                    b,
                    v,
                ][
                    iy,
                    ix,
                ].bool()

                fg = foreground[
                    b,
                    v,
                ][
                    iy,
                    ix,
                ].bool()

                q_valid = (
                    vm
                    & fg
                    & torch.isfinite(q)
                    .all(dim=-1)
                    & torch.isfinite(
                        q[:, 2]
                    )
                    & (q[:, 2] > 0)
                )

                # Candidate camera-space depth.
                w2c = torch.linalg.inv(
                    calc_v[0].float()
                )

                p_cam_blender = (
                    cand_h_dev
                    @ w2c.T
                )[:, :3]

                p_z_cv = (
                    -p_cam_blender[:, 2]
                ).cpu()

                # Only MoGe radial gauge is normalised.
                q_z = (
                    q[:, 2]
                    * per_view_scale[
                        b,
                        v,
                    ]
                )

                dz_vox = (
                    torch.abs(
                        p_z_cv
                        - q_z
                    )
                    / voxel_size
                )

                usable = (
                    proj_valid
                    & q_valid
                    & torch.isfinite(
                        dz_vox
                    )
                )

                residual_vox[
                    usable,
                    v,
                ] = dz_vox[
                    usable
                ]

                support_v = (
                    usable
                    & (
                        dz_vox
                        <= float(
                            support_threshold_voxels
                        )
                    )
                )

                support[
                    support_v,
                    v,
                ] = True

                ev = torch.zeros(
                    N,
                    dtype=torch.float32,
                )

                ev[
                    usable
                ] = torch.exp(
                    -0.5
                    * (
                        dz_vox[
                            usable
                        ]
                        / float(
                            evidence_tau_voxels
                        )
                    ) ** 2
                )

                evidence[
                    :,
                    v,
                ] = ev

            support_count = (
                support.sum(dim=1)
            )

            evidence_sum = (
                evidence.sum(dim=1)
            )

            accepted = (
                (
                    support_count
                    >= int(
                        min_support_views
                    )
                )
                & (
                    evidence_sum
                    >= float(
                        min_evidence_sum
                    )
                )
            )

            print(
                f"[SS-Repair S4 PROD] "
                f"accepted={int(accepted.sum().item())}/"
                f"{int(accepted.shape[0])}"
            )

            accepted_idx = (
                cand_idx[
                    accepted
                ]
                .round()
                .long()
            )

            added_count = int(
                accepted_idx.shape[0]
            )

            support_hist = {
                n: int(
                    (
                        support_count
                        == n
                    ).sum().item()
                )
                for n in range(
                    V + 1
                )
            }

            print(
                f"[SS-Repair S4] b={b}: "
                f"accepted={added_count}/{N}, "
                f"support_counts={support_hist}"
            )

            # ====================================================
            # JSON candidate report
            # ====================================================

            candidate_report = []

            for i, meta in enumerate(
                bridge_meta
            ):

                rv = []

                for v in range(V):

                    value = (
                        residual_vox[
                            i,
                            v,
                        ]
                    )

                    if torch.isfinite(
                        value
                    ):
                        rv.append(
                            float(
                                value.item()
                            )
                        )
                    else:
                        rv.append(
                            None
                        )

                item = dict(meta)

                item.update(
                    {
                        "depth_residual_vox":
                            rv,

                        "support_views": [
                            int(v)
                            for v in range(V)
                            if bool(
                                support[
                                    i,
                                    v,
                                ]
                            )
                        ],

                        "support_count":
                            int(
                                support_count[
                                    i
                                ].item()
                            ),

                        "evidence_sum":
                            float(
                                evidence_sum[
                                    i
                                ].item()
                            ),

                        "accepted":
                            bool(
                                accepted[
                                    i
                                ].item()
                            ),
                    }
                )

                candidate_report.append(
                    item
                )

            report["batches"].append(
                {
                    "batch": b,

                    "stock_voxels":
                        int(
                            occ.sum()
                        ),

                    "near_surface_empty":
                        int(
                            near_surface.sum()
                        ),

                    "bridge_candidates":
                        N,

                    "accepted":
                        added_count,

                    "support_histogram":
                        support_hist,

                    "per_view_scale": [
                        float(
                            per_view_scale[
                                b,
                                v,
                            ].item()
                        )
                        for v in range(V)
                    ],

                    "candidates":
                        candidate_report,
                }
            )

            # ====================================================
            # Debug overlay:
            #
            # grey    = stock B
            # yellow  = topology candidate rejected by MoGe
            # magenta = voxel actually added by S4
            # ====================================================

            stock_idx = torch.tensor(
                idx_np,
                dtype=torch.float32,
            )

            stock_lattice = (
                -1.0
                + 2.0
                * stock_idx
                / float(R - 1)
            )

            stock_world = (
                stock_lattice
                @ grid_rotation.T
                / mesh_scale
                / 2.0
            )

            rejected_world = (
                cand_world[
                    ~accepted
                ]
            )

            accepted_world = (
                cand_world[
                    accepted
                ]
            )

            vertices = [
                stock_world.numpy()
            ]

            colors = [
                np.tile(
                    np.array(
                        [
                            170,
                            170,
                            170,
                            255,
                        ],
                        dtype=np.uint8,
                    )[None, :],
                    (
                        stock_world.shape[0],
                        1,
                    ),
                )
            ]

            if rejected_world.shape[0]:

                vertices.append(
                    rejected_world.numpy()
                )

                colors.append(
                    np.tile(
                        np.array(
                            [
                                255,
                                210,
                                40,
                                255,
                            ],
                            dtype=np.uint8,
                        )[None, :],
                        (
                            rejected_world.shape[0],
                            1,
                        ),
                    )
                )

            if accepted_world.shape[0]:

                vertices.append(
                    accepted_world.numpy()
                )

                colors.append(
                    np.tile(
                        np.array(
                            [
                                255,
                                40,
                                220,
                                255,
                            ],
                            dtype=np.uint8,
                        )[None, :],
                        (
                            accepted_world.shape[0],
                            1,
                        ),
                    )
                )

            overlay_path = os.path.join(
                output_dir,
                f"ss_repair_s4_overlay_"
                f"{stamp}_b{b}.ply",
            )

            trimesh.points.PointCloud(
                np.concatenate(
                    vertices,
                    axis=0,
                ).astype(
                    np.float32
                ),
                colors=np.concatenate(
                    colors,
                    axis=0,
                ).astype(
                    np.uint8
                ),
            ).export(
                overlay_path
            )

            print(
                f"[SS-Repair S4] "
                f"overlay PLY: {overlay_path}"
            )

            # ====================================================
            # Stock + accepted voxels.
            # ====================================================

            base_out = torch.cat(
                [
                    torch.full(
                        (
                            batch_coords.shape[0],
                            1,
                        ),
                        b,
                        dtype=torch.int32,
                    ),

                    batch_coords.int(),
                ],
                dim=1,
            )

            if added_count:

                add_out = torch.cat(
                    [
                        torch.full(
                            (
                                added_count,
                                1,
                            ),
                            b,
                            dtype=torch.int32,
                        ),

                        accepted_idx.int(),
                    ],
                    dim=1,
                )

                batch_out = torch.cat(
                    [
                        base_out,
                        add_out,
                    ],
                    dim=0,
                )

            else:

                batch_out = base_out

            repaired_batches.append(
                batch_out
            )

        json_path = os.path.join(
            output_dir,
            f"ss_repair_s4_"
            f"{stamp}.json",
        )

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                report,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"[SS-Repair S4] "
            f"JSON report: {json_path}"
        )

        if not repaired_batches:
            return coords

        return torch.cat(
            repaired_batches,
            dim=0,
        ).int()


    @torch.no_grad()
    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
        fill_holes: bool = False,
        hole_structure: int = 1,
        hole_iterations: int = 1,
        hole_fill_algorithm: str = "remove_small_holes",
        keep_only_shell: bool = False,
        verbose: bool = False,
        dino_lock: float = 0.0,
        dino_substeps: int = 2,
        dino_foundation_cap: float = 0.92
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        if self.low_vram:
            cond = self._cond_to(cond, self.device)                         
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=verbose,
            tqdm_desc="Sampling sparse structure",
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap
        ).samples
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()        
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)
        decoded = decoder(z_s) > 0
        if self.low_vram:
            decoder.cpu()
            self._cleanup_cuda()
            
        # if resolution != decoded.shape[2]:
            # ratio = decoded.shape[2] // resolution
            # decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        if resolution != decoded.shape[2]:
            if resolution < decoded.shape[2]:
                ratio = decoded.shape[2] // resolution
                decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
            else:
                decoded = torch.nn.functional.interpolate(decoded.float(), size=(resolution, resolution, resolution), mode='nearest') > 0.5

        # Optionally fill holes in the sparse voxel grid using the selected algorithm
        if fill_holes:
            try:
                from scipy.ndimage import binary_closing, label, binary_fill_holes
                arr = decoded.cpu().numpy()
                if arr.ndim == 5:
                    arr = arr[:, 0]
                closed = np.zeros_like(arr)
                for b in range(arr.shape[0]):
                    filled = arr[b].astype(np.bool_)
                    inv = ~filled
                    labeled, num_features = label(inv)
                    border_mask = np.zeros_like(inv)
                    border_mask[0,:,:] = border_mask[-1,:,:] = 1
                    border_mask[:,0,:] = border_mask[:,-1,:] = 1
                    border_mask[:,:,0] = border_mask[:,:,-1] = 1
                    border_labels = np.unique(labeled[border_mask==1])
                    holes = np.isin(labeled, border_labels, invert=True) & (labeled > 0)
                    n_holes = np.unique(labeled[holes]).size
                    print(f"[Sparse HoleFill] Batch {b}: Found {n_holes} holes before filling.")
                    if hole_fill_algorithm == "morphological_closing":
                        closed[b] = binary_closing(arr[b], structure=np.ones((hole_structure,)*3), iterations=hole_iterations)
                    elif hole_fill_algorithm == "flood_fill":
                        # Robust structure-preserving hole filling:
                        # 1. Morphological closing to connect small gaps
                        # 2. Fill internal holes
                        # 3. Keep only the largest connected component
                        from scipy.ndimage import binary_closing, label, binary_fill_holes
                        # Step 1: Morphological closing (small structure, 1 iter)
                        closed1 = binary_closing(arr[b], structure=np.ones((hole_structure,)*3), iterations=hole_iterations)
                        # Step 2: Fill internal holes
                        filled = binary_fill_holes(closed1)
                        # Step 3: Keep only the largest connected component
                        labeled, num = label(filled)
                        if num > 0:
                            sizes = np.bincount(labeled.ravel())
                            sizes[0] = 0  # background
                            largest = sizes.argmax()
                            closed[b] = (labeled == largest)
                        else:
                            closed[b] = filled                     
                    elif hole_fill_algorithm == "remove_small_holes":
                        # Remove small holes by area (2D slices)
                        from skimage.morphology import remove_small_holes
                        # Apply per-slice (z axis)
                        temp = np.copy(arr[b])
                        for z in range(temp.shape[0]):
                            temp[z] = remove_small_holes(temp[z].astype(bool), area_threshold=hole_structure**2)
                        closed[b] = temp
                    else:
                        print(f"[Sparse HoleFill] Unknown algorithm: {hole_fill_algorithm}, skipping.")
                        closed[b] = arr[b]
                    # Count holes after filling
                    filled2 = closed[b].astype(np.bool_)
                    inv2 = ~filled2
                    labeled2, num_features2 = label(inv2)
                    border_labels2 = np.unique(labeled2[border_mask==1])
                    holes2 = np.isin(labeled2, border_labels2, invert=True) & (labeled2 > 0)
                    n_holes2 = np.unique(labeled2[holes2]).size
                    print(f"[Sparse HoleFill] Batch {b}: {n_holes-n_holes2} holes filled, {n_holes2} remain after filling.")

                    # Optionally remove deeply interior voxels, keeping surface and near-surface structure
                    if keep_only_shell:
                        from scipy.ndimage import binary_erosion

                        filled = closed[b].astype(np.bool_)
                        before_count = int(filled.sum())
                        struct = np.ones((3, 3, 3), dtype=bool)
                        # Erode twice: only voxels surviving 2 erosion passes are >=2 layers deep
                        # This preserves thin structures (e.g. necks with 3x3 cross-section)
                        eroded = binary_erosion(filled, structure=struct, border_value=0)
                        eroded = binary_erosion(eroded, structure=struct, border_value=0)
                        # Remove only deeply interior voxels (>=2 layers from any surface)
                        shell = filled & ~eroded
                        closed[b] = shell
                        after_count = int(shell.sum())
                        if verbose:
                            print(f"[Sparse Shell] Batch {b}: {before_count} -> {after_count} voxels (removed {before_count - after_count} deeply interior)")

                decoded = torch.from_numpy(closed).to(decoded.device)


                # Debug: print tensor info before extracting coordinates
                if verbose:
                    print(f"[Sparse HoleFill] decoded shape: {decoded.shape}, dtype: {decoded.dtype}, device: {decoded.device}")
                    print(f"[Sparse HoleFill] decoded min: {decoded.min().item()}, max: {decoded.max().item()}, unique: {torch.unique(decoded)}")
                # Safety: ensure tensor is contiguous and on CPU for argwhere
                decoded = decoded.contiguous().cpu()

            except ImportError:
                print("[Warning] scipy or skimage not installed, skipping hole filling.")
            except Exception as e:
                print(f"[Warning] Hole filling failed: {e}")

            try:
                coords = torch.argwhere(decoded)[:, [0, 1, 2, 3]].int()
            except Exception as e:
                print(f"[Sparse HoleFill] Error in torch.argwhere: {e}")
                raise

            if verbose:
                print(f"[Sparse HoleFill] coords shape: {coords.shape}, min: {coords.min(dim=0).values.tolist() if coords.numel()>0 else 'empty'}, max: {coords.max(dim=0).values.tolist() if coords.numel()>0 else 'empty'}")

            if coords.numel() == 0:
                raise RuntimeError("No voxels remain after hole filling/shell extraction. The mask is empty. Adjust your input, mask, or hole filling parameters.")
        else:
            coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        coords = coords.cpu()

        # SS_ANCHORED_SOFT_OCCLUSION_V2
        # Exact whole-grid signature BEFORE S4.
        stock_ss_rows, stock_ss_hash = (
            self._ss_coords_signature(
                coords,
                "stock_pre_s4",
            )
        )
        stock_ss_snapshot = coords.clone()

        # ------------------------------------------------------
        # Preserve the MoGe snapshot locally for S4.
        #
        # Diagnostic cleanup is allowed to clear the attribute,
        # but cannot invalidate this Python reference.
        # ------------------------------------------------------
        ss_moge_repair_state = getattr(
            self,
            "_ss_moge_diag_state",
            None,
        )

        print(
            "[SS-Repair S4 state] "
            f"local snapshot captured="
            f"{ss_moge_repair_state is not None}"
        )

        # MoGe diagnostic is deliberately executed only AFTER
        # the baseline SS has already been sampled.
        if getattr(
            self,
            "_ss_moge_diag_state",
            None,
        ) is not None:

            try:
                self._diagnose_ss_moge_against_coords(
                    coords,
                    resolution,
                )

            except Exception as e:
                import traceback

                print(
                    f"[SS-MoGe DIAG] "
                    f"FAILED: {e}"
                )

                traceback.print_exc()

            finally:
                # Prevent accidental reuse on the next prompt.
                self._ss_moge_diag_state = None

        del decoded
        del z_s
        if self.low_vram:
            cond = self._cond_cpu(cond)
            self._cleanup_cuda()
        # S4 is strictly post-SS.
        # Stock Pixal SS and existing diagnostics have already run.
        try:
            coords = self._repair_ss_with_moge(
                coords,
                resolution,
                state=ss_moge_repair_state,
            )

        except Exception as e:
            import traceback

            print(
                f"[SS-Repair S4] FAILED: {e}; "
                "stock SS returned unchanged"
            )

            traceback.print_exc()

        # SS_ANCHORED_SOFT_OCCLUSION_V2
        # Exact whole-grid signature AFTER S4 plus explicit diff.
        post_s4_rows, post_s4_hash = (
            self._ss_coords_signature(
                coords,
                "post_s4",
            )
        )

        stock_set = set(stock_ss_rows)
        post_set = set(post_s4_rows)

        s4_added = sorted(
            post_set - stock_set
        )

        s4_removed = sorted(
            stock_set - post_set
        )

        print(
            f"[SS-Coords] S4 diff: "
            f"added={len(s4_added)}, "
            f"removed={len(s4_removed)}, "
            f"stock_hash={stock_ss_hash}, "
            f"post_hash={post_s4_hash}"
        )

        if len(s4_added) <= 32:
            print(
                f"[SS-Coords] S4 added coords="
                f"{s4_added}"
            )
        else:
            print(
                f"[SS-Coords] S4 added coords "
                f"first32={s4_added[:32]}"
            )

        if s4_removed:
            print(
                f"[SS-Coords] WARNING: "
                f"S4 removed coords="
                f"{s4_removed[:32]}"
            )

        del stock_ss_snapshot

        # Release large CPU MoGe point maps after S4.
        ss_moge_repair_state = None

        return coords

    @torch.no_grad()
    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
        verbose: bool = True,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        if self.low_vram:
            cond = self._cond_to(cond, self.device)

        coords_dev = coords.to(self.device)                         
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords_dev,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=verbose,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()                                

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        del coords_dev
        if self.low_vram:
            cond = self._cond_cpu(cond)
            self._cleanup_cuda()

        return slat
    

    @torch.no_grad()
    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
        use_tiled: bool = True,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        
        self.load_shape_slat_decoder()
        
        self.models['shape_slat_decoder'].set_resolution(resolution)
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        ret = self.models['shape_slat_decoder'](slat, return_subs=True, useTiled=use_tiled)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
            self._cleanup_cuda()                
        
        if not self.keep_models_loaded:        
            self.unload_shape_slat_decoder()
            
        return ret
    

    
    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
        use_tiled: bool = True,
    ) -> List[MeshWithVoxel]:
        """Decode a shape latent into the mesh-only Orbit representation.

        The ``tex_slat`` argument is intentionally retained for socket/call compatibility
        with visualbruno's original ``Trellis2DecodeLatents`` node. OrbitTo3D does not
        ship texture generation or texture decoding; a non-None texture latent is rejected
        explicitly rather than being silently ignored.
        """
        if tex_slat is not None:
            raise RuntimeError(
                "ComfyUI-OrbitTo3D is mesh-only: texture_slat decoding is not supported. "
                "Use texture-capable nodes/pipelines from the original ComfyUI-Trellis2 package."
            )

        meshes, _subs = self.decode_shape_slat(shape_slat, resolution, use_tiled=use_tiled)
        if self.low_vram:
            self._cleanup_cuda()

        out_mesh = []
        for m in meshes:
            m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin=[-0.5, -0.5, -0.5],
                    voxel_size=1 / resolution,
                    coords=None,
                    attrs=None,
                    voxel_shape=None,
                    layout={},
                )
            )
        return out_mesh
    
            
    def GetSamplerName(self, sampler):
        if sampler == 'euler':
            return 'Euler'
        elif sampler == 'rk4':
            return 'RK4'
        elif sampler == 'rk5':
            return 'RK5'
        elif sampler == 'heun':
            return 'Heun'
        else:
            return 'Euler'       
            

            
                       








        

    # Reorientations that map an incoming mesh into the frame preprocess_mesh expects,
    # named and defined exactly like Trellis2MeshWithVoxelToTrimesh's reorient_vertices.
    # The pipeline's own TRIMESH frame is Z-up (that node's "90 degrees" output), so a
    # plain Y-up glb needs "90 degrees" applied again to get there.
    MESH_ORIENTATIONS = {
        'none': np.eye(4),
        '90 degrees': np.array([[1., 0., 0., 0.],      # (x, y, z) -> (x, z, -y)
                                [0., 0., 1., 0.],
                                [0., -1., 0., 0.],
                                [0., 0., 0., 1.]]),
        '-90 degrees': np.array([[1., 0., 0., 0.],     # (x, y, z) -> (x, -z, y)
                                 [0., 0., -1., 0.],
                                 [0., 1., 0., 0.],
                                 [0., 0., 0., 1.]]),
    }




        
    
            
