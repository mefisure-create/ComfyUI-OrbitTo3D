# ORBIT3D_CORE_MV_ONLY_V1
"""
Runtime core subset for ComfyUI-OrbitTo3D.

Only the Pixal3D multi-view reconstruction path is retained here:
  * model loading,
  * multiview camera/view preparation,
  * Sparse Structure,
  * Shape512,
  * ShapeHR cascade.

Legacy single-view conditioning / camera-config generator paths are intentionally
removed.  The pipeline object keeps the Trellis-compatible TRELLIS2PIPELINE
socket so original downstream Trellis2 mesh/decode nodes can still consume it.
"""

import gc
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import folder_paths
import comfy.model_management as mm

from .trellis2.pipelines import Trellis2ImageTo3DPipeline
from .trellis2.pipelines import samplers
from .trellis2.modules.attention import config
from .trellis2.modules.sparse import config as sparseconfig
from .trellis2.modules.sparse import SparseTensor

script_directory = os.path.dirname(os.path.abspath(__file__))

def reset_cuda():    
    # Synchronize to ensure all GPU operations complete
    torch.cuda.synchronize()     
    
    # Force garbage collection of Python objects
    gc.collect()    
    
    # Clear PyTorch CUDA cache
    torch.cuda.empty_cache()

def check_pixal3d_mv_pipeline(pipeline):
    """
    The multi-view conditioning only lines up with the *_mv denoisers.

    Feeding it to the single-view checkpoints silently produces garbage rather
    than an error, so refuse it here: the pipeline has to have been loaded with
    pixal3d_multiview on (pipeline_mv.json).
    """
    if not getattr(pipeline, 'isPixal3DMV', False):
        raise Exception(
            'pixal3d_mv_views needs the Pixal3D multi-view weights. Turn on '
            '"pixal3d_multiview" in Trellis2 - LoadModel (loads pipeline_mv.json / ckpts/*_mv).')

class Trellis2LoadModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "modelname": (
                    [
                        "microsoft/TRELLIS.2-4B",
                        "visualbruno/TRELLIS.2-4B-FP8",
                        "TencentARC/Pixal3D",
                        "TencentARC/Pixal3D-T",
                    ],
                    {"default": "microsoft/TRELLIS.2-4B"}
                ),
                "backend": (["flash_attn","xformers","sdpa","flash_attn_3"],{"default":"flash_attn"}),
                "device": (["cpu","cuda"],{"default":"cuda"}),
                "low_vram": ("BOOLEAN",{"default":True}),
                "keep_models_loaded": ("BOOLEAN", {"default":True}),
                "conv_backend": (["spconv","torchsparse","flex_gemm"],{"default":"flex_gemm"}),
                "sparse_backend": (["xformers","flash_attn"],{"default":"flash_attn"}),
                "pixal3d_multiview": ("BOOLEAN",{"default":False,"tooltip":"Pixal3D only: load the multi-view denoisers (pipeline_mv.json / ckpts/*_mv) instead of the single-view ones"}),
                #"naf_chunk_size":(["None","144","208","272","336","400","464","528","592","656","720","784","848","912","976","1024"],{"default":"None"}),
            }
        }

    RETURN_TYPES = ("TRELLIS2PIPELINE", )
    RETURN_NAMES = ("pipeline", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, modelname, backend, device, low_vram, keep_models_loaded, conv_backend, sparse_backend, pixal3d_multiview = False):
        import requests
        
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
        #os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json')
        #os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = '1'        
        os.environ['ATTN_BACKEND'] = backend
        
        config.set_backend(backend)
        sparseconfig.set_attn_backend(sparse_backend)
        sparseconfig.set_conv_backend(conv_backend)
        
        reset_cuda()
        
        torch.backends.cudnn.benchmark = False        
            
        model_path = os.path.join(folder_paths.models_dir, modelname)
        
        if not os.path.exists(model_path):
            print(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=modelname,
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )
        elif modelname == "TencentARC/Pixal3D" and pixal3d_multiview and not os.path.exists(os.path.join(model_path, 'pipeline_mv.json')):
            # A Pixal3D folder downloaded before the multi-view release has no
            # pipeline_mv.json / ckpts/*_mv. snapshot_download skips what is already
            # there, so this only pulls the missing multi-view files.
            print(f"Multi-view weights missing in {model_path}, downloading them ...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=modelname,
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )

        dinov3_model_path = os.path.join(folder_paths.models_dir,"facebook","dinov3-vitl16-pretrain-lvd1689m","model.safetensors")
        if not os.path.exists(dinov3_model_path):
            print('Downloading dinov3 model files ...')            
            from huggingface_hub import hf_hub_download
            
            repo_id = "visualbruno/dinov3-vitl16-pretrain-lvd1689m"
            target_dir = os.path.join(folder_paths.models_dir, "facebook", "dinov3-vitl16-pretrain-lvd1689m")
            
            os.makedirs(target_dir, exist_ok=True)

            # Enable the fast Rust-based transfer backend (multi-connection, much faster)
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

            for fname in ["model.safetensors", "config.json", "preprocessor_config.json"]:
                print(f"downloading {fname} ...")
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=fname,
                    local_dir=target_dir,
                    local_dir_use_symlinks=False,
                )
                print(f"{fname} -> {downloaded_path}")            
        
        trellis_image_large_path = os.path.join(folder_paths.models_dir,"microsoft","TRELLIS-image-large","ckpts","ss_dec_conv3d_16l8_fp16.safetensors")
        if not os.path.exists(trellis_image_large_path):
            print('Trellis-Image-Large ss_dec_conv3d_16l8_fp16 files not found. Trying to download the files from huggingface ...')            
            url = "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/ss_dec_conv3d_16l8_fp16.json?download=true"
            filename = os.path.join(folder_paths.models_dir,"microsoft","TRELLIS-image-large","ckpts","ss_dec_conv3d_16l8_fp16.json")
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                print("Download ss_dec_conv3d_16l8_fp16.json complete!")
            else:
                raise Exception("Cannot download Trellis-Image-Large file ss_dec_conv3d_16l8_fp16.json")
            
            url = "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/ckpts/ss_dec_conv3d_16l8_fp16.safetensors?download=true"
            filename = os.path.join(folder_paths.models_dir,"microsoft","TRELLIS-image-large","ckpts","ss_dec_conv3d_16l8_fp16.safetensors")

            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                print("Download ss_dec_conv3d_16l8_fp16.safetensors complete!")
            else:
                raise Exception("Cannot download Trellis-Image-Large file ss_dec_conv3d_16l8_fp16.safetensors")
        
        if modelname == "visualbruno/TRELLIS.2-4B-FP8":
            use_fp8 = True
        else:
            use_fp8 = False
        
        isPixal3D = modelname in ("TencentARC/Pixal3D", "TencentARC/Pixal3D-T")
        isPixal3DMVCapable = modelname == "TencentARC/Pixal3D"

        if pixal3d_multiview and not isPixal3DMVCapable:
            raise Exception('pixal3d_multiview works only with TencentARC/Pixal3D, not Pixal3D-T')


        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path, keep_models_loaded = keep_models_loaded, use_fp8=use_fp8, isPixal3D = isPixal3D, isPixal3DMV=(pixal3d_multiview and isPixal3DMVCapable))
        pipeline.low_vram = low_vram
        
        # if naf_chunk_size == "None":
            # pipeline.naf_chunk_size = None
        # else:
            # pipeline.naf_chunk_size = int(naf_chunk_size)
        
        if device=="cuda":
            if low_vram:
                pipeline.cuda()
            else:
                pipeline.to(device)
        else:
            pipeline.to(device)
        
        return (pipeline,)

class Trellis2SparseGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "sparse_structure_steps": ("INT",{"default":12, "min":1, "max":100},),
                "sparse_structure_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "sparse_structure_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),
                "sparse_structure_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "sparse_structure_resolution": ("INT", {"default":32,"min":32,"max":128,"step":4}),
                "sparse_structure_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "sparse_structure_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "fill_holes":("BOOLEAN",{"default":True}),
                "hole_iterations": ("INT",{"default":1,"min":1,"max":9,"step":1}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "hole_fill_algorithm": (["morphological_closing","flood_fill","remove_small_holes"],{"default":"flood_fill"}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
                "keep_only_shell": ("BOOLEAN",{"default":True}),
            },
            "optional":{
                "pixal3d_mv_views":("PIXAL3D_MV_VIEWS",)
            }
        }

    RETURN_TYPES = ("COORDS", "INT", "TRELLIS2PIPELINE",)
    RETURN_NAMES = ("coords", "sparse_structure_resolution", "pipeline",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, seed,
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
        pixal3d_mv_views=None
        ):
        self.seed_all(seed)

        if pixal3d_mv_views is None:
            raise ValueError("Pixal3D MV sparse generation requires pixal3d_mv_views")
        check_pixal3d_mv_pipeline(pipeline)

        sparse_structure_guidance_interval = [sparse_structure_guidance_interval_start,sparse_structure_guidance_interval_end]
        sparse_structure_sampler_params = {"steps":sparse_structure_steps,"guidance_strength":sparse_structure_guidance_strength,"guidance_rescale":sparse_structure_guidance_rescale,"guidance_interval":sparse_structure_guidance_interval,"rescale_t":sparse_structure_rescale_t}

        args = pipeline._pretrained_args
        sparse_sampler_prefix = pipeline.GetSamplerName(sparse_structure_sampler)
        pipeline.sparse_structure_sampler = getattr(samplers, f"Flow{sparse_sampler_prefix}GuidanceIntervalSampler")(**args['sparse_structure_sampler']['args'])

        if pixal3d_mv_views.get("ss_geometry", None) is not None:
            geo = pixal3d_mv_views["ss_geometry"]
            print(
                "[Pixal3D SS] MoGe surface geometry enabled: "
                f"points={tuple(geo['points'].shape)}"
            )
        else:
            print(
                "[Pixal3D SS] no ss_geometry: "
                "falling back to stock all-view fusion"
            )

        image_cond_model = pipeline.load_pixal3d_mv_image_cond_ss()
        image_cond = pipeline.get_proj_cond_ss_mv(
            pixal3d_mv_views,
            image_cond_model=image_cond_model
        )
        if not pipeline.keep_models_loaded:
            pipeline.unload_pixal3d_mv_image_cond_ss()

        pipeline.load_sparse_structure_model()

        coords = pipeline.sample_sparse_structure(
            image_cond, sparse_structure_resolution,
            1, sparse_structure_sampler_params,
            fill_holes=fill_holes,
            hole_iterations=hole_iterations,
            verbose=verbose,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            hole_fill_algorithm=hole_fill_algorithm,
            dino_foundation_cap=dino_foundation_cap,
            keep_only_shell=keep_only_shell
        )

        if not pipeline.keep_models_loaded:
            pipeline.unload_sparse_structure_model()

        return (coords, sparse_structure_resolution, pipeline,)

    def seed_all(self, seed: int = 0):
        import random
        """
        Set random seeds of all components.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class Trellis2ShapeGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "coords": ("COORDS",),
                "resolution": ([512,1024],{"default":1024}),
                "shape_steps": ("INT",{"default":12, "min":1, "max":100},),
                "shape_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "shape_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "shape_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),
                "shape_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "shape_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "shape_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
            },
            "optional":
            {
                "pixal3d_mv_views": ("PIXAL3D_MV_VIEWS",),
            }
        }

    RETURN_TYPES = ("SHAPE_SLAT", "INT", "TRELLIS2PIPELINE",)
    RETURN_NAMES = ("shape_slat", "resolution", "pipeline",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, coords, resolution,
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
        pixal3d_mv_views=None
        ):
        if pixal3d_mv_views is None:
            raise ValueError("Pixal3D MV shape generation requires pixal3d_mv_views")
        check_pixal3d_mv_pipeline(pipeline)

        shape_guidance_interval = [shape_guidance_interval_start, shape_guidance_interval_end]
        shape_slat_sampler_params = {"steps":shape_steps,"guidance_strength":shape_guidance_strength,"guidance_rescale":shape_guidance_rescale,"guidance_interval":shape_guidance_interval,"rescale_t":shape_rescale_t}

        args = pipeline._pretrained_args
        shape_sampler_prefix = pipeline.GetSamplerName(shape_sampler)
        pipeline.shape_slat_sampler = getattr(samplers, f"Flow{shape_sampler_prefix}GuidanceIntervalSampler")(**args['shape_slat_sampler']['args'])

        if resolution == 512:
            pipeline.unload_shape_slat_flow_model_1024()

            image_cond_model = pipeline.load_pixal3d_mv_image_cond_shape_512()
            image_cond = pipeline.get_proj_cond_shape_mv(
                image_cond_model, pixal3d_mv_views, coords,
            )
            if not pipeline.keep_models_loaded:
                pipeline.unload_pixal3d_mv_image_cond_shape_512()

            pipeline.load_shape_slat_flow_model_512()

            shape_slat = pipeline.sample_shape_slat(
                image_cond, pipeline.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params,
                verbose=verbose,
                dino_lock=dino_lock,
                dino_substeps=dino_substeps,
                dino_foundation_cap=dino_foundation_cap
            )

            if not pipeline.keep_models_loaded:
                pipeline.unload_shape_slat_flow_model_512()

        elif resolution == 1024:
            pipeline.unload_shape_slat_flow_model_512()

            image_cond_model = pipeline.load_pixal3d_mv_image_cond_shape_1024()
            image_cond = pipeline.get_proj_cond_shape_mv(
                image_cond_model, pixal3d_mv_views, coords,
            )
            if not pipeline.keep_models_loaded:
                pipeline.unload_pixal3d_mv_image_cond_shape_1024()

            pipeline.load_shape_slat_flow_model_1024()

            shape_slat = pipeline.sample_shape_slat(
                image_cond, pipeline.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params,
                verbose=verbose,
                dino_lock=dino_lock,
                dino_substeps=dino_substeps,
                dino_foundation_cap=dino_foundation_cap
            )

            if not pipeline.keep_models_loaded:
                pipeline.unload_shape_slat_flow_model_1024()
        else:
            raise ValueError(f"unsupported shape resolution: {resolution}")

        return (shape_slat, resolution, pipeline,)

class Trellis2ShapeCascadeGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2PIPELINE",),
                "shape_slat": ("SHAPE_SLAT",),
                "from_resolution": ("INT",),
                "to_resolution": ([1024,1536],{"default":1024}),
                "sparse_structure_resolution": ("INT", {"default":32,"min":32,"max":128,"step":4}),
                "max_num_tokens": ("INT",{"default":999999,"min":0,"max":999999}),
                "shape_steps": ("INT",{"default":12, "min":1,"max":100},),
                "shape_guidance_strength": ("FLOAT",{"default":6.50,"min":0.00,"max":99.99,"step":0.01}),
                "shape_guidance_rescale": ("FLOAT",{"default":0.05,"min":0.00,"max":1.00,"step":0.01}),
                "shape_rescale_t": ("FLOAT",{"default":4.00,"min":0.00,"max":9.99,"step":0.01}),
                "shape_sampler": (["euler", "heun", "rk4", "rk5"], {"default": "euler"}),
                "shape_guidance_interval_start": ("FLOAT",{"default":0.10,"min":0.00,"max":1.00,"step":0.01}),
                "shape_guidance_interval_end": ("FLOAT",{"default":1.00,"min":0.00,"max":1.00,"step":0.01}),
                "verbose": ("BOOLEAN",{"default":False}),
                "dino_lock": ("FLOAT",{"default":0.00,"min":0.00,"max":1.00,"step":0.01}),
                "dino_substeps": ("INT",{"default":4,"min":1,"max":99,"step":1}),
                "dino_foundation_cap": ("FLOAT",{"default":1.00,"min":0.01,"max":1.00,"step":0.01}),
                "shape_hr_mode": (
                    ["off", "normal", "depth", "normal+depth"],
                    {
                        "default": "off",
                        "tooltip":
                            "Geometry re-ranking for the Cascade Generator HR shape pass. "
                            "off preserves the current 512->HR baseline; normal/depth/both "
                            "reuse the same native Marigold/MoGe maps as Shape512."
                    },
                ),
                "shape_hr_normals_source": (
                    ["Marigold", "MoGe"],
                    {
                        "default": "Marigold",
                        "tooltip": "Normal source used only by the cascade HR shape pass."
                    },
                ),
                "shape_hr_depth_source": (
                    ["Marigold", "MoGe"],
                    {
                        "default": "Marigold",
                        "tooltip": "Depth source used only by the cascade HR shape pass."
                    },
                ),
            },
            "optional":
            {
                "pixal3d_mv_views": ("PIXAL3D_MV_VIEWS",),
            }
        }

    RETURN_TYPES = ("SHAPE_SLAT","INT","TRELLIS2PIPELINE","INT",)
    RETURN_NAMES = ("shape_slat","resolution","pipeline","num_tokens")
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, shape_slat, from_resolution, to_resolution, sparse_structure_resolution, max_num_tokens,
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
        pixal3d_mv_views=None
        ):
        if pixal3d_mv_views is None:
            raise ValueError("Pixal3D MV cascade generation requires pixal3d_mv_views")
        check_pixal3d_mv_pipeline(pipeline)

        shape_guidance_interval = [shape_guidance_interval_start, shape_guidance_interval_end]
        shape_slat_sampler_params = {"steps":shape_steps,"guidance_strength":shape_guidance_strength,"guidance_rescale":shape_guidance_rescale,"guidance_interval":shape_guidance_interval,"rescale_t":shape_rescale_t}

        args = pipeline._pretrained_args
        shape_sampler_prefix = pipeline.GetSamplerName(shape_sampler)
        pipeline.shape_slat_sampler = getattr(samplers, f"Flow{shape_sampler_prefix}GuidanceIntervalSampler")(**args['shape_slat_sampler']['args'])

        shape_hr_mode = str(shape_hr_mode).strip().lower()
        shape_hr_normals_source = str(shape_hr_normals_source).strip().lower()
        shape_hr_depth_source = str(shape_hr_depth_source).strip().lower()

        if shape_hr_mode not in ("off", "normal", "depth", "normal+depth"):
            raise ValueError(f"invalid shape_hr_mode={shape_hr_mode!r}")
        if shape_hr_normals_source not in ("marigold", "moge"):
            raise ValueError(
                f"invalid shape_hr_normals_source={shape_hr_normals_source!r}"
            )
        if shape_hr_depth_source not in ("marigold", "moge"):
            raise ValueError(
                f"invalid shape_hr_depth_source={shape_hr_depth_source!r}"
            )

        hr_views = dict(pixal3d_mv_views)
        hr_views["shape_hr_mode"] = shape_hr_mode
        hr_views["shape_hr_normals_source"] = shape_hr_normals_source
        hr_views["shape_hr_depth_source"] = shape_hr_depth_source

        print(
            f"[ShapeHR config] mode={shape_hr_mode}; "
            f"normal={shape_hr_normals_source}; "
            f"depth={shape_hr_depth_source}"
        )

        slat, hr_resolution, num_tokens = self.sample(
            pipeline, shape_slat, from_resolution, to_resolution,
            sparse_structure_resolution, max_num_tokens,
            shape_slat_sampler_params, verbose, dino_lock, dino_substeps,
            dino_foundation_cap, hr_views
        )

        if not pipeline.keep_models_loaded:
            pipeline.unload_shape_slat_flow_model_1024()

        return (slat, hr_resolution, pipeline, num_tokens,)

    def sample(self, pipeline, slat, lr_resolution, resolution, sparse_structure_resolution,
               max_num_tokens, sampler_params, verbose, dino_lock, dino_substeps,
               dino_foundation_cap, pixal3d_mv_views):
        pipeline.load_shape_slat_decoder()
        if pipeline.low_vram:
            pipeline.models['shape_slat_decoder'].to(pipeline.device)
            pipeline.models['shape_slat_decoder'].low_vram = True
        hr_coords = pipeline.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if pipeline.low_vram:
            pipeline.models['shape_slat_decoder'].cpu()
            pipeline.models['shape_slat_decoder'].low_vram = False

        if not pipeline.keep_models_loaded:
            pipeline.unload_shape_slat_decoder()

        hr_resolution = resolution
        ratio = (sparse_structure_resolution / 32)

        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / (lr_resolution * ratio) * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                print(f"Num Tokens: {num_tokens}")
                break
            hr_resolution -= 128
            if hr_resolution < 1024 and resolution >= 1024:
                print(f"Num Tokens: {num_tokens}")
                hr_resolution = 1024
                break
            if hr_resolution < 512:
                print(f"Num Tokens: {num_tokens}")
                hr_resolution = 512
                break

        actual_grid_res = hr_resolution // 16
        image_cond_model = pipeline.load_pixal3d_mv_image_cond_shape_1024()

        cond = pipeline.get_proj_cond_shape_mv(
            image_cond_model, pixal3d_mv_views, coords,
            grid_resolution_override=actual_grid_res,
            geometry_stage="shape_hr",
        )

        if not pipeline.keep_models_loaded:
            pipeline.unload_pixal3d_mv_image_cond_shape_1024()

        pipeline.load_shape_slat_flow_model_1024()
        flow_model = pipeline.models['shape_slat_flow_model_1024']

        if pipeline.low_vram:
            cond = pipeline._cond_to(cond, pipeline.device)

        coords_dev = coords.to(pipeline.device)
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=pipeline.device),
            coords=coords_dev,
        )
        sampler_params = {**pipeline.shape_slat_sampler_params, **sampler_params}
        if pipeline.low_vram:
            flow_model.to(pipeline.device)
        slat = pipeline.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=verbose,
            dino_lock=dino_lock,
            dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
            tqdm_desc="Sampling shape SLat (HR)",
        ).samples
        if pipeline.low_vram:
            flow_model.cpu()
            pipeline._cleanup_cuda()

        std = torch.tensor(pipeline.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(pipeline.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        del coords_dev
        if pipeline.low_vram:
            cond = pipeline._cond_cpu(cond)
            pipeline._cleanup_cuda()

        return slat, hr_resolution, num_tokens

def comfy_images_to_rgba_pils(images, masks=None, invert_mask=False, remove_background=False,
                              max_views=16):
    """
    Turn a ComfyUI IMAGE batch into the RGBA PIL views the Pixal3D MV path expects.

    Alpha is taken, in order of preference, from an explicit MASK input, from the
    image's own alpha channel, or from rembg. The views are never cropped or
    rescaled here: transforms.json describes the framing as given, so changing it
    would break the correspondence between the pixels and the cameras.
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for IMAGE, got {type(images)}")
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4:
        raise ValueError(f"Unsupported IMAGE tensor shape: {tuple(images.shape)}")

    V = min(int(images.shape[0]), int(max_views))
    if V < int(images.shape[0]):
        print(f"[Pixal3D MV] Warning: {images.shape[0]} views given, using the first {V}")

    if masks is not None:
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if masks.shape[0] != images.shape[0]:
            raise ValueError(
                f"got {images.shape[0]} images but {masks.shape[0]} masks")

    out = []
    for i in range(V):
        img = images[i]
        alpha = None

        if masks is not None:
            alpha = masks[i].detach().cpu().float().clamp(0, 1)
            if invert_mask:
                alpha = 1.0 - alpha
            if alpha.shape != img.shape[:2]:
                alpha = F.interpolate(
                    alpha[None, None], size=tuple(img.shape[:2]),
                    mode='bilinear', align_corners=False)[0, 0]
        elif img.shape[-1] == 4:
            a = img[..., 3].detach().cpu().float().clamp(0, 1)
            # A fully opaque alpha channel counts as no mask, as in preprocess_image.
            if not torch.all(a >= 254.0 / 255.0):
                alpha = a

        rgb = img[..., :3].detach().cpu().float().clamp(0, 1)
        pil = Image.fromarray((rgb.numpy() * 255.0).astype(np.uint8), mode='RGB')

        if alpha is None:
            if not remove_background:
                raise ValueError(
                    f"view {i} has no alpha channel and no mask. Give it a MASK, an "
                    f"RGBA image, or turn remove_background on.")
            from rembg import remove
            pil = remove(pil).convert('RGBA')
        else:
            pil = pil.convert('RGBA')
            pil.putalpha(Image.fromarray((alpha.numpy() * 255.0).astype(np.uint8), mode='L'))

        out.append(pil)

    return out

def pixal3d_views_to_preview(views):
    """The alpha-premultiplied 512px views, as a ComfyUI IMAGE batch, for previewing."""
    size = min(views['images'].keys())
    return views['images'][size][0].permute(0, 2, 3, 1).contiguous().cpu()

class Trellis2ImagesToViewConfigs:
    """Private compatibility helper retained only for angle parsing."""

    def _parse_angles(self, angle_string):
        """Parse comma-separated angles into list of floats."""
        if not angle_string or angle_string.strip() == "":
            return []
        try:
            return [float(x.strip()) for x in angle_string.split(",") if x.strip()]
        except ValueError:
            print(f"[MultiView] Warning: Could not parse angles: {angle_string}")
            return []

class Trellis2Pixal3DMultiViewConfig:
    """
    Build the Pixal3D multi-view conditioning bundle from a batch of posed views.

    The first image is the MAIN view and its pose must be the canonical front view
    (azimuth 0, elevation 0), because every other view is placed relative to it.
    Azimuth / elevation follow the same convention as
    Trellis2RenderMultiViewNvdiffrast: 0/90/180/270 -> front/left/back/right, and
    positive elevation is above the object.

    fov / distance / mesh_scale describe the camera the views were SHOT with, and
    the views are never cropped or rescaled here, so they have to match how the
    images are actually framed. Getting the distance wrong scales the whole
    projection: at 10% too near, the surface of the model samples the background
    instead of itself, which shows up as washed-out texture long before the mesh
    suffers. `framing` picks how the distance is derived:

      auto         measure the object in the alpha channel and fit the distance to
                   it. Works whatever the margin is; the default.
      pixal3d_rig  the rig the multi-view weights were trained on, 10% margin
                   (the shipped example: fov 20 deg, distance 3.1192).
      fill_frame   object touches the frame edges. This is right for views
                   you explicitly cropped to fill the frame and wrong for raw
                   renders with margins.

    A distance widget above 0 always wins.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "azimuths": ("STRING", {"default": "0,90,180,270"}),
                "elevations": ("STRING", {"default": "0,0,0,0"}),
                "fov": ("FLOAT", {"default": 20.0, "min": 0.001, "max": 179.999, "step": 0.001}),
                "fov_unit": (["deg", "rad"], {"default": "deg"}),
                "mesh_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 9.9, "step": 0.1}),
                "distance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 99.99, "step": 0.0001,
                                       "tooltip": "0 = derive it from `framing`. Above 0 always wins."}),
                "remove_background": ("BOOLEAN", {"default": False}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "framing": (["auto", "pixal3d_rig", "fill_frame"],
                            {"default": "auto",
                             "tooltip": "How to derive the camera distance. auto = fit it to the "
                                        "object in the alpha channel; pixal3d_rig = the 10% margin "
                                        "the MV weights were trained on; fill_frame = object touches "
                                        "the frame edges (what Trellis2PreProcessImage produces)."}),
                # MARIGOLD512_DIAG_V2
                "marigold_shape512_mode": (
                    ["off", "normal", "depth", "normal+depth"],
                    {
                        "default": "normal",
                        "tooltip":
                            "Shape512 Marigold ablation. normal = current B1.2N; "
                            "depth = affine-calibrated depth agreement; "
                            "normal+depth = both; off = neither. "
                            "Diagnostics are generated for all four counterfactuals."
                    },
                ),
                "shape512_normals_source": (
                    ["Marigold", "MoGe"],
                    {
                        "default": "Marigold",
                        "tooltip":
                            "Shape512 NORMAL source. Marigold = current B1.2N confidence; "
                            "MoGe = the same confidence law evaluated on MoGe normals/mask."
                    },
                ),
                "shape512_depth_source": (
                    ["Marigold", "MoGe"],
                    {
                        "default": "Marigold",
                        "tooltip":
                            "Shape512 DEPTH source. Marigold = native relative depth; "
                            "MoGe = camera-space Z from MoGe points[...,2]."
                    },
                ),
                # SS_ANCHORED_SOFT_OCCLUSION_V2
                "ss_visibility_mode": (
                    [
                        "all_views",
                        "soft_occlusion",
                        "anchored_soft_occlusion",
                    ],
                    {
                        "default": "anchored_soft_occlusion",
                        "tooltip":
                            "SS projected-feature fusion. "
                            "all_views = exact equal-average baseline; "
                            "soft_occlusion = V1 depth occlusion over all usable cells; "
                            "anchored_soft_occlusion = only re-rank a cell if at least "
                            "one view first confirms that the cell lies on a visible "
                            "MoGe surface."
                    },
                ),
                "ss_occlusion_floor": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.05,
                        "max": 1.00,
                        "step": 0.05,
                        "tooltip": "Minimum SS weight of a strongly occluded view; never hard-gates it."
                    },
                ),
                "ss_occlusion_margin_voxels": (
                    "FLOAT",
                    {
                        "default": 1.50,
                        "min": 0.00,
                        "max": 6.00,
                        "step": 0.25,
                        "tooltip": "No occlusion penalty until the candidate lies this many SS voxels behind the MoGe surface."
                    },
                ),
                "ss_occlusion_tau_voxels": (
                    "FLOAT",
                    {
                        "default": 1.50,
                        "min": 0.25,
                        "max": 8.00,
                        "step": 0.25,
                        "tooltip": "Gaussian falloff width after the occlusion margin."
                    },
                ),
                "ss_anchor_band_voxels": (
                    "FLOAT",
                    {
                        "default": 1.50,
                        "min": 0.25,
                        "max": 4.00,
                        "step": 0.25,
                        "tooltip":
                            "Anchored mode only: a candidate SS cell is considered "
                            "surface-confirmed when its camera depth is within this "
                            "many SS voxels of the calibrated MoGe surface in at "
                            "least one view."
                    },
                ),
                "ss_anchored_floor": (
                    "FLOAT",
                    {
                        "default": 0.50,
                        "min": 0.10,
                        "max": 1.00,
                        "step": 0.05,
                        "tooltip":
                            "Anchored mode only: minimum weight of a view that is "
                            "occluded for an already surface-confirmed SS cell."
                    },
                ),
            },
            "optional": {
                "masks": ("MASK",),
                "normal_maps": ("IMAGE",),
                "depth_maps": ("IMAGE",),

                # SS-only geometry from MoGe.
                # This must remain separate from Marigold geometry_confidence,
                # which is used later by Shape.
                "moge_geometry": ("MOGE_GEOMETRY",),

                # #33 is a natural synchronization barrier: all preprocessing
                # branches have completed before this node can execute.
                "cleanup_before_pixal3d": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip":
                            "Unload Comfy-managed preprocessing models and "
                            "clear CUDA cache before Pixal3D starts."
                    },
                ),
            }
        }

    RETURN_TYPES = ("PIXAL3D_MV_VIEWS", "IMAGE",)
    RETURN_NAMES = ("pixal3d_mv_views", "preview",)
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"
    OUTPUT_NODE = True

    def process(self, images, azimuths, elevations, fov, fov_unit, mesh_scale, distance,
                remove_background, invert_mask, framing="auto",
                marigold_shape512_mode="normal",
                shape512_normals_source="Marigold",
                shape512_depth_source="Marigold",
                ss_visibility_mode="anchored_soft_occlusion",
                ss_occlusion_floor=0.35,
                ss_occlusion_margin_voxels=1.50,
                ss_occlusion_tau_voxels=1.50,
                ss_anchor_band_voxels=1.50,
                ss_anchored_floor=0.50,
                masks=None, normal_maps=None, depth_maps=None,
                moge_geometry=None, cleanup_before_pixal3d=True):
        from .trellis2.utils import mv_camera

        az_list = Trellis2ImagesToViewConfigs()._parse_angles(azimuths)
        el_list = Trellis2ImagesToViewConfigs()._parse_angles(elevations)
        if not az_list or not el_list:
            raise Exception("azimuths and elevations are required")
        if len(az_list) != len(el_list):
            raise Exception("azimuths and elevations must have the same amount of values")

        # PIXAL_MV_ARBITRARY_V_V1
        # The MV pipeline is V-dynamic.  Make the contract explicit here so
        # ComfyUI cannot silently truncate/reorder a batch relative to azimuths.
        expected_v = len(az_list)

        if not isinstance(images, torch.Tensor):
            raise TypeError(
                f"Pixal3D MV images must be a torch.Tensor, got {type(images).__name__}"
            )

        if images.ndim == 3:
            image_v = 1
        elif images.ndim == 4:
            image_v = int(images.shape[0])
        else:
            raise ValueError(
                f"Pixal3D MV images must be [V,H,W,C] or [H,W,C], got {tuple(images.shape)}"
            )

        if image_v != expected_v:
            raise ValueError(
                f"Pixal3D MV view-count mismatch: images={image_v}, "
                f"azimuths={expected_v}, elevations={len(el_list)}. "
                "Batch order must exactly match azimuth/elevation order."
            )

        if masks is not None:
            mask_v = 1 if masks.ndim == 2 else int(masks.shape[0])
            if mask_v != expected_v:
                raise ValueError(
                    f"Pixal3D MV mask-count mismatch: masks={mask_v}, views={expected_v}"
                )

        order_text = ", ".join(
            f"{i}:az={float(a):g},el={float(e):g}"
            for i, (a, e) in enumerate(zip(az_list, el_list))
        )
        print(f"[Pixal3D MV VIEWS] V={expected_v}; order=[{order_text}]")

        camera_angle_x = float(fov) if fov_unit == "rad" else math.radians(float(fov))

        pils = comfy_images_to_rgba_pils(
            images, masks=masks, invert_mask=invert_mask,
            remove_background=remove_background, max_views=len(az_list),
        )
        if len(pils) != len(az_list):
            raise Exception(
                f"got {len(pils)} images but {len(az_list)} azimuth/elevation pairs")

        fill = mv_camera.measure_object_fill(pils)
        fill_frame = mv_camera.camera_distance_for_fov(camera_angle_x, mesh_scale)
        print(f"[Pixal3D MV] views fill {100 * fill:.1f}% of the frame "
              f"(fill_frame distance would be {fill_frame:.4f})")

        if distance > 0:
            print(f"[Pixal3D MV] framing: distance {distance:.4f} set explicitly")
        elif framing == "auto":
            distance = mv_camera.camera_distance_for_extent(
                camera_angle_x, fill * 512 / 2, mesh_scale)
            print(f"[Pixal3D MV] framing=auto: distance {distance:.4f} fitted to the views")
        elif framing == "pixal3d_rig":
            distance = fill_frame * mv_camera.PIXAL3D_RIG_MARGIN
            print(f"[Pixal3D MV] framing=pixal3d_rig: distance {distance:.4f} "
                  f"({mv_camera.PIXAL3D_RIG_MARGIN}x fill_frame, the trained 10% margin)")
        elif framing == "fill_frame":
            distance = fill_frame
            print(f"[Pixal3D MV] framing=fill_frame: distance {distance:.4f}")
        else:
            raise Exception(f"unknown framing {framing!r}")

        # The projection scales with distance, so a mismatch here quietly makes every
        # grid point sample the wrong pixel -- loudest in the texture stage.
        fitted = mv_camera.camera_distance_for_extent(camera_angle_x, fill * 512 / 2, mesh_scale)
        if abs(distance - fitted) / fitted > 0.03:
            print(f"[Pixal3D MV] Warning: distance {distance:.4f} does not match how the views "
                  f"are framed ({fitted:.4f} would). The projection is off by "
                  f"{100 * abs(distance - fitted) / fitted:.1f}%, which washes out the texture. "
                  f"Try framing=auto.")

        views = mv_camera.build_views_from_angles(
            pils, az_list, el_list, camera_angle_x,
            mesh_scale=mesh_scale, distance=distance,
        )

        # MARIGOLD512_DIAG_V2
        marigold_shape512_mode = str(marigold_shape512_mode).strip().lower()
        if marigold_shape512_mode not in (
            "off", "normal", "depth", "normal+depth",
        ):
            raise ValueError(
                f"invalid marigold_shape512_mode={marigold_shape512_mode!r}"
            )
        views["marigold_shape512_mode"] = marigold_shape512_mode
        print(
            f"[Marigold512] requested mode={marigold_shape512_mode}"
        )

        shape512_normals_source = str(shape512_normals_source).strip().lower()
        shape512_depth_source = str(shape512_depth_source).strip().lower()

        if shape512_normals_source not in ("marigold", "moge"):
            raise ValueError(
                f"invalid shape512_normals_source={shape512_normals_source!r}"
            )
        if shape512_depth_source not in ("marigold", "moge"):
            raise ValueError(
                f"invalid shape512_depth_source={shape512_depth_source!r}"
            )

        views["shape512_normals_source"] = shape512_normals_source
        views["shape512_depth_source"] = shape512_depth_source
        print(
            f"[Shape512 sources] normal={shape512_normals_source}, "
            f"depth={shape512_depth_source}"
        )

        # SS_SOFT_OCCLUSION_V1
        ss_visibility_mode = str(ss_visibility_mode).strip().lower()
        if ss_visibility_mode not in (
            "all_views",
            "soft_occlusion",
            "anchored_soft_occlusion",
        ):
            raise ValueError(
                f"invalid ss_visibility_mode={ss_visibility_mode!r}"
            )

        ss_occlusion_floor = float(ss_occlusion_floor)
        ss_occlusion_margin_voxels = float(ss_occlusion_margin_voxels)
        ss_occlusion_tau_voxels = float(ss_occlusion_tau_voxels)
        ss_anchor_band_voxels = float(ss_anchor_band_voxels)
        ss_anchored_floor = float(ss_anchored_floor)

        if not (0.0 < ss_occlusion_floor <= 1.0):
            raise ValueError("ss_occlusion_floor must be in (0,1]")
        if ss_occlusion_margin_voxels < 0.0:
            raise ValueError("ss_occlusion_margin_voxels must be >= 0")
        if ss_occlusion_tau_voxels <= 0.0:
            raise ValueError("ss_occlusion_tau_voxels must be > 0")
        if ss_anchor_band_voxels <= 0.0:
            raise ValueError("ss_anchor_band_voxels must be > 0")
        if not (0.0 < ss_anchored_floor <= 1.0):
            raise ValueError("ss_anchored_floor must be in (0,1]")

        views["ss_visibility_mode"] = ss_visibility_mode
        views["ss_soft_occlusion_params"] = {
            "floor": ss_occlusion_floor,
            "margin_voxels": ss_occlusion_margin_voxels,
            "tau_voxels": ss_occlusion_tau_voxels,
            "anchor_band_voxels": ss_anchor_band_voxels,
            "anchored_floor": ss_anchored_floor,
        }

        print(
            f"[SS-SoftOcc config] mode={ss_visibility_mode}; "
            f"soft_floor={ss_occlusion_floor:.3f}, "
            f"anchored_floor={ss_anchored_floor:.3f}, "
            f"anchor_band={ss_anchor_band_voxels:.2f} vox, "
            f"margin={ss_occlusion_margin_voxels:.2f} vox, "
            f"tau={ss_occlusion_tau_voxels:.2f} vox"
        )

        if (normal_maps is None) != (depth_maps is None):
            raise Exception(
                "Geometry weighting needs BOTH normal_maps and depth_maps, or neither."
            )

        shape512_uses_normal = marigold_shape512_mode in ("normal", "normal+depth")
        shape512_uses_depth = marigold_shape512_mode in ("depth", "normal+depth")

        need_marigold_shape512 = (
            (shape512_uses_normal and shape512_normals_source == "marigold")
            or (shape512_uses_depth and shape512_depth_source == "marigold")
        )

        need_moge_shape512 = (
            (shape512_uses_normal and shape512_normals_source == "moge")
            or (shape512_uses_depth and shape512_depth_source == "moge")
        )

        if need_marigold_shape512 and normal_maps is None:
            raise Exception(
                "Shape512 source selection requires BOTH normal_maps and depth_maps "
                "when a Marigold source is selected for an active mode."
            )

        if need_moge_shape512 and moge_geometry is None:
            raise Exception(
                "Shape512 source selection requires moge_geometry when a MoGe "
                "source is selected for an active mode."
            )

        if normal_maps is not None:
            mv_camera.attach_geometry_confidence(
                views, normal_maps, depth_maps, pils,
            )

        mv_camera.check_main_view(views)

        # --------------------------------------------------------------
        # SS-only MoGe geometry.
        #
        # Input convention from SSGeometryBatch:
        #   points      [V,H,W,3]
        #   depth       [V,H,W]
        #   normal      [V,H,W,3]
        #   mask        [V,H,W]
        #   intrinsics  [V,3,3]
        #
        # Internally PIXAL3D_MV_VIEWS has an object batch dimension,
        # so store it as [B=1,V,...].
        #
        # Keep the large maps on CPU. SS needs only ~4096 samples from
        # each map, so copying full maps to VRAM would be wasteful.
        # --------------------------------------------------------------

        if moge_geometry is not None:

            required_ss = (
                "points",
                "depth",
                "normal",
                "mask",
                "intrinsics",
            )

            missing = [
                k for k in required_ss
                if k not in moge_geometry
            ]

            if missing:
                raise Exception(
                    f"moge_geometry missing fields: {missing}"
                )

            ss_cpu = {}

            for key in required_ss:

                value = moge_geometry[key]

                if not torch.is_tensor(value):
                    raise TypeError(
                        f"moge_geometry[{key!r}] must be torch.Tensor, "
                        f"got {type(value).__name__}"
                    )

                value = value.detach().to("cpu")

                if key == "mask":
                    value = value.bool()
                else:
                    value = value.float()

                # Mutate the incoming packet as well. If the batching node's
                # output is cached, it should retain CPU tensors rather than
                # keeping another GPU reference alive.
                moge_geometry[key] = value
                ss_cpu[key] = value

            V = len(az_list)

            points = ss_cpu["points"]
            depth = ss_cpu["depth"]
            normal = ss_cpu["normal"]
            mask = ss_cpu["mask"]
            intrinsics = ss_cpu["intrinsics"]

            if points.ndim != 4 or points.shape[-1] != 3:
                raise Exception(
                    f"ss points must be [V,H,W,3], got {tuple(points.shape)}"
                )

            if normal.shape != points.shape:
                raise Exception(
                    f"ss normal shape {tuple(normal.shape)} != "
                    f"points shape {tuple(points.shape)}"
                )

            H, W = points.shape[1:3]

            if depth.shape != (V, H, W):
                raise Exception(
                    f"ss depth must be [{V},{H},{W}], "
                    f"got {tuple(depth.shape)}"
                )

            if mask.shape != (V, H, W):
                raise Exception(
                    f"ss mask must be [{V},{H},{W}], "
                    f"got {tuple(mask.shape)}"
                )

            if intrinsics.shape != (V, 3, 3):
                raise Exception(
                    f"ss intrinsics must be [{V},3,3], "
                    f"got {tuple(intrinsics.shape)}"
                )

            if points.shape[0] != V:
                raise Exception(
                    f"moge_geometry has V={points.shape[0]} but "
                    f"Pixal config has {V} views"
                )

            views["ss_geometry"] = {
                k: value.unsqueeze(0)
                for k, value in ss_cpu.items()
            }

            approx_mb = (
                sum(
                    value.numel() * value.element_size()
                    for value in ss_cpu.values()
                )
                / 1024**2
            )

            print(
                f"[Pixal3D SS-MoGe] attached V={V}, "
                f"{H}x{W} CPU geometry "
                f"({approx_mb:.1f} MiB)"
            )

            mv_camera.attach_moge_shape512_geometry(
                views,
                ss_cpu,
                pils,
            )

        # --------------------------------------------------------------
        # Pre-Pixal VRAM barrier.
        #
        # At this point RGB, masks, Marigold and MoGe outputs all exist,
        # so no preprocessing branch needs its model in VRAM any more.
        # --------------------------------------------------------------

        if cleanup_before_pixal3d:

            before_free = None

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                before_free = free

                print(
                    f"[Pixal3D preflight] before cleanup: "
                    f"{free / 1024**3:.2f} GiB free / "
                    f"{total / 1024**3:.2f} GiB total"
                )

            print(
                "[Pixal3D preflight] unloading preprocessing models ..."
            )

            mm.free_memory(
                memory_required=1e30,
                device=mm.get_torch_device(),
            )

            mm.soft_empty_cache()

            gc.collect()
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

                free, total = torch.cuda.mem_get_info()

                print(
                    f"[Pixal3D preflight] after cleanup: "
                    f"{free / 1024**3:.2f} GiB free / "
                    f"{total / 1024**3:.2f} GiB total"
                )

                if before_free is not None:
                    print(
                        f"[Pixal3D preflight] released: "
                        f"{(free - before_free) / 1024**3:.2f} GiB"
                    )

        return (views, pixal3d_views_to_preview(views),)
