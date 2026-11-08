"""Isolated experimental checkpoints and HR scaffold experiments.

No production node or sampler is monkey-patched by this module.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .. import nodes as _trellis
from ..orbit_nodes import _require_pixal3d_mv, _normalize_mode, _normalize_source, _require_geometry_sources
from ..trellis2.modules.sparse import SparseTensor


def _checkpoint_dir(path):
    raw = str(path or '').strip()
    if not raw:
        raise ValueError('checkpoint_path must be a non-empty directory path')
    return Path(raw).expanduser().resolve()


def _ensure_fresh_dir(root):
    if root.exists():
        if any(root.iterdir()) if root.is_dir() else True:
            raise FileExistsError(f'Checkpoint already exists; refusing to overwrite: {root}')
    root.mkdir(parents=True, exist_ok=True)


def _atomic_safetensors(path, tensors):
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(tmp))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_json(path, record):
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True), encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_checkpoint(path, kind):
    root = _checkpoint_dir(path)
    meta = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if meta.get('format') != 'orbit3d_stage_checkpoint_v1' or meta.get('kind') != kind:
        raise ValueError(f'Unexpected checkpoint kind/format in {root}: {meta}')
    tensors = load_file(str(root / 'tensors.safetensors'), device='cpu')
    return meta, tensors


class Orbit3DSaveSparseStage:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'pipeline': ('TRELLIS2PIPELINE',),
            'coords': ('COORDS',),
            'sparse_structure_resolution': ('INT', {'forceInput': True}),
            'checkpoint_path': ('STRING', {'default': '/home/ai/orbit3d-cache/experiments/left_hand/sparse', 'multiline': False}),
        }}
    RETURN_TYPES = ('COORDS', 'INT', 'TRELLIS2PIPELINE', 'STRING')
    RETURN_NAMES = ('coords', 'sparse_structure_resolution', 'pipeline', 'checkpoint_path')
    CATEGORY = 'Orbit to 3D/Experimental'
    FUNCTION = 'run'
    OUTPUT_NODE = True

    def run(self, pipeline, coords, sparse_structure_resolution, checkpoint_path):
        _require_pixal3d_mv(pipeline)
        if not isinstance(coords, torch.Tensor) or coords.ndim != 2 or coords.shape[1] != 4:
            raise TypeError(f'Expected COORDS tensor [N,4], got {type(coords)} {getattr(coords, "shape", None)}')
        resolution = int(sparse_structure_resolution)
        if resolution <= 0:
            raise ValueError('sparse_structure_resolution must be positive')
        root = _checkpoint_dir(checkpoint_path)
        _ensure_fresh_dir(root)
        _atomic_safetensors(root/'tensors.safetensors', {'coords': coords})
        _atomic_json(root/'manifest.json', {'format':'orbit3d_stage_checkpoint_v1', 'kind':'sparse',
                                         'sparse_structure_resolution':resolution, 'num_tokens':int(coords.shape[0])})
        print(f'[Orbit3D Stage] saved Sparse: {root}, coords={tuple(coords.shape)}', flush=True)
        return (coords, resolution, pipeline, str(root))


class Orbit3DLoadSparseStage:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'pipeline': ('TRELLIS2PIPELINE',),
            'checkpoint_path': ('STRING', {'default':'/home/ai/orbit3d-cache/experiments/left_hand/sparse', 'multiline':False}),
        }}
    RETURN_TYPES = ('COORDS', 'INT', 'TRELLIS2PIPELINE')
    RETURN_NAMES = ('coords', 'sparse_structure_resolution', 'pipeline')
    CATEGORY = 'Orbit to 3D/Experimental'
    FUNCTION = 'run'

    def run(self, pipeline, checkpoint_path):
        _require_pixal3d_mv(pipeline)
        meta, tensors = _read_checkpoint(checkpoint_path, 'sparse')
        coords = tensors['coords']
        if coords.ndim != 2 or coords.shape[1] != 4 or int(coords.shape[0]) != meta['num_tokens']:
            raise ValueError('Sparse checkpoint coords shape mismatch')
        return (coords.to(pipeline.device), int(meta['sparse_structure_resolution']), pipeline)


class Orbit3DSaveShape512Stage:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'pipeline': ('TRELLIS2PIPELINE',),
            'shape_slat': ('SHAPE_SLAT',),
            'resolution': ('INT', {'forceInput':True}),
            'checkpoint_path': ('STRING', {'default':'/home/ai/orbit3d-cache/experiments/left_hand/shape512', 'multiline':False}),
        }}
    RETURN_TYPES = ('SHAPE_SLAT', 'INT', 'TRELLIS2PIPELINE', 'STRING')
    RETURN_NAMES = ('shape_slat', 'resolution', 'pipeline', 'checkpoint_path')
    CATEGORY = 'Orbit to 3D/Experimental'
    FUNCTION = 'run'
    OUTPUT_NODE = True

    def run(self, pipeline, shape_slat, resolution, checkpoint_path):
        _require_pixal3d_mv(pipeline)
        if int(resolution) != 512:
            raise ValueError(f'Shape512 checkpoint expects resolution=512, got {resolution}')
        if not isinstance(shape_slat, SparseTensor):
            raise TypeError(f'Expected SparseTensor SHAPE_SLAT, got {type(shape_slat)}')
        coords, feats = shape_slat.coords, shape_slat.feats
        if coords.ndim != 2 or coords.shape[1] != 4 or feats.shape[0] != coords.shape[0]:
            raise ValueError('Shape512 sparse tensor coords/features mismatch')
        root = _checkpoint_dir(checkpoint_path)
        _ensure_fresh_dir(root)
        _atomic_safetensors(root/'tensors.safetensors', {'coords': coords, 'feats':feats})
        _atomic_json(root/'manifest.json', {'format':'orbit3d_stage_checkpoint_v1', 'kind':'shape512',
                                         'resolution':512, 'num_tokens':int(coords.shape[0]),
                                         'channels':int(feats.shape[1]), 'dtype':str(feats.dtype)})
        print(f'[Orbit3D Stage] saved Shape512: {root}, coords={tuple(coords.shape)} feats={tuple(feats.shape)}', flush=True)
        return (shape_slat, 512, pipeline, str(root))


class Orbit3DLoadShape512Stage:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'pipeline': ('TRELLIS2PIPELINE',),
            'checkpoint_path': ('STRING', {'default':'/home/ai/orbit3d-cache/experiments/left_hand/shape512', 'multiline':False}),
        }}
    RETURN_TYPES = ('SHAPE_SLAT', 'INT', 'TRELLIS2PIPELINE')
    RETURN_NAMES = ('shape_slat', 'resolution', 'pipeline')
    CATEGORY = 'Orbit to 3D/Experimental'
    FUNCTION = 'run'

    def run(self, pipeline, checkpoint_path):
        _require_pixal3d_mv(pipeline)
        meta, tensors = _read_checkpoint(checkpoint_path, 'shape512')
        coords, feats = tensors['coords'], tensors['feats']
        if (int(meta['resolution']) != 512 or coords.ndim != 2 or coords.shape[1] != 4
                or feats.ndim != 2 or feats.shape[0] != coords.shape[0]
                or int(meta['num_tokens']) != coords.shape[0]
                or int(meta['channels']) != feats.shape[1]):
            raise ValueError('Shape512 checkpoint metadata/tensors mismatch')
        slat = SparseTensor(feats=feats.to(pipeline.device), coords=coords.to(pipeline.device))
        return (slat, 512, pipeline)


class Orbit3DExperimentalShapeCascade(_trellis.Trellis2ShapeCascadeGenerator):
    """Independent Cascade with optional coordinate inspection/dilation, never modifies production nodes."""
    CATEGORY = 'Orbit to 3D/Experimental'

    @classmethod
    def INPUT_TYPES(cls):
        base = _trellis.Trellis2ShapeCascadeGenerator.INPUT_TYPES()
        required = dict(base['required'])
        required.update({
            'scaffold_mode': (['off','inspect','dilate_6'], {'default':'off'}),
            'roi_xmin': ('FLOAT', {'default':0.0,'min':0.0,'max':1.0,'step':0.01}),
            'roi_xmax': ('FLOAT', {'default':1.0,'min':0.0,'max':1.0,'step':0.01}),
            'roi_ymin': ('FLOAT', {'default':0.0,'min':0.0,'max':1.0,'step':0.01}),
            'roi_ymax': ('FLOAT', {'default':1.0,'min':0.0,'max':1.0,'step':0.01}),
            'roi_zmin': ('FLOAT', {'default':0.0,'min':0.0,'max':1.0,'step':0.01}),
            'roi_zmax': ('FLOAT', {'default':1.0,'min':0.0,'max':1.0,'step':0.01}),
            'max_added_tokens': ('INT', {'default':10000,'min':1,'max':500000,'step':100}),
            'scaffold_debug_path': ('STRING', {'default':'/home/ai/orbit3d-cache/experiments/left_hand/scaffold', 'multiline':False}),
        })
        return {'required': required, 'optional':dict(base.get('optional',{}))}

    def process(self, pipeline, shape_slat, from_resolution, to_resolution,
                sparse_structure_resolution, max_num_tokens, shape_steps,
                shape_guidance_strength, shape_guidance_rescale, shape_rescale_t,
                shape_sampler, shape_guidance_interval_start, shape_guidance_interval_end,
                verbose, dino_lock, dino_substeps, dino_foundation_cap,
                shape_hr_mode='off', shape_hr_normals_source='Marigold',
                shape_hr_depth_source='Marigold', shape_hr_seed_mode='inherit', shape_hr_seed=0,
                shape_hr_noise_cache_mode='off', shape_hr_noise_cache_path='',
                scaffold_mode='off', roi_xmin=0., roi_xmax=1., roi_ymin=0., roi_ymax=1.,
                roi_zmin=0., roi_zmax=1., max_added_tokens=10000,
                scaffold_debug_path='', pixal3d_mv_views=None):
        _require_pixal3d_mv(pipeline)
        mode = _normalize_mode(shape_hr_mode, 'shape_hr_mode')
        normal_source = _normalize_source(shape_hr_normals_source,'shape_hr_normals_source')
        depth_source = _normalize_source(shape_hr_depth_source,'shape_hr_depth_source')
        _require_geometry_sources(pixal3d_mv_views, mode, normal_source, depth_source, 'Experimental ShapeHR')
        self._experimental_config = {
            'mode':scaffold_mode, 'bounds':(roi_xmin, roi_xmax, roi_ymin, roi_ymax, roi_zmin, roi_zmax),
            'max_added_tokens':int(max_added_tokens), 'debug_path':str(scaffold_debug_path),
        }
        # Use exactly the standard stage process (sampler parameters, conditioning, cleanup).
        return super().process(
            pipeline=pipeline, shape_slat=shape_slat, from_resolution=from_resolution,
            to_resolution=to_resolution, sparse_structure_resolution=sparse_structure_resolution,
            max_num_tokens=max_num_tokens, shape_steps=shape_steps,
            shape_guidance_strength=shape_guidance_strength,
            shape_guidance_rescale=shape_guidance_rescale, shape_rescale_t=shape_rescale_t,
            shape_sampler=shape_sampler,
            shape_guidance_interval_start=shape_guidance_interval_start,
            shape_guidance_interval_end=shape_guidance_interval_end,
            verbose=verbose, dino_lock=dino_lock, dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap, shape_hr_mode=mode,
            shape_hr_normals_source=normal_source, shape_hr_depth_source=depth_source,
            shape_hr_seed_mode=shape_hr_seed_mode, shape_hr_seed=shape_hr_seed,
            shape_hr_noise_cache_mode=shape_hr_noise_cache_mode,
            shape_hr_noise_cache_path=shape_hr_noise_cache_path,
            pixal3d_mv_views=pixal3d_mv_views,
        )

    def sample(self, pipeline, slat, lr_resolution, resolution, sparse_structure_resolution,
               max_num_tokens, sampler_params, verbose, dino_lock, dino_substeps,
               dino_foundation_cap, pixal3d_mv_views, shape_hr_seed_mode='inherit', shape_hr_seed=0,
               shape_hr_noise_cache_mode='off', shape_hr_noise_cache_path=''):
        cfg = self._experimental_config
        if cfg['mode'] == 'off':
            return super().sample(
                pipeline, slat, lr_resolution, resolution, sparse_structure_resolution,
                max_num_tokens, sampler_params, verbose, dino_lock, dino_substeps,
                dino_foundation_cap, pixal3d_mv_views,
                shape_hr_seed_mode=shape_hr_seed_mode, shape_hr_seed=shape_hr_seed,
                shape_hr_noise_cache_mode=shape_hr_noise_cache_mode,
                shape_hr_noise_cache_path=shape_hr_noise_cache_path,
            )
        return self._experimental_sample(
            pipeline, slat, lr_resolution, resolution, sparse_structure_resolution,
            max_num_tokens, sampler_params, verbose, dino_lock, dino_substeps,
            dino_foundation_cap, pixal3d_mv_views,
            shape_hr_seed_mode, shape_hr_seed, shape_hr_noise_cache_mode, shape_hr_noise_cache_path,
        )

    def _experimental_sample(self, pipeline, slat, lr_resolution, resolution, sparse_structure_resolution,
                             max_num_tokens, sampler_params, verbose, dino_lock, dino_substeps,
                             dino_foundation_cap, pixal3d_mv_views, shape_hr_seed_mode, shape_hr_seed,
                             shape_hr_noise_cache_mode, shape_hr_noise_cache_path):
        # Same upsample/quantization/conditioning/sampling as production; only optional
        # coordinate inspection/dilation occurs between unique() and conditioning.
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
        ratio = sparse_structure_resolution / 32
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / (lr_resolution * ratio) * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens:
                if hr_resolution != resolution:
                    print(f'Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.')
                print(f'Num Tokens: {num_tokens}')
                break
            hr_resolution -= 128
            if hr_resolution < 1024 and resolution >= 1024:
                print(f'Num Tokens: {num_tokens}')
                hr_resolution = 1024
                break
            if hr_resolution < 512:
                print(f'Num Tokens: {num_tokens}')
                hr_resolution = 512
                break

        coords = self._modify_coords(coords, hr_resolution, max_num_tokens)
        num_tokens = int(coords.shape[0])
        actual_grid_res = hr_resolution // 16
        image_cond_model = pipeline.load_pixal3d_mv_image_cond_shape_1024()
        cond = pipeline.get_proj_cond_shape_mv(image_cond_model, pixal3d_mv_views, coords,
                                               grid_resolution_override=actual_grid_res,
                                               geometry_stage='shape_hr')
        if not pipeline.keep_models_loaded:
            pipeline.unload_pixal3d_mv_image_cond_shape_1024()
        pipeline.load_shape_slat_flow_model_1024()
        flow_model = pipeline.models['shape_slat_flow_model_1024']
        if pipeline.low_vram:
            cond = pipeline._cond_to(cond, pipeline.device)
        coords_dev = coords.to(pipeline.device)
        if str(shape_hr_seed_mode).lower() == 'inherit' and _trellis._orbit3d_normalize_cache_mode(shape_hr_noise_cache_mode) == 'off':
            noise_feats = torch.randn(coords.shape[0], flow_model.in_channels, device=pipeline.device)
        else:
            noise_feats = _trellis._orbit3d_build_hr_noise(
                coords=coords, in_channels=flow_model.in_channels, device=pipeline.device,
                seed_mode=shape_hr_seed_mode, seed=shape_hr_seed,
                cache_mode=shape_hr_noise_cache_mode, cache_path=shape_hr_noise_cache_path,
            )
        noise = SparseTensor(feats=noise_feats, coords=coords_dev)
        sampler_params = {**pipeline.shape_slat_sampler_params, **sampler_params}
        if pipeline.low_vram:
            flow_model.to(pipeline.device)
        slat = pipeline.shape_slat_sampler.sample(
            flow_model, noise, **cond, **sampler_params,
            verbose=verbose, dino_lock=dino_lock, dino_substeps=dino_substeps,
            dino_foundation_cap=dino_foundation_cap,
            tqdm_desc='Sampling shape SLat (HR)',
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

    @staticmethod
    def _export_scaffold_preview(path, coords, roi, grid):
        """Small deterministic PLY point cloud (red = ROI, gray = elsewhere)."""
        # File is visualization only; exact coordinates remain in safetensors.
        n = int(coords.shape[0])
        stride = max(1, (n + 199999) // 200000)
        xyz = ((coords[::stride, 1:].detach().cpu().float() + 0.5) / grid).tolist()
        flags = roi[::stride].detach().cpu().tolist()
        with path.open('w', encoding='ascii') as out:
            out.write('ply\nformat ascii 1.0\n')
            out.write(f'element vertex {len(xyz)}\n')
            out.write('property float x\nproperty float y\nproperty float z\n')
            out.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
            for point, selected in zip(xyz, flags):
                r, g, b = (245, 80, 60) if selected else (130, 130, 130)
                out.write(f'{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {r} {g} {b}\n')

    def _modify_coords(self, coords, hr_resolution, max_num_tokens):
        cfg = self._experimental_config
        grid = hr_resolution // 16
        bounds = cfg['bounds']
        if any(bounds[i] >= bounds[i+1] for i in (0,2,4)):
            raise ValueError(f'Invalid ROI bounds: {bounds}')
        if cfg['mode'] not in ('inspect','dilate_6'):
            raise ValueError(f'Invalid scaffold_mode={cfg["mode"]}')
        xyz = coords[:, 1:].float()
        norm = (xyz + 0.5) / grid
        roi = ((norm[:,0] >= bounds[0]) & (norm[:,0] <= bounds[1]) &
               (norm[:,1] >= bounds[2]) & (norm[:,1] <= bounds[3]) &
               (norm[:,2] >= bounds[4]) & (norm[:,2] <= bounds[5]))
        selected = coords[roi]
        path = _checkpoint_dir(cfg['debug_path'])
        path.mkdir(parents=True, exist_ok=True)
        # Save compact coordinate tensors for exact inspection and ROI refinement.
        _atomic_safetensors(path / 'scaffold_original.safetensors', {'coords':coords, 'roi_coords':selected})
        self._export_scaffold_preview(path / 'scaffold_preview.ply', coords, roi, grid)
        if cfg['mode'] == 'inspect':
            print(f'[Experimental ShapeHR] inspect grid={grid}, ROI tokens={selected.shape[0]}, total={coords.shape[0]}',flush=True)
            return coords
        if selected.shape[0] == 0:
            raise ValueError('ROI contains no HR tokens')
        if (bounds[1]-bounds[0])*(bounds[3]-bounds[2])*(bounds[5]-bounds[4]) > 0.2:
            raise ValueError('Dilation ROI is too broad; narrow ROI bounds before running')
        offsets = torch.tensor([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], device=coords.device, dtype=coords.dtype)
        batch = selected[:,None,:1].expand(-1,6,-1)
        neigh_xyz = selected[:,None,1:] + offsets[None,:,:]
        candidates = torch.cat((batch,neigh_xyz),dim=2).reshape(-1,4)
        valid = ((candidates[:,1:] >= 0) & (candidates[:,1:] < grid)).all(dim=1)
        candidates = candidates[valid]
        # Keep additions inside declared 3D ROI.
        cnorm = (candidates[:,1:].float()+0.5)/grid
        inside = ((cnorm[:,0]>=bounds[0])&(cnorm[:,0]<=bounds[1]) &
                  (cnorm[:,1]>=bounds[2])&(cnorm[:,1]<=bounds[3]) &
                  (cnorm[:,2]>=bounds[4])&(cnorm[:,2]<=bounds[5]))
        candidates = candidates[inside]
        merged = torch.cat((coords,candidates),dim=0).unique(dim=0)
        added = merged.shape[0]-coords.shape[0]
        if added > cfg['max_added_tokens']:
            raise ValueError(f'Dilation would add {added} tokens; max_added_tokens={cfg["max_added_tokens"]}')
        if merged.shape[0] >= max_num_tokens:
            raise ValueError(f'Dilation would exceed max_num_tokens: {merged.shape[0]} >= {max_num_tokens}')
        _atomic_safetensors(path / 'scaffold_modified.safetensors', {'coords':merged})
        print(f'[Experimental ShapeHR] dilate_6 grid={grid}, ROI={selected.shape[0]} tokens, added={added}, total={merged.shape[0]}',flush=True)
        return merged
