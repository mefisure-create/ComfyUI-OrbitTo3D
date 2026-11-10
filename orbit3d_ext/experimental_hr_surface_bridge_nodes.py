"""Read-only V9b export of the EXACT HR scaffold offered to Pixal3D Shape Cascade.

It deliberately does not edit Shape512 SLat, VAE features, Cascade or conditioning.
No conversion of mesh faces into 'safe-to-delete' HR cells occurs here.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ..orbit_nodes import _require_pixal3d_mv
from ..trellis2.modules.sparse import SparseTensor


def _slat_fingerprint(slat: SparseTensor) -> str:
    h = hashlib.sha256()
    for name in ('coords', 'feats'):
        tensor = getattr(slat, name).detach().contiguous().to('cpu')
        h.update(name.encode('ascii'))
        h.update(str(tuple(tensor.shape)).encode('ascii'))
        h.update(str(tensor.dtype).encode('ascii'))
        h.update(tensor.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _aggregate_quantized(raw: np.ndarray, grid: int, denom: float, chunk=200000) -> tuple[np.ndarray, np.ndarray]:
    """Cascader's ((hr_coords+.5)/denom*grid).int() for nonnegative coords.

    B=1. Use 3D bincount to avoid per-child tuples/large Python objects.
    """
    counts = np.zeros(grid**3, dtype=np.int64)
    for start in range(0, len(raw), chunk):
        block = raw[start:start+chunk]
        if np.any(block[:, 0] != 0):
            raise ValueError('V9b scaffold export supports a single batch (batch_id=0)')
        xyz = np.trunc((block[:, 1:].astype(np.float64) + 0.5) / denom * grid).astype(np.int64)
        if xyz.size and (xyz.min() < 0 or xyz.max() >= grid):
            raise ValueError(f'HR quantization outside [0,{grid}): check from_resolution/SS resolution')
        key = (xyz[:, 0] * grid + xyz[:, 1]) * grid + xyz[:, 2]
        counts += np.bincount(key, minlength=grid**3)
    idx = np.flatnonzero(counts)
    coords = np.stack([idx // (grid*grid), (idx // grid) % grid, idx % grid], axis=1).astype(np.int32)
    return coords, counts[idx]


def _write_tokens_ply(path: Path, coords: np.ndarray, grid: int, counts: np.ndarray) -> None:
    # Display only: coordinate centre of each conditioning token.
    rgba = np.zeros((len(coords),), dtype=[('xyz','<f4',(3,)),('rgb','u1',(3,))])
    rgba['xyz'] = (coords.astype(np.float32) + 0.5) / float(grid) - 0.5
    # Density visualization: brighter = more VAE child coords collapsed to token.
    vmax = max(1, int(np.percentile(counts, 98)))
    t = np.clip(counts / vmax, 0, 1)
    rgba['rgb'][:,0] = np.uint8(64+191*t)
    rgba['rgb'][:,1] = np.uint8(65+100*(1-t))
    rgba['rgb'][:,2] = np.uint8(200-160*t)
    with path.open('wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\n'
                 'comment HR conditioning tokens; geometric sample points ONLY\n'
                 f'element vertex {len(coords)}\n'
                 'property float x\nproperty float y\nproperty float z\n'
                 'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                 'end_header\n').encode('ascii'))
        f.write(rgba.tobytes())


class Orbit3DExportCascadeHRScaffoldV9:
    """Read-only VAE upsample and EXACT production coordinate quantization, no Cascade."""
    CATEGORY = 'Orbit to 3D/Experimental'
    FUNCTION = 'run'
    OUTPUT_NODE = True
    RETURN_TYPES = ('SHAPE_SLAT', 'STRING', 'INT', 'INT')
    RETURN_NAMES = ('shape_slat_unchanged', 'scaffold_npz', 'hr_resolution', 'hr_token_count')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'pipeline': ('TRELLIS2PIPELINE',),
            'shape_slat': ('SHAPE_SLAT',),
            'from_resolution': ('INT', {'default':512,'min':512,'max':512}),
            'to_resolution': ([1024, 1536], {'default':1536}),
            'sparse_structure_resolution': ('INT', {'default':64,'min':32,'max':128,'step':4}),
            'max_num_tokens': ('INT', {'default':999999,'min':1,'max':999999}),
            'output_directory': ('STRING', {'default':'/home/ai/orbit3d-cache/experiments/v9_hr_scaffold','multiline':False}),
        }}

    @torch.no_grad()
    def run(self, pipeline, shape_slat, from_resolution=512, to_resolution=1536,
            sparse_structure_resolution=64, max_num_tokens=999999,
            output_directory='/home/ai/orbit3d-cache/experiments/v9_hr_scaffold'):
        _require_pixal3d_mv(pipeline)
        if not isinstance(shape_slat, SparseTensor):
            raise TypeError('shape_slat must be a sparse SHAPE_SLAT from the saved filled Shape512 run')
        if int(from_resolution) != 512:
            raise ValueError('Only Shape512 starting resolution supported')
        if int(to_resolution) not in (1024, 1536):
            raise ValueError('Only HR resolution 1024/1536 supported')
        denom = float(from_resolution) * (float(sparse_structure_resolution) / 32.0)
        if denom <= 0:
            raise ValueError('Invalid sparse_structure_resolution')
        outroot=Path(output_directory).expanduser().resolve()
        outroot.mkdir(parents=True, exist_ok=True)
        out=outroot/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
        out.mkdir()
        dec=None
        try:
            pipeline.load_shape_slat_decoder()
            dec=pipeline.models['shape_slat_decoder']
            if pipeline.low_vram:
                dec.to(pipeline.device)
                dec.low_vram=True
            raw=dec.upsample(shape_slat, upsample_times=4)
            if raw.ndim!=2 or raw.shape[1]!=4 or raw.dtype not in (torch.int32, torch.int64):
                raise ValueError('Unexpected VAE upsample coordinates')
            raw_cpu=raw.detach().to(device='cpu', dtype=torch.int32).contiguous().numpy()
            del raw
        finally:
            if dec is not None and pipeline.low_vram:
                dec.cpu()
                dec.low_vram=False
            if dec is not None and not pipeline.keep_models_loaded:
                pipeline.unload_shape_slat_decoder()

        trial=[]
        chosen=None
        resolution=int(to_resolution)
        while True:
            grid=resolution//16
            coords,counts=_aggregate_quantized(raw_cpu,grid,denom)
            n=len(coords)
            trial.append({'hr_resolution':resolution,'grid_resolution':grid,'unique_tokens':n})
            if n<int(max_num_tokens):
                chosen=(resolution,grid,coords,counts)
                break
            resolution-=128
            if resolution<1024 and to_resolution>=1024:
                chosen=(1024,64,*_aggregate_quantized(raw_cpu,64,denom))
                trial.append({'hr_resolution':1024,'grid_resolution':64,'unique_tokens':len(chosen[2]),
                              'forced_minimum':True})
                break
            if resolution<512:
                raise RuntimeError('Cascade coordinate quantization below 512 not supported')
        selected_res,grid,coords,counts=chosen
        hash_value=_slat_fingerprint(shape_slat)
        npz=out/'hr_scaffold.npz'
        np.savez_compressed(npz,
            token_xyz=coords, child_counts=counts.astype(np.int32),
            grid_resolution=np.int32(grid), hr_resolution=np.int32(selected_res),
            from_resolution=np.int32(from_resolution),
            sparse_structure_resolution=np.int32(sparse_structure_resolution),
            vae_child_coordinate_count=np.int64(len(raw_cpu)),
            shape_slat_sha256=np.array(hash_value),
            voxel_world_min=np.float32(-0.5),voxel_world_max=np.float32(0.5),
        )
        _write_tokens_ply(out/'hr_scaffold_centers.ply',coords,grid,counts)
        report={
            'format':'orbit3d_v9_hr_scaffold_readonly_v1',
            'shape512_regenerated':False,'cascade_executed':False,'input_slat_modified':False,
            'shape_slat_sha256':hash_value,'from_resolution':int(from_resolution),
            'requested_hr_resolution':int(to_resolution),'hr_resolution':int(selected_res),
            'hr_grid':grid,'vae_children_1024':int(len(raw_cpu)),
            'hr_token_count':len(coords),'quantization_denominator':denom,
            'token_occupancy_count_min':int(counts.min()) if len(counts) else 0,
            'token_occupancy_count_max':int(counts.max()) if len(counts) else 0,
            'resolution_trials':trial,
            'notes':[
                'Same VAE upsample and quantization formula as production Cascade sample().',
                'This is only the spatial scaffold BEFORE MV conditioning; no HR generation.',
                'The point cloud represents spatial token centers, not mesh vertices or surface labels.',
                'Multiple disconnected mesh surfaces can project to the SAME HR token; deletion then unsafe.'
            ]
        }
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(f'[Orbit V9 HR scaffold] {len(raw_cpu):,} VAE coordinates -> {len(coords):,} '
              f'conditioning tokens grid={grid}; {npz}',flush=True)
        return (shape_slat, str(npz), int(selected_res), int(len(coords)))
