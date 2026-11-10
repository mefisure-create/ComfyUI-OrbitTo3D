"""Global S5/S7 VAE-prune ablations from the EXACT cached Shape512 SLat.

Experimental only: no Shape512 regeneration, no Cascade, no production changes.
One decoder is temporarily hooked after the 128->256 subdivision. The hook
is removed in a finally block; weights and SLat are untouched.
"""
from __future__ import annotations

import gc
import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ..orbit_nodes import _require_pixal3d_mv
from ..trellis2.modules.sparse import SparseTensor
from .experimental_vae_prune_contract import validate_vae_prune_labels, pack_256


_GRID = 256
_VARIANTS = ("baseline", "intersection", "s7", "union", "all_audit_scope")


def _validated_4d(coords, resolution, label):
    arr = np.asarray(coords)
    if arr.ndim != 2 or arr.shape[1] != 4 or arr.dtype.kind not in "iu":
        raise ValueError(f"{label}: expected integer COORDS [N,4], got {arr.shape}/{arr.dtype}")
    if not len(arr):
        raise ValueError(f"{label}: empty coordinates")
    arr = arr.astype(np.int64, copy=False)
    if np.any(arr[:, 0] != 0) or np.any(arr[:, 1:] < 0) or np.any(arr[:, 1:] >= resolution):
        raise ValueError(f"{label}: only one batch with coordinates in [0,{resolution}) supported")
    keys = _pack(arr, resolution)
    if len(np.unique(keys)) != len(keys):
        raise ValueError(f"{label}: duplicate coordinates")
    return arr, keys


def _pack(arr, resolution):
    """Collision-free key for B=1 and 3-D [0,R), including in the hook."""
    return ((arr[:, 1].astype(np.int64) * resolution + arr[:, 2]) * resolution + arr[:, 3])


def _prepare_live_selections(shape_slat, vae_prune_labels):
    """Validate exact SLat provenance and attach in-memory S5/S7 selections."""
    validated=validate_vae_prune_labels(shape_slat,vae_prune_labels)
    s5=validated['s5_marked_keys']
    s7=validated['s7_all_keys']
    return {
        'source_audit':validated.get('audit_report_path'),
        's5_marked':s5,'s7_all':s7,
        'audit_scope':validated['audit_scope'],
        'audit_scope_keys':validated['audit_scope_keys'],
        'all_vae256_keys':validated['all_vae256_keys'],
        'selection':{
            'baseline':np.empty(0,dtype=np.int64),
            'intersection':np.intersect1d(s5,s7,assume_unique=True),
            's7':s7,
            'union':np.union1d(s5,s7),
            'all_audit_scope':validated['audit_scope_keys'],
        },
    }


def _check_baseline_stage_256(allcoords, inputs):
    coords, keys=_validated_4d(allcoords,_GRID,'live VAE256')
    if not np.array_equal(np.sort(keys),inputs['all_vae256_keys']):
        raise ValueError('Audit VAE256 occupancy differs from V8 baseline: wrong Shape512 SLat, '
                         'decoder changed, or stale ComfyUI cache')
    if (not np.isin(inputs['s5_marked'],keys).all()
            or not np.isin(inputs['s7_all'],keys).all()
            or not np.isin(inputs['audit_scope_keys'],keys).all()):
        raise ValueError('Audit labels/scope include nonexistent baseline VAE children')
    return keys


def _write_points(path, coords, grid, color):
    """Streaming binary PLY of grid-point centers; NO surface inference."""
    xyz = np.asarray(coords, dtype=np.float32)
    if len(xyz) and xyz.shape[1] == 4:
        xyz = xyz[:, 1:]
    # EXACT V5/V7 morphology and raw-lamellar PLY convention.
    # Do not use _write_score_ply's separate parent64-centred convention here:
    # doing so shifts stage256_removed relative to Audit candidate-only PLYs.
    xyz=(xyz+0.5)/float(grid)-0.5
    with Path(path).open("wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                "comment Orbit3D VAE coordinates, point cloud (not mesh)\n"
                f"element vertex {len(xyz)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n").encode("ascii"))
        color = np.array(color, dtype=np.uint8)
        for lo in range(0, len(xyz), 131072):
            p = xyz[lo:lo+131072]
            record = np.empty(len(p), dtype=[("xyz","<f4",(3,)),("rgb","u1",(3,))])
            record["xyz"] = p
            record["rgb"] = color
            f.write(record.tobytes())


def _write_mesh_ply(path, mesh):
    """Write native decoder vertices/faces, chunked to avoid trimesh/RAM copies."""
    v, fcs = mesh.vertices, mesh.faces
    if not (isinstance(v, torch.Tensor) and isinstance(fcs, torch.Tensor)):
        raise TypeError("Decoder mesh must expose torch Tensor vertices and faces")
    if v.ndim != 2 or v.shape[1] != 3 or fcs.ndim != 2 or fcs.shape[1] != 3:
        raise ValueError(f"Invalid mesh vertex/face shapes: {v.shape}, {fcs.shape}")
    nv,nf = int(len(v)), int(len(fcs))
    if nv < 1 or nf < 1:
        raise ValueError("Decoder returned an empty mesh")
    if nv >= 2**31:
        raise ValueError("PLY writer supports fewer than 2^31 vertices")
    bb_lo=np.array([float('inf')]*3)
    bb_hi=np.array([-float('inf')]*3)
    with Path(path).open("wb") as out:
        out.write(("ply\nformat binary_little_endian 1.0\n"
                   "comment Native Shape512 VAE decoder mesh; no mesh postprocessing\n"
                   f"element vertex {nv}\n"
                   "property float x\nproperty float y\nproperty float z\n"
                   f"element face {nf}\n"
                   "property list uchar int vertex_indices\nend_header\n").encode("ascii"))
        for lo in range(0,nv,250000):
            block=v[lo:lo+250000].detach().to('cpu',dtype=torch.float32).numpy()
            if not np.isfinite(block).all():
                raise ValueError("Decoded mesh contains nonfinite vertices")
            bb_lo=np.minimum(bb_lo,block.min(0));bb_hi=np.maximum(bb_hi,block.max(0))
            out.write(np.asarray(block,dtype='<f4',order='C').tobytes())
        for lo in range(0,nf,250000):
            block=fcs[lo:lo+250000].detach().to('cpu',dtype=torch.int64).numpy()
            if block.min()<0 or block.max()>=nv:
                raise ValueError("Decoded mesh contains out-of-range face indices")
            row=np.empty(len(block),dtype=[('n','u1'),('f','<i4',(3,))])
            row['n']=3;row['f']=block.astype(np.int32,copy=False)
            out.write(row.tobytes())
    return {"vertices":nv,"faces":nf,"bbox_min":bb_lo.tolist(),"bbox_max":bb_hi.tolist()}


class Orbit3DShape512VAEGlobalPruneDecode:
    """Run global pruning variants on exact cached SLat, export meshes for Blender."""
    CATEGORY="Orbit to 3D/Experimental"
    FUNCTION="run"
    RETURN_TYPES=("STRING","STRING","INT")
    RETURN_NAMES=("report_path","last_mesh_path","last_removed_256")
    OUTPUT_NODE=True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pipeline":("TRELLIS2PIPELINE",),
            "shape_slat":("SHAPE_SLAT",),
            "decode_resolution":("INT",{"forceInput":True}),
            "vae_prune_labels":("ORBIT3D_VAE_PRUNE_LABELS",),
            "variant":(["all_four","all_five",*_VARIANTS],{"default":"all_four"}),
            "use_tiled":("BOOLEAN",{"default":True}),
            "save_256_previews":("BOOLEAN",{"default":True}),
            "report_directory":("STRING",{"default":"/home/ai/orbit3d-cache/experiments/vae_prune_v8"}),
        }}

    @torch.no_grad()
    def run(self,pipeline,shape_slat,decode_resolution,vae_prune_labels,variant="all_four",
            use_tiled=True,save_256_previews=True,
            report_directory="/home/ai/orbit3d-cache/experiments/vae_prune_v8"):
        _require_pixal3d_mv(pipeline)
        if not isinstance(shape_slat,SparseTensor):
            raise TypeError("shape_slat must be a SHAPE_SLAT SparseTensor, not decoded mesh")
        if int(decode_resolution)!=1024:
            raise ValueError("V8 is validated only for 64^3 Shape512 SLat decoded at resolution=1024")
        if variant not in ("all_four","all_five",*_VARIANTS):
            raise ValueError(f"Unrecognized variant: {variant}")
        inputs=_prepare_live_selections(shape_slat,vae_prune_labels)
        # Whole-model means no local ROI. "all_audit_scope" is deliberately
        # restricted to Force Fill descendants; never erase every VAE voxel.
        if variant in ("all_audit_scope","all_five") and inputs['audit_scope'] != 'filled_descendants':
            raise ValueError(
                "all_audit_scope requires Audit audit_scope=filled_descendants. "
                "Using all_descendants would delete the entire model and is forbidden.")
        outdir=Path(report_directory).expanduser().resolve()
        outdir.mkdir(parents=True,exist_ok=True)
        out=outdir/(datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8])
        out.mkdir()
        device=torch.device(pipeline.device)
        pipeline.load_shape_slat_decoder()
        dec=pipeline.models['shape_slat_decoder']
        hooks=[]
        report={
            'format':'ORBIT_SHAPE512_VAE_GLOBAL_PRUNE_V8',
            'shape512_regenerated':False,'cascade_executed':False,
            'input_slat_modified':False,'spatial_scope':'whole_model',
            'resolution':int(decode_resolution),'prune_grid_resolution':256,
            'audit_report_path':inputs['source_audit'],
            'labels_source':'live_Audit_&_Filter',
            'input_slat_tokens':int(len(shape_slat.coords)),
            's5_marked':int(len(inputs['s5_marked'])),
            's7_marked':int(len(inputs['s7_all'])),
            's5_s7_intersection':int(len(inputs['selection']['intersection'])),
            's5_s7_union':int(len(inputs['selection']['union'])),
            'audit_scope':inputs['audit_scope'],
            'audit_scope_count':int(len(inputs['audit_scope_keys'])),
            'all_audit_scope_note':'All descendants of Force Fill parents at VAE256; no S5/S7 classification filter',
            'variants':{},
        }
        last_mesh='';last_removed=0
        try:
            if pipeline.low_vram:
                dec.to(device);dec.low_vram=True
            dec.set_resolution(int(decode_resolution))
            if not getattr(dec,'pred_subdiv',False) or len(dec.blocks)<5:
                raise RuntimeError("Unsupported VAE architecture: expected four 2x subdivision levels")
            stage_modules=[dec.blocks[i][-1] for i in range(4)]
            if not all(getattr(m,'pred_subdiv',False) for m in stage_modules):
                raise RuntimeError("VAE does not expose predicted subdivision at all four stages")

            # PRE-FLIGHT: compare ALL actual VAE256 children against the attached
            # audit's full occupancy. Detect stale/mismatched objects before pruning.
            seen=[]
            def baseline_capture(_mod,_arg,output):
                h=output[0] if isinstance(output,tuple) else output
                seen.append(h.coords.detach().cpu().numpy().copy())
            prehook=stage_modules[1].register_forward_hook(baseline_capture)
            try:
                shape_slat.clear_spatial_cache()
                dec.upsample(shape_slat,upsample_times=4)
            finally:
                prehook.remove()
                shape_slat.clear_spatial_cache()
            if len(seen)!=1:
                raise RuntimeError(f"Expected one baseline VAE256 capture, received {len(seen)}")
            baseline_256=seen[0]
            baseline_keys=_check_baseline_stage_256(baseline_256,inputs)
            del seen
            report['baseline_vae_256_children']=int(len(baseline_keys))
            report['baseline_vae_256_verified']=True

            variants=(list(_VARIANTS[:-1]) if variant=='all_four'
                      else list(_VARIANTS) if variant=='all_five'
                      else [variant])
            for name in variants:
                target=inputs['selection'][name]
                stats={'selection_requested':int(len(target)),
                       'stages':{},'selected_absent':0}
                kept_at_256=[]
                removed_at_256=[]
                print(f"[Orbit3D V8] starting {name}: target256={len(target)} whole-model",flush=True)

                def hook_fn(level):
                    def hook(_mod,_args,output):
                        if not isinstance(output,tuple) or len(output)!=2:
                            raise RuntimeError(f"Unexpected VAE hook return at level {level}")
                        h,sub=output
                        coords=h.coords
                        if level==256:
                            actual=coords.detach().cpu().numpy()
                            _,key=_validated_4d(actual,256,'live decoder VAE256')
                            if not np.array_equal(np.sort(key),np.sort(baseline_keys)):
                                raise RuntimeError("VAE256 coords changed across identical decoder runs; abort")
                            remove=np.isin(key,target,assume_unique=True)
                            if int(remove.sum())!=len(target):
                                raise RuntimeError("Not all requested VAE256 children exist in live decoder")
                            nrm=int(remove.sum())
                            stats['removed_256']=nrm
                            stats['stages']['256_before']=int(len(key))
                            if save_256_previews:
                                # Save only bounded 256^3 snapshots; later stages may have millions of tokens.
                                kept_at_256.append(actual[~remove].copy())
                                removed_at_256.append(actual[remove].copy())
                            if nrm:
                                keep=torch.as_tensor(~remove,device=coords.device,dtype=torch.bool)
                                # SparseTensor.__getitem__ indexes BATCH, not POINTS.
                                # Rebuild with retained rows; discard spatial caches built
                                # on pre-prune coordinates to avoid corrupt neighbor maps.
                                fresh=SparseTensor(
                                    feats=h.feats[keep].contiguous(),
                                    coords=coords[keep].contiguous(),
                                    shape=h.shape,
                                )
                                # Some sparse backends do not accept 'scale' as
                                # a constructor kwarg. Preserve it explicitly.
                                fresh._scale=h._scale
                                h=fresh
                            stats['stages']['256']=int(h.coords.shape[0])
                            return (h,sub)
                        stats['stages'][str(level)]=int(coords.shape[0])
                        return None
                    return hook

                for lvl,mod in zip((128,256,512,1024),stage_modules):
                    hooks.append(mod.register_forward_hook(hook_fn(lvl)))
                try:
                    shape_slat.clear_spatial_cache()
                    decoded=dec(shape_slat,return_subs=False,useTiled=bool(use_tiled))
                finally:
                    for h in hooks:h.remove()
                    hooks.clear()
                    shape_slat.clear_spatial_cache()

                if not isinstance(decoded,(tuple,list)) or len(decoded)!=1:
                    raise RuntimeError(f"Expected single-mesh decoder output, got {type(decoded)}")
                mesh=decoded[0]
                meshfile=out/f'{name}_shape512_mesh.ply'
                stats['mesh']=_write_mesh_ply(meshfile,mesh)
                stats['mesh_file']=str(meshfile)
                if save_256_previews:
                    if len(kept_at_256)!=1 or len(removed_at_256)!=1:
                        raise RuntimeError("Missing stage256 snapshots")
                    keptfile=out/f'{name}_stage256_kept.ply'
                    removedfile=out/f'{name}_stage256_removed.ply'
                    _write_points(keptfile,kept_at_256[0],256,(58,188,160))
                    stats['stage256_kept_ply']=str(keptfile)
                    if len(removed_at_256[0]):
                        _write_points(removedfile,removed_at_256[0],256,(248,53,48))
                        stats['stage256_removed_ply']=str(removedfile)
                    else:
                        stats['stage256_removed_ply']=None  # baseline only
                report['variants'][name]=stats
                (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
                print(f"[Orbit3D V8] completed {name}: removed={stats['removed_256']}, "
                      f"mesh V={stats['mesh']['vertices']:,} F={stats['mesh']['faces']:,}; "
                      f"stages={stats['stages']}; file={meshfile}",flush=True)
                last_mesh=str(meshfile);last_removed=stats['removed_256']
                del mesh,decoded,kept_at_256,removed_at_256
                gc.collect()
                if torch.cuda.is_available():torch.cuda.empty_cache()
        finally:
            for h in hooks:h.remove()
            if pipeline.low_vram:
                dec.cpu();dec.low_vram=False
            if not pipeline.keep_models_loaded:
                pipeline.unload_shape_slat_decoder()
            del dec
        final=out/'report.json'
        final.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        print(f'[Orbit3D V8] report={final}',flush=True)
        return (str(final),last_mesh,int(last_removed))
