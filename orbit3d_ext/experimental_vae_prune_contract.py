"""Strict in-memory interface between the experimental HR audit and V8 decoder.

The contract contains only coordinate labels, plus a fingerprint of the exact
Shape512 SLat. No latent features or entire SparseTensor are stored in it.
"""
from __future__ import annotations

import hashlib
import numpy as np
import torch

_FORMAT = 'ORBIT3D_VAE_PRUNE_LABELS_V2'
_GRID = 256


def _validate_coords(coords,grid,name):
    arr=np.asarray(coords)
    if arr.ndim!=2 or arr.shape[1]!=4 or arr.dtype.kind not in 'iu':
        raise ValueError(f'{name}: expected integer coordinates [N,4], got {arr.shape}/{arr.dtype}')
    if not len(arr) and name == 'full VAE256':
        raise ValueError(f'{name}: empty occupancy')
    arr=np.ascontiguousarray(arr,dtype=np.int64)
    if len(arr) and (np.any(arr[:,0]!=0) or np.any(arr[:,1:]<0) or np.any(arr[:,1:]>=grid)):
        raise ValueError(f'{name}: expected a single batch and coordinates in [0,{grid})')
    return arr


def pack_256(coords):
    """Collision-free 256^3 packed coordinate keys, batch zero only."""
    arr=_validate_coords(coords,_GRID,'VAE256 coords')
    return ((arr[:,1]*_GRID+arr[:,2])*_GRID+arr[:,3]).astype(np.int64,copy=False)


def _sha_tensor(hash_obj, t):
    if not isinstance(t,torch.Tensor):
        raise TypeError('Shape512 SLat must contain torch Tensor coords and feats')
    tt=t.detach().contiguous().to('cpu')
    hash_obj.update(str(tuple(tt.shape)).encode('ascii'))
    hash_obj.update(str(tt.dtype).encode('ascii'))
    # Byte view handles bfloat16 without loss or unsafe conversion.
    hash_obj.update(memoryview(tt.view(torch.uint8).numpy()).cast('B'))


def shape_slat_fingerprint(slat):
    digest=hashlib.sha256()
    _sha_tensor(digest,slat.coords)
    _sha_tensor(digest,slat.feats)
    return digest.hexdigest()


def _unique_sorted_keys(keys,name):
    keys=np.asarray(keys)
    if keys.ndim!=1 or keys.dtype.kind not in 'iu':
        raise ValueError(f'{name}: invalid packed keys')
    keys=np.ascontiguousarray(keys,dtype=np.int64)
    if np.any(keys<0) or np.any(keys>=_GRID**3):
        raise ValueError(f'{name}: keys outside 256^3')
    unique=np.unique(keys)
    if len(unique)!=len(keys):
        raise ValueError(f'{name}: repeated VAE child keys')
    return unique


def make_vae_prune_labels(*,shape_slat,fine_coords,scope_coords,s5_coords,s5_class,
                          s7_coords,audit_scope):
    fine=pack_256(fine_coords)
    if not len(fine):
        raise ValueError('No VAE256 children')
    scope=pack_256(scope_coords)
    if audit_scope not in ('filled_descendants','all_descendants'):
        raise ValueError('Unsupported audit_scope for VAE prune labels')
    s5=pack_256(s5_coords)
    s7=pack_256(s7_coords)
    klass=np.asarray(s5_class)
    if klass.shape!=(len(s5),) or klass.dtype.kind not in 'iu':
        raise ValueError('Invalid S5 morphology class labels')
    if np.any(klass>3) or np.any(klass<0):
        raise ValueError('S5 classification must be in [0,3]')
    all_keys=_unique_sorted_keys(fine,'full VAE256')
    scope_keys=_unique_sorted_keys(scope,'complete Audit scope')
    _unique_sorted_keys(s5,'S5 audited coordinates')
    if len(scope_keys)==0:
        raise ValueError('Empty complete Audit scope')
    if not np.isin(scope_keys,all_keys).all():
        raise ValueError('Audit scope contains coordinates outside complete VAE256 occupancy')
    if not np.array_equal(np.sort(s5),scope_keys):
        raise ValueError('S5 audited coordinates must cover the entire Audit scope; partial samples forbidden')
    if audit_scope == 'all_descendants' and not np.array_equal(scope_keys,all_keys):
        raise ValueError('all_descendants must cover all VAE256 coordinates')
    if audit_scope == 'filled_descendants' and len(scope_keys)>=len(all_keys):
        raise ValueError('filled_descendants cannot include the entire VAE256 occupancy')
    _unique_sorted_keys(s7,'S7 raw lamellar coordinates')
    if not np.isin(s5,all_keys).all() or not np.isin(s7,all_keys).all():
        raise ValueError('S5/S7 contains coordinates outside captured VAE256 occupancy')
    return {
        'format':_FORMAT,
        'grid_resolution':_GRID,
        'slat_fingerprint':shape_slat_fingerprint(shape_slat),
        'all_vae256_keys':all_keys,
        'audit_scope_keys':scope_keys,
        'audit_scope_complete':True,
        's5_marked_keys':_unique_sorted_keys(s5[klass>0],'S5 marked'),
        's7_all_keys':_unique_sorted_keys(s7,'S7 marked'),
        'audit_scope':str(audit_scope),
        'audit_report_path':None,
    }


def validate_vae_prune_labels(shape_slat, labels):
    if not isinstance(labels,dict) or labels.get('format')!=_FORMAT:
        raise ValueError('Connect vae_prune_labels from the current Orbit3D HR Subvoxel Audit; '
                         'old audit or incompatible output supplied')
    if int(labels.get('grid_resolution',0))!=_GRID:
        raise ValueError('V8 requires labels from 256^3 audit')
    if shape_slat_fingerprint(shape_slat)!=labels.get('slat_fingerprint'):
        raise ValueError('The connected audit labels were generated using a DIFFERENT Shape512 SLat')
    allkeys=_unique_sorted_keys(labels['all_vae256_keys'],'full VAE256 labels')
    if labels.get('audit_scope') not in ('filled_descendants','all_descendants'):
        raise ValueError('Unknown Audit scope')
    if labels.get('audit_scope_complete') is not True:
        raise ValueError('VAE prune requires fully audited scope')
    scope_keys=_unique_sorted_keys(labels['audit_scope_keys'],'complete Audit scope')
    if not len(scope_keys) or not np.isin(scope_keys,allkeys).all():
        raise ValueError('Invalid Audit scope coordinates')
    if labels['audit_scope']=='all_descendants' and not np.array_equal(scope_keys,allkeys):
        raise ValueError('Incomplete all_descendants Audit scope')
    if labels['audit_scope']=='filled_descendants' and len(scope_keys)>=len(allkeys):
        raise ValueError('Invalid filled_descendants Audit scope: includes entire VAE')
    for k in ('s5_marked_keys','s7_all_keys'):
        ck=_unique_sorted_keys(labels[k],k)
        if not np.isin(ck,allkeys).all():
            raise ValueError(f'{k}: labels are outside full VAE256 occupancy')
    if not np.isin(labels['s5_marked_keys'],scope_keys).all():
        raise ValueError('S5 labels lie outside Audit scope')
    return labels
