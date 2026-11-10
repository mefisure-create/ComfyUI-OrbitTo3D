"""Diagnostic-only free-space audit at a VAE subdivision level (128/256/512).

Do not feed subvoxel COORDS straight to ShapeHR. No geometry is edited;
Shape512 and Cascade behavior remain unchanged. Fail closed on untrusted depth.
"""
from __future__ import annotations

import csv
from collections import deque
import json
import math
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..orbit_nodes import _require_pixal3d_mv
from ..trellis2.modules.sparse import SparseTensor
from ..trellis2.trainers.flow_matching.mixins.image_conditioned_proj import (
    ProjGrid, compute_relative_calc_mat, project_points_to_image_batch,
)
from .experimental_vae_prune_contract import make_vae_prune_labels
from .experimental_hr_free_space_nodes import (
    _validate, _getmap, _sample, _apply_axis_x_offset, _fit_depth,
)


def _fine_world(xyz: np.ndarray, grid: int, mesh_scale: float, device):
    """Map children to the same parent-center grid as production ProjGrid(R=64).

    Parent centers are at 2*i/63-1, not (i+.5)*2/64-1. A child's
    center is a within-parent offset, so our exact SS64 lattice is preserved.
    """
    factor = grid // 64
    arr = torch.as_tensor(xyz, dtype=torch.float32, device=device)
    parent = torch.floor(arr / factor)
    local = (torch.remainder(arr, factor) + .5) / factor - .5
    mesh = 2.0 * (parent + local) / 63.0 - 1.0
    world = torch.stack((mesh[:, 0], -mesh[:, 2], mesh[:, 1]), dim=1)
    return (world / (2.0 * mesh_scale)).unsqueeze(0)


def _project_points(pts, calc, fov, dx):
    xy, dep, _ = project_points_to_image_batch(pts, calc, fov, 512)
    cpu_xy = xy[0].detach().cpu().float()
    cpu_dep = dep[0].detach().cpu().float()
    corrected, valid = _apply_axis_x_offset(cpu_xy, cpu_dep, dx)
    return corrected, cpu_dep.numpy(), valid


def _subcell_probe(centers, half_span, calc, fov, dx, chunk=12000):
    """Project center and eight 3D corners, return valid [N] and XY [N,9,2]."""
    from itertools import product
    signs = torch.tensor(list(product((-1., 1.), repeat=3)),
                         device=centers.device, dtype=centers.dtype)
    n = centers.shape[1]
    points = np.empty((n, 9, 2), np.float32)
    valid = np.zeros(n, dtype=np.bool_)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        p = centers[:, start:stop, :]
        # The Cartesian axes are already Blender coordinates: offsets are
        # isotropic, and all sign combinations are invariant under rotation.
        samples = torch.cat((p[:, :, None, :],
                  p[:, :, None, :] + signs[None,None,:,:] * float(half_span)), dim=2)
        xy, dep, _ = project_points_to_image_batch(
            samples.reshape(1, -1, 3), calc, fov, 512)
        xy_c, okay = _apply_axis_x_offset(
            xy[0].detach().cpu().float(), dep[0].detach().cpu().float(), dx)
        m = stop - start
        points[start:stop] = xy_c.numpy().reshape(m, 9, 2)
        valid[start:stop] = okay.reshape(m, 9).all(axis=1)
    return points, valid


def _probe_foreground_fraction(mask: torch.Tensor, points: np.ndarray):
    """Fraction of nine samples outside a conservatively dilated foreground."""
    img = mask.numpy()
    H,W = img.shape
    flat = points.reshape(-1,2)
    good = np.isfinite(flat).all(axis=1)
    x = np.clip(np.nan_to_num(flat[:,0],nan=0,posinf=0,neginf=0).round().astype(np.int64),0,W-1)
    y = np.clip(np.nan_to_num(flat[:,1],nan=0,posinf=0,neginf=0).round().astype(np.int64),0,H-1)
    outside = ((img[y,x] < .05) & good).reshape(-1,9)
    return outside.mean(axis=1).astype(np.float32)


def _visible_anchor_zbuffer(pts, calc, fov, dx, mask, max_depth_anchors, unit64):
    """Nearest projected ORIGINAL-parent child surface per unique camera pixel.

    This is a pixel-wise zbuffer rather than a fit to all occupied SS voxels.
    It is still a proxy for the physical surface, so residual QA is required.
    """
    count = pts.shape[1]
    step = max(1, math.ceil(count / max_depth_anchors))
    xy, z, valid = _project_points(pts[:,::step,:], calc, fov, dx)
    fg = _sample(mask, xy) > .8
    valid &= fg
    ix = xy[:,0].round().long().clamp(0,511).numpy()
    iy = xy[:,1].round().long().clamp(0,511).numpy()
    idx = iy*512 + ix
    zbuf = np.full(512*512, np.inf, dtype=np.float32)
    np.minimum.at(zbuf, idx[valid], z[valid].astype(np.float32))
    return zbuf, int(valid.sum()), int(np.isfinite(zbuf).sum()), step


def _write_score_ply(path, xyz, resolution, score, parent_original, confirmed):
    """Write colored point cloud in model coordinates; no mesh is generated."""
    factor=resolution//64
    arr=xyz.astype(np.float64)
    parent=np.floor_divide(arr,factor)
    local=(np.remainder(arr,factor)+.5)/factor-.5
    norm=(parent+local)/63.0-.5
    with open(path,'w',encoding='utf-8') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(xyz)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('property float score\nproperty uchar parent_original\nend_header\n')
        for p,s,orig,conf in zip(norm,score,parent_original,confirmed):
            # cyan = original-parent descendants; orange/red = contested
            # filled descendants, dark blue = no evidence.
            if orig: col=(38,190,176)
            elif conf: col=(247,57,45)
            elif s>=.35: col=(248,162,40)
            else: col=(90,110,148)
            f.write(f'{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} '
                    f'{col[0]} {col[1]} {col[2]} {float(s):.5f} {int(orig)}\n')


def _components6(xyz: np.ndarray, confirmed: np.ndarray, grid: int):
    """6-connected components in CONFIRMED sampled children; never infer negatives.

    Returns sample-aligned component IDs (-1 for unconfirmed) plus
    component metadata sorted by descending population. All operations are
    sparse; no full 512^3 dense mask allocation.
    """
    selected = np.flatnonzero(confirmed)
    labels = np.full(len(xyz), -1, dtype=np.int32)
    if not len(selected):
        return labels, []
    p = xyz[selected].astype(np.int32)
    ids = (p[:, 0].astype(np.int64)*grid*grid +
           p[:, 1].astype(np.int64)*grid + p[:, 2])
    idx = {int(k): i for i, k in enumerate(ids)}
    roots = np.arange(len(ids), dtype=np.int32)
    sizes = np.ones(len(ids), dtype=np.int32)
    def find(j):
        while roots[j] != j:
            roots[j] = roots[roots[j]]
            j = int(roots[j])
        return j
    def merge(i, j):
        a, b = find(i), find(j)
        if a == b:
            return
        if sizes[a] < sizes[b]:
            a, b = b, a
        roots[b] = a
        sizes[a] += sizes[b]
    directions = ((0, grid*grid), (1, grid), (2, 1))
    for i, (x, y, z) in enumerate(p):
        xyz_i = (x, y, z)
        for axis, offset in directions:
            if xyz_i[axis] + 1 < grid:
                j = idx.get(int(ids[i] + offset))
                if j is not None:
                    merge(i, j)
    grouped = {}
    for i in range(len(p)):
        root = find(i)
        grouped.setdefault(root, []).append(i)
    components = sorted(grouped.values(), key=lambda members: (-len(members), members[0]))
    metadata = []
    for cid, members in enumerate(components):
        pts = p[members]
        labels[selected[np.asarray(members)]] = cid
        bbox_min = pts.min(axis=0).tolist()
        bbox_max = pts.max(axis=0).tolist()
        dims = (pts.max(axis=0) - pts.min(axis=0) + 1).tolist()
        # PCA is an inspection hint only: thin actual surfaces can be planar.
        shape = None
        if len(members) >= 6:
            covariance = np.cov(pts.astype(np.float64), rowvar=False)
            eig = np.maximum(np.linalg.eigvalsh(covariance), 0)
            shape = {
                'eigenvalues_ascending': [float(a) for a in eig],
                'thinness_ratio': float(eig[0] / max(eig[2], 1e-12)),
                'width_ratio': float(eig[1] / max(eig[2], 1e-12)),
            }
        parent_count = int(np.unique(pts // (grid//64), axis=0).shape[0])
        metadata.append({
            'component_id':cid,'n_confirmed_children':len(members),
            'parent64_cells_touched':parent_count,
            'bbox_min_subvoxel':bbox_min,'bbox_max_subvoxel':bbox_max,
            'bbox_size_subvoxel':dims,'shape_diagnostic':shape,
        })
    return labels, metadata


def _parent_topology(coords, parentkeys_all, selected_xyz, selected_parentkeys,
                     originalkeys, confirmed, component_id, mean_score):
    """Exact denominator from ALL generated in-scope VAE children.

    Report incomplete sampling explicitly. Counts and components are never
    presented as a validated mesh removal operation.
    """
    all_keys, total_children = np.unique(parentkeys_all, return_counts=True)
    parent_idx = np.searchsorted(all_keys, selected_parentkeys)
    if np.any(all_keys[parent_idx] != selected_parentkeys):
        raise AssertionError('Selected parent not in complete VAE hierarchy')
    n_parents = len(all_keys)
    measured = np.bincount(parent_idx, minlength=n_parents).astype(np.int32)
    positive = np.bincount(parent_idx, weights=confirmed.astype(np.int32),
                          minlength=n_parents).astype(np.int32)
    mean_sum = np.bincount(parent_idx, weights=mean_score,
                           minlength=n_parents)
    pmean = mean_sum / np.maximum(measured, 1)
    largest = np.zeros(n_parents, np.int32)
    component_count = np.zeros(n_parents, np.int32)
    ok = confirmed & (component_id >= 0)
    if np.any(ok):
        pairs = np.stack((parent_idx[ok], component_id[ok]), axis=1)
        seen, pair_counts = np.unique(pairs, axis=0, return_counts=True)
        np.maximum.at(largest, seen[:,0], pair_counts)
        np.add.at(component_count, seen[:,0], 1)
    covered = measured == total_children
    ratio = positive / np.maximum(measured, 1)
    connected_share = largest / np.maximum(positive, 1)
    risk = np.zeros(n_parents, dtype=np.uint8)
    risk[positive > 0] = 1      # evidence exists, not yet connected
    risk[(positive >= 4) & (largest >= 3)] = 2  # coherent local patch
    risk[(ratio >= 0.5) & (positive >= 8) & (largest >= 6)] = 3
    # Incomplete parents cannot receive 'dominant' confidence by sampling.
    risk[~covered] = np.minimum(risk[~covered], 1)
    parent_xyz = np.stack((all_keys//4096, (all_keys//64)%64,
                           all_keys%64), axis=1).astype(np.int16)
    return {
        'coords':parent_xyz,
        'n_generated_children':total_children.astype(np.int32),
        'n_audited_children':measured,
        'n_confirmed_children':positive,
        'sample_coverage':(measured/np.maximum(total_children,1)).astype(np.float32),
        'confirmed_fraction_of_audited':ratio.astype(np.float32),
        'largest_component_children':largest,
        'largest_component_fraction_of_confirmed':connected_share.astype(np.float32),
        'n_connected_components':component_count,
        'mean_score':pmean.astype(np.float32),
        'is_complete':covered,
        'risk_class':risk,
        'original_parent':np.isin(all_keys,originalkeys),
    }


def _write_parent_csv(path, summary):
    fields = ('x','y','z','n_generated_children','n_audited_children',
              'n_confirmed_children','sample_coverage','confirmed_fraction_of_audited',
              'largest_component_children','largest_component_fraction_of_confirmed',
              'n_connected_components','mean_score','is_complete',
              'risk_class','original_parent')
    order = np.lexsort((
        -summary['n_confirmed_children'],
        -summary['largest_component_children'],
        -summary['confirmed_fraction_of_audited'],
        -summary['risk_class'].astype(np.int32),
    ))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for i in order:
            p = summary['coords'][i]
            record = (int(p[0]),int(p[1]),int(p[2]))
            record += tuple(int(summary[k][i]) if summary[k].dtype.kind in 'biu'
                            else round(float(summary[k][i]), 6)
                            for k in fields[3:])
            writer.writerow(record)


def _write_parent_ply(path, summary):
    # Risk categories are diagnostics only, not labels of erroneous geometry.
    palette = np.array([[120,140,155],[251,186,71],
                        [238,102,53],[216,45,56]],dtype=np.uint8)
    pts = (summary['coords'].astype(np.float64)+.5)/64.0-.5
    with path.open('w',encoding='utf-8') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(pts)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('property float confirmed_fraction\nproperty int confirmed_children\nend_header\n')
        for i,point in enumerate(pts):
            c = palette[summary['risk_class'][i]]
            f.write(f'{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} '
                    f'{int(c[0])} {int(c[1])} {int(c[2])} '
                    f'{summary["confirmed_fraction_of_audited"][i]:.5f} '
                    f'{int(summary["n_confirmed_children"][i])}\n')


def _depth_spatial_holdout_qa(depth_img, zbuffer, goodflat, unit64,
                              tiles=4, min_anchors=120):
    """Out-of-sample LOCAL affine depth QA. Never used in deletion evidence.

    Divide frame into tiles, fit even checkerboard rays, score odd pixels.
    A local improvement alone does not prove physical calibration correctness.
    """
    measured=depth_img.reshape(512,512)
    rendered=zbuffer.reshape(512,512)
    accepted=goodflat.reshape(512,512)
    records=[]
    for row in range(tiles):
        y0,y1=row*512//tiles,(row+1)*512//tiles
        for col in range(tiles):
            x0,x1=col*512//tiles,(col+1)*512//tiles
            m=measured[y0:y1,x0:x1].ravel()
            z=rendered[y0:y1,x0:x1].ravel()
            ok=accepted[y0:y1,x0:x1].ravel().copy()
            if ok.sum() < min_anchors:
                continue
            ys,xs=np.indices((y1-y0,x1-x0))
            even=((ys+y0+xs+x0)&1)==0
            even=even.ravel()
            fit=_fit_depth(m,z,ok&even,unit64,min_samples=30)
            val=ok&(~even)&np.isfinite(m)&np.isfinite(z)
            if not fit.get('usable',False) or val.sum()<30:
                continue
            resid=(fit['a']*m[val]+fit['b']-z[val])/unit64
            p90=float(np.quantile(np.abs(resid),.9))
            records.append({'row':row,'col':col,
                            'train_n':int(np.sum(ok&even)),
                            'test_n':int(np.sum(val)),
                            'train_p90_vox':float(fit['p90_error_vox']),
                            'heldout_p90_vox':p90,
                            'heldout_signed_median_vox':float(np.median(resid))})
    vals=np.array([v['heldout_p90_vox'] for v in records],dtype=np.float64)
    return {
       'tile_grid':tiles,'usable_tiles':len(records),
       'heldout_p90_median_vox':float(np.median(vals)) if len(vals) else None,
       'heldout_p90_best_vox':float(vals.min()) if len(vals) else None,
       'heldout_tiles_below_4vox':int(np.sum(vals<4.0)),
       'tiles':records,
       'used_for_scoring':False,
    }


def _morphology_sheet_candidates(xyz, confirmed, original_parent, grid,
                                  radius_voxels=4.0, min_neighbors=16,
                                  planar_ratio_max=0.12, width_ratio_min=0.18,
                                  propagation_steps=12,
                                  min_unseeded_component=48):
    """Weak morphological hypotheses from VAE CHILD occupancy, not mesh faces.

    Local PCA: lambda0/lambda1 estimates thinness; lambda1/lambda2 distinguishes
    a 2-D sheet from a 1-D rod. A mask-confirmed seed can propagate a bounded
    number of 26-connected steps across locally coherent sheet candidates.

    Category 2 (purple) is *mask-seeded morphological extension*, category 3
    (amber) is *unseeded morphology only*. Neither authorizes deletion.
    The original SS64 descendants are NEVER morphological candidates.
    """
    from scipy.spatial import cKDTree

    pts=np.asarray(xyz,dtype=np.int32)
    seeds=np.asarray(confirmed,dtype=bool)
    orig=np.asarray(original_parent,dtype=bool)
    n=len(pts)
    if seeds.shape!=(n,) or orig.shape!=(n,):
        raise ValueError('Morphology labels must match child coordinates')
    selected=np.flatnonzero(~orig)
    cls=np.zeros(n,dtype=np.uint8)
    cls[seeds & ~orig]=1
    normal=np.zeros((n,3),dtype=np.float32)
    planar_ratio=np.full(n,np.nan,dtype=np.float32)
    width_ratio=np.full(n,np.nan,dtype=np.float32)
    support=np.zeros(n,dtype=np.int16)
    sheet=np.zeros(n,dtype=bool)
    cluster_labels=np.full(n,-1,dtype=np.int32)
    if len(selected)<min_neighbors:
        return {'class':cls,'planar_ratio':planar_ratio,
                'width_ratio':width_ratio,'neighbors':support,
                'sheet':sheet,'cluster_id':cluster_labels,'clusters':[],
                'seeded_extension':int((cls==2).sum()),
                'unseeded_candidates':int((cls==3).sum())}

    # Query only Fill descendants; this avoids averaging the valid shell into
    # thin bridge PCA measurements. Original-parent geometry stays protected.
    fill=pts[selected].astype(np.float32)
    tree=cKDTree(fill)
    k=min(64,len(fill))
    radius=float(radius_voxels)
    # Bounded chunking avoids allocating an enormous [N,K,3] tensor.
    for start in range(0,len(fill),2048):
        stop=min(len(fill),start+2048)
        d,idx=tree.query(fill[start:stop],k=k,
                         distance_upper_bound=radius,workers=1)
        if k==1: d=d[:,None];idx=idx[:,None]
        good=np.isfinite(d)&(idx<len(fill))
        count=good.sum(axis=1)
        safe=np.where(good,idx,0)
        neighbor=fill[safe]
        w=good.astype(np.float32)
        center=(neighbor*w[:,:,None]).sum(axis=1)/np.maximum(count[:,None],1)
        delta=(neighbor-center[:,None,:])*w[:,:,None]
        covariance=np.einsum('nki,nkj->nij',delta,delta,optimize=True)/np.maximum(count[:,None,None],1)
        eigen,evec=np.linalg.eigh(covariance)
        eigen=np.maximum(eigen,0.0)
        r0=eigen[:,0]/np.maximum(eigen[:,1],1e-6)
        r1=eigen[:,1]/np.maximum(eigen[:,2],1e-6)
        # Reject near-zero spatial spreads where any PCA normal is meaningless.
        reliable=(count>=int(min_neighbors))&(eigen[:,1]>=.8)
        keep=reliable&(r0<=float(planar_ratio_max))&(r1>=float(width_ratio_min))
        rows=selected[start:stop]
        normal[rows]=evec[:,:,0]
        planar_ratio[rows]=r0
        width_ratio[rows]=r1
        support[rows]=count.astype(np.int16)
        sheet[rows]=keep

    sheet_idx=np.flatnonzero(sheet)
    lookup={int(p[0])*grid*grid+int(p[1])*grid+int(p[2]):int(i)
            for i,p in zip(sheet_idx,pts[sheet_idx])}
    neighbor_offsets=[(dx,dy,dz) for dx in (-1,0,1)
                      for dy in (-1,0,1) for dz in (-1,0,1)
                      if (dx,dy,dz)!=(0,0,0)]
    def neighbors(i):
        x,y,z=map(int,pts[i])
        for dx,dy,dz in neighbor_offsets:
            xx,yy,zz=x+dx,y+dy,z+dz
            if xx<0 or yy<0 or zz<0 or xx>=grid or yy>=grid or zz>=grid:
                continue
            j=lookup.get(xx*grid*grid+yy*grid+zz)
            if j is not None: yield j

    # A sheet-like patch is only a geometric *hypothesis*. We need not assume
    # that legitimate thin clothing is an artifact. Record all components.
    visited=np.zeros(n,dtype=bool)
    clusters=[]
    for i in sheet_idx:
        if visited[i]:continue
        cid=len(clusters)
        members=[]
        pending=[int(i)]
        visited[i]=True
        while pending:
            v=pending.pop();members.append(v)
            for j in neighbors(v):
                if not visited[j]:
                    visited[j]=True;pending.append(j)
        ix=np.asarray(members,dtype=np.int64)
        cluster_labels[ix]=cid
        box_min=pts[ix].min(axis=0)
        box_max=pts[ix].max(axis=0)
        clusters.append({'component_id':cid,'children':len(ix),
                         'bbox_min':box_min.tolist(),'bbox_max':box_max.tolist(),
                         'extent':(box_max-box_min+1).tolist(),
                         'mask_seed_touching':0,'extended_children':0,
                         'morphology_only_children':0})

    # Bridge the first 2 subvoxels to a sheet even if the visible mask evidence
    # is a line at its edge. This is controlled by the max propagation steps.
    # Source red points stay red regardless of local PCA shape.
    seed_idx=np.flatnonzero(seeds & ~orig)
    queue=deque()
    steps=np.full(n,-1,dtype=np.int16)
    if len(seed_idx) and len(sheet_idx):
        seedtree=cKDTree(pts[seed_idx].astype(np.float32))
        d,_=seedtree.query(pts[sheet_idx].astype(np.float32),k=1,workers=1)
        init=sheet_idx[d<=2.25]
        for i in init:
            steps[i]=0
            queue.append(int(i))
        while queue:
            i=queue.popleft()
            if steps[i]>=int(propagation_steps):continue
            for j in neighbors(i):
                if steps[j]>=0:continue
                # Limit normal turns: avoid crossing intersections of true
                # clothing layers or branching into volumetric support.
                if abs(float(np.dot(normal[i],normal[j])))<.819152:
                    continue
                steps[j]=steps[i]+1
                queue.append(j)

    # Seed extension class; red mask witness has priority.
    cls[(steps>=0)&(~seeds)&(~orig)]=2
    # Unseeded sheets are only hypotheses if the component has a minimum size.
    if clusters:
        counts=np.asarray([c['children'] for c in clusters])
        accept=np.flatnonzero(counts>=int(min_unseeded_component))
        is_unseeded=(sheet)&(steps<0)&(~seeds)&np.isin(cluster_labels,accept)
        cls[is_unseeded&(~orig)]=3
        for c in clusters:
            member=cluster_labels==c['component_id']
            c['mask_seed_touching']=int(np.sum(member&(steps==0)))
            c['extended_children']=int(np.sum(member&(cls==2)))
            c['morphology_only_children']=int(np.sum(member&(cls==3)))
    return {'class':cls,'planar_ratio':planar_ratio,'width_ratio':width_ratio,
            'neighbors':support,'sheet':sheet,'cluster_id':cluster_labels,
            'clusters':sorted(clusters,key=lambda c:-c['children']),
            'seeded_extension':int((cls==2).sum()),
            'unseeded_candidates':int((cls==3).sum())}


def _detect_lamellar_stacks(xyz, morphology, grid,
                            min_sheet_children=35, max_sheet_thickness=2,
                            min_sheet_extent=6, max_spacing=9,
                            min_projection_overlap=0.25, min_layers=3,
                            max_spacing_cv=0.35):
    """Audit repeated parallel VAE occupancy sheets (``radiator`` structure).

    This is NOT mesh-plane detection. Only compares existing V5 PCA clusters;
    score is an anatomical *hypothesis* and never authorizes token removal.
    Each pair requires true 2-D coordinate footprint overlap (not only bboxes).
    """
    from scipy.spatial import cKDTree
    from collections import defaultdict

    pts=np.asarray(xyz,dtype=np.int32)
    labels=np.asarray(morphology['cluster_id'],dtype=np.int32)
    orig=np.asarray(morphology.get('original_parent',np.zeros(len(pts),bool)),dtype=bool)
    n=len(pts)
    found=np.full(n,-1,dtype=np.int32)
    sheet_candidates=[]
    for c in morphology['clusters']:
        if c['children'] < int(min_sheet_children):continue
        lo=np.asarray(c['bbox_min'],dtype=np.int32)
        hi=np.asarray(c['bbox_max'],dtype=np.int32)
        ext=hi-lo+1
        ix=np.flatnonzero((labels==c['component_id'])&~orig)
        if len(ix)<int(min_sheet_children):continue
        for axis in range(3):
            other=[d for d in range(3) if d!=axis]
            if ext[axis]>int(max_sheet_thickness) or min(ext[other])<int(min_sheet_extent):continue
            # Approximate a plane by a narrow, axis-aligned cluster. This is
            # intentionally conservative and misses oblique sheets.
            proj=pts[ix][:,other].astype(np.float32)
            sheet_candidates.append({
                'component_id':int(c['component_id']),
                'axis':int(axis),'center':float((lo[axis]+hi[axis])/2),
                'indices':ix,'proj':proj,
                'bbox_min':lo.tolist(),'bbox_max':hi.tolist(),
                'children':int(len(ix)),
            })
            break
    groups=[]
    for axis in range(3):
        layers=sorted([c for c in sheet_candidates if c['axis']==axis],
                      key=lambda c:c['center'])
        if len(layers)<int(min_layers):continue
        edges={i:set() for i in range(len(layers))}
        evidence={}
        for i,a in enumerate(layers):
            for j in range(i+1,len(layers)):
                b=layers[j]
                sep=b['center']-a['center']
                if sep>float(max_spacing): break
                if sep<2.0:continue
                oa=a['proj'];ob=b['proj']
                # Reject pairs whose rectangles do not even overlap.
                overlap=True
                for k in range(2):
                    if oa[:,k].max()+1<ob[:,k].min() or \
                       ob[:,k].max()+1<oa[:,k].min():overlap=False;break
                if not overlap:continue
                small,large=(oa,ob) if len(oa)<=len(ob) else (ob,oa)
                tree=cKDTree(large)
                dist,_=tree.query(small,k=1,workers=1)
                fraction=float(np.mean(dist<=2.25))
                if fraction<float(min_projection_overlap):continue
                edges[i].add(j);edges[j].add(i)
                evidence[f'{i}:{j}']={'spacing':float(sep),
                                      'projected_overlap':fraction}
        visited=set()
        for i in range(len(layers)):
            if i in visited or not edges[i]:continue
            frontier=[i];visited.add(i);ids=[]
            while frontier:
                k=frontier.pop();ids.append(k)
                for j in edges[k]:
                    if j not in visited:visited.add(j);frontier.append(j)
            if len(ids)<int(min_layers):continue
            ids=sorted(ids,key=lambda k:layers[k]['center'])
            centers=np.asarray([layers[k]['center'] for k in ids])
            spacings=np.diff(centers)
            cv=float(spacings.std()/spacings.mean()) if spacings.mean()>0 else float('inf')
            if cv>float(max_spacing_cv):continue
            members=np.concatenate([layers[k]['indices'] for k in ids])
            sid=len(groups)
            found[members]=sid
            pair_evidence=[]
            for pi in range(len(ids)):
                for pj in range(pi+1,len(ids)):
                    ii,jj=sorted((ids[pi],ids[pj]))
                    datum=evidence.get(f'{ii}:{jj}')
                    if datum is not None:pair_evidence.append({
                        'a_component_id':layers[ii]['component_id'],
                        'b_component_id':layers[jj]['component_id'],**datum})
            groups.append({
                'stack_id':sid,'axis':axis,'n_layers':len(ids),
                'layer_component_ids':[layers[k]['component_id'] for k in ids],
                'layer_centers':centers.tolist(),
                'successive_spacings':spacings.tolist(),
                'spacing_cv':cv,
                'children':int(len(members)),
                'pair_support':pair_evidence,
                'bbox_min':pts[members].min(axis=0).tolist(),
                'bbox_max':pts[members].max(axis=0).tolist(),
                'mask_confirmed_children':int(np.sum(morphology['class'][members]==1)),
            })
    return {'stack_id':found,'groups':sorted(groups,key=lambda c:-c['children']),
            'n_stacks':len(groups),'children':int(np.sum(found>=0)),
            'axis_aligned_only':True}



def _detect_raw_periodic_lamellae(fine_xyz, original_parent, grid,
                                  min_layers=4, min_columns=12,
                                  min_spacing=3, max_spacing=9,
                                  max_thickness=2, spacing_tolerance=1):
    """Find repeated separated layers in *raw* VAE occupancy, independently of V5.

    Each 2-D voxel column is inspected for >=min_layers regularly spaced,
    1--max_thickness-voxel-thick segments. Candidate columns must form a
    4-connected 2-D patch. No mesh-based evidence is implied. This is a
    read-only sensitivity test; regular clothing pleats may yield false hits.
    """
    from scipy.ndimage import label

    pts = np.asarray(fine_xyz, dtype=np.int32)
    orig = np.asarray(original_parent, dtype=bool)
    N = len(pts)
    if pts.shape != (N,3) or orig.shape != (N,):
        raise ValueError('Invalid raw VAE occupancy inputs')
    axes = np.zeros(N, dtype=np.uint8)
    groups = []
    for axis in range(3):
        other = [d for d in range(3) if d != axis]
        key = pts[:,other[0]].astype(np.int64)*grid + pts[:,other[1]]
        order = np.lexsort((pts[:,axis], key))
        sorted_key = key[order]
        cuts = np.concatenate(([0],np.flatnonzero(np.diff(sorted_key))+1,[N]))
        # For one projected column, record all layers participating in periodic runs.
        candidate_column = np.zeros((grid,grid), dtype=bool)
        hits_per_column = {}
        for a,b in zip(cuts[:-1],cuts[1:]):
            if b-a < int(min_layers):
                continue
            indices = order[a:b]
            heights = pts[indices,axis].astype(np.float32)
            # Contiguous heights belong to one slab, not separate layers.
            sep = np.flatnonzero(np.diff(heights) > 1)+1
            bounds = np.concatenate(([0],sep,[len(indices)]))
            thickness = np.diff(bounds)
            centers = np.array([heights[l:r].mean() for l,r in zip(bounds[:-1],bounds[1:])])
            nseg = len(centers)
            selected = []
            j = 0
            while j + int(min_layers) <= nseg:
                if thickness[j] > int(max_thickness):
                    j += 1
                    continue
                k = j + 1
                baseline = None
                while k < nseg and thickness[k] <= int(max_thickness):
                    dist = float(centers[k] - centers[k-1])
                    if not (float(min_spacing) <= dist <= float(max_spacing)):
                        break
                    if baseline is None:
                        baseline = dist
                    elif abs(dist - baseline) > float(spacing_tolerance):
                        break
                    k += 1
                if k - j >= int(min_layers):
                    selected.extend(indices[bounds[j]:bounds[k]].tolist())
                    j = k
                else:
                    j += 1
            if selected:
                col = int(sorted_key[a])
                u,v = divmod(col,grid)
                candidate_column[u,v] = True
                hits_per_column[col] = np.unique(np.asarray(selected,dtype=np.int32))
        # Require spatial support, not a single chance periodic line.
        connectivity = np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)
        labels, nlabels = label(candidate_column, structure=connectivity)
        for comp in range(1,nlabels+1):
            uv = np.argwhere(labels==comp)
            if len(uv) < int(min_columns):
                continue
            members = np.concatenate([hits_per_column[int(u)*grid+int(v)] for u,v in uv])
            axes[members] |= (1 << axis)
            region = pts[members]
            groups.append({
                'axis':axis,'support_columns':int(len(uv)),
                'children':int(len(members)),
                'original_children':int(orig[members].sum()),
                'fill_children':int((~orig[members]).sum()),
                'bbox_min':region.min(axis=0).tolist(),
                'bbox_max':region.max(axis=0).tolist(),
                'note':'Periodicity in VAE occupancy only; not validated against mesh',
            })
    return {'axis_flags':axes,
            'groups':sorted(groups,key=lambda x:-x['children']),
            'n_regions':len(groups),
            'children':int((axes>0).sum()),
            'original_children':int(((axes>0)&orig).sum()),
            'fill_children':int(((axes>0)&(~orig)).sum())}


def _write_raw_lamellar_ply(path, coords, grid, orig):
    """Export periodicity candidates as colored points, not polygons."""
    xyz=np.asarray(coords,dtype=np.float32)
    p=(xyz+.5)/float(grid)-.5
    with path.open('w',encoding='utf-8') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(p)}\n')
        for x in ('x','y','z'): f.write('property float '+x+'\n')
        for x in ('red','green','blue'):f.write('property uchar '+x+'\n')
        f.write('end_header\n')
        for x,source in zip(p,orig):
            color=(255,185,46) if source else (234,47,184)
            f.write(f'{x[0]:.7f} {x[1]:.7f} {x[2]:.7f} '
                    f'{color[0]} {color[1]} {color[2]}\n')


def _write_morphology_ply(path, xyz, resolution, labels, planar_ratio, width_ratio):
    colors=np.array([
        [87,119,159],    # blue: unknown
        [247,57,45],     # red: multiview mask evidence
        [180,74,222],    # violet: planar, mask-seeded propagation
        [255,177,55],    # amber: unseeded planar candidate
        [234, 47,184],    # magenta: repeated lamellar stack (hypothesis)
    ],dtype=np.uint8)
    pts=(np.asarray(xyz,dtype=np.float64)+.5)/float(resolution)-.5
    with path.open('w',encoding='utf-8') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(pts)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('property uchar morph_class\nend_header\n')
        for xyz0,label in zip(pts,labels):
            color=colors[int(label)]
            f.write(f'{xyz0[0]:.7f} {xyz0[1]:.7f} {xyz0[2]:.7f} '
                    f'{color[0]} {color[1]} {color[2]} {int(label)}\n')


def _write_v8_prune_candidates_only_ply(path, xyz, resolution, s5, s7):
    """Export ONLY the V8 deletion targets, using the Audit PLY coordinate scale.

    Classes: 1=S5 only (cyan); 2=S7 only (magenta); 3=both (yellow).
    Class 0 is forbidden: these must be exact union candidate previews.
    """
    xyz=np.asarray(xyz)
    s5=np.asarray(s5,dtype=bool)
    s7=np.asarray(s7,dtype=bool)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or s5.shape!=(len(xyz),) or s7.shape!=(len(xyz),):
        raise ValueError('Invalid V8 candidate-only PLY arrays')
    if np.any(~(s5|s7)):
        raise ValueError('Candidate-only PLY cannot include unmarked occupancy')
    pts=(xyz.astype(np.float64)+0.5)/float(resolution)-0.5
    cols=((s5.astype(np.uint8))+(s7.astype(np.uint8)*2)).astype(np.uint8)
    palette={1:(47,198,236),2:(234,47,184),3:(250,209,44)}
    with path.open('w',encoding='utf-8') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write('comment ONLY the live V8 deletion targets, not the full Audit occupancy\n')
        f.write(f'element vertex {len(pts)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('property uchar prune_class\nend_header\n')
        for coord,klass in zip(pts,cols):
            red,green,blue=palette[int(klass)]
            f.write(f'{coord[0]:.7f} {coord[1]:.7f} {coord[2]:.7f} '
                    f'{red} {green} {blue} {int(klass)}\n')


class Orbit3DHRSubvoxelFreeSpaceAudit:
    """Inspect VAE's 128/256/512 subdivision coordinates, not ShapeHR output."""
    CATEGORY='Orbit to 3D/Experimental'
    FUNCTION='process'
    RETURN_TYPES=('COORDS','INT','STRING','INT','INT','ORBIT3D_VAE_PRUNE_LABELS')
    RETURN_NAMES=('hr64_coords_unchanged','hr64_grid_resolution',
                  'report_path','proposed_subvoxel_candidates','sampled_subvoxel_count',
                  'vae_prune_labels')
    OUTPUT_NODE=True

    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'pipeline':('TRELLIS2PIPELINE',),
            'shape_slat':('SHAPE_SLAT',),
            'decode_resolution':('INT',{'forceInput':True}),
            'original_ss64_coords':('COORDS',),
            'original_ss64_resolution':('INT',{'forceInput':True}),
            'pixal3d_mv_views':('PIXAL3D_MV_VIEWS',),
            'audit_grid_resolution':(['128','256','512'],{'default':'256'}),
            'audit_scope':(['filled_descendants','all_descendants'],{'default':'filled_descendants'}),
            'depth_sources':(['both','moge','marigold','none'],{'default':'both'}),
            'max_audit_points':('INT',{'default':80000,'min':1000,'max':300000,'step':1000}),
            'max_surface_anchors':('INT',{'default':30000,'min':1000,'max':100000,'step':1000}),
            'min_independent_views':('INT',{'default':2,'min':2,'max':4}),
            'min_view_angle_deg':('FLOAT',{'default':15.,'min':1.,'max':90.,'step':1.}),
            'mask_background_fraction_threshold':('FLOAT',{'default':0.78,'min':.1,'max':1.,'step':.01}),
            'mask_guard_pixels':('INT',{'default':1,'min':0,'max':8}),
            'depth_front_margin_voxels':('FLOAT',{'default':3.,'min':.5,'max':15.,'step':.5}),
            'max_depth_fit_p90_voxels':('FLOAT',{'default':4.,'min':1.,'max':10.,'step':.5}),
            'normal_confidence_min':('FLOAT',{'default':.60,'min':.5,'max':1.,'step':.02}),
            'axis_x_source_px':('FLOAT',{'default':1243.5,'min':0.,'max':10000.,'step':.25}),
            'source_image_width_px':('INT',{'default':2432,'min':32,'max':16384}),
            'report_directory':('STRING',{'default':'/home/ai/orbit3d-cache/experiments/hr_free_space','multiline':False}),
            'preview_views':('INT',{'default':8,'min':0,'max':16}),
            'export_annotation_npz':('BOOLEAN',{'default':False}),
            'morphology_mode':(['seeded_and_unseeded','seeded_only','off'],{'default':'seeded_and_unseeded'}),
            'morphology_radius_voxels':('FLOAT',{'default':4.0,'min':2.0,'max':8.0,'step':.5}),
            'morphology_min_neighbors':('INT',{'default':16,'min':8,'max':64}),
            'morphology_planar_ratio_max':('FLOAT',{'default':.12,'min':.01,'max':.5,'step':.01}),
            'morphology_width_ratio_min':('FLOAT',{'default':.18,'min':.01,'max':.9,'step':.01}),
            'morphology_propagation_steps':('INT',{'default':12,'min':1,'max':64}),
            'morphology_min_unseeded_children':('INT',{'default':48,'min':8,'max':1000}),
            'lamellar_mode':(['diagnostic','off'],{'default':'diagnostic'}),
            'lamellar_min_sheet_children':('INT',{'default':35,'min':12,'max':1000}),
            'lamellar_max_sheet_thickness':('INT',{'default':2,'min':1,'max':4}),
            'lamellar_min_sheet_extent':('INT',{'default':6,'min':3,'max':20}),
            'lamellar_max_spacing':('INT',{'default':9,'min':3,'max':20}),
            'lamellar_min_projection_overlap':('FLOAT',{'default':.25,'min':.05,'max':.95,'step':.05}),
            'lamellar_min_layers':('INT',{'default':3,'min':2,'max':12}),
            'raw_lamellar_mode':(['all_descendants','fill_only','off'],{'default':'all_descendants'}),
            'raw_lamellar_min_layers':('INT',{'default':4,'min':3,'max':10}),
            'raw_lamellar_min_columns':('INT',{'default':12,'min':3,'max':100}),
        }}

    @torch.no_grad()
    def process(self,pipeline,shape_slat,decode_resolution,original_ss64_coords,
                original_ss64_resolution,pixal3d_mv_views,
                audit_grid_resolution=256,audit_scope='filled_descendants',
                depth_sources='both',max_audit_points=80000,max_surface_anchors=30000,
                min_independent_views=2,min_view_angle_deg=15.,
                mask_background_fraction_threshold=.78,mask_guard_pixels=1,
                depth_front_margin_voxels=3.,max_depth_fit_p90_voxels=4.,
                normal_confidence_min=.6,axis_x_source_px=1243.5,
                source_image_width_px=2432,
                report_directory='/home/ai/orbit3d-cache/experiments/hr_free_space',
                preview_views=8,export_annotation_npz=False,
                morphology_mode='seeded_and_unseeded',
                morphology_radius_voxels=4.0,
                morphology_min_neighbors=16,
                morphology_planar_ratio_max=.12,
                morphology_width_ratio_min=.18,
                morphology_propagation_steps=12,
                morphology_min_unseeded_children=48,
                lamellar_mode='diagnostic',
                lamellar_min_sheet_children=35,
                lamellar_max_sheet_thickness=2,
                lamellar_min_sheet_extent=6,
                lamellar_max_spacing=9,
                lamellar_min_projection_overlap=.25,
                lamellar_min_layers=3,
                raw_lamellar_mode='all_descendants',
                raw_lamellar_min_layers=4,raw_lamellar_min_columns=12):
        _require_pixal3d_mv(pipeline)
        if not isinstance(shape_slat,SparseTensor):
            raise TypeError('shape_slat must be the actual Shape512 SparseTensor')
        if int(original_ss64_resolution)!=64 or int(decode_resolution)!=1024:
            raise ValueError('Requires original SS64 and Shape512 SLat decode_resolution=1024')
        R=int(audit_grid_resolution)
        if R not in (128,256,512):
            raise ValueError('audit_grid_resolution must be 128, 256, 512')
        if audit_scope not in ('filled_descendants','all_descendants'):
            raise ValueError('Unsupported audit_scope')
        if depth_sources not in ('both','moge','marigold','none'):
            raise ValueError('Unsupported depth_sources')
        original=_validate(original_ss64_coords,64,'original_ss64_coords')
        slat=_validate(shape_slat.coords,64,'shape_slat.coords')
        flat=lambda a:a[:,0]*4096+a[:,1]*64+a[:,2]
        originals=flat(original[:,1:])
        slats=flat(slat[:,1:])
        if not np.isin(originals,slats).all():
            raise ValueError('Original SS64 must be subset of Shape512 SLat coordinates')
        views=pixal3d_mv_views
        if views.get('alphas') is None:
            raise ValueError('Missing foreground masks in PIXAL3D_MV_VIEWS')
        device=torch.device(pipeline.device)
        cam=views['transform_matrix'].to(device=device,dtype=torch.float32)
        fov=views['camera_angle_x'].to(device=device,dtype=torch.float32)
        dist=views['camera_distance'].to(device=device,dtype=torch.float32)
        B,V=cam.shape[:2]
        if B!=1 or V<int(min_independent_views):
            raise ValueError('Requires batch=1 and sufficient views')
        scale=float(torch.as_tensor(views['mesh_scale']).reshape(-1)[0])
        if not math.isfinite(scale) or scale<=0:
            raise ValueError('Invalid mesh_scale')
        axis=float(axis_x_source_px); width=int(source_image_width_px)
        if not math.isfinite(axis) or not (0<=axis<width):
            raise ValueError('axis_x_source_px outside the source frame')
        dx=(axis-width/2)*(512/width)
        wanted={'both':['moge','marigold'],'moge':['moge'],
                'marigold':['marigold'],'none':[]}[depth_sources]
        available=[k for k in wanted if views.get(k+'_depth_native') is not None
                                    and views.get(k+'_alpha_native') is not None]
        if wanted and not available:
            raise ValueError('No requested native geometry maps in view bundle')

        # Capture an intermediate subdivision *during the existing single
        # upsample-to-1024 pass*. Read-only forward hook, removed in finally.
        level={128:1,256:2,512:3}[R]
        captured=[]
        pipeline.load_shape_slat_decoder()
        dec=pipeline.models['shape_slat_decoder']
        if pipeline.low_vram:
            dec.to(device);dec.low_vram=True
        hook=None
        try:
            stage=dec.blocks[level-1][-1]
            def capture(_module,_inputs,output):
                out=output[0] if isinstance(output,tuple) else output
                captured.append(out.coords.detach().to('cpu').contiguous())
            hook=stage.register_forward_hook(capture)
            raw=dec.upsample(shape_slat,upsample_times=4)
        finally:
            if hook is not None:hook.remove()
            if pipeline.low_vram:
                dec.cpu();dec.low_vram=False
            if not pipeline.keep_models_loaded:
                pipeline.unload_shape_slat_decoder()
        # Release the local decoder reference after the checkpoint is captured.
        del dec, stage
        if len(captured)!=1:
            raise RuntimeError('Expected exactly one VAE subdivision hook call; got '+str(len(captured)))
        fine=_validate(captured[0],R,f'VAE {R} coords')
        raw_cpu=raw.detach().to('cpu').long()
        hr64=torch.unique(torch.cat((raw_cpu[:,:1],
             ((raw_cpu[:,1:].float()+.5)/1024*64).long()),dim=1),dim=0).contiguous()
        _validate(hr64,64,'hr64_coords')
        del raw,raw_cpu,captured

        factor=R//64
        parents=np.floor_divide(fine[:,1:],factor)
        parentkeys=flat(parents)
        origin_all=np.isin(parentkeys,originals)
        fill_all=np.isin(parentkeys,slats)&(~origin_all)
        if not np.all(origin_all|fill_all):
            raise RuntimeError('VAE yielded a child outside the input SS64/Fill occupancy')
        in_scope=fill_all if audit_scope=='filled_descendants' else np.ones(len(fine),bool)
        selected=np.flatnonzero(in_scope)
        if len(selected)==0:
            raise ValueError('No VAE child coordinates in requested audit_scope')
        requested_count=len(selected)
        if requested_count>int(max_audit_points):
            # Uniform deterministic sampling over sorted VAE coords; no hard
            # filtering is permitted on a partial sample.
            picked=np.linspace(0,requested_count-1,int(max_audit_points),dtype=np.int64)
            selected=selected[picked]
        sampled=len(selected)
        xyz=fine[selected,1:]
        child_parents=parents[selected]
        original_parent=origin_all[selected]
        print(f'[HR Subvoxel] R={R}: VAE produced {len(fine)} children, '
              f'Fill descendants={int(fill_all.sum())}, selected={sampled}/{requested_count}. '
              'Audit ONLY; Shape1024 not run.',flush=True)

        # Physical footprint matches original 64 grid and 2^k subdivisions.
        pts=_fine_world(xyz,R,scale,device)
        half_span=1/(63*factor*scale)
        # Reference anchors from children descending ONLY from genuine SS64.
        anchor_idx=np.flatnonzero(origin_all)
        if len(anchor_idx)==0:
            raise ValueError('No original-parent VAE children available for depth calibration')
        anchor_pts=_fine_world(fine[anchor_idx,1:],R,scale,device)
        proj=ProjGrid(grid_resolution=64,image_resolution=512).to(device)
        calc=compute_relative_calc_mat(cam,dist,proj.front_view_transform_matrix)
        dirs=cam[0,:,:3,2].detach().cpu().numpy().astype(np.float64)
        dirs/=np.maximum(np.linalg.norm(dirs,axis=1,keepdims=True),1e-12)
        cos_angle=math.cos(math.radians(float(min_view_angle_deg)))
        K=int(min_independent_views)
        witnesses=np.full((sampled,K),-1,dtype=np.int16)
        score_sum=np.zeros(sampled,dtype=np.float32)
        valid_views=np.zeros(sampled,dtype=np.uint16)
        mask_sum=np.zeros(sampled,dtype=np.float32)
        mask_high=np.zeros(sampled,dtype=np.uint16)
        depth_high=np.zeros(sampled,dtype=np.uint16)
        max_mask=np.zeros(sampled,dtype=np.float32)
        max_depth=np.zeros(sampled,dtype=np.float32)
        reports=[]
        depth_spatial_reports=[]
        preview_ids=set(np.linspace(0,V-1,num=min(int(preview_views),V),dtype=int))
        preview_data={}
        unit64=1/(63*scale)
        threshold=float(mask_background_fraction_threshold)

        for v in range(V):
            alpha=_getmap(views,'alphas',v)
            guard=int(mask_guard_pixels)
            if guard:
                k=2*guard+1
                alpha_guard=F.max_pool2d(alpha[None,None],kernel_size=k,
                          stride=1,padding=guard)[0,0]
            else:
                alpha_guard=alpha
            # Nine-point sampling at subvoxel scale, not the large R64 box.
            probe_xy,all_valid=_subcell_probe(pts,half_span,
                                  calc[0:1,v],fov[0:1,v],dx)
            frac_bg=_probe_foreground_fraction(alpha_guard,probe_xy)
            frac_bg[~all_valid]=0
            valid_views+=all_valid.astype(np.uint16)
            mask_sum+=frac_bg
            max_mask=np.maximum(max_mask,frac_bg)
            mask_vote=all_valid&(frac_bg>=threshold)
            mask_high+=mask_vote.astype(np.uint16)
            center_xy=torch.from_numpy(probe_xy[:,0,:]).float()
            center_depth=None
            depth_score=np.zeros(sampled,np.float32)
            dfits={}
            if available:
                # Unique foreground camera rays, nearest original-child depth.
                zbuffer,n_vis,n_rays,anchor_stride=_visible_anchor_zbuffer(
                     anchor_pts,calc[0:1,v],fov[0:1,v],dx,alpha,
                     int(max_surface_anchors),unit64)
                candidate_xy,candidate_z,proj_valid=_project_points(pts,calc[0:1,v],fov[0:1,v],dx)
                for key in available:
                    dm=_getmap(views,key+'_depth_native',v)
                    am=_getmap(views,key+'_alpha_native',v)
                    nm=_getmap(views,'moge_geometry_confidence' if key=='moge'
                               else 'geometry_confidence',v)
                    fg=(_sample(alpha,candidate_xy)>.8)&(_sample(am,candidate_xy)>.8)
                    if nm is not None:
                        fg&=(_sample(nm,candidate_xy)>=float(normal_confidence_min))
                    # Pixel-unique z-buffer samples are visible-surface proxies.
                    good=np.isfinite(zbuffer).reshape(512,512)
                    good&=(alpha.numpy()>.8)&(am.numpy()>.8)
                    if nm is not None: good&=(nm.numpy()>=float(normal_confidence_min))
                    goodflat=good.ravel()
                    fit=_fit_depth(dm.numpy().ravel(),zbuffer,goodflat,unit64,
                                   min_samples=25)
                    # Independent spatial QA ONLY; not allowed to create depth votes.
                    local_qa = _depth_spatial_holdout_qa(
                        dm.numpy(), zbuffer, goodflat, unit64, tiles=4)
                    depth_spatial_reports.append({'view':v,'source':key,
                                                  'global_p90_vox':fit.get('p90_error_vox'),
                                                  'local_holdout':local_qa})
                    permitted=bool(fit.get('usable',False)) and \
                              fit.get('p90_error_vox',1e10)<=float(max_depth_fit_p90_voxels)
                    nfront=0
                    if permitted:
                        observed=_sample(dm,candidate_xy)
                        projected=fit['a']*observed+fit['b']
                        # Positive means candidate is in FRONT of visible geometry.
                        front_vox=(projected-candidate_z)/unit64
                        margin=max(float(depth_front_margin_voxels),
                                   2.0*fit['p90_error_vox'])
                        magnitude=np.clip((front_vox-margin)/max(margin,1.),0,1)
                        depth_ok=proj_valid&fg&np.isfinite(projected)&np.isfinite(observed)
                        magnitude[~depth_ok]=0
                        depth_score=np.maximum(depth_score,magnitude.astype(np.float32))
                        nfront=int(np.count_nonzero(magnitude>=threshold))
                    dfits[key]={'calibration':fit,'accepted_for_scoring':bool(permitted),
                                'front_votes':nfront,'unique_visible_anchor_rays':n_rays,
                                'visible_anchor_points':n_vis,'anchor_subsample_stride':anchor_stride}
            max_depth=np.maximum(max_depth,depth_score)
            depth_high+=(depth_score>=threshold).astype(np.uint16)
            score=np.maximum(frac_bg,depth_score)
            score_sum+=score
            accepted=score>=threshold
            # Greedily accumulate witnesses from distinct view directions.
            for k in range(K):
                allowed=accepted&(witnesses[:,k]<0)
                for earlier in range(k):
                    prev=witnesses[:,earlier]
                    allowed&=(prev>=0)
                    if np.any(allowed):
                        allowed&=(np.einsum('ij,j->i',dirs[np.maximum(prev,0)],dirs[v])<cos_angle)
                witnesses[allowed,k]=v
            if v in preview_ids:
                preview_data[v]=(probe_xy[:,0].copy(),all_valid.copy(),
                                 alpha.numpy().copy(),frac_bg.copy())
            reports.append({'view':v,'name':views.get('view_names',[f'view{i}' for i in range(V)])[v],
                   'valid_subcell_count':int(all_valid.sum()),
                   'mask_high_count':int(mask_vote.sum()),
                   'mask_background_fraction_mean':float(frac_bg.mean()),
                   'depth_fits':dfits})
            if (v+1)%20==0 or v==V-1:
                print(f'[HR Subvoxel] audited {v+1}/{V} views',flush=True)

        confirmed=witnesses[:,K-1]>=0
        mean_score=score_sum/np.maximum(valid_views,1)
        mean_mask=mask_sum/np.maximum(valid_views,1)
        # Connected clusters and true parent denominators, including VAE children
        # that were not audited when max_audit_points forced a subset.
        component_id, components = _components6(xyz,confirmed,R)
        pkeys=flat(child_parents)
        summary=_parent_topology(fine, parentkeys[in_scope], xyz, pkeys,
                                 originals, confirmed, component_id, mean_score)
        is_full_sample = bool(sampled == requested_count)
        if morphology_mode not in ('seeded_and_unseeded','seeded_only','off'):
            raise ValueError('Unsupported morphology_mode')
        morph=None
        if morphology_mode!='off':
            # Do not infer shape/topology from a partial coordinate sample.
            if not is_full_sample:
                raise ValueError('Morphology requires full audit coverage: increase max_audit_points')
            morph=_morphology_sheet_candidates(
                xyz,confirmed,original_parent,R,
                radius_voxels=morphology_radius_voxels,
                min_neighbors=morphology_min_neighbors,
                planar_ratio_max=morphology_planar_ratio_max,
                width_ratio_min=morphology_width_ratio_min,
                propagation_steps=morphology_propagation_steps,
                min_unseeded_component=morphology_min_unseeded_children)
            if morphology_mode=='seeded_only':
                morph['class'][morph['class']==3]=0
                morph['unseeded_candidates']=0
        if lamellar_mode not in ('diagnostic','off'):
            raise ValueError('Unsupported lamellar_mode')
        lamellar=None
        if morph is not None and lamellar_mode!='off':
            morph['original_parent']=original_parent
            lamellar=_detect_lamellar_stacks(
                xyz,morph,R,min_sheet_children=lamellar_min_sheet_children,
                max_sheet_thickness=lamellar_max_sheet_thickness,
                min_sheet_extent=lamellar_min_sheet_extent,
                max_spacing=lamellar_max_spacing,
                min_projection_overlap=lamellar_min_projection_overlap,
                min_layers=lamellar_min_layers)
            print('[HR Subvoxel Lamellar] stacks=',lamellar['n_stacks'],
                  'children=',lamellar['children'],flush=True)
        # Independent scan over raw VAE child occupancy -- V5 morphology labels
        # and the restricted free-space audit sample are NOT prerequisites.
        if raw_lamellar_mode not in ('off','fill_only','all_descendants'):
            raise ValueError('Unsupported raw_lamellar_mode')
        raw_scan = None
        if raw_lamellar_mode != 'off':
            if R > 256:
                raise ValueError('Raw periodicity V7 is limited to R<=256; select 128 or 256')
            scan_mask=fill_all if raw_lamellar_mode=='fill_only' else np.ones(len(fine),bool)
            scan_indices=np.flatnonzero(scan_mask)
            scan_xyz=fine[scan_indices,1:]
            raw_scan=_detect_raw_periodic_lamellae(
                scan_xyz,origin_all[scan_indices],R,
                min_layers=int(raw_lamellar_min_layers),
                min_columns=int(raw_lamellar_min_columns))
            raw_scan['scanned_children']=int(len(scan_indices))
            raw_scan['indices_in_fine']=scan_indices[raw_scan['axis_flags']>0]
            raw_scan['full_flags']=np.zeros(len(fine),dtype=np.uint8)
            raw_scan['full_flags'][scan_indices]=raw_scan['axis_flags']
            print('[HR Subvoxel Raw Lamellar] regions=',raw_scan['n_regions'],
                  'hits=',raw_scan['children'],
                  'original=',raw_scan['original_children'],
                  'fill=',raw_scan['fill_children'],flush=True)
        # Direct in-memory S5/S7 data contract for the experimental V8 decoder.
        # No NPZ is required and NOTHING is pruned here.
        prune_labels=None
        if (R==256 and is_full_sample and morph is not None
                and raw_scan is not None and raw_lamellar_mode=='all_descendants'):
            prune_labels=make_vae_prune_labels(
                shape_slat=shape_slat,
                fine_coords=fine,
                scope_coords=fine[selected],
                s5_coords=fine[selected],
                s5_class=morph['class'],
                s7_coords=fine[raw_scan['indices_in_fine']],
                audit_scope=audit_scope,
            )
        else:
            print('[HR Subvoxel] V8 direct-prune output is unavailable: '
                  'requires R=256, full audit, morphology!=off, '
                  'raw_lamellar_mode=all_descendants',flush=True)
        # A red subvoxel does NOT justify deleting an entire parent token.
        parent_proposals=(summary['risk_class']==3) & summary['is_complete']

        root=Path(report_directory).expanduser().resolve()
        root.mkdir(parents=True,exist_ok=True)
        out=root/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(exist_ok=False)
        # Export side-by-side targets from the EXACT same live label contract
        # that V8 Prune consumes. Unlike morphology/lamellar PLYs, these
        # contain NO class-0/background occupancy and can be overlaid directly
        # against V8's stage256_removed PLY for a fair comparison.
        if prune_labels is not None:
            from .experimental_vae_prune_contract import pack_256
            keys=pack_256(fine)
            in_s5=np.isin(keys,prune_labels['s5_marked_keys'])
            in_s7=np.isin(keys,prune_labels['s7_all_keys'])
            candidate_masks={
                's5':in_s5,
                's7':in_s7,
                'intersection':in_s5 & in_s7,
                'union':in_s5 | in_s7,
            }
            preview_counts={}
            for name,mask in candidate_masks.items():
                preview_counts[name]=int(mask.sum())
                _write_v8_prune_candidates_only_ply(
                    out/f'subvoxel_{R}_v8_{name}_candidates_only.ply',
                    fine[mask,1:],R,in_s5[mask],in_s7[mask])
            if audit_scope == 'filled_descendants':
                # The complete audited cohort, including UNMARKED S5/S7 points.
                # Only used by the separate intentionally aggressive ablation.
                scope_mask=np.isin(keys,prune_labels['audit_scope_keys'])
                preview_counts['all_audit_scope']=int(scope_mask.sum())
                _write_raw_lamellar_ply(
                    out/f'subvoxel_{R}_v8_all_audit_scope_candidates_only.ply',
                    fine[scope_mask,1:],R,
                    np.zeros(int(scope_mask.sum()),dtype=bool))
            (out/'v8_prune_candidate_counts.json').write_text(
                json.dumps({'resolution':R,
                            'note':'V8 live deletion targets only; matches stage256_removed exactly',
                            'counts':preview_counts},indent=2),encoding='utf-8')
            print('[HR Subvoxel] V8 candidate-only PLYs:',preview_counts,flush=True)
        _write_score_ply(out/f'subvoxel_{R}_score.ply',xyz,R,mean_score,original_parent,confirmed)
        if export_annotation_npz:
            np.savez_compressed(out/f'subvoxel_{R}_audit.npz',
                coords=fine[selected].astype(np.int32),parent_coords=child_parents.astype(np.int16),
                original_parent=original_parent,mean_score=mean_score,mean_mask_score=mean_mask,
                max_mask_score=max_mask,max_depth_score=max_depth,mask_high_views=mask_high,
                depth_high_views=depth_high,valid_views=valid_views,
                witness_indices=witnesses,confirmed=confirmed,
                connected_component_id=component_id,
                sampling_fraction=np.float32(sampled/requested_count))
        if morph is not None:
            _write_morphology_ply(out/f'subvoxel_{R}_morphology.ply',
                                  xyz,R,morph['class'],
                                  morph['planar_ratio'],morph['width_ratio'])
            if export_annotation_npz:
                np.savez_compressed(out/f'subvoxel_{R}_morphology.npz',
                    coords=fine[selected].astype(np.int32),
                    parent_coords=child_parents.astype(np.int16),
                    morphology_class=morph['class'],
                    mask_confirmed=confirmed,original_parent=original_parent,
                    planar_ratio=morph['planar_ratio'],
                    width_ratio=morph['width_ratio'],
                    neighbor_support=morph['neighbors'],
                    local_sheet_candidate=morph['sheet'],
                    morphology_cluster_id=morph['cluster_id'])
            (out/'morphology_components.json').write_text(
                json.dumps({'note':'Local VAE occupancy morphology, not mesh surface morphology. '
                           'No label licenses automatic removal.',
                           'clusters':morph['clusters']},
                           indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
            print('[HR Subvoxel Morphology] mask-only=',int(np.sum(morph['class']==1)),
                  'seeded_extension=',morph['seeded_extension'],
                  'morphology_only=',morph['unseeded_candidates'],flush=True)
        if lamellar is not None:
            lamellar_labels=morph['class'].copy()
            lamellar_labels[(lamellar['stack_id']>=0)&(lamellar_labels!=1)]=4
            _write_morphology_ply(out/f'subvoxel_{R}_lamellar.ply',
                                  xyz,R,lamellar_labels,
                                  morph['planar_ratio'],morph['width_ratio'])
            if export_annotation_npz:
                np.savez_compressed(out/f'subvoxel_{R}_lamellar.npz',
                    coords=fine[selected].astype(np.int32),
                    parent_coords=child_parents.astype(np.int16),
                    lamellar_stack_id=lamellar['stack_id'],
                    lamellar_display_class=lamellar_labels,
                    morphology_class=morph['class'],
                    mask_confirmed=confirmed,
                    original_parent=original_parent)
            (out/'lamellar_stacks.json').write_text(json.dumps({
                'note':'Axis-aligned, repeated near-parallel VAE occupancy layers. '
                       'Diagnostic only; not confirmed mesh artifacts.',
                'axis_aligned_only':True,
                'n_stacks':lamellar['n_stacks'],
                'children':lamellar['children'],
                'stacks':lamellar['groups']},
                indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        if raw_scan is not None:
            hit=raw_scan['indices_in_fine']
            _write_raw_lamellar_ply(out/f'subvoxel_{R}_raw_lamellar.ply',
                      fine[hit,1:],R,origin_all[hit])
            if export_annotation_npz:
                np.savez_compressed(out/f'subvoxel_{R}_raw_lamellar.npz',
                          coords=fine[hit].astype(np.int32),
                          axis_flags=raw_scan['full_flags'][hit],
                          original_parent=origin_all[hit])
            (out/'raw_lamellar_regions.json').write_text(json.dumps({
                'note':'Independent periodicity search on raw VAE occupancy; diagnostic ONLY. '
                       'Even repeated layers may be legitimate geometry.',
                'mode':raw_lamellar_mode,
                'scanned_children':raw_scan['scanned_children'],
                'n_regions':raw_scan['n_regions'],
                'children':raw_scan['children'],
                'original_children':raw_scan['original_children'],
                'fill_children':raw_scan['fill_children'],
                'regions':raw_scan['groups']},indent=2,ensure_ascii=False,allow_nan=False),
                encoding='utf-8')
        if export_annotation_npz:
            np.savez_compressed(out/'parent64_summary.npz',**summary)
        _write_parent_csv(out/'parent64_topology.csv',summary)
        _write_parent_ply(out/'parent64_risk.ply',summary)
        (out/'confirmed_components.json').write_text(
            json.dumps({'full_coverage':is_full_sample,'components':components},
                       ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        (out/'depth_spatial_holdout_qa.json').write_text(
            json.dumps({'used_for_scoring':False,
                        'note':'Local held-out fits are only diagnostics. All V3 strict global scoring remains unchanged.',
                        'views':depth_spatial_reports},
                       ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        if preview_data:
            from PIL import Image,ImageDraw
            for v,(xy,valid,alpha,frac) in preview_data.items():
                gray=np.clip(alpha*180+35,0,255).astype(np.uint8)
                im=Image.fromarray(gray,'L').convert('RGB')
                draw=ImageDraw.Draw(im)
                stride=max(1,len(xyz)//40000)
                for i in range(0,len(xyz),stride):
                    if not valid[i]:continue
                    x,y=xy[i]
                    if original_parent[i]:color=(35,195,180)
                    elif confirmed[i]:color=(245,62,57)
                    elif mean_score[i]>.35:color=(255,164,50)
                    else:color=(95,130,180)
                    draw.point((float(x),float(y)),fill=color)
                im.save(out/f'subvoxel_overlay_view_{v:03}.png')
                if morph is not None:
                    mor=Image.fromarray(gray,'L').convert('RGB')
                    md=ImageDraw.Draw(mor)
                    palette=((95,130,180),(245,62,57),(180,74,222),(255,170,40))
                    for i in range(0,len(xyz),stride):
                        if not valid[i]:continue
                        x,y=xy[i]
                        md.point((float(x),float(y)),fill=palette[int(morph['class'][i])])
                    mor.save(out/f'subvoxel_morphology_overlay_view_{v:03}.png')
                    if lamellar is not None:
                        lam=Image.fromarray(gray,'L').convert('RGB')
                        ld=ImageDraw.Draw(lam)
                        for i in range(0,len(xyz),stride):
                            if not valid[i]:continue
                            x,y=xy[i]
                            cls=int(morph['class'][i])
                            if lamellar['stack_id'][i]>=0 and cls!=1:cls=4
                            palette2=((95,130,180),(245,62,57),
                                      (180,74,222),(255,170,40),(234,47,184))
                            ld.point((float(x),float(y)),fill=palette2[cls])
                        lam.save(out/f'subvoxel_lamellar_overlay_view_{v:03}.png')
                if raw_scan is not None:
                    # Magenta: descendants of Fill. Yellow: original SS64.
                    # The raw scan is independent from blue/red V3-V6 labels.
                    probe_hit=raw_scan['indices_in_fine']
                    raw_pts=_fine_world(fine[probe_hit,1:],R,scale,device)
                    raw_xy,_,raw_valid=_project_points(raw_pts,calc[0:1,v],fov[0:1,v],dx)
                    raw_im=Image.fromarray(gray,'L').convert('RGB')
                    rd=ImageDraw.Draw(raw_im)
                    for i in np.flatnonzero(raw_valid):
                        x,y=raw_xy[i]
                        rd.point((float(x),float(y)),fill=(255,185,46) if origin_all[probe_hit[i]] else (234,47,184))
                    raw_im.save(out/f'raw_lamellar_overlay_view_{v:03}.png')

        report={'version':'ORBIT_HR_SUBVOXEL_AUDIT_V7','mode':'audit_only',
            'cascade_executed':False,'sLat_modified':False,'hr64_coords_modified':False,
            'input_ss64_tokens':int(len(original)),'input_shape512_tokens':int(len(slat)),
            'vae_subdivision_level':R,'total_vae_children':int(len(fine)),
            'total_fill_descendants':int(fill_all.sum()),
            'scope':audit_scope,'scope_candidates':int(requested_count),
            'sampled_candidates':int(sampled),'sampling_fraction':float(sampled/requested_count),
            'confirmed_subvoxel_candidates_in_sample':int(confirmed.sum()),
            'confirmed_fraction_in_sample':float(confirmed.mean()),
            'hr64_coords_returned_unchanged':int(len(hr64)),
            'parent64_sampled':int(len(summary['coords'])),
            'parent64_confirmed_any':int(np.sum(summary['n_confirmed_children']>0)),
            'full_coverage':is_full_sample,
            'connected_components6':int(len(components)),
            'largest_connected_component_children':int(components[0]['n_confirmed_children']) if components else 0,
            'risk_parent_counts':{str(i):int((summary['risk_class']==i).sum()) for i in range(4)},
            'diagnostic_parent_proposals_not_deleted':int(parent_proposals.sum()),
            'incomplete_parent_count':int(np.sum(~summary['is_complete'])),
            'depth_spatial_holdout_scoring_enabled':False,
            'morphology_scoring_used_for_deletion':False,
            'morphology_mode':morphology_mode,
            'morphology_local_sheet_children':int(morph['sheet'].sum()) if morph else 0,
            'morphology_seeded_extension_children':morph['seeded_extension'] if morph else 0,
            'morphology_only_children':morph['unseeded_candidates'] if morph else 0,
            'morphology_clusters':len(morph['clusters']) if morph else 0,
            'morphology_file':f'subvoxel_{R}_morphology.ply' if morph else None,
            'morphology_components_file':'morphology_components.json' if morph else None,
            'lamellar_mode':lamellar_mode,
            'lamellar_stacks':lamellar['n_stacks'] if lamellar is not None else 0,
            'lamellar_children':lamellar['children'] if lamellar is not None else 0,
            'lamellar_stacks_file':'lamellar_stacks.json' if lamellar is not None else None,
            'raw_lamellar_mode':raw_lamellar_mode,
            'raw_lamellar_regions':raw_scan['n_regions'] if raw_scan else 0,
            'raw_lamellar_children':raw_scan['children'] if raw_scan else 0,
            'raw_lamellar_original_children':raw_scan['original_children'] if raw_scan else 0,
            'raw_lamellar_fill_children':raw_scan['fill_children'] if raw_scan else 0,
            'raw_lamellar_file':'raw_lamellar_regions.json' if raw_scan else None,
            'depth_spatial_holdout_file':'depth_spatial_holdout_qa.json',
            'parent64_topology_file':'parent64_topology.csv',
            'parent64_risk_file':'parent64_risk.ply',
            'confirmed_components_file':'confirmed_components.json',
            'sources_requested':wanted,'sources_available':available,
            'depth_accepted_views':{k:sum(int(r['depth_fits'].get(k,{}).get('accepted_for_scoring',False))
                     for r in reports) for k in available},
            'parameters':{'min_independent_views':K,'min_view_angle_deg':float(min_view_angle_deg),
               'mask_background_fraction_threshold':threshold,'mask_guard_pixels':int(mask_guard_pixels),
               'depth_front_margin_voxels':float(depth_front_margin_voxels),
               'max_depth_fit_p90_voxels':float(max_depth_fit_p90_voxels),
               'axis_shift_x_px_512':dx,'axis_x_source_px':axis,
               'source_image_width_px':width,
               'max_audit_points':int(max_audit_points),
               'max_surface_anchors':int(max_surface_anchors),
               'morphology_radius_voxels':float(morphology_radius_voxels),
               'morphology_min_neighbors':int(morphology_min_neighbors),
               'morphology_planar_ratio_max':float(morphology_planar_ratio_max),
               'morphology_width_ratio_min':float(morphology_width_ratio_min),
               'morphology_propagation_steps':int(morphology_propagation_steps),
               'morphology_min_unseeded_children':int(morphology_min_unseeded_children),
               'lamellar_min_sheet_children':int(lamellar_min_sheet_children),
               'lamellar_max_sheet_thickness':int(lamellar_max_sheet_thickness),
               'lamellar_min_sheet_extent':int(lamellar_min_sheet_extent),
               'lamellar_max_spacing':int(lamellar_max_spacing),
               'lamellar_min_projection_overlap':float(lamellar_min_projection_overlap),
               'lamellar_min_layers':int(lamellar_min_layers),
               'raw_lamellar_mode':raw_lamellar_mode,
               'raw_lamellar_min_layers':int(raw_lamellar_min_layers),
               'raw_lamellar_min_columns':int(raw_lamellar_min_columns)},
            'topology_note':'Risk classes 1-3 are candidate analysis only; do not delete a parent because its children contain free space evidence.',
            'colors':{'teal':'original-SS64 descendants protected for interpretation',
                      'red':'independent free-space evidence in subvoxel sample',
                      'orange':'medium score','blue':'low/unknown'},
            'views':reports}
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        print(f'[HR Subvoxel] R={R} sampled={sampled}/{requested_count}, '
              f'flagged={int(confirmed.sum())}, depth accepted={report["depth_accepted_views"]}, '
              f'report={out/"report.json"}',flush=True)
        if prune_labels is not None:
            prune_labels['audit_report_path']=str(out/'report.json')
        return hr64,64,str(out/'report.json'),int(confirmed.sum()),sampled,prune_labels


NODE_CLASS_MAPPINGS={'Orbit3DHRSubvoxelFreeSpaceAudit':Orbit3DHRSubvoxelFreeSpaceAudit}
NODE_DISPLAY_NAME_MAPPINGS={'Orbit3DHRSubvoxelFreeSpaceAudit':
    'Orbit3D - HR Subvoxel Free-Space Audit [Experimental]'}
