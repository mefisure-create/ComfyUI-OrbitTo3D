"""Independent pre-Cascade HR64 free-space diagnostic (does not run Shape1024).

All tests are conservative and driven by the user's exact MV cameras, masks,
relative depth maps, and original unfilled SS64 occupancy.  No production math
is monkeypatched. Unknown and occluded points are NEVER treated as free space.
"""
from __future__ import annotations

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
    ProjGrid,
    compute_relative_calc_mat,
    project_points_to_image_batch,
)


def _validate(coords, R, label):
    if not isinstance(coords, torch.Tensor) or coords.ndim != 2 or coords.shape[1] != 4:
        raise TypeError(f'{label} must be [N,4] COORDS')
    if coords.is_floating_point() or coords.is_complex():
        raise TypeError(f'{label} must be integer COORDS')
    arr = coords.detach().to('cpu', dtype=torch.int64).numpy()
    if arr.shape[0] == 0 or np.any(arr[:, 0] != 0) or np.any(arr[:, 1:] < 0) or np.any(arr[:, 1:] >= R):
        raise ValueError(f'{label} must be non-empty, batch=0, grid [0,{R-1}]')
    if len(np.unique(arr, axis=0)) != len(arr):
        raise ValueError(f'{label} has duplicate coords')
    return arr


def _getmap(views, key, v, res=512):
    obj = views.get(key)
    if obj is None:
        return None
    # [1,V,H,W] for `alphas`; [1,V,1,H,W] for native depth/alpha/normals.
    img = obj[0, v]
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    if img.ndim != 2:
        raise ValueError(f'Unexpected {key} view tensor shape: {tuple(img.shape)}')
    img = img.detach().to('cpu', dtype=torch.float32)
    if tuple(img.shape) != (res, res):
        img = F.interpolate(img[None, None], (res, res), mode='bilinear', align_corners=False)[0, 0]
    return img


def _sample(img, xy):
    h, w = img.shape
    xi = xy[:, 0].round().long().clamp(0, w-1)
    yi = xy[:, 1].round().long().clamp(0, h-1)
    return img[yi, xi].numpy()


def _apply_axis_x_offset(xy, depth, dx_px, image_size=512):
    """Diagnostic image-plane translation only, after stock Pixal3D projection.

    Re-evaluate bounds AFTER the shift, so points moving into/out of the
    image are counted correctly. This does not modify camera matrices or 3D.
    The caller supplies CPU float tensors [N,2], [N].
    """
    adjusted = xy.clone()
    adjusted[:, 0] += float(dx_px)
    valid = (torch.isfinite(adjusted).all(dim=1) & torch.isfinite(depth)
             & (adjusted[:, 0] >= 0) & (adjusted[:, 0] < image_size)
             & (adjusted[:, 1] >= 0) & (adjusted[:, 1] < image_size)
             & (depth > 0))
    return adjusted, valid.numpy().astype(bool)



def _cell_corners(points, unit):
    """Eight corners of each 64^3 voxel, in the *same* Blender 3-D frame.

    ProjGrid maps [x,y,z] -> [x,-z,y], and its spacing after mesh_scale/2
    is 1/(63*mesh_scale). `unit` is exactly that half-cell spacing.
    """
    from itertools import product
    sign=torch.tensor(list(product((-1.,1.),repeat=3)),
                      device=points.device,dtype=points.dtype)
    offsets=torch.stack((sign[:,0],-sign[:,2],sign[:,1]),dim=-1)*float(unit)
    return (points[:, :, None, :]+offsets[None,None,:,:]).reshape(1,-1,3)


def _project_corners(corners, calc_mat, fov, dx_px, n_cells):
    """Project each voxel's 8 corners, applying the fixed-axis offset equally.

    Returns [N,8,2] image pixels and [N] all-corners-valid indicator.
    A voxel intersecting/exiting the camera frame is UNKNOWN, not background.
    """
    xy, depth, _=project_points_to_image_batch(corners,calc_mat,fov,512)
    xy_cpu=xy[0].detach().cpu().float()
    z_cpu=depth[0].detach().cpu().float()
    xy_fix,valid=_apply_axis_x_offset(xy_cpu,z_cpu,dx_px)
    return xy_fix.numpy().reshape(n_cells,8,2),valid.reshape(n_cells,8).all(axis=1)


def _footprint_mask_hit(alpha, corner_xy, corners_valid, guard=3):
    """Conservative silhouette overlap using a projected 8-corner AABB.

    The AABB encloses the projected cell. A point is confirmed background only
    when *no* nonzero foreground mask pixel intersects this rectangle plus
    guard pixels. This intentionally yields false UNKNOWN rather than false FREE.
    The method is more conservative than an exact projected hexahedron test.
    """
    h,w=alpha.shape
    if corner_xy.ndim!=3 or corner_xy.shape[1:]!=(8,2):
        raise ValueError(f'Expected corner pixels [N,8,2], got {corner_xy.shape}')
    valid=np.asarray(corners_valid,dtype=bool)
    finite=np.isfinite(corner_xy).all(axis=(1,2))
    valid &= finite
    finite_xy=np.nan_to_num(corner_xy,nan=0.0,posinf=0.0,neginf=0.0)
    pad=int(guard)
    if pad<0: raise ValueError('guard must be nonnegative')
    x0=np.floor(finite_xy[:,:,0].min(axis=1)).astype(np.int64)-pad
    x1=np.ceil(finite_xy[:,:,0].max(axis=1)).astype(np.int64)+pad
    y0=np.floor(finite_xy[:,:,1].min(axis=1)).astype(np.int64)-pad
    y1=np.ceil(finite_xy[:,:,1].max(axis=1)).astype(np.int64)+pad
    x0=np.clip(x0,0,w-1);x1=np.clip(x1,0,w-1)
    y0=np.clip(y0,0,h-1);y1=np.clip(y1,0,h-1)
    foreground=np.asarray(alpha)>0.05  # include low-alpha antialiasing boundary
    prefix=np.zeros((h+1,w+1),dtype=np.int64)
    prefix[1:,1:]=np.cumsum(np.cumsum(foreground.astype(np.int32),axis=0),axis=1)
    totals=(prefix[y1+1,x1+1]-prefix[y0,x1+1]
            -prefix[y1+1,x0]+prefix[y0,x0])
    intersects=valid&(totals>0)
    empty=valid&(totals==0)
    return intersects,empty,valid


def _fit_depth(x, y, ok, unit, min_samples=25):
    """Robust per-view affine calibration: camera_depth = a*relative_depth+b."""
    x, y = x[ok].astype(np.float64), y[ok].astype(np.float64)
    sel = np.isfinite(x) & np.isfinite(y)
    x, y = x[sel], y[sel]
    if len(x) < min_samples or np.std(x) < 1e-7 or np.std(y) < 1e-7:
        return {'usable': False, 'reason': 'insufficient or constant depth', 'n': len(x)}
    def fit(a, b):
        xc = a-a.mean()
        den = float(np.dot(xc, xc))
        if den < 1e-12:
            return None
        s = float(np.dot(xc, b-b.mean())/den)
        return s, float(b.mean()-s*a.mean())
    p = fit(x,y)
    if p is None:
        return {'usable': False, 'reason':'degenerate fit', 'n':len(x)}
    pred = p[0]*x+p[1]
    err = y-pred
    m = np.median(err)
    sigma = 1.4826*np.median(np.abs(err-m))
    keep = np.abs(err-m) <= max(3*sigma, unit)
    if np.count_nonzero(keep) >= min_samples:
        p2=fit(x[keep],y[keep])
        if p2 is not None:
            p=p2
            x,y=x[keep],y[keep]
    corr=float(np.corrcoef(x,y)[0,1])
    p90=float(np.quantile(np.abs(y-(p[0]*x+p[1])),.90)/unit)
    good=np.isfinite(corr) and np.isfinite(p90) and abs(corr)>=0.30 and p90<=8.0
    return {'usable':bool(good), 'a':p[0], 'b':p[1], 'n':len(x),
            'corr':corr, 'p90_error_vox':p90}


def _angular_agree(evidence, dirs, min_views, angle):
    """Only count supporting views with sufficiently separated camera axes."""
    out=np.zeros(evidence.shape[1],dtype=np.bool_)
    threshold=math.cos(math.radians(angle))
    for i in np.flatnonzero(evidence.sum(axis=0)>=min_views):
        chosen=[]
        for v in np.flatnonzero(evidence[:,i]):
            if all(np.dot(dirs[v],dirs[j])<threshold for j in chosen):
                chosen.append(v)
            if len(chosen)>=min_views:
                out[i]=True
                break
    return out


def _export_ply(path, xyz, cls):
    palette=((150,150,150),(245,62,57),(248,164,30),(32,191,161))
    with path.open('w',encoding='utf-8') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(xyz)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for p,c in zip(xyz,cls):
            r,g,b=palette[int(c)]
            x,y,z=(p+.5)/64-.5
            f.write(f'{x:.7f} {y:.7f} {z:.7f} {r} {g} {b}\n')


class Orbit3DHRFreeSpaceAuditFilter:
    CATEGORY='Orbit to 3D/Experimental'
    FUNCTION='process'
    RETURN_TYPES=('COORDS','INT','STRING','INT','INT')
    RETURN_NAMES=('hr_coords','hr_grid_resolution','report_path','confirmed','removed')
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
            'mode':(['audit_only','filter_confirmed'],{'default':'audit_only'}),
            'depth_sources':(['both','moge','marigold','none'],{'default':'both'}),
            'min_independent_views':('INT',{'default':2,'min':2,'max':4}),
            'min_view_angle_deg':('FLOAT',{'default':15.,'min':1.,'max':90.,'step':1.}),
            # axis_fixed (calibrated on opposite-view silhouettes): the real
            # rotation axis is at x=1243.5 in the 2432-px source image.
            # The diagnostic image-plane shift is +5.78947 px on 512 maps.
            'axis_x_source_px':('FLOAT',{'default':1243.5,'min':0.,'max':10000.,'step':0.25}),
            'source_image_width_px':('INT',{'default':2432,'min':32,'max':16384}),
            'mask_evidence_mode':(['voxel_footprint','center_legacy'],{'default':'voxel_footprint',
                'tooltip':'Use projected volume of full voxel, not only center, for silhouette free-space evidence.'}),
            'mask_guard_pixels':('INT',{'default':3,'min':0,'max':12}),
            'depth_front_margin_voxels':('FLOAT',{'default':3.,'min':.5,'max':15.,'step':.5}),
            'normal_confidence_min':('FLOAT',{'default':.60,'min':.5,'max':1.,'step':.02}),
            'protect_original_radius_voxels':('INT',{'default':1,'min':0,'max':3}),
            'report_directory':('STRING',{'default':'/home/ai/orbit3d-cache/experiments/hr_free_space','multiline':False}),
            'preview_views':('INT',{'default':8,'min':0,'max':16}),
        }}

    @torch.no_grad()
    def process(self, pipeline, shape_slat, decode_resolution,
                original_ss64_coords, original_ss64_resolution, pixal3d_mv_views,
                mode='audit_only', depth_sources='both', min_independent_views=2,
                min_view_angle_deg=15., mask_evidence_mode='voxel_footprint', mask_guard_pixels=3,
                depth_front_margin_voxels=3., normal_confidence_min=.6,
                axis_x_source_px=1243.5, source_image_width_px=2432,
                protect_original_radius_voxels=1,
                report_directory='/home/ai/orbit3d-cache/experiments/hr_free_space',
                preview_views=8):
        _require_pixal3d_mv(pipeline)
        if not isinstance(shape_slat,SparseTensor):
            raise TypeError('shape_slat must be SparseTensor')
        if int(decode_resolution)!=1024 or int(original_ss64_resolution)!=64:
            raise ValueError('This audit requires 64^3 SLat, decode_resolution=1024 and original SS64')
        original=_validate(original_ss64_coords,64,'original_ss64_coords')
        slat_coords=_validate(shape_slat.coords,64,'shape_slat.coords')
        xyz_key=lambda arr: arr[:,1]*4096+arr[:,2]*64+arr[:,3]
        if not np.isin(xyz_key(original), xyz_key(slat_coords)).all():
            raise ValueError('Original SS64 is not contained in the Shape512 SLat coords')
        if mode not in ('audit_only','filter_confirmed'):
            raise ValueError(f'Unknown mode={mode!r}')
        if depth_sources not in ('both','moge','marigold','none'):
            raise ValueError(f'Unknown depth_sources={depth_sources!r}')
        if mask_evidence_mode not in ('voxel_footprint','center_legacy'):
            raise ValueError(f'Unknown mask_evidence_mode={mask_evidence_mode!r}')
        views=pixal3d_mv_views
        if views.get('alphas') is None:
            raise ValueError('PIXAL3D_MV_VIEWS is missing foreground alphas')
        cam=views['transform_matrix'].float()
        B,V=cam.shape[:2]
        if B!=1 or V<int(min_independent_views):
            raise ValueError('Only B=1 and enough MV cameras supported')
        device=torch.device(pipeline.device)
        cam=cam.to(device)
        distance=views['camera_distance'].to(device,dtype=torch.float32)
        fov=views['camera_angle_x'].to(device,dtype=torch.float32)
        scale=float(torch.as_tensor(views['mesh_scale']).reshape(-1)[0].item())
        if scale<=0:
            raise ValueError('Invalid mesh_scale')
        unit=1.0/63.0/scale
        # Offset the diagnostic 2D sampling coordinates, not the physical
        # camera transforms, SLat coordinates, or production projections.
        src_w=int(source_image_width_px)
        axis_x=float(axis_x_source_px)
        if not math.isfinite(axis_x) or not (0. <= axis_x < src_w):
            raise ValueError('axis_x_source_px must be finite and inside source image width')
        dx_512=(axis_x - src_w / 2.0) * (512.0 / src_w)
        print(f'[HR FreeSpace] axis_fixed diagnostic: source X={axis_x:.3f}/'
              f'{src_w}, sample shift X={dx_512:+.5f} px @512; production unchanged',
              flush=True)

        # Extract HR coords through the trained VAE subdivision path. No Shape1024.
        pipeline.load_shape_slat_decoder()
        decoder=pipeline.models['shape_slat_decoder']
        if pipeline.low_vram:
            decoder.to(device)
            decoder.low_vram=True
        try:
            raw=decoder.upsample(shape_slat,upsample_times=4)
        finally:
            if pipeline.low_vram:
                decoder.cpu()
                decoder.low_vram=False
            if not pipeline.keep_models_loaded:
                pipeline.unload_shape_slat_decoder()
        xyz64=((raw[:,1:].float()+.5)/float(decode_resolution)*64).long()
        hr=torch.unique(torch.cat((raw[:,:1].long(),xyz64),dim=1),dim=0).contiguous()
        xyz=_validate(hr,64,'hr_coords')[:,1:]
        n=len(xyz)
        if n>200000:
            raise ValueError('HR64 token budget above audit limit (200000)')
        del raw

        orig_mask=np.zeros((64,64,64),dtype=bool)
        orig_mask[original[:,1],original[:,2],original[:,3]]=True
        radius=int(protect_original_radius_voxels)
        if radius:
            from scipy.ndimage import binary_dilation
            struct=np.zeros((3,3,3),dtype=bool)
            struct[1,1,:]=True
            struct[1,:,1]=True
            struct[:,1,1]=True
            protected_grid=binary_dilation(orig_mask,structure=struct,iterations=radius)
        else:
            protected_grid=orig_mask
        protected=protected_grid[xyz[:,0],xyz[:,1],xyz[:,2]]

        # Match native Pixal3D ProjGrid (including Blender rotation and mesh scale).
        pj=ProjGrid(grid_resolution=64,image_resolution=512).to(device)
        cm=compute_relative_calc_mat(cam,distance,pj.front_view_transform_matrix)
        base=pj.grid_points
        def world(rows):
            flat=rows[:,0]*4096+rows[:,1]*64+rows[:,2]
            return (base[torch.as_tensor(flat,device=device)]/scale/2.)[None]
        pts=world(xyz)
        anchors=world(original[:,1:])
        # Prepared once; per-view calls only project. Memory O(8*N), not O(V*8*N).
        hr_corners=_cell_corners(pts,unit)
        original_corners=_cell_corners(anchors,unit)
        dirs=cam[0,:,:3,2].detach().cpu().numpy().astype(np.float64)
        dirs/=np.maximum(np.linalg.norm(dirs,axis=1,keepdims=True),1e-12)
        bg=np.zeros((V,n),dtype=np.bool_)
        moge=np.zeros((V,n),dtype=np.bool_)
        marigold=np.zeros((V,n),dtype=np.bool_)
        diagnostics=[]
        previews={}
        footprint_hr_totals=[]
        footprint_original_totals=[]
        requested=({'both':['moge','marigold'],'moge':['moge'],
                     'marigold':['marigold'],'none':[]})[depth_sources]
        available=[k for k in requested if views.get(k+'_depth_native') is not None
                                     and views.get(k+'_alpha_native') is not None]
        if requested and not available:
            raise ValueError('No requested native depth maps in PIXAL3D_MV_VIEWS')
        if len(available)!=len(requested):
            print('[HR FreeSpace] Some depth sources missing: '+str(set(requested)-set(available)),flush=True)
        preview_ids=set(np.linspace(0,V-1,num=min(int(preview_views),V),dtype=int).tolist())
        for v in range(V):
            pr,d,valid=project_points_to_image_batch(pts,cm[0:1,v],fov[0:1,v],512)
            px_raw=pr[0].detach().cpu()
            depth_cpu=d[0].detach().cpu().float()
            depth=depth_cpu.numpy()
            px,pv=_apply_axis_x_offset(px_raw,depth_cpu,dx_512)
            apr,ad,avalid=project_points_to_image_batch(anchors,cm[0:1,v],fov[0:1,v],512)
            apx_raw=apr[0].detach().cpu()
            adepth_cpu=ad[0].detach().cpu().float()
            adepth=adepth_cpu.numpy()
            apx,apv=_apply_axis_x_offset(apx_raw,adepth_cpu,dx_512)
            alpha=_getmap(views,'alphas',v)
            if int(mask_guard_pixels):
                k=2*int(mask_guard_pixels)+1
                a_guard=F.max_pool2d(alpha[None,None],kernel_size=k,stride=1,
                       padding=int(mask_guard_pixels))[0,0]
            else:
                a_guard=alpha
            obs_alpha=_sample(alpha,px)
            obs_guard=_sample(a_guard,px)
            bg_center=pv&(obs_guard<.05)
            corner_px, corner_ok=_project_corners(hr_corners,cm[0:1,v],fov[0:1,v],
                                                   dx_512,n)
            cell_fg,cell_bg,cell_valid=_footprint_mask_hit(alpha.numpy(),corner_px,
                                    corner_ok,guard=int(mask_guard_pixels))
            bg[v] = (cell_bg if mask_evidence_mode=='voxel_footprint' else bg_center)
            anchor_alpha=_sample(alpha,apx)
            orig_corner_px,orig_corner_ok=_project_corners(original_corners,
                                     cm[0:1,v],fov[0:1,v],dx_512,len(original))
            orig_fg,orig_bg,orig_valid=_footprint_mask_hit(alpha.numpy(),orig_corner_px,
                                         orig_corner_ok,guard=int(mask_guard_pixels))
            footprint_hr_totals.append((int(cell_valid.sum()),int(cell_bg.sum()),int(bg_center.sum())))
            footprint_original_totals.append((int(orig_valid.sum()),int(orig_fg.sum()),int(orig_bg.sum())))
            ax=apx[:,0].round().long().clamp(0,511).numpy()
            ay=apx[:,1].round().long().clamp(0,511).numpy()
            aid=ay*512+ax
            zbuf=np.full(512*512,np.inf,dtype=np.float32)
            np.minimum.at(zbuf,aid[apv],adepth[apv])
            anchor_front=apv&(adepth<=zbuf[aid]+1.5*unit)&(anchor_alpha>.8)
            # Alignment audit: project the *unmodified* SS64 against the exact
            # mask. This is only QA, not a translation/correction to the camera.
            al_np=alpha.numpy()
            fy,fx=np.where(al_np>.5)
            anchor_pixels=apx.numpy()
            raw_anchor_pixels=apx_raw.numpy()
            raw_anchor_valid=avalid[0].detach().cpu().numpy().astype(bool)
            finite_anchor=apv & np.isfinite(anchor_pixels).all(axis=1)
            if len(fx) and int(finite_anchor.sum()):
                xy_ref=anchor_pixels[finite_anchor]
                mask_box=[int(fx.min()),int(fy.min()),int(fx.max()),int(fy.max())]
                projected_box=[float(xy_ref[:,0].min()),float(xy_ref[:,1].min()),
                               float(xy_ref[:,0].max()),float(xy_ref[:,1].max())]
                bbox_delta=[float((projected_box[0]+projected_box[2]-mask_box[0]-mask_box[2])/2),
                            float((projected_box[1]+projected_box[3]-mask_box[1]-mask_box[3])/2)]
                inside_fraction=float((anchor_alpha[finite_anchor]>.5).mean())
            else:
                mask_box=None; projected_box=None; bbox_delta=None; inside_fraction=None
            rd={'view':v,'name':views.get('view_names',[f'view{i}' for i in range(V)])[v],
                'bg_evidence':int(bg[v].sum()), 'depth_fits':{},
                'silhouette_footprint':{
                    'hr_valid_full_cell':int(cell_valid.sum()),
                    'hr_provably_background':int(cell_bg.sum()),
                    'hr_center_background_legacy':int(bg_center.sum()),
                    'original_ss64_valid_full_cell':int(orig_valid.sum()),
                    'original_ss64_footprint_intersects_foreground':int(orig_fg.sum()),
                    'original_ss64_footprint_provably_background':int(orig_bg.sum()),
                },
                'alignment':{
                    'original_ss64_projected_count':int(finite_anchor.sum()),
                    'original_ss64_inside_foreground_fraction':inside_fraction,
                    'original_ss64_footprint_intersects_fraction':float(orig_fg.sum()/max(int(orig_valid.sum()),1)),
                    'original_ss64_footprint_valid_count':int(orig_valid.sum()),
                    'mask_bbox_xyxy':mask_box,
                    'original_ss64_projected_bbox_xyxy':projected_box,
                    'original_minus_mask_bbox_center_xy_px':bbox_delta,
                    'axis_shift_x_px_512':float(dx_512),
                    'original_inside_mask_before_offset':float(
                        (_sample(alpha,apx_raw)[raw_anchor_valid]>.5).mean()
                    ) if raw_anchor_valid.any() else None,
                    'note':'Bounds and inside fractions are measured after the fixed-axis diagnostic shift; no camera transforms changed.'}}
            for key in available:
                dm=_getmap(views,key+'_depth_native',v)
                am=_getmap(views,key+'_alpha_native',v)
                nm=_getmap(views,'moge_geometry_confidence' if key=='moge' else 'geometry_confidence',v)
                axmask=_sample(am,apx)>.8
                if nm is not None:
                    axmask &= _sample(nm,apx)>=float(normal_confidence_min)
                fit=_fit_depth(_sample(dm,apx),adepth,anchor_front&axmask,unit)
                nfront=0
                if fit['usable']:
                    obsd=_sample(dm,px)
                    alpha_d=_sample(am,px)
                    norm_d=_sample(nm,px) if nm is not None else np.ones(n)
                    estimated=fit['a']*obsd+fit['b']
                    margin=max(float(depth_front_margin_voxels),2*fit['p90_error_vox'])*unit
                    front=(pv&(obs_alpha>.8)&(alpha_d>.8)&np.isfinite(obsd)
                           &(norm_d>=float(normal_confidence_min))&(depth<estimated-margin))
                    if key=='moge':
                        moge[v]=front
                    else:
                        marigold[v]=front
                    nfront=int(front.sum())
                rd['depth_fits'][key]={'calibration':fit,'front_evidence':nfront}
            diagnostics.append(rd)
            if v in preview_ids:
                previews[v]=(px.numpy(),pv,alpha.numpy(),anchor_pixels,finite_anchor,
                             mask_box,projected_box,raw_anchor_pixels,raw_anchor_valid)
            if (v+1)%20==0 or v+1==V:
                print(f'[HR FreeSpace] audited {v+1}/{V} views',flush=True)

        evidence=bg|moge|marigold
        confirmed=_angular_agree(evidence,dirs,int(min_independent_views),
                                 float(min_view_angle_deg))
        removable=confirmed&~protected
        class_id=np.zeros(n,dtype=np.uint8)
        class_id[protected]=3
        class_id[confirmed&protected]=2
        class_id[removable]=1
        root=Path(str(report_directory)).expanduser().resolve()
        root.mkdir(parents=True,exist_ok=True)
        out=root/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
        out.mkdir(parents=False,exist_ok=False)
        _export_ply(out/'hr_free_space_audit.ply',xyz,class_id)
        np.savez_compressed(out/'hr_free_space_audit.npz',
            hr_coords=hr.detach().cpu().numpy().astype(np.int32),
            original_ss64_coords=original.astype(np.int32),
            classification=class_id, mask_votes=bg.sum(axis=0).astype(np.int16),
            moge_front_votes=moge.sum(axis=0).astype(np.int16),
            marigold_front_votes=marigold.sum(axis=0).astype(np.int16),
            total_evidence_views=evidence.sum(axis=0).astype(np.int16),
            confirmed=confirmed,protected=protected)
        if previews:
            from PIL import Image,ImageDraw
            for v,(px,valid,alpha,original_xy,original_valid,mask_box,projected_box,
                   original_xy_raw,original_valid_raw) in previews.items():
                pixels=(np.clip(alpha,0,1)*180+35).astype(np.uint8)
                im=Image.fromarray(pixels,'L').convert('RGB')
                draw=ImageDraw.Draw(im)
                for i in np.flatnonzero(valid):
                    if class_id[i]==0 and i%4:
                        continue
                    x,y=px[i]
                    if class_id[i]==1: color=(250,50,40); rad=2
                    elif class_id[i]==2: color=(255,172,30); rad=2
                    elif class_id[i]==3: color=(44,185,158); rad=1
                    else: color=(155,155,155); rad=1
                    draw.ellipse((x-rad,y-rad,x+rad,y+rad),fill=color)
                im.save(out/f'overlay_view_{v:03d}.png')

                # The reference overlay is intentionally SEPARATE from HR and
                # ignores all SLat / Fill coords. Cyan is original SS64 only.
                ref=Image.fromarray(pixels,'L').convert('RGB')
                rdw=ImageDraw.Draw(ref)
                for x,y in original_xy[original_valid]:
                    rdw.point((float(x),float(y)),fill=(25,235,240))
                if mask_box is not None:
                    rdw.rectangle(mask_box,outline=(110,240,60),width=1)
                if projected_box is not None:
                    rdw.rectangle(projected_box,outline=(234,70,223),width=1)
                ref.save(out/f'original_ss64_alignment_view_{v:03d}.png')
                # Control overlay uses the prior, unshifted projection.
                control=Image.fromarray(pixels,'L').convert('RGB')
                cd=ImageDraw.Draw(control)
                for x,y in original_xy_raw[original_valid_raw]:
                    cd.point((float(x),float(y)),fill=(25,235,240))
                if mask_box is not None:
                    cd.rectangle(mask_box,outline=(110,240,60),width=1)
                control.save(out/f'original_ss64_alignment_unshifted_view_{v:03d}.png')
        count_flagged=int(confirmed.sum())
        count_removed=int(removable.sum()) if mode=='filter_confirmed' else 0
        report={
            'version':'ORBIT_HR_FREE_SPACE_FOOTPRINT_V2',
            'mode':mode,'stage':'pre_cascade_hr64','cascade_executed':False,
            'num_views':int(V),'requested_depth_sources':requested,'available_depth_sources':available,
            'original_ss64_tokens':len(original),'shape512_slat_tokens':len(slat_coords),
            'hr64_tokens':n,'confirmed_free_space':count_flagged,
            'confirmed_but_protected':int((confirmed&protected).sum()),
            'confirmed_removable':int(removable.sum()),
            'removed':count_removed,'kept':n-count_removed,
            'mask_evidence_any':int(bg.any(axis=0).sum()),
            'silhouette_footprint_summary':{
                'hr_valid_cell_projections_total':int(sum(r[0] for r in footprint_hr_totals)),
                'hr_provably_background_total':int(sum(r[1] for r in footprint_hr_totals)),
                'hr_center_background_legacy_total':int(sum(r[2] for r in footprint_hr_totals)),
                'original_ss64_valid_cell_projections_total':int(sum(r[0] for r in footprint_original_totals)),
                'original_ss64_intersecting_foreground_total':int(sum(r[1] for r in footprint_original_totals)),
                'original_ss64_provably_background_total':int(sum(r[2] for r in footprint_original_totals)),
            },
            'moge_depth_front_any':int(moge.any(axis=0).sum()),
            'marigold_depth_front_any':int(marigold.any(axis=0).sum()),
            'parameters':{
               'min_independent_views':int(min_independent_views),
               'min_view_angle_deg':float(min_view_angle_deg),
               'axis_x_source_px':float(axis_x),
               'source_image_width_px':int(src_w),
               'axis_shift_x_px_512':float(dx_512),
               'mask_evidence_mode':str(mask_evidence_mode),
               'mask_guard_pixels':int(mask_guard_pixels),
               'depth_front_margin_voxels':float(depth_front_margin_voxels),
               'normal_confidence_min':float(normal_confidence_min),
               'protect_original_radius_voxels':radius,
            },
            'colors':{'gray':'unknown','red':'confirmed removable',
                      'orange':'confirmed but protected','teal':'original SS64 vicinity'},
            'views':diagnostics,
        }
        (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,
                                allow_nan=False),encoding='utf-8')
        if mode=='filter_confirmed':
            keep=torch.from_numpy(~removable).to(device=hr.device)
            result=hr[keep].contiguous()
        else:
            result=hr
        print(f'[HR FreeSpace] mode={mode}, HR={n}, flagged={count_flagged}, '
              f'protected={int((confirmed&protected).sum())}, removed={count_removed}. '
              f'Report {out}/report.json',flush=True)
        return (result,64,str(out/'report.json'),count_flagged,count_removed)
