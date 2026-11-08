"""Read-only export of the exact Pixal3D MultiView bundle camera parameters.

Does not rebuild camera poses, alter view tensors, or touch sparse/shape generation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _small_numpy(value, field: str) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if not np.isfinite(arr).all():
        raise ValueError(f'{field}: non-finite camera value')
    return arr


def build_camera_metadata(views: dict) -> dict:
    """Only extract camera metadata; preserve the original views object unchanged."""
    if not isinstance(views, dict):
        raise TypeError('pixal3d_mv_views must be a dictionary')
    for name in ('transform_matrix', 'camera_angle_x', 'camera_distance', 'mesh_scale'):
        if name not in views:
            raise KeyError(f'PIXAL3D_MV_VIEWS lacks {name}')

    poses = _small_numpy(views['transform_matrix'], 'transform_matrix')
    fov = _small_numpy(views['camera_angle_x'], 'camera_angle_x')
    dist = _small_numpy(views['camera_distance'], 'camera_distance')
    if poses.ndim != 4 or poses.shape[0] != 1 or poses.shape[2:] != (4, 4):
        raise ValueError(f'Expected transform_matrix [1,V,4,4], got {poses.shape}')
    vcount = poses.shape[1]
    if fov.shape != (1, vcount) or dist.shape != (1, vcount):
        raise ValueError(f'FOV/distance must be [1,V], got {fov.shape}/{dist.shape}')
    if not np.allclose(poses[:, :, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError('Invalid affine camera_to_world last row')
    if not (np.all((fov > 0) & (fov < math.pi)) and np.all(dist > 0)):
        raise ValueError('Invalid FOV or camera distance')

    names = list(views.get('view_names', []))
    if names and len(names) != vcount:
        raise ValueError('view_names count does not match transform_matrix')
    if not names:
        names = [f'view{i:03d}' for i in range(vcount)]

    native_images = views.get('native_images')
    native_size = None
    if native_images is not None and len(native_images):
        native_size = [int(x) for x in native_images[0].size]
        if len(native_images) != vcount:
            raise ValueError('native_images count does not match camera matrices')

    # This extracts exactly what the Pixal3D node passed, NOT VGGT poses.
    # Pixel K is a derived convenience for the documented centered camera model;
    # it does not encode the measured turntable axis.
    derived_resolutions = []
    if native_size is not None:
        derived_resolutions.append(native_size)
    for res in (512, 1024):
        r = [res, res]
        if r not in derived_resolutions:
            derived_resolutions.append(r)

    camera_matrices = poses[0].astype(np.float64)
    inverse = np.linalg.inv(camera_matrices)
    actual_distance = np.linalg.norm(camera_matrices[:, :3, 3], axis=1)
    if not np.allclose(actual_distance, dist[0], rtol=1e-5, atol=1e-5):
        raise ValueError('camera_distance differs from norm(camera_to_world translation)')

    records = []
    for i in range(vcount):
        angle = float(fov[0, i])
        record = {
            'index': i,
            'view_name': names[i],
            'camera_angle_x_rad': angle,
            'camera_angle_x_deg': math.degrees(angle),
            'camera_distance': float(dist[0, i]),
            'camera_to_world': camera_matrices[i].tolist(),
            'world_to_camera': inverse[i].tolist(),
            'derived_centered_K_pixels': {},
        }
        for w, h in derived_resolutions:
            fx = (w / 2.0) / math.tan(angle / 2.0)
            # Native views are square in this production pipeline. This K is
            # only a derived standard pinhole convention, NOT an exported K.
            record['derived_centered_K_pixels'][f'{w}x{h}'] = [
                [fx, 0.0, w / 2.0],
                [0.0, fx, h / 2.0],
                [0.0, 0.0, 1.0],
            ]
        records.append(record)

    fingerprint = hashlib.sha256(
        np.ascontiguousarray(poses.astype('<f4')).tobytes()
        + np.ascontiguousarray(fov.astype('<f4')).tobytes()
        + np.ascontiguousarray(dist.astype('<f4')).tobytes()
    ).hexdigest()

    return {
        'schema': 'orbit3d_pixal3d_mv_camera_checkpoint_v1',
        'export_time_utc': datetime.now(timezone.utc).isoformat(),
        'source': 'PIXAL3D_MV_VIEWS in memory; no camera reconstruction',
        'camera_bundle_sha256': fingerprint,
        'view_count': vcount,
        'native_image_size_wh': native_size,
        'mesh_scale': float(views['mesh_scale']),
        'conventions': {
            'world': 'Blender/NeRF Z-up',
            'camera_to_world': '4x4; camera looks along local -Z; local +Y up',
            'intrinsics': 'camera_angle_x is horizontal FOV (radians)',
            'derived_centered_K_pixels': 'CONVENIENCE ONLY, not supplied by Pixal3D; principal point at image center',
            'turntable_axis_x': 'NOT STORED in PIXAL3D_MV_VIEWS; do not equate with principal point',
            'view_angles': 'view_name is a label generated from configured azimuth/elevation, not actual VGGT measurements',
            'pixel_transforms': 'native_image_size is not enough to recover original video->VSR->square transforms',
        },
        'views': records,
    }


class Orbit3DExportPixal3DMVCameras:
    """Pass through the SAME PIXAL3D_MV_VIEWS object after atomic JSON export."""

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'pixal3d_mv_views': ('PIXAL3D_MV_VIEWS',),
            'shape512_checkpoint_dir': ('STRING', {
                'default': '/home/ai/orbit3d-cache/experiments/left_hand/shape512',
                'multiline': False,
            }),
            'filename': ('STRING', {'default': 'camera_config.json', 'multiline': False}),
        }}

    RETURN_TYPES = ('PIXAL3D_MV_VIEWS', 'STRING')
    RETURN_NAMES = ('pixal3d_mv_views', 'export_path')
    FUNCTION = 'execute'
    CATEGORY = 'Orbit to 3D/Diagnostics'
    OUTPUT_NODE = True

    def execute(self, pixal3d_mv_views, shape512_checkpoint_dir, filename):
        # Prevent path traversal in filename, but allow explicit external directories.
        filename = str(filename).strip()
        if not filename or Path(filename).name != filename or not filename.endswith('.json'):
            raise ValueError('filename must be a simple .json filename')
        root = Path(os.path.expanduser(shape512_checkpoint_dir)).resolve()
        root.mkdir(parents=True, exist_ok=True)
        destination = root / filename
        metadata = build_camera_metadata(pixal3d_mv_views)
        fd, tmp_name = tempfile.mkstemp(prefix='.orbit3d-camera-', suffix='.tmp', dir=str(root))
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(metadata, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_name, destination)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        print(f'[Orbit3D Camera Export] views={metadata["view_count"]} sha256={metadata["camera_bundle_sha256"][:16]} path={destination}', flush=True)
        return (pixal3d_mv_views, str(destination))
