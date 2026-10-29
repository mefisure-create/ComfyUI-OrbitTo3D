"""
Camera / view-bundle helpers for the Pixal3D multi-view path.

Pixal3D's multi-view conditioning wants the same "dataset-style" bundle that
inference_mv.py builds from a `transforms.json` directory:

    {
        'images':           {image_size: [1, V, 3, S, S]},  alpha-premultiplied, in [0,1]
        'camera_angle_x':   [1, V]    horizontal fov, radians
        'camera_distance':  [1, V]    camera distance (norm of the c2w translation)
        'transform_matrix': [1, V, 4, 4]  camera-to-world, index 0 = MAIN view
        'mesh_scale':       float
        'view_names':       [str] * V
    }

`transform_matrix` follows the Blender/NeRF convention used by the training
renders: a world that is Z-up, each camera looking along its own -Z with its own
+Y as up. Frame 0 is the main view and should be the canonical front view -- a
camera at (0, -d, 0) -- because every other view is placed relative to it
(calc_mat_i = F @ inv(C_0) @ C_i, see compute_relative_calc_mat).

Azimuth / elevation here match the convention used by
Trellis2RenderMultiViewNvdiffrast, which works in the model's own Y-up mesh
frame:

    eye_mesh = d * [cos(el)sin(az), sin(el), cos(el)cos(az)]

so az=0/90/180/270 -> front/left/back/right and el=+/-90 -> top/bottom. ProjGrid
rotates the mesh frame into the Blender frame with (x, y, z) -> (x, -z, y), so
the same camera in Blender coordinates sits at

    eye = d * [cos(el)sin(az), -cos(el)cos(az), sin(el)]

which is exactly (0, -d, 0) at az=el=0 -- the canonical front view.
"""

import os
import json
import math
from typing import *

import numpy as np
import torch
from PIL import Image

from .camera import compute_f_pixels, distance_from_fov


# Blender-frame world up (Z-up).
_WORLD_UP = (0.0, 0.0, 1.0)

# The Pixal3D training rig leaves a 10% margin around the object: the shipped
# assets/mv_images/example/transforms.json has distance 3.1192050 at a 20 deg fov,
# and the fill-the-frame distance for that fov is 2.8356409 -- exactly 1.1x smaller.
# Multi-view renders (and the outputs of most multi-view diffusion models) come
# framed this way, unlike the single-view path, where preprocess_image crops the
# object until it fills the frame.
PIXAL3D_RIG_MARGIN = 1.1

# Resolution the per-view masks are kept at, for silhouette checks only.
ALPHA_SIZE = 512


# Diagnostic-only quantiles can receive V*H*W values.  torch.quantile() has
# implementation limits for very large reductions on some PyTorch versions.
# Keep diagnostics independent of the number/resolution of views by evaluating
# quantiles on a deterministic, uniformly-strided subset.  This helper is NEVER
# used to build conditioning, confidence, depth, visibility, or any generation
# weight.
_DIAG_QUANTILE_MAX_SAMPLES = 1_000_000


def _diag_quantiles(values: torch.Tensor, qs, max_samples: int = _DIAG_QUANTILE_MAX_SAMPLES):
    values = values.reshape(-1)
    n = int(values.numel())
    if n == 0:
        raise ValueError("_diag_quantiles received an empty tensor")

    max_samples = int(max_samples)
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")

    if n > max_samples:
        # Deterministic and allocation-light: this is a view for strided tensors
        # and caps the reduction length without creating a giant index tensor.
        step = (n + max_samples - 1) // max_samples
        values = values[::step]

    q = torch.as_tensor(qs, dtype=torch.float32, device=values.device)
    return torch.quantile(values.float(), q)


def camera_distance_for_extent(camera_angle_x: float, half_extent_px: float,
                               mesh_scale: float = 1.0, image_resolution: int = 512) -> float:
    """
    Distance at which the unit box half-extent projects to `half_extent_px` pixels.

    This is the one relation that has to hold between the views and the cameras: the
    projection maps the [-0.5, 0.5]^3 grid (scaled by mesh_scale) into each image, so
    if the object is drawn smaller than the camera says, every grid point samples too
    far out -- the surface ends up reading the background.
    """
    if half_extent_px <= 0:
        raise ValueError("half_extent_px must be positive")
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    return float(f_pixels * 0.5 / (mesh_scale * half_extent_px))


def camera_distance_for_fov(camera_angle_x: float, mesh_scale: float = 1.0,
                            image_resolution: int = 512, extend_pixel: int = 0) -> float:
    """
    Distance at which a unit object of `mesh_scale` exactly fills the frame.

    Same formula the single-view Pixal3D path uses (Trellis2FovMoGeCameraConfig /
    get_camera_params_wild_moge). Note that the single-view path only gets away with
    it because preprocess_image crops the object to fill the frame first; un-cropped
    multi-view renders usually want PIXAL3D_RIG_MARGIN times this.
    """
    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    target_point = torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel])
    return float(distance_from_fov(
        camera_angle_x, grid_point, target_point, mesh_scale, image_resolution
    )["distance_from_x"])


def measure_object_fill(images: List[Image.Image], alpha_threshold: float = 0.8) -> float:
    """
    Fraction of the frame the object occupies, as the largest silhouette extent
    across the views.

    Mirrors the framing convention preprocess_image establishes for a single view:
    a square crop of side max(bbox_w, bbox_h), so the object's *longest* axis is what
    maps to the frame. Taking the max over views is the tightest rig that still
    contains every view.
    """
    best = 0.0
    for im in images:
        alpha = np.array(im.convert('RGBA').getchannel(3))
        ys, xs = np.nonzero(alpha > alpha_threshold * 255)
        if len(xs) == 0:
            continue
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        best = max(best, max(w, h) / max(im.size))
    if best <= 0:
        raise ValueError("every view is empty; cannot measure the framing")
    return best


def blender_c2w_from_azimuth_elevation(azimuth_deg: float, elevation_deg: float,
                                       distance: float) -> np.ndarray:
    """
    Build a 4x4 camera-to-world matrix in the Blender/NeRF convention Pixal3D uses.

    Args:
        azimuth_deg: 0 = front, 90 = left, 180 = back, 270 = right.
        elevation_deg: positive = above the object.
        distance: camera distance from the origin.

    Returns:
        [4, 4] float64 c2w matrix. Columns are (right, up, back, eye); the camera
        looks along -back.
    """
    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))

    eye = np.array([
        math.cos(el) * math.sin(az),
        -math.cos(el) * math.cos(az),
        math.sin(el),
    ], dtype=np.float64) * float(distance)

    # +Z of the camera points away from the object (the camera looks along -Z).
    back = eye / (np.linalg.norm(eye) + 1e-12)

    world_up = np.array(_WORLD_UP, dtype=np.float64)
    if abs(float(np.dot(back, world_up))) > 0.999:
        # Looking straight down / up: the world up is degenerate, so pick the
        # in-plane reference that the surrounding elevations converge to. At the
        # top pole (back_z > 0) that limit is +Y, at the bottom pole it is -Y;
        # using one fixed vector for both rolls one of the two views by 180 deg.
        world_up = np.array([0.0, 1.0 if back[2] > 0 else -1.0, 0.0], dtype=np.float64)

    right = np.cross(world_up, back)
    right = right / (np.linalg.norm(right) + 1e-12)
    up = np.cross(back, right)
    up = up / (np.linalg.norm(up) + 1e-12)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = back
    c2w[:3, 3] = eye
    return c2w


def to_cond_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    """
    Turn an RGBA view into a conditioning tensor the way training read its views:
    LANCZOS resize, then premultiply by alpha so the background is black.
    """
    image = image.convert('RGBA').resize((image_size, image_size), Image.Resampling.LANCZOS)
    alpha = torch.tensor(np.array(image.getchannel(3))).float() / 255.0
    rgb = torch.tensor(np.array(image.convert('RGB'))).permute(2, 0, 1).float() / 255.0
    return rgb * alpha.unsqueeze(0)

def to_native_cond_tensor(image: Image.Image) -> torch.Tensor:
    """
    Keep the original view resolution, but apply the same RGB/alpha
    convention as to_cond_tensor().
    """
    image = image.convert('RGBA')

    alpha = torch.from_numpy(
        np.array(image.getchannel(3), copy=True)
    ).float() / 255.0

    rgb = torch.from_numpy(
        np.array(image.convert('RGB'), copy=True)
    ).permute(2, 0, 1).float() / 255.0

    return rgb * alpha.unsqueeze(0)

def build_views(
    images: List[Image.Image],
    transform_matrix: Union[np.ndarray, torch.Tensor, List],
    camera_angle_x: Union[float, Sequence[float]],
    mesh_scale: float = 1.0,
    image_sizes: Sequence[int] = (512, 1024),
    view_names: Optional[List[str]] = None,
) -> dict:
    """
    Assemble the views bundle the MV conditioning path consumes.

    Args:
        images: V RGBA PIL images. Alpha is used as the object mask, so views that
            carry no real mask must be matted BEFORE they get here (the views are
            never cropped or rescaled -- the framing has to be the framing the
            cameras describe).
        transform_matrix: [V, 4, 4] c2w matrices, index 0 = main view.
        camera_angle_x: horizontal fov in radians, one value or one per view.
        mesh_scale: per-object mesh scale.
        image_sizes: the stage resolutions to pre-decode (512 for SS / shape-512,
            1024 for the two 1024 stages).
        view_names: optional labels, used only for logging.
    """
    V = len(images)
    if V == 0:
        raise ValueError("build_views needs at least one view")
        
    native_sizes = {im.size for im in images}

    if len(native_sizes) != 1:
        raise ValueError(
            f"Native MV views must have identical sizes, got: "
            f"{sorted(native_sizes)}"
        )

    native_w, native_h = next(iter(native_sizes))

    native_images = torch.stack(
        [to_native_cond_tensor(im) for im in images],
        dim=0,
    )[None]  # [1, V, 3, H, W]

    print(
        f"[Pixal3D MV] native views kept at "
        f"{native_w}x{native_h}"
    )

    tm = torch.as_tensor(np.asarray(transform_matrix, dtype=np.float32), dtype=torch.float32)
    if tm.ndim != 3 or tm.shape[-2:] != (4, 4):
        raise ValueError(f"transform_matrix must be [V, 4, 4], got {tuple(tm.shape)}")
    if tm.shape[0] != V:
        raise ValueError(f"got {V} images but {tm.shape[0]} transform matrices")
    tm = tm[None]                                                        # [1, V, 4, 4]

    if isinstance(camera_angle_x, (int, float)):
        cax = torch.full((1, V), float(camera_angle_x), dtype=torch.float32)
    else:
        cax = torch.tensor([float(a) for a in camera_angle_x], dtype=torch.float32)[None]
        if cax.shape[1] != V:
            raise ValueError(f"got {V} images but {cax.shape[1]} camera_angle_x values")

    # Derive the distance from the pose rather than trusting a separate field, so it
    # can never disagree with transform_matrix.
    camera_distance = torch.norm(tm[:, :, :3, 3], dim=-1)                 # [1, V]

    bundle_images = {
        int(size): torch.stack([to_cond_tensor(im, int(size)) for im in images], dim=0)[None]
        for size in image_sizes                                           # [1, V, 3, S, S]
    }

    # The conditioning images are alpha-premultiplied, so a dark object is
    # indistinguishable from background in them. Keep the masks separately -- that is
    # what a silhouette check (fit_mesh_orientation) needs to tell whether a mesh is
    # posed the way these cameras describe.
    alphas = torch.stack([
        torch.tensor(np.array(im.convert('RGBA')
                              .resize((ALPHA_SIZE, ALPHA_SIZE), Image.Resampling.LANCZOS)
                              .getchannel(3))).float() / 255.0
        for im in images
    ], dim=0)[None]                                                       # [1, V, S, S]

    if view_names is None:
        view_names = [f"view{i:02d}" for i in range(V)]

    print(f"[Pixal3D MV] V={V} ({', '.join(view_names)})")
    print(f"[Pixal3D MV] fov={math.degrees(float(cax[0, 0])):.2f}deg, "
          f"distance={float(camera_distance[0, 0]):.4f}, mesh_scale={float(mesh_scale):.4f}")

    return {
        'images': bundle_images,
        'native_images': native_images,
        'alphas': alphas,
        'camera_angle_x': cax,
        'camera_distance': camera_distance,
        'transform_matrix': tm,
        'mesh_scale': float(mesh_scale),
        'view_names': list(view_names),
    }


def attach_geometry_confidence(
    views: dict,
    normal_maps: torch.Tensor,
    depth_maps: torch.Tensor,
    rgba_images: List[Image.Image],
    base_confidence: float = 0.50,
    normal_bonus: float = 0.50,
) -> dict:
    """
    Geometry B1.1.

    Conservative view ranking from Marigold V2 camera-space normals.

    Important differences from B1.0:

      * depth does NOT affect fusion;
      * mask/background can NOT reduce confidence to zero;
      * grazing surfaces are NOT strongly penalised;
      * normals only provide a small positive preference for a view.

    Confidence:

        C = base_confidence
            + normal_bonus * alpha * max(0, Nz)

    With defaults:

        outside mask        -> 0.50
        grazing, Nz = 0     -> 0.50
        Nz = 0.5            -> 0.75
        frontal, Nz = 1     -> 1.00

    The absolute scale mostly cancels during normalized weighted_average.
    What matters is that a good front-facing view can be preferred by at
    most 1.25x over a grazing/unknown view.

    depth_maps is deliberately kept in the function signature so the
    existing ComfyUI workflow does not need to change. It is validated
    for view count/alignment, but is NOT used to calculate confidence.
    """

    if normal_maps is None or depth_maps is None:
        raise ValueError(
            "normal_maps and depth_maps must both be provided "
            "(B1.1 ignores depth values, but keeps the existing workflow interface)"
        )

    if normal_maps.ndim == 3:
        normal_maps = normal_maps.unsqueeze(0)

    if depth_maps.ndim == 3:
        depth_maps = depth_maps.unsqueeze(0)

    if normal_maps.ndim != 4 or depth_maps.ndim != 4:
        raise ValueError(
            "geometry maps must be [V,H,W,C], got "
            f"normals={tuple(normal_maps.shape)}, "
            f"depth={tuple(depth_maps.shape)}"
        )

    V = len(rgba_images)

    if normal_maps.shape[0] != V:
        raise ValueError(
            f"normal view count mismatch: "
            f"RGB={V}, normals={normal_maps.shape[0]}"
        )

    if depth_maps.shape[0] != V:
        raise ValueError(
            f"depth view count mismatch: "
            f"RGB={V}, depth={depth_maps.shape[0]}"
        )

    native_sizes = {im.size for im in rgba_images}

    if len(native_sizes) != 1:
        raise ValueError(
            f"RGB views have different native sizes: "
            f"{sorted(native_sizes)}"
        )

    W, H = next(iter(native_sizes))

    if tuple(normal_maps.shape[1:3]) != (H, W):
        raise ValueError(
            "normal maps must be pixel-aligned with RGB: "
            f"normals={tuple(normal_maps.shape[1:3])}, "
            f"RGB={(H, W)}"
        )

    if tuple(depth_maps.shape[1:3]) != (H, W):
        raise ValueError(
            "depth maps must be pixel-aligned with RGB: "
            f"depth={tuple(depth_maps.shape[1:3])}, "
            f"RGB={(H, W)}"
        )

    # --------------------------------------------------------------
    # Decode Marigold V2 normals.
    #
    # Stored RGB [0,1] -> camera-space XYZ [-1,1].
    # Camera looks along -Z; visible/front-facing normals have +Z.
    # --------------------------------------------------------------

    n = (
        normal_maps[..., :3]
        .detach()
        .cpu()
        .float()
        .clamp(0.0, 1.0)
    )

    n = n.mul(2.0).sub(1.0)

    # PNG quantization / resize can slightly change vector length.
    n = n / torch.linalg.vector_norm(
        n,
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-6)

    facing = n[..., 2].clamp(0.0, 1.0)

    # --------------------------------------------------------------
    # Alpha only gates the BONUS.
    #
    # It never gates the baseline confidence.
    #
    # Background:
    #     C = 0.80
    #
    # Object:
    #     C = 0.80 ... 1.00
    # --------------------------------------------------------------

    alpha = torch.stack(
        [
            torch.from_numpy(
                np.array(
                    im.convert("RGBA").getchannel(3),
                    copy=True,
                )
            ).float() / 255.0
            for im in rgba_images
        ],
        dim=0,
    )

    base = float(base_confidence)
    bonus = float(normal_bonus)

    if not (0.0 < base <= 1.0):
        raise ValueError(
            f"base_confidence must be in (0,1], got {base}"
        )

    if bonus < 0.0:
        raise ValueError(
            f"normal_bonus must be >= 0, got {bonus}"
        )

    if base + bonus > 1.000001:
        raise ValueError(
            "base_confidence + normal_bonus must not exceed 1.0, "
            f"got {base + bonus}"
        )

    conf = base + bonus * alpha * facing

    conf = conf.clamp(
        min=base,
        max=base + bonus,
    )

    # [V,H,W] -> [1,V,1,H,W]
    views["geometry_confidence"] = (
        conf.unsqueeze(0)
        .unsqueeze(2)
        .contiguous()
    )

    # MARIGOLD512_DIAG_V2
    # Depth still does NOT participate in B1.2N itself.  Preserve the raw native
    # map separately so Shape512 can fit each view's affine depth gauge against
    # the already-built SS/S4 surface and evaluate it independently.
    depth_native = (
        depth_maps[..., 0]
        .detach()
        .cpu()
        .float()
        .contiguous()
    )

    views["marigold_depth_native"] = (
        depth_native.unsqueeze(0)
        .unsqueeze(2)
        .contiguous()
    )

    views["marigold_alpha_native"] = (
        alpha.unsqueeze(0)
        .unsqueeze(2)
        .contiguous()
    )

    inside = alpha > 0.5

    # PIXAL_MV_ARBITRARY_V_V1
    view_names = list(
        views.get(
            "view_names",
            [f"view{v:02d}" for v in range(V)],
        )
    )
    if len(view_names) != V:
        raise ValueError(
            f"view_names has {len(view_names)} entries but RGB/Marigold V={V}"
        )

    if inside.any():
        vals = conf[inside]
        q_vals = _diag_quantiles(vals, (0.10, 0.90))

        print(
            f"[Pixal3D geometry B1.2N] "
            f"normal-only confidence {W}x{H}, V={V}: "
            f"mean={vals.mean().item():.3f}, "
            f"p10={q_vals[0].item():.3f}, "
            f"p90={q_vals[1].item():.3f}; "
            f"depth retained for Shape512 diagnostics"
        )

        for v in range(V):
            m = inside[v] & torch.isfinite(depth_native[v])
            if not m.any():
                continue
            dv = depth_native[v][m]
            nv = conf[v][inside[v]]
            q_nv = _diag_quantiles(nv, (0.10, 0.50, 0.90))
            q_dv = _diag_quantiles(dv, (0.10, 0.50, 0.90))
            print(
                f"[Marigold512 native] view={v} {view_names[v]}: "
                f"normal_conf p10/50/90="
                f"{q_nv[0].item():.3f}/"
                f"{q_nv[1].item():.3f}/"
                f"{q_nv[2].item():.3f}; "
                f"depth p10/50/90="
                f"{q_dv[0].item():.5f}/"
                f"{q_dv[1].item():.5f}/"
                f"{q_dv[2].item():.5f}"
            )

    return views


def attach_moge_shape512_geometry(
    views: dict,
    ss_geometry: dict,
    rgba_images: List[Image.Image],
    base_confidence: float = 0.50,
    normal_bonus: float = 0.50,
) -> dict:
    """
    Attach Shape512 geometry sources derived from the already-batched MoGe views.

    NORMAL source:
        same conservative confidence law as Marigold B1.2N, but evaluated on
        MoGe camera-space normals and MoGe foreground mask.

    DEPTH source:
        use MoGe camera-space Z from `points[..., 2]` rather than the relative
        depth image.  This keeps the signal in the same gauge family that the
        SS / anchored-occlusion code already calibrates successfully.
    """

    required = ("points", "normal", "mask")
    missing = [k for k in required if k not in ss_geometry]
    if missing:
        raise ValueError(f"ss_geometry missing fields for Shape512 MoGe source: {missing}")

    points = ss_geometry["points"]
    normal = ss_geometry["normal"]
    mask = ss_geometry["mask"]

    if not (torch.is_tensor(points) and torch.is_tensor(normal) and torch.is_tensor(mask)):
        raise TypeError("ss_geometry points/normal/mask must all be torch.Tensor")

    V = len(rgba_images)

    points = points.detach().cpu().float()
    normal = normal.detach().cpu().float()
    mask = mask.detach().cpu().bool()

    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"ss_geometry points must be [V,H,W,3], got {tuple(points.shape)}")

    if normal.shape != points.shape:
        raise ValueError(
            f"ss_geometry normal shape {tuple(normal.shape)} != points shape {tuple(points.shape)}"
        )

    if mask.ndim != 3 or tuple(mask.shape) != tuple(points.shape[:3]):
        raise ValueError(
            f"ss_geometry mask must be [V,H,W] matching points, got {tuple(mask.shape)}"
        )

    if points.shape[0] != V:
        raise ValueError(
            f"MoGe Shape512 source view count mismatch: RGB={V}, ss_geometry={points.shape[0]}"
        )

    n = normal / torch.linalg.vector_norm(
        normal, dim=-1, keepdim=True
    ).clamp_min(1e-6)

    # SHAPE_HR_GEOMETRY_V1
    # MoGe normals are in OpenCV camera coordinates (+Z forward into scene).
    # A front-facing surface normal points back toward the camera, i.e. -Z.
    # Using +Nz made every MoGe confidence collapse to the 0.50 floor.
    facing = (-n[..., 2]).clamp(0.0, 1.0)
    alpha = mask.float()

    base = float(base_confidence)
    bonus = float(normal_bonus)

    if not (0.0 < base <= 1.0):
        raise ValueError(f"base_confidence must be in (0,1], got {base}")
    if bonus < 0.0:
        raise ValueError(f"normal_bonus must be >= 0, got {bonus}")
    if base + bonus > 1.000001:
        raise ValueError(
            "base_confidence + normal_bonus must not exceed 1.0, "
            f"got {base + bonus}"
        )

    conf = (base + bonus * alpha * facing).clamp(min=base, max=base + bonus)
    depth_native = points[..., 2].contiguous()

    views["moge_geometry_confidence"] = (
        conf.unsqueeze(0).unsqueeze(2).contiguous()
    )
    views["moge_depth_native"] = (
        depth_native.unsqueeze(0).unsqueeze(2).contiguous()
    )
    views["moge_alpha_native"] = (
        alpha.unsqueeze(0).unsqueeze(2).contiguous()
    )

    inside = mask & torch.isfinite(depth_native)
    view_names = list(views.get("view_names", [f"view{v:02d}" for v in range(V)]))

    if inside.any():
        vals = conf[inside]
        q_vals = _diag_quantiles(vals, (0.10, 0.90))
        print(
            f"[ShapeGeom MoGe] normal-only confidence {points.shape[2]}x{points.shape[1]}, V={V}: "
            f"mean={vals.mean().item():.3f}, "
            f"p10={q_vals[0].item():.3f}, "
            f"p90={q_vals[1].item():.3f}; "
            "depth source = camera-space Z from MoGe points"
        )

        for v in range(V):
            m = inside[v]
            if not m.any():
                continue
            dv = depth_native[v][m]
            nv = conf[v][mask[v]]
            if nv.numel() == 0:
                continue
            q_nv = _diag_quantiles(nv, (0.10, 0.50, 0.90))
            q_dv = _diag_quantiles(dv, (0.10, 0.50, 0.90))
            print(
                f"[ShapeGeom MoGe native] view={v} {view_names[v]}: "
                f"normal_conf p10/50/90="
                f"{q_nv[0].item():.3f}/"
                f"{q_nv[1].item():.3f}/"
                f"{q_nv[2].item():.3f}; "
                f"depth(z) p10/50/90="
                f"{q_dv[0].item():.5f}/"
                f"{q_dv[1].item():.5f}/"
                f"{q_dv[2].item():.5f}"
            )

    return views

def build_views_from_angles(
    images: List[Image.Image],
    azimuths: Sequence[float],
    elevations: Sequence[float],
    camera_angle_x: float,
    mesh_scale: float = 1.0,
    distance: Optional[float] = None,
    image_sizes: Sequence[int] = (512, 1024),
) -> dict:
    """
    Build the views bundle for an orbit described by azimuth / elevation angles.

    `distance` defaults to the framing distance implied by the fov and mesh scale,
    the same one the single-view path derives from MoGe.
    """
    if len(images) != len(azimuths) or len(images) != len(elevations):
        raise ValueError(
            f"images ({len(images)}), azimuths ({len(azimuths)}) and elevations "
            f"({len(elevations)}) must have the same length")

    if distance is None:
        distance = camera_distance_for_fov(camera_angle_x, mesh_scale)

    transform_matrix = np.stack([
        blender_c2w_from_azimuth_elevation(a, e, distance)
        for a, e in zip(azimuths, elevations)
    ], axis=0)

    view_names = [f"azim{int(round(a)) % 360:03d}_elev{int(round(e)):+03d}"
                  for a, e in zip(azimuths, elevations)]

    return build_views(
        images, transform_matrix, camera_angle_x,
        mesh_scale=mesh_scale, image_sizes=image_sizes, view_names=view_names,
    )


def load_rgba(path: str, rembg: Optional[Callable] = None) -> Tuple[Image.Image, bool]:
    """
    Read one view as RGBA, matting it first if it does not already carry a mask.

    Returns (image, was_matted). The alpha test matches preprocess_image: a fully
    opaque alpha channel counts as no mask.
    """
    image = Image.open(path)
    alpha = np.array(image.getchannel(3)) if image.mode == 'RGBA' else None
    if alpha is not None and not np.all(alpha == 255):
        return image.convert('RGBA'), False
    if rembg is None:
        raise ValueError(f"{path} has no alpha channel and no matting model was given")
    return rembg(image).convert('RGBA'), True


def load_views_from_dir(
    views_dir: str,
    num_views: Optional[int] = None,
    rembg: Optional[Callable] = None,
    image_sizes: Sequence[int] = (512, 1024),
) -> dict:
    """
    Load a `transforms.json` view directory, the input format of inference_mv.py.

        <views_dir>/
            transforms.json     mesh_scale + per-frame file_path / transform_matrix
            view00_azim000.png  RGBA alpha is used as the mask if present
            ...

    `camera_angle_x` may be given per frame or once at the top level.
    """
    with open(os.path.join(views_dir, 'transforms.json')) as f:
        meta = json.load(f)
    frames = meta['frames']
    if num_views is not None:
        if num_views > len(frames):
            raise ValueError(f"num_views {num_views} > {len(frames)} views in {views_dir}")
        frames = frames[:num_views]

    def camera_angle_x_of(frame):
        for src in (frame, meta):
            if 'camera_angle_x' in src:
                return float(src['camera_angle_x'])
        raise KeyError(f"camera_angle_x missing for {frame.get('file_path')}")

    paths = [os.path.join(views_dir, fr['file_path']) for fr in frames]
    loaded = [load_rgba(p, rembg) for p in paths]
    rgba = [im for im, _ in loaded]
    matted = sum(was_matted for _, was_matted in loaded)
    if matted:
        print(f"[Pixal3D MV] matted {matted}/{len(paths)} view(s) that had no alpha channel")

    views = build_views(
        rgba,
        np.array([fr['transform_matrix'] for fr in frames], dtype=np.float32),
        [camera_angle_x_of(fr) for fr in frames],
        mesh_scale=float(meta.get('mesh_scale', 1.0)),
        image_sizes=image_sizes,
        view_names=[fr.get('name', os.path.splitext(fr['file_path'])[0]) for fr in frames],
    )
    print(f"[Pixal3D MV] loaded from {views_dir}")
    return views


def check_main_view(views: dict, atol: float = 1e-4) -> float:
    """
    Warn if frame 0 is not the canonical front view, and return the deviation.

    The extractor maps every view through calc_mat_i = F @ inv(C_0) @ C_i, so the
    main view is always snapped onto F -- but if C_0 is not itself a front view,
    the whole rig gets rotated relative to the object and the generated mesh comes
    out in a different frame than the models were trained for.
    """
    from ..trainers.flow_matching.mixins.image_conditioned_proj import ProjGridMV

    F_mat = ProjGridMV(grid_resolution=2, image_resolution=64).front_view_transform_matrix.clone()
    F_mat[1, 3] = -views['camera_distance'][0, 0]
    err = float((views['transform_matrix'][0, 0] - F_mat).abs().max())
    if err > atol:
        print(f"[Pixal3D MV] Warning: main view (frame 0) is not the canonical front view "
              f"(max deviation {err:.3e}). The result will be posed in that view's frame.")
    else:
        print(f"[Pixal3D MV] main view == canonical front view (max err {err:.1e})")
    return err


def views_to(views: dict, device) -> dict:
    """Move every tensor in a views bundle to `device` (images included)."""
    out = dict(views)
    out['images'] = {k: v.to(device) for k, v in views['images'].items()}
    for k in ('camera_angle_x', 'camera_distance', 'transform_matrix'):
        out[k] = views[k].to(device)
    return out
