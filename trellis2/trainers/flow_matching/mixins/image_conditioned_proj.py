"""
View-Aligned (Projection) Image Conditioned Mixin for TRELLIS2

This module implements DINOv3-based feature extraction with view-aligned projection,
supporting camera-aware 3D-to-2D feature mapping.
"""

from typing import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import DINOv3ViTModel
import numpy as np
from PIL import Image, ImageDraw



# =============================================================================
# Projection Utilities
# =============================================================================

def project_points_to_image_batch(
    points_3d: torch.Tensor, 
    transform_matrix: torch.Tensor, 
    camera_angle_x: torch.Tensor, 
    resolution: int = 518
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project 3D points to 2D image coordinates (batch processing).
    
    Args:
        points_3d: torch.Tensor, shape [N, 3] or [B, N, 3], 3D point coordinates (in [-1, 1] range)
        transform_matrix: torch.Tensor, shape [B, 4, 4], camera transformation matrix
        camera_angle_x: torch.Tensor, shape [B], horizontal field of view angle (radians)
        resolution: int, image resolution, default 518
    
    Returns:
        points_2d: torch.Tensor, shape [B, N, 2], image coordinates [x, y]
        depth: torch.Tensor, shape [B, N], depth values
        valid_mask: torch.Tensor, shape [B, N], mask for points within view
    """
    device = points_3d.device
    B = transform_matrix.shape[0]
    
    # Ensure inputs are torch.Tensor on correct device
    if not isinstance(transform_matrix, torch.Tensor):
        transform_matrix = torch.tensor(transform_matrix, dtype=torch.float32, device=device)
    if not isinstance(points_3d, torch.Tensor):
        points_3d = torch.tensor(points_3d, dtype=torch.float32, device=device)
    if not isinstance(camera_angle_x, torch.Tensor):
        camera_angle_x = torch.tensor(camera_angle_x, dtype=torch.float32, device=device)
    
    # Expand points_3d to batch dimension: [N, 3] -> [B, N, 3]
    if points_3d.dim() == 2:
        points_3d_batch = points_3d.unsqueeze(0).expand(B, -1, -1)
    else:
        points_3d_batch = points_3d
    N = points_3d_batch.shape[1]
    
    # Add homogeneous coordinates: [B, N, 3] -> [B, N, 4]
    ones = torch.ones(B, N, 1, device=device, dtype=points_3d_batch.dtype)
    points_homogeneous = torch.cat([points_3d_batch, ones], dim=-1)  # [B, N, 4]
    
    # Compute world to camera transformation matrix
    world_to_camera = torch.linalg.inv(transform_matrix.float()).to(transform_matrix.dtype)  # linalg.inv requires fp32+
    
    # Batch transform to camera coordinate system: [B, N, 4] @ [B, 4, 4]^T -> [B, N, 3]
    points_camera = torch.bmm(points_homogeneous, world_to_camera.transpose(-2, -1))[..., :3]  # [B, N, 3]
    
    # Extract camera coordinates
    x_cam = points_camera[..., 0]  # [B, N]
    y_cam = points_camera[..., 1]  # [B, N]
    z_cam = points_camera[..., 2]  # [B, N]
    
    # Depth value (Z value in camera coordinate system, note Blender camera faces -Z direction)
    depth = -z_cam  # [B, N]
    
    # Compute camera intrinsics (batch processing)
    sensor_width = 32.0  # mm
    focal_length = 16.0 / torch.tan(camera_angle_x / 2.0)  # [B]
    focal_length_pixels = focal_length * resolution / sensor_width  # [B]
    
    # Expand focal_length_pixels dimension for broadcasting: [B] -> [B, 1]
    focal_length_pixels = focal_length_pixels.unsqueeze(1)  # [B, 1]
    
    # Perspective projection to NDC coordinates
    x_ndc = focal_length_pixels * x_cam / (-z_cam + 1e-8)  # [B, N]
    y_ndc = focal_length_pixels * y_cam / (-z_cam + 1e-8)  # [B, N]
    
    # Convert to image coordinates (pixel coordinates)
    x_pixel = x_ndc + resolution / 2.0  # [B, N]
    y_pixel = -y_ndc + resolution / 2.0  # [B, N], flip Y axis
    
    # Create validity mask (points within image range and in front of camera)
    valid_mask = (
        (x_pixel >= 0) & (x_pixel < resolution) &
        (y_pixel >= 0) & (y_pixel < resolution) &
        (depth > 0)  # In front of camera
    )  # [B, N]
    
    points_2d = torch.stack([x_pixel, y_pixel], dim=-1)  # [B, N, 2]
    
    return points_2d, depth, valid_mask


def sample_features(fmap: torch.Tensor, queries_ndc: torch.Tensor) -> torch.Tensor:
    """
    Sample features from feature map at specified NDC coordinates.
    
    Args:
        fmap: torch.Tensor, shape [B, C, H, W], feature map
        queries_ndc: torch.Tensor, shape [B, K, 2], normalized device coordinates
    
    Returns:
        torch.Tensor, shape [B, C, K], sampled features
    """
    B, C, H, W = fmap.shape
    Bq, K, _ = queries_ndc.shape
    assert Bq == B, "Batch size mismatch"

    # grid_sample requires (B, out_h, out_w, 2), here we want K points -> out_h=K, out_w=1
    grid = queries_ndc.view(B, K, 1, 2)  # (B, K, 1, 2)

    # Bilinear interpolation, align_corners=False (consistent with [-1,1] pixel center convention)
    feat = F.grid_sample(
        fmap, grid, mode='bilinear',
        align_corners=False, padding_mode='border'  # border avoids out-of-bound becoming 0
    )  # (B, C, K, 1)

    return feat.squeeze(-1)  # (B, C, K)


# =============================================================================
# Projection Grid Module
# =============================================================================

class ProjGrid(nn.Module):
    """
    3D Grid Projection Module.
    
    Projects a 3D grid of points to 2D image coordinates and samples features 
    from the image feature map at those locations.
    
    This is the core module for view-aligned feature extraction.
    """
    def __init__(self, grid_resolution: int = 16, image_resolution: int = 518):
        super().__init__()
        self.grid_resolution = grid_resolution
        self.image_resolution = image_resolution
        
        # Create 3D grid points
        one_dim = torch.linspace(-1, 1, grid_resolution)
        x, y, z = torch.meshgrid(one_dim, one_dim, one_dim, indexing='ij')
        grid_points = torch.stack((x, y, z), dim=-1)
        
        # Rotation matrix to align with Blender coordinate system
        rotation_matrix = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0]
        ])
        grid_points = torch.matmul(grid_points, rotation_matrix.T)
        grid_points = grid_points.reshape(-1, 3)
        self.register_buffer('grid_points', grid_points)  # [R³, 3]
        
        # Default front view transformation matrix
        front_view_transform_matrix = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.register_buffer("front_view_transform_matrix", front_view_transform_matrix)
        
    def forward(
        self, 
        features_map: torch.Tensor, 
        camera_angle_x: torch.Tensor,
        distance: torch.Tensor,
        mesh_scale: torch.Tensor,
        transform_matrix: Optional[torch.Tensor] = None,
        BHWC: bool = True
    ) -> torch.Tensor:
        """
        Project 3D grid points to image and sample features.
        
        Args:
            features_map: Feature map, shape [B, H, W, C] if BHWC else [B, C, H, W]
            camera_angle_x: Camera FOV angle, shape [B]
            distance: Camera distance, shape [B]
            mesh_scale: Mesh scale factor, shape [B]
            transform_matrix: Optional camera transform matrix, shape [B, 4, 4]
            BHWC: Whether features_map is in BHWC format
            
        Returns:
            Projected features, shape [B, grid_resolution³, C]
        """
        if BHWC:
            B, H, W, C = features_map.shape
        else:
            B, C, H, W = features_map.shape
            
        grid_points = self.grid_points
        grid_points = grid_points.expand(B, -1, -1)
        grid_points = grid_points / mesh_scale.unsqueeze(-1).unsqueeze(-1) / 2  # Scale alignment
        assert transform_matrix is None, "transform_matrix is not None"
        if transform_matrix is None:
            transform_matrix = self.front_view_transform_matrix
            transform_matrix = transform_matrix.expand(B, -1, -1).clone()
            transform_matrix[:, 1, 3] = -distance  # Set camera distance
            
        # Project to image coordinates (simulate Blender projection)
        image_points, depth, valid_mask = project_points_to_image_batch(
            grid_points, transform_matrix, camera_angle_x, self.image_resolution
        )
        
        # Normalize to [-1, 1] for grid_sample
        image_points_norm = (image_points + 0.5) / self.image_resolution * 2 - 1
        
        if BHWC:
            features_map = features_map.permute(0, 3, 1, 2)  # [B, C, H, W]
            
        # Sample features from DINOv3 patch feature map
        x = sample_features(features_map, image_points_norm)  # [B, C, K]
        x = x.permute(0, 2, 1)  # [B, K, C]
   
        return x
    
    def visualize_projection(
        self,
        image: torch.Tensor,
        camera_angle_x: torch.Tensor,
        distance: torch.Tensor,
        mesh_scale: torch.Tensor,
        transform_matrix: Optional[torch.Tensor] = None,
        save_dir: Optional[str] = None,
        prefix: str = "proj_vis",
    ) -> List[Image.Image]:
        """
        Visualize the projected 3D grid points on the input image.
        
        Args:
            image: Input image tensor [B, C, H, W], assumed to be in [0, 1] range
            camera_angle_x: Camera FOV angle, shape [B]
            distance: Camera distance, shape [B]
            mesh_scale: Mesh scale factor, shape [B]
            transform_matrix: Optional camera transform matrix, shape [B, 4, 4]
            save_dir: Directory to save visualizations (optional)
            prefix: Prefix for saved files
            
        Returns:
            List of PIL Images with projected points overlaid
        """
        B = image.shape[0]
        
        # Get projected points
        grid_points = self.grid_points.expand(B, -1, -1)
        grid_points = grid_points / mesh_scale.unsqueeze(-1).unsqueeze(-1) / 2
        assert transform_matrix is None, "transform_matrix is not None"
        if transform_matrix is None:
            transform_matrix = self.front_view_transform_matrix
            transform_matrix = transform_matrix.expand(B, -1, -1).clone()
            transform_matrix[:, 1, 3] = -distance
            
        image_points, depth, valid_mask = project_points_to_image_batch(
            grid_points, transform_matrix, camera_angle_x, self.image_resolution
        )
        
        # Convert image to PIL for visualization
        vis_images = []
        for b in range(B):
            # Convert tensor to PIL image
            img_np = image[b].cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            
            # Resize to image_resolution if needed
            pil_img = Image.fromarray(img_np)
            if pil_img.size != (self.image_resolution, self.image_resolution):
                pil_img = pil_img.resize((self.image_resolution, self.image_resolution), Image.LANCZOS)
            
            # Create a copy for drawing
            vis_img = pil_img.copy()
            draw = ImageDraw.Draw(vis_img)
            
            # Get points for this batch
            pts = image_points[b].cpu().numpy()  # [K, 2]
            depths = depth[b].cpu().numpy()  # [K]
            mask = valid_mask[b].cpu().numpy()  # [K]
            
            # Normalize depth for coloring
            valid_depths = depths[mask]
            if len(valid_depths) > 0:
                d_min, d_max = valid_depths.min(), valid_depths.max()
                if d_max - d_min > 1e-6:
                    depths_norm = (depths - d_min) / (d_max - d_min)
                else:
                    depths_norm = np.ones_like(depths) * 0.5
            else:
                depths_norm = np.ones_like(depths) * 0.5
            
            # Draw projected points
            R = self.grid_resolution
            for i, (pt, d, m, dn) in enumerate(zip(pts, depths, mask, depths_norm)):
                if not m:
                    continue
                    
                x, y = pt
                
                # Color by depth (blue=near, red=far)
                r = int(255 * dn)
                g = int(255 * (1 - abs(2 * dn - 1)))
                b_color = int(255 * (1 - dn))
                color = (r, g, b_color)
                
                # Draw small circle
                radius = 2
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    fill=color,
                    outline=color
                )
            
            vis_images.append(vis_img)
            
            # Save if directory is specified
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{prefix}_batch{b}.png")
                vis_img.save(save_path)
                print(f"Saved projection visualization to: {save_path}")
        
        return vis_images


class ProjGridMV(ProjGrid):
    """
    Multi-view variant of ProjGrid.

    Identical to ProjGrid except it ALLOWS an explicit per-view transform_matrix
    (calc_mat) to be passed in for projection (the base ProjGrid asserts it is
    None and always uses the fixed front-view matrix). This is used by the
    multi-view feature extractor to project the SAME 3D grid through each view's
    relative pose (calc_mat_i = F @ inv(C_0) @ C_i).

    When transform_matrix is None it falls back to the front-view + distance
    behavior, so it is a strict superset of ProjGrid.
    """

    def forward(
        self,
        features_map: torch.Tensor,
        camera_angle_x: torch.Tensor,
        distance: torch.Tensor,
        mesh_scale: torch.Tensor,
        transform_matrix: Optional[torch.Tensor] = None,
        BHWC: bool = True,
    ) -> torch.Tensor:
        if BHWC:
            B, H, W, C = features_map.shape
        else:
            B, C, H, W = features_map.shape

        grid_points = self.grid_points
        grid_points = grid_points.expand(B, -1, -1)
        grid_points = grid_points / mesh_scale.unsqueeze(-1).unsqueeze(-1) / 2  # Scale alignment

        if transform_matrix is None:
            transform_matrix = self.front_view_transform_matrix
            transform_matrix = transform_matrix.expand(B, -1, -1).clone()
            transform_matrix[:, 1, 3] = -distance  # Set camera distance
        # else: use the provided per-view calc_mat directly

        image_points, depth, valid_mask = project_points_to_image_batch(
            grid_points, transform_matrix, camera_angle_x, self.image_resolution
        )

        image_points_norm = (image_points + 0.5) / self.image_resolution * 2 - 1

        if BHWC:
            features_map = features_map.permute(0, 3, 1, 2)  # [B, C, H, W]

        x = sample_features(features_map, image_points_norm)  # [B, C, K]
        x = x.permute(0, 2, 1)  # [B, K, C]
        return x

    def forward_crop(
        self,
        features_map: torch.Tensor,
        camera_angle_x: torch.Tensor,
        distance: torch.Tensor,
        mesh_scale: torch.Tensor,
        transform_matrix: torch.Tensor,
        crop_rect,
        feather: float = 0.20,
        BHWC: bool = True,
    ):
        """
        Project the normal full-image camera into a cropped/resized DINO image.

        crop_rect = (x0, y0, x1, y1), normalized in the ORIGINAL full image.
        Returns:
            features: [B, R^3, C]
            weights:  [B, R^3, 1]
        """

        if BHWC:
            B, H, W, C = features_map.shape
        else:
            B, C, H, W = features_map.shape

        grid_points = self.grid_points.expand(B, -1, -1)
        grid_points = (
            grid_points
            / mesh_scale.unsqueeze(-1).unsqueeze(-1)
            / 2
        )

        image_points, depth, valid_mask = project_points_to_image_batch(
            grid_points,
            transform_matrix,
            camera_angle_x,
            self.image_resolution,
        )

        x0, y0, x1, y1 = crop_rect

        # Full-image pixel position -> normalized [0,1] full-image coordinate.
        full_u = (image_points[..., 0] + 0.5) / self.image_resolution
        full_v = (image_points[..., 1] + 0.5) / self.image_resolution

        # Full-image coordinate -> crop-local [0,1].
        local_u = (full_u - x0) / (x1 - x0)
        local_v = (full_v - y0) / (y1 - y0)

        inside = (
            valid_mask
            & (local_u >= 0.0)
            & (local_u <= 1.0)
            & (local_v >= 0.0)
            & (local_v <= 1.0)
        )

        # Coordinates for grid_sample on the resized crop.
        queries_ndc = torch.stack(
            [
                local_u * 2.0 - 1.0,
                local_v * 2.0 - 1.0,
            ],
            dim=-1,
        )

        if BHWC:
            features_map = features_map.permute(0, 3, 1, 2)

        feat = sample_features(
            features_map,
            queries_ndc,
        ).permute(0, 2, 1)

        # Smooth feathering towards crop boundaries.
        du = torch.minimum(local_u, 1.0 - local_u)
        dv = torch.minimum(local_v, 1.0 - local_v)

        wu = (du / feather).clamp(0.0, 1.0)
        wv = (dv / feather).clamp(0.0, 1.0)

        weight = wu * wv * inside.to(wu.dtype)

        return feat, weight.unsqueeze(-1)

def compute_relative_calc_mat(
    transform_matrix: torch.Tensor,
    distance: torch.Tensor,
    front_view_transform_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the per-view projection matrix (calc_mat) that maps each view into
    the coordinate frame where the MAIN view (index 0) is snapped to the fixed
    front-view pose F (with F's camera distance set to the main view's distance).

    calc_mat_i = F @ inv(C_0) @ C_i

    where C_i = transform_matrix[:, i] (c2w). For i == 0, calc_mat_0 == F exactly,
    which reproduces the single-view behavior.

    Args:
        transform_matrix: [B, V, 4, 4] c2w matrices (index 0 = main view).
        distance: [B, V] camera distances (only index 0 used for F).
        front_view_transform_matrix: [4, 4] the canonical front-view c2w.

    Returns:
        calc_mat: [B, V, 4, 4]
    """
    B, V = transform_matrix.shape[:2]
    device = transform_matrix.device

    # Fixed front matrix with main-view distance in the translation slot.
    F_mat = front_view_transform_matrix.to(device).unsqueeze(0).expand(B, -1, -1).clone()  # [B,4,4]
    F_mat[:, 1, 3] = -distance[:, 0]
    F_mat = F_mat.unsqueeze(1)  # [B,1,4,4]

    C0 = transform_matrix[:, 0:1]  # [B,1,4,4]

    # Do the matrix math in fp32 for numerical stability (inv is sensitive).
    with torch.amp.autocast('cuda', enabled=False):
        C0f = C0.float().expand(B, V, 4, 4).reshape(B * V, 4, 4)
        Cif = transform_matrix.float().reshape(B * V, 4, 4)
        Ff = F_mat.float().expand(B, V, 4, 4).reshape(B * V, 4, 4)
        rel = torch.bmm(torch.linalg.inv(C0f), Cif)   # inv(C_0) @ C_i
        calc = torch.bmm(Ff, rel)                      # F @ rel
    calc_mat = calc.reshape(B, V, 4, 4)
    return calc_mat


# =============================================================================
# DINOv3 Feature Extractor with Projection
# =============================================================================

class DinoV3ProjFeatureExtractor(nn.Module):
    """
    DINOv3 Feature Extractor with View-Aligned Projection.
    
    This extractor produces both:
    1. Global features (CLS token + register tokens) in embed_dim
    2. View-aligned projected features (3D grid projected to 2D and sampled)
       - Without NAF: [B, R³, embed_dim]
       - With NAF:    [B, R³, embed_dim * 2]  (concat of lr and hr features)
    
    NOTE: proj_linear has been moved to per-block ProjectAttention / SparseProjectAttention.
    This module now outputs raw DINOv3 features for proj (optionally concatenated with NAF-upsampled features).
    
    Args:
        model_name: Name of the pretrained DINOv3 model
        image_size: Input image size (default: 512)
        grid_resolution: Resolution of the 3D projection grid (default: 16)
        use_naf_upsample: Whether to use NAF to upsample features (default: False)
        naf_target_size: Target spatial size for NAF upsampling (default: [128, 128])
    """
    def __init__(
        self, 
        model_name: str,
        image_size: int = 512,
        grid_resolution: int = 16,
        use_naf_upsample: bool = False,
        naf_target_size: Optional[List[int]] = None
    ):
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size
        self.grid_resolution = grid_resolution
        self.use_naf_upsample = use_naf_upsample
        if naf_target_size is None:
            self.naf_target_size = (128, 128)
        elif isinstance(naf_target_size, int):
            self.naf_target_size = (naf_target_size, naf_target_size)
        else:
            self.naf_target_size = tuple(naf_target_size)
        
        # Load DINOv3 model (frozen, no trainable params in this module)
        self.model = DINOv3ViTModel.from_pretrained(model_name)
        self.model.eval()
        self.model.requires_grad_(False)
        
        # Image transform (only normalize, no resize - assume already resized)
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Get patch info
        self.patch_size = self.model.config.patch_size
        self.patch_number = image_size // self.patch_size
        self.embed_dim = self.model.config.hidden_size
        
        # Projection grid for view-aligned features
        self.proj_grid = ProjGrid(
            grid_resolution=grid_resolution,
            image_resolution=image_size
        )
        
        # NAF upsampler (frozen, no trainable params)
        self.naf_model = None  # Lazy-loaded on first use to avoid import if not needed

        # Per-axis spatial tile factor for the NAF + proj_grid streaming path. 1 = un-tiled
        # (current behavior, default). >1 enables a streaming wrapper that tiles NAF and
        # the projection together, capping the transient HR-feature-map peak. Set transiently
        # by the pipeline (see Pixal3DImageTo3DPipeline.get_proj_cond_shape).
        self.naf_tile_factor: int = 1

        # proj_channels: the output dimension of proj features
        # Without NAF: embed_dim (e.g. 1024)
        # With NAF: embed_dim * 2 (e.g. 2048, concat of lr and hr)
        self.proj_channels = self.embed_dim * 2 if use_naf_upsample else self.embed_dim
        
        # NOTE: proj_linear removed — now lives in each denoiser block's ProjectAttention
    
    def _cached_naf_repo(self) -> Optional[str]:
        import torch.hub

        hub_dir = torch.hub.get_dir()
        if not os.path.isdir(hub_dir):
            return None
        for name in os.listdir(hub_dir):
            repo_dir = os.path.join(hub_dir, name)
            if name.startswith("valeoai_NAF_") and os.path.exists(os.path.join(repo_dir, "hubconf.py")):
                return repo_dir
        return None    
    
    def _load_naf(self):
        """Lazy-load pretrained NAF model."""
        if self.naf_model is None:
            import torch.hub
            device = next(self.model.parameters()).device            
            cached_repo = self._cached_naf_repo()
            
            if cached_repo is not None:
                self.naf_model = torch.hub.load(
                    cached_repo, "naf", pretrained=True, device=device, source="local", trust_repo=True
                )
            else:
                self.naf_model = torch.hub.load(
                    "valeoai/NAF", "naf", pretrained=True, device=device, trust_repo=True
                )            

            self.naf_model.eval()
            self.naf_model.requires_grad_(False)
        
    def to(self, device):
        super().to(device)
        self.model.to(device)
        self.proj_grid.to(device)
        if self.naf_model is not None:
            self.naf_model.to(device)
        return self

    def cuda(self):
        super().cuda()
        self.model.cuda()
        self.proj_grid.cuda()
        if self.naf_model is not None:
            self.naf_model.cuda()
        return self

    def cpu(self):
        super().cpu()
        self.model.cpu()
        self.proj_grid.cpu()
        if self.naf_model is not None:
            self.naf_model.cpu()
        return self
    
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract features using DINOv3."""
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)

        # transformers < 5
        if hasattr(self.model,'layer'):
            for layer_module in self.model.layer:
                hidden_states = layer_module(
                    hidden_states,
                    position_embeddings=position_embeddings,
                )
        elif hasattr(self.model,'model') and hasattr(self.model.model,'layer'): # transformers >= 5
            for layer_module in self.model.model.layer:
                hidden_states = layer_module(
                    hidden_states,
                    position_embeddings=position_embeddings,
                )
        else:
            raise Exception("Cannot extract features")

        return F.layer_norm(hidden_states, hidden_states.shape[-1:])
    
    def forward(
        self,
        image: Union[torch.Tensor, List[Image.Image]],
        camera_angle_x: Optional[torch.Tensor] = None,
        distance: Optional[torch.Tensor] = None,
        mesh_scale: Optional[torch.Tensor] = None,
        transform_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract view-aligned features from the image.
        
        Args:
            image: Input image tensor [B, C, H, W] or list of PIL images
            camera_angle_x: Camera FOV angle in radians [B]
            distance: Camera distance [B]
            mesh_scale: Mesh scale factor [B]
            transform_matrix: Optional camera transform matrix [B, 4, 4]
        
        Returns:
            Tuple of (global_features, proj_features):
            - global_features: [B, num_global_tokens, embed_dim]
            - proj_features: [B, grid_resolution³, proj_channels]
              where proj_channels = embed_dim (no NAF) or embed_dim*2 (with NAF)
        """
        # Handle input types
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        B = image.shape[0]
        
        # Keep a copy of the unnormalized image for NAF guide
        if self.use_naf_upsample:
            image_for_naf = image.clone()  # [B, 3, H, W], in [0, 1] range
        
        # Apply transform (ImageNet normalization)
        image = self.transform(image)
        
        # Extract DINOv3 features (frozen, no gradients)
        with torch.no_grad():
            z = self.extract_features(image)
            
            # Split into CLS token, register tokens, and patch tokens
            z_clstoken = z[:, 0:1]  # [B, 1, D]
            num_reg = getattr(self.model.config, 'num_register_tokens', 4)
            z_regtokens = z[:, 1:1+num_reg]  # [B, num_reg, D]
            z_patchtokens = z[:, 1+num_reg:]  # [B, num_patches, D]
            
            # Reshape patch tokens to spatial grid: [B, h, w, D]
            z_patchtokens_spatial = z_patchtokens.reshape(
                B, self.patch_number, self.patch_number, -1
            )  # [B, h, w, D]
            
            if camera_angle_x is None or distance is None or mesh_scale is None:
                raise ValueError("camera_angle_x, distance, and mesh_scale must be provided")
            
            # --- Low-resolution branch: sample from DINOv3 patch feature map ---
            z_proj_lr = self.proj_grid(
                z_patchtokens_spatial, 
                camera_angle_x, 
                distance, 
                mesh_scale,
                transform_matrix
            )  # [B, grid_res³, D]
            
            # --- High-resolution branch (NAF): upsample then sample ---
            if self.use_naf_upsample:
                self._load_naf()
                # NAF expects: guide [B, 3, H, W], lr_features [B, C, h, w], target_size (H', W')
                lr_features_bchw = z_patchtokens_spatial.permute(0, 3, 1, 2)  # [B, D, h, w]

                K = getattr(self, 'naf_tile_factor', 1) or 1
                if K <= 1:
                    # Un-tiled path (default, unchanged behavior).
                    hr_features = self.naf_model(
                        image_for_naf, lr_features_bchw, self.naf_target_size
                    )  # [B, D, H', W']

                    # Sample from high-res feature map using same projection coordinates
                    z_proj_hr = self.proj_grid(
                        hr_features,
                        camera_angle_x,
                        distance,
                        mesh_scale,
                        transform_matrix,
                        BHWC=False  # hr_features is [B, C, H', W']
                    )  # [B, grid_res³, D]
                    del hr_features
                else:
                    # Tiled streaming path: bound the transient HR feature map by running
                    # NAF and the projection sample one image-space tile at a time.
                    z_proj_hr = self._proj_naf_tiled(
                        image_for_naf, lr_features_bchw,
                        camera_angle_x, distance, mesh_scale, transform_matrix,
                        tile_factor=int(K),
                    )

                # Concatenate lr and hr: [B, grid_res³, D*2]
                z_proj = torch.cat([z_proj_lr, z_proj_hr], dim=-1)
            else:
                z_proj = z_proj_lr  # [B, grid_res³, D]
                
            # Combine global tokens
            z_global = torch.cat([z_clstoken, z_regtokens], dim=1)  # [B, 1+num_reg, D]
        
        # proj_linear has been moved to per-block ProjectAttention
        # z_proj stays in proj_channels, each block will project independently
        
        return z_global, z_proj

    @torch.no_grad()
    def _proj_naf_tiled(
        self,
        image_for_naf: torch.Tensor,
        lr_features_bchw: torch.Tensor,
        camera_angle_x: torch.Tensor,
        distance: torch.Tensor,
        mesh_scale: torch.Tensor,
        transform_matrix: Optional[torch.Tensor],
        tile_factor: int,
    ) -> torch.Tensor:
        """Streaming NAF + grid_sample. Materialises only one image-space tile of the NAF
        output at a time, capping the transient HR-feature-map peak. Returns the same
        [B, grid_res^3, embed_dim] tensor that proj_grid(naf_model(...)) would produce in
        the un-tiled path.

        Each 3D grid point is routed to exactly one tile (the tile its projection falls in).
        Halo at the NAF output scale = kernel_size//2 (4 for kernel=9) + 2 px safety for the
        2-layer image encoder receptive field. Off-image grid points are clamped to the
        boundary tile so grid_sample's padding_mode='border' recovers the un-tiled border
        behaviour for them.

        adaptive_avg_pool inside NAF re-buckets when called per-tile, so the output is
        near-equivalent to (not bit-equivalent with) the un-tiled call. Boundary diffs are
        small in practice; if they manifest, fall back to naf_tile_factor=1.
        """
        K = int(tile_factor)
        naf_h, naf_w = self.naf_target_size  # always a 2-tuple of ints
        img_size = self.image_size
        img_res = self.proj_grid.image_resolution  # equals image_size for the configured stages

        # Halo at output (NAF) scale, then converted to image scale by the resolution ratio.
        # NAF: single CrossAttention with kernel_size=9 -> 4 px, plus 2 px safety for the
        # 2-conv image encoder. Negligible relative to tile size at K<=8.
        HALO_OUT = 6
        halo_img = max(1, int(round(HALO_OUT * img_size / max(naf_h, naf_w))))

        B = image_for_naf.shape[0]
        device = image_for_naf.device

        # Project all 3D grid points once. The output tensor is small (R^3 * 2 floats).
        grid_pts = self.proj_grid.grid_points.to(device)  # [R^3, 3]
        grid_pts_b = grid_pts.unsqueeze(0).expand(B, -1, -1)
        grid_pts_b = grid_pts_b / mesh_scale.unsqueeze(-1).unsqueeze(-1) / 2

        if transform_matrix is None:
            tm = self.proj_grid.front_view_transform_matrix.to(device)
            tm = tm.unsqueeze(0).expand(B, -1, -1).clone()
            tm[:, 1, 3] = -distance
        else:
            tm = transform_matrix

        image_points, _, _ = project_points_to_image_batch(
            grid_pts_b, tm, camera_angle_x, img_res
        )  # [B, R^3, 2] in pixel coords
        image_points_norm = (image_points + 0.5) / img_res * 2 - 1  # [B, R^3, 2] in [-1, 1]

        # Clamp off-image points to the boundary so they get routed to a boundary tile and
        # produce the same border-feature value as padding_mode='border' would in the
        # un-tiled call.
        image_points_norm = image_points_norm.clamp(-1.0, 1.0)

        u_global = image_points_norm[..., 0]  # [B, R^3]
        v_global = image_points_norm[..., 1]

        R3 = grid_pts.shape[0]
        embed_dim = lr_features_bchw.shape[1]
        z_proj_hr: Optional[torch.Tensor] = None

        for ti in range(K):
            for tj in range(K):
                u_lo = -1.0 + 2.0 * tj / K
                u_hi = -1.0 + 2.0 * (tj + 1) / K
                v_lo = -1.0 + 2.0 * ti / K
                v_hi = -1.0 + 2.0 * (ti + 1) / K

                # Half-open partition with the last tile inclusive on the right/bottom.
                u_in = (u_global >= u_lo) & (u_global <= u_hi if tj == K - 1 else u_global < u_hi)
                v_in = (v_global >= v_lo) & (v_global <= v_hi if ti == K - 1 else v_global < v_hi)
                core_mask = u_in & v_in  # [B, R^3]
                if not core_mask.any():
                    continue

                # Output (NAF) tile pixel bounds with halo, clamped to the full extent.
                out_lo_y = max(0, int(round(naf_h * ti / K)) - HALO_OUT)
                out_hi_y = min(naf_h, int(round(naf_h * (ti + 1) / K)) + HALO_OUT)
                out_lo_x = max(0, int(round(naf_w * tj / K)) - HALO_OUT)
                out_hi_x = min(naf_w, int(round(naf_w * (tj + 1) / K)) + HALO_OUT)
                tile_out_h = out_hi_y - out_lo_y
                tile_out_w = out_hi_x - out_lo_x

                # Image (guide) tile pixel bounds with halo, clamped to the full extent.
                img_lo_y = max(0, int(round(img_size * ti / K)) - halo_img)
                img_hi_y = min(img_size, int(round(img_size * (ti + 1) / K)) + halo_img)
                img_lo_x = max(0, int(round(img_size * tj / K)) - halo_img)
                img_hi_x = min(img_size, int(round(img_size * (tj + 1) / K)) + halo_img)
                image_tile = image_for_naf[:, :, img_lo_y:img_hi_y, img_lo_x:img_hi_x].contiguous()

                tile_hr = self.naf_model(
                    image_tile, lr_features_bchw, (tile_out_h, tile_out_w)
                )  # [B, embed_dim, tile_out_h, tile_out_w]

                if z_proj_hr is None:
                    z_proj_hr = torch.zeros(
                        B, R3, embed_dim,
                        device=tile_hr.device, dtype=tile_hr.dtype,
                    )

                # Tile's normalised bounds in the FULL feature map's [-1, 1] coord system.
                tile_v_lo_n = 2.0 * out_lo_y / naf_h - 1.0
                tile_v_hi_n = 2.0 * out_hi_y / naf_h - 1.0
                tile_u_lo_n = 2.0 * out_lo_x / naf_w - 1.0
                tile_u_hi_n = 2.0 * out_hi_x / naf_w - 1.0

                # Per-batch sampling — core_mask varies across batch so we can't share an index.
                for b in range(B):
                    mask_b = core_mask[b]
                    if not mask_b.any():
                        continue
                    pts = image_points_norm[b][mask_b]  # [N_core, 2] in global [-1, 1]
                    local_u = 2.0 * (pts[..., 0] - tile_u_lo_n) / (tile_u_hi_n - tile_u_lo_n) - 1.0
                    local_v = 2.0 * (pts[..., 1] - tile_v_lo_n) / (tile_v_hi_n - tile_v_lo_n) - 1.0
                    local_pts = torch.stack([local_u, local_v], dim=-1).unsqueeze(0)  # [1, N_core, 2]

                    feats = sample_features(tile_hr[b:b + 1], local_pts)  # [1, C, N_core]
                    z_proj_hr[b][mask_b] = feats.squeeze(0).transpose(0, 1).to(z_proj_hr.dtype)

                del tile_hr, image_tile

        if z_proj_hr is None:
            # No grid point projected on-image at all (degenerate). Return zeros — same shape
            # as the un-tiled call would produce.
            z_proj_hr = torch.zeros(
                B, R3, embed_dim,
                device=lr_features_bchw.device, dtype=lr_features_bchw.dtype,
            )
        return z_proj_hr

    @torch.no_grad()
    def visualize_projection(
        self,
        image: torch.Tensor,
        camera_angle_x: torch.Tensor,
        distance: torch.Tensor,
        mesh_scale: torch.Tensor,
        transform_matrix: Optional[torch.Tensor] = None,
        save_dir: Optional[str] = None,
        prefix: str = "proj_vis",
    ) -> List[Image.Image]:
        """
        Visualize the projected 3D grid points on the input image.
        
        This is a convenience method that delegates to ProjGrid.visualize_projection.
        
        Args:
            image: Input image tensor [B, C, H, W], in [0, 1] range (before ImageNet normalization)
            camera_angle_x: Camera FOV angle, shape [B]
            distance: Camera distance, shape [B]
            mesh_scale: Mesh scale factor, shape [B]
            transform_matrix: Optional camera transform matrix, shape [B, 4, 4]
            save_dir: Directory to save visualizations (optional)
            prefix: Prefix for saved files
            
        Returns:
            List of PIL Images with projected points overlaid
        """
        return self.proj_grid.visualize_projection(
            image=image,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
            transform_matrix=transform_matrix,
            save_dir=save_dir,
            prefix=prefix,
        )


# =============================================================================
# Multi-View DINOv3 Feature Extractor with Projection
# =============================================================================

class DinoV3ProjMultiViewFeatureExtractor(DinoV3ProjFeatureExtractor):
    """
    Multi-view DINOv3 feature extractor with view-aligned projection.

    Reuses DinoV3ProjFeatureExtractor's DINOv3 backbone + NAF, but:
    - accepts multi-view inputs: image [B, V, 3, H, W], camera_angle_x [B, V],
      distance [B, V], mesh_scale [B], transform_matrix [B, V, 4, 4] (c2w, index0=main).
    - computes per-view calc_mat = F @ inv(C_0) @ C_i and projects the SAME 3D grid
      through each view's relative pose (uses ProjGridMV).
    - fuses the V views according to `multiview_fusion`:
        * "average":  z_proj -> [B, R^3, C] (mean over V, matches single-view);
                      z_global -> [B, 1+num_reg, C] (mean over V).
        * "attention": z_proj -> [B, R^3, V, C] (keep V dim for per-voxel attn);
                      z_global -> [B, V*(1+num_reg), C] (concat over V as longer KV).

    proj_channels is inherited (embed_dim, or embed_dim*2 with NAF), so downstream
    per-block proj_linear / proj cross-attn weights are compatible with single-view.
    """

    def __init__(
        self,
        *args,
        multiview_fusion: str = "average",
        detail_tiles: bool = False,
        detail_tile_fraction: float = 0.60,
        detail_full_strength: float = 0.35,
        detail_tile_strength: float = 1.0,
        detail_feather: float = 0.20,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert multiview_fusion in ("average", "weighted_average", "attention"), \
            f"multiview_fusion must be average, weighted_average or attention, got {multiview_fusion}"
        self.multiview_fusion = multiview_fusion
        self.detail_tiles = detail_tiles
        self.detail_tile_fraction = float(detail_tile_fraction)
        self.detail_full_strength = float(detail_full_strength)
        self.detail_tile_strength = float(detail_tile_strength)
        self.detail_feather = float(detail_feather)
        # Replace proj_grid with the multi-view variant that accepts calc_mat.
        self.proj_grid = ProjGridMV(
            grid_resolution=self.grid_resolution,
            image_resolution=self.image_size,
        )

    def _project_single_view(self, z_patchtokens_spatial, image_for_naf,
                             camera_angle_x_v, distance_v, mesh_scale_v, calc_mat_v):
        """Project one view of DINOv3 features to the 3D grid. Returns [b, R^3, C]."""
        z_proj_lr = self.proj_grid(
            z_patchtokens_spatial, camera_angle_x_v, distance_v, mesh_scale_v, calc_mat_v
        )
        if not self.use_naf_upsample:
            return z_proj_lr  # [b, R^3, D]

        self._load_naf()
        lr_features_bchw = z_patchtokens_spatial.permute(0, 3, 1, 2)

        # Honour the same NAF tiling knob the single-view path uses (see
        # DinoV3ProjFeatureExtractor.naf_tile_factor). 1 = un-tiled.
        K = getattr(self, 'naf_tile_factor', 1) or 1
        if K <= 1:
            hr_features = self.naf_model(image_for_naf, lr_features_bchw, self.naf_target_size)
            z_proj_hr = self.proj_grid(
                hr_features, camera_angle_x_v, distance_v, mesh_scale_v, calc_mat_v, BHWC=False
            )
            del hr_features
        else:
            z_proj_hr = self._proj_naf_tiled(
                image_for_naf, lr_features_bchw,
                camera_angle_x_v, distance_v, mesh_scale_v, calc_mat_v,
                tile_factor=int(K),
            )
        return torch.cat([z_proj_lr, z_proj_hr], dim=-1)  # [b, R^3, 2D]
    def _project_single_view_lr(
        self,
        z_patchtokens_spatial,
        camera_angle_x_v,
        distance_v,
        mesh_scale_v,
        calc_mat_v,
    ):
        return self.proj_grid(
            z_patchtokens_spatial,
            camera_angle_x_v,
            distance_v,
            mesh_scale_v,
            calc_mat_v,
        )


    def _project_single_view_hr(
        self,
        z_patchtokens_spatial,
        image_for_naf_v,
        camera_angle_x_v,
        distance_v,
        mesh_scale_v,
        calc_mat_v,
    ):
        if not self.use_naf_upsample:
            return None

        self._load_naf()

        lr_features_bchw = z_patchtokens_spatial.permute(
            0, 3, 1, 2
        )

        K = getattr(self, "naf_tile_factor", 1) or 1

        if K <= 1:
            hr_features = self.naf_model(
                image_for_naf_v,
                lr_features_bchw,
                self.naf_target_size,
            )

            z_proj_hr = self.proj_grid(
                hr_features,
                camera_angle_x_v,
                distance_v,
                mesh_scale_v,
                calc_mat_v,
                BHWC=False,
            )

            del hr_features

        else:
            z_proj_hr = self._proj_naf_tiled(
                image_for_naf_v,
                lr_features_bchw,
                camera_angle_x_v,
                distance_v,
                mesh_scale_v,
                calc_mat_v,
                tile_factor=int(K),
            )

        return z_proj_hr

    @torch.no_grad()
    def _project_detail_tile_lr(
        self,
        image_v,
        camera_angle_x_v,
        distance_v,
        mesh_scale_v,
        calc_mat_v,
        crop_rect,
    ):
        """
        Crop one region from the original view, resize it back to image_size,
        run DINO, and project its LR DINO features into the SAME 3D grid.
        """

        B, C, H, W = image_v.shape

        x0, y0, x1, y1 = crop_rect

        ix0 = max(0, min(W - 1, int(round(x0 * W))))
        iy0 = max(0, min(H - 1, int(round(y0 * H))))
        ix1 = max(ix0 + 1, min(W, int(round(x1 * W))))
        iy1 = max(iy0 + 1, min(H, int(round(y1 * H))))

        tile = image_v[:, :, iy0:iy1, ix0:ix1].contiguous()

        tile = F.interpolate(
            tile,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        tile_norm = self.transform(tile)

        z = self.extract_features(tile_norm)

        num_reg = getattr(self.model.config, "num_register_tokens", 4)
        z_patchtokens = z[:, 1 + num_reg:]

        z_patchtokens_spatial = z_patchtokens.reshape(
            B,
            self.patch_number,
            self.patch_number,
            -1,
        )

        feat, weight = self.proj_grid.forward_crop(
            z_patchtokens_spatial,
            camera_angle_x_v,
            distance_v,
            mesh_scale_v,
            calc_mat_v,
            crop_rect,
            feather=self.detail_feather,
            BHWC=True,
        )

        del z, z_patchtokens, z_patchtokens_spatial
        del tile, tile_norm

        return feat, weight

    def forward(
        self,
        image: torch.Tensor,
        camera_angle_x: Optional[torch.Tensor] = None,
        distance: Optional[torch.Tensor] = None,
        mesh_scale: Optional[torch.Tensor] = None,
        transform_matrix: Optional[torch.Tensor] = None,
        visibility_weights: Optional[torch.Tensor] = None,
        detail_image: Optional[torch.Tensor] = None,
        geometry_confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: [B, V, 3, H, W]
            camera_angle_x: [B, V]
            distance: [B, V]
            mesh_scale: [B]
            transform_matrix: [B, V, 4, 4] (c2w, index 0 = main view)
        Returns:
            (z_global, z_proj) - shapes depend on multiview_fusion (see class doc).
        """
        assert image.ndim == 5, f"Expected image [B,V,3,H,W], got {tuple(image.shape)}"
        B, V = image.shape[:2]
        if camera_angle_x is None or distance is None or mesh_scale is None or transform_matrix is None:
            raise ValueError("camera_angle_x, distance, mesh_scale, transform_matrix must be provided")

        # Compute per-view calc_mat = F @ inv(C_0) @ C_i  -> [B, V, 4, 4]
        calc_mat = compute_relative_calc_mat(
            transform_matrix, distance, self.proj_grid.front_view_transform_matrix
        )

        # ------------------------------------------------------------------
        # TRUE PER-VIEW STREAMING
        # ------------------------------------------------------------------

        fuse_by_average = self.multiview_fusion in (
            "average",
            "weighted_average",
        )
        use_weights = self.multiview_fusion == "weighted_average"

        if use_weights and visibility_weights is None:
            raise ValueError(
                "multiview_fusion='weighted_average' "
                "requires visibility_weights"
            )

        calc_mat = compute_relative_calc_mat(
            transform_matrix,
            distance,
            self.proj_grid.front_view_transform_matrix,
        )

        z_proj_acc = None
        z_weight_acc = None

        z_global_acc = None
        per_view_proj = [] if not fuse_by_average else None
        per_view_global = [] if not fuse_by_average else None

        for v in range(V):

            # --------------------------------------------------------------
            # 1. ONE RGB VIEW ONLY
            # --------------------------------------------------------------

            image_v = image[:, v]
            detail_image_v = None

            if self.detail_tiles and detail_image is not None:
                detail_image_v = detail_image[:, v].to(
                    device=image_v.device,
                    dtype=image_v.dtype,
                    non_blocking=True,
                )

                tile_source_v = detail_image_v
            else:
                tile_source_v = image_v

            img_norm_v = self.transform(image_v)

            z = self.extract_features(img_norm_v)

            z_clstoken_v = z[:, 0:1]

            num_reg = getattr(
                self.model.config,
                "num_register_tokens",
                4,
            )

            z_regtokens_v = z[:, 1:1 + num_reg]
            z_patchtokens_v = z[:, 1 + num_reg:]

            z_global_v = torch.cat(
                [z_clstoken_v, z_regtokens_v],
                dim=1,
            )

            z_patchtokens_spatial_v = z_patchtokens_v.reshape(
                B,
                self.patch_number,
                self.patch_number,
                -1,
            )

            # Global cond is tiny.
            if fuse_by_average:
                if z_global_acc is None:
                    z_global_acc = z_global_v.clone()
                else:
                    z_global_acc.add_(z_global_v)
            else:
                per_view_global.append(z_global_v)

            # --------------------------------------------------------------
            # 2. FULL-VIEW LR DINO PROJECTION
            # --------------------------------------------------------------

            lr_view = self._project_single_view_lr(
                z_patchtokens_spatial_v,
                camera_angle_x[:, v],
                distance[:, v],
                mesh_scale,
                calc_mat[:, v],
            )

            # --------------------------------------------------------------
            # 3. DETAIL TILES — LR ONLY
            # --------------------------------------------------------------

            if self.detail_tiles:

                s = self.detail_tile_fraction

                positions = (
                    0.0,
                    (1.0 - s) * 0.5,
                    1.0 - s,
                )

                tile_rects = tuple(
                    (x, y, x + s, y + s)
                    for y in positions
                    for x in positions
                )

                # IMPORTANT:
                # reuse lr_view itself as the tile accumulator.
                # Do not create another R^3 x D lr_acc tensor.
                lr_view.mul_(self.detail_full_strength)

                lr_weight_acc = torch.full(
                    (*lr_view.shape[:-1], 1),
                    self.detail_full_strength,
                    device=lr_view.device,
                    dtype=lr_view.dtype,
                )

                for tile_i, crop_rect in enumerate(tile_rects):

                    tile_feat, tile_weight = (
                        self._project_detail_tile_lr(
                            tile_source_v,
                            camera_angle_x[:, v],
                            distance[:, v],
                            mesh_scale,
                            calc_mat[:, v],
                            crop_rect,
                        )
                    )

                    tile_weight = tile_weight.to(
                        device=tile_feat.device,
                        dtype=tile_feat.dtype,
                    )

                    tile_weight.mul_(
                        self.detail_tile_strength
                    )

                    # Reuse tile_feat too.
                    tile_feat.mul_(tile_weight)

                    lr_view.add_(tile_feat)
                    lr_weight_acc.add_(tile_weight)

                    del tile_feat, tile_weight

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                lr_view.div_(
                    lr_weight_acc.clamp_min_(1e-6)
                )

                del lr_weight_acc

            # --------------------------------------------------------------
            # 4. INITIALISE ONE FINAL ACCUMULATOR
            # --------------------------------------------------------------

            if fuse_by_average:

                if use_weights:
                    w = visibility_weights[:, v].to(
                        device=lr_view.device,
                        dtype=lr_view.dtype,
                    )

                    if geometry_confidence is not None:
                        # Native-resolution confidence is sampled directly
                        # with exactly the same camera projection as RGB.
                        #
                        # geometry_confidence[:, v]:
                        #     [B, 1, native_H, native_W]
                        #
                        # ProjGridMV uses normalized image coordinates, so
                        # the confidence map does NOT have to be resized to
                        # 512/1024 and does not have to be physically split
                        # into DINO tiles.

                        geom_v = geometry_confidence[:, v].to(
                            device=lr_view.device,
                            dtype=torch.float32,
                            non_blocking=True,
                        )

                        geom_w = self.proj_grid(
                            geom_v,
                            camera_angle_x[:, v],
                            distance[:, v],
                            mesh_scale,
                            calc_mat[:, v],
                            BHWC=False,
                        ).to(
                            dtype=lr_view.dtype
                        )

                        # B1.1 is deliberately conservative:
                        # geometry may only re-rank views within 0.50..1.00.
                        #
                        # This floor also prevents out-of-image grid_sample
                        # values (normally 0) from completely killing a view.
                        geom_w.clamp_(0.50, 1.0)

                        visible = w > 0

                        if visible.any():
                            vals = geom_w[visible]

                            print(
                                f"[Pixal3D geometry B1.2N] view={v}: "
                                f"projected mean="
                                f"{vals.float().mean().item():.3f}, "
                                f"min="
                                f"{vals.float().min().item():.3f}, "
                                f"max="
                                f"{vals.float().max().item():.3f}"
                            )

                        # Existing silhouette/depth visibility decides
                        # whether the view is valid at all.
                        # Geometry confidence decides HOW MUCH to trust it.
                        w.mul_(geom_w)

                        del geom_v, geom_w

                else:
                    w = None

                if z_proj_acc is None:

                    out_channels = self.proj_channels

                    z_proj_acc = torch.zeros(
                        B,
                        lr_view.shape[1],
                        out_channels,
                        device=lr_view.device,
                        dtype=lr_view.dtype,
                    )

                    if use_weights:
                        z_weight_acc = torch.zeros(
                            B,
                            lr_view.shape[1],
                            1,
                            device=lr_view.device,
                            dtype=lr_view.dtype,
                        )

                d = self.embed_dim

                # ----------------------------------------------------------
                # 5. ACCUMULATE LR DIRECTLY INTO FIRST HALF
                # ----------------------------------------------------------

                if use_weights:
                    lr_view.mul_(w)

                z_proj_acc[..., :d].add_(lr_view)

                del lr_view

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # ----------------------------------------------------------
                # 6. HR/NAF ONLY AFTER LR IS GONE
                # ----------------------------------------------------------

                if self.use_naf_upsample:

                    hr_view = self._project_single_view_hr(
                        z_patchtokens_spatial_v,
                        image_v,
                        camera_angle_x[:, v],
                        distance[:, v],
                        mesh_scale,
                        calc_mat[:, v],
                    )

                    if use_weights:
                        hr_view.mul_(w)

                    z_proj_acc[..., d:].add_(hr_view)

                    del hr_view

                # Weight is accumulated ONCE PER VIEW,
                # not once for LR and again for HR.
                if use_weights:
                    z_weight_acc.add_(w)
                    del w

            else:
                # Attention mode is not our current experiment.
                # Keep conventional full per-view tensor here.
                if self.use_naf_upsample:
                    hr_view = self._project_single_view_hr(
                        z_patchtokens_spatial_v,
                        image_v,
                        camera_angle_x[:, v],
                        distance[:, v],
                        mesh_scale,
                        calc_mat[:, v],
                    )

                    z_view = torch.cat(
                        [lr_view, hr_view],
                        dim=-1,
                    )

                    del lr_view, hr_view
                else:
                    z_view = lr_view

                per_view_proj.append(z_view)

            # --------------------------------------------------------------
            # 7. KILL EVERYTHING BELONGING TO THIS VIEW
            # --------------------------------------------------------------

            del z
            del z_patchtokens_v
            del z_patchtokens_spatial_v
            del z_clstoken_v
            del z_regtokens_v
            del img_norm_v
            del image_v
            if detail_image_v is not None:
                del detail_image_v

            del tile_source_v

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

                free, total = torch.cuda.mem_get_info()

                print(
                    f"[Pixal3D streaming] view={v} done: "
                    f"{free / 1024**3:.2f} GiB free / "
                    f"{total / 1024**3:.2f} GiB total"
                )

        # ------------------------------------------------------------------
        # FINAL NORMALISATION
        # ------------------------------------------------------------------

        if fuse_by_average:

            if use_weights:
                z_proj_acc.div_(
                    z_weight_acc.clamp_min_(1e-6)
                )

                del z_weight_acc

            else:
                z_proj_acc.div_(V)

            z_proj = z_proj_acc
            z_global = z_global_acc.div_(V)

        else:
            z_proj = torch.stack(
                per_view_proj,
                dim=2,
            )

            z_global = torch.cat(
                per_view_global,
                dim=1,
            )

        return z_global, z_proj
