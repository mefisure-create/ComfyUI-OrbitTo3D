# Nodes and reconstruction features

Orbit keeps image-conditioned Pixal3D as the primary reconstruction model. External geometry is used mainly to decide **which observation to trust**, rather than to replace the generated shape. Most geometry mechanisms are therefore soft weighting signals; S4 bridge repair is the deliberately narrow topology exception.

# Public nodes

### Pixal3D reconstruction

- `Orbit3DLoadModel` — model loader; default `TencentARC/Pixal3D`.
- `Orbit3DMultiViewConfig` — builds the structured multiview packet from images, masks, geometry, FOV/framing data, standard `MOGE_GEOMETRY`, and yaw/pitch-derived azimuth/elevation strings.
- `Orbit3DSparseGenerator` — Pixal3D multiview sparse-structure stage.
- `Orbit3DShapeGenerator` — Shape512 generation.
- `Orbit3DShapeCascadeGenerator` — high-resolution shape cascade.

The Orbit generators require a compatible Pixal3D multiview pipeline and fail explicitly when an incompatible `TRELLIS2PIPELINE` is connected.

### View preparation

- `Orbit3DVGGTViewSelector` — selects turntable frames for requested yaw angles. Angle 0 is the first frame; rotation direction is supplied explicitly by `clockwise`.
- `Orbit3DRealBasicVSR` — cyclic RealBasicVSR processing while preserving the selected frame order.
- `Orbit3DAngleGrid` — generates synchronized `azimuths` / `elevations` strings from exact integer divisors of 360 (`yaw_step`, `pitch_step`) and returns `view_count`.

The verified production path is yaw-only (`pitch_step=360`, elevations all zero). Full 0–360 pitch orbits are not currently supported/tested because `mv_camera.py` interprets elevation as a spherical camera elevation with world-up handling rather than an independent rigid pitch rotation.

### MoGe utilities

- `Orbit3DMoGeInference` — inherits ComfyUI's native `MoGeInference.execute()` unchanged while extending only the `resolution_level` schema maximum to 30 for the high-resolution Orbit workflow.
- `Orbit3DMoGeGeometryListToBatch`
- `Orbit3DMoGeGeometryBatchToList`

Both use the standard `MOGE_GEOMETRY` socket type. `Orbit3DMultiViewConfig` also accepts standard `MOGE_GEOMETRY` directly.

## Current reconstruction behavior

The current regression baseline preserves the tested Pixal3D multiview behavior, including:

- Sparse Structure resolution 32 and anchored soft occlusion.
- Shape512 normal/depth conditioning with independently selectable Marigold or MoGe sources.
- ShapeHR / Cascade normal/depth conditioning with independently selectable Marigold or MoGe sources.
- S4 post-sparse bridge repair.
- high-resolution DINO detail tiles sourced from `native_images` for Shape512 and ShapeHR.
- MoGe camera-space normal convention using `-Nz` for front-facing confidence.

`IMAGE_COND` is not part of the public Orbit multiview dataflow; projected conditioning is rebuilt from `PIXAL3D_MV_VIEWS` inside the multiview stages.

## Orbit-specific high-precision multiview extensions

OrbitTo3D keeps the original Pixal3D multi-view denoisers and projection model from [`visualbruno/ComfyUI-Trellis2`](https://github.com/visualbruno/ComfyUI-Trellis2), pinned for this project to upstream commit `b0e84c0f29857a444e8bcb5b9be2ea218e96efd0`, and layers additional mechanisms on top of that baseline for **high-fidelity reconstruction from many ordered views**.

The upstream visualbruno implementation already provides DINOv3, NAF, camera-aware multi-view projection and the basic Pixal3D/TRELLIS2 runtime. The features below describe Orbit-specific behavior or locally modified inference behavior rather than claiming those upstream components as Orbit inventions.

### 1. Native-Resolution Tiled Conditioning

**Purpose:** preserve small visual details that disappear when native multiview images are reduced to the standard 512/1024 DINO inputs.

The normal Pixal3D conditioning path operates on stage-resolution images:

- Shape512: 512 px
- ShapeHR: 1024 px

Orbit additionally preserves every original view in `native_images` at its source resolution.

For Shape512 and ShapeHR, each view receives:

1. the normal full-frame DINO projection;
2. a 3×3 grid of overlapping crops from the original native-resolution image;
3. an independent DINO extraction for every crop;
4. projection of each crop back into the **same 3D grid and camera**;
5. feathered blending between the full-frame and crop-derived features.

Implemented by:

- `mv_camera.to_native_cond_tensor()`
- modified `mv_camera.build_views()`
- `ProjGridMV.forward_crop()`
- `DinoV3ProjMultiViewFeatureExtractor._project_detail_tile_lr()`
- the detail-tile branch in `DinoV3ProjMultiViewFeatureExtractor.forward()`

Current tested configuration:

| Stage | Tile fraction | Full-frame strength | Tile strength | Feather |
|---|---:|---:|---:|---:|
| Shape512 | 0.41 | 0.35 | 1.00 | 0.20 |
| ShapeHR | 0.40 | 0.35 | 1.00 | 0.20 |

A crop is not treated as a new camera. `forward_crop()` converts the original camera projection into crop-local coordinates, so cropped DINO features are written only into the 3D cells actually covered by that crop.

**Why it helps:** fingers, laces, thin weapon parts, facial elements, clothing seams and other small structures remain visible to DINO instead of being reduced to only a few pixels in the standard stage input.

**Important:** this is different from `naf_tile_factor`. NAF tiling is primarily a memory/streaming mechanism. Native-resolution detail tiles introduce **new high-resolution source information** into conditioning.

### 2. Visibility-Aware Weighted Fusion

**Purpose:** prevent every view from contributing equally to every 3D cell.

The upstream multi-view path can average projected features. Orbit adds and uses a `weighted_average` path that accumulates view-specific confidence:

`feature = sum(view_feature × view_weight) / sum(view_weight)`

Implemented in:

- `DinoV3ProjMultiViewFeatureExtractor.forward()`
- `Trellis2ImageTo3DPipeline.get_proj_cond_ss_mv()`
- `Trellis2ImageTo3DPipeline.get_proj_cond_shape_mv()`

Weights are calculated independently for each view and each 3D grid location.

The feature dimensionality remains compatible with the original Pixal3D MV denoisers.

**Why it helps:** contradictory observations no longer have identical authority. A view that actually sees a surface can contribute more strongly than a view looking through, behind or tangentially across it.

### 3. Sparse-Token Visibility Awareness

**Purpose:** determine which cameras can actually see an occupied 3D token during Shape generation.

After Sparse Structure has produced occupied coordinates, Orbit projects each occupied token into every camera and builds a temporary per-view Z-buffer.

A token is considered visible when its projected depth lies within a small front-surface band of the nearest projected token in that view.

Implemented by:

`Trellis2ImageTo3DPipeline._build_sparse_mv_visibility()`

The current visibility tolerance is expressed in **voxel units**, rather than arbitrary camera-depth units.

If a token cannot be confidently classified as visible from any camera, Orbit deliberately falls back to the all-view behavior instead of starving that token of conditioning.

**Why it helps:** Shape conditioning is driven primarily by views that geometrically observe the surface, reducing feature contamination from the opposite side of the object.

### 4. Anchored Soft Occlusion

**Purpose:** improve Sparse Structure fusion before occupied tokens exist, without allowing external geometry to dictate occupancy.

Sparse Structure is the difficult stage for visibility reasoning because there is not yet a generated surface to Z-buffer.

Orbit can therefore use MoGe geometry only as **soft evidence for view reliability**.

Two modes are available:

- `soft_occlusion`
- `anchored_soft_occlusion`

The tested/default mode is `anchored_soft_occlusion`.

Implemented by:

`Trellis2ImageTo3DPipeline._build_ss_soft_occlusion_weights()`

For each candidate SS grid cell Orbit:

1. projects the candidate into every camera;
2. samples the calibrated MoGe visible surface;
3. measures candidate depth relative to that surface;
4. searches for at least one view in which the candidate lies close enough to the observed surface to act as an **anchor**;
5. only then downweights other views where the candidate lies clearly behind the visible surface.

Current regression parameters:

| Parameter | Value |
|---|---:|
| Soft floor | 0.35 |
| Occlusion margin | 1.5 voxels |
| Gaussian tau | 1.5 voxels |
| Anchor band | 1.5 voxels |
| Anchored floor | 0.50 |

The weighting is deliberately soft. An occluded view is reduced, not removed.

Most importantly, this mechanism:

- does **not** create voxels;
- does **not** delete voxels;
- does **not** move voxels;
- does **not** replace Pixal3D occupancy prediction.

It changes only the relative influence of projected image conditioning.

**Why it helps:** Sparse Structure becomes less likely to mix front-side and back-side visual evidence while preserving Pixal3D as the authority on actual occupancy.

### 5. Surface-Normal Confidence

**Purpose:** prefer views that observe a surface more frontally.

Orbit can derive a conservative confidence field from either Marigold V2 normals or MoGe normals.

For Marigold camera-space normals:

`confidence = base + bonus × alpha × max(0, Nz)`

For MoGe, the sign is corrected for its +Z-forward camera convention:

`confidence = base + bonus × mask × max(0, -Nz)`

Implemented by:

- `mv_camera.attach_geometry_confidence()`
- `mv_camera.attach_moge_shape512_geometry()`
- geometry-weight sampling in `DinoV3ProjMultiViewFeatureExtractor.forward()`

The confidence has a non-zero floor and therefore never acts as a hard visibility mask.

Geometry confidence is sampled through the **same ProjGrid camera projection** as DINO rather than being resized to the DINO input resolution.

**Why it helps:** frontal observations receive a moderate preference over grazing-angle observations, where silhouettes and fine surface details are less reliable.

### 6. Affine Depth Agreement

**Purpose:** use depth as a consistency signal without assuming that Marigold or MoGe depth is already in the exact Pixal3D world-space scale.

For each view, Orbit compares the selected native depth source against the already generated SS/S4 surface.

It fits:

`z_pixal ≈ a × depth_source + b`

using currently visible occupied tokens.

The first fit is followed by a MAD-based robust outlier rejection and a second affine fit when enough samples remain.

Depth residual is then converted into voxel units and mapped to a soft Gaussian confidence:

`weight = floor + (1 - floor) × exp(-0.5 × (residual / tau)²)`

Implemented by:

`Trellis2ImageTo3DPipeline._shape512_marigold_diagnostics()`

Despite its historical name, this routine supports both Marigold and MoGe depth and can actively provide Shape weighting.

Current depth floor:

`0.70`

ShapeHR automatically scales its voxel-space tau with grid resolution so that the effective tolerance stays approximately constant in **world space** rather than becoming artificially stricter at higher resolutions.

**Why it helps:** views whose predicted depth agrees with the already reconstructed surface receive more influence, while relative-depth scale differences do not immediately invalidate the source.

### 7. Hybrid Geometry Guidance

**Purpose:** allow the strongest geometry source to be selected independently for normals and depth.

Shape512 and ShapeHR support four modes:

- `off`
- `normal`
- `depth`
- `normal+depth`

For each stage, the normal and depth sources can independently be:

- `Marigold`
- `MoGe`

Implemented by:

- `Trellis2ImageTo3DPipeline._shape512_selected_sources()`
- `Trellis2ImageTo3DPipeline._shape_geometry_stage_config()`
- Orbit Shape / Cascade public wrappers

For example, the tested regression configuration uses:

`Normal = Marigold`

`Depth = MoGe`

This is not a geometry replacement pipeline. Both signals only **re-rank projected image conditioning**.

**Why it helps:** different geometry estimators have different strengths. Orbit can use the sharper or more stable normal source while simultaneously using the more geometrically useful depth gauge.

### 8. ShapeHR Geometry Carry-Through

**Purpose:** prevent geometry awareness from stopping after Shape512.

Orbit extends the same optional normal/depth weighting model to the high-resolution Cascade stage.

Implemented through:

- `geometry_stage="shape_hr"`
- `_shape_geometry_stage_config()`
- `get_proj_cond_shape_mv()`

The high-resolution stage reuses the original native geometry maps instead of deriving geometry again from an intermediate render.

**Why it helps:** high-resolution shape refinement remains constrained by the same observed multiview evidence that guided the lower-resolution shape, reducing the opportunity for fine geometry to drift away from the references.

### 9. S4 Conservative Bridge Repair

**Purpose:** recover small topology connections that Sparse Structure nearly produced but missed.

S4 runs **after** the normal Pixal3D Sparse Structure prediction.

Implemented by:

`Trellis2ImageTo3DPipeline._repair_ss_with_moge()`

It performs one conservative repair iteration.

A missing cell is considered only when local topology indicates that adding that one cell could connect separate nearby pieces. MoGe evidence is then evaluated independently from multiple views.

A candidate must satisfy both:

- minimum supporting-view count;
- minimum accumulated geometric evidence.

Current defaults include:

- support threshold: 2.0 voxels;
- evidence tau: 1.25 voxels;
- minimum support views: 2;
- minimum evidence sum: 0.90.

S4 is intentionally asymmetric:

- it may **add** a strongly supported missing bridge;
- it may not delete existing Pixal3D voxels;
- it may not move existing voxels;
- it performs only one local repair pass.

**Why it helps:** thin connections such as weapon continuity, limbs, straps and other narrow bridges can survive Sparse Structure without letting external geometry rewrite the generated object.

### 10. Per-View Streaming Conditioning

**Purpose:** make high-view-count reconstruction feasible without changing the Pixal3D denoiser architecture.

Orbit processes projected multiview conditioning incrementally by view and releases temporary detail-tile / projection intermediates before advancing to the next view.

Implemented in:

`DinoV3ProjMultiViewFeatureExtractor.forward()`

The retained workflow performs per-view DINO projection, native-resolution detail-tile projection, weighted accumulation and early release of intermediates before continuing through the remaining views.

The final fused tensor keeps the layout expected by Pixal3D.

**Why it helps:** peak memory scales less aggressively with view count, making 24-view conditioning and native-resolution detail extraction practical on workstation GPUs. This is primarily a memory mechanism, but it directly enables using **more views and more source detail**.

### 11. Native Geometry Projection

**Purpose:** avoid degrading geometry maps just to match DINO input resolution.

Marigold and MoGe geometry remains at native spatial resolution.

Normal confidence, depth and masks are sampled by projecting the actual 3D grid into those maps using the same camera geometry as image conditioning.

Geometry therefore does not need to be:

- resized to 512;
- resized to 1024;
- tiled in the same way as DINO;
- regenerated from an intermediate clay render.

**Why it helps:** geometry weighting preserves the spatial precision of the original estimator instead of introducing another resize/re-depth step.

### 12. Camera and Framing Consistency

**Purpose:** ensure that the camera model used by 3D projection matches the way the supplied images are actually framed.

`Orbit3DMultiViewConfig` explicitly validates:

- image count;
- mask count;
- azimuth count;
- elevation count;
- view ordering.

It also derives or validates camera distance against image framing.

Available framing strategies are:

- `auto` — measure object fill from alpha and fit camera distance;
- `pixal3d_rig` — use the Pixal3D training-rig margin;
- `fill_frame` — assume the object fills the frame;
- explicit distance — use the supplied value unchanged.

A warning is emitted when the selected distance disagrees significantly with the measured image framing.

**Why it helps:** camera-distance errors shift every 3D-to-2D lookup. Even good DINO features become incorrect conditioning if a voxel samples the wrong image pixel.

### 13. View-Order Integrity

**Purpose:** prevent silent correspondence errors between images and camera angles.

Orbit treats the view batch as an ordered dataset:

`image[i] ↔ azimuth[i] ↔ elevation[i] ↔ geometry[i]`

The MultiView Config fails explicitly if counts disagree instead of truncating or silently reordering inputs.

The MoGe adapters apply the same rule to geometry packet fields.

**Why it helps:** a mathematically correct projection becomes useless if a 90° image is accidentally paired with a 105° camera or with another view's depth map.

### 14. VGGT Orbit-Aware Frame Selection

**Purpose:** obtain reproducible camera-aligned views from a turntable video rather than selecting frames by uniform frame number.

`Orbit3DVGGTViewSelector` uses VGGT camera predictions to reconstruct the camera-center trajectory.

The selector:

- defines 0° strictly as the first source frame;
- never invents a semantic "front" direction;
- accepts rotation direction explicitly;
- fits the predicted camera centers to an orbit plane and circle;
- unwraps angular progression;
- applies monotonic/isotonic stabilization;
- performs a coarse pass followed by local refinement windows;
- selects the nearest recovered frame for every requested yaw;
- derives median FOV from VGGT intrinsics;
- validates orbit span, planarity, monotonicity and radius stability.

**Why it helps:** generated turntables frequently have non-uniform angular speed. Selecting every N-th frame can therefore assign incorrect camera angles even when the video looks visually smooth.

VGGT selection ties each chosen image to a measured camera trajectory instead.

### 15. Cyclic View Super-Resolution

**Purpose:** upscale selected turntable views without breaking their ordering or treating the first/last view as unrelated endpoints.

`Orbit3DRealBasicVSR` processes the exact source-frame sequence received from the view selector and preserves the supplied order.

Its turntable-aware processing treats the sequence as cyclic rather than as an ordinary open-ended video clip.

**Why it helps:** the 0° and near-360° views represent neighboring camera positions. Preserving cyclic context reduces endpoint behavior while providing higher-resolution inputs for Marigold, MoGe and native-detail conditioning.

### 16. Native MoGe Packet Integrity

**Purpose:** use ComfyUI's standard `MOGE_GEOMETRY` type without introducing an incompatible Orbit-only geometry socket.

Orbit provides:

- `Orbit3DMoGeGeometryListToBatch`
- `Orbit3DMoGeGeometryBatchToList`

The adapters validate field dimensions, view counts and metadata consistency.

Tensor fields are moved to CPU before concatenation.

Recognized geometry remains associated by view:

- points;
- depth;
- normals;
- mask;
- intrinsics;
- image.

Unknown metadata is retained only when all list items agree.

**Why it helps:** mapped per-view MoGe inference can be combined into one multiview geometry packet without silently misaligning fields or retaining unnecessary GPU copies.

### 17. Pre-Pixal Memory Barrier

**Purpose:** release preprocessing models before the expensive Pixal3D stages begin.

`Orbit3DMultiViewConfig` is a natural synchronization point: RGB preparation, VSR, Marigold and MoGe have all completed before it can execute.

With `cleanup_before_pixal3d` enabled, Orbit asks the ComfyUI model manager to release preprocessing GPU allocations before loading Pixal3D.

This is separate from the recommended global `--cache-none` mode, which addresses retained **system RAM** outputs between executions.

**Why it helps:** Pixal3D does not have to compete with preprocessing models for GPU memory during Sparse and Shape generation.

### 18. Geometry Diagnostics and Regression Observability

**Purpose:** make geometry-assisted reconstruction auditable instead of silently changing the result.

Orbit contains dedicated diagnostic routines for:

- MoGe cross-view alignment;
- MoGe-to-SS agreement;
- per-view depth gauge calibration;
- normal/depth weight distributions;
- visibility statistics;
- S4 candidate and acceptance reports;
- before/after sparse-coordinate signatures.

Relevant methods include:

- `_prepare_ss_moge_diag_state()`
- `_diagnose_moge_views_alignment()`
- `_diagnose_ss_moge_against_coords()`
- `_moge_views_fit_similarity()`
- `_ss_coords_signature()`
- diagnostic portions of `_shape512_marigold_diagnostics()`

Several diagnostics export JSON reports and debug point clouds.

These routines are intentionally separate from occupancy authority: diagnostics can inspect geometry disagreement without automatically forcing Pixal3D to follow the auxiliary estimator.

**Why it helps:** changes to multiview weighting can be regression-tested quantitatively instead of relying only on visual comparison of final meshes.

### Design principle

Orbit's geometry extensions follow one conservative rule:

> **Image-conditioned Pixal3D remains the primary reconstruction model; external geometry is used mainly to decide which observation should be trusted, not to replace the generated shape.**

This is why most Orbit geometry mechanisms operate as **soft weights** rather than hard masks.

The main exception is S4, which is deliberately restricted to one-step, multi-view-supported topology bridge repair after the normal Sparse Structure has already been generated.

This approach is intended to improve high-precision multiview consistency while preserving the learned Pixal3D prior and avoiding direct reconstruction from noisy monocular geometry.
