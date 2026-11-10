# Experimental SS64 -> Shape512 weights (global resolution probe)

This is **not** ROI repair and does **not** replace any production node.

Node: `Orbit3D - SS64 to Shape512 Weights [Experimental]`.

## Connect

1. Use existing `Orbit3D - Sparse Generator` with `sparse_structure_resolution=64`. Connect its `coords`, `sparse_structure_resolution`, and `pipeline` outputs to this experimental node. Alternatively use `Orbit3D - Load Sparse Stage [Experimental]` with a verified SS64 checkpoint.
2. Connect the **same** `PIXAL3D_MV_VIEWS` packet used by SS. Do not crop, regenerate camera configurations, or change view order.
3. Set `geometry_world_scale_mode`: `legacy`, `visibility_only`, `depth_only`, or `world_matched` (default). Only the experimental node changes its visibility/depth voxel-tolerance parameters. In `world_matched`, visibility uses `1.5*(R-1)/31` and depth tau uses `2.0*(R-1)/31`. R64 values: 3.048 and 4.065 voxels respectively. `legacy` preserves the original experiment exactly.
4. Set normal/depth modes and seeds to the baseline experiment values (recommended `normal+depth`, `Marigold` normals, `MoGe` depth when those maps are present). Keep `detail_tile_grid=3`, `detail_cond_mode=blend` to match the previously tested tiled baseline.
5. Route `shape_slat`, `decode_resolution`, and `pipeline` to your **mesh decoder**, not to Shape Cascade. For SS64, `decode_resolution=1024`. The node returns Shape512 weights applied on a 64^3 sparse grid; it does not run Shape1024 weights or a cascade.
6. If the number of SS64 tokens exceeds `max_num_tokens`, the node **fails without filtering**. Raise the guard only after considering VRAM pressure. Try a smaller number of views first (24 vs 120) if conditioning cost is prohibitive; that is a separate control and must be kept fixed during comparisons.

Supported SS grid values: 32 and 64 (32 is intended only as a baseline sanity check). `decode_resolution = SS grid * 16`.

## Assumptions, limitations, and interpretation

- Pixal3D SS decoder natively emits a 64^3 occupancy grid. The stock SS32 setting downsamples that occupancy. SS64 retains the higher-resolution decoder output; SS Flow is still generated from its original latent grid.
- Shape512 checkpoint weights are frozen and originally used for a 32^3 Shape-token grid. Sampling at 64^3 is *out of distribution*, even though the current sparse transformer implementation accepts the coordinates.
- The DINO/NAF image size is still 512. The experiment increases only the **3D projected conditioning grid and sparse tokens**, not image resolution or training capacity.
- The R64 Shape512 experiment is out of the training distribution. The scale-mode controls isolate two confounds: sparse-visibility tolerance and depth agreement tau. `legacy` keeps the old settings; the other modes match one or both tolerances to their R32 world-space extents. These settings do not rescale RoPE coordinates or retrain the weights.
- Compare four modes with **identical saved SS64**, images, seed, sampler, and decoding. Inspect global missing surfaces, the inner shell, fingers, and staff ornament. Neither presence nor absence of the inner shell by itself proves the source of degradation.
- Decoding at 1024 is a reasonable scale for 64^3 SLat coordinates, **not** a validated compatibility guarantee for the Shape512 weights.
- The feature projection at R64 has 8x as many grid positions as R32. More occupied tokens may multiply attention cost even more. Full-body SS64 can exceed 24-GB VRAM; monitor GPU memory.
- The output resolution is only meant for the decoder; the existing Shape Cascade rescaling path is not validated for this SLat. Do **not** attach the new output to it for this experiment.
- Keep the same SS source and its unchanged coordinates and maintain the same denoiser settings for a controlled 32-vs-64 comparison. Since Shape512 consumes different token sets for R32 and R64, fixed RNG seed does not imply identical per-token initial noise.

## Code isolation

New `orbit3d_ext/experimental_resolution_nodes.py` and three registration lines in `__init__.py` only. No production class, pipeline, sampler, projection math, checkpoint, or decoder is changed. Remove the registration and the new file to revert.
