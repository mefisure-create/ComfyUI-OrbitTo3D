# SS64 completion (resolution_test)

## Nodes

- `Orbit3D - Sparse64 Generator T5 [Experimental]`: the original opt-in T5-only generator, with the same node ID and output contract as the local experiment.
- `Orbit3D - Sparse64 Generator T5 + SurfaceAll`: final SS64 completion candidate. Uses the existing Pixal3D MV conditioning and sparse flow sampler. Native SS64 only, one sample; use the SS64 flow checkpoint.

The default preset is C25_NoLock: seed 1234, Heun, 25 steps, guidance 7.5, rescale 0.70, rescale_t 5, interval [0.60,1.0], dino_lock 0, substeps 1, foundation cap 0.92. Visibility settings are inherited from Orbit3D Sparse Generator (anchored soft occlusion defaults). Keep geometry and reference views identical to your comparison run.

BASE is raw decoder logits > 0. T5 uses logits > -5 by default. SurfaceAll evaluates empty cells against the fixed T5 occupancy, adds cells having at least two opposite occupied face-neighbor pairs, and never repeats the operation. There is no additional logit threshold for selection; no flood fill, dilation, shell removal, S4 or resampling. The historical -41 partition is retained solely for exact coordinate-row ordering compatibility with the evaluated SurfaceAll checkpoints: T5, then Surface41 rows, then remaining rows. It does not affect membership.

## Output sockets

The first five outputs retain the T5 experiment's positions: coords (final for SurfaceAll), sparse_structure_resolution (64), pipeline, base_coords, added_tokens (final minus BASE). SurfaceAll additionally exposes t5_coords and surface_added_tokens (final minus T5). The optional checkpoint_path must be a new or empty directory; an empty string disables saving. BASE, T5 and T5_SurfaceAll use the existing sparse stage checkpoint format. The flow dump includes original logits and occupancy; report.json distinguishes T5 additions from SurfaceAll additions. No files are overwritten.

Connect the final coords, resolution, pipeline and the same MV views to `Orbit3D - SS64 to Shape512 Weights [Experimental]`. Shape512 weights remain unchanged; its decode_resolution for SS64 is 1024, not 512. No Shape or Cascade implementation is changed by this update.

## Observed validation

Six GT test objects were visually inspected by the project author after T5 + SurfaceAll. Calliope was also tested, but its side defect is confounded by missing foreground in some of the 120 masks. These are visual observations, not automated topology/accuracy metrics.

| Model | BASE | T5 | SurfaceAll | Residual observations |
|---|---:|---:|---:|---|
| Item_01 | 6672 | 6681 | 6711 | Armpits |
| Item_02 | 11184 | 11210 | 11375 | Left chest/armpit |
| Item_03 | 5171 | 5179 | 5199 | Left hand |
| Item_04 | 4923 | 4935 | 4965 | Breastplate/abdomen gap |
| Item_05 | 3199 | 3204 | 3235 | No visible new defects reported |
| Item_06 | 3998 | 4003 | 4067 | One of four dress gaps remains, substantially smaller |
| Calliope | 5199 | 5210 | 5252 | Mask-confounded side; residual dress/chest defects |

Surface41 improved several models, but a fixed additional threshold is unnecessary for the tested one-pass SurfaceAll candidate. The largest adjacent logit gap was not a reliable selection rule. R30 was too restrictive on Item_02. SurfaceAll does not guarantee watertightness or absence of artifacts on unseen inputs. Existing GT64-to-Shape512 results also show residual holes. Next investigation: Shape512 subdivision/mesh connectivity versus SS32, followed by unchanged Shape Cascade tests; no presumption that HR Cascade will repair every hole.

## Regression checks

Run from repo root with the ComfyUI Python environment:

```sh
python -m unittest discover -s tests -p 'test_ss64_completion.py'
```

For integration, compare exported BASE/T5/SurfaceAll coordinate tensors with previously saved captures for the same latent and threshold (exact rows, not just count). Run Shape512 with the same fixed seed and settings; unchanged ordering reduces confounds but does not guarantee identical noise for different token counts.

The standard generator remains on the original path when no scoped capture is active.
