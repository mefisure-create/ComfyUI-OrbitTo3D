# Regression and release validation

## Accepted clean-room baseline

The current release candidate passed an end-to-end clean-room run with:

```text
ComfyUI               0.42.3
Python                 3.12.3
PyTorch                2.8.0+cu128
CUDA toolkit/runtime   12.8
GPU                     RTX 4090 24 GB
System RAM              64 GB
Views                   24 yaw views, 15° step, 0°..345°
Pitch                   disabled (pitch_step=360; elevations 0°)
RealBasicVSR            2×
Sparse Structure        resolution 32, anchored_soft_occlusion
Shape512                normal+depth; Marigold normals + MoGe depth
ShapeHR                 normal+depth; Marigold normals + MoGe depth
ShapeHR / mesh target   1536
ComfyUI cache mode      --cache-none
```

The final clean-room run used stock ComfyUI MoGe core code plus `Orbit3DMoGeInference`; no manual `comfy_extras/nodes_moge.py` patch was present. The generated mesh was visually accepted and the full prompt completed without `ERROR`, `Traceback` or `ModuleNotFoundError`.

## Release regression procedure

1. Start from a clean ComfyUI environment, not the development/production venv.
2. Install the external custom-node dependencies listed in [INSTALLATION.md](INSTALLATION.md).
3. Install a working `ComfyUI-Trellis2` runtime and matching platform-specific native extensions.
4. Install Orbit from the candidate source tree.
5. Confirm `python tools/check_release.py` passes.
6. Load `example_workflows/OrbitTo3D_HighPrecision_MultiView.json` and confirm there are no missing nodes/models.
7. Start ComfyUI with `--cache-none` and run the complete 24-view workflow through final mesh reconstruction/export.
8. Confirm the mesh is consistent with the accepted visual baseline.
9. Search the complete run log for `ERROR`, `Traceback` and `ModuleNotFoundError`.
10. If Trellis2 upstream-derived code changed, complete the procedure in [../UPSTREAM.md](../UPSTREAM.md) before advancing the pinned baseline.

## Release source checks

Run:

```bash
python tools/check_release.py
```

The checker validates Python compilation, the public Orbit node contract, the inherited MoGe execution contract, package/manifest integrity, dependency metadata alignment and required release documentation.

For upstream status, run separately:

```bash
python tools/check_upstream.py \
  --upstream-repo /path/to/ComfyUI-Trellis2 \
  --target origin/main \
  --fetch \
  --json-report upstream-review.json
```

A changed upstream contract is not an automatic failure of Orbit runtime; it means manual review is required before declaring the release current against the newer Trellis2 commit.
