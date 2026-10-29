# ComfyUI-OrbitTo3D

**Repository:** https://github.com/mefisure-create/ComfyUI-OrbitTo3D  
**Comfy Registry package ID:** `orbit-to-3d` (publisher `@fadj`)

`ComfyUI-OrbitTo3D` is a **mesh-only, high-precision multiview reconstruction extension for ComfyUI**, derived from and designed to coexist with [`visualbruno/ComfyUI-Trellis2`](https://github.com/visualbruno/ComfyUI-Trellis2).

Orbit is intended for a specific workflow: take an ordered turntable / multiview sequence, select camera-aligned views, preserve native geometry evidence, and reconstruct a Pixal3D mesh with stronger multiview consistency. It keeps Trellis2-compatible socket types so the original Trellis2 decode / mesh reconstruction / export nodes remain downstream-compatible.

This repository does **not** replace `ComfyUI-Trellis2`. The pinned Trellis2 reference baseline is commit `b0e84c0f29857a444e8bcb5b9be2ea218e96efd0`; retained upstream-derived code is tracked explicitly and checked for upstream changes.

## Current release status

The bundled 24-view / 15° yaw workflow has passed a clean-room end-to-end regression on ComfyUI `0.42.3` with Python 3.12, PyTorch `2.8.0+cu128`, CUDA 12.8 and an RTX 4090 24 GB. The validated path includes RealBasicVSR 2×, native-resolution MoGe/Marigold geometry, Sparse anchored soft occlusion, Shape512 normal+depth guidance, ShapeHR 1536 and final mesh reconstruction.

`Orbit3DMoGeInference` subclasses ComfyUI's native `MoGeInference` and changes only the `resolution_level` schema bound (up to 30); the original ComfyUI inference implementation is inherited unchanged. **No patch to `comfy_extras/nodes_moge.py` is required.**

## Documentation

- [Installation and requirements](docs/INSTALLATION.md) — host requirements, platform-specific dependencies, custom-node dependencies, model/checkpoint layout, tested stack and `--cache-none` guidance.
- [Nodes and reconstruction features](docs/FEATURES.md) — public node API, reconstruction behavior and the high-precision multiview mechanisms added by Orbit.
- [Regression and release validation](docs/REGRESSION.md) — accepted clean-room profile and release/regression procedure.
- [Trellis2 upstream synchronization](UPSTREAM.md) — pinned baseline, provenance contract, automated upstream detection and the manual review/port/rebaseline procedure.
- [Release checklist](RELEASE_CHECKLIST.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

## Scope

Orbit stops at reconstruction and is intentionally **mesh-only**. Texture/PBR generation stays in the original Trellis2 ecosystem. Print-specific cleanup such as internal-shell removal, floating-island cleanup, hole repair and remeshing belongs downstream.

## License and upstream attribution

Orbit retains the upstream MIT licensing/attribution for Trellis-derived code. See [LICENSE](LICENSE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [UPSTREAM.md](UPSTREAM.md).
