# ComfyUI-OrbitTo3D

**Version:** `0.1.1` · **Repository:** https://github.com/mefisure-create/ComfyUI-OrbitTo3D
**Comfy Registry:** `orbit-to-3d` (publisher `@fadj`)

`ComfyUI-OrbitTo3D` is a **mesh-only, high-precision multiview reconstruction extension for ComfyUI**. It takes a turntable video, estimates the frame-to-yaw correspondence with VGGT-Omega, enhances selected views with cyclic RealBasicVSR, prepares Marigold V2/MoGe-3 geometric evidence, and reconstructs a Pixal3D mesh using consistent multiview observations.

The project is derived from, and coexists with, [`visualbruno/ComfyUI-Trellis2`](https://github.com/visualbruno/ComfyUI-Trellis2). Its compatible `TRELLIS2PIPELINE` / `MESHWITHVOXEL` sockets enable downstream Trellis2 decoding and mesh operations. The reviewed reference baseline is `b0e84c0f29857a444e8bcb5b9be2ea218e96efd0`; see [UPSTREAM.md](UPSTREAM.md) for provenance and the update contract.

## What's new in 0.1.1

- **Disk-backed multiview preprocessing:** an Akatz-driven per-view loop stores VSR inputs and synchronized image/mask, MoGe, Marigold normal/depth data on disk instead of retaining every view in RAM. Unified manifests support opening the reconstruction packet later, persistent snapshots, and cleanup of temporary sessions.
- **WTiVo mesh post-processing integration:** a pre-clean → external WTiVo remesh/repair → post-clean route accepts and returns `MESHWITHVOXEL`, uses transient resource-limited worker execution, and reports topology/resource diagnostics. **WTiVo itself and the privileged launcher are separate local installations**, not bundled binaries.
- **Checkpoints and diagnostics:** optional Sparse/Shape and mesh checkpoints, debug/repair helpers and RAM-profiling support.

The bundled [0.1.1 workflow](example_workflows/OrbitTo3D_HighPrecision_MultiView.json) is configured for a **3° yaw grid (120 views)** with 1536 ShapeHR and WTiVo 1536→2048. It captures the supplied working configuration, including **machine-specific absolute paths and external debug-export nodes**; adapt those before running on another host. The earlier **24-view / 15° clean-room regression** is retained as a reference, **not** presented as validation of the entire new 120-view + WTiVo graph.

## Documentation

- [Installation & dependencies](docs/INSTALLATION.md) — ComfyUI/Trellis2 environment, node packs, external binaries and workflow adaptation.
- [Features & node API](docs/FEATURES.md) — multiview conditioning and public nodes.
- [Disk cache and execution lifecycle](docs/CACHE.md) — session storage, manifests, persistence and cleanup.
- [WTiVo integration](docs/WTIVO.md) — external worker requirements, parameters, topology checks and limitations.
- [Regression & release checks](docs/REGRESSION.md), [changelog](CHANGELOG.md), [release checklist](RELEASE_CHECKLIST.md).
- [Upstream compatibility](UPSTREAM.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

## Scope and compatibility

Orbit remains **mesh-only**: it does not implement Pixal3D texture/PBR generation. WTiVo integration extends the post-reconstruction mesh stage; it does **not** replace or modify the Pixal3D sparse/shape denoising algorithms. External WTiVo output is **geometry-only** and loses voxel-specific metadata.

`Orbit3DMoGeInference` inherits the stock ComfyUI `MoGeInference.execute()` and changes only the `resolution_level` input bound. **No patch to `comfy_extras/nodes_moge.py` is required.**

See [LICENSE](LICENSE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [UPSTREAM.md](UPSTREAM.md) for licensing and upstream-derived code attribution.
