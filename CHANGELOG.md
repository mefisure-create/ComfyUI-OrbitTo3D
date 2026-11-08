# Changelog

## 0.1.1 — 2026-11-08

### Added
- Disk-backed preprocessing and Unified per-view cache: session creation, VSR mmap spill/read, Akatz static loop context, synchronized per-view spill, manifest finalization, cache opening/loading, and cleanup; optional persistent-session snapshot and reconstruction from saved data.
- ComfyUI nodes for WTiVo integration: streamed mesh PLY serialization, external component cleaner (before/after), transient resource-limited WTiVo worker, reporting, cancellation, and geometry-only PLY import.
- Mesh and intermediate-stage checkpoints, optional experimental topology repair/shape diagnostics and RAM profiling.
- Updated example workflow with 120-view / 3° angular sampling, cache lifecycle, and WTiVo cleanup/remeshing chain.
- Documentation for cache persistence, external runtime installation, limitations, and regression procedures.

### Changed
- Public node registry now includes production cache, diagnostics, checkpoint and WTiVo integration nodes.
- Release integrity checker handles composed node registries and validates new workflow public Orbit node IDs.
- Updated release manifest/hashes without changing the pinned Trellis2 upstream reference baseline.

### Known limitations
- WTiVo solver, systemd launcher and `orbit_raw_component_tool` are not included; see `docs/WTIVO.md`.
- The supplied workflow contains host-specific paths and `Pixal3DDebugExport*` nodes that are not exported by Orbit itself.
- A clean-room end-to-end regression of the entire 120-view cache + WTiVo graph is not claimed by this source-only release check.

## 0.1.0 — 2026-10-30

- Initial mesh-only Pixal3D multiview release with VGGT turntable selection, cyclic RealBasicVSR, native MoGe/Marigold conditioning, and pinned Trellis2 upstream contract.
- Accepted 24-view / 15° yaw clean-room reconstruction baseline.
