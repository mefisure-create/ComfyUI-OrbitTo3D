# Disk-backed preprocessing and cache lifecycle (0.1.1)

The production pipeline processes many selected turntable views without holding all Marigold, MoGe and image intermediates in Python RAM simultaneously. **Caching changes data ownership/lifetime, not the reconstruction equations.** It does not by itself guarantee a particular peak RAM footprint; GPU model loading, ComfyUI's execution cache and selected input resolution still matter.

## Production dataflow

1. `Orbit3DAngleGrid` creates requested azimuth/elevation values. `Orbit3DVGGTViewSelector` selects the corresponding video frames; `Orbit3DRealBasicVSR` enhances them in their original selected order.
2. `Orbit3DCacheSession` creates the run-specific cache directory. `Orbit3DVSRBatchCache` stores the VSR batch in a memory-mapped backing file. `Orbit3DVSRViewAtIndex` retrieves one view per iteration.
3. `WhileLoopOpen | akatz-loops` / `WhileLoopClose | akatz-loops` drive the per-view Marigold V2 and MoGe-3 processing. `Orbit3DLoopContextPack` / `Orbit3DLoopContextUnpack` carry loop-invariant resources; the loop must not re-run expensive model-loading or VSR branches on every iteration.
4. `Orbit3DUnifiedViewSpill` writes a complete synchronized packet (image, mask, normal/depth maps, MoGe geometry) for each index. `Orbit3DUnifiedCacheFinalize` verifies and finalizes the view-order manifest, then releases CPU objects.
5. `Orbit3DMultiViewCacheOpen` supplies a file-backed handle to `Orbit3DMultiViewConfig`, which builds `PIXAL3D_MV_VIEWS`. This is consumed by the existing Pixal3D Sparse/Shape512/ShapeHR stages.
6. `Orbit3DCleanupCacheSession` is ordered after the workflow's final output. Set `delete_cache=true` to remove temporary session data when it is no longer required.

Relevant implementation files: `orbit3d_ext/production_cache_nodes.py`, `unified_cache_nodes.py`, `generic_cache_nodes.py`, `image_mmap_for_nodes.py`, `akatz_static_context_nodes.py`.

## Temporary vs persistent

- **Temporary mode:** intermediate files are stored under the configured cache root (ComfyUI temp by default) and can be deleted after the output is saved. Do not delete them while `PIXAL3D_MV_VIEWS` still references them.
- **Persistent mode:** the Unified cache finalization path takes a separate snapshot in `persistent_path/<session-id>` with rebased absolute paths. Cleanup of the temporary session does not remove that persistent snapshot. Allow enough free disk space for **both** trees during the copy.
- **Reopen:** `Orbit3DUnifiedCacheManifestLoad` and the cache-open nodes can reuse previously saved compatible manifests. Use the directory/manifest path the node actually expects; a cache directory is not interchangeable with a manifest file path.

The supplied 0.1.1 workflow selects **persistent** storage, with a `persistent_path` pointing to the original developer's host. Replace `/home/ai/orbit3d-cache` with a real writable location or choose temporary storage. Disk usage grows with view count and image resolution, and may amount to many gigabytes.

## Operational cautions

- Keep a stable view index/order between VGGT selection, VSR, geometry prediction, and the cache manifest. Mixing sessions, changing the view grid or manually editing per-view files can invalidate reconstruction metadata.
- Prefer ComfyUI `--cache-none` for large multiview jobs so ComfyUI does not retain large completed upstream results alongside the disk cache. The cache directory and ComfyUI's execution-result cache are **different mechanisms**.
- The final cleanup node is explicitly scheduled through the output dependency. If a prompt aborts or a node errors before cleanup, temporary files may remain. Inspect the session metadata and remove only abandoned run directories after verifying no workflow is using them.
- Persistent sessions are not automatically erased by temporary cleanup. Manage their retention separately.
- RAM limits and throughput depend on actual view resolution, batch shape and underlying models; validating a new 72/120-view configuration requires monitoring RAM/VRAM and successful postprocessing.
