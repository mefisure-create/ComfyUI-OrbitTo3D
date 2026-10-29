# Release checklist

Before publishing a public release:

- [ ] Run `python tools/check_release.py` and resolve every failure.
- [ ] Run `python tools/check_upstream.py --upstream-repo <local-clone> --target origin/main --fetch --json-report upstream-review.json`.
- [ ] Review every upstream `CHANGED`, `REMOVED*`, `NEW_*` and every local `LOCAL_*` finding manually.
- [ ] If upstream changes are adopted or the reviewed Trellis2 baseline is advanced, follow `UPSTREAM.md`; never auto-merge/cherry-pick the upstream repository.
- [ ] Run the complete clean-room 24-view regression from `docs/REGRESSION.md` with ComfyUI started using `--cache-none`.
- [ ] Confirm the generated mesh is consistent with the accepted visual baseline and the complete log has no `ERROR`, `Traceback` or `ModuleNotFoundError`.
- [ ] Confirm the clean-room host uses stock ComfyUI MoGe code; `Orbit3DMoGeInference` must provide the extended `resolution_level` without editing `comfy_extras/nodes_moge.py`.
- [ ] Confirm platform-specific runtime dependencies match the target Python/Torch/CUDA stack. The tested reference uses `flash_attn` and `natten`; see `docs/INSTALLATION.md`.
- [ ] Verify the README remains a short project/front-page overview and detailed requirements/features live in `docs/` / `UPSTREAM.md`.
- [ ] Verify the README documentation links and the `docs/INSTALLATION.md` workflow dependency table still match the bundled example, including nodes nested inside subgraphs.
- [ ] Keep `LICENSE`, `THIRD_PARTY_NOTICES.md`, `UPSTREAM.md`, `ORBIT_BASELINE.md`, `docs/` and the pinned upstream metadata in the release.
- [ ] Replace `[tool.comfy].PublisherId = "local-orbit-to-3d"` with the real Comfy Registry publisher ID before Registry publication.
- [ ] Add the final repository/documentation/issue URLs to `pyproject.toml` once the public GitHub location is known.

The package intentionally remains mesh-only. Texture/PBR generation belongs to the original Trellis2 package or another downstream workflow.
