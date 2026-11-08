# Release checklist

Before publishing a public release:

- [ ] Run `python tools/check_release.py` and resolve every failure.
- [ ] Run `python tools/check_upstream.py --upstream-repo <local-clone> --target origin/main --fetch --json-report upstream-review.json`.
- [ ] Review every upstream `CHANGED`, `REMOVED*`, `NEW_*` and every local `LOCAL_*` finding manually.
- [ ] If upstream changes are adopted or the reviewed Trellis2 baseline is advanced, follow `UPSTREAM.md`; never auto-merge/cherry-pick the upstream repository.
- [ ] Run the previous 24-view regression and a separate complete 0.1.1 120-view cache/WTiVo regression from `docs/REGRESSION.md` with ComfyUI started using `--cache-none`.
- [ ] Confirm the generated mesh is consistent with the accepted visual baseline and the complete log has no `ERROR`, `Traceback` or `ModuleNotFoundError`.
- [ ] Confirm the clean-room host uses stock ComfyUI MoGe code; `Orbit3DMoGeInference` must provide the extended `resolution_level` without editing `comfy_extras/nodes_moge.py`.
- [ ] Confirm platform-specific runtime dependencies match the target Python/Torch/CUDA stack. The tested reference uses `flash_attn` and `natten`; see `docs/INSTALLATION.md`.
- [ ] Verify the README remains a short project/front-page overview and detailed requirements/features live in `docs/` / `UPSTREAM.md`.
- [ ] Verify the README documentation links and the `docs/INSTALLATION.md` workflow dependency table still match the bundled example, including nodes nested inside subgraphs.
- [ ] Keep `LICENSE`, `THIRD_PARTY_NOTICES.md`, `UPSTREAM.md`, `ORBIT_BASELINE.md`, `docs/` and the pinned upstream metadata in the release.
- [ ] Confirm `[tool.comfy].PublisherId` still matches the Registry publisher (`fadj`) before publishing.
- [ ] Confirm the `Repository`, `Documentation` and `Bug Tracker` URLs in `pyproject.toml` still point to `https://github.com/mefisure-create/ComfyUI-OrbitTo3D`.

## 0.1.1 gates

- [ ] Validate Akatz loop completion for every requested view, cache manifest order and reopen from a persistent snapshot.
- [ ] Confirm cleanup only runs after final mesh/export use; distinguish temporary-session deletion from persistent retention.
- [ ] Validate WTiVo on a target Linux host with the *separate* WTiVo solver, cleaner binary and audited systemd launcher in place. Record runtime/peak RAM/VRAM and topology audit.
- [ ] Resolve the unbundled `Pixal3DDebugExport*` nodes used by the working graph or provide a tested portable replacement.
- [ ] Audit `example_workflows/OrbitTo3D_HighPrecision_MultiView.json` for host-specific absolute paths before publication.
- [ ] Ensure no `.git`, `.pyc`, `node.zip`, model files, prebuilt local tools, cache session artifacts or backup sources are accidentally included.
- [ ] Verify the source branch/tag and Registry version are both `0.1.1`; only create a GitHub release once end-to-end tests have passed.

The package intentionally remains mesh-only for reconstruction. The new WTiVo stage is optional **geometry-only** downstream remeshing; texture/PBR generation stays outside Orbit.
