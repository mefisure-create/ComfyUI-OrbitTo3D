# Trellis2 upstream synchronization contract

## Primary upstream

- Repository: `https://github.com/visualbruno/ComfyUI-Trellis2`
- Orbit reference baseline: `b0e84c0f29857a444e8bcb5b9be2ea218e96efd0`
- Base date: 2026-09-25
- Last reviewed upstream check for this release candidate: 2026-10-29
- Upstream `main` at that review: `b0e84c0f29857a444e8bcb5b9be2ea218e96efd0`

Orbit is derived from Trellis2 but intentionally removes large subsystems (single-view paths, texture/PBR generation, training/rendering utilities and other code not needed by the mesh-only multiview path). For that reason Orbit must **not** periodically merge the whole Trellis2 repository.

The maintained synchronization unit is the retained file/symbol contract recorded in `upstream_symbols.json`.

## Provenance categories

`upstream_symbols.json` classifies shipped code as:

- `upstream_verbatim_base` — retained from the pinned Trellis2 baseline without Orbit edits;
- `derived_modified` — upstream-derived code modified/pruned for Orbit;
- `orbit_original` — Orbit-specific code with no corresponding upstream implementation;
- `metadata` — packaging, documentation, workflows and tooling.

The first working Orbit tree modified four primary Trellis-derived files relative to the pinned baseline:

1. `nodes.py`
2. `trellis2/pipelines/trellis2_image_to_3d.py`
3. `trellis2/trainers/flow_matching/mixins/image_conditioned_proj.py`
4. `trellis2/utils/mv_camera.py`

Later mesh-only/release cleanup also changed package initializers, `trellis2/utils/camera.py`, and the retained mesh representation layer while removing unused subsystems.

## What is tracked

The release contract currently tracks:

- retained upstream files at file level, with explicit provenance mode;
- selected classes/functions/methods in actively modified upstream-derived files at AST level;
- hashes of every shipped Orbit source/metadata file through the release manifest.

This gives three different signals:

1. **upstream changed** — Trellis2 changed a retained file/symbol after Orbit's pinned baseline;
2. **local drift** — Orbit source changed without refreshing the reviewed manifest;
3. **release drift** — a shipped file no longer matches the release manifest.

A new upstream feature in a completely unretained subsystem is intentionally not treated as a required Orbit update. If an upstream refactor moves/removes a retained symbol or file, the existing tracked path reports that break and forces manual review.

## Automated detection

`.github/workflows/upstream-contract.yml` runs the upstream contract checker weekly and can also be started manually.

The workflow clones the current `visualbruno/ComfyUI-Trellis2` `main` branch and runs:

```bash
python tools/check_upstream.py \
  --upstream-repo .upstream/ComfyUI-Trellis2 \
  --target origin/main \
  --fail-on-change
```

Any tracked upstream change makes the workflow fail visibly. **The workflow never edits Orbit, merges commits, or advances the baseline.**

For a local review, use:

```bash
git -C /path/to/ComfyUI-Trellis2 fetch origin

python tools/check_upstream.py \
  --upstream-repo /path/to/ComfyUI-Trellis2 \
  --target origin/main \
  --json-report upstream-review.json
```

The checker also verifies that the target is a descendant of the pinned base, so an unrelated/rebased history cannot be accepted accidentally.

## Checker statuses

Important statuses include:

- `CHANGED` — tracked upstream file/symbol changed;
- `REMOVED` / `REMOVED_UPSTREAM_FILE` — retained upstream contract disappeared;
- `NEW_COLLISION` / `NEW_UPSTREAM_FILE` — a previously absent tracked contract now exists upstream;
- `LOCAL_DRIFT` — a tracked Orbit symbol changed relative to the reviewed manifest;
- `LOCAL_SYMBOL_MISSING` / `LOCAL_FILE_MISSING` — a tracked Orbit contract was deleted locally.

All of these require review. They are signals, not instructions to copy upstream wholesale.

## Manual update procedure

When Trellis2 changes:

1. Fetch the latest upstream and run `tools/check_upstream.py` with `--json-report`.
2. Review **old upstream -> new upstream** for every reported retained file/symbol. Do not begin with a noisy whole-repository `new upstream -> Orbit` merge/diff.
3. Decide for each change whether it is:
   - relevant and must be ported;
   - irrelevant because Orbit intentionally removed that path;
   - incompatible and requires an Orbit-side adaptation;
   - a structural refactor that requires changing the tracked file/symbol contract itself.
4. Port only the relevant change. Preserve Orbit's multiview, mesh-only and geometry-weighting semantics.
5. If a previously verbatim upstream file must diverge, reclassify it to `derived_modified` before advancing the baseline.
6. Run `python tools/check_release.py`.
7. Run the complete clean-room 24-view regression described in `docs/REGRESSION.md` and visually compare the output mesh with the accepted baseline.
8. Only after the review and regression pass, advance the Trellis2 baseline with `tools/update_upstream_manifest.py`.

Example rebaseline command:

```bash
python tools/update_upstream_manifest.py \
  --upstream-repo /path/to/ComfyUI-Trellis2 \
  --reviewed-report upstream-review.json \
  --new-base <exact-reviewed-upstream-commit> \
  --accept-review \
  --regression-passed

python tools/check_release.py
```

`update_upstream_manifest.py` deliberately refuses to run without both explicit confirmations. It also refuses to advance a file labelled `upstream_verbatim_base` unless the local file is still byte-for-byte identical to that file at the new upstream baseline.

The tool refreshes local file/symbol hashes and the pinned commit in `upstream_symbols.json`. Human-readable baseline metadata in this document must be updated in the same reviewed change.

## High-priority review areas

Even when surrounding APIs change, always inspect Trellis2 changes touching:

- model loading and Pixal3D / Pixal3D-MV flags;
- DINOv3 feature extraction, especially `model.layer` / `model.model.layer` compatibility;
- NAF loading/projection and its platform dependencies;
- sparse/shape samplers and decoder loading;
- `decode_latent` / mesh-decode interoperability with downstream Trellis2 nodes;
- sparse convolution / attention backend selection (`flex_gemm`, `flash_attn`, etc.);
- camera/multiview conventions and socket/API changes;
- native-extension packaging/wheels used by Trellis2.

## Policy

- No automatic merge/cherry-pick from Trellis2.
- No baseline advancement merely because upstream `main` moved.
- Every relevant upstream change is reviewed against Orbit's retained semantics.
- Every adopted/runtime-affecting change requires the full regression before rebaseline.
- The original Trellis2 package remains installed alongside Orbit; Orbit reuses its downstream decode/reconstruct/export nodes and compatible socket types.

## Licensing

Retained Trellis-derived code follows the upstream MIT license. Keep `LICENSE`, `THIRD_PARTY_NOTICES.md`, this file and the pinned upstream metadata in public releases.
