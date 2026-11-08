# Orbit3D release baseline

Stage: historical v0.1.0 clean-room baseline (retained for regression)
Date: 2026-10-29

The runtime reconstruction baseline remains the verified Stage 6B7 mesh-only Pixal3D multiview implementation. Release hardening has now also passed a complete clean-room regression.

Validated release-candidate additions include:

- `Orbit3DMoGeInference`, which inherits ComfyUI's native `MoGeInference.execute()` unchanged and only extends the schema validation range for `resolution_level` to 30;
- removal of the previous hidden requirement to patch `comfy_extras/nodes_moge.py`;
- explicit documentation of the platform-specific `flash_attn` and `natten` dependencies discovered by clean-room execution;
- the maintained Trellis2 upstream contract and rebaseline tooling documented in `UPSTREAM.md`.

The accepted clean-room profile is documented in `docs/REGRESSION.md`. Detailed runtime requirements are in `docs/INSTALLATION.md`; reconstruction behavior and public nodes are in `docs/FEATURES.md`.

## v0.1.1 release note

The 0.1.1 codebase adds disk-backed preprocessing and external WTiVo integration. The above clean-room validation refers exclusively to the earlier 24-view 0.1.0 reconstruction pipeline. It is **not** an end-to-end validation of the new 120-view cache + WTiVo example. See `docs/REGRESSION.md` for the new acceptance gates. The pinned Trellis2 upstream reference commit remains unchanged.
