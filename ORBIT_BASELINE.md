# Orbit3D release baseline

Stage: v0.1.0 release candidate hardening
Date: 2026-10-29

The runtime reconstruction baseline remains the verified Stage 6B7 mesh-only Pixal3D multiview implementation. Release hardening has now also passed a complete clean-room regression.

Validated release-candidate additions include:

- `Orbit3DMoGeInference`, which inherits ComfyUI's native `MoGeInference.execute()` unchanged and only extends the schema validation range for `resolution_level` to 30;
- removal of the previous hidden requirement to patch `comfy_extras/nodes_moge.py`;
- explicit documentation of the platform-specific `flash_attn` and `natten` dependencies discovered by clean-room execution;
- the maintained Trellis2 upstream contract and rebaseline tooling documented in `UPSTREAM.md`.

The accepted clean-room profile is documented in `docs/REGRESSION.md`. Detailed runtime requirements are in `docs/INSTALLATION.md`; reconstruction behavior and public nodes are in `docs/FEATURES.md`.
