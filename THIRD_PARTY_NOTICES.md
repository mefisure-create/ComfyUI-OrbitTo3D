# Third-party notices

## visualbruno/ComfyUI-Trellis2

Parts of the TRELLIS/Pixal3D runtime in this repository are derived from
`visualbruno/ComfyUI-Trellis2`:

- https://github.com/visualbruno/ComfyUI-Trellis2
- Orbit baseline commit: `b0e84c0f29857a444e8bcb5b9be2ea218e96efd0`

That repository is distributed under the MIT License and retains the Microsoft
Corporation copyright notice found in this repository's `LICENSE` file.

Orbit-specific wrappers, multiview reconstruction changes, VGGT/VSR helpers,
MoGe adapters, Angle Grid and packaging/tooling are maintained separately in
ComfyUI-OrbitTo3D.

See `UPSTREAM.md` and `upstream_symbols.json` for the maintained provenance and
update contract.

## External workflow/runtime dependencies (not vendored)

The 0.1.1 workflow uses Akatz Execution Inversion (`akatz-ai/ComfyUI-Execution-Inversion`) for per-view while loops. WTiVo, `orbit_raw_component_tool`, and the audited systemd launcher are separately installed on the host. The solver and component cleaner are not bundled; consult their own distribution/license terms. The experimental PMP helper source in `orbit3d_ext/pmp_helper/` is based on CGAL libraries, which must be installed under their applicable licenses when built.
