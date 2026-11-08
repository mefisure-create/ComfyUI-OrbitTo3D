# WTiVo mesh repair / remeshing integration (0.1.1)

Orbit provides **ComfyUI nodes that orchestrate an external WTiVo executable**; the WTiVo solver is **not vendored** into this repository, and installing Orbit's Python requirements does not install it. The current integration targets a Linux host with systemd, NVIDIA CUDA and a separately configured privileged launcher.

## Mesh route in the supplied workflow

`Trellis2DecodeLatents` → `Orbit3DComponentCleanerPLY` (`pre`) → `Orbit3DWTiVoExternalRunner` → `Orbit3DComponentCleanerPLY` (`post`) → downstream simplify/export.

WTiVo operates on triangles/vertices, not the original Pixal3D voxel attributes. Its result is returned as `MESHWITHVOXEL` **geometry-only** (without voxel coords, semantic attributes or texture). A passing edge audit means every edge has an appropriate face incidence under that audit; do not equate this alone with a guarantee of a printable, intersection-free or semantically correct sculpture.

## External tools required

1. A working, separately built **WTiVo** checkout with the native CUDA and CGAL dependencies appropriate for your machine. The workflow's `wtivo_root` points to its directory. This repository does not carry WTiVo source or publish its license on WTiVo's behalf.
2. A separately compiled `orbit_raw_component_tool` executable for the pre/post component cleaning. Set its absolute executable path in both `Orbit3DComponentCleanerPLY` nodes, or provide it through `ORBIT3D_COMPONENT_TOOL`/`PATH`. **Its source is not included in this archive.**
3. A root-owned, locally audited **`orbit3d-wtivo-systemd-launch`** helper (default `/usr/local/sbin/orbit3d-wtivo-systemd-launch`; configurable via `ORBIT3D_WTIVO_LAUNCHER`). The ComfyUI service user must be permitted to invoke **only that exact helper** using noninteractive `sudo -n`; do not grant unrestricted passwordless `systemctl`, arbitrary shell execution or broad root privileges.
4. A working `/usr/bin/systemctl`, `/usr/bin/sudo`, `/usr/bin/nvidia-smi` and enough disk/RAM/VRAM resources. This specific WTiVo path is **not portable to Windows or machines without systemd**.

The current Python runner invokes the audited helper as `run <unit> <input-ply> <output-ply> <input-res> <final-res> <proxy-points> <lambda-fill> <feature-weight> <threads> <memory-limit-gib> <gpu-index> <wtivo-root>` and `stop <unit>`. That is an **interface contract**, not a distributed launcher implementation. A missing launcher fails explicitly rather than silently executing WTiVo without resource isolation. Configure/test this privileged component outside the ComfyUI process according to your local security policy.

## Run controls and output

The runner uses `comfy.model_management.unload_all_models()` before launching the solver, checks host available RAM and GPU free memory, then starts a **transient** systemd unit with a memory cap. It samples resource utilization, aborts the worker when ComfyUI cancels, and rejects an absent/non-passing WTiVo edge audit.

The 0.1.1 example is tuned to one local setup: `input_res=1536`, `final_res=2048`, `proxy_points=12000000`, `lambda_fill=10`, `feature_weight=1.5`, `threads=4`, `memory_limit_gib=40`, `minimum_mem_available_gib=24`, and `gpu_index=0`. These values are **not universal hardware requirements**, nor are they an assurance that every mesh can be repaired.

The runner writes temporary PLY files and log/JSON reports in ComfyUI's output directory under `orbit3d_wtivo_runs`. They can consume substantial disk space and are not covered by the preprocessing cache cleanup node. Inspect reports and prune completed runs according to your retention policy.

## Additional experimental CGAL PMP helper

The **experimental** PMP polyline repair nodes are separate from WTiVo. Their helper is supplied as **C++ source** under `orbit3d_ext/pmp_helper/`, not as a portable precompiled executable. If you use those experimental nodes on Linux, build locally with CGAL and CMake:

```bash
cmake -S orbit3d_ext/pmp_helper -B /tmp/orbit3d-pmp-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/orbit3d-pmp-build --parallel
mkdir -p orbit3d_ext/bin
install -m 755 /tmp/orbit3d-pmp-build/orbit3d_pmp_polyline orbit3d_ext/bin/orbit3d_pmp_polyline
```

The built binary is deliberately ignored by Git: it depends on the local compiler, platform and CGAL environment. This helper **does not replace** WTiVo or `orbit_raw_component_tool`.

## Validation boundary

The 0.1.1 package passes static syntax/registry/manifest tests when reported as passed. Full verification still requires an end-to-end run on a system that has the exact external WTiVo solver, launcher, cleaner, GPU backends and appropriate permissions installed. See [REGRESSION.md](REGRESSION.md).
