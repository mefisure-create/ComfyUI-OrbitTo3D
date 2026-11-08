# Installation and requirements

This document contains the detailed runtime and workflow requirements for `ComfyUI-OrbitTo3D`. The root README intentionally stays short.

# Requirements

### Host / runtime

- A current **ComfyUI** installation. The bundled example workflow was saved against ComfyUI core `0.42.3`; older builds may not provide the core MoGe / background-removal / 3D preview nodes used by the example.
- A working [`visualbruno/ComfyUI-Trellis2`](https://github.com/visualbruno/ComfyUI-Trellis2) installation with its platform-specific Pixal3D/TRELLIS2 runtime backends. Orbit intentionally does not vendor Trellis2 CUDA/Torch-specific wheels. Keep the original package installed: the example workflow also uses its downstream decode/reconstruct/export nodes.
- For `Orbit3DVGGTViewSelector`, a **VGGT-Omega** checkout and checkpoint. Portable defaults use `~/vggt-omega`.
- For `Orbit3DRealBasicVSR`, the RealBasicVSR checkpoint under `ComfyUI/models/video_sr/realbasicvsr/`.

### Orbit Python dependencies

`requirements.txt` contains only Orbit's direct, ordinary pip-level additions that are not assumed from the ComfyUI host environment:

```text
easydict
meshlib
trimesh
```

`requests` and `scipy` are used by Orbit but are already mandatory ComfyUI dependencies, so they are not duplicated here. Platform-specific Trellis/Pixal3D backends such as `cumesh`, `flex_gemm`, `o_voxel`, sparse-convolution backends and the selected attention backend are deliberately **not** placed in the generic pip requirements; use the working Trellis2 environment/wheels appropriate for your Torch/CUDA platform.

Optional features:

- `rembg` — only if `Orbit3DMultiViewConfig.remove_background=true` is used without an input mask / usable alpha channel. The bundled print workflow already supplies masks and does not need Orbit's internal background-removal fallback.
- `scikit-image` — only for Sparse `hole_fill_algorithm=remove_small_holes`. The verified workflow uses `flood_fill`.

### Required custom nodes for the bundled example workflow

These are **workflow dependencies**, not Python dependencies of the Orbit package itself. Install them through ComfyUI Manager or from their repositories before loading `example_workflows/OrbitTo3D_HighPrecision_MultiView.json`. Custom nodes used only inside a subgraph are listed too; otherwise ComfyUI will still report missing nodes when the example is opened.

| Custom-node package | Nodes used by the example | Purpose | Version recorded in the example |
|---|---|---|---|
| [`visualbruno/ComfyUI-Trellis2`](https://github.com/visualbruno/ComfyUI-Trellis2) | `Trellis2DecodeLatents`, `Trellis2SimplifyMesh` | downstream mesh decode / simplification and the companion runtime stack Orbit is based on | commit `b0e84c0f29857a444e8bcb5b9be2ea218e96efd0` |
| [`akatz-ai/ComfyUI-Execution-Inversion`](https://github.com/akatz-ai/ComfyUI-Execution-Inversion) | `WhileLoopOpen`, `WhileLoopClose`, `AccumulateNode`, `AccumulationToListNode` (`| akatz-loops`) | per-view iteration and ordered cache reference collection | commit `711a5fb244fc0b39a22f5b5603cb2a86b3534631` |
| [`city96/ComfyUI-GGUF`](https://github.com/city96/ComfyUI-GGUF) | `UnetLoaderGGUF` | loads the Qwen-Image-Edit GGUF backbone used by Marigold V2 | not pinned in workflow metadata |
| [`visualbruno/ComfyUI-Marigold-v2`](https://github.com/visualbruno/ComfyUI-Marigold-v2) | `MarigoldV2LoRALoader`, `MarigoldV2Predict` | native-resolution normal and depth prediction | not pinned in workflow metadata |
| [`Kosinkadink/ComfyUI-VideoHelperSuite`](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | `VHS_LoadVideo` | loads the turntable video as an ordered frame batch | `1.7.9` |
| [`TJ16th/comfyUI_TJ_NormalLighting`](https://github.com/TJ16th/comfyUI_TJ_NormalLighting) | `EulerLightingNode` (inside the Clay Render subgraph) | normal-map virtual lighting for the clay render | commit `f02c6094d92f974a2dbcaacc4019a71906d2b917` |

The supplied workflow also contains **`Pixal3DDebugExportMesh`, `Pixal3DDebugExportShape`, and `Pixal3DDebugExportSparse`** nodes with no publisher metadata. They are **not registered by OrbitTo3D or guaranteed by the installed Trellis2 baseline**. These are local debug/export dependencies. You must install the exact implementation already present on the source machine, or intentionally replace/remove these branches and rewire the export/cleanup dependency before calling the workflow portable. In particular, cache cleanup is gated by a debug-export output path; removing the debug export without replacing that ordering dependency is unsafe.

The example also uses **ComfyUI core** nodes, including `ComfyMathExpression`, `ImageBlend` and `GetImageSize` (in subgraphs), `LoadMoGeModel`, `VAELoader`, background-removal and standard image/mask compositing nodes. These do not require separate custom-node packages. The bundled workflow uses `Orbit3DMoGeInference` instead of the core `MoGeInference`: it inherits the native ComfyUI execution implementation unchanged and only extends the `resolution_level` validation range to 30.

If you build a different Orbit graph and provide already prepared images, masks, normal/depth maps and/or `MOGE_GEOMETRY`, you do **not** need every example-workflow package above.

# Tested reference configuration

The previous version was validated with the following reference setup. This is a **historical tested configuration**, not a minimum hardware requirement or a claim that the complete 0.1.1 workflow has been revalidated.

| Component | Tested configuration |
|---|---|
| ComfyUI | `0.42.3` |
| Input views | 24 yaw views |
| Angular step | `15°` (`0° ... 345°`) |
| Pitch | disabled for the validated path (`pitch_step=360`, elevations remain `0°`) |
| Frame super-resolution | RealBasicVSR `2×` |
| Shape512 | `512` |
| High-resolution mesh / ShapeHR | `1536` |
| GPU | NVIDIA GeForce RTX 4090, 24 GB VRAM |
| System RAM | 64 GB |

This is the **0.1.0 clean-room regression baseline**, not the new 0.1.1 example configuration: 24 selected turntable frames are spaced at 15-degree yaw intervals, each selected view is upscaled 2× before geometry preparation, and the final ShapeHR / mesh generation stage runs at resolution 1536.

For repeated or automated execution, `--cache-none` remains recommended to prevent ComfyUI's execution cache from retaining large preprocessing outputs in system RAM between runs.

# Platform-specific runtime dependencies discovered by clean-room validation

Orbit intentionally does not place Torch/CUDA-specific binary packages in generic `requirements.txt`. They must match the installed Python / PyTorch / CUDA stack.

The validated Linux reference environment was:

```text
Python 3.12
PyTorch 2.8.0+cu128
CUDA 12.8
RTX 4090 (sm_89)
```

Two additional runtime dependencies were required by the full regression path:

- `flash_attn` — used by the retained Trellis2/Pixal3D attention path. The clean-room validation used `flash-attn 2.8.3` built for Torch 2.8 / CUDA 12 / CPython 3.12 / CXX11 ABI TRUE.
- `natten` — required by NAF detail conditioning. The clean-room validation used `natten 0.21.1+torch280cu128`.

Example commands for that **specific tested stack**:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)"

pip install --no-deps \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'

pip install "natten==0.21.1+torch280cu128" -f https://whl.natten.org

python - <<'PY'
import torch, flash_attn, natten
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("flash_attn:", flash_attn.__version__)
print("natten:", natten.__version__)
PY

pip check
```

Do not copy these binary package versions blindly to a different Torch/CUDA/Python combination. Use matching wheels for the target platform.

## Trellis2 native-extension responsibility

`ComfyUI-Trellis2` remains responsible for its platform-specific native runtime stack (for example `cumesh`, `flex_gemm`, `o_voxel`, `nvdiffrast`, and their transitive dependencies). Orbit does not duplicate those packages in its generic pip metadata. A working Trellis2 installation is a prerequisite.

## Extended MoGe resolution without patching ComfyUI

The bundled workflow uses `Orbit3DMoGeInference` rather than ComfyUI's core `MoGeInference`.

`Orbit3DMoGeInference` subclasses the core node, obtains the native schema, changes only the maximum accepted `resolution_level` to 30, and inherits the native `MoGeInference.execute()` implementation unchanged. Therefore **do not edit `ComfyUI/comfy_extras/nodes_moge.py`**.

# Memory / production execution

For heavy automated Orbit workflows, run ComfyUI with `--cache-none`.  The default ComfyUI execution cache can retain very large upstream outputs (especially Marigold/Qwen image batches and other multiview intermediates) in system RAM after a prompt finishes.  In the tested 24-view workflow, retained RAM was about 88% with the normal cache and dropped to about 34% with `--cache-none`.

`--cache-none` is therefore the recommended mode for production/automation where the workflow is expected to run from start to finish and cross-run node reuse is not required.  The trade-off is deliberate: ComfyUI will recompute nodes on subsequent queues instead of reusing cached outputs.  This is a ComfyUI execution-cache behavior rather than a requirement to restart the Orbit package itself.

Example launch flag:

```text
--cache-none
```

For an automated combiner/orchestrator, it is reasonable to restart or select the ComfyUI service with this flag for Orbit jobs.

# Example workflow

The 0.1.1 release ships `example_workflows/OrbitTo3D_HighPrecision_MultiView.json` copied from the supplied working graph. It is configured for **120 yaw views at 3°** (Angle Grid `yaw_step=3`, `pitch_step=360`) and contains an **Akatz preprocessing loop, persistent Unified Cache and WTiVo postprocess route**. It uses original-host absolute paths (notably `/home/ai/orbit3d-cache`, `/home/ai/wtivo-lab/...` and `~/vggt-omega`); edit them for a new host. Earlier versions were arranged into six compact numbered sections: model loading/patching, task configuration, frame selection/preparation, geometry/normal/depth/render preparation, 3D mesh generation, and output. The expected model/checkpoint locations are listed later in this document. The complete list of third-party custom-node packages required by this example is documented under **Requirements → Required custom nodes for the bundled example workflow** above.

`Orbit3DAngleGrid.azimuths` is connected to both `Orbit3DVGGTViewSelector.target_angles` and `Orbit3DMultiViewConfig.azimuths`, so the requested yaw list has a single source of truth. `Orbit3DAngleGrid.elevations` is connected to `Orbit3DMultiViewConfig.elevations`. The current example remains yaw-only: keep `pitch_step=360`. For cache lifetime details see [CACHE.md](CACHE.md); for WTiVo and local systemd launcher setup see [WTIVO.md](WTIVO.md).

# Model/checkpoint paths used by the validated workflow

The clean-room regression reused the following model assets from the production model directory:

```text
ComfyUI/models/
  TencentARC/Pixal3D/
  facebook/dinov3-vitl16-pretrain-lvd1689m/
  microsoft/TRELLIS-image-large/ckpts/
  unet/Qwen-Image-Edit-2509-Q4_K_M.gguf
  vae/qwen_image_vae.safetensors
  loras/marigold-v2-normals.safetensors
  loras/marigold-v2-depth-Log-stage2.safetensors
  geometry_estimation/moge_3_vitg_fp16.safetensors
  RMBG/RMBG-2.0/model.safetensors
  video_sr/realbasicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth
```

VGGT-Omega was loaded from:

```text
~/vggt-omega/checkpoints/VGGT-Omega-1B-512/vggt_omega_1b_512.pt
```

Portable defaults use `~/vggt-omega`; adjust the node/path if your checkout lives elsewhere.

# Version 0.1.1 additional requirements

- Install [Akatz Execution Inversion](https://github.com/akatz-ai/ComfyUI-Execution-Inversion), matching its recursive while-loop behavior; it is essential to the per-view cache branch.
- Ensure sufficient fast local SSD space for the Unified cache and, if persistent mode is enabled, a second copy during snapshot finalization. Set `persistent_path` to a writable directory on the target host.
- WTiVo is an **optional external runtime for the node package, but required by the bundled full workflow**. The solver, `orbit_raw_component_tool`, and the narrowly authorized systemd launcher are *not shipped by Orbit*. See [WTIVO.md](WTIVO.md).
- The experimental CGAL PMP helper is distributed as C++ source. See [WTIVO.md](WTIVO.md#additional-experimental-cgal-pmp-helper) for local build instructions; it is not the WTiVo cleaner.
- Avoid running the example unchanged on a new host: debug-export node implementations and local WTiVo paths must first be resolved.
