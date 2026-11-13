"""Public ComfyUI node registry for ComfyUI-OrbitTo3D.

Only classes listed in NODE_CLASS_MAPPINGS are exposed to ComfyUI.  Internal
Trellis-derived implementations remain private to the Orbit runtime.

Registry structure:
  1. Import implementation classes and module-provided WTiVo mappings.
  2. Define the public node IDs and their UI names in one place.
  3. Optionally enable RAM profiling after the complete registry is assembled.

Existing node IDs are stable to preserve saved ComfyUI workflows.
"""

# Core Pixal3D reconstruction
from .orbit_nodes import (
    Orbit3DLoadModel,
    Orbit3DMultiViewConfig,
    Orbit3DSparseGenerator,
    Orbit3DShapeGenerator,
    Orbit3DShapeCascadeGenerator,
)

# View selection, preprocessing and geometry
from .orbit3d_ext.vggt_nodes import VGGTTurntableFourViewsNative
from .orbit3d_ext.vsr_nodes import TurntableCyclicRealBasicVSR
from .orbit3d_ext.geometry_nodes import (
    Orbit3DMoGeGeometryListToBatch,
    Orbit3DMoGeGeometryBatchToList,
)
from .orbit3d_ext.axis_canvas_nodes import Orbit3DAxisBBoxCanvas
from .orbit3d_ext.angle_nodes import Orbit3DAngleGrid
from .orbit3d_ext.moge_nodes import Orbit3DMoGeInference

# Production cache. Keep the production session/cleanup implementations:
# unified_cache_nodes contains the frame decoder, not their replacements.
from .orbit3d_ext.production_cache_nodes import (
    Orbit3DProductionCacheSession,
    Orbit3DVSRBatchCache,
    Orbit3DVSRViewAtIndex,
    Orbit3DLoopContextPack,
    Orbit3DLoopContextUnpack,
    Orbit3DUnifiedViewSpill,
    Orbit3DUnifiedCacheFinalize,
    Orbit3DUnifiedCacheManifestLoad,
    Orbit3DMultiViewCacheOpen,
    Orbit3DProductionCleanupCacheSession,
)
from .orbit3d_ext.unified_cache_nodes import Orbit3DDecodeUnifiedCacheFrame
from .orbit3d_ext.camera_checkpoint_nodes import Orbit3DExportPixal3DMVCameras
from .orbit3d_ext.experimental_shape1024_direct_nodes import Orbit3DSS64Shape1024DirectExperimental
from .orbit3d_ext.experimental_shape_nodes import (
    Orbit3DSaveSparseStage,
    Orbit3DLoadSparseStage,
    Orbit3DSaveShape512Stage,
    Orbit3DLoadShape512Stage,
    Orbit3DExperimentalShapeCascade,
)
from .orbit3d_ext.experimental_resolution_nodes import Orbit3DSS64Shape512Experimental

# Mesh repair and checkpoints
from .orbit3d_ext.meshlib_repair_nodes import Orbit3DMeshLibFillHolesNicelyPack
from .orbit3d_ext.mesh_checkpoint_nodes import (
    Orbit3DSaveMeshWithVoxelCheckpoint,
    Orbit3DLoadMeshWithVoxelCheckpoint,
)
from .orbit3d_ext.experimental_repair_nodes import (
    Orbit3DPMPPolylineHoleFillExperimental,
    Orbit3DWarpSparseWindingRemeshExperimental,
)
from .orbit3d_ext.repair_diagnostics_nodes import Orbit3DRepairDiagnosticsExperimental

# WTiVo maintains its own set of public nodes. Expand it into the final
# dictionaries below rather than registering another post-hoc update block.
from .orbit3d_ext.wtivo_integration_nodes import (
    NODE_CLASS_MAPPINGS as _orbit3d_wtivo_classes,
    NODE_DISPLAY_NAME_MAPPINGS as _orbit3d_wtivo_display,
)

WEB_DIRECTORY = None

VGGTTurntableFourViewsNative.CATEGORY = "Orbit to 3D/Views"
TurntableCyclicRealBasicVSR.CATEGORY = "Orbit to 3D/Views"

# Stable ComfyUI node IDs -> Python implementation classes.
NODE_CLASS_MAPPINGS = {
    # Production cache
    "Orbit3DCacheSession": Orbit3DProductionCacheSession,
    "Orbit3DVSRBatchCache": Orbit3DVSRBatchCache,
    "Orbit3DVSRViewAtIndex": Orbit3DVSRViewAtIndex,
    "Orbit3DLoopContextPack": Orbit3DLoopContextPack,
    "Orbit3DLoopContextUnpack": Orbit3DLoopContextUnpack,
    "Orbit3DUnifiedViewSpill": Orbit3DUnifiedViewSpill,
    "Orbit3DUnifiedCacheFinalize": Orbit3DUnifiedCacheFinalize,
    "Orbit3DUnifiedCacheManifestLoad": Orbit3DUnifiedCacheManifestLoad,
    "Orbit3DMultiViewCacheOpen": Orbit3DMultiViewCacheOpen,
    "Orbit3DCleanupCacheSession": Orbit3DProductionCleanupCacheSession,
    "Orbit3DDecodeUnifiedCacheFrame": Orbit3DDecodeUnifiedCacheFrame,
    "Orbit3DExportPixal3DMVCameras": Orbit3DExportPixal3DMVCameras,
    # Isolated local-repair experiments
    "Orbit3DSaveSparseStage": Orbit3DSaveSparseStage,
    "Orbit3DLoadSparseStage": Orbit3DLoadSparseStage,
    "Orbit3DSaveShape512Stage": Orbit3DSaveShape512Stage,
    "Orbit3DLoadShape512Stage": Orbit3DLoadShape512Stage,
    "Orbit3DSS64Shape1024DirectExperimental": Orbit3DSS64Shape1024DirectExperimental,
    "Orbit3DExperimentalShapeCascade": Orbit3DExperimentalShapeCascade,
    "Orbit3DSS64Shape512Experimental": Orbit3DSS64Shape512Experimental,

    # Core reconstruction and preparation
    "Orbit3DLoadModel": Orbit3DLoadModel,
    "Orbit3DMultiViewConfig": Orbit3DMultiViewConfig,
    "Orbit3DSparseGenerator": Orbit3DSparseGenerator,
    "Orbit3DShapeGenerator": Orbit3DShapeGenerator,
    "Orbit3DShapeCascadeGenerator": Orbit3DShapeCascadeGenerator,
    "Orbit3DVGGTViewSelector": VGGTTurntableFourViewsNative,
    "Orbit3DRealBasicVSR": TurntableCyclicRealBasicVSR,
    "Orbit3DMoGeGeometryListToBatch": Orbit3DMoGeGeometryListToBatch,
    "Orbit3DMoGeGeometryBatchToList": Orbit3DMoGeGeometryBatchToList,
    "Orbit3DAxisBBoxCanvas": Orbit3DAxisBBoxCanvas,
    "Orbit3DAngleGrid": Orbit3DAngleGrid,
    "Orbit3DMoGeInference": Orbit3DMoGeInference,

    # Mesh repair and checkpoints
    "Orbit3DMeshLibFillHolesNicelyPack": Orbit3DMeshLibFillHolesNicelyPack,
    "Orbit3DSaveMeshWithVoxelCheckpoint": Orbit3DSaveMeshWithVoxelCheckpoint,
    "Orbit3DLoadMeshWithVoxelCheckpoint": Orbit3DLoadMeshWithVoxelCheckpoint,
    "Orbit3DPMPPolylineHoleFillExperimental": Orbit3DPMPPolylineHoleFillExperimental,
    "Orbit3DWarpSparseWindingRemeshExperimental": Orbit3DWarpSparseWindingRemeshExperimental,
    "Orbit3DRepairDiagnosticsExperimental": Orbit3DRepairDiagnosticsExperimental,

    # WTiVo node IDs come from its integration module; same precedence as before.
    **_orbit3d_wtivo_classes,
}

# Node IDs -> user-visible names. Keys must match NODE_CLASS_MAPPINGS.
NODE_DISPLAY_NAME_MAPPINGS = {
    # Production cache
    "Orbit3DCacheSession": "Orbit3D - Create Cache Session",
    "Orbit3DVSRBatchCache": "Orbit3D - Cache VSR Batch (MMap)",
    "Orbit3DVSRViewAtIndex": "Orbit3D - Read VSR View",
    "Orbit3DLoopContextPack": "Orbit3D - Pack Loop Context",
    "Orbit3DLoopContextUnpack": "Orbit3D - Unpack Loop Context",
    "Orbit3DUnifiedViewSpill": "Orbit3D - Spill Unified View",
    "Orbit3DUnifiedCacheFinalize": "Orbit3D - Finalize Unified Cache",
    "Orbit3DUnifiedCacheManifestLoad": "Orbit3D - Load Unified Cache Manifest",
    "Orbit3DMultiViewCacheOpen": "Orbit3D - Open MultiView Cache",
    "Orbit3DCleanupCacheSession": "Orbit3D - Delete Cache Session",
    "Orbit3DDecodeUnifiedCacheFrame": "Orbit3D - Decode Unified Cache Frame",
    "Orbit3DExportPixal3DMVCameras": "Orbit3D - Export Pixal3D MV Cameras",
    "Orbit3DSaveSparseStage": "Orbit3D - Save Sparse Stage [Experimental]",
    "Orbit3DLoadSparseStage": "Orbit3D - Load Sparse Stage [Experimental]",
    "Orbit3DSaveShape512Stage": "Orbit3D - Save Shape512 Stage [Experimental]",
    "Orbit3DLoadShape512Stage": "Orbit3D - Load Shape512 Stage [Experimental]",
    "Orbit3DSS64Shape1024DirectExperimental": "Orbit3D - SS64 to Shape1024 Direct [Experimental]",
    "Orbit3DExperimentalShapeCascade": "Orbit3D - Experimental Shape Cascade",
    "Orbit3DSS64Shape512Experimental": "Orbit3D - SS64 to Shape512 Weights [Experimental]",

    # Core reconstruction and preparation
    "Orbit3DLoadModel": "Orbit3D - Load Model",
    "Orbit3DMultiViewConfig": "Orbit3D - MultiView Config",
    "Orbit3DSparseGenerator": "Orbit3D - Sparse Generator",
    "Orbit3DShapeGenerator": "Orbit3D - Shape 512 Generator",
    "Orbit3DShapeCascadeGenerator": "Orbit3D - Shape Cascade Generator",
    "Orbit3DVGGTViewSelector": "Orbit3D - VGGT Turntable View Selector",
    "Orbit3DRealBasicVSR": "Orbit3D - Cyclic RealBasicVSR",
    "Orbit3DMoGeGeometryListToBatch": "Orbit3D - MoGe Geometry List To Batch",
    "Orbit3DMoGeGeometryBatchToList": "Orbit3D - MoGe Geometry Batch To List",
    "Orbit3DAxisBBoxCanvas": "Orbit3D - Axis BBox Canvas (MMap)",
    "Orbit3DAngleGrid": "Orbit3D - Yaw/Pitch Angle Grid",
    "Orbit3DMoGeInference": "Orbit3D - MoGe Inference (Extended Resolution)",

    # Mesh repair and checkpoints
    "Orbit3DMeshLibFillHolesNicelyPack": "Orbit3D - MeshLib Fill + Filter + Pack",
    "Orbit3DSaveMeshWithVoxelCheckpoint": "Orbit3D - Save MESHWITHVOXEL Checkpoint",
    "Orbit3DLoadMeshWithVoxelCheckpoint": "Orbit3D - Load MESHWITHVOXEL Checkpoint",
    "Orbit3DPMPPolylineHoleFillExperimental": "Orbit3D - PMP Polyline Hole Fill [Experimental]",
    "Orbit3DWarpSparseWindingRemeshExperimental": "Orbit3D - Warp Sparse Winding Remesh [Experimental]",
    "Orbit3DRepairDiagnosticsExperimental": "Orbit3D - Repair Diagnostics [Experimental]",

    # WTiVo display names
    **_orbit3d_wtivo_display,
}

# Optional, isolated HR64 diagnostics: register before profiler instrumentation.
# Keep existing production and experimental node mappings unmodified.
from .orbit3d_ext.experimental_hr64_compare_nodes import (
    Orbit3DExtractCascadeHR64CoordsExperimental,
    Orbit3DCompareSS64CascadeHR64Experimental,
)
NODE_CLASS_MAPPINGS["Orbit3DExtractCascadeHR64CoordsExperimental"] = Orbit3DExtractCascadeHR64CoordsExperimental
NODE_CLASS_MAPPINGS["Orbit3DCompareSS64CascadeHR64Experimental"] = Orbit3DCompareSS64CascadeHR64Experimental
NODE_DISPLAY_NAME_MAPPINGS["Orbit3DExtractCascadeHR64CoordsExperimental"] = "Orbit3D - Extract Cascade HR64 Coords [Experimental]"
NODE_DISPLAY_NAME_MAPPINGS["Orbit3DCompareSS64CascadeHR64Experimental"] = "Orbit3D - Compare SS64 vs Cascade HR64 [Experimental]"

# Separate SS64 occupancy-fill experiment (registration before RAM profiler).
from .orbit3d_ext.experimental_ss_force_fill_nodes import Orbit3DSSForceFillExperimental
NODE_CLASS_MAPPINGS["Orbit3DSSForceFillExperimental"] = Orbit3DSSForceFillExperimental
NODE_DISPLAY_NAME_MAPPINGS["Orbit3DSSForceFillExperimental"] = "Orbit3D - SS Force Fill [Experimental]"

# Isolated coarse-to-fine SS32 / SS64 occupancy fusion experiment.
from .orbit3d_ext.experimental_ss32_ss64_fusion_nodes import Orbit3DSS32SS64FusionExperimental
NODE_CLASS_MAPPINGS["Orbit3DSS32SS64FusionExperimental"] = Orbit3DSS32SS64FusionExperimental
NODE_DISPLAY_NAME_MAPPINGS["Orbit3DSS32SS64FusionExperimental"] = "Orbit3D - Fuse SS32 and SS64 [Experimental]"


# Isolated pre-Cascade HR64 free-space audit: no Shape1024 generation.
from .orbit3d_ext.experimental_hr_free_space_nodes import Orbit3DHRFreeSpaceAuditFilter
NODE_CLASS_MAPPINGS["Orbit3DHRFreeSpaceAuditFilter"] = Orbit3DHRFreeSpaceAuditFilter
NODE_DISPLAY_NAME_MAPPINGS["Orbit3DHRFreeSpaceAuditFilter"] = "Orbit3D - HR Free-Space Audit & Filter [Experimental]"

# HR VAE subdivision free-space audit (independent from the existing HR64 audit).
from .orbit3d_ext.experimental_hr_subvoxel_audit_nodes import Orbit3DHRSubvoxelFreeSpaceAudit
NODE_CLASS_MAPPINGS["Orbit3DHRSubvoxelFreeSpaceAudit"] = Orbit3DHRSubvoxelFreeSpaceAudit
NODE_DISPLAY_NAME_MAPPINGS["Orbit3DHRSubvoxelFreeSpaceAudit"] = "Orbit3D - HR Subvoxel Free-Space Audit [Experimental]"

# V8: global Shape512 VAE pruning; experiment-only, no production generator edits.
from .orbit3d_ext.experimental_vae_prune_decode_nodes import Orbit3DShape512VAEGlobalPruneDecode
NODE_CLASS_MAPPINGS["Orbit3DShape512VAEGlobalPruneDecode"] = Orbit3DShape512VAEGlobalPruneDecode
NODE_DISPLAY_NAME_MAPPINGS["Orbit3DShape512VAEGlobalPruneDecode"] = "Orbit3D - Shape512 VAE Global Prune & Export [Experimental]"

# V9 read-only export: exact ShapeHR scaffold, no change to generation.
from .orbit3d_ext.experimental_hr_surface_bridge_nodes import Orbit3DExportCascadeHRScaffoldV9
NODE_CLASS_MAPPINGS["Orbit3DExportCascadeHRScaffoldV9"] = Orbit3DExportCascadeHRScaffoldV9
NODE_DISPLAY_NAME_MAPPINGS["Orbit3DExportCascadeHRScaffoldV9"] = "Orbit3D - Export Cascade HR Scaffold [V9 Diagnostic]"

# SS64 completion: opt-in T5 and one-pass SurfaceAll.
from .orbit3d_ext.experimental_ss64_t5_nodes import (
    Orbit3DSS64T5Experimental,
    Orbit3DSS64SurfaceAll,
)
NODE_CLASS_MAPPINGS["Orbit3DSS64T5Experimental"] = Orbit3DSS64T5Experimental
NODE_CLASS_MAPPINGS["Orbit3DSS64SurfaceAll"] = Orbit3DSS64SurfaceAll
NODE_DISPLAY_NAME_MAPPINGS["Orbit3DSS64T5Experimental"] = "Orbit3D - Sparse64 Generator T5 [Experimental]"
NODE_DISPLAY_NAME_MAPPINGS["Orbit3DSS64SurfaceAll"] = "Orbit3D - Sparse64 Generator T5 + SurfaceAll"

# Optional diagnostics. Instrument the final registry, including experimental
# and WTiVo nodes, not a partially assembled registry.
try:
    from .orbit3d_ext.ram_profiler import install_profiling as _orbit3d_install_ram_profiling
    _orbit3d_install_ram_profiling(NODE_CLASS_MAPPINGS)
except Exception as _orbit3d_ram_profile_error:
    print(f"[Orbit3D RAM Profile] instrumentation unavailable: {_orbit3d_ram_profile_error!r}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
