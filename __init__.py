# ORBIT3D_PUBLIC_API_V1
# Public registration for the retained Orbit runtime nodes.  Trellis-derived
# implementation classes remain internal and are not exported as separate nodes.

from .orbit_nodes import (
    Orbit3DLoadModel,
    Orbit3DMultiViewConfig,
    Orbit3DSparseGenerator,
    Orbit3DShapeGenerator,
    Orbit3DShapeCascadeGenerator,
)
from .orbit3d_ext.vggt_nodes import VGGTTurntableFourViewsNative
from .orbit3d_ext.vsr_nodes import TurntableCyclicRealBasicVSR
from .orbit3d_ext.geometry_nodes import (
    Orbit3DMoGeGeometryListToBatch,
    Orbit3DMoGeGeometryBatchToList,
)
from .orbit3d_ext.angle_nodes import Orbit3DAngleGrid
from .orbit3d_ext.moge_nodes import Orbit3DMoGeInference

WEB_DIRECTORY = None

VGGTTurntableFourViewsNative.CATEGORY = "Orbit to 3D/Views"
TurntableCyclicRealBasicVSR.CATEGORY = "Orbit to 3D/Views"

NODE_CLASS_MAPPINGS = {
    "Orbit3DLoadModel": Orbit3DLoadModel,
    "Orbit3DMultiViewConfig": Orbit3DMultiViewConfig,
    "Orbit3DSparseGenerator": Orbit3DSparseGenerator,
    "Orbit3DShapeGenerator": Orbit3DShapeGenerator,
    "Orbit3DShapeCascadeGenerator": Orbit3DShapeCascadeGenerator,
    "Orbit3DVGGTViewSelector": VGGTTurntableFourViewsNative,
    "Orbit3DRealBasicVSR": TurntableCyclicRealBasicVSR,
    "Orbit3DMoGeGeometryListToBatch": Orbit3DMoGeGeometryListToBatch,
    "Orbit3DMoGeGeometryBatchToList": Orbit3DMoGeGeometryBatchToList,
    "Orbit3DAngleGrid": Orbit3DAngleGrid,
    "Orbit3DMoGeInference": Orbit3DMoGeInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Orbit3DLoadModel": "Orbit3D - Load Model",
    "Orbit3DMultiViewConfig": "Orbit3D - MultiView Config",
    "Orbit3DSparseGenerator": "Orbit3D - Sparse Generator",
    "Orbit3DShapeGenerator": "Orbit3D - Shape 512 Generator",
    "Orbit3DShapeCascadeGenerator": "Orbit3D - Shape Cascade Generator",
    "Orbit3DVGGTViewSelector": "Orbit3D - VGGT Turntable View Selector",
    "Orbit3DRealBasicVSR": "Orbit3D - Cyclic RealBasicVSR",
    "Orbit3DMoGeGeometryListToBatch": "Orbit3D - MoGe Geometry List To Batch",
    "Orbit3DMoGeGeometryBatchToList": "Orbit3D - MoGe Geometry Batch To List",
    "Orbit3DAngleGrid": "Orbit3D - Yaw/Pitch Angle Grid",
    "Orbit3DMoGeInference": "Orbit3D - MoGe Inference (Extended Resolution)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
