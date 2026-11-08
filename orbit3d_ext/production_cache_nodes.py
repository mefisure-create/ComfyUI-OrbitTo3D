"""Production cache/loop API for ComfyUI-OrbitTo3D.

This module is intentionally a thin public facade over the validated internal
implementations accumulated during the memory-optimization work.

Public nodes are task-specific.  Experimental/generic nodes remain internal so
the production UI does not expose implementation knobs that the validated
workflow does not need.

No reconstruction formulas, camera math, conditioning, seeds, VSR/VGGT model
mechanics, or geometry guidance are changed here.
"""
from __future__ import annotations

import os
from pathlib import Path

from .unified_cache_nodes import (
    Orbit3DCacheSession as _CacheSession,
    Orbit3DReleaseCPUMemory as _ReleaseCPUMemory,
    Orbit3DCleanupCacheSession as _CleanupCacheSession,
    Orbit3DUnifiedViewPayload as _UnifiedViewPayload,
    Orbit3DUnifiedCachePrepare as _UnifiedCachePrepare,
)
from .generic_cache_nodes import (
    Orbit3DGenericCacheSpill as _GenericCacheSpill,
    Orbit3DGenericCacheRefReport as _GenericCacheRefReport,
)
from .image_mmap_for_nodes import (
    Orbit3DImageBatchSpillMMapIndexed as _ImageBatchSpillMMapIndexed,
    Orbit3DMMapImageAtIndex as _MMapImageAtIndex,
)
from .akatz_static_context_nodes import (
    Orbit3DStaticLoopContextPack as _StaticLoopContextPack,
    Orbit3DStaticLoopContextUnpack as _StaticLoopContextUnpack,
    Orbit3DStaticLoopFOVFromSession as _StaticLoopFOVFromSession,
)


class Orbit3DProductionCacheSession(_CacheSession):
    CATEGORY = "Orbit to 3D/Cache"


class Orbit3DVSRBatchCache(_ImageBatchSpillMMapIndexed):
    """Persist the complete VSR IMAGE batch once and expose a file-backed mmap."""

    CATEGORY = "Orbit to 3D/Cache"


class Orbit3DVSRViewAtIndex(_MMapImageAtIndex):
    """Return one zero-copy VSR view from the persistent mmap batch."""

    CATEGORY = "Orbit to 3D/Loop"


class Orbit3DLoopContextPack(_StaticLoopContextPack):
    """Pack all loop-invariant objects once for Akatz value4 state."""

    CATEGORY = "Orbit to 3D/Loop"


class Orbit3DLoopContextUnpack(_StaticLoopContextUnpack):
    """Unpack loop-invariant state inside one recursive Akatz iteration."""

    CATEGORY = "Orbit to 3D/Loop"


class Orbit3DUnifiedViewSpill:
    """Persist one complete synchronized view to the Unified disk cache.

    This is the production facade for:
      UnifiedViewPayload -> GenericCacheSpill(stream="Unified",
                                              verify_roundtrip=False)

    The hard-coded options match the validated 72/120-view workflow.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_session": ("ORBIT3D_CACHE_SESSION",),
                "iteration_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0x7FFFFFFFFFFFFFFF,
                        "forceInput": True,
                    },
                ),
                "moge_geometry": ("MOGE_GEOMETRY",),
                "normal_map": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("ORBIT3D_CACHE_REF", "STRING")
    RETURN_NAMES = ("cache_ref", "report")
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"

    def run(
        self,
        cache_session,
        iteration_index,
        moge_geometry,
        normal_map,
        depth_map,
        image,
        mask,
    ):
        payload, = _UnifiedViewPayload().run(
            moge_geometry=moge_geometry,
            normal_map=normal_map,
            depth_map=depth_map,
            image=image,
            mask=mask,
        )
        return _GenericCacheSpill().run(
            cache_session=cache_session,
            iteration_index=iteration_index,
            stream="Unified",
            value_1=payload,
            verify_roundtrip=False,
        )


class Orbit3DUnifiedCacheFinalize:
    """Build the ordered Unified manifest and perform the proven final CPU release."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cache_ref": ("ORBIT3D_CACHE_REF",)}}

    INPUT_IS_LIST = True
    RETURN_TYPES = (
        "STRING",
        "ORBIT3D_GENERIC_CACHE_MANIFEST",
        "STRING",
    )
    RETURN_NAMES = ("report", "cache_manifest", "release_report")
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"

    def run(self, cache_ref):
        result = _GenericCacheRefReport().run(cache_ref)
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError(
                "Orbit3D production cache requires Generic Cache Ref Report "
                "with (report, cache_manifest) outputs"
            )
        report, cache_manifest = result
        release_report, = _ReleaseCPUMemory().run(
            after=report,
            python_gc=True,
            malloc_trim=True,
        )
        return report, cache_manifest, release_report


class Orbit3DUnifiedCacheManifestLoad:
    """Restore an existing Unified spill manifest as a typed runtime handle.

    Only JSON metadata and filesystem structure are read here. No safetensors
    payload is materialized. The existing 3.6.12 MultiView integrity gate
    remains responsible for checking safetensors headers/keys downstream.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Existing spill_Unified_manifest.json or its "
                            "Orbit3D cache-session directory."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("ORBIT3D_GENERIC_CACHE_MANIFEST", "STRING")
    RETURN_NAMES = ("cache_manifest", "report")
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @staticmethod
    def _resolve_manifest_path(value):
        raw = str(value or "").strip()
        if not raw:
            raise ValueError("manifest_path is empty")
        path = Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
        if path.is_dir():
            path = path / "spill_Unified_manifest.json"
        return path

    def run(self, manifest_path):
        import json

        path = self._resolve_manifest_path(manifest_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Orbit3D Unified cache manifest not found: {path}"
            )

        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(
                f"Invalid Orbit3D Unified cache manifest JSON: {path}: {exc}"
            ) from exc

        if not isinstance(manifest, dict):
            raise TypeError(
                f"Unified cache manifest must be a JSON object, got "
                f"{type(manifest).__name__}"
            )

        if manifest.get("orbit3d_cache_type") != "generic_spill_manifest":
            raise ValueError(
                "Expected orbit3d_cache_type='generic_spill_manifest', got "
                f"{manifest.get('orbit3d_cache_type')!r}"
            )

        stream = str(manifest.get("stream", "")).strip()
        if stream.lower() != "unified":
            raise ValueError(
                f"Expected Unified spill manifest, got stream={stream!r}"
            )

        version = int(manifest.get("version", 1))
        if version != 1:
            raise ValueError(
                f"Unsupported Unified cache manifest version: {version}"
            )

        session_id = str(manifest.get("session_id", "")).strip()
        root_text = str(manifest.get("root", "")).strip()
        if not session_id or not root_text:
            raise ValueError(
                "Unified cache manifest is missing session_id/root"
            )

        root = Path(
            os.path.expandvars(os.path.expanduser(root_text))
        ).resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"Unified cache session root is missing: {root}"
            )

        session_json = root / "session.json"
        if not session_json.is_file():
            raise FileNotFoundError(
                f"Unified cache session marker is missing: {session_json}"
            )

        try:
            session = json.loads(session_json.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(
                f"Invalid Orbit3D session marker: {session_json}: {exc}"
            ) from exc

        if str(session.get("session_id", "")).strip() != session_id:
            raise ValueError(
                "Unified cache manifest/session.json session_id mismatch"
            )

        marker_root = str(session.get("root", "")).strip()
        if marker_root and Path(
            os.path.expandvars(os.path.expanduser(marker_root))
        ).resolve() != root:
            raise ValueError(
                "Unified cache session.json root does not match manifest root"
            )

        # Finalize adds manifest_path to its returned in-memory dict after
        # writing JSON. Recreate that runtime-only field when reopening.
        manifest["manifest_path"] = str(path)

        refs = list(manifest.get("items") or [])
        if not refs:
            raise ValueError("Unified cache manifest contains no items")

        declared_count = int(manifest.get("item_count", len(refs)))
        if declared_count != len(refs):
            raise ValueError(
                f"Unified cache item_count={declared_count}, "
                f"but items has {len(refs)} entries"
            )

        seen_indices = set()
        missing = []

        for ordinal, ref in enumerate(refs):
            if not isinstance(ref, dict):
                raise TypeError(
                    f"Unified cache item {ordinal} is not an object"
                )

            try:
                iteration_index = int(ref["iteration_index"])
            except Exception as exc:
                raise ValueError(
                    f"Unified cache item {ordinal} has invalid/missing "
                    "iteration_index"
                ) from exc

            if iteration_index in seen_indices:
                raise ValueError(
                    f"Duplicate Unified cache iteration_index={iteration_index}"
                )
            seen_indices.add(iteration_index)

            item_path_text = str(ref.get("path", "")).strip()
            if not item_path_text:
                missing.append(f"item[{ordinal}] path=<empty>")
                continue

            item_dir = Path(
                os.path.expandvars(os.path.expanduser(item_path_text))
            ).resolve()

            if not item_dir.is_dir():
                missing.append(f"item[{ordinal}] dir={item_dir}")
                continue

            item_json = item_dir / "item.json"
            tensor_file = item_dir / "tensors.safetensors"
            if not item_json.is_file():
                missing.append(str(item_json))
            if not tensor_file.is_file():
                missing.append(str(tensor_file))

        if missing:
            preview = "\n  ".join(missing[:12])
            suffix = "" if len(missing) <= 12 else (
                f"\n  ... and {len(missing) - 12} more"
            )
            raise FileNotFoundError(
                "Unified cache is incomplete; missing paths:\n  "
                + preview + suffix
            )

        ordered = sorted(seen_indices)
        if ordered and ordered != list(range(len(ordered))):
            raise ValueError(
                "Unified cache iteration indices are not contiguous 0..V-1: "
                f"first={ordered[:8]!r}, last={ordered[-8:]!r}"
            )

        report = (
            "Orbit3D existing Unified cache loaded: "
            f"V={len(refs)}; session={session_id}; manifest={path}; "
            "tensor payloads loaded=0"
        )
        print(f"[Orbit3D Cache] {report}", flush=True)
        return manifest, report

class Orbit3DMultiViewCacheOpen:
    """Open the validated disk-backed MultiView handles and restore scalar FOV."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_manifest": ("ORBIT3D_GENERIC_CACHE_MANIFEST",),
                "after_release": (
                    "STRING",
                    {"forceInput": True, "multiline": True},
                ),
            }
        }

    RETURN_TYPES = (
        "ORBIT3D_VIEW_CACHE",
        "ORBIT3D_MOGE_CACHE",
        "ORBIT3D_MARIGOLD_CACHE",
        "ORBIT3D_CACHE_SESSION",
        "FLOAT",
        "STRING",
    )
    RETURN_NAMES = (
        "view_cache",
        "moge_cache",
        "marigold_cache",
        "cache_session",
        "fov",
        "report",
    )
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Cache"

    def run(self, cache_manifest, after_release):
        (
            view_cache,
            moge_cache,
            marigold_cache,
            cache_session,
            report,
        ) = _UnifiedCachePrepare().run(
            cache_manifest=cache_manifest,
            after_release=after_release,
            snapshot_persistent=False,  # Read-only open: do not duplicate archive.
        )
        fov, = _StaticLoopFOVFromSession().run(cache_session)
        return (
            view_cache,
            moge_cache,
            marigold_cache,
            cache_session,
            fov,
            report,
        )


class Orbit3DProductionCleanupCacheSession(_CleanupCacheSession):
    CATEGORY = "Orbit to 3D/Cache"


NODE_CLASS_MAPPINGS = {
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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Orbit3DCacheSession": "Orbit3D - Create Cache Session",
    "Orbit3DVSRBatchCache": "Orbit3D - Cache VSR Batch (MMap)",
    "Orbit3DVSRViewAtIndex": "Orbit3D - Read VSR View",
    "Orbit3DLoopContextPack": "Orbit3D - Pack Loop Context",
    "Orbit3DLoopContextUnpack": "Orbit3D - Unpack Loop Context",
    "Orbit3DUnifiedViewSpill": "Orbit3D - Spill Unified View",
    "Orbit3DUnifiedCacheFinalize": "Orbit3D - Finalize Unified Cache",
    "Orbit3DUnifiedCacheManifestLoad": 'Orbit3D - Load Unified Cache Manifest',
    "Orbit3DMultiViewCacheOpen": "Orbit3D - Open MultiView Cache",
    "Orbit3DCleanupCacheSession": "Orbit3D - Delete Cache Session",
}
