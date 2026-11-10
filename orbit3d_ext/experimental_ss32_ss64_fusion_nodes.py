"""Experimental coarse SS32 scaffold + detailed SS64 occupancy fusion.

No trained weights, image features, or production graph behavior are modified.
The SS64 occupied cells are ALWAYS preserved. Candidate additions come only
from 2x expanded SS32 occupancy; protection comes from the ORIGINAL SS64 air.
"""
from __future__ import annotations

import json
import numpy as np
import torch


class Orbit3DSS32SS64FusionExperimental:
    CATEGORY = "Orbit to 3D/Experimental"
    FUNCTION = "process"
    RETURN_TYPES = ("COORDS", "INT", "COORDS", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "fused_coords", "resolution", "added_coords", "added_voxels",
        "final_voxels", "report_json",
    )
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "ss32_coords": ("COORDS",),
            "ss32_resolution": ("INT", {"forceInput": True}),
            "ss64_coords": ("COORDS",),
            "ss64_resolution": ("INT", {"forceInput": True}),
            "fusion_mode": (
                ["protected_coarse", "protected_core6", "protected_core26",
                 "core6_only", "full_coarse_diagnostic"],
                {"default": "protected_coarse", "tooltip":
                 "protected_coarse=SS32 envelope with 4-axis SS64 open-air protection. "
                 "protected_core6/26=also restrict to interior of coarse envelope. "
                 "core6_only=core with no air protection. "
                 "full_coarse_diagnostic=unsafe unguarded union (reference only)."},
            ),
            "max_added_voxels": ("INT", {"default": 10000, "min": 0, "max": 250000}),
            "min_parent_agreement_percent": (
                "FLOAT", {"default": 75.0, "min": 0.0, "max": 100.0, "step": 1.0,
                          "tooltip": "Fail if too many SS64 voxels fall outside the expanded SS32. Use 0 to disable; checks alignment, not quality."},
            ),
            "verbose": ("BOOLEAN", {"default": True}),
        }}

    @staticmethod
    def _grid(coords: torch.Tensor, resolution: int, label: str) -> np.ndarray:
        if not isinstance(coords, torch.Tensor) or coords.ndim != 2 or coords.shape[1] != 4:
            raise TypeError(f"{label}: expected COORDS torch.Tensor [N,4]")
        if coords.is_floating_point() or coords.is_complex():
            raise TypeError(f"{label}: coordinates must be integer")
        if coords.shape[0] == 0:
            raise ValueError(f"{label}: empty occupancy")
        data = coords.detach().cpu().to(torch.int64).contiguous()
        if bool((data[:, 0] != 0).any()):
            raise ValueError(f"{label}: only batch 0 is supported")
        if bool((data[:, 1:] < 0).any()) or bool((data[:, 1:] >= resolution).any()):
            raise ValueError(f"{label}: coordinates outside [0,{resolution-1}]^3")
        if int(torch.unique(data, dim=0).shape[0]) != data.shape[0]:
            raise ValueError(f"{label}: duplicate coordinates")
        xyz = data[:, 1:].numpy()
        out = np.zeros((resolution, resolution, resolution), dtype=np.bool_)
        out[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = True
        return out

    @staticmethod
    def _coords(grid: np.ndarray) -> torch.Tensor:
        xyz = np.argwhere(grid)
        rows = np.concatenate(
            (np.zeros((xyz.shape[0], 1), dtype=np.int32), xyz.astype(np.int32)),
            axis=1,
        )
        return torch.from_numpy(np.ascontiguousarray(rows))

    @staticmethod
    def _four_direction_behind_surface(fine: np.ndarray) -> np.ndarray:
        """True if a fine-grid cell has fine SS occupancy on both sides of X and Z.

        This is an approximate horizontal opening guard, NOT true turntable
        camera/mask visibility, nor a semantic object/air classifier.
        """
        x = np.cumsum(fine, axis=0, dtype=np.int32)
        z = np.cumsum(fine, axis=2, dtype=np.int32)
        x_hidden = (x - fine > 0) & ((x[-1:, :, :] - x) > 0)
        z_hidden = (z - fine > 0) & ((z[:, :, -1:] - z) > 0)
        return x_hidden & z_hidden

    @staticmethod
    def _coarse_core(coarse: np.ndarray, mode: str) -> np.ndarray:
        try:
            from scipy.ndimage import binary_erosion, generate_binary_structure
        except ImportError as exc:
            raise RuntimeError("coarse-core fusion requires scipy") from exc
        neighbors = 1 if mode in ("protected_core6", "core6_only") else 3
        return binary_erosion(
            coarse, structure=generate_binary_structure(3, neighbors),
            iterations=1, border_value=0,
        )

    def process(
        self, ss32_coords, ss32_resolution,
        ss64_coords, ss64_resolution,
        fusion_mode="protected_coarse", max_added_voxels=10000,
        min_parent_agreement_percent=75.0, verbose=True,
    ):
        if int(ss32_resolution) != 32 or int(ss64_resolution) != 64:
            raise ValueError(
                f"SS32+SS64 fusion requires grids 32 and 64, got {ss32_resolution} and {ss64_resolution}"
            )
        allowed = {"protected_coarse", "protected_core6", "protected_core26",
                   "core6_only", "full_coarse_diagnostic"}
        if fusion_mode not in allowed:
            raise ValueError(f"Unknown fusion_mode={fusion_mode!r}")

        coarse32 = self._grid(ss32_coords, 32, "SS32")
        fine = self._grid(ss64_coords, 64, "SS64")
        coarse64 = coarse32.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
        fine_n = int(fine.sum())
        coarse_n = int(coarse64.sum())
        in_coarse = int((fine & coarse64).sum())
        agreement = 100.0 * in_coarse / fine_n
        if agreement < float(min_parent_agreement_percent):
            raise RuntimeError(
                f"SS32/SS64 overlap too low: {agreement:.1f}% of fine voxels lie in the "
                f"SS32 envelope; required {float(min_parent_agreement_percent):.1f}%. "
                "Verify same object, framing, seed and alignment."
            )

        candidates = coarse64 & ~fine
        if fusion_mode in ("protected_core6", "protected_core26", "core6_only"):
            candidates &= self._coarse_core(coarse64, fusion_mode)
        pre_air_guard = int(candidates.sum())
        if fusion_mode.startswith("protected_"):
            candidates &= self._four_direction_behind_surface(fine)

        added = int(candidates.sum())
        if added > int(max_added_voxels):
            raise RuntimeError(
                f"Fusion would add {added} tokens (max_added_voxels={max_added_voxels}); "
                "nothing was truncated."
            )
        fused = fine | candidates
        assert np.all(fused[fine])  # never remove any fine cell
        report = {
            "mode": fusion_mode,
            "ss32_voxels": int(coarse32.sum()),
            "ss32_expanded_voxels": coarse_n,
            "ss64_voxels": fine_n,
            "ss64_inside_ss32_expanded": in_coarse,
            "ss64_parent_agreement_percent": round(agreement, 3),
            "candidate_before_air_guard": pre_air_guard,
            "added_voxels": added,
            "fused_voxels": int(fused.sum()),
            "candidate_rejected_by_air_guard": pre_air_guard - added,
            "limits": "SS32 support + optional coarse erosion + cardinal X/Z air exclusion",
            "notes": "NOT exact camera-space visual hull; run mesh tests before drawing conclusions",
        }
        if verbose:
            print("[Orbit3D SS32+SS64 FUSE] " + json.dumps(report, ensure_ascii=False), flush=True)
        return (
            self._coords(fused), 64, self._coords(candidates),
            added, int(fused.sum()), json.dumps(report, ensure_ascii=False, indent=2),
        )
