from __future__ import annotations

import numpy as np
import torch


class Orbit3DSSForceFillExperimental:
    """Force-fill a sparse voxel occupancy after SS generation.

    This is a pure post-process on COORDS. It does NOT rerun Sparse Structure,
    Shape512, Shape1024, or change cameras/conditioning.

    Main method: axis-span voting.
    For each principal axis, every line is filled between its first and last
    occupied voxel. A voxel is added only if it is supported by at least a given
    number of axes (default: 2 of 3), which generally reduces indiscriminate filling compared with filling
    everything inside a bounding volume; it can still bridge distinct parts.
    """

    CATEGORY = "Orbit to 3D/Experimental"
    FUNCTION = "process"
    RETURN_TYPES = ("COORDS", "INT", "INT", "INT")
    RETURN_NAMES = ("coords", "sparse_structure_resolution", "added_voxels", "final_voxels")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coords": ("COORDS",),
                "sparse_structure_resolution": (
                    "INT",
                    {
                        "forceInput": True,
                        "tooltip": "Resolution of incoming SS coordinates. Designed primarily for 64.",
                    },
                ),
                "method": (
                    ["axis_vote", "axis_vote_then_close", "neck_limited"],
                    {
                        "default": "axis_vote",
                        "tooltip": "neck_limited: fill deep pockets accessible only through narrow openings; protect direct horizontal air rays. axis_vote: previous behavior; axis_vote_then_close: previous behavior plus closing.",
                    },
                ),
                "axis_support_threshold": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 3,
                        "step": 1,
                        "tooltip": "How many axis spans must include a voxel before it is added. 2 is the recommended safe default; 3 is stricter, 1 is aggressive.",
                    },
                ),
                "closing_iterations": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 3,
                        "step": 1,
                        "tooltip": "Used only by axis_vote_then_close.",
                    },
                ),
                "max_added_voxels": (
                    "INT",
                    {
                        "default": 25000,
                        "min": 0,
                        "max": 1000000,
                        "step": 100,
                        "tooltip": "Safety guard. Abort if more than this many voxels would be added.",
                    },
                ),
                "neck_clearance_voxels": (
                    "FLOAT",
                    {
                        "default": 2.0, "min": 1.25, "max": 5.0, "step": 0.25,
                        "tooltip": "neck_limited only: minimum clearance for an exterior rolling-ball probe, in SS voxels. Larger radii classify more air as unreachable and can fill genuine gaps.",
                    },
                ),
                "protect_horizontal_air": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "neck_limited only: never add a voxel directly visible from any X or Z axis direction in the ORIGINAL occupancy; does not use camera calibration.",
                    },
                ),
                "min_pocket_voxels": (
                    "INT",
                    {
                        "default": 64, "min": 1, "max": 100000, "step": 1,
                        "tooltip": "neck_limited only: ignore small disconnected candidate pockets after horizontal-visibility protection. Larger values reduce tiny false bridges but may miss small real cavities.",
                    },
                ),
                "verbose": ("BOOLEAN", {"default": True}),
            }
        }

    @staticmethod
    def _coords_to_grid(coords: torch.Tensor, R: int) -> np.ndarray:
        grid = np.zeros((R, R, R), dtype=np.bool_)
        xyz = coords[:, 1:4].detach().cpu().numpy().astype(np.int64)
        grid[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = True
        return grid

    @staticmethod
    def _grid_to_coords(grid: np.ndarray) -> torch.Tensor:
        idx = np.argwhere(grid)
        if idx.size == 0:
            return torch.zeros((0, 4), dtype=torch.int32)
        batch = np.zeros((idx.shape[0], 1), dtype=np.int32)
        out = np.concatenate([batch, idx.astype(np.int32)], axis=1)
        return torch.from_numpy(np.ascontiguousarray(out))

    @staticmethod
    def _axis_span_votes(grid: np.ndarray) -> np.ndarray:
        R = grid.shape[0]
        votes = np.zeros_like(grid, dtype=np.uint8)

        # X-lines for each (y,z)
        for y in range(R):
            for z in range(R):
                xs = np.flatnonzero(grid[:, y, z])
                if xs.size >= 2:
                    votes[xs[0]:xs[-1] + 1, y, z] += 1

        # Y-lines for each (x,z)
        for x in range(R):
            for z in range(R):
                ys = np.flatnonzero(grid[x, :, z])
                if ys.size >= 2:
                    votes[x, ys[0]:ys[-1] + 1, z] += 1

        # Z-lines for each (x,y)
        for x in range(R):
            for y in range(R):
                zs = np.flatnonzero(grid[x, y, :])
                if zs.size >= 2:
                    votes[x, y, zs[0]:zs[-1] + 1] += 1

        return votes

    @staticmethod
    def _neck_limited_candidates(
        grid: np.ndarray,
        clearance_voxels: float,
        protect_horizontal_air: bool,
        min_pocket_voxels: int,
    ) -> tuple[np.ndarray, dict]:
        """Rolling-probe accessibility, with a conservative horizontal-ray veto.

        This is a geometry-only hypothesis test, NOT a semantic cavity detector.
        Outside air is connected through 6-neighbor centers whose EDT clearance
        permits a probe of the requested radius. The probe's reachable swept
        volume is reconstructed by Euclidean dilation. Deep regions not reached
        by the probe become filling candidates.

        The horizontal ray veto suppresses candidates that are openly visible
        along +/-X or +/-Z in the ORIGINAL SS; vertical Y openings are left
        unprotected because the test targets unknown top/bottom cavities.
        """
        try:
            from scipy.ndimage import distance_transform_edt, label, generate_binary_structure
        except ImportError as exc:
            raise RuntimeError("neck_limited requires scipy.ndimage") from exc

        if not np.isfinite(clearance_voxels) or not 1.25 <= clearance_voxels <= 5.0:
            raise ValueError("neck_clearance_voxels must be between 1.25 and 5")
        if min_pocket_voxels < 1:
            raise ValueError("min_pocket_voxels must be >= 1")

        R = int(grid.shape[0])
        pad = max(7, int(np.ceil(clearance_voxels)) + 3)
        occ = np.pad(grid, pad, mode="constant", constant_values=False)
        air = ~occ
        air_clearance = distance_transform_edt(air)
        traversable_centers = air & (air_clearance >= float(clearance_voxels))

        cc6 = generate_binary_structure(3, 1)
        components, component_count = label(traversable_centers, structure=cc6)
        exterior_labels = np.unique(np.concatenate([
            components[0].ravel(), components[-1].ravel(),
            components[:, 0].ravel(), components[:, -1].ravel(),
            components[:, :, 0].ravel(), components[:, :, -1].ravel(),
        ]))
        exterior_labels = exterior_labels[exterior_labels != 0]
        exterior_centers = np.isin(components, exterior_labels)
        if not np.any(exterior_centers):
            raise RuntimeError("No exterior air reached by the rolling probe")

        # Sweep the rolling probe around each reachable center, recovering the
        # exterior-accessible air up to the probe radius. Use the same radius
        # in both erosion (clearance) and reconstruction (dilation).
        swept_exterior_air = (distance_transform_edt(~exterior_centers)
                              < float(clearance_voxels)) & air
        slices = (slice(pad, pad + R),) * 3
        enclosed_by_probe = (air & ~swept_exterior_air)[slices]
        raw_candidate_count = int(enclosed_by_probe.sum())

        if protect_horizontal_air:
            # Direct horizontal rays through any exterior opening are a
            # conservative approximation to turntable silhouette protection.
            # This is evaluated BEFORE filling, so added voxels cannot shield
            # later candidates from their original visibility.
            x_cumulative = np.cumsum(grid, axis=0)
            z_cumulative = np.cumsum(grid, axis=2)
            enclosed_x = ((x_cumulative - grid) > 0) & ((x_cumulative[-1:] - x_cumulative) > 0)
            enclosed_z = ((z_cumulative - grid) > 0) & ((z_cumulative[:, :, -1:] - z_cumulative) > 0)
            enclosed_by_probe &= (enclosed_x & enclosed_z)

        after_visibility_count = int(enclosed_by_probe.sum())
        pocket_labels, pocket_count = label(enclosed_by_probe, structure=cc6)
        sizes = np.bincount(pocket_labels.ravel(), minlength=pocket_count + 1)
        keep = sizes >= int(min_pocket_voxels)
        keep[0] = False
        candidates = keep[pocket_labels]
        kept_components = int(np.count_nonzero(keep[1:]))
        return candidates, {
            "radius": float(clearance_voxels),
            "core_components": int(component_count),
            "candidate_before_horizontal_guard": raw_candidate_count,
            "candidate_after_horizontal_guard": after_visibility_count,
            "candidate_after_pocket_filter": int(candidates.sum()),
            "kept_pockets": kept_components,
            "protect_horizontal_air": bool(protect_horizontal_air),
        }

    def process(
        self,
        coords,
        sparse_structure_resolution,
        method="axis_vote",
        axis_support_threshold=2,
        closing_iterations=1,
        max_added_voxels=25000,
        neck_clearance_voxels=2.0,
        protect_horizontal_air=True,
        min_pocket_voxels=64,
        verbose=True,
    ):
        if not isinstance(coords, torch.Tensor) or coords.ndim != 2 or coords.shape[1] != 4:
            raise TypeError(f"Expected COORDS tensor [N,4], got {type(coords)} with shape {getattr(coords, 'shape', None)}")

        R = int(sparse_structure_resolution)
        if R < 8:
            raise ValueError(f"Unsupported sparse_structure_resolution={R}")

        thresh = int(axis_support_threshold)
        if thresh < 1 or thresh > 3:
            raise ValueError("axis_support_threshold must be in [1,3]")

        if coords.is_floating_point() or coords.is_complex():
            raise TypeError("Sparse COORDS must contain integer coordinates")
        if coords.shape[0] == 0:
            raise ValueError("Sparse COORDS are empty")
        coords_cpu = coords.detach().to(device="cpu", dtype=torch.int64)
        if bool((coords_cpu[:, 0] != 0).any()):
            raise ValueError("Only batch index 0 is supported")
        if bool((coords_cpu[:, 1:] < 0).any()) or bool((coords_cpu[:, 1:] >= R).any()):
            raise ValueError(f"Spatial coordinates must be inside [0,{R-1}]^3")
        if int(torch.unique(coords_cpu, dim=0).shape[0]) != int(coords_cpu.shape[0]):
            raise ValueError("Duplicate sparse coordinates detected")
        if method not in ("axis_vote", "axis_vote_then_close", "neck_limited"):
            raise ValueError(f"Unknown fill method: {method!r}")
        if not 1 <= int(closing_iterations) <= 3:
            raise ValueError("closing_iterations must be in [1,3]")

        grid = self._coords_to_grid(coords_cpu, R)
        before = int(grid.sum())
        votes = None
        diagnostics = None
        if method == "neck_limited":
            filled, diagnostics = self._neck_limited_candidates(
                grid,
                float(neck_clearance_voxels),
                bool(protect_horizontal_air),
                int(min_pocket_voxels),
            )
        else:
            votes = self._axis_span_votes(grid)
            filled = votes >= thresh
        merged = grid | filled

        if method == "axis_vote_then_close":
            try:
                from scipy.ndimage import binary_closing
            except Exception as e:
                raise RuntimeError("axis_vote_then_close requires scipy") from e
            # scipy.binary_closing erodes near volume boundaries. The experiment
            # must never silently delete SS input tokens, so explicitly preserve
            # all original and axis-voted cells.
            closed = binary_closing(
                merged,
                structure=np.ones((3, 3, 3), dtype=bool),
                iterations=int(closing_iterations),
            )
            merged |= closed

        added = merged & ~grid
        added_count = int(added.sum())
        after = int(merged.sum())

        if max_added_voxels >= 0 and added_count > int(max_added_voxels):
            raise RuntimeError(
                f"Force fill would add {added_count} voxels, exceeding max_added_voxels={int(max_added_voxels)}. "
                f"No output was produced."
            )

        out_coords = self._grid_to_coords(merged)
        if int(out_coords.shape[0]) != after or after < before:
            raise RuntimeError("Force-fill violated the no-token-loss invariant")

        if verbose:
            print(
                f"[Orbit3D SS Force Fill EXP] R={R}, method={method}, threshold={thresh}; "
                f"before={before}, added={added_count}, after={after}; "
                f"growth={100.0*added_count/max(before,1):.1f}%",
                flush=True,
            )
            # Diagnostic histogram of vote strengths among added voxels.
            if method == "neck_limited":
                print(f"[Orbit3D SS Force Fill EXP] neck diagnostics: {diagnostics}", flush=True)
            elif added_count > 0:
                vals = votes[added]
                hist = {int(k): int((vals == k).sum()) for k in (1, 2, 3)}
                print(f"[Orbit3D SS Force Fill EXP] added-vote histogram: {hist}", flush=True)

        return out_coords, R, added_count, after
