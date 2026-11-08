from __future__ import annotations

from pathlib import Path

import copy
import gc
import time

import torch
from tqdm import tqdm

import meshlib.mrmeshnumpy as mrmeshnumpy
import meshlib.mrmeshpy as mrmeshpy

from comfy.utils import ProgressBar


# --- ORBIT3D PATCH 3.6.16 MESHLIB DIAGNOSTICS BEGIN ---
def _orbit3d_meshlib_proc_mem():
    # Linux-only diagnostic snapshot; never changes geometry or allocator state.
    rss_kib = anon_kib = None
    avail_kib = None
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                rss_kib = int(line.split()[1])
            elif line.startswith("RssAnon:"):
                anon_kib = int(line.split()[1])
    except Exception:
        pass
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                avail_kib = int(line.split()[1])
                break
    except Exception:
        pass

    def gib(kib):
        return None if kib is None else kib / (1024.0 * 1024.0)

    return {
        "rss_gib": gib(rss_kib),
        "anon_gib": gib(anon_kib),
        "available_gib": gib(avail_kib),
    }


def _orbit3d_meshlib_heap_gib(mesh):
    try:
        return float(mesh.heapBytes()) / (1024.0 ** 3)
    except Exception:
        return None


def _orbit3d_fmt_gib(value):
    return "n/a" if value is None else f"{value:.3f} GiB"


def _orbit3d_log_meshlib_mem(prefix, mesh):
    mem = _orbit3d_meshlib_proc_mem()
    heap = _orbit3d_meshlib_heap_gib(mesh)
    print(
        f"[Orbit3D MeshLib Diag] {prefix}; "
        f"RSS={_orbit3d_fmt_gib(mem['rss_gib'])}; "
        f"RssAnon={_orbit3d_fmt_gib(mem['anon_gib'])}; "
        f"MemAvailable={_orbit3d_fmt_gib(mem['available_gib'])}; "
        f"Mesh.heapBytes={_orbit3d_fmt_gib(heap)}",
        flush=True,
    )


def _orbit3d_scan_hole_complexity(mesh, hole_edges, top_k=32, first_n=128):
    """
    Read-only scan of the INITIAL hole set.

    For every representative edge:
      * trackRightBoundaryLoop() gives combinatorial boundary size
      * holePerimeter() gives geometric size

    The returned list is indexed exactly like hole_edges and is used only for
    diagnostics. No fill order, topology, geometry, metric or settings change.
    """
    import heapq
    import math

    n = len(hole_edges)
    records = [None] * n
    top_edges = []
    total_edges = 0
    edge_ok = 0
    edge_failed = 0
    perimeter_ok = 0
    perimeter_failed = 0
    total_perimeter = 0.0
    max_edges = None
    max_perimeter = None
    thresholds = {10: 0, 100: 0, 1000: 0, 10000: 0, 100000: 0}

    print(
        f"[Orbit3D MeshLib Diag] scanning initial hole complexity: "
        f"count={n:,}; read-only",
        flush=True,
    )

    for i, e in enumerate(hole_edges):
        edge_count = None
        perimeter = None

        try:
            loop = mrmeshpy.trackRightBoundaryLoop(mesh.topology, e)
            edge_count = int(len(loop))
            del loop
            edge_ok += 1
            total_edges += edge_count
            max_edges = edge_count if max_edges is None else max(max_edges, edge_count)
            for threshold in thresholds:
                if edge_count >= threshold:
                    thresholds[threshold] += 1
        except Exception as exc:
            edge_failed += 1
            if edge_failed <= 8:
                print(
                    f"[Orbit3D MeshLib Diag] boundary-loop read FAILED "
                    f"hole={i}: {type(exc).__name__}: {exc}",
                    flush=True,
                )

        try:
            perimeter = float(mesh.holePerimeter(e))
            if not math.isfinite(perimeter):
                raise ValueError(f"non-finite perimeter {perimeter!r}")
            perimeter_ok += 1
            total_perimeter += perimeter
            max_perimeter = (
                perimeter if max_perimeter is None else max(max_perimeter, perimeter)
            )
        except Exception as exc:
            perimeter_failed += 1
            if perimeter_failed <= 8:
                print(
                    f"[Orbit3D MeshLib Diag] perimeter read FAILED "
                    f"hole={i}: {type(exc).__name__}: {exc}",
                    flush=True,
                )

        records[i] = (edge_count, perimeter)

        if edge_count is not None:
            # i is second so ties never compare perimeter(None/float).
            item = (edge_count, i, perimeter)
            if len(top_edges) < top_k:
                heapq.heappush(top_edges, item)
            elif item > top_edges[0]:
                heapq.heapreplace(top_edges, item)

    mean_edges = (total_edges / edge_ok) if edge_ok else None
    mean_perimeter = (total_perimeter / perimeter_ok) if perimeter_ok else None

    print(
        f"[Orbit3D MeshLib Diag] complexity scan complete: "
        f"edges_ok={edge_ok:,}/{n:,}; edges_failed={edge_failed:,}; "
        f"mean_edges={mean_edges if mean_edges is not None else 'n/a'}; "
        f"max_edges={max_edges if max_edges is not None else 'n/a'}; "
        f"perimeter_ok={perimeter_ok:,}/{n:,}; perimeter_failed={perimeter_failed:,}; "
        f"mean_perimeter={mean_perimeter if mean_perimeter is not None else 'n/a'}; "
        f"max_perimeter={max_perimeter if max_perimeter is not None else 'n/a'}",
        flush=True,
    )

    print(
        "[Orbit3D MeshLib Diag] boundary-edge thresholds: "
        + ", ".join(f">={k}:{v:,}" for k, v in thresholds.items()),
        flush=True,
    )

    detail_n = min(first_n, n)
    print(
        f"[Orbit3D MeshLib Diag] first {detail_n} initial holes "
        f"(index edges perimeter):",
        flush=True,
    )
    for i in range(detail_n):
        edge_count, perimeter = records[i]
        print(
            f"[Orbit3D MeshLib Diag] HOLE_INIT index={i}; "
            f"edges={'ERR' if edge_count is None else edge_count}; "
            f"perimeter={'ERR' if perimeter is None else format(perimeter, '.9g')}",
            flush=True,
        )

    if top_edges:
        desc = sorted(top_edges, reverse=True)
        print(
            f"[Orbit3D MeshLib Diag] top {len(desc)} initial holes by boundary edges:",
            flush=True,
        )
        for rank, (edge_count, i, perimeter) in enumerate(desc, 1):
            print(
                f"[Orbit3D MeshLib Diag] HOLE_TOP rank={rank}; index={i}; "
                f"edges={edge_count}; "
                f"perimeter={'ERR' if perimeter is None else format(perimeter, '.9g')}",
                flush=True,
            )

    return records
# --- ORBIT3D PATCH 3.6.16 MESHLIB DIAGNOSTICS END ---

# --- ORBIT3D PATCH 3.6.17 MESHLIB CRASH-SAFE LOGGING BEGIN ---
def _orbit3d_crashlog_root():
    try:
        import folder_paths
        root = Path(folder_paths.get_output_directory())
    except Exception:
        # .../ComfyUI/custom_nodes/ComfyUI-OrbitTo3D/orbit3d_ext/file.py
        root = Path(__file__).resolve().parents[3] / "output"
    root = root / "orbit3d_meshlib_diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _orbit3d_crashlog_sync(fp):
    try:
        fp.flush()
        import os
        os.fsync(fp.fileno())
    except Exception:
        pass


def _orbit3d_crashlog_emit(run, message, *, stdout=True, fsync=False):
    # Diagnostic failure must never change repair behavior.
    if stdout:
        print(message, flush=True)
    if run is None:
        return
    fp = run.get("log_fp")
    if fp is None:
        return
    try:
        fp.write(message + "\n")
        fp.flush()  # survives process SIGKILL/OOM kill; kernel already owns bytes
        if fsync:
            import os
            os.fsync(fp.fileno())
    except Exception as exc:
        if not run.get("write_error_reported", False):
            run["write_error_reported"] = True
            print(
                "[Orbit3D MeshLib CrashLog] write FAILED: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )


def _orbit3d_crashlog_mem_line(prefix, mesh):
    mem = _orbit3d_meshlib_proc_mem()
    heap = _orbit3d_meshlib_heap_gib(mesh)
    return (
        f"[Orbit3D MeshLib Diag] {prefix}; "
        f"RSS={_orbit3d_fmt_gib(mem['rss_gib'])}; "
        f"RssAnon={_orbit3d_fmt_gib(mem['anon_gib'])}; "
        f"MemAvailable={_orbit3d_fmt_gib(mem['available_gib'])}; "
        f"Mesh.heapBytes={_orbit3d_fmt_gib(heap)}"
    )


def _orbit3d_crashlog_mem(run, prefix, mesh, *, stdout=True, fsync=False):
    _orbit3d_crashlog_emit(
        run,
        _orbit3d_crashlog_mem_line(prefix, mesh),
        stdout=stdout,
        fsync=fsync,
    )


def _orbit3d_crashlog_start(mesh_copy, ml_mesh, holes_found):
    import datetime
    import os

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = _orbit3d_crashlog_root() / f"meshlib_{stamp}_pid{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    log_path = run_dir / "run.log"
    fp = log_path.open("a", encoding="utf-8", buffering=1)
    run = {
        "dir": run_dir,
        "log_path": log_path,
        "holes_csv_path": run_dir / "holes_initial.csv",
        "summary_path": run_dir / "summary.json",
        "log_fp": fp,
        "write_error_reported": False,
    }

    _orbit3d_crashlog_emit(
        run,
        f"[Orbit3D MeshLib CrashLog] START run_dir={run_dir}",
        fsync=True,
    )

    try:
        vertices = mesh_copy.vertices
        faces = mesh_copy.faces
        coords = getattr(mesh_copy, "coords", None)
        attrs = getattr(mesh_copy, "attrs", None)
        _orbit3d_crashlog_emit(
            run,
            "[Orbit3D MeshLib CrashLog] INPUT "
            f"vertices={int(vertices.shape[0]):,}; "
            f"faces={int(faces.shape[0]):,}; "
            f"vertices_dtype={vertices.dtype}; "
            f"faces_dtype={faces.dtype}; "
            f"vertices_device={vertices.device}; "
            f"faces_device={faces.device}; "
            f"coords={'none' if coords is None else tuple(coords.shape)}; "
            f"attrs={'none' if attrs is None else tuple(attrs.shape)}; "
            f"voxel_shape={getattr(mesh_copy, 'voxel_shape', None)}; "
            f"holes_found={holes_found:,}",
            fsync=True,
        )
    except Exception as exc:
        _orbit3d_crashlog_emit(
            run,
            "[Orbit3D MeshLib CrashLog] INPUT metadata FAILED: "
            f"{type(exc).__name__}: {exc}",
            fsync=True,
        )

    _orbit3d_crashlog_mem(
        run,
        f"crashlog start; holes={holes_found:,}",
        ml_mesh,
        fsync=True,
    )
    return run


def _orbit3d_crashlog_write_initial_holes(run, records):
    # Pure Python post-processing of the already-computed 3.6.16 records.
    # No MeshLib topology/geometry calls are made here.
    if run is None or records is None:
        return

    import csv
    import heapq
    import json
    import math
    import os

    top_k = 64
    top_edges = []
    top_perimeter = []
    thresholds = {10: 0, 100: 0, 1000: 0, 10000: 0, 100000: 0}
    edge_ok = 0
    perimeter_ok = 0
    total_edges = 0
    total_perimeter = 0.0
    max_edges = None
    max_perimeter = None

    csv_path = run["holes_csv_path"]
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["hole_index", "boundary_edges", "perimeter"])

        for i, rec in enumerate(records):
            edge_count, perimeter = rec if rec is not None else (None, None)
            writer.writerow([
                i,
                "" if edge_count is None else int(edge_count),
                "" if perimeter is None else format(float(perimeter), ".17g"),
            ])

            if edge_count is not None:
                edge_count = int(edge_count)
                edge_ok += 1
                total_edges += edge_count
                max_edges = edge_count if max_edges is None else max(max_edges, edge_count)
                for threshold in thresholds:
                    if edge_count >= threshold:
                        thresholds[threshold] += 1
                item = (edge_count, i, perimeter)
                if len(top_edges) < top_k:
                    heapq.heappush(top_edges, item)
                elif item > top_edges[0]:
                    heapq.heapreplace(top_edges, item)

            if perimeter is not None and math.isfinite(float(perimeter)):
                perimeter = float(perimeter)
                perimeter_ok += 1
                total_perimeter += perimeter
                max_perimeter = (
                    perimeter if max_perimeter is None else max(max_perimeter, perimeter)
                )
                item = (perimeter, i, edge_count)
                if len(top_perimeter) < top_k:
                    heapq.heappush(top_perimeter, item)
                elif item > top_perimeter[0]:
                    heapq.heapreplace(top_perimeter, item)

        _orbit3d_crashlog_sync(fp)

    summary = {
        "holes_total": len(records),
        "edges_ok": edge_ok,
        "edges_failed": len(records) - edge_ok,
        "mean_edges": (total_edges / edge_ok) if edge_ok else None,
        "max_edges": max_edges,
        "perimeter_ok": perimeter_ok,
        "perimeter_failed": len(records) - perimeter_ok,
        "mean_perimeter": (total_perimeter / perimeter_ok) if perimeter_ok else None,
        "max_perimeter": max_perimeter,
        "edge_threshold_counts": {str(k): v for k, v in thresholds.items()},
        "top_by_edges": [
            {
                "rank": rank,
                "hole_index": i,
                "boundary_edges": edge_count,
                "perimeter": perimeter,
            }
            for rank, (edge_count, i, perimeter)
            in enumerate(sorted(top_edges, reverse=True), 1)
        ],
        "top_by_perimeter": [
            {
                "rank": rank,
                "hole_index": i,
                "boundary_edges": edge_count,
                "perimeter": perimeter,
            }
            for rank, (perimeter, i, edge_count)
            in enumerate(sorted(top_perimeter, reverse=True), 1)
        ],
    }

    summary_path = run["summary_path"]
    tmp = summary_path.with_suffix(summary_path.suffix + f".{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, sort_keys=True)
            fp.write("\n")
            _orbit3d_crashlog_sync(fp)
        os.replace(tmp, summary_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

    _orbit3d_crashlog_emit(
        run,
        "[Orbit3D MeshLib CrashLog] INITIAL TABLE READY "
        f"csv={csv_path}; summary={summary_path}; "
        f">=1000_edges={thresholds[1000]:,}; "
        f">=10000_edges={thresholds[10000]:,}; "
        f">=100000_edges={thresholds[100000]:,}",
        fsync=True,
    )
# --- ORBIT3D PATCH 3.6.17 MESHLIB CRASH-SAFE LOGGING END ---


class Orbit3DMeshLibFillHolesNicelyPack:
    """
    OrbitTo3D final mesh repair.

    One live MeshLib topology is used for the whole stage:
      1. import MESHWITHVOXEL once
      2. fill holes with MeshLib fillHoleNicely
      3. optionally remove disconnected components in MeshLib topology
      4. pack the SAME live mesh
      5. export dense vertices/faces back to MESHWITHVOXEL

    Critical invariant:
    never export/re-import between fill, filtering and pack.
    That path was observed to destroy the topology on high-poly Pixal3D meshes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "component_filter": (
                    ["none", "keep_largest", "min_area_ratio"],
                    {"default": "min_area_ratio"},
                ),
                "min_area_ratio": (
                    "FLOAT",
                    {
                        "default": 0.00001,
                        "min": 0.0,
                        "max": 0.1,
                        "step": 0.000001,
                        "tooltip": (
                            "Used only for min_area_ratio. "
                            "0.00001 = 0.001% of total mesh surface area; "
                            "0.0001 = 0.01%."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "MESHWITHVOXEL",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "mesh",
        "holes_found",
        "holes_processed",
        "valid_faces_before_pack",
        "valid_faces_after_pack",
        "valid_vertices_after_pack",
    )
    FUNCTION = "process"
    CATEGORY = "Orbit3D/Mesh"
    DESCRIPTION = (
        "Fill holes with MeshLib, optionally remove disconnected components, "
        "then pack the same live MeshLib topology before returning the mesh."
    )

    def process(self, mesh, component_filter, min_area_ratio):
        if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
            raise TypeError(
                "Expected MESHWITHVOXEL-like object with vertices and faces"
            )

        if not isinstance(mesh.vertices, torch.Tensor):
            raise TypeError("mesh.vertices must be torch.Tensor")
        if not isinstance(mesh.faces, torch.Tensor):
            raise TypeError("mesh.faces must be torch.Tensor")

        if mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3:
            raise ValueError(
                f"Expected vertices [V,3], got {tuple(mesh.vertices.shape)}"
            )
        if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
            raise ValueError(
                f"Expected triangular faces [F,3], got {tuple(mesh.faces.shape)}"
            )

        input_vertices = int(mesh.vertices.shape[0])
        input_faces = int(mesh.faces.shape[0])

        print(
            "[Orbit3D MeshLib Fill+Filter+Pack] Input: "
            f"{input_vertices:,} vertices / {input_faces:,} faces",
            flush=True,
        )

        # Preserve the surrounding MESHWITHVOXEL metadata contract.
        mesh_copy = copy.deepcopy(mesh)

        ml_mesh = mrmeshnumpy.meshFromFacesVerts(
            mesh_copy.faces.detach().clone().cpu().numpy(),
            mesh_copy.vertices.detach().clone().cpu().numpy(),
        )

        valid_faces_initial = int(ml_mesh.topology.getValidFaces().count())
        valid_vertices_initial = int(ml_mesh.topology.getValidVerts().count())

        print(
            "[Orbit3D MeshLib Fill+Filter+Pack] Initial topology: "
            f"valid_vertices={valid_vertices_initial:,}, "
            f"valid_faces={valid_faces_initial:,}",
            flush=True,
        )

        hole_edges = ml_mesh.topology.findHoleRepresentiveEdges()
        holes_found = int(len(hole_edges))
        holes_processed = 0

        print(
            f"[Orbit3D MeshLib Fill+Filter+Pack] {holes_found:,} holes found",
            flush=True,
        )

        # 3.6.17 crash-safe logging starts BEFORE the first dangerous fill.
        # The 3.6.16 MeshLib scan itself remains unchanged and read-only.
        _orbit3d_diag_run = _orbit3d_crashlog_start(
            mesh_copy,
            ml_mesh,
            holes_found,
        )
        _orbit3d_log_meshlib_mem(
            f"after findHoleRepresentiveEdges; holes={holes_found:,}",
            ml_mesh,
        )
        _orbit3d_crashlog_mem(
            _orbit3d_diag_run,
            f"after findHoleRepresentiveEdges; holes={holes_found:,}",
            ml_mesh,
            stdout=False,
        )
        _orbit3d_hole_diag = None
        if holes_found > 0:
            _orbit3d_hole_diag = _orbit3d_scan_hole_complexity(
                ml_mesh,
                hole_edges,
                top_k=32,
                first_n=128,
            )
            _orbit3d_crashlog_write_initial_holes(
                _orbit3d_diag_run,
                _orbit3d_hole_diag,
            )
            _orbit3d_log_meshlib_mem(
                "after read-only hole-complexity scan",
                ml_mesh,
            )
            _orbit3d_crashlog_mem(
                _orbit3d_diag_run,
                "after read-only hole-complexity scan",
                ml_mesh,
                stdout=False,
                fsync=True,
            )
            _orbit3d_crashlog_emit(
                _orbit3d_diag_run,
                "[Orbit3D MeshLib CrashLog] BEGIN FILL PHASE",
                fsync=True,
            )

        if holes_found > 0:
            progress_bar = tqdm(
                total=holes_found,
                desc="MeshLib fillHoleNicely",
            )
            comfy_pbar = ProgressBar(holes_found)

            for _orbit3d_hole_i, e in enumerate(hole_edges):
                # Repair mechanics below are deliberately unchanged from production.
                _orbit3d_initial_edges = None
                _orbit3d_initial_perim = None
                if (
                    _orbit3d_hole_diag is not None
                    and _orbit3d_hole_i < len(_orbit3d_hole_diag)
                ):
                    (
                        _orbit3d_initial_edges,
                        _orbit3d_initial_perim,
                    ) = _orbit3d_hole_diag[_orbit3d_hole_i]

                _orbit3d_suspicious = (
                    _orbit3d_initial_edges is not None
                    and _orbit3d_initial_edges >= 1000
                )

                _orbit3d_crashlog_emit(
                    _orbit3d_diag_run,
                    f"[Orbit3D MeshLib Diag] ENTER hole={_orbit3d_hole_i}/{holes_found}; "
                    f"initial_edges="
                    f"{'ERR' if _orbit3d_initial_edges is None else _orbit3d_initial_edges}; "
                    f"initial_perimeter="
                    f"{'ERR' if _orbit3d_initial_perim is None else format(_orbit3d_initial_perim, '.9g')}",
                    fsync=_orbit3d_suspicious,
                )

                _orbit3d_diag_detail = (
                    _orbit3d_hole_i < 128
                    or _orbit3d_suspicious
                    or (_orbit3d_hole_i > 0 and _orbit3d_hole_i % 1000 == 0)
                )
                if _orbit3d_diag_detail:
                    _orbit3d_log_meshlib_mem(
                        f"before metric hole={_orbit3d_hole_i}",
                        ml_mesh,
                    )
                    _orbit3d_crashlog_mem(
                        _orbit3d_diag_run,
                        f"before metric hole={_orbit3d_hole_i}",
                        ml_mesh,
                        stdout=False,
                        fsync=_orbit3d_suspicious,
                    )

                params = mrmeshpy.FillHoleNicelySettings()
                params.triangulateParams.metric = mrmeshpy.getMinAreaMetric(
                    ml_mesh
                )
                params.smoothCurvature = False

                _orbit3d_crashlog_emit(
                    _orbit3d_diag_run,
                    f"[Orbit3D MeshLib Diag] CALL fillHoleNicely "
                    f"hole={_orbit3d_hole_i}/{holes_found}",
                    fsync=_orbit3d_suspicious,
                )
                if _orbit3d_diag_detail:
                    _orbit3d_log_meshlib_mem(
                        f"after getMinAreaMetric hole={_orbit3d_hole_i}",
                        ml_mesh,
                    )
                    _orbit3d_crashlog_mem(
                        _orbit3d_diag_run,
                        f"after getMinAreaMetric hole={_orbit3d_hole_i}",
                        ml_mesh,
                        stdout=False,
                        fsync=_orbit3d_suspicious,
                    )

                mrmeshpy.fillHoleNicely(ml_mesh, e, params)
                holes_processed += 1

                # Persist DONE for every successful hole. Keep stdout sparse.
                _orbit3d_crashlog_emit(
                    _orbit3d_diag_run,
                    f"[Orbit3D MeshLib Diag] DONE fillHoleNicely "
                    f"hole={_orbit3d_hole_i}/{holes_found}",
                    stdout=_orbit3d_diag_detail,
                    fsync=_orbit3d_suspicious,
                )

                if _orbit3d_diag_detail:
                    _orbit3d_log_meshlib_mem(
                        f"after fill hole={_orbit3d_hole_i}",
                        ml_mesh,
                    )
                    _orbit3d_crashlog_mem(
                        _orbit3d_diag_run,
                        f"after fill hole={_orbit3d_hole_i}",
                        ml_mesh,
                        stdout=False,
                        fsync=_orbit3d_suspicious,
                    )

                progress_bar.update(1)
                comfy_pbar.update(1)

            progress_bar.close()

        components_before = int(
            mrmeshpy.MeshComponents.getNumComponents(ml_mesh)
        )
        total_area_before = float(ml_mesh.area())
        valid_faces_before_filter = int(
            ml_mesh.topology.getValidFaces().count()
        )

        print(
            "[Orbit3D MeshLib Fill+Filter+Pack] After fill: "
            f"components={components_before}, "
            f"valid_faces={valid_faces_before_filter:,}, "
            f"area={total_area_before:.9g}",
            flush=True,
        )

        faces_removed_by_filter = 0
        removed_area_ratio = 0.0
        components_after = components_before

        if component_filter != "none":
            valid_faces = ml_mesh.topology.getValidFaces()

            if component_filter == "keep_largest":
                keep_faces = mrmeshpy.MeshComponents.getLargestComponent(
                    ml_mesh
                )

            elif component_filter == "min_area_ratio":
                threshold_area = (
                    float(min_area_ratio) * total_area_before
                )
                keep_faces = (
                    mrmeshpy.MeshComponents.getLargeByAreaComponents(
                        ml_mesh,
                        threshold_area,
                        None,
                    )
                )

                if int(keep_faces.count()) == 0:
                    raise RuntimeError(
                        "MeshLib component threshold selected zero faces; "
                        "refusing to erase the mesh."
                    )
            else:
                raise ValueError(
                    f"Unknown component_filter: {component_filter}"
                )

            remove_faces = valid_faces - keep_faces
            faces_removed_by_filter = int(remove_faces.count())

            print(
                "[Orbit3D MeshLib Fill+Filter+Pack] Component filter "
                f"'{component_filter}': removing "
                f"{faces_removed_by_filter:,} faces",
                flush=True,
            )

            if faces_removed_by_filter > 0:
                ml_mesh.deleteFaces(remove_faces)

            components_after = int(
                mrmeshpy.MeshComponents.getNumComponents(ml_mesh)
            )
            area_after_filter = float(ml_mesh.area())

            removed_area_ratio = (
                0.0
                if total_area_before <= 0.0
                else max(
                    0.0,
                    min(
                        1.0,
                        (total_area_before - area_after_filter)
                        / total_area_before,
                    ),
                )
            )

            print(
                "[Orbit3D MeshLib Fill+Filter+Pack] Filter result: "
                f"components {components_before} -> {components_after}, "
                f"removed_area_ratio={removed_area_ratio:.9g}",
                flush=True,
            )

        valid_faces_before_pack = int(
            ml_mesh.topology.getValidFaces().count()
        )
        valid_vertices_before_pack = int(
            ml_mesh.topology.getValidVerts().count()
        )

        print(
            "[Orbit3D MeshLib Fill+Filter+Pack] Before pack: "
            f"valid_vertices={valid_vertices_before_pack:,}, "
            f"valid_faces={valid_faces_before_pack:,}",
            flush=True,
        )

        t0 = time.perf_counter()
        try:
            ml_mesh.pack(rearrangeTriangles=False)
        except TypeError:
            # Compatibility fallback for MeshLib bindings exposing only the
            # no-argument overload/default.
            ml_mesh.pack()
        pack_seconds = time.perf_counter() - t0

        valid_faces_after_pack = int(
            ml_mesh.topology.getValidFaces().count()
        )
        valid_vertices_after_pack = int(
            ml_mesh.topology.getValidVerts().count()
        )

        print(
            "[Orbit3D MeshLib Fill+Filter+Pack] After pack: "
            f"valid_vertices={valid_vertices_after_pack:,}, "
            f"valid_faces={valid_faces_after_pack:,}, "
            f"time={pack_seconds:.3f}s",
            flush=True,
        )

        if valid_faces_after_pack != valid_faces_before_pack:
            raise RuntimeError(
                "MeshLib pack changed VALID face count: "
                f"{valid_faces_before_pack:,} -> "
                f"{valid_faces_after_pack:,}. Refusing output."
            )

        new_vertices = mrmeshnumpy.getNumpyVerts(ml_mesh)
        new_faces = mrmeshnumpy.getNumpyFaces(ml_mesh.topology)

        output_vertices = int(new_vertices.shape[0])
        output_faces = int(new_faces.shape[0])

        if output_faces != valid_faces_after_pack:
            raise RuntimeError(
                "Packed NumPy face count differs from valid topology: "
                f"numpy={output_faces:,}, "
                f"valid={valid_faces_after_pack:,}"
            )

        if output_vertices != valid_vertices_after_pack:
            raise RuntimeError(
                "Packed NumPy vertex count differs from valid topology: "
                f"numpy={output_vertices:,}, "
                f"valid={valid_vertices_after_pack:,}"
            )

        del ml_mesh
        gc.collect()

        mesh_copy.vertices = torch.from_numpy(
            new_vertices
        ).float().to(mesh_copy.device)

        mesh_copy.faces = torch.from_numpy(
            new_faces
        ).int().to(mesh_copy.device)

        del new_vertices
        del new_faces
        gc.collect()

        print(
            "[Orbit3D MeshLib Fill+Filter+Pack] Done: "
            f"holes={holes_processed}/{holes_found}, "
            f"components={components_before}->{components_after}, "
            f"faces_removed_by_filter={faces_removed_by_filter:,}, "
            f"removed_area_ratio={removed_area_ratio:.9g}.",
            flush=True,
        )

        return (
            mesh_copy,
            holes_found,
            holes_processed,
            valid_faces_before_pack,
            valid_faces_after_pack,
            valid_vertices_after_pack,
        )


NODE_CLASS_MAPPINGS = {
    "Orbit3DMeshLibFillHolesNicelyPack":
        Orbit3DMeshLibFillHolesNicelyPack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Orbit3DMeshLibFillHolesNicelyPack":
        "Orbit3D - MeshLib Fill + Filter + Pack",
}
