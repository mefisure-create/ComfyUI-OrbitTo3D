from __future__ import annotations

import copy
import gc
import os
import re
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import folder_paths

from ..trellis2.representations import MeshWithVoxel

try:
    import mrmeshpy
    import mrmeshnumpy
except ImportError:
    from meshlib import mrmeshpy, mrmeshnumpy


# --- ORBIT3D PATCH 3.6.19 PMP STREAMING BEGIN ---
_PMP_MAGIC_IN = b"O3DPMP01"
_PMP_MAGIC_OUT = b"O3DPMPO1"
_PMP_VERSION = 2
_PMP_STREAM_BATCH_LOOPS = 2048
_PMP_STREAM_BATCH_EDGES = 65536
_PMP_HARD_SKIP_EDGES = 50000


def _proc_mem_gib():
    result = {"rss": None, "anon": None, "available": None}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                result["rss"] = int(line.split()[1]) / (1024.0 * 1024.0)
            elif line.startswith("RssAnon:"):
                result["anon"] = int(line.split()[1]) / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                result["available"] = int(line.split()[1]) / (1024.0 * 1024.0)
                break
    except Exception:
        pass
    return result


def _fmt_mem(prefix: str) -> str:
    m = _proc_mem_gib()
    def f(v):
        return "n/a" if v is None else f"{v:.3f} GiB"
    return (
        f"{prefix}; RSS={f(m['rss'])}; RssAnon={f(m['anon'])}; "
        f"MemAvailable={f(m['available'])}"
    )


def _id_to_int(value) -> int:
    for name in ("get", "value", "id"):
        attr = getattr(value, name, None)
        if callable(attr):
            try:
                return int(attr())
            except Exception:
                pass
        elif attr is not None:
            try:
                return int(attr)
            except Exception:
                pass
    try:
        return int(value)
    except Exception:
        pass
    text = str(value)
    m = re.search(r"-?\d+", text)
    if m:
        return int(m.group(0))
    raise TypeError(f"Cannot convert MeshLib id to int: {value!r}")


def _build_pmp_boundary_context(mesh):
    """
    Build MeshLib topology once, but retain only scalar metadata per hole.
    No point payloads and no third_points are retained for all 86k loops.
    """
    vertices = np.asarray(
        mesh.vertices.detach().cpu().numpy(), dtype=np.float32
    )
    faces = np.asarray(
        mesh.faces.detach().cpu().numpy(), dtype=np.int32
    )

    print(_fmt_mem("[Orbit3D PMP] before MeshLib boundary build"), flush=True)
    ml_mesh = mrmeshnumpy.meshFromFacesVerts(faces, vertices)
    reps = list(ml_mesh.topology.findHoleRepresentiveEdges())
    print(
        f"[Orbit3D PMP] boundary loops found={len(reps):,}; "
        + _fmt_mem("after boundary build"),
        flush=True,
    )

    topo = ml_mesh.topology
    meta = []
    total_edges = 0
    max_edges = 0

    for hole_index, rep in enumerate(reps):
        # MeshLib documents findHoleRepresentiveEdges() as returning one edge
        # with no valid LEFT face per hole; use the same right-boundary
        # traversal that completed the full 3.6.17 diagnostic scan.
        loop = mrmeshpy.trackRightBoundaryLoop(topo, rep)
        n = int(len(loop))
        del loop
        if n < 3:
            continue

        perimeter = float(ml_mesh.holePerimeter(rep))
        meta.append((int(hole_index), n, perimeter))
        total_edges += n
        max_edges = max(max_edges, n)

        if (hole_index + 1) % 1000 == 0:
            print(
                f"[Orbit3D PMP] scan {hole_index + 1:,}/{len(reps):,}; "
                f"running_edges={total_edges:,}; max_edges={max_edges:,}; "
                + _fmt_mem("scan memory"),
                flush=True,
            )

    print(
        f"[Orbit3D PMP] boundary scan complete loops={len(meta):,}; "
        f"total_edges={total_edges:,}; max_edges={max_edges:,}; "
        + _fmt_mem("after lightweight scan"),
        flush=True,
    )
    return vertices, faces, ml_mesh, reps, meta



def _id_is_valid(value) -> bool:
    valid = getattr(value, "valid", None)
    if callable(valid):
        try:
            return bool(valid())
        except Exception:
            pass
    try:
        return _id_to_int(value) >= 0
    except Exception:
        return False


def _map_boundary_org_to_input_vertex(
    vertices,
    faces,
    ml_mesh,
    topo,
    edge,
    hole_index,
    edge_index,
):
    """
    Map a MeshLib topology vertex back to an ORIGINAL MeshWithVoxel vertex id.

    meshFromFacesVerts() may duplicate topology vertices while resolving
    non-manifold triangle soup. The unique adjacent face restricts the lookup
    to the three original input vertices of that triangle; the MeshLib origin
    point is then coordinate-matched to one of those three vertices.
    """
    raw_vid = _id_to_int(topo.org(edge))
    p = ml_mesh.orgPnt(edge)
    px = float(p.x)
    py = float(p.y)
    pz = float(p.z)

    left = topo.left(edge)
    right = topo.right(edge)

    valid_faces = []
    for side, face_obj in (("left", left), ("right", right)):
        if not _id_is_valid(face_obj):
            continue
        fid = _id_to_int(face_obj)
        if 0 <= fid < int(faces.shape[0]):
            valid_faces.append((side, fid))

    if len(valid_faces) != 1:
        raise RuntimeError(
            f"Boundary loop {hole_index} edge {edge_index}: expected exactly "
            f"one valid adjacent input face, got {valid_faces}; raw_vid={raw_vid}"
        )

    side, fid = valid_faces[0]
    tri = faces[fid]

    best_orig = -1
    best_d2 = float("inf")
    d2s = []

    for j in range(3):
        orig_vid = int(tri[j])
        if orig_vid < 0 or orig_vid >= int(vertices.shape[0]):
            raise RuntimeError(
                f"Input face {fid} contains invalid vertex id {orig_vid}"
            )

        q = vertices[orig_vid]
        dx = float(q[0]) - px
        dy = float(q[1]) - py
        dz = float(q[2]) - pz
        d2 = dx * dx + dy * dy + dz * dz
        d2s.append(d2)

        if d2 < best_d2:
            best_d2 = d2
            best_orig = orig_vid

    # Source coordinates and MeshLib points are float32-derived.
    # 1e-6 positional tolerance is tiny in the normalized ~[-0.5,0.5] mesh.
    if best_d2 > 1.0e-12:
        raise RuntimeError(
            f"Boundary loop {hole_index} edge {edge_index}: MeshLib org point "
            f"does not match any vertex of adjacent input face {fid} ({side}); "
            f"raw_vid={raw_vid}; tri={list(map(int, tri))}; d2={d2s}"
        )

    return best_orig, raw_vid, best_d2


def _extract_pmp_record(
    vertices,
    faces,
    ml_mesh,
    rep,
    hole_index,
    expected_edges,
    perimeter,
):
    """
    Materialize exactly ONE boundary loop on demand.

    MeshLib topology vertex ids are remapped through each boundary edge's
    adjacent ORIGINAL input triangle. This handles duplicate topology vertices
    without reindexing or replacing the source MeshWithVoxel.
    """
    topo = ml_mesh.topology
    loop = mrmeshpy.trackRightBoundaryLoop(topo, rep)
    n = int(len(loop))
    if n != int(expected_edges):
        raise RuntimeError(
            f"Boundary loop {hole_index} changed during read-only staging: "
            f"scan={expected_edges}, extract={n}"
        )

    ids = np.empty((n,), dtype=np.uint64)
    points = np.empty((n, 3), dtype=np.float64)

    remapped = 0
    raw_out_of_range = 0
    max_raw_vid = -1
    max_match_error = 0.0

    for i, edge in enumerate(loop):
        orig_vid, raw_vid, d2 = _map_boundary_org_to_input_vertex(
            vertices,
            faces,
            ml_mesh,
            topo,
            edge,
            hole_index,
            i,
        )
        ids[i] = orig_vid
        points[i] = vertices[orig_vid]

        max_raw_vid = max(max_raw_vid, raw_vid)
        max_match_error = max(max_match_error, d2)
        if raw_vid != orig_vid:
            remapped += 1
        if raw_vid < 0 or raw_vid >= int(vertices.shape[0]):
            raw_out_of_range += 1

    del loop

    adjacent_duplicates = 0
    if n:
        adjacent_duplicates = int(np.count_nonzero(ids[1:] == ids[:-1]))
        if int(ids[0]) == int(ids[-1]):
            adjacent_duplicates += 1

    if remapped or n >= 1000:
        print(
            f"[Orbit3D PMP] mapped hole={hole_index} edges={n:,}; "
            f"remapped_topology_vertices={remapped:,}; "
            f"raw_out_of_range={raw_out_of_range:,}; "
            f"max_raw_vid={max_raw_vid:,}; "
            f"adjacent_duplicate_source_ids={adjacent_duplicates:,}; "
            f"max_match_error={max_match_error ** 0.5:.3e}",
            flush=True,
        )

    return {
        "hole_index": int(hole_index),
        "ids": ids,
        "points": points,
        "third_points": None,
        "edges": int(n),
        "perimeter": float(perimeter),
    }


def _write_pmp_input(path: Path, records):
    """
    Protocol v2 uses block I/O instead of struct.pack per boundary vertex.

    Record:
      uint64 hole_index
      uint64 edge_count
      uint32 flags (bit0: third_points present)
      uint64 ids[n]
      float64 points[n][3]
      optional float64 third_points[n][3]
    """
    with path.open("wb") as fp:
        fp.write(_PMP_MAGIC_IN)
        fp.write(struct.pack("<IQ", _PMP_VERSION, len(records)))

        for rec in records:
            n = int(rec["edges"])
            thirds = rec.get("third_points")
            flags = 1 if thirds is not None else 0
            fp.write(struct.pack("<QQI", int(rec["hole_index"]), n, flags))

            ids = np.asarray(rec["ids"], dtype="<u8")
            pts = np.asarray(rec["points"], dtype="<f8").reshape(n, 3)
            ids.tofile(fp)
            pts.tofile(fp)

            if thirds is not None:
                np.asarray(thirds, dtype="<f8").reshape(n, 3).tofile(fp)

        fp.flush()
        os.fsync(fp.fileno())


def _read_pmp_output(path: Path):
    results = []
    with path.open("rb") as fp:
        magic = fp.read(8)
        if magic != _PMP_MAGIC_OUT:
            raise RuntimeError(f"Bad PMP helper output magic: {magic!r}")
        raw = fp.read(12)
        if len(raw) != 12:
            raise RuntimeError("Truncated PMP helper output header")
        version, count = struct.unpack("<IQ", raw)
        if version != _PMP_VERSION:
            raise RuntimeError(f"Unsupported PMP output version {version}")

        for _ in range(count):
            hdr = fp.read(36)
            if len(hdr) != 36:
                raise RuntimeError("Truncated PMP output record header")
            hole_index, edges, status, tri_count, elapsed = struct.unpack(
                "<QQiQd", hdr
            )
            faces = None
            if tri_count:
                raw_faces = fp.read(tri_count * 3 * 8)
                if len(raw_faces) != tri_count * 3 * 8:
                    raise RuntimeError("Truncated PMP output triangle payload")
                faces = (
                    np.frombuffer(raw_faces, dtype="<u8")
                    .reshape(-1, 3)
                    .copy()
                )
            results.append(
                {
                    "hole_index": int(hole_index),
                    "edges": int(edges),
                    "status": int(status),
                    "faces": faces,
                    "elapsed": float(elapsed),
                }
            )
    return results


def _pmp_helper_path() -> Path:
    return Path(__file__).resolve().parent / "bin" / "orbit3d_pmp_polyline"


def _run_pmp_helper(records, threads: int, timeout_s: int, temp_root: Path):
    helper = _pmp_helper_path()
    if not helper.is_file() or not os.access(helper, os.X_OK):
        raise RuntimeError(
            "CGAL PMP helper is not built. See docs/INSTALLATION.md for local CMake build instructions."
        )

    token = f"pmp_{os.getpid()}_{time.time_ns()}"
    in_path = temp_root / f"{token}.in.bin"
    out_path = temp_root / f"{token}.out.bin"
    _write_pmp_input(in_path, records)

    cmd = [
        str(helper),
        str(in_path),
        str(out_path),
        str(max(1, int(threads))),
    ]
    start = time.perf_counter()
    try:
        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=max(1, int(timeout_s)),
            check=False,
        )
        elapsed = time.perf_counter() - start
        if cp.stdout:
            print(cp.stdout.rstrip(), flush=True)

        if cp.returncode != 0:
            return [], elapsed, False, int(cp.returncode)

        return _read_pmp_output(out_path), elapsed, False, 0

    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        if exc.stdout:
            text = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout
            print(text.rstrip(), flush=True)
        return [], elapsed, True, None

    finally:
        for p in (in_path, out_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass


def _append_result_faces(results, face_chunks, counters):
    for result in results:
        if result["status"] == 0 and result["faces"] is not None:
            counters["success"] += 1
            if len(result["faces"]):
                face_chunks.append(result["faces"])
        else:
            counters["failed"] += 1


class Orbit3DPMPPolylineHoleFillExperimental:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "threads": (
                    "INT",
                    {"default": 8, "min": 1, "max": 64, "step": 1},
                ),
                "isolate_edges": (
                    "INT",
                    {"default": 10000, "min": 100, "max": 1000000, "step": 100},
                ),
                "isolated_timeout_s": (
                    "INT",
                    {"default": 120, "min": 5, "max": 3600, "step": 5},
                ),
                "batch_timeout_s": (
                    "INT",
                    {"default": 900, "min": 30, "max": 7200, "step": 30},
                ),
            }
        }

    RETURN_TYPES = ("MESHWITHVOXEL", "STRING")
    RETURN_NAMES = ("mesh", "report")
    FUNCTION = "fill"
    CATEGORY = "Orbit3D/Mesh Repair/Experimental"
    DESCRIPTION = (
        "Experimental streaming boundary-only CGAL PMP triangulate_hole_polyline "
        "backend. Original vertices/faces are preserved; successful patch "
        "triangles are appended."
    )

    def fill(self, mesh, threads, isolate_edges, isolated_timeout_s, batch_timeout_s):
        if getattr(mesh, "coords", None) is not None or getattr(mesh, "attrs", None) is not None:
            raise RuntimeError(
                "PMP experimental node currently supports mesh-only MeshWithVoxel "
                "checkpoints (coords/attrs must be None)."
            )

        t0 = time.perf_counter()
        vertices, faces, ml_mesh, reps, meta = _build_pmp_boundary_context(mesh)
        scan_s = time.perf_counter() - t0
        total = len(meta)

        if total == 0:
            report = "Orbit3D PMP: no boundary loops found; mesh returned unchanged."
            print(report, flush=True)
            return (mesh, report)

        temp_root = Path(folder_paths.get_temp_directory()) / "orbit3d_pmp"
        temp_root.mkdir(parents=True, exist_ok=True)

        isolate_edges = int(isolate_edges)
        isolated_meta = [m for m in meta if m[1] >= isolate_edges]
        regular_meta = [m for m in meta if m[1] < isolate_edges]
        isolated_meta.sort(key=lambda m: m[1])

        regular_total = len(regular_meta)
        isolated_total = len(isolated_meta)

        face_chunks = []
        counters = {"success": 0, "failed": 0}
        timed_out = []
        helper_failed = []
        hard_skipped = []
        batch_timeouts = 0
        batch_failures = 0
        isolated_lines = []

        # Ordinary holes FIRST.
        batch_records = []
        batch_edges = 0
        batch_index = 0
        processed_regular = 0

        def flush_batch():
            nonlocal batch_records, batch_edges, batch_index
            nonlocal batch_timeouts, batch_failures, processed_regular

            if not batch_records:
                return

            batch_index += 1
            batch_loop_count = len(batch_records)
            first_hole = batch_records[0]["hole_index"]
            last_hole = batch_records[-1]["hole_index"]

            print(
                f"[Orbit3D PMP] batch={batch_index} RUN "
                f"loops={batch_loop_count:,} edges={batch_edges:,} "
                f"holes={first_hole}..{last_hole}; "
                + _fmt_mem("before helper"),
                flush=True,
            )

            results, sec, timeout, rc = _run_pmp_helper(
                batch_records,
                int(threads),
                int(batch_timeout_s),
                temp_root,
            )
            processed_regular += batch_loop_count

            if timeout:
                batch_timeouts += 1
                counters["failed"] += batch_loop_count
                print(
                    f"[Orbit3D PMP] batch={batch_index} TIMEOUT "
                    f"wall={sec:.3f}s loops_unfilled={batch_loop_count:,}",
                    flush=True,
                )
            elif rc != 0:
                batch_failures += 1
                counters["failed"] += batch_loop_count
                print(
                    f"[Orbit3D PMP] batch={batch_index} HELPER_FAILED "
                    f"rc={rc} wall={sec:.3f}s "
                    f"loops_unfilled={batch_loop_count:,}",
                    flush=True,
                )
            else:
                _append_result_faces(results, face_chunks, counters)
                print(
                    f"[Orbit3D PMP] batch={batch_index} DONE "
                    f"wall={sec:.3f}s results={len(results):,} "
                    f"regular_progress={processed_regular:,}/{regular_total:,}; "
                    + _fmt_mem("after helper"),
                    flush=True,
                )

            batch_records = []
            batch_edges = 0
            gc.collect()

        for hole_index, edges, perimeter in regular_meta:
            if batch_records and (
                len(batch_records) >= _PMP_STREAM_BATCH_LOOPS
                or batch_edges + edges > _PMP_STREAM_BATCH_EDGES
            ):
                flush_batch()

            rec = _extract_pmp_record(
                vertices,
                faces,
                ml_mesh,
                reps[hole_index],
                hole_index,
                edges,
                perimeter,
            )
            batch_records.append(rec)
            batch_edges += edges

        flush_batch()

        print(
            f"[Orbit3D PMP] regular phase complete "
            f"processed={processed_regular:,}/{regular_total:,}; "
            f"success_so_far={counters['success']:,}; "
            + _fmt_mem("before isolated phase"),
            flush=True,
        )

        # Isolated holes LAST, one subprocess each.
        for hole_index, edges, perimeter in isolated_meta:
            if edges > _PMP_HARD_SKIP_EDGES:
                hard_skipped.append(hole_index)
                counters["failed"] += 1
                line = (
                    f"isolated hole {hole_index} edges={edges:,}: "
                    f"SKIP > {_PMP_HARD_SKIP_EDGES:,}-edge experimental cap"
                )
                isolated_lines.append(line)
                print(f"[Orbit3D PMP] {line}", flush=True)
                continue

            print(
                f"[Orbit3D PMP] isolated EXTRACT hole={hole_index} "
                f"edges={edges:,} perimeter={perimeter:.9g}; "
                + _fmt_mem("before extract"),
                flush=True,
            )

            rec = _extract_pmp_record(
                vertices,
                faces,
                ml_mesh,
                reps[hole_index],
                hole_index,
                edges,
                perimeter,
            )

            print(
                f"[Orbit3D PMP] isolated RUN hole={hole_index} "
                f"edges={edges:,}; "
                + _fmt_mem("after extract"),
                flush=True,
            )

            results, sec, timeout, rc = _run_pmp_helper(
                [rec],
                1,
                int(isolated_timeout_s),
                temp_root,
            )
            del rec
            gc.collect()

            if timeout:
                timed_out.append(hole_index)
                counters["failed"] += 1
                line = (
                    f"isolated hole {hole_index} edges={edges:,}: "
                    f"TIMEOUT at {sec:.3f}s"
                )
            elif rc != 0:
                helper_failed.append((hole_index, rc))
                counters["failed"] += 1
                signal_text = ""
                if rc is not None and rc < 0:
                    signal_text = f" signal={-rc}"
                line = (
                    f"isolated hole {hole_index} edges={edges:,}: "
                    f"HELPER_FAILED rc={rc}{signal_text} wall={sec:.3f}s"
                )
            else:
                _append_result_faces(results, face_chunks, counters)
                status = results[0]["status"] if results else -99
                tris = (
                    0
                    if not results or results[0]["faces"] is None
                    else len(results[0]["faces"])
                )
                line = (
                    f"isolated hole {hole_index} edges={edges:,}: "
                    f"status={status} tris={tris:,} wall={sec:.3f}s"
                )

            isolated_lines.append(line)
            print(
                f"[Orbit3D PMP] {line}; "
                + _fmt_mem("after isolated"),
                flush=True,
            )

        del faces, ml_mesh, reps, meta, isolated_meta, regular_meta
        gc.collect()
        print(_fmt_mem("[Orbit3D PMP] boundary topology released"), flush=True)

        out = copy.copy(mesh)
        added_faces_count = 0

        if face_chunks:
            patch_np = np.concatenate(face_chunks, axis=0)
            if patch_np.max(initial=0) >= int(mesh.vertices.shape[0]):
                raise RuntimeError(
                    "PMP helper returned an out-of-range global vertex id"
                )
            patch = torch.from_numpy(
                patch_np.astype(np.int64, copy=False)
            ).to(device=mesh.faces.device, dtype=mesh.faces.dtype)
            out.faces = torch.cat((mesh.faces, patch), dim=0)
            added_faces_count = int(patch.shape[0])
        else:
            out.faces = mesh.faces

        out.vertices = mesh.vertices
        total_wall = time.perf_counter() - t0

        report_lines = [
            "=== Orbit3D PMP Polyline Hole Fill [Experimental / 3.6.23] ===",
            f"Input vertices:          {int(mesh.vertices.shape[0]):,}",
            f"Input faces:             {int(mesh.faces.shape[0]):,}",
            f"Boundary loops:          {total:,}",
            f"Boundary scan:           {scan_s:.3f} s",
            f"Regular loops processed: {processed_regular:,}/{regular_total:,}",
            f"Isolated loops:          {isolated_total:,}",
            f"Isolated threshold:      >= {isolate_edges:,} edges",
            f"Hard skip threshold:     > {_PMP_HARD_SKIP_EDGES:,} edges",
            f"Successful loops:        {counters['success']:,}",
            f"Unfilled/failed loops:   {counters['failed']:,}",
            f"Hard-skipped holes:      {len(hard_skipped):,} {hard_skipped}",
            f"Isolated timeouts:       {len(timed_out):,} {timed_out}",
            f"Isolated helper failures:{len(helper_failed):,} {helper_failed}",
            f"Regular batch timeouts:  {batch_timeouts:,}",
            f"Regular batch failures:  {batch_failures:,}",
            f"Streaming batch loops:   {_PMP_STREAM_BATCH_LOOPS:,}",
            f"Streaming edge budget:   {_PMP_STREAM_BATCH_EDGES:,}",
            f"Added faces:             {added_faces_count:,}",
            f"Output faces:            {int(out.faces.shape[0]):,}",
            f"Total wall:              {total_wall:.3f} s",
            "",
            "Execution order: regular bounded batches first, isolated holes last.",
            "CGAL mode: points-only triangulate_hole_polyline; "
            "2D constrained Delaunay -> 3D Delaunay; cubic fallback disabled.",
            "Original vertices and original faces are unchanged; "
            "successful patch triangles are append-only.",
        ]

        if isolated_lines:
            report_lines += ["", "Isolated-hole results:"] + isolated_lines

        report = "\n".join(report_lines)
        print("\n" + report, flush=True)
        return (out, report)

# --- ORBIT3D PATCH 3.6.19 PMP STREAMING END ---


# Warp is intentionally lazy.  Missing/incompatible Warp must not prevent the
# entire OrbitTo3D package from importing and exposing the PMP node.
_WARP_IMPORT_ERROR = None
try:
    import warp as wp
    import warp.geometry  # noqa: F401
except Exception as _exc:
    wp = None
    _WARP_IMPORT_ERROR = _exc


if wp is not None:
    @wp.kernel(enable_backward=False, module="orbit3d_warp_repair")
    def _orbit3d_winding_sdf_kernel(
        mesh_id: wp.uint64,
        points: wp.array(dtype=wp.vec3),
        max_dist: float,
        accuracy: float,
        threshold: float,
        values: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        p = points[i]
        q = wp.mesh_query_point_sign_winding_number(
            mesh_id, p, max_dist, accuracy, threshold
        )
        if q.result:
            closest = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
            values[i] = q.sign * wp.length(p - closest)
        else:
            values[i] = max_dist


def _parse_version_tuple(text):
    nums = re.findall(r"\d+", str(text))
    return tuple(int(x) for x in nums[:3]) + (0,) * max(0, 3 - len(nums[:3]))


class Orbit3DWarpSparseWindingRemeshExperimental:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "resolution": (["256", "512", "1024", "2048"], {"default": "1024"}),
                "padding_fraction": (
                    "FLOAT",
                    {"default": 0.03, "min": 0.005, "max": 0.25, "step": 0.005},
                ),
                "winding_accuracy": (
                    "FLOAT",
                    {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.25},
                ),
                "winding_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.05},
                ),
                "bvh_constructor": (
                    ["default", "lbvh", "sah", "median"],
                    {"default": "lbvh"},
                ),
                "output_device": (["cpu", "same", "cuda"], {"default": "cpu"}),
            }
        }

    RETURN_TYPES = ("MESHWITHVOXEL", "STRING")
    RETURN_NAMES = ("mesh", "report")
    FUNCTION = "remesh"
    CATEGORY = "Orbit3D/Mesh Repair/Experimental"
    DESCRIPTION = (
        "Experimental NVIDIA Warp 1.18+ winding-number SDF + sparse marching cubes. "
        "This is a GLOBAL REMESH and does not preserve original vertex/face identities."
    )

    def remesh(
        self,
        mesh,
        resolution,
        padding_fraction,
        winding_accuracy,
        winding_threshold,
        bvh_constructor,
        output_device,
    ):
        if wp is None:
            raise RuntimeError(
                "NVIDIA Warp is not importable. Install Warp 1.18+ with a CUDA build "
                f"compatible with the server. Import error: {_WARP_IMPORT_ERROR!r}"
            )
        if _parse_version_tuple(getattr(wp, "__version__", "0")) < (1, 18, 0):
            raise RuntimeError(
                f"Warp >=1.18 required for sparse_marching_cubes; found {getattr(wp, '__version__', '?')}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this experimental high-resolution Warp node")
        if getattr(mesh, "coords", None) is not None or getattr(mesh, "attrs", None) is not None:
            raise RuntimeError(
                "Warp experimental node currently supports mesh-only MeshWithVoxel "
                "checkpoints (coords/attrs must be None)."
            )

        # Warp's torch bridge expects the Warp runtime to have already
        # enumerated CUDA devices. Importing warp alone does not guarantee that.
        # Keep Warp and PyTorch on the same explicit cuda:N device.
        cuda_index = int(torch.cuda.current_device())
        torch_cuda_device = torch.device(f"cuda:{cuda_index}")
        warp_device = f"cuda:{cuda_index}"

        try:
            wp.init()
        except Exception as exc:
            raise RuntimeError(
                f"Warp runtime initialization failed: {type(exc).__name__}: {exc}"
            ) from exc

        if not wp.is_cuda_available():
            raise RuntimeError(
                "Warp imported successfully, but its CUDA backend is unavailable "
                f"after wp.init(); requested device={warp_device}"
            )

        try:
            wp.set_device(warp_device)
        except Exception as exc:
            raise RuntimeError(
                f"Warp could not select {warp_device}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        print(
            f"[Orbit3D Warp] runtime initialized; "
            f"warp_version={getattr(wp, '__version__', '?')}; "
            f"warp_device={warp_device}; "
            f"torch_device={torch_cuda_device}",
            flush=True,
        )

        resolution = int(resolution)
        n_nodes = resolution + 1
        t0 = time.perf_counter()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        src_device = mesh.vertices.device

        v = mesh.vertices.detach().to(
            device=torch_cuda_device,
            dtype=torch.float32,
        ).contiguous()
        f = mesh.faces.detach().to(
            device=torch_cuda_device,
            dtype=torch.int32,
        ).contiguous().view(-1)

        lower_src = v.amin(dim=0)
        upper_src = v.amax(dim=0)
        center = 0.5 * (lower_src + upper_src)
        extent = upper_src - lower_src
        side = float(extent.max().item())
        if not np.isfinite(side) or side <= 0.0:
            raise RuntimeError(f"Invalid mesh bounding box side={side}")
        half = 0.5 * side * (1.0 + 2.0 * float(padding_fraction))
        lower = center - half
        upper = center + half
        diag = float((upper - lower).norm().item())
        max_dist = max(diag * 4.0, 1.0)

        points_wp = wp.from_torch(v, dtype=wp.vec3)
        indices_wp = wp.from_torch(f, dtype=wp.int32)
        bvh = None if bvh_constructor == "default" else str(bvh_constructor)

        print(
            f"[Orbit3D Warp] building wp.Mesh V={v.shape[0]:,} F={f.numel()//3:,}; "
            f"support_winding_number=True; bvh={bvh or 'default'}",
            flush=True,
        )
        wp_mesh = wp.Mesh(
            points=points_wp,
            indices=indices_wp,
            support_winding_number=True,
            bvh_constructor=bvh,
        )
        wp.synchronize_device()
        build_s = time.perf_counter() - t0

        lo = [float(x) for x in lower.detach().cpu().tolist()]
        hi = [float(x) for x in upper.detach().cpu().tolist()]
        lower_wp = wp.vec3(lo[0], lo[1], lo[2])
        upper_wp = wp.vec3(hi[0], hi[1], hi[2])

        def evaluator(points):
            values = wp.empty(points.shape[0], dtype=wp.float32, device=points.device)
            wp.launch(
                _orbit3d_winding_sdf_kernel,
                dim=points.shape[0],
                inputs=[
                    wp_mesh.id,
                    points,
                    float(max_dist),
                    float(winding_accuracy),
                    float(winding_threshold),
                ],
                outputs=[values],
                device=points.device,
            )
            return values

        extract_t0 = time.perf_counter()
        verts_wp, inds_wp, stats = wp.geometry.sparse_marching_cubes(
            evaluator,
            n_nodes,
            n_nodes,
            n_nodes,
            lower=lower_wp,
            upper=upper_wp,
            threshold=0.0,
            return_stats=True,
        )
        wp.synchronize_device()
        extract_s = time.perf_counter() - extract_t0

        out_v_cuda = wp.to_torch(verts_wp).clone().reshape(-1, 3)
        out_f_cuda = wp.to_torch(inds_wp).clone().reshape(-1, 3).to(dtype=torch.int64)

        if output_device == "cpu":
            out_v = out_v_cuda.cpu()
            out_f = out_f_cuda.cpu()
        elif output_device == "same":
            out_v = out_v_cuda.to(src_device)
            out_f = out_f_cuda.to(src_device)
        else:
            out_v = out_v_cuda
            out_f = out_f_cuda

        # New topology: deliberately do not copy voxel/attribute payloads that
        # describe the original topology.  Mesh-only geometry is the target.
        out = MeshWithVoxel(
            out_v,
            out_f,
            origin=mesh.origin,
            voxel_size=mesh.voxel_size,
            coords=None,
            attrs=None,
            voxel_shape=None,
            layout=mesh.layout,
        )

        peak_gib = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
        reserved_gib = torch.cuda.max_memory_reserved() / (1024.0 ** 3)
        total_s = time.perf_counter() - t0
        dense_evals = n_nodes ** 3
        field_evals = int(stats.get("field_evaluations", -1)) if isinstance(stats, dict) else -1
        leaf_cells = int(stats.get("leaf_cells", -1)) if isinstance(stats, dict) else -1

        report = "\n".join([
            "=== Orbit3D Warp Sparse Winding Remesh [Experimental] ===",
            f"Warp version:            {getattr(wp, '__version__', '?')}",
            f"Input vertices:          {int(mesh.vertices.shape[0]):,}",
            f"Input faces:             {int(mesh.faces.shape[0]):,}",
            f"Resolution:              {resolution}^3 cells ({n_nodes}^3 nodes)",
            f"Domain lower:            {lo}",
            f"Domain upper:            {hi}",
            f"Padding fraction:        {float(padding_fraction):.4f}",
            f"Winding accuracy:        {float(winding_accuracy):.3f}",
            f"Winding threshold:       {float(winding_threshold):.3f}",
            f"BVH constructor:         {bvh or 'default'}",
            f"Mesh/BVH build:          {build_s:.3f} s",
            f"Sparse extraction:       {extract_s:.3f} s",
            f"Leaf cells:              {leaf_cells:,}",
            f"Field evaluations:       {field_evals:,}",
            f"Dense-equivalent evals:  {dense_evals:,}",
            f"Output vertices:         {int(out_v.shape[0]):,}",
            f"Output faces:            {int(out_f.shape[0]):,}",
            f"CUDA peak allocated:     {peak_gib:.3f} GiB",
            f"CUDA peak reserved:      {reserved_gib:.3f} GiB",
            f"Output device:           {out_v.device}",
            f"Total wall:              {total_s:.3f} s",
            "",
            "NOTE: global implicit remesh; original vertex/face identities are not preserved.",
        ])
        print("\n" + report, flush=True)

        # Drop Warp acceleration structures as soon as the output has been copied.
        del wp_mesh, points_wp, indices_wp, v, f, verts_wp, inds_wp, out_v_cuda, out_f_cuda
        torch.cuda.empty_cache()
        return (out, report)


NODE_CLASS_MAPPINGS = {
    "Orbit3DPMPPolylineHoleFillExperimental": Orbit3DPMPPolylineHoleFillExperimental,
    "Orbit3DWarpSparseWindingRemeshExperimental": Orbit3DWarpSparseWindingRemeshExperimental,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Orbit3DPMPPolylineHoleFillExperimental": "Orbit3D - PMP Polyline Hole Fill [Experimental]",
    "Orbit3DWarpSparseWindingRemeshExperimental": "Orbit3D - Warp Sparse Winding Remesh [Experimental]",
}
