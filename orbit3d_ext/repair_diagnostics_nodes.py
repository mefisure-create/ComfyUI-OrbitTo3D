from __future__ import annotations

import math
import time

import numpy as np
import pymeshlab


_GEOM_CHUNK_FACES = 500_000
_SAMPLE_FACES = 1_000_000


def _chunked_geometry_metrics(vertices, faces, chunk_faces=_GEOM_CHUNK_FACES):
    total_area = 0.0
    signed_volume = 0.0
    degenerate = 0

    n = len(faces)
    for start in range(0, n, chunk_faces):
        tri_idx = faces[start:start + chunk_faces]
        a = vertices[tri_idx[:, 0]]
        b = vertices[tri_idx[:, 1]]
        c = vertices[tri_idx[:, 2]]

        cross = np.cross(b - a, c - a)
        dbl_area = np.linalg.norm(cross, axis=1)
        total_area += float(0.5 * dbl_area.sum(dtype=np.float64))
        degenerate += int(np.count_nonzero(dbl_area <= 1e-15))

        signed_volume += float(
            np.einsum("ij,ij->i", a, np.cross(b, c)).sum(dtype=np.float64) / 6.0
        )

    return total_area, signed_volume, degenerate


def _sample_shape_metrics(vertices, faces, max_faces=_SAMPLE_FACES):
    """Deterministic bounded sample; diagnostic only, never changes geometry."""
    n = len(faces)
    if n == 0:
        return {
            "sample_faces": 0,
            "area_q": (0.0, 0.0, 0.0),
            "edge_q": (0.0, 0.0, 0.0),
        }

    stride = max(1, int(math.ceil(n / max_faces)))
    tri_idx = faces[::stride][:max_faces]
    a = vertices[tri_idx[:, 0]]
    b = vertices[tri_idx[:, 1]]
    c = vertices[tri_idx[:, 2]]

    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    edges = np.concatenate((ab, bc, ca))

    aq = tuple(float(x) for x in np.quantile(areas, [0.50, 0.90, 0.99]))
    eq = tuple(float(x) for x in np.quantile(edges, [0.50, 0.90, 0.99]))
    return {
        "sample_faces": int(len(tri_idx)),
        "area_q": aq,
        "edge_q": eq,
    }


class Orbit3DRepairDiagnosticsExperimental:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESHWITHVOXEL",),
                "label": ("STRING", {"default": "candidate"}),
            }
        }

    RETURN_TYPES = ("MESHWITHVOXEL", "STRING")
    RETURN_NAMES = ("mesh", "report")
    FUNCTION = "analyze"
    CATEGORY = "Orbit3D/Diagnostics"
    DESCRIPTION = (
        "Topology plus bounded-memory geometry metrics for comparing repair backends. "
        "Does not mutate the mesh."
    )

    def analyze(self, mesh, label):
        t0 = time.perf_counter()
        src_vertices = mesh.vertices
        src_faces = mesh.faces

        # Same basic representation used by the user's existing diagnostic node.
        vertices = np.asarray(src_vertices.detach().cpu().numpy(), dtype=np.float64)
        faces = np.asarray(src_faces.detach().cpu().numpy(), dtype=np.int32)

        if len(vertices):
            bbox_min = vertices.min(axis=0)
            bbox_max = vertices.max(axis=0)
            extent = bbox_max - bbox_min
            bbox_diag = float(np.linalg.norm(extent))
        else:
            bbox_min = bbox_max = extent = np.zeros(3, dtype=np.float64)
            bbox_diag = 0.0

        surface_area, signed_volume_raw, degenerate_faces = _chunked_geometry_metrics(
            vertices, faces
        )
        sampled = _sample_shape_metrics(vertices, faces)

        pm_mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pm_mesh, label)
        topo = ms.get_topological_measures()

        vertices_number = int(topo.get("vertices_number", len(vertices)))
        edges_number = int(topo.get("edges_number", -1))
        faces_number = int(topo.get("faces_number", len(faces)))
        boundary_edges = int(topo.get("boundary_edges", -1))
        connected_components = int(topo.get("connected_components_number", -1))
        non_manifold_edges = int(topo.get("non_two_manifold_edges", -1))
        non_manifold_vertices = int(topo.get("non_two_manifold_vertices", -1))
        is_two_manifold = bool(topo.get("is_mesh_two_manifold", False))
        holes = int(topo.get("number_holes", -1))
        genus = int(topo.get("genus", -1))
        watertight = is_two_manifold and boundary_edges == 0

        euler = (
            vertices_number - edges_number + faces_number
            if edges_number >= 0 else None
        )
        mean_triangle_area = surface_area / faces_number if faces_number > 0 else 0.0
        characteristic_edge = math.sqrt(max(mean_triangle_area, 0.0))
        physical_volume = abs(signed_volume_raw) if watertight else None

        def fmt_vec(v):
            return "[" + ", ".join(f"{float(x):.9g}" for x in v) + "]"

        area_p50, area_p90, area_p99 = sampled["area_q"]
        edge_p50, edge_p90, edge_p99 = sampled["edge_q"]
        physical_volume_text = (
            f"{physical_volume:.12g}" if physical_volume is not None
            else "n/a (mesh not watertight)"
        )
        elapsed = time.perf_counter() - t0

        report = (
            f"=== Orbit3D Repair Diagnostics [Experimental]: {label} ===\n"
            f"Tensor vertices:       dtype={src_vertices.dtype} device={src_vertices.device}\n"
            f"Tensor faces:          dtype={src_faces.dtype} device={src_faces.device}\n"
            f"Vertices:              {vertices_number:,}\n"
            f"Faces:                 {faces_number:,}\n"
            f"Edges:                 {edges_number:,}\n"
            f"Euler characteristic:  {euler if euler is not None else 'n/a'}\n"
            f"\n"
            f"Connected components:  {connected_components:,}\n"
            f"Boundary edges:         {boundary_edges:,}\n"
            f"Holes:                  {holes:,}\n"
            f"Non-manifold edges:     {non_manifold_edges:,}\n"
            f"Non-manifold vertices:  {non_manifold_vertices:,}\n"
            f"Two-manifold:           {is_two_manifold}\n"
            f"Watertight:             {watertight}\n"
            f"Genus:                  {genus}\n"
            f"\n"
            f"BBox min:               {fmt_vec(bbox_min)}\n"
            f"BBox max:               {fmt_vec(bbox_max)}\n"
            f"BBox extent:            {fmt_vec(extent)}\n"
            f"BBox diagonal:          {bbox_diag:.9g}\n"
            f"Surface area:           {surface_area:.12g}\n"
            f"Signed volume (raw):    {signed_volume_raw:.12g}\n"
            f"Physical volume:        {physical_volume_text}\n"
            f"Mean triangle area:     {mean_triangle_area:.12g}\n"
            f"sqrt(mean tri area):    {characteristic_edge:.12g}\n"
            f"Degenerate faces*:      {degenerate_faces:,}\n"
            f"\n"
            f"Shape sample faces:     {sampled['sample_faces']:,}\n"
            f"Triangle area p50/p90/p99: {area_p50:.9g} / {area_p90:.9g} / {area_p99:.9g}\n"
            f"Edge length p50/p90/p99:   {edge_p50:.9g} / {edge_p90:.9g} / {edge_p99:.9g}\n"
            f"Diagnostics wall:       {elapsed:.3f} s\n"
            f"\n"
            f"* Degenerate threshold: double-area <= 1e-15. Quantiles use a deterministic\n"
            f"  bounded sample of at most {_SAMPLE_FACES:,} faces.\n"
        )
        print("\n" + report, flush=True)
        return (mesh, report)


NODE_CLASS_MAPPINGS = {
    "Orbit3DRepairDiagnosticsExperimental": Orbit3DRepairDiagnosticsExperimental,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Orbit3DRepairDiagnosticsExperimental": "Orbit3D - Repair Diagnostics [Experimental]",
}
