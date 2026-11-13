"""Scoped SS64 logit experiment; inactive for all stock nodes."""
from contextlib import contextmanager
from contextvars import ContextVar
import math
import torch

_request = ContextVar("orbit3d_ss64_t5_request", default=None)

@contextmanager
def request_capture(threshold, surface_all=False):
    threshold = float(threshold)
    if not math.isfinite(threshold) or not -100.0 <= threshold <= 0.0:
        raise ValueError("occupancy_threshold must be finite and in [-100, 0]")
    if _request.get() is not None:
        raise RuntimeError("Nested SS64 T5 capture is not supported")
    record = {"threshold": threshold, "surface_all": bool(surface_all)}
    token = _request.set(record)
    try:
        yield record
    finally:
        _request.reset(token)


def active_request():
    return _request.get()


def capture_logits(record, latent, logits):
    if "base_coords" in record:
        raise RuntimeError("SS64 T5 capture unexpectedly called twice")
    if tuple(latent.shape) != (1, 8, 16, 16, 16):
        raise RuntimeError(f"Expected native SS64 latent, got {tuple(latent.shape)}")
    if tuple(logits.shape) != (1, 1, 64, 64, 64):
        raise RuntimeError(f"Expected native SS64 logits, got {tuple(logits.shape)}")
    raw = logits.detach().cpu().contiguous()
    values = raw.float()
    if not torch.isfinite(values).all():
        raise RuntimeError("Non-finite occupancy logits")
    base = values > 0
    selected = values > record["threshold"]
    additions = selected & ~base
    base_coords = torch.argwhere(base)[:, [0, 2, 3, 4]].int().contiguous()
    extra_coords = torch.argwhere(additions)[:, [0, 2, 3, 4]].int().contiguous()
    if not len(base_coords):
        raise RuntimeError("Empty raw SS64 occupancy")
    # Match the validated offline T5 experiment: do NOT resort existing rows.
    t5_coords = torch.cat((base_coords, extra_coords), dim=0).contiguous()
    if int(selected.sum()) != len(t5_coords):
        raise RuntimeError("Threshold coordinate count mismatch")
    record.update(latent=latent.detach().cpu().contiguous(), logits=raw,
                  occupancy=base.contiguous(), t5_occupancy=selected.contiguous(),
                  base_coords=base_coords, t5_coords=t5_coords,
                  added_coords=extra_coords)
    print(f"[Orbit3D SS64 T5] threshold={record['threshold']:.4f}; "
          f"BASE={len(base_coords)}; added={len(extra_coords)}; "
          f"T5={len(t5_coords)}; removed=0; S4=skipped (experimental)", flush=True)
    if record["surface_all"]:
        completed, surface_coords = complete_surface_all(selected, t5_coords, values)
        record.update(surface_all_coords=completed, surface_added_coords=surface_coords)
        print(f"[Orbit3D SS64 SurfaceAll] T5={len(t5_coords)}; "
              f"surface_added={len(surface_coords)}; final={len(completed)}; "
              "iterations=1; removed=0; no additional logit cutoff", flush=True)
    return base_coords


def complete_surface_all(t5_occupancy, t5_coords, values):
    """Single pass over the immutable T5 mask; preserve tested row ordering."""
    if tuple(t5_occupancy.shape) != (1, 1, 64, 64, 64):
        raise ValueError("SurfaceAll requires one native SS64 occupancy")
    occ = t5_occupancy[0, 0].bool()
    p = torch.nn.functional.pad(occ, (1, 1, 1, 1, 1, 1), value=False)
    pairs = (
        (p[:-2, 1:-1, 1:-1] & p[2:, 1:-1, 1:-1]).to(torch.int8)
        + (p[1:-1, :-2, 1:-1] & p[1:-1, 2:, 1:-1]).to(torch.int8)
        + (p[1:-1, 1:-1, :-2] & p[1:-1, 1:-1, 2:]).to(torch.int8)
    )
    additions = (~occ) & (pairs >= 2)
    # The former Surface41 cutoff is used ONLY to reproduce the exact tested
    # coordinate order: old Surface41 rows first, remaining rows afterward.
    # It does not affect membership in SurfaceAll.
    order_boundary = values[0, 0] > -41.0
    groups = []
    for mask in (additions & order_boundary, additions & ~order_boundary):
        xyz = torch.argwhere(mask).int()
        batch = torch.zeros((len(xyz), 1), dtype=torch.int32, device=xyz.device)
        groups.append(torch.cat((batch, xyz), dim=1))
    extra = torch.cat(groups, dim=0).contiguous()
    completed = torch.cat((t5_coords, extra), dim=0).contiguous()
    if len(extra) != int(additions.sum()) or len(completed) != int(occ.sum()) + len(extra):
        raise RuntimeError("SurfaceAll coordinate count mismatch")
    return completed, extra
