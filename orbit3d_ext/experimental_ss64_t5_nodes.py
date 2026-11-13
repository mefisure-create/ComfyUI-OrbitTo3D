"""Paired SS64 BASE/T5 node. Reuses stock Orbit3D conditioning and sampler."""
from pathlib import Path
import json
import torch
from ..orbit_nodes import Orbit3DSparseGenerator
from ..trellis2.pipelines.ss64_t5_capture import request_capture
from .experimental_shape_nodes import _atomic_safetensors, _atomic_json


class Orbit3DSS64T5Experimental(Orbit3DSparseGenerator):
    CATEGORY = "Orbit to 3D/Experimental"
    FUNCTION = "process"
    OUTPUT_NODE = True
    SURFACE_ALL = False
    RETURN_TYPES = ("COORDS", "INT", "TRELLIS2PIPELINE", "COORDS", "INT")
    RETURN_NAMES = ("coords", "sparse_structure_resolution", "pipeline", "base_coords", "added_tokens")

    @classmethod
    def INPUT_TYPES(cls):
        base = Orbit3DSparseGenerator.INPUT_TYPES()
        required = dict(base["required"])
        for key in ("fill_holes", "hole_iterations", "hole_fill_algorithm", "keep_only_shell"):
            required.pop(key)
        defaults = {
            "seed": 1234,
            "sparse_structure_steps": 25,
            "sparse_structure_guidance_strength": 7.5,
            "sparse_structure_guidance_rescale": 0.70,
            "sparse_structure_rescale_t": 5.0,
            "sparse_structure_sampler": "heun",
            "sparse_structure_resolution": 64,
            "sparse_structure_guidance_interval_start": 0.60,
            "sparse_structure_guidance_interval_end": 1.0,
            "dino_lock": 0.0,
            "dino_substeps": 1,
            "dino_foundation_cap": 0.92,
        }
        for key, value in defaults.items():
            kind, settings = required[key]
            settings = dict(settings, default=value)
            if key == "sparse_structure_resolution":
                settings.update(min=64, max=64, step=1)
            required[key] = (kind, settings)
        required["occupancy_threshold"] = ("FLOAT", {
            "default": -5.0, "min": -100.0, "max": 0.0, "step": 0.1,
            "tooltip": "Raw decoder logits > threshold. 0 = exact BASE; -5 = T5. No morphology or S4."})
        required["checkpoint_path"] = ("STRING", {
            "default": "", "multiline": False,
            "tooltip": "Optional NEW directory for paired BASE/T5 checkpoints, flow dump and report. Empty = do not save. Never overwrites existing files."})
        return {"required": required, "optional": dict(base.get("optional", {}))}

    def process(self, pipeline, seed, occupancy_threshold, checkpoint_path,
                sparse_structure_resolution=64, pixal3d_mv_views=None, **kwargs):
        if int(sparse_structure_resolution) != 64:
            raise ValueError("SS64 T5 requires sparse_structure_resolution=64")
        root = None
        if str(checkpoint_path).strip():
            root = Path(str(checkpoint_path).strip()).expanduser().resolve()
            if root.exists() and (not root.is_dir() or any(root.iterdir())):
                raise FileExistsError(f"SS64 T5 checkpoint already exists: {root}")
        with request_capture(occupancy_threshold, surface_all=self.SURFACE_ALL) as record:
            result = super().process(
                pipeline=pipeline, seed=seed, sparse_structure_resolution=64,
                fill_holes=False, hole_iterations=1,
                hole_fill_algorithm="flood_fill", keep_only_shell=False,
                pixal3d_mv_views=pixal3d_mv_views, **kwargs)
        if "base_coords" not in record:
            raise RuntimeError("SS64 T5 pipeline hook missing; install failed or wrong pipeline")
        returned, res, result_pipeline = result
        if int(res) != 64 or not torch.equal(returned.detach().cpu().int(), record["base_coords"]):
            raise RuntimeError("BASE changed after capture; refusing inconsistent experiment")
        base = record["base_coords"]
        t5 = record["t5_coords"]
        extras = record["added_coords"]
        if root is not None:
            self._save_pair(root, record, seed, kwargs)
        if self.SURFACE_ALL:
            final = record["surface_all_coords"]
            surface_count = int(len(record["surface_added_coords"]))
            return (final, 64, result_pipeline, base,
                    int(len(final) - len(base)), t5, surface_count)
        return (t5, 64, result_pipeline, base, int(len(extras)))

    @staticmethod
    def _save_pair(root, record, seed, settings):
        # Fail closed on collisions. Atomic per-file writes; a partial failed
        # export is intentionally not overwritten by the next run.
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise FileExistsError(f"SS64 T5 checkpoint not empty: {root}")
        stages = [("BASE", "base_coords"), ("T5", "t5_coords")]
        if record["surface_all"]:
            stages.append(("T5_SurfaceAll", "surface_all_coords"))
        for name, key in stages:
            path = root / name
            path.mkdir()
            coords = record[key]
            _atomic_safetensors(path / "tensors.safetensors", {"coords": coords})
            _atomic_json(path / "manifest.json", {
                "format": "orbit3d_stage_checkpoint_v1", "kind": "sparse",
                "sparse_structure_resolution": 64, "num_tokens": int(len(coords))})
        dump = {key: record[key] for key in
                ("latent", "logits", "occupancy", "t5_occupancy", "added_coords")}
        if record["surface_all"]:
            dump["surface_added_coords"] = record["surface_added_coords"]
        _atomic_safetensors(root / "flow_dump.safetensors", dump)
        values = record["logits"][0,0].float()
        additions = [{"native": [int(x), int(y), int(z)],
                      "logit": float(values[x,y,z])}
                     for _, x, y, z in record["added_coords"].tolist()]
        additions.sort(key=lambda row: row["logit"], reverse=True)
        report = {
            "experiment": "SS64 paired logit threshold",
            "seed": int(seed), "threshold": record["threshold"],
            "base_tokens": int(len(record["base_coords"])),
            "t5_tokens": int(len(record["t5_coords"])),
            "added_tokens": len(additions), "removed_tokens": 0,
            "coordinate_space": "Pixal3D native", "base_order_preserved": True,
            "fill_holes": False, "S4": False,
            "sampler_settings": settings, "additions": additions}
        if record["surface_all"]:
            final = record["surface_all_coords"]
            surface_extra = record["surface_added_coords"]
            report.update(
                experiment="SS64 T5 + SurfaceAll",
                completion="one_pass_opposite_pairs_ge_2",
                support_reference="T5",
                additional_logit_threshold=None,
                surface_added_tokens=int(len(surface_extra)),
                total_added_tokens=int(len(final) - len(record["base_coords"])),
                total_tokens=int(len(final)),
                surface_additions=[
                    {"native": [int(x), int(y), int(z)],
                     "logit": float(values[x,y,z])}
                    for _, x, y, z in surface_extra.tolist()
                ])
        _atomic_json(root / "report.json", report)
        print(f"[Orbit3D SS64 T5] saved paired checkpoints: {root}", flush=True)


class Orbit3DSS64SurfaceAll(Orbit3DSS64T5Experimental):
    """Validated T5 followed by one-pass local SS64 completion."""

    CATEGORY = "Orbit to 3D/Experimental"
    SURFACE_ALL = True
    RETURN_TYPES = ("COORDS", "INT", "TRELLIS2PIPELINE", "COORDS", "INT", "COORDS", "INT")
    RETURN_NAMES = ("coords", "sparse_structure_resolution", "pipeline", "base_coords",
                    "added_tokens", "t5_coords", "surface_added_tokens")

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        required = dict(schema["required"])
        kind, options = required["occupancy_threshold"]
        required["occupancy_threshold"] = (kind, dict(
            options, tooltip="T5 threshold, default -5. SurfaceAll then adds every empty cell with >=2 opposite occupied face-neighbor pairs in one pass."))
        kind, options = required["checkpoint_path"]
        required["checkpoint_path"] = (kind, dict(
            options, tooltip="Optional NEW directory for BASE, T5, T5_SurfaceAll, flow dump and report. Empty = no export."))
        return {"required": required, "optional": dict(schema.get("optional", {}))}
