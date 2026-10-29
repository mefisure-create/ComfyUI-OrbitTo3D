"""Orbit-side schema extensions for native ComfyUI MoGe nodes.

The inference implementation is intentionally *not* copied here.  Orbit inherits
ComfyUI's native ``MoGeInference.execute`` verbatim and only widens the
``resolution_level`` validation range used by the high-resolution workflow.
"""

from comfy_extras.nodes_moge import MoGeInference as _ComfyMoGeInference


class Orbit3DMoGeInference(_ComfyMoGeInference):
    """Native ComfyUI MoGe inference with an extended resolution-level range."""

    @classmethod
    def define_schema(cls):
        schema = _ComfyMoGeInference.define_schema()

        # Keep the execution contract owned by ComfyUI; only give this derived
        # node its own identity and widen the prompt/UI validation bound.
        schema.node_id = "Orbit3DMoGeInference"
        schema.display_name = "Orbit3D - MoGe Inference (Extended Resolution)"

        matches = [inp for inp in schema.inputs if inp.id == "resolution_level"]
        if len(matches) != 1:
            raise RuntimeError(
                "Orbit3D: native MoGeInference schema changed: expected exactly "
                "one 'resolution_level' input."
            )

        resolution = matches[0]
        if not hasattr(resolution, "max"):
            raise RuntimeError(
                "Orbit3D: native MoGeInference resolution_level input no longer "
                "exposes a numeric max bound."
            )

        upstream_max = resolution.max
        resolution.max = max(30, upstream_max) if upstream_max is not None else 30
        resolution.tooltip = (
            "Native ComfyUI MoGe resolution level. Orbit extends the accepted "
            "range to 30; execution remains the native MoGeInference.execute()."
        )

        return schema
