"""Runtime-only trainer namespace.

Orbit retains only flow_matching.mixins.image_conditioned_proj because Pixal3D
uses its DINO projection feature extractors at inference time.  Training
implementations are intentionally not shipped.
"""
