# ORBIT3D_STAGE5_ANGLE_GRID_V1
"""Small deterministic yaw/pitch angle-grid generator for Orbit3D workflows."""

from __future__ import annotations


_DIVISORS_360 = tuple(str(v) for v in range(1, 361) if 360 % v == 0)


def _parse_step(value, label: str) -> int:
    try:
        step = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{label} must be an integer divisor of 360, got {value!r}") from e

    if step <= 0 or step > 360 or 360 % step != 0:
        raise ValueError(
            f"{label}={step} is invalid; expected a positive integer divisor of 360"
        )
    return step


def build_angle_grid(yaw_step, pitch_step):
    """
    Build a full yaw x pitch grid over [0, 360), excluding the duplicate 360° pose.

    Ordering is pitch-major / yaw-minor: for each pitch layer, walk one complete
    yaw revolution.  This preserves the natural turntable ordering when
    pitch_step=360 and also gives deterministic multi-elevation grids.
    """
    yaw_step = _parse_step(yaw_step, "yaw_step")
    pitch_step = _parse_step(pitch_step, "pitch_step")

    yaws = tuple(range(0, 360, yaw_step))
    pitches = tuple(range(0, 360, pitch_step))

    azimuths = []
    elevations = []
    for pitch in pitches:
        for yaw in yaws:
            azimuths.append(str(yaw))
            elevations.append(str(pitch))

    view_count = len(azimuths)
    if view_count != len(yaws) * len(pitches):
        raise RuntimeError("internal Orbit3D angle-grid size mismatch")

    return ", ".join(azimuths), ", ".join(elevations), view_count


class Orbit3DAngleGrid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Dropdowns intentionally expose only exact divisors of 360,
                # so an invalid angular step cannot be entered accidentally.
                "yaw_step": (_DIVISORS_360, {"default": "15"}),
                "pitch_step": (_DIVISORS_360, {"default": "360"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("azimuths", "elevations", "view_count")
    FUNCTION = "run"
    CATEGORY = "Orbit to 3D/Views"

    def run(self, yaw_step, pitch_step):
        return build_angle_grid(yaw_step, pitch_step)
