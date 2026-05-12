from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

ACTION_OUTPUT_COUNT = 2
ACTION_OUTPUT_NAMES = ("accelerate", "rotate")
NEUTRAL_NETWORK_OUTPUT = 0.5


@dataclass(slots=True)
class Action:
    accelerate: float
    rotate: float

    def clamped(self) -> Action:
        return Action(
            accelerate=max(-1.0, min(1.0, self.accelerate)),
            rotate=max(-1.0, min(1.0, self.rotate)),
        )


def signed_output(value: float) -> float:
    return (value - NEUTRAL_NETWORK_OUTPUT) * 2.0


def acceleration_force_vector(
    accelerate: float,
    heading: float,
    max_forward_force: float,
    max_backward_force: float,
) -> tuple[float, float]:
    if accelerate >= 0.0:
        magnitude = max_forward_force * accelerate
    else:
        magnitude = max_backward_force * accelerate

    return (
        cos(heading) * magnitude,
        sin(heading) * magnitude,
    )
