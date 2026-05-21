from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

ACTION_OUTPUT_COUNT = 7
ACTION_OUTPUT_NAMES = (
    "forward",
    "backward",
    "left",
    "right",
    "want_reproduce",
    "want_eat",
    "reset_chronometer",
)
NEUTRAL_NETWORK_OUTPUT = 0.5


@dataclass(slots=True)
class Action:
    forward: float
    backward: float
    left: float
    right: float
    want_reproduce: float
    want_eat: float
    reset_chronometer: float

    def clamped(self) -> Action:
        return Action(
            forward=max(0.0, min(1.0, self.forward)),
            backward=max(0.0, min(1.0, self.backward)),
            left=max(0.0, min(1.0, self.left)),
            right=max(0.0, min(1.0, self.right)),
            want_reproduce=max(0.0, min(1.0, self.want_reproduce)),
            want_eat=max(0.0, min(1.0, self.want_eat)),
            reset_chronometer=max(0.0, min(1.0, self.reset_chronometer)),
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
