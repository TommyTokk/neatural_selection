from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

ACTION_OUTPUT_COUNT = 7
ACTION_OUTPUT_NAMES = (
    "accelerate",
    "rotate",
    "want_reproduce",
    "want_eat",
    "reset_chronometer",
    "want_grab",
    "want_release",
)
NEUTRAL_NETWORK_OUTPUT = 0.5


@dataclass(slots=True)
class Action:
    accelerate: float
    rotate: float
    want_reproduce: float
    want_eat: float
    reset_chronometer: float
    want_grab: float
    want_release: float

    def clamped(self) -> Action:
        return Action(
            accelerate=max(-1.0, min(1.0, self.accelerate)),
            rotate=max(-1.0, min(1.0, self.rotate)),
            want_reproduce=max(0.0, min(1.0, self.want_reproduce)),
            want_eat=max(0.0, min(1.0, self.want_eat)),
            reset_chronometer=max(0.0, min(1.0, self.reset_chronometer)),
            want_grab=max(0.0, min(1.0, self.want_grab)),
            want_release=max(0.0, min(1.0, self.want_release)),
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
