from __future__ import annotations

from dataclasses import dataclass

ACTION_OUTPUT_COUNT = 2
ACTION_OUTPUT_NAMES = ("accelerate", "rotate")


@dataclass(slots=True)
class Action:
    accelerate: float
    rotate: float

    def clamped(self) -> Action:
        return Action(
            accelerate=max(-1.0, min(1.0, self.accelerate)),
            rotate=max(-1.0, min(1.0, self.rotate)),
        )
