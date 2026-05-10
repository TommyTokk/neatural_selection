from __future__ import annotations

from dataclasses import dataclass

ACTION_OUTPUT_COUNT = 3
ACTION_OUTPUT_NAMES = ("accelerate", "rotate", "herding")


@dataclass(slots=True)
class Action:
    accelerate: float
    rotate: float
    herding: float = 0.0

    def clamped(self) -> Action:
        return Action(
            accelerate=max(-1.0, min(1.0, self.accelerate)),
            rotate=max(-1.0, min(1.0, self.rotate)),
            herding=max(-1.0, min(1.0, self.herding)),
        )