from __future__ import annotations

from dataclasses import dataclass

import arcade


@dataclass(slots=True)
class Food:
    x_ratio: float
    y_ratio: float
    radius: float

    def get_screen_position(self, bounds: arcade.Rect) -> tuple[float, float]:
        return (
            bounds.left + bounds.width * self.x_ratio,
            bounds.bottom + bounds.height * self.y_ratio,
        )
