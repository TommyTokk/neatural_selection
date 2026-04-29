from __future__ import annotations

from math import cos, sin
from src.world import World
import arcade

from configs.sim_config import SimConfig


class EnvironmentRenderer:
    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.theme = config.theme

    def draw(self, world: World) -> None:
        bounds = world.layout.environment
        self._draw_panel(bounds)
        self._draw_grid(bounds)
        self._draw_environment_header(bounds, world)

    def _draw_panel(self, bounds: arcade.Rect) -> None:
        arcade.draw_lrbt_rectangle_filled(
            bounds.left,
            bounds.right,
            bounds.bottom,
            bounds.top,
            self.theme.environment_background,
        )

        arcade.draw_lrbt_rectangle_outline(
            bounds.left,
            bounds.right,
            bounds.bottom,
            bounds.top,
            self.theme.environment_border,
            border_width=2,
        )

    def _draw_grid(self, bounds: arcade.Rect) -> None:
        step = 48
        x = bounds.left + step
        while x < bounds.right:
            arcade.draw_line(
                x, bounds.bottom, x, bounds.top, self.theme.environment_grid, 1
            )
            x += step

        y = bounds.bottom + step
        while y < bounds.top:
            arcade.draw_line(
                bounds.left, y, bounds.right, y, self.theme.environment_grid, 1
            )
            y += step

    def _draw_environment_header(self, bounds: arcade.Rect, world: World) -> None:
        arcade.draw_text(
            world.stats.environment_name,
            bounds.left + 16,
            bounds.top - 30,
            self.theme.text_primary,
            16,
            bold=True,
        )
        arcade.draw_text(
            "Central simulation viewport",
            bounds.left + 16,
            bounds.top - 52,
            self.theme.text_muted,
            11,
        )
