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
        self._draw_rounded_rect(
            bounds,
            self.theme.environment_background,
            self.theme.environment_border,
            self.config.layout.environment_radius,
            2,
        )

    def _draw_rounded_rect(
        self,
        bounds: arcade.Rect,
        fill_color: arcade.Color | tuple[int, ...],
        border_color: arcade.Color | tuple[int, ...],
        radius: float,
        border_width: float,
    ) -> None:
        self._draw_rounded_rect_fill(bounds, border_color, radius)
        inner = arcade.LBWH(
            bounds.left + border_width,
            bounds.bottom + border_width,
            max(0, bounds.width - border_width * 2),
            max(0, bounds.height - border_width * 2),
        )
        self._draw_rounded_rect_fill(inner, fill_color, max(0, radius - border_width))

    def _draw_rounded_rect_fill(
        self,
        bounds: arcade.Rect,
        color: arcade.Color | tuple[int, ...],
        radius: float,
    ) -> None:
        radius = min(radius, bounds.width / 2, bounds.height / 2)
        arcade.draw_lrbt_rectangle_filled(
            bounds.left + radius,
            bounds.right - radius,
            bounds.bottom,
            bounds.top,
            color,
        )
        arcade.draw_lrbt_rectangle_filled(
            bounds.left,
            bounds.right,
            bounds.bottom + radius,
            bounds.top - radius,
            color,
        )
        arcade.draw_circle_filled(bounds.left + radius, bounds.bottom + radius, radius, color)
        arcade.draw_circle_filled(bounds.right - radius, bounds.bottom + radius, radius, color)
        arcade.draw_circle_filled(bounds.left + radius, bounds.top - radius, radius, color)
        arcade.draw_circle_filled(bounds.right - radius, bounds.top - radius, radius, color)

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
        self._draw_text(
            world.stats.environment_name,
            bounds.left + 16,
            bounds.top - 30,
            self.theme.environment_text,
            16,
            bold=True,
        )
        self._draw_text(
            "Central simulation viewport",
            bounds.left + 16,
            bounds.top - 52,
            self.theme.environment_text_muted,
            11,
        )

    def _draw_text(
        self,
        text: str,
        x: float,
        y: float,
        color: arcade.Color | tuple[int, ...],
        size: float,
        *,
        bold: bool = False,
    ) -> None:
        arcade.Text(text, round(x), round(y), color, size, bold=bold).draw()
