from __future__ import annotations

import arcade

from configs.sim_config import SimConfig
from src.world import World


class UiRenderer:
    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.theme = config.theme
        self._text_cache: dict[str, arcade.Text] = {}

    def draw(self, world: World) -> None:
        self._draw_top_bar(world)
        self._draw_sidebar(world)

    def _draw_top_bar(self, world: World) -> None:
        bounds = world.layout.top_bar
        self._draw_panel(bounds)

        self._draw_text(
            "top_title",
            "Neat Game Of Life",
            bounds.left + 18,
            bounds.top - 34,
            self.theme.text_primary,
            24,            
            bold=True,
        )

        self._draw_text(
            "top_subtitle",
            "Window container with nested environment + UI panels",
            bounds.left + 18,
            bounds.top - 60,
            self.theme.text_muted,
            12,
        )

        status = "Debug vision on" if world.debug_vision_enabled else "Debug vision off"
        self._draw_text(
            "top_status",
            status,
            bounds.right - 180,
            bounds.top - 40,
            self.theme.accent,
            13,
            bold=True,
        )

    def _draw_sidebar(self, world: World) -> None:
        bounds = world.layout.left_sidebar
        self._draw_panel(bounds, fill_color=self.theme.panel_background_alt)

        title_y = bounds.top - 28
        self._draw_text(
            "sidebar_title",
            "Inspector",
            bounds.left + 18,
            title_y,
            self.theme.text_primary,
            18,
            bold=True,
        )

        card_width = bounds.width - 36
        card_height = 146
        gap = 16

        first_card = arcade.LBWH(
            bounds.left + 18, bounds.top - 58 - card_height, card_width, card_height
        )
        second_card = arcade.LBWH(
            bounds.left + 18,
            first_card.bottom - gap - card_height,
            card_width,
            card_height,
        )
        third_card = arcade.LBWH(
            bounds.left + 18, second_card.bottom - gap - 122, card_width, 122
        )

        self._draw_card(first_card, "Selected Creature")
        self._draw_selected_creature(world, first_card)

        self._draw_card(second_card, "Environment Stats")
        self._draw_environment_stats(world, second_card)

        self._draw_card(third_card, "Controls")
        self._draw_controls(world, third_card)

    def _draw_selected_creature(self, world: World, bounds: arcade.Rect) -> None:
        selected = world.selected_creature
        lines: list[str]
        if selected is None:
            lines = [
                "No creature selected.",
                "Click inside the environment",
                "to inspect a herbivore.",
            ]
        else:
            lines = [
                selected.name,
                f"Energy: {selected.energy:.0%}",
                f"Drift speed: {selected.drift_speed:.2f}",
                f"Heading speed: {selected.heading_speed:.2f}",
            ]

        y = bounds.top - 50
        line_index = 0
        for line in lines:
            self._draw_text(
                f"selected_line_{line_index}",
                line,
                bounds.left + 16,
                y,
                self.theme.text_primary
                if y == bounds.top - 50
                else self.theme.text_muted,
                12,
                bold=y == bounds.top - 50,
            )
            y -= 26
            line_index += 1

    def _draw_environment_stats(self, world: World, bounds: arcade.Rect) -> None:
        lines = [
            f"Herbivores: {world.stats.herbivore_count}",
            f"Food nodes: {world.stats.food_count}",
            f"Elapsed time: {world.elapsed_time:0.1f}s",
            f"Zoom: {world.environment_zoom:.2f}x",
            world.stats.generation_label,
        ]
        y = bounds.top - 50
        line_index = 0
        for line in lines:
            self._draw_text(
                f"stats_line_{line_index}",
                line,
                bounds.left + 16,
                y,
                self.theme.text_muted,
                12,
            )
            y -= 24
            line_index += 1

    def _draw_controls(self, world: World, bounds: arcade.Rect) -> None:
        lines = [
            "Click a herbivore to select it.",
            f"Press {self.config.debug.vision_toggle_label} to toggle vision.",
            "Sidebar reserved for future controls.",
        ]
        y = bounds.top - 50
        line_index = 0
        for line in lines:
            self._draw_text(
                f"controls_line_{line_index}",
                line,
                bounds.left + 16,
                y,
                self.theme.text_muted,
                11,
                width=bounds.width - 32,
                multiline=True,
            )
            y -= 28
            line_index += 1

    def _draw_panel(
        self, bounds: arcade.Rect, fill_color: arcade.Color | tuple[int, ...] | None = None
    ) -> None:
        self._draw_rounded_rect(
            bounds,
            fill_color or self.theme.panel_background,
            self.theme.panel_border,
            self.config.layout.panel_radius,
            2,
        )

    def _draw_card(self, bounds: arcade.Rect, title: str) -> None:
        self._draw_rounded_rect(
            bounds,
            self.theme.card_background,
            self.theme.panel_border,
            self.config.layout.card_radius,
            2,
        )
        self._draw_text(
            f"card_title_{title}",
            title,
            bounds.left + 16,
            bounds.top - 24,
            self.theme.text_primary,
            14,
            bold=True,
        )

    def _draw_text(
        self,
        key: str,
        text: str,
        x: float,
        y: float,
        color: arcade.Color | tuple[int, ...],
        size: float,
        *,
        bold: bool = False,
        width: float | None = None,
        multiline: bool = False,
    ) -> None:
        rx = round(x)
        ry = round(y)
        cached = self._text_cache.get(key)
        if cached is None:
            cached = arcade.Text(
                text,
                rx,
                ry,
                color,
                size,
                font_name=("Verdana", "DejaVu Sans", "Arial"),
                bold=bold,
                width=width,
                multiline=multiline,
            )
            self._text_cache[key] = cached
        else:
            cached.text = text
            cached.x = rx
            cached.y = ry
            cached.color = color
            cached.font_size = size
            cached.bold = bold
            cached.width = width
            cached.multiline = multiline
        cached.draw()

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
