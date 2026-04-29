from __future__ import annotations

import arcade

from configs.sim_config import SimConfig
from src.world import World


class UiRenderer:
    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.theme = config.theme

    def draw(self, world: World) -> None:
        self._draw_top_bar(world)
        self._draw_sidebar(world)
        self._draw_bottom_bar(world)

    def _draw_top_bar(self, world: World) -> None:
        bounds = world.layout.top_bar
        self._draw_panel(bounds)

        arcade.draw_text(
            "NEAT Game Of Life",
            bounds.left + 18,
            bounds.top - 34,
            self.theme.text_primary,
            24,
            bold=True,
        )
        arcade.draw_text(
            "Window container with nested environment + UI panels",
            bounds.left + 18,
            bounds.top - 60,
            self.theme.text_muted,
            12,
        )

        status = "Debug vision on" if world.debug_vision_enabled else "Debug vision off"
        arcade.draw_text(
            status,
            bounds.right - 180,
            bounds.top - 40,
            self.theme.accent,
            13,
            bold=True,
        )

    def _draw_sidebar(self, world: World) -> None:
        bounds = world.layout.right_sidebar
        self._draw_panel(bounds)

        title_y = bounds.top - 28
        arcade.draw_text(
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

    def _draw_bottom_bar(self, world: World) -> None:
        bounds = world.layout.bottom_bar
        self._draw_panel(bounds)

        arcade.draw_text(
            "Prototype Notes",
            bounds.left + 18,
            bounds.top - 28,
            self.theme.text_primary,
            18,
            bold=True,
        )
        arcade.draw_text(
            "The center region is reserved for the actual ecosystem. The surrounding panels are ready for stats, debug controls, and future trait graphs.",
            bounds.left + 18,
            bounds.top - 58,
            self.theme.text_muted,
            12,
            width=bounds.width - 36,
            multiline=True,
        )
        arcade.draw_text(
            f"Window: {int(world.layout.window.width)} x {int(world.layout.window.height)}    "
            f"Environment: {int(world.layout.environment.width)} x {int(world.layout.environment.height)}",
            bounds.left + 18,
            bounds.bottom + 20,
            self.theme.accent,
            11,
        )

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
        for line in lines:
            arcade.draw_text(
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

    def _draw_environment_stats(self, world: World, bounds: arcade.Rect) -> None:
        lines = [
            f"Herbivores: {world.stats.herbivore_count}",
            f"Food nodes: {world.stats.food_count}",
            f"Elapsed time: {world.elapsed_time:0.1f}s",
            world.stats.generation_label,
        ]
        y = bounds.top - 50
        for line in lines:
            arcade.draw_text(
                line,
                bounds.left + 16,
                y,
                self.theme.text_muted,
                12,
            )
            y -= 24

    def _draw_controls(self, world: World, bounds: arcade.Rect) -> None:
        lines = [
            "Click a herbivore to select it.",
            f"Press {self.config.debug.vision_toggle_label} to toggle vision.",
            "Sidebar reserved for future controls.",
        ]
        y = bounds.top - 50
        for line in lines:
            arcade.draw_text(
                line,
                bounds.left + 16,
                y,
                self.theme.text_muted,
                11,
                width=bounds.width - 32,
                multiline=True,
            )
            y -= 28

    def _draw_panel(self, bounds: arcade.Rect) -> None:
        arcade.draw_lrbt_rectangle_filled(
            bounds.left,
            bounds.right,
            bounds.bottom,
            bounds.top,
            self.theme.panel_background,
        )
        arcade.draw_lrbt_rectangle_outline(
            bounds.left,
            bounds.right,
            bounds.bottom,
            bounds.top,
            self.theme.panel_border,
            border_width=3,
        )

    def _draw_card(self, bounds: arcade.Rect, title: str) -> None:
        arcade.draw_lrbt_rectangle_filled(
            bounds.left,
            bounds.right,
            bounds.bottom,
            bounds.top,
            self.theme.card_background,
        )
        arcade.draw_lrbt_rectangle_outline(
            bounds.left,
            bounds.right,
            bounds.bottom,
            bounds.top,
            self.theme.panel_border,
            border_width=2,
        )
        arcade.draw_text(
            title,
            bounds.left + 16,
            bounds.top - 24,
            self.theme.text_primary,
            14,
            bold=True,
        )
