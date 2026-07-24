from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from math import ceil, cos, floor, hypot, isfinite, log1p, log10, pi, sin
from pathlib import Path
import re

import arcade

from configs.sim_config import SimConfig
from src.action import ACTION_OUTPUT_NAMES
from src.analysis import (
    BEHAVIOR_RADAR_LABELS,
    InspectorReport,
    calculate_behavior_scores,
    generate_inspector_report,
    generate_radar_chart_image,
    profile_morphology,
)
from src.speciation import SpeciesRecord
from src.ui.common.interaction import rect_contains
from src.ui.components.state import (
    SpeciesInspectorRow as _SpeciesInspectorRow,
    SpeciesInspectorSection as _SpeciesInspectorSection,
    SpeciesTreeLabel as _SpeciesTreeLabel,
)
from src.ui.layouts.brain_graph import (
    BrainEdgeKind,
    BrainGraphEdge,
    BrainGraphLayout,
    BrainGraphNode,
    BrainNodeKind,
    build_brain_graph_layout,
    highlighted_path_through_node,
)
from src.ui.layouts.species_tree import (
    SpeciesTreeLayout,
    SpeciesTreeRoute,
    TreeLayoutManager,
    TreeViewportSlice,
    species_tree_line_width,
)
from src.vision import SENSOR_INPUT_NAMES
from src.world import World

_EMPTY_NEAT_NODE_LABELS: dict[int, str] = {}

class SettingsPanelComponent:
    """Group related behavior extracted from ``UiRenderer``."""

    def _draw_settings_panel(self, world: World) -> None:
        """Draw settings panel.

        Parameters
        ----------
        world
            Simulation world providing current state.
        """
        self._control_hitboxes.pop("reset_speed", None)
        bounds = self._settings_panel_bounds(world)
        content = self._draw_floating_panel(
            bounds,
            "",
            "settings",
            body_top_padding=14.0,
        )

        row_center_y = content.top - 33.0
        self._draw_text(
            "settings_speed_title",
            "Speed",
            content.left + 8.0,
            row_center_y + 10.0,
            self.theme.text_primary,
            13,
            bold=True,
        )
        self._draw_text(
            "settings_speed_value",
            f"{world.simulation_speed:.2f}x",
            content.left + 8.0,
            row_center_y - 8.0,
            self.theme.text_primary,
            12,
        )

        control_size = 26.0
        pause_size = 36.0
        control_gap = 12.0
        slider_gap = 26.0
        controls_right_padding = 8.0
        controls_width = control_size * 4 + pause_size + control_gap * 4
        start_x = content.right - controls_width - controls_right_padding
        slider_left = content.left + 90.0
        slider_right = start_x - slider_gap
        slider = arcade.LBWH(
            slider_left,
            row_center_y - 9.0,
            max(36.0, slider_right - slider_left),
            18,
        )
        self._control_hitboxes["speed_slider"] = slider
        self._draw_speed_slider(slider, world)

        controls = (
            ("speed_min", "<<"),
            ("speed_down", "<"),
            ("pause", ""),
            ("speed_up", ">"),
            ("speed_max", ">>"),
        )
        next_x = start_x
        for key, label in controls:
            size = pause_size if key == "pause" else control_size
            button = arcade.LBWH(
                next_x,
                row_center_y - size / 2.0,
                size,
                size,
            )
            self._control_hitboxes[key] = button
            if key == "pause":
                self._draw_play_pause_button(button, is_paused=world.is_paused)
            else:
                self._draw_icon_text_button(button, label, key, fill_color=None, size=14)
            next_x += size + control_gap

        divider_y = content.bottom + 50
        draw_line = getattr(arcade, "draw_line", None)
        if draw_line is not None:
            draw_line(
                content.left,
                divider_y,
                content.right,
                divider_y,
                self.theme.panel_border,
                1,
            )
        hints = (("SPACE", "PLAY/PAUSE"), ("A", "BACK"), ("D", "NEXT"))
        hint_positions = (
            content.left + 68,
            content.left + 238,
            content.left + 326,
        )
        for index, (key_label, label) in enumerate(hints):
            x = hint_positions[index]
            self._draw_keycap(
                f"settings_hint_{index}", key_label, label, x, content.bottom + 22
            )
    def _draw_controls(self, world: World, bounds: arcade.Rect) -> None:
        """Draw controls.

        Parameters
        ----------
        world
            Simulation world providing current state.
        bounds
            Rectangle defining the relevant UI area.
        """
        button_top = bounds.top - 50
        button_height = 32
        button_gap = 10
        button_width = (bounds.width - 32 - button_gap) / 2
        pause_button = arcade.LBWH(
            bounds.left + 16, button_top - button_height, button_width, button_height
        )
        reset_button = arcade.LBWH(
            pause_button.right + button_gap,
            button_top - button_height,
            button_width,
            button_height,
        )
        self._control_hitboxes["pause"] = pause_button
        self._control_hitboxes["reset_speed"] = reset_button
        self._draw_button(
            pause_button, "> Space" if world.is_paused else "|| Space", "pause"
        )
        self._draw_button(reset_button, "1x 0", "reset_speed")

        slider_y = reset_button.bottom - 42
        slider = arcade.LBWH(bounds.left + 16, slider_y, bounds.width - 32, 18)
        self._control_hitboxes["speed_slider"] = slider
        self._draw_speed_slider(slider, world)

        small_button_width = (bounds.width - 32 - button_gap) / 2
        small_button_top = slider.bottom - 28
        slow_button = arcade.LBWH(
            bounds.left + 16,
            small_button_top - button_height,
            small_button_width,
            button_height,
        )
        fast_button = arcade.LBWH(
            slow_button.right + button_gap,
            small_button_top - button_height,
            small_button_width,
            button_height,
        )
        self._control_hitboxes["speed_down"] = slow_button
        self._control_hitboxes["speed_up"] = fast_button
        self._draw_button(slow_button, "<< A/<-", "speed_down")
        self._draw_button(fast_button, ">> D/->", "speed_up")
    def _draw_play_pause_button(self, bounds: arcade.Rect, *, is_paused: bool) -> None:
        """Draw play pause button.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        is_paused
            Whether the corresponding behavior is enabled.
        """
        self._draw_rounded_rect(bounds, self.theme.accent_soft, self.theme.accent_soft, 8, 1)
        if is_paused:
            points = [
                (bounds.center_x - 5.0, bounds.center_y + 8.0),
                (bounds.center_x - 5.0, bounds.center_y - 8.0),
                (bounds.center_x + 8.0, bounds.center_y),
            ]
            draw_polygon = getattr(arcade, "draw_polygon_filled", None)
            if draw_polygon is not None:
                draw_polygon(points, self.theme.accent)
                return
            self._draw_text(
                "icon_text_button_pause",
                ">",
                bounds.center_x,
                bounds.center_y - 1.0,
                self.theme.accent,
                19,
                bold=True,
                anchor_x="center",
                anchor_y="center",
            )
            return

        bar_width = 4.0
        bar_height = 18.0
        gap = 5.0
        left_bar = arcade.LBWH(
            bounds.center_x - gap / 2.0 - bar_width,
            bounds.center_y - bar_height / 2.0,
            bar_width,
            bar_height,
        )
        right_bar = arcade.LBWH(
            bounds.center_x + gap / 2.0,
            bounds.center_y - bar_height / 2.0,
            bar_width,
            bar_height,
        )
        self._draw_rounded_rect_fill(left_bar, self.theme.accent, 1.5)
        self._draw_rounded_rect_fill(right_bar, self.theme.accent, 1.5)
    def _draw_keycap(
        self,
        key: str,
        key_label: str,
        label: str,
        x: float,
        y: float,
    ) -> None:
        """Draw keycap.

        Parameters
        ----------
        key
            Stable identifier used by the UI.
        key_label
            Value used by the operation.
        label
            Text displayed by the UI.
        x
            Logical screen coordinate.
        y
            Logical screen coordinate.
        """
        keycap = arcade.LBWH(x, y - 8, 42 if len(key_label) > 1 else 24, 22)
        self._draw_rounded_rect(
            keycap, self.theme.card_background, self.theme.panel_border, 4, 1
        )
        self._draw_text(
            f"{key}_key",
            key_label,
            keycap.center_x,
            keycap.center_y,
            self.theme.text_muted,
            8,
            anchor_x="center",
            anchor_y="center",
        )
        self._draw_text(
            f"{key}_label",
            label,
            keycap.right + 12,
            keycap.center_y,
            self.theme.text_muted,
            8,
            anchor_y="center",
        )
    def _draw_speed_slider(self, bounds: arcade.Rect, world: World) -> None:
        """Draw speed slider.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        world
            Simulation world providing current state.
        """
        track_height = 6
        track_bottom = bounds.center_y - track_height / 2
        ratio = (world.simulation_speed - world.MIN_SIMULATION_SPEED) / (
            world.MAX_SIMULATION_SPEED - world.MIN_SIMULATION_SPEED
        )
        knob_x = bounds.left + bounds.width * ratio

        arcade.draw_lrbt_rectangle_filled(
            bounds.left,
            bounds.right,
            track_bottom,
            track_bottom + track_height,
            self.theme.panel_border,
        )
        arcade.draw_lrbt_rectangle_filled(
            bounds.left,
            knob_x,
            track_bottom,
            track_bottom + track_height,
            self.theme.accent,
        )
        arcade.draw_circle_filled(knob_x, bounds.center_y, 8, self.theme.accent_soft)
        arcade.draw_circle_outline(knob_x, bounds.center_y, 8, self.theme.accent, 2)

        self._draw_text(
            "speed_min_label",
            f"{world.MIN_SIMULATION_SPEED:.2f}x",
            bounds.left,
            bounds.bottom - 13,
            self.theme.text_muted,
            9,
        )
        self._draw_text(
            "speed_value_label",
            f"{world.simulation_speed:.2f}x",
            bounds.center_x - 18,
            bounds.top + 7,
            self.theme.text_primary,
            10,
            bold=True,
        )
        self._draw_text(
            "speed_max_label",
            f"{world.MAX_SIMULATION_SPEED:.2f}x",
            bounds.right - 36,
            bounds.bottom - 13,
            self.theme.text_muted,
            9,
        )
    def _set_speed_from_slider(self, world: World, x: float) -> None:
        """Set speed from slider.

        Parameters
        ----------
        world
            Simulation world providing current state.
        x
            Logical screen coordinate.
        """
        bounds = self._control_hitboxes["speed_slider"]
        ratio = (x - bounds.left) / bounds.width
        ratio = max(0.0, min(1.0, ratio))
        speed = world.MIN_SIMULATION_SPEED + ratio * (
            world.MAX_SIMULATION_SPEED - world.MIN_SIMULATION_SPEED
        )
        world.set_simulation_speed(speed)

