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

    LIVE_FOOD_SLIDERS = (
        (
            "Biome fertility",
            (
                ("forest_spawn_weight", "Forest", 0.0, 5.0, 0.05, 2),
                ("bushes_spawn_weight", "Bushes", 0.0, 5.0, 0.05, 2),
                ("prairie_spawn_weight", "Prairie", 0.0, 5.0, 0.05, 2),
            ),
        ),
        (
            "Capacity",
            (("max_food_items", "Maximum food", 0.0, 2000.0, 1.0, 0),),
        ),
        (
            "Low-food bursts",
            (
                (
                    "low_food_pressure_threshold",
                    "Low-food threshold",
                    0.0,
                    1.0,
                    0.01,
                    2,
                ),
                (
                    "critical_food_ratio",
                    "Critical ratio",
                    0.0,
                    1.0,
                    0.01,
                    2,
                ),
                ("low_food_burst_items", "Burst items", 0.0, 2000.0, 1.0, 0),
                (
                    "low_food_burst_interval",
                    "Burst interval (s)",
                    0.05,
                    10.0,
                    0.05,
                    2,
                ),
            ),
        ),
    )
    LIVE_FOOD_SLIDER_FIELDS = tuple(
        field_name
        for _section, rows in LIVE_FOOD_SLIDERS
        for field_name, _label, _minimum, _maximum, _step, _precision in rows
    )

    def _draw_settings_panel(self, world: World) -> None:
        """Draw settings panel.

        Parameters
        ----------
        world
            Simulation world providing current state.
        """
        self._control_hitboxes.pop("reset_speed", None)
        for field_name in self.LIVE_FOOD_SLIDER_FIELDS:
            self._control_hitboxes.pop(
                self._live_food_slider_key(field_name),
                None,
            )
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

        divider_y = content.bottom + 100.0
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

        # Keep the expander in a dedicated padded footer row. The divider
        # separates it from simulation controls and the keyboard hints stay
        # below it, so neither label can collide with the button.
        toggle_horizontal_padding = 16.0
        toggle_vertical_padding = 14.0
        toggle_height = 30.0
        toggle_width = min(
            156.0,
            max(0.0, content.width - toggle_horizontal_padding * 2.0),
        )
        toggle = arcade.LBWH(
            content.center_x - toggle_width / 2.0,
            divider_y - toggle_vertical_padding - toggle_height,
            toggle_width,
            toggle_height,
        )
        self._control_hitboxes["settings_food_toggle"] = toggle
        self._draw_rounded_rect(
            toggle,
            self.theme.card_background,
            self.theme.panel_border,
            7.0,
            1.0,
        )
        self._draw_text(
            "settings_food_toggle_label",
            f"Food settings {'^' if self._settings_expanded else 'v'}",
            toggle.center_x,
            toggle.center_y,
            self.theme.text_primary,
            10,
            bold=True,
            anchor_x="center",
            anchor_y="center",
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
        if self._settings_expanded:
            food_viewport = arcade.LBWH(
                content.left,
                divider_y + 12.0,
                content.width,
                max(0.0, content.top - 82.0 - divider_y - 12.0),
            )
            self._draw_live_food_controls(world, food_viewport)

    def _draw_live_food_controls(
        self,
        world: World,
        viewport: arcade.Rect,
    ) -> None:
        """Draw scrollable live food settings inside ``viewport``.

        Parameters
        ----------
        world
            Simulation world providing current values.
        viewport
            Clipped rectangle available for food controls.
        """
        settings = world.live_food_config
        section_height = 30.0
        row_height = 58.0
        content_height = sum(
            section_height + len(rows) * row_height
            for _title, rows in self.LIVE_FOOD_SLIDERS
        )
        scroll_limit = max(0.0, content_height - viewport.height + 8.0)
        scroll_offset = max(
            0.0,
            min(
                scroll_limit,
                self._scroll_offsets.get("settings_food", 0.0),
            ),
        )
        self._scroll_offsets["settings_food"] = scroll_offset
        self._scroll_limits["settings_food"] = scroll_limit
        self._scroll_regions["settings_food"] = viewport

        with self._ui_clip(viewport):
            y = viewport.top + scroll_offset
            for section_index, (section_title, rows) in enumerate(
                self.LIVE_FOOD_SLIDERS
            ):
                heading_y = y - 18.0
                if viewport.bottom <= heading_y <= viewport.top:
                    self._draw_text(
                        f"settings_food_section_{section_index}",
                        section_title,
                        viewport.left + 8.0,
                        heading_y,
                        self.theme.text_primary,
                        12,
                        bold=True,
                    )
                y -= section_height
                for (
                    field_name,
                    label,
                    minimum,
                    maximum,
                    _step,
                    precision,
                ) in rows:
                    row_center = y - row_height / 2.0
                    slider = arcade.LBWH(
                        viewport.left + 188.0,
                        row_center - 9.0,
                        max(40.0, viewport.width - 210.0),
                        18.0,
                    )
                    if (
                        slider.bottom >= viewport.bottom
                        and slider.top <= viewport.top
                    ):
                        self._control_hitboxes[
                            self._live_food_slider_key(field_name)
                        ] = slider
                    if (
                        row_center + 19.0 >= viewport.bottom
                        and row_center - 19.0 <= viewport.top
                    ):
                        value = getattr(settings, field_name)
                        formatted = (
                            str(int(value))
                            if precision == 0
                            else f"{float(value):.{precision}f}"
                        )
                        self._draw_text(
                            f"settings_food_label_{field_name}",
                            label,
                            viewport.left + 12.0,
                            row_center + 7.0,
                            self.theme.text_primary,
                            10,
                            bold=True,
                        )
                        self._draw_text(
                            f"settings_food_value_{field_name}",
                            formatted,
                            viewport.left + 12.0,
                            row_center - 11.0,
                            self.theme.text_muted,
                            10,
                        )
                        self._draw_live_food_slider(
                            slider,
                            float(value),
                            minimum,
                            maximum,
                        )
                    y -= row_height

        if scroll_limit > 0.0:
            self._draw_scrollbar(viewport, scroll_offset, scroll_limit)

    def _draw_live_food_slider(
        self,
        bounds: arcade.Rect,
        value: float,
        minimum: float,
        maximum: float,
    ) -> None:
        """Draw a normalized live food value slider.

        Parameters
        ----------
        bounds
            Slider hit and drawing bounds.
        value
            Current configuration value.
        minimum, maximum
            Inclusive slider range.
        """
        ratio = (value - minimum) / max(0.000001, maximum - minimum)
        ratio = max(0.0, min(1.0, ratio))
        knob_x = bounds.left + bounds.width * ratio
        track_bottom = bounds.center_y - 3.0
        arcade.draw_lrbt_rectangle_filled(
            bounds.left,
            bounds.right,
            track_bottom,
            track_bottom + 6.0,
            self.theme.panel_border,
        )
        arcade.draw_lrbt_rectangle_filled(
            bounds.left,
            knob_x,
            track_bottom,
            track_bottom + 6.0,
            self.theme.accent,
        )
        arcade.draw_circle_filled(
            knob_x,
            bounds.center_y,
            7.0,
            self.theme.accent_soft,
        )
        arcade.draw_circle_outline(
            knob_x,
            bounds.center_y,
            7.0,
            self.theme.accent,
            2,
        )

    @staticmethod
    def _live_food_slider_key(field_name: str) -> str:
        """Return the stable hitbox key for a live food field."""
        return f"food_slider_{field_name}"

    @staticmethod
    def _live_food_field_from_slider_key(key: str) -> str | None:
        """Return the field encoded in a live food slider hitbox key."""
        prefix = "food_slider_"
        return key[len(prefix):] if key.startswith(prefix) else None

    def _live_food_slider_spec(
        self,
        field_name: str,
    ) -> tuple[float, float, float, int]:
        """Return range, step, and precision metadata for ``field_name``."""
        for _section, rows in self.LIVE_FOOD_SLIDERS:
            for name, _label, minimum, maximum, step, precision in rows:
                if name == field_name:
                    return minimum, maximum, step, precision
        raise KeyError(field_name)

    def _set_live_food_from_slider(
        self,
        world: World,
        field_name: str,
        x: float,
    ) -> None:
        """Snap a pointer position and apply its live configuration value.

        Parameters
        ----------
        world
            Simulation world receiving the update.
        field_name
            Live configuration field represented by the slider.
        x
            Horizontal pointer position.
        """
        bounds = self._control_hitboxes[
            self._live_food_slider_key(field_name)
        ]
        minimum, maximum, step, precision = self._live_food_slider_spec(
            field_name
        )
        ratio = max(0.0, min(1.0, (x - bounds.left) / bounds.width))
        step_count = round((ratio * (maximum - minimum)) / step)
        value = minimum + step_count * step
        value = max(minimum, min(maximum, value))
        normalized: int | float = (
            int(round(value)) if precision == 0 else round(value, precision)
        )
        world.set_live_food_config_value(field_name, normalized)
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
