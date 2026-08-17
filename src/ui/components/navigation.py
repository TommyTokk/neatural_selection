from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from math import ceil, cos, floor, hypot, isfinite, log1p, log10, pi, sin
from pathlib import Path
import re

import arcade

from configs.sim_config import SimConfig
from src.creature.action import ACTION_OUTPUT_NAMES
from src.analysis import (
    BEHAVIOR_RADAR_LABELS,
    InspectorReport,
    calculate_behavior_scores,
    generate_inspector_report,
    generate_radar_chart_image,
    profile_morphology,
)
from src.creature.speciation import SpeciesRecord
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
from src.creature.vision import SENSOR_INPUT_NAMES
from src.world import World

_EMPTY_NEAT_NODE_LABELS: dict[int, str] = {}

class NavigationComponent:
    """Group related behavior extracted from ``UiRenderer``."""

    def _draw_icon_rail(self, world: World) -> None:
        """Draw icon rail.

        Parameters
        ----------
        world
            Simulation world providing current state.
        """
        bounds = world.layout.left_sidebar
        self._draw_rounded_rect(
            bounds,
            self.theme.panel_background_alt,
            self.theme.panel_border,
            14,
            1.5,
        )
        self._control_hitboxes["icon_rail"] = bounds

        self._control_hitboxes.pop("open_behavior_report", None)
        button_count = 6
        button_size = self.ICON_BUTTON_SIZE
        button_gap = self.ICON_BUTTON_GAP
        available_height = max(0.0, bounds.height - self.RAIL_VERTICAL_PADDING)
        preferred_buttons_height = (
            button_size * button_count + button_gap * (button_count - 1)
        )
        if preferred_buttons_height > available_height:
            button_gap = min(button_gap, max(4.0, available_height * 0.05))
            button_size = min(
                button_size,
                max(
                    24.0,
                    (available_height - button_gap * (button_count - 1))
                    / button_count,
                ),
            )
        top = bounds.top - self.RAIL_VERTICAL_PADDING / 2.0 - button_size / 2.0
        step = button_size + button_gap
        environment_map_mode = self._environment_map_mode(world)
        icon_buttons = (
            (
                "panel_toggle_inspector",
                "search",
                self._panel_open["inspector"],
                top,
            ),
            (
                "panel_toggle_stats",
                "analytics",
                self._panel_open["stats"],
                top - step,
            ),
            (
                "panel_toggle_settings",
                "tune",
                self._panel_open["settings"],
                top - step * 2,
            ),
            (
                "open_map_submenu",
                "globe",
                self._map_submenu_open or environment_map_mode != "none",
                top - step * 3,
            ),
            (
                "save_simulation",
                "save_sim",
                getattr(world, "save_in_progress", False),
                top - step * 4,
            ),
            (
                "open_species_tree",
                "species",
                self._species_tree_open,
                top - step * 5,
            ),
        )
        map_button = None
        for key, icon_name, active, center_y in icon_buttons:
            button = arcade.LBWH(
                bounds.center_x - button_size / 2.0,
                center_y - button_size / 2.0,
                button_size,
                button_size,
            )
            self._control_hitboxes[key] = button
            if key == "open_map_submenu":
                map_button = button
            self._draw_icon_button(
                button,
                icon_name,
                key,
                active=active,
            )
        if self._map_submenu_open and map_button is not None:
            self._draw_map_submenu(world, map_button)
    @staticmethod
    def _environment_map_mode(world: World) -> str:
        """Return environment map mode.

        Parameters
        ----------
        world
            Simulation world providing current state.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        mode = getattr(world, "environment_map_mode", None)
        if mode in {"none", "biome", "pheromones"}:
            return mode
        return (
            "biome"
            if getattr(world, "show_biome_background", False)
            else "none"
        )
    def _draw_map_submenu(
        self,
        world: World,
        anchor: arcade.Rect,
    ) -> None:
        """Draw map submenu.

        Parameters
        ----------
        world
            Simulation world providing current state.
        anchor
            Rectangle defining the relevant UI area.
        """
        window = world.layout.window
        padding = 8.0
        row_height = 48.0
        row_gap = 6.0
        width = min(184.0, max(142.0, window.width - 16.0))
        height = padding * 2.0 + row_height * 2.0 + row_gap
        left = min(anchor.right + 10.0, window.right - width - 8.0)
        left = max(window.left + 8.0, left)
        bottom = anchor.center_y - height / 2.0
        bottom = min(bottom, window.top - height - 8.0)
        bottom = max(window.bottom + 8.0, bottom)
        card = arcade.LBWH(left, bottom, width, height)
        self._control_hitboxes["map_submenu"] = card
        self._draw_rounded_rect(
            card,
            self.theme.panel_background,
            self.theme.panel_border,
            12.0,
            1.5,
        )

        mode = self._environment_map_mode(world)
        rows = (
            ("map_layer_biome", "biome_map", "Biome Map", "biome"),
            (
                "map_layer_pheromones",
                "pheromone_map",
                "Pheromones",
                "pheromones",
            ),
        )
        row_top = card.top - padding
        for index, (key, icon_name, label, row_mode) in enumerate(rows):
            row = arcade.LBWH(
                card.left + padding,
                row_top - row_height - index * (row_height + row_gap),
                card.width - padding * 2.0,
                row_height,
            )
            active = mode == row_mode
            self._control_hitboxes[key] = row
            self._draw_rounded_rect(
                row,
                (
                    self.theme.accent_soft
                    if active
                    else self.theme.panel_background_alt
                ),
                self.theme.accent if active else self.theme.panel_border,
                8.0,
                1.0,
            )
            icon_size = 30.0
            self._draw_icon(
                arcade.LBWH(
                    row.left + 9.0,
                    row.center_y - icon_size / 2.0,
                    icon_size,
                    icon_size,
                ),
                icon_name,
                key,
            )
            self._draw_text(
                f"{key}_label",
                label,
                row.left + 48.0,
                row.center_y,
                self.theme.text_primary,
                11.0,
                bold=active,
                anchor_y="center",
            )
