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

class PanelManagerComponent:
    """Group related behavior extracted from ``UiRenderer``."""

    def _draw_floating_panels(self, world: World) -> None:
        """Draw floating panels.

        Parameters
        ----------
        world
            Simulation world providing current state.
        """
        if self._panel_open["stats"]:
            self._draw_stats_panel(world)
        if self._panel_open["inspector"]:
            self._draw_inspector_panel(world)
        if self._panel_open["settings"]:
            self._draw_settings_panel(world)
    def _stats_panel_bounds(self, world: World) -> arcade.Rect:
        """Return stats panel bounds.

        Parameters
        ----------
        world
            Simulation world providing current state.

        Returns
        -------
        arcade.Rect
            Computed UI rectangle.
        """
        width = min(330.0, max(270.0, world.layout.window.width - 140.0))
        height = min(238.0, max(190.0, world.layout.window.height * 0.28))
        default_bounds = arcade.LBWH(
            world.layout.window.right - width - 18.0,
            world.layout.window.top - height - 18.0,
            width,
            height,
        )
        return self._panel_bounds_for(world, "stats", default_bounds)
    def _inspector_panel_bounds(self, world: World) -> arcade.Rect:
        """Return inspector panel bounds.

        Parameters
        ----------
        world
            Simulation world providing current state.

        Returns
        -------
        arcade.Rect
            Computed UI rectangle.
        """
        width = min(368.0, max(310.0, world.layout.window.width - 148.0))
        height = min(414.0, max(330.0, world.layout.window.height * 0.48))
        default_bounds = arcade.LBWH(
            world.layout.window.right - width - 18.0,
            max(18.0, world.layout.window.center_y - height * 0.38),
            width,
            height,
        )
        return self._panel_bounds_for(world, "inspector", default_bounds)
    def _settings_panel_bounds(self, world: World) -> arcade.Rect:
        """Return settings panel bounds.

        Parameters
        ----------
        world
            Simulation world providing current state.

        Returns
        -------
        arcade.Rect
            Computed UI rectangle.
        """
        width = min(540.0, max(400.0, world.layout.window.width - 160.0))
        height = 164.0
        default_bounds = arcade.LBWH(
            world.layout.window.center_x - width / 2.0,
            32.0,
            width,
            height,
        )
        return self._panel_bounds_for(world, "settings", default_bounds)
    def _panel_bounds_for(
        self,
        world: World,
        key: str,
        default_bounds: arcade.Rect,
    ) -> arcade.Rect:
        """Return panel bounds for.

        Parameters
        ----------
        world
            Simulation world providing current state.
        key
            Stable identifier used by the UI.
        default_bounds
            Value used by the operation.

        Returns
        -------
        arcade.Rect
            Computed UI rectangle.
        """
        bounds = self._panel_bounds.get(key, default_bounds)
        bounds = self._clamp_panel_bounds(world, bounds)
        self._panel_bounds[key] = bounds
        return bounds
    def _clamp_panel_bounds(self, world: World, bounds: arcade.Rect) -> arcade.Rect:
        """Return clamp panel bounds.

        Parameters
        ----------
        world
            Simulation world providing current state.
        bounds
            Rectangle defining the relevant UI area.

        Returns
        -------
        arcade.Rect
            Computed UI rectangle.
        """
        margin = float(self.config.layout.outer_padding)
        max_left = max(margin, world.layout.window.width - margin - bounds.width)
        max_bottom = max(margin, world.layout.window.height - margin - bounds.height)
        return arcade.LBWH(
            max(margin, min(max_left, bounds.left)),
            max(margin, min(max_bottom, bounds.bottom)),
            bounds.width,
            bounds.height,
        )

