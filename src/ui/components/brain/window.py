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

class BrainWindowComponent:
    """Group related behavior extracted from ``UiRenderer``."""

    BRAIN_INSPECTOR_MIN_WIDTH = 320.0
    BRAIN_INSPECTOR_MAX_WIDTH = 420.0
    BRAIN_INSPECTOR_WIDTH_RATIO = 0.28

    def _draw_brain_window(self, world: World) -> None:
        """Draw brain window.

        Parameters
        ----------
        world
            Simulation world providing current state.
        """
        if not self._brain_window_open:
            return

        selected = world.selected_creature
        if selected is None:
            self._close_brain_window()
            return

        brain = world.neat_controller.brain_for(selected.creature_id)
        self._ensure_brain_window_bounds(world)
        bounds = self._brain_window_bounds
        if bounds is None:
            return

        self._draw_rounded_rect(
            bounds,
            self.theme.panel_background,
            self.theme.panel_border,
            14,
            1.5,
        )

        header_height = 64.0
        header = arcade.LBWH(
            bounds.left + 1.5,
            bounds.top - header_height,
            bounds.width - 3.0,
            header_height - 1.5,
        )
        self._draw_rounded_rect_fill(header, self.theme.panel_background, 12.5)
        arcade.draw_line(
            bounds.left,
            header.bottom,
            bounds.right,
            header.bottom,
            self.theme.panel_border,
            1.0,
        )
        close_button = arcade.LBWH(bounds.right - 48, bounds.top - 45, 28, 28)
        self._control_hitboxes["brain_window_close"] = close_button
        genome_label = f"Genome {brain.genome_id}" if brain is not None else "No genome"
        self._draw_text(
            "brain_window_title_text",
            f"Brain: {getattr(selected, 'name', selected.creature_id)}  /  {genome_label}",
            bounds.left + 26,
            bounds.top - 33,
            self.theme.text_primary,
            17,
            bold=True,
            anchor_y="center",
        )
        self._draw_panel_close_button(close_button, "brain_window")

        footer_bounds = arcade.LBWH(
            bounds.left + 20,
            bounds.bottom + 16,
            bounds.width - 40,
            72,
        )
        body_bounds = arcade.LBWH(
            bounds.left + 20,
            footer_bounds.top + 14,
            bounds.width - 40,
            max(120.0, header.bottom - 14 - (footer_bounds.top + 14)),
        )

        inspector_bounds: arcade.Rect | None = None
        gap = 14.0
        if self._brain_node_inspector_open:
            desired_width = max(
                self.BRAIN_INSPECTOR_MIN_WIDTH,
                min(
                    self.BRAIN_INSPECTOR_MAX_WIDTH,
                    bounds.width * self.BRAIN_INSPECTOR_WIDTH_RATIO,
                ),
            )
            inspector_width = min(
                desired_width,
                max(220.0, body_bounds.width - 360.0),
            )
            inspector_bounds = arcade.LBWH(
                body_bounds.right - inspector_width,
                body_bounds.bottom,
                inspector_width,
                body_bounds.height,
            )
            main_right = inspector_bounds.left - gap
        else:
            toggle_bounds = arcade.LBWH(
                body_bounds.right - 34,
                body_bounds.bottom,
                34,
                body_bounds.height,
            )
            self._control_hitboxes["brain_node_inspector_toggle"] = toggle_bounds
            self._draw_rounded_rect(
                toggle_bounds,
                self.theme.card_background,
                self.theme.panel_border,
                8,
                1,
            )
            self._draw_text(
                "brain_node_inspector_reopen",
                ">",
                toggle_bounds.center_x,
                toggle_bounds.center_y,
                self.theme.accent,
                16,
                bold=True,
                anchor_x="center",
                anchor_y="center",
            )
            main_right = toggle_bounds.left - gap

        main_bounds = arcade.LBWH(
            body_bounds.left,
            body_bounds.bottom,
            max(220.0, main_right - body_bounds.left),
            body_bounds.height,
        )
        desired_legend_width = max(160.0, min(210.0, bounds.width * 0.16))
        legend_width = min(
            desired_legend_width,
            max(120.0, main_bounds.width - 320.0),
        )
        legend_bounds = arcade.LBWH(
            main_bounds.right - legend_width,
            main_bounds.bottom,
            legend_width,
            main_bounds.height,
        )
        graph_bounds = arcade.LBWH(
            main_bounds.left,
            main_bounds.bottom,
            max(180.0, legend_bounds.left - gap - main_bounds.left),
            main_bounds.height,
        )
        self._control_hitboxes["brain_window_graph"] = graph_bounds

        layout: BrainGraphLayout | None = None
        if brain is None:
            self._clear_brain_render_caches()
            self._brain_selection_identity = None
            self._brain_selected_node_key = None
            self._draw_rounded_rect(
                graph_bounds,
                self.theme.card_background,
                self.theme.panel_border,
                self.config.layout.card_radius,
                1.0,
            )
            self._draw_text(
                "brain_window_empty",
                "No brain assigned.",
                graph_bounds.left + 18,
                graph_bounds.top - 28,
                self.theme.text_muted,
                13,
            )
        else:
            layout = self._draw_brain_graph(world, graph_bounds)

        self._draw_brain_legend(legend_bounds)
        if inspector_bounds is not None:
            self._draw_brain_side_inspector(
                world,
                brain,
                layout,
                inspector_bounds,
            )
        if brain is not None:
            self._draw_brain_footer(world, selected, brain, footer_bounds)
    def _sync_brain_graph_selection(
        self,
        selected: object,
        brain: object,
        layout: BrainGraphLayout,
    ) -> None:
        """Synchronize brain graph selection.

        Parameters
        ----------
        selected
            Value used by the operation.
        brain
            Value used by the operation.
        layout
            Value used by the operation.
        """
        identity = (int(selected.creature_id), int(brain.genome_id))
        if identity != self._brain_selection_identity:
            self._brain_selection_identity = identity
            self._brain_selected_node_key = None
            self._scroll_offsets["brain_node_inspector"] = 0.0
            self._clear_brain_selection_caches()
        elif (
            self._brain_selected_node_key is not None
            and self._brain_selected_node_key not in layout.nodes
        ):
            self._brain_selected_node_key = None
            self._scroll_offsets["brain_node_inspector"] = 0.0
    def _close_brain_window(self) -> None:
        """Close brain window.
        """
        self._brain_window_open = False
        self._brain_selected_node_key = None
        self._brain_selection_identity = None
        self._brain_inspector_page = "node"
        self._brain_node_bounds.clear()
        self._scroll_offsets["brain_node_inspector"] = 0.0
        self._scroll_offsets["brain_behavior_inspector"] = 0.0
        self._brain_behavior_scroll_offset = 0.0
        self._brain_expanded_behavior = None
        self._clear_brain_render_caches()
    def _clear_brain_selection_caches(self) -> None:
        """Release cached data derived from the selected graph node."""
        state = self._brain_state
        state.highlight_layout = None
        state.highlight_node_key = None
        state.highlight = None
        state.inspector_brain = None
        state.inspector_layout = None
        state.inspector_node_key = None
        state.inspector_lines = ()
    def _clear_brain_render_caches(self) -> None:
        """Release cached brain layout, selection, and wrapped content."""
        state = self._brain_state
        state.layout_cache_key = None
        state.layout = None
        self._clear_brain_selection_caches()
        self._painter.wrapped_line_block_cache.clear()
        self._painter.curve_cache.clear()
    def _brain_node_at(self, x: float, y: float) -> int | None:
        """Return brain node at.

        Parameters
        ----------
        x
            Logical screen coordinate.
        y
            Logical screen coordinate.

        Returns
        -------
        int | None
            Computed result.
        """
        candidates = (
            (node_key, node_bounds)
            for node_key, node_bounds in self._brain_node_bounds.items()
            if self._contains_bounds(node_bounds, x, y)
        )
        nearest = min(
            candidates,
            key=lambda item: (
                (item[1].center_x - x) ** 2 + (item[1].center_y - y) ** 2
            ),
            default=None,
        )
        return None if nearest is None else nearest[0]
    def _ensure_brain_window_bounds(self, world: World) -> None:
        """Ensure brain window bounds.

        Parameters
        ----------
        world
            Simulation world providing current state.
        """
        window = world.layout.window
        shorter_side = max(1.0, min(window.width, window.height))
        margin = max(
            8.0,
            min(self.config.layout.outer_padding * 2.0, shorter_side * 0.05),
        )
        self._brain_window_bounds = arcade.LBWH(
            window.left + margin,
            window.bottom + margin,
            max(1.0, window.width - margin * 2.0),
            max(1.0, window.height - margin * 2.0),
        )
    def _brain_graph_screen_position(
        self,
        position: tuple[float, float],
        bounds: arcade.Rect,
    ) -> tuple[float, float]:
        """Return brain graph screen position.

        Parameters
        ----------
        position
            Value used by the operation.
        bounds
            Rectangle defining the relevant UI area.

        Returns
        -------
        tuple[float, float]
            Computed collection.
        """
        return (
            bounds.center_x + (position[0] - bounds.center_x) * self._brain_graph_zoom,
            bounds.center_y + (position[1] - bounds.center_y) * self._brain_graph_zoom,
        )
