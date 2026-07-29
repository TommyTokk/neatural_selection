from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Sequence
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
    BrainGraphHighlight,
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

class BrainGraphComponent:
    """Group related behavior extracted from ``UiRenderer``."""

    def _draw_brain_graph(
        self,
        world: World,
        bounds: arcade.Rect,
    ) -> BrainGraphLayout | None:
        """Draw brain graph.

        Parameters
        ----------
        world
            Simulation world providing current state.
        bounds
            Rectangle defining the relevant UI area.

        Returns
        -------
        BrainGraphLayout | None
            Computed result.
        """
        selected = world.selected_creature
        if selected is None:
            return None

        brain = world.neat_controller.brain_for(selected.creature_id)
        if brain is None:
            return None

        layout_bounds = arcade.LBWH(
            bounds.left + 8,
            bounds.bottom + 10,
            max(1.0, bounds.width - 16),
            max(1.0, bounds.height - 54),
        )

        input_keys = list(world.neat_controller.config.genome_config.input_keys)
        output_keys = list(world.neat_controller.config.genome_config.output_keys)
        layout = self._brain_graph_layout(
            selected,
            brain,
            input_keys,
            output_keys,
            layout_bounds,
        )
        lanes = self._draw_brain_graph_lanes(bounds, layout)
        self._sync_brain_graph_selection(selected, brain, layout)
        positions = self._brain_graph_node_positions(layout, lanes)
        base_radius, label_size = self._brain_graph_node_metrics(
            layout,
            layout_bounds,
        )
        selected_key = self._brain_selected_node_key
        highlight = (
            self._brain_highlight_for_node(layout, selected_key)
            if selected_key is not None
            else None
        )

        disabled_edges = [edge for edge in layout.edges if not edge.enabled]
        enabled_edges = [edge for edge in layout.edges if edge.enabled]
        if highlight is None:
            edge_groups = [
                (disabled_edges, True, False, False, False),
                (enabled_edges, False, False, False, False),
            ]
        else:
            direct_edges = highlight.direct_edges
            edge_groups = [
                (
                    [
                        edge
                        for edge in disabled_edges
                        if (edge.source, edge.target) not in direct_edges
                    ],
                    True,
                    False,
                    False,
                    False,
                ),
                (
                    [
                        edge
                        for edge in enabled_edges
                        if (edge.source, edge.target) not in highlight.edges
                    ],
                    False,
                    False,
                    True,
                    False,
                ),
                (
                    [
                        edge
                        for edge in enabled_edges
                        if (edge.source, edge.target) in highlight.edges
                        and (edge.source, edge.target) not in direct_edges
                    ],
                    False,
                    True,
                    False,
                    False,
                ),
                (
                    [
                        edge
                        for edge in disabled_edges
                        if (edge.source, edge.target) in direct_edges
                    ],
                    True,
                    False,
                    False,
                    True,
                ),
                (
                    [
                        edge
                        for edge in enabled_edges
                        if (edge.source, edge.target) in direct_edges
                    ],
                    False,
                    True,
                    False,
                    True,
                ),
            ]
        for edges, disabled, highlighted, dimmed, direct in edge_groups:
            for edge in edges:
                self._draw_brain_graph_edge(
                    edge,
                    positions,
                    layout_bounds,
                    disabled=disabled,
                    highlighted=highlighted,
                    dimmed=dimmed,
                    direct=direct,
                )

        direct_nodes = (
            {
                endpoint
                for edge in highlight.direct_edges
                for endpoint in edge
            }
            if highlight is not None
            else set()
        )
        for key, node in layout.nodes.items():
            position = positions.get(key)
            if position is None:
                continue

            fill_color = self.theme.panel_background
            outline_color = self._brain_node_kind_color(node.kind)
            radius = min(
                13.0,
                base_radius + (1.5 if node.kind == BrainNodeKind.HIDDEN else 0.0),
            )

            in_selection = (
                highlight is None
                or key in highlight.nodes
                or key in direct_nodes
            )
            if not in_selection:
                outline_color = self._brain_color_alpha(outline_color, 70)
            elif highlight is not None and key not in direct_nodes:
                outline_color = self._brain_color_alpha(outline_color, 160)
            elif key != selected_key and highlight is not None:
                arcade.draw_circle_outline(
                    position[0],
                    position[1],
                    radius + 4.0,
                    self._brain_color_alpha(outline_color, 120),
                    1.5,
                )
            if key == selected_key:
                arcade.draw_circle_outline(
                    position[0],
                    position[1],
                    radius + 7.0,
                    self._brain_node_kind_color(node.kind),
                    3.0,
                )
                arcade.draw_circle_outline(
                    position[0],
                    position[1],
                    radius + 11.0,
                    self._brain_color_alpha(self._brain_node_kind_color(node.kind), 90),
                    2.0,
                )
            self._draw_brain_node(position, fill_color, outline_color, radius=radius)
            hit_radius = max(12.0, radius + 4.0)
            self._brain_node_bounds[key] = arcade.LBWH(
                position[0] - hit_radius,
                position[1] - hit_radius,
                hit_radius * 2.0,
                hit_radius * 2.0,
            )
            if node.kind != BrainNodeKind.HIDDEN:
                label_bounds = self._draw_brain_graph_label(
                    key,
                    self._brain_node_display_name(node),
                    node.kind,
                    position,
                    lanes[node.kind],
                    radius=radius,
                    font_size=label_size,
                    color=(
                        self._brain_color_alpha(self.theme.text_muted, 70)
                        if not in_selection
                        else self._brain_color_alpha(self.theme.text_muted, 160)
                        if highlight is not None and key not in direct_nodes
                        else self.theme.text_muted
                    ),
                )
                if label_bounds is not None:
                    node_bounds = self._brain_node_bounds[key]
                    left = min(node_bounds.left, label_bounds.left)
                    right = max(node_bounds.right, label_bounds.right)
                    bottom = min(node_bounds.bottom, label_bounds.bottom)
                    top = max(node_bounds.top, label_bounds.top)
                    self._brain_node_bounds[key] = arcade.LBWH(
                        left,
                        bottom,
                        right - left,
                        top - bottom,
                    )
        return layout
    def _brain_graph_layout(
        self,
        selected: object,
        brain: object,
        input_keys: list[int],
        output_keys: list[int],
        layout_bounds: arcade.Rect,
    ) -> BrainGraphLayout:
        """Return a cached layout for the selected immutable genome."""
        identity = (
            int(getattr(selected, "creature_id")),
            int(getattr(brain, "genome_id")),
        )
        genome = getattr(brain, "genome")
        genome_nodes = getattr(genome, "nodes", {})
        genome_connections = getattr(genome, "connections", {})
        input_names = tuple(getattr(brain, "last_input_names", ()))
        if len(input_names) != len(input_keys):
            input_names = SENSOR_INPUT_NAMES
        cache_key = (
            identity,
            id(genome),
            id(genome_nodes),
            len(genome_nodes),
            id(genome_connections),
            len(genome_connections),
            tuple(input_keys),
            input_names,
            tuple(output_keys),
            layout_bounds.left,
            layout_bounds.bottom,
            layout_bounds.width,
            layout_bounds.height,
        )
        state = self._brain_state
        if state.layout_cache_key == cache_key and state.layout is not None:
            return state.layout

        layout = build_brain_graph_layout(
            genome,
            input_keys,
            output_keys,
            layout_bounds,
            input_names,
            ACTION_OUTPUT_NAMES,
        )
        state.layout_cache_key = cache_key
        state.layout = layout
        return layout
    def _brain_highlight_for_node(
        self,
        layout: BrainGraphLayout,
        node_key: int,
    ) -> BrainGraphHighlight:
        """Return a cached signal-path highlight for one graph selection."""
        state = self._brain_state
        if (
            state.highlight_layout is layout
            and state.highlight_node_key == node_key
            and state.highlight is not None
        ):
            return state.highlight

        highlight = highlighted_path_through_node(layout, node_key)
        state.highlight_layout = layout
        state.highlight_node_key = node_key
        state.highlight = highlight
        return highlight
    def _draw_brain_graph_edge(
        self,
        edge: BrainGraphEdge,
        positions: dict[int, tuple[float, float]],
        bounds: arcade.Rect,
        *,
        disabled: bool = False,
        highlighted: bool = False,
        dimmed: bool = False,
        direct: bool = False,
    ) -> None:
        """Draw brain graph edge.

        Parameters
        ----------
        edge
            Value used by the operation.
        positions
            Value used by the operation.
        bounds
            Rectangle defining the relevant UI area.
        disabled
            Whether the corresponding behavior is enabled.
        highlighted
            Value used by the operation.
        dimmed
            Value used by the operation.
        direct
            Value used by the operation.
        """
        start = positions.get(edge.source)
        end = positions.get(edge.target)
        if start is None or end is None:
            return

        color = self._brain_edge_color(edge.weight)
        width = max(1.0, min(4.5, 0.9 + abs(edge.weight) * 0.65))
        draw_arrow = not disabled
        dashed = False
        if disabled:
            if direct:
                color = self._brain_color_alpha(color, 165)
                width = max(1.25, width * 0.85)
                draw_arrow = True
                dashed = True
            else:
                color = self._brain_color_alpha(self.theme.panel_border, 34)
                width = 0.75
                draw_arrow = False
        elif dimmed:
            color = self._brain_color_alpha(color, 38)
            width = max(0.75, width * 0.62)
        elif highlighted:
            color = self._brain_color_alpha(color, 255 if direct else 175)
            width += 2.25 if direct else 0.85
        else:
            color = self._brain_color_alpha(color, 105)
        if edge.kind == BrainEdgeKind.SELF_LOOP:
            self._draw_self_loop(
                start,
                color,
                width,
                dashed=dashed,
                draw_arrow=draw_arrow,
            )
            return
        if edge.kind == BrainEdgeKind.RECURRENT:
            control_y = (
                bounds.top - 18.0 if start[1] <= end[1] else bounds.bottom + 18.0
            )
            control = ((start[0] + end[0]) * 0.5, control_y)
            points = self._quadratic_bezier_points(start, control, end)
            if dashed:
                self._draw_dashed_curve(points, color, width)
            else:
                self._draw_brain_solid_curve(points, color, width)
            if draw_arrow:
                self._draw_brain_arrowhead(points, color, width)
            return

        direction = 1.0 if end[0] >= start[0] else -1.0
        handle = max(28.0, abs(end[0] - start[0]) * 0.42)
        first_control = (start[0] + direction * handle, start[1])
        second_control = (end[0] - direction * handle, end[1])
        points = self._cubic_bezier_points(
            start,
            first_control,
            second_control,
            end,
        )
        if dashed:
            self._draw_dashed_curve(points, color, width)
        else:
            self._draw_brain_solid_curve(points, color, width)
        if draw_arrow:
            self._draw_brain_arrowhead(points, color, width)
    def _draw_brain_graph_label(
        self,
        node_key: int,
        label: str,
        kind: BrainNodeKind,
        position: tuple[float, float],
        lane: arcade.Rect,
        *,
        radius: float,
        font_size: float,
        color: arcade.Color | tuple[int, ...] | None = None,
    ) -> arcade.Rect | None:
        """Draw brain graph label.

        Parameters
        ----------
        node_key
            Value used by the operation.
        label
            Text displayed by the UI.
        kind
            Value used by the operation.
        position
            Value used by the operation.
        lane
            Value used by the operation.
        radius
            Requested logical size.
        font_size
            Value used by the operation.
        color
            Arcade-compatible color.

        Returns
        -------
        arcade.Rect | None
            Computed UI rectangle.
        """
        label_text = self._short_brain_label(label)
        if kind == BrainNodeKind.INPUT:
            anchor_x = "right"
            x = position[0] - radius - 8.0
            label_width = max(24.0, x - lane.left - 10.0)
        elif kind == BrainNodeKind.OUTPUT:
            anchor_x = "left"
            x = position[0] + radius + 8.0
            label_width = max(24.0, lane.right - x - 10.0)
        else:
            return None

        cache_key = f"brain_window_node_label_{node_key}"
        self._draw_text(
            cache_key,
            self._fit_line(label_text, label_width),
            x,
            position[1],
            color or self.theme.text_muted,
            font_size,
            anchor_x=anchor_x,
            anchor_y="center",
        )
        rendered_height = max(24.0, font_size + 8.0)
        label_left = x - label_width if anchor_x == "right" else x
        return arcade.LBWH(
            label_left - 5.0,
            position[1] - rendered_height / 2.0,
            label_width + 10.0,
            rendered_height,
        )
    def _draw_brain_graph_lanes(
        self,
        bounds: arcade.Rect,
        layout: BrainGraphLayout,
    ) -> dict[BrainNodeKind, arcade.Rect]:
        """Draw brain graph lanes.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        layout
            Value used by the operation.

        Returns
        -------
        dict[BrainNodeKind, arcade.Rect]
            Computed UI rectangle.
        """
        lane_bounds = self._brain_graph_lane_bounds(bounds)
        input_count = sum(
            node.kind == BrainNodeKind.INPUT for node in layout.nodes.values()
        )
        hidden_nodes = [
            node
            for node in layout.nodes.values()
            if node.kind == BrainNodeKind.HIDDEN
        ]
        output_count = sum(
            node.kind == BrainNodeKind.OUTPUT for node in layout.nodes.values()
        )
        hidden_layer_count = len({node.depth for node in hidden_nodes})
        hidden_heading = "HIDDEN LAYER" if hidden_layer_count <= 1 else "HIDDEN LAYERS"
        lane_specs = (
            (
                BrainNodeKind.INPUT,
                "brain_lane_inputs",
                f"INPUTS ({input_count})",
                (232, 241, 255),
            ),
            (
                BrainNodeKind.HIDDEN,
                "brain_lane_hidden",
                f"{hidden_heading} ({len(hidden_nodes)})",
                (246, 238, 255),
            ),
            (
                BrainNodeKind.OUTPUT,
                "brain_lane_outputs",
                f"OUTPUTS ({output_count})",
                (235, 247, 239),
            ),
        )
        for kind, key, label, header_fill in lane_specs:
            lane = lane_bounds[kind]
            self._draw_rounded_rect(
                lane,
                self.theme.panel_background,
                self.theme.panel_border,
                self.config.layout.card_radius,
                1.0,
            )
            lane_header = arcade.LBWH(
                lane.left + 1,
                lane.top - 38,
                max(1.0, lane.width - 2),
                37,
            )
            self._draw_rounded_rect_fill(lane_header, header_fill, 7)
            self._draw_text(
                key,
                label,
                lane.center_x,
                lane_header.center_y,
                self._brain_node_kind_color(kind),
                10,
                bold=True,
                anchor_x="center",
                anchor_y="center",
            )
        return lane_bounds
    def _brain_graph_lane_bounds(
        self,
        bounds: arcade.Rect,
    ) -> dict[BrainNodeKind, arcade.Rect]:
        """Return brain graph lane bounds.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.

        Returns
        -------
        dict[BrainNodeKind, arcade.Rect]
            Computed UI rectangle.
        """
        input_width = bounds.width * 0.21
        first_gap = bounds.width * 0.18
        hidden_width = bounds.width * 0.25
        second_gap = bounds.width * 0.18
        output_width = max(
            1.0,
            bounds.width - input_width - first_gap - hidden_width - second_gap,
        )
        input_lane = arcade.LBWH(
            bounds.left,
            bounds.bottom,
            input_width,
            bounds.height,
        )
        hidden_lane = arcade.LBWH(
            input_lane.right + first_gap,
            bounds.bottom,
            hidden_width,
            bounds.height,
        )
        output_lane = arcade.LBWH(
            hidden_lane.right + second_gap,
            bounds.bottom,
            output_width,
            bounds.height,
        )
        return {
            BrainNodeKind.INPUT: input_lane,
            BrainNodeKind.HIDDEN: hidden_lane,
            BrainNodeKind.OUTPUT: output_lane,
        }
    def _brain_graph_node_positions(
        self,
        layout: BrainGraphLayout,
        lanes: dict[BrainNodeKind, arcade.Rect],
    ) -> dict[int, tuple[float, float]]:
        """Return brain graph node positions.

        Parameters
        ----------
        layout
            Value used by the operation.
        lanes
            Value used by the operation.

        Returns
        -------
        dict[int, tuple[float, float]]
            Computed collection.
        """
        hidden_depths = sorted(
            {
                node.depth
                for node in layout.nodes.values()
                if node.kind == BrainNodeKind.HIDDEN
            }
        )
        hidden_x: dict[int, float] = {}
        hidden_lane = lanes[BrainNodeKind.HIDDEN]
        hidden_padding = min(28.0, hidden_lane.width * 0.16)
        if len(hidden_depths) == 1:
            hidden_x[hidden_depths[0]] = hidden_lane.center_x
        elif hidden_depths:
            span = max(0.0, hidden_lane.width - hidden_padding * 2.0)
            step = span / max(1, len(hidden_depths) - 1)
            hidden_x = {
                depth: hidden_lane.left + hidden_padding + index * step
                for index, depth in enumerate(hidden_depths)
            }

        positions: dict[int, tuple[float, float]] = {}
        for key, node in layout.nodes.items():
            _, y = layout.positions[key]
            lane = lanes[node.kind]
            edge_padding = min(30.0, lane.width * 0.18)
            if node.kind == BrainNodeKind.INPUT:
                x = lane.right - edge_padding
            elif node.kind == BrainNodeKind.OUTPUT:
                x = lane.left + edge_padding
            else:
                x = hidden_x.get(node.depth, hidden_lane.center_x)
            positions[key] = (x, y)
        return positions
    def _brain_graph_node_metrics(
        self,
        layout: BrainGraphLayout,
        bounds: arcade.Rect,
    ) -> tuple[float, float]:
        """Return brain graph node metrics.

        Parameters
        ----------
        layout
            Value used by the operation.
        bounds
            Rectangle defining the relevant UI area.

        Returns
        -------
        tuple[float, float]
            Computed collection.
        """
        depth_counts: dict[int, int] = {}
        for node in layout.nodes.values():
            depth_counts[node.depth] = depth_counts.get(node.depth, 0) + 1
        densest_column = max(depth_counts.values(), default=1)
        usable_height = max(1.0, bounds.height - 56.0)
        row_step = (
            usable_height
            if densest_column <= 1
            else usable_height / (densest_column - 1)
        )
        radius = max(5.0, min(11.0, (row_step - 2.0) * 0.55))
        font_size = max(8.0, min(11.0, row_step * 0.6))
        return radius, font_size
    def _draw_brain_legend(self, bounds: arcade.Rect) -> None:
        """Draw brain legend.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        """
        self._draw_rounded_rect(
            bounds,
            self.theme.card_background,
            self.theme.panel_border,
            self.config.layout.card_radius,
            1.0,
        )
        x = bounds.left + 20
        y = bounds.top - 28
        self._draw_text(
            "brain_legend_title",
            "LEGEND",
            bounds.center_x,
            y,
            self.theme.text_primary,
            10,
            bold=True,
            anchor_x="center",
        )
        y -= 38
        for index, (kind, label) in enumerate(
            (
                (BrainNodeKind.INPUT, "Input node"),
                (BrainNodeKind.HIDDEN, "Hidden node"),
                (BrainNodeKind.OUTPUT, "Output node"),
            )
        ):
            arcade.draw_circle_filled(x + 6, y + 3, 7, self.theme.panel_background)
            arcade.draw_circle_outline(
                x + 6,
                y + 3,
                7,
                self._brain_node_kind_color(kind),
                1.7,
            )
            self._draw_text(
                f"brain_legend_node_{index}",
                label,
                x + 28,
                y,
                self.theme.text_muted,
                10,
            )
            y -= 34

        arcade.draw_line(
            bounds.left + 14,
            y + 12,
            bounds.right - 14,
            y + 12,
            self.theme.panel_border,
            1,
        )
        y -= 18
        self._draw_text(
            "brain_legend_weight_title",
            "WEIGHT",
            x,
            y,
            self.theme.text_primary,
            9,
            bold=True,
        )
        y -= 31
        weight_items = (
            ("Positive", self._brain_edge_color(0.8), 2.0),
            ("Negative", self._brain_edge_color(-0.8), 2.0),
            ("Near zero", self._brain_edge_color(0.1), 1.0),
            (
                "Disabled gene",
                self._brain_color_alpha(self.theme.panel_border, 60),
                0.75,
            ),
        )
        for index, (label, color, width) in enumerate(weight_items):
            arcade.draw_line(x, y + 4, x + 28, y + 4, color, width)
            self._draw_text(
                f"brain_legend_weight_{index}",
                label,
                x + 40,
                y,
                self.theme.text_muted,
                10,
            )
            y -= 30

        y -= 6
        self._draw_text(
            "brain_legend_strength_title",
            "SELECTION DETAIL",
            x,
            y,
            self.theme.text_primary,
            9,
            bold=True,
        )
        y -= 28
        selection_items = (
            (
                "Direct gene",
                self._brain_color_alpha(self._brain_edge_color(0.8), 255),
                4.0,
                False,
            ),
            (
                "Enabled signal route",
                self._brain_color_alpha(self._brain_edge_color(0.8), 175),
                2.5,
                False,
            ),
            (
                "Unrelated while selected",
                self._brain_color_alpha(self._brain_edge_color(0.8), 38),
                1.0,
                False,
            ),
            (
                "Disabled direct gene",
                self._brain_color_alpha(self._brain_edge_color(0.8), 165),
                1.5,
                True,
            ),
        )
        for index, (label, color, width, dashed) in enumerate(selection_items):
            points = [(x, y + 4), (x + 28, y + 4)]
            if dashed:
                self._draw_dashed_curve(points, color, width)
            else:
                self._draw_curve(points, color, width)
            self._draw_text(
                f"brain_legend_strength_{index}",
                label,
                x + 40,
                y,
                self.theme.text_muted,
                9.5,
            )
            y -= 29
    def _brain_color_alpha(
        self,
        color: arcade.Color | tuple[int, ...],
        alpha: int,
    ) -> tuple[int, int, int, int]:
        """Return brain color alpha.

        Parameters
        ----------
        color
            Arcade-compatible color.
        alpha
            Value used by the operation.

        Returns
        -------
        tuple[int, int, int, int]
            Computed collection.
        """
        components = tuple(color)
        return (
            int(components[0]),
            int(components[1]),
            int(components[2]),
            max(0, min(255, int(alpha))),
        )
    def _brain_blend_color(
        self,
        background: arcade.Color | tuple[int, ...],
        foreground: arcade.Color | tuple[int, ...],
        amount: float,
    ) -> tuple[int, int, int]:
        """Return brain blend color.

        Parameters
        ----------
        background
            Value used by the operation.
        foreground
            Value used by the operation.
        amount
            Value used by the operation.

        Returns
        -------
        tuple[int, int, int]
            Computed collection.
        """
        background_components = tuple(background)
        foreground_components = tuple(foreground)
        mix = max(0.0, min(1.0, amount))
        return tuple(
            int(background_components[index] * (1.0 - mix) + foreground_components[index] * mix)
            for index in range(3)
        )
    def _brain_edge_color(self, weight: float) -> arcade.Color | tuple[int, ...]:
        """Return brain edge color.

        Parameters
        ----------
        weight
            Value used by the operation.

        Returns
        -------
        arcade.Color | tuple[int, ...]
            Computed result.
        """
        if abs(weight) < 0.25:
            return (180, 188, 200)
        return (43, 108, 246) if weight >= 0.0 else (245, 62, 62)
    def _draw_self_loop(
        self,
        position: tuple[float, float],
        color: arcade.Color | tuple[int, ...],
        width: float,
        *,
        dashed: bool = False,
        draw_arrow: bool = True,
    ) -> None:
        """Draw self loop.

        Parameters
        ----------
        position
            Value used by the operation.
        color
            Arcade-compatible color.
        width
            Requested logical size.
        dashed
            Value used by the operation.
        draw_arrow
            Value used by the operation.
        """
        x, y = position
        points = [
            (x + 8.0, y + 2.0),
            (x + 32.0, y + 26.0),
            (x + 22.0, y - 24.0),
            (x + 8.0, y - 2.0),
        ]
        final_curve: Sequence[tuple[float, float]] = ()
        for start, control, end in (
            (points[0], points[1], points[2]),
            (points[2], points[3], points[0]),
        ):
            curve = self._quadratic_bezier_points(start, control, end, steps=10)
            if dashed:
                self._draw_dashed_curve(curve, color, width)
            else:
                self._draw_brain_solid_curve(curve, color, width)
            final_curve = curve
        if draw_arrow:
            self._draw_brain_arrowhead(final_curve, color, width)
    def _draw_curve(
        self,
        points: Sequence[tuple[float, float]],
        color: arcade.Color | tuple[int, ...],
        width: float,
    ) -> None:
        """Draw curve.

        Parameters
        ----------
        points
            Value used by the operation.
        color
            Arcade-compatible color.
        width
            Requested logical size.
        """
        for start, end in zip(points, points[1:]):
            arcade.draw_line(start[0], start[1], end[0], end[1], color, width)
    def _draw_brain_solid_curve(
        self,
        points: Sequence[tuple[float, float]],
        color: arcade.Color | tuple[int, ...],
        width: float,
    ) -> None:
        """Draw one brain connection with a single line-strip call."""
        draw_line_strip = getattr(arcade, "draw_line_strip", None)
        if draw_line_strip is not None:
            draw_line_strip(points, color, width)
            return
        self._draw_curve(points, color, width)
    def _draw_dashed_curve(
        self,
        points: Sequence[tuple[float, float]],
        color: arcade.Color | tuple[int, ...],
        width: float,
        *,
        dash_length: float = 6.0,
        gap_length: float = 4.0,
    ) -> None:
        """Draw dashed curve.

        Parameters
        ----------
        points
            Value used by the operation.
        color
            Arcade-compatible color.
        width
            Requested logical size.
        dash_length
            Value used by the operation.
        gap_length
            Value used by the operation.
        """
        drawing = True
        remaining = dash_length
        for start, end in zip(points, points[1:]):
            delta_x = end[0] - start[0]
            delta_y = end[1] - start[1]
            segment_length = (delta_x * delta_x + delta_y * delta_y) ** 0.5
            if segment_length <= 0.0001:
                continue
            consumed = 0.0
            while consumed < segment_length:
                step = min(remaining, segment_length - consumed)
                if drawing:
                    start_ratio = consumed / segment_length
                    end_ratio = (consumed + step) / segment_length
                    arcade.draw_line(
                        start[0] + delta_x * start_ratio,
                        start[1] + delta_y * start_ratio,
                        start[0] + delta_x * end_ratio,
                        start[1] + delta_y * end_ratio,
                        color,
                        width,
                    )
                consumed += step
                remaining -= step
                if remaining <= 0.0001:
                    drawing = not drawing
                    remaining = dash_length if drawing else gap_length
    def _quadratic_bezier_points(
        self,
        start: tuple[float, float],
        control: tuple[float, float],
        end: tuple[float, float],
        *,
        steps: int = 24,
    ) -> tuple[tuple[float, float], ...]:
        """Return quadratic bezier points.

        Parameters
        ----------
        start
            Value used by the operation.
        control
            Value used by the operation.
        end
            Value used by the operation.
        steps
            Value used by the operation.

        Returns
        -------
        tuple[tuple[float, float], ...]
            Computed collection.
        """
        cache_key = ("quadratic", start, control, end, steps)
        cached = self._painter.curve_cache.get(cache_key)
        if cached is not None:
            return cached
        points: list[tuple[float, float]] = []
        for index in range(steps + 1):
            t = index / steps
            inverse = 1.0 - t
            points.append(
                (
                    inverse * inverse * start[0]
                    + 2.0 * inverse * t * control[0]
                    + t * t * end[0],
                    inverse * inverse * start[1]
                    + 2.0 * inverse * t * control[1]
                    + t * t * end[1],
                )
            )
        curve = tuple(points)
        if len(self._painter.curve_cache) >= 4096:
            self._painter.curve_cache.clear()
        self._painter.curve_cache[cache_key] = curve
        return curve
    def _cubic_bezier_points(
        self,
        start: tuple[float, float],
        first_control: tuple[float, float],
        second_control: tuple[float, float],
        end: tuple[float, float],
        *,
        steps: int = 28,
    ) -> tuple[tuple[float, float], ...]:
        """Return cubic bezier points.

        Parameters
        ----------
        start
            Value used by the operation.
        first_control
            Value used by the operation.
        second_control
            Value used by the operation.
        end
            Value used by the operation.
        steps
            Value used by the operation.

        Returns
        -------
        tuple[tuple[float, float], ...]
            Computed collection.
        """
        cache_key = (
            "cubic",
            start,
            first_control,
            second_control,
            end,
            steps,
        )
        cached = self._painter.curve_cache.get(cache_key)
        if cached is not None:
            return cached
        points: list[tuple[float, float]] = []
        for index in range(steps + 1):
            t = index / steps
            inverse = 1.0 - t
            points.append(
                (
                    inverse**3 * start[0]
                    + 3.0 * inverse * inverse * t * first_control[0]
                    + 3.0 * inverse * t * t * second_control[0]
                    + t**3 * end[0],
                    inverse**3 * start[1]
                    + 3.0 * inverse * inverse * t * first_control[1]
                    + 3.0 * inverse * t * t * second_control[1]
                    + t**3 * end[1],
                )
            )
        curve = tuple(points)
        if len(self._painter.curve_cache) >= 4096:
            self._painter.curve_cache.clear()
        self._painter.curve_cache[cache_key] = curve
        return curve
    def _draw_brain_arrowhead(
        self,
        points: Sequence[tuple[float, float]],
        color: arcade.Color | tuple[int, ...],
        width: float,
    ) -> None:
        """Draw brain arrowhead.

        Parameters
        ----------
        points
            Value used by the operation.
        color
            Arcade-compatible color.
        width
            Requested logical size.
        """
        if len(points) < 4:
            return
        tip_index = max(2, min(len(points) - 2, int((len(points) - 1) * 0.86)))
        tip = points[tip_index]
        previous = points[tip_index - 1]
        delta_x = tip[0] - previous[0]
        delta_y = tip[1] - previous[1]
        length = (delta_x * delta_x + delta_y * delta_y) ** 0.5
        if length <= 0.0001:
            return
        unit_x = delta_x / length
        unit_y = delta_y / length
        size = 4.0 + min(3.0, width)
        base_x = tip[0] - unit_x * size
        base_y = tip[1] - unit_y * size
        perpendicular_x = -unit_y * size * 0.55
        perpendicular_y = unit_x * size * 0.55
        line_width = max(1.0, width * 0.8)
        arcade.draw_line(
            tip[0],
            tip[1],
            base_x + perpendicular_x,
            base_y + perpendicular_y,
            color,
            line_width,
        )
        arcade.draw_line(
            tip[0],
            tip[1],
            base_x - perpendicular_x,
            base_y - perpendicular_y,
            color,
            line_width,
        )
    def _node_column_positions(
        self,
        node_keys: list[int],
        x: float,
        bottom: float,
        top: float,
    ) -> dict[int, tuple[float, float]]:
        """Return node column positions.

        Parameters
        ----------
        node_keys
            Value used by the operation.
        x
            Logical screen coordinate.
        bottom
            Logical screen coordinate.
        top
            Logical screen coordinate.

        Returns
        -------
        dict[int, tuple[float, float]]
            Computed collection.
        """
        if not node_keys:
            return {}
        if len(node_keys) == 1:
            return {node_keys[0]: (x, (bottom + top) * 0.5)}

        step = (top - bottom) / (len(node_keys) - 1)
        return {key: (x, top - index * step) for index, key in enumerate(node_keys)}
    def _draw_brain_node(
        self,
        position: tuple[float, float],
        fill_color: arcade.Color | tuple[int, ...],
        outline_color: arcade.Color | tuple[int, ...],
        *,
        radius: float = 5.0,
    ) -> None:
        """Draw brain node.

        Parameters
        ----------
        position
            Value used by the operation.
        fill_color
            Arcade-compatible color.
        outline_color
            Arcade-compatible color.
        radius
            Requested logical size.
        """
        arcade.draw_circle_filled(position[0], position[1], radius, fill_color)
        arcade.draw_circle_outline(position[0], position[1], radius, outline_color, 1.5)
    def _brain_activity_color(self, value: float) -> arcade.Color | tuple[int, ...]:
        """Return brain activity color.

        Parameters
        ----------
        value
            Value used by the operation.

        Returns
        -------
        arcade.Color | tuple[int, ...]
            Computed result.
        """
        strength = max(0.0, min(1.0, abs(value)))
        if value < 0.0:
            base = self.theme.selected_outline
        else:
            base = self.theme.accent
        return (
            int(235 * (1.0 - strength) + base[0] * strength),
            int(235 * (1.0 - strength) + base[1] * strength),
            int(235 * (1.0 - strength) + base[2] * strength),
        )
    def _draw_brain_node_label(
        self,
        key: str,
        text: str,
        position: tuple[float, float],
        bounds: arcade.Rect,
        *,
        side: str,
    ) -> None:
        """Draw brain node label.

        Parameters
        ----------
        key
            Stable identifier used by the UI.
        text
            Text displayed by the UI.
        position
            Value used by the operation.
        bounds
            Rectangle defining the relevant UI area.
        side
            Value used by the operation.
        """
        label_width = min(68.0, max(28.0, bounds.width * 0.32))
        if side == "left":
            x = position[0] - label_width - 10
        else:
            x = position[0] + 10

        x = max(bounds.left, min(bounds.right - label_width, x))
        y = max(bounds.bottom + 4, min(bounds.top - 12, position[1] - 4))

        self._draw_text(
            key,
            self._fit_line(text, label_width),
            x,
            y,
            self.theme.text_muted,
            9,
            width=label_width,
        )
    def _short_brain_label(self, label: str) -> str:
        """Return short brain label.

        Parameters
        ----------
        label
            Text displayed by the UI.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        replacements = {
            "constant": "const",
            "feeding_drive": "feed",
            "reproductive_readiness": "repr",
            "energy_percent": "energy",
            "creature_count": "near",
            "food_count": "food#",
            "clock_tik_tok": "tik",
            "clock_chronometer": "chrono",
            "clock_time_alive": "age",
            "food_proximity": "f_p",
            "food_angle": "f_a",
            "creature_proximity": "n_p",
            "creature_angle": "n_a",
            "wall_proximity": "w_p",
            "wall_angle": "w_a",
            "is_grabbing": "holding",
            "biome_fertility_here": "bio_here",
            "biome_fertility_left_gradient": "bio_left",
            "biome_fertility_right_gradient": "bio_right",
            "biome_fertility_trend": "bio_trend",
            "own_infant_proximity": "baby_p",
            "own_infant_angle": "baby_a",
            "flock_center_proximity": "flock_p",
            "flock_center_angle": "flock_a",
            "flock_average_relative_heading": "flock_h",
            "flockmate_count": "flock_n",
            "flock_presence": "flock_on",
            "flock_effective_count": "flock_n",
            "flock_center_forward": "flock_f",
            "flock_center_right": "flock_r",
            "flock_relative_velocity_forward": "flock_vf",
            "flock_relative_velocity_right": "flock_vr",
            "long_range_social_intensity": "social_i",
            "long_range_social_direction_forward": "social_f",
            "long_range_social_direction_right": "social_r",
            "stomach_fullness": "stomach",
            "sound_strength": "sound",
            "sound_dir_sin": "sound_sin",
            "sound_dir_cos": "sound_cos",
            "sound_tone": "tone",
            "trail_pheromone_here": "trail_here",
            "trail_pheromone_forward_left": "trail_left",
            "trail_pheromone_forward_right": "trail_right",
            "alarm_pheromone_here": "alarm_here",
            "alarm_pheromone_forward_left": "alarm_left",
            "alarm_pheromone_forward_right": "alarm_right",
            "accelerate": "acc",
            "rotate": "rot",
            "want_reproduce": "repr",
            "want_eat": "eat",
            "reset_chronometer": "reset",
            "want_grab": "grab",
            "want_release": "drop",
            "want_nurse": "nurse",
            "flee_panic_intensity": "panic",
            "herding": "herd",
            "emit_sound": "sound",
            "emit_trail_pheromone": "trail",
            "emit_alarm_pheromone": "alarm",
        }
        return replacements.get(label, label)
    def _brain_value_readout(
        self,
        inputs: list[float],
        outputs: list[float],
    ) -> str:
        """Return brain value readout.

        Parameters
        ----------
        inputs
            Value used by the operation.
        outputs
            Value used by the operation.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        return f"{self._brain_input_readout(inputs)}\n{self._brain_output_readout(outputs)}"
    def _brain_input_readout(self, inputs: list[float]) -> str:
        """Return brain input readout.

        Parameters
        ----------
        inputs
            Value used by the operation.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        def value(index: int, values: list[float]) -> float:
            """Return value.

            Parameters
            ----------
            index
                Value used by the operation.
            values
                Value used by the operation.

            Returns
            -------
            float
                Computed result.
            """
            return values[index] if index < len(values) else 0.0

        return (
            f"F {value(10, inputs):.2f}/{value(11, inputs):.2f}  "
            f"C {value(12, inputs):.2f}/{value(13, inputs):.2f}  "
            f"W {value(14, inputs):.2f}/{value(15, inputs):.2f}  "
            f"G {value(16, inputs):.2f}  "
            f"B {value(17, inputs):.2f}/{value(18, inputs):.2f}/"
            f"{value(19, inputs):.2f}/{value(20, inputs):.2f}  "
            f"FL {value(23, inputs):.2f}/{value(24, inputs):.2f}/"
            f"{value(25, inputs):.2f}/{value(26, inputs):.2f}  "
            f"E {value(3, inputs):.2f}"
        )
    def _brain_output_readout(self, outputs: list[float]) -> str:
        """Return brain output readout.

        Parameters
        ----------
        outputs
            Value used by the operation.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        def value(index: int, values: list[float]) -> float:
            """Return value.

            Parameters
            ----------
            index
                Value used by the operation.
            values
                Value used by the operation.

            Returns
            -------
            float
                Computed result.
            """
            return values[index] if index < len(values) else 0.0

        return (
            f"Centered outputs: {value(0, outputs):.2f}/"
            f"{value(1, outputs):.2f}  "
            f"Intent: {value(2, outputs):.2f}/{value(3, outputs):.2f}/"
            f"{value(4, outputs):.2f}  "
            f"Carry: {value(5, outputs):.2f}/{value(6, outputs):.2f}  "
            f"Social: panic {value(8, outputs):.2f} / "
            f"herding {value(9, outputs):.2f}"
        )
