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

class BrainInspectorComponent:
    """Group related behavior extracted from ``UiRenderer``."""

    def _draw_brain_node_inspector(
        self,
        brain: object | None,
        layout: BrainGraphLayout | None,
        bounds: arcade.Rect,
    ) -> None:
        """Draw brain node inspector.

        Parameters
        ----------
        brain
            Value used by the operation.
        layout
            Value used by the operation.
        bounds
            Rectangle defining the relevant UI area.
        """
        self._draw_rounded_rect(
            bounds,
            self.theme.panel_background,
            self.theme.panel_border,
            self.config.layout.card_radius,
            1.0,
        )
        header = arcade.LBWH(bounds.left, bounds.top - 44, bounds.width, 44)
        close_button = arcade.LBWH(bounds.right - 34, bounds.top - 34, 22, 22)
        self._control_hitboxes["brain_node_inspector_toggle"] = close_button
        self._draw_text(
            "brain_node_inspector_title",
            "NODE INSPECTOR",
            bounds.left + 16,
            bounds.top - 26,
            self.theme.text_primary,
            10,
            bold=True,
            anchor_y="center",
        )
        self._draw_panel_close_button(close_button, "brain_node_inspector")
        arcade.draw_line(
            bounds.left + 12,
            header.bottom,
            bounds.right - 12,
            header.bottom,
            self.theme.panel_border,
            1,
        )

        selected_key = self._brain_selected_node_key
        node = (
            layout.nodes.get(selected_key)
            if layout is not None and selected_key is not None
            else None
        )
        if brain is None or layout is None or node is None:
            content = arcade.LBWH(
                bounds.left + 16,
                bounds.bottom + 16,
                bounds.width - 32,
                max(1.0, header.bottom - bounds.bottom - 28),
            )
            self._draw_scrollable_lines_in_bounds(
                "brain_node_inspector",
                content,
                [
                    "Select a node",
                    "Click any node to inspect its genome properties and signal path.",
                ],
                line_spacing=22,
                first_line_color=self.theme.text_primary,
                body_color=self.theme.text_muted,
                first_line_bold=True,
                wrap_lines=True,
            )
            return

        summary = arcade.LBWH(
            bounds.left + 14,
            header.bottom - 68,
            bounds.width - 28,
            56,
        )
        kind_color = self._brain_node_kind_color(node.kind)
        arcade.draw_circle_filled(
            summary.left + 10,
            summary.center_y,
            9,
            self.theme.panel_background,
        )
        arcade.draw_circle_outline(
            summary.left + 10,
            summary.center_y,
            9,
            kind_color,
            2,
        )
        kind_label = f"{node.kind.value.title()} Node"
        badge, name_bounds = self._brain_node_badge_layout(summary, kind_label)
        self._draw_rounded_rect(
            badge,
            self._brain_blend_color(
                self.theme.panel_background,
                kind_color,
                0.12,
            ),
            self._brain_blend_color(
                self.theme.panel_background,
                kind_color,
                0.48,
            ),
            6,
            1,
        )
        self._draw_text(
            "brain_node_inspector_kind",
            kind_label,
            badge.center_x,
            badge.center_y,
            kind_color,
            8.5,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )
        self._draw_text(
            "brain_node_inspector_name",
            self._brain_node_name_text(
                self._brain_node_display_name(node),
                name_bounds.width,
            ),
            name_bounds.left,
            summary.center_y,
            self.theme.text_primary,
            14,
            bold=True,
            width=name_bounds.width,
            multiline=True,
            align="left",
            anchor_y="center",
        )

        content = arcade.LBWH(
            bounds.left + 16,
            bounds.bottom + 14,
            bounds.width - 32,
            max(1.0, summary.bottom - bounds.bottom - 22),
        )
        self._draw_scrollable_lines_in_bounds(
            "brain_node_inspector",
            content,
            self._brain_node_inspector_lines(brain, layout, node),
            line_spacing=20,
            first_line_color=self.theme.text_primary,
            body_color=self.theme.text_muted,
            first_line_bold=True,
            wrap_lines=True,
        )
    def _brain_node_badge_layout(
        self,
        summary: arcade.Rect,
        label: str,
    ) -> tuple[arcade.Rect, arcade.Rect]:
        """Return brain node badge layout.

        Parameters
        ----------
        summary
            Value used by the operation.
        label
            Text displayed by the UI.

        Returns
        -------
        tuple[arcade.Rect, arcade.Rect]
            Computed UI rectangle.
        """
        estimated_width = len(label) * 5.8 + 22.0
        maximum_width = max(68.0, summary.width * 0.46)
        badge_width = max(68.0, min(108.0, estimated_width, maximum_width))
        badge = arcade.LBWH(
            summary.right - badge_width,
            summary.center_y - 13.0,
            badge_width,
            26.0,
        )
        name_left = summary.left + 30.0
        name_bounds = arcade.LBWH(
            name_left,
            summary.bottom,
            max(16.0, badge.left - 10.0 - name_left),
            summary.height,
        )
        return badge, name_bounds
    def _brain_node_name_text(self, name: str, width: float) -> str:
        """Return brain node name text.

        Parameters
        ----------
        name
            Stable identifier used by the UI.
        width
            Requested logical size.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        lines = self._wrap_line(name, width)
        if len(lines) <= 2:
            return "\n".join(lines)

        max_chars = max(4, int(width / 7.0))
        second_line = lines[1].rstrip()
        second_line = second_line[: max(1, max_chars - 3)].rstrip() + "..."
        return "\n".join((lines[0], second_line))
    def _brain_node_inspector_lines(
        self,
        brain: object,
        layout: BrainGraphLayout,
        node: BrainGraphNode,
    ) -> list[str]:
        """Return brain node inspector lines.

        Parameters
        ----------
        brain
            Value used by the operation.
        layout
            Value used by the operation.
        node
            Value used by the operation.

        Returns
        -------
        list[str]
            Computed collection.
        """
        if node.kind == BrainNodeKind.INPUT:
            layer_label = "Input"
        elif node.kind == BrainNodeKind.OUTPUT:
            layer_label = "Output"
        else:
            layer_label = f"Hidden {node.depth}"
        lines = [
            "NODE DETAILS",
            f"Layer: {layer_label}",
            f"ID: {node.key}",
        ]
        gene = getattr(brain.genome, "nodes", {}).get(node.key)
        if node.kind == BrainNodeKind.INPUT:
            lines.append(f"Sensor: {node.label}")
        elif gene is not None:
            lines.extend(
                (
                    f"Activation: {getattr(gene, 'activation', 'Unavailable')}",
                    f"Aggregation: {getattr(gene, 'aggregation', 'Unavailable')}",
                    "Bias: "
                    + self._format_optional_number(
                        getattr(gene, "bias", None),
                        signed=True,
                    ),
                    "Response: "
                    + self._format_optional_number(
                        getattr(gene, "response", None),
                    ),
                )
            )

        order = {key: index for index, key in enumerate(layout.nodes)}
        connections = [
            connection
            for connection in getattr(brain.genome, "connections", {}).values()
            if connection.key[0] in layout.nodes and connection.key[1] in layout.nodes
        ]
        connections_by_key = {
            connection.key: connection for connection in connections
        }
        highlight = highlighted_path_through_node(layout, node.key)
        incoming = sorted(
            (connection for connection in connections if connection.key[1] == node.key),
            key=lambda connection: order.get(connection.key[0], len(order)),
        )
        outgoing = sorted(
            (connection for connection in connections if connection.key[0] == node.key),
            key=lambda connection: order.get(connection.key[1], len(order)),
        )
        lines.extend(("", f"INCOMING CONNECTIONS ({len(incoming)})"))
        if not incoming:
            lines.append("No incoming connections")
        else:
            for connection in incoming:
                source = layout.nodes[connection.key[0]]
                lines.append(
                    self._brain_connection_inspector_line(
                        self._brain_node_display_name(source),
                        source.key,
                        connection,
                    )
                )
        lines.extend(("", f"OUTGOING CONNECTIONS ({len(outgoing)})"))
        if not outgoing:
            lines.append("No outgoing connections")
        else:
            for connection in outgoing:
                target = layout.nodes[connection.key[1]]
                lines.append(
                    self._brain_connection_inspector_line(
                        self._brain_node_display_name(target),
                        target.key,
                        connection,
                    )
                )
        additional_route_keys = sorted(
            highlight.edges - highlight.direct_edges,
            key=lambda key: (
                order.get(key[0], len(order)),
                order.get(key[1], len(order)),
            ),
        )
        lines.extend(
            (
                "",
                f"ADDITIONAL ENABLED SIGNAL ROUTE ({len(additional_route_keys)})",
            )
        )
        if not additional_route_keys:
            lines.append("No additional route connections")
        else:
            for connection_key in additional_route_keys:
                connection = connections_by_key[connection_key]
                in_upstream = connection_key in highlight.upstream_edges
                in_downstream = connection_key in highlight.downstream_edges
                relation = (
                    "Upstream + downstream"
                    if in_upstream and in_downstream
                    else "Upstream"
                    if in_upstream
                    else "Downstream"
                )
                source = layout.nodes[connection_key[0]]
                target = layout.nodes[connection_key[1]]
                lines.append(
                    self._brain_route_connection_inspector_line(
                        relation,
                        source,
                        target,
                        connection,
                    )
                )
        return lines
    def _brain_connection_inspector_line(
        self,
        endpoint_label: str,
        endpoint_key: int,
        connection: object,
    ) -> str:
        """Return brain connection inspector line.

        Parameters
        ----------
        endpoint_label
            Value used by the operation.
        endpoint_key
            Value used by the operation.
        connection
            Value used by the operation.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        return (
            f"{endpoint_label} [ID {endpoint_key}] | "
            f"{self._brain_connection_inspector_details(connection)}"
        )
    def _brain_route_connection_inspector_line(
        self,
        relation: str,
        source: BrainGraphNode,
        target: BrainGraphNode,
        connection: object,
    ) -> str:
        """Return brain route connection inspector line.

        Parameters
        ----------
        relation
            Value used by the operation.
        source
            Value used by the operation.
        target
            Value used by the operation.
        connection
            Value used by the operation.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        return (
            f"{relation}: {self._brain_node_display_name(source)} "
            f"[ID {source.key}] -> {self._brain_node_display_name(target)} "
            f"[ID {target.key}] | "
            f"{self._brain_connection_inspector_details(connection)}"
        )
    def _brain_connection_inspector_details(self, connection: object) -> str:
        """Return brain connection inspector details.

        Parameters
        ----------
        connection
            Value used by the operation.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        try:
            weight = float(getattr(connection, "weight", 0.0))
        except (TypeError, ValueError):
            weight = 0.0
        state = "Enabled" if bool(getattr(connection, "enabled", False)) else "Disabled"
        return f"{weight:+.3f} | {state}"
    def _draw_brain_footer(
        self,
        world: World,
        selected: object,
        brain: object,
        bounds: arcade.Rect,
    ) -> None:
        """Draw brain footer.

        Parameters
        ----------
        world
            Simulation world providing current state.
        selected
            Value used by the operation.
        brain
            Value used by the operation.
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
        enabled_connections = sum(
            bool(connection.enabled)
            for connection in brain.genome.connections.values()
        )
        input_keys = list(world.neat_controller.config.genome_config.input_keys)
        output_keys = list(world.neat_controller.config.genome_config.output_keys)
        biome_path_count = sum(
            item.has_enabled_path
            for item in brain.sensor_usage(input_keys, output_keys)[17:21]
        )
        action = brain.last_action
        action_label = (
            f"acc {action.accelerate:.2f} / rot {action.rotate:.2f}"
            if action is not None
            else "waiting"
        )
        metrics = (
            (
                "NODES",
                str(len(brain.genome.nodes)),
                self._brain_node_kind_color(BrainNodeKind.INPUT),
            ),
            (
                "CONNECTIONS",
                f"{enabled_connections} / {len(brain.genome.connections)} enabled",
                self._brain_node_kind_color(BrainNodeKind.HIDDEN),
            ),
            (
                "FITNESS",
                self._selected_fitness_label(world, selected),
                (24, 126, 70),
            ),
            ("SIGNED ACTION", action_label, (180, 83, 9)),
            ("BIOME PATHS", f"{biome_path_count} / 4", (0, 112, 122)),
        )
        cell_width = bounds.width / len(metrics)
        for index, (label, value, label_color) in enumerate(metrics):
            cell = arcade.LBWH(
                bounds.left + index * cell_width,
                bounds.bottom,
                cell_width,
                bounds.height,
            )
            if index:
                arcade.draw_line(
                    cell.left,
                    bounds.bottom + 9,
                    cell.left,
                    bounds.top - 9,
                    self.theme.panel_border,
                    1,
                )
            self._draw_text(
                f"brain_footer_label_{index}",
                label,
                cell.center_x,
                cell.center_y + 13,
                label_color,
                11,
                bold=True,
                anchor_x="center",
                anchor_y="center",
            )
            self._draw_text(
                f"brain_footer_value_{index}",
                self._fit_line(value, cell_width - 18),
                cell.center_x,
                cell.center_y - 13,
                self.theme.text_primary,
                12,
                bold=True,
                anchor_x="center",
                anchor_y="center",
            )
    def _brain_node_display_name(self, node: BrainGraphNode) -> str:
        """Return brain node display name.

        Parameters
        ----------
        node
            Value used by the operation.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        if node.kind == BrainNodeKind.HIDDEN:
            return f"Hidden {node.key}"
        return node.label
    def _brain_node_kind_color(
        self,
        kind: BrainNodeKind,
    ) -> arcade.Color | tuple[int, ...]:
        """Return brain node kind color.

        Parameters
        ----------
        kind
            Value used by the operation.

        Returns
        -------
        arcade.Color | tuple[int, ...]
            Computed result.
        """
        if kind == BrainNodeKind.INPUT:
            return (39, 110, 241)
        if kind == BrainNodeKind.OUTPUT:
            return (31, 168, 82)
        return (130, 54, 224)

