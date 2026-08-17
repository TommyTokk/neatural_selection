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

class SpeciesTreeCanvasComponent:
    """Group related behavior extracted from ``UiRenderer``."""

    def _draw_species_tree_legend(self, bounds: arcade.Rect) -> None:
        """Draw species tree legend.

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
        left = bounds.left + 16.0
        right = bounds.right - 16.0
        y = bounds.top - 24.0
        self._draw_text(
            "species_tree_legend_title",
            "LEGEND & CONTROLS",
            left,
            y,
            self.theme.text_primary,
            10,
            bold=True,
            anchor_y="center",
        )
        arcade.draw_line(
            left,
            y - 19.0,
            right,
            y - 19.0,
            self.theme.panel_border,
            1.0,
        )

        y -= 48.0
        self._draw_text(
            "species_tree_legend_nodes_title",
            "NODE STATES",
            left,
            y,
            self.theme.text_primary,
            9,
            bold=True,
        )
        node_rows = (
            ("Species node", "node"),
            ("Extant endpoint", "extant"),
            ("Extinct endpoint", "extinct"),
            ("Selected lineage", "selected"),
            ("Mutation activity", "mutation"),
        )
        for index, (label, kind) in enumerate(node_rows):
            row_y = y - 31.0 - index * 29.0
            marker_x = left + 7.0
            if kind == "extinct":
                arcade.draw_circle_outline(
                    marker_x,
                    row_y,
                    6.0,
                    self._brain_color_alpha(self.theme.text_muted, 110),
                    1.0,
                )
                arcade.draw_line(
                    marker_x - 5.0,
                    row_y,
                    marker_x + 5.0,
                    row_y,
                    self.theme.text_muted,
                    1.5,
                )
                arcade.draw_line(
                    marker_x,
                    row_y - 5.0,
                    marker_x,
                    row_y + 5.0,
                    self.theme.text_muted,
                    1.5,
                )
            elif kind == "mutation":
                arcade.draw_circle_filled(
                    marker_x,
                    row_y,
                    4.0,
                    self.theme.herbivore_fill,
                )
                for tick_index in range(3):
                    angle = pi * (0.18 + tick_index * 0.32)
                    arcade.draw_line(
                        marker_x + cos(angle) * 6.0,
                        row_y + sin(angle) * 6.0,
                        marker_x + cos(angle) * 10.0,
                        row_y + sin(angle) * 10.0,
                        self.theme.accent,
                        1.25,
                    )
            else:
                fill = (
                    self.theme.accent
                    if kind == "selected"
                    else self.theme.herbivore_fill
                )
                if kind == "extant":
                    arcade.draw_circle_filled(
                        marker_x,
                        row_y,
                        8.0,
                        self._brain_color_alpha(fill, 50),
                    )
                arcade.draw_circle_filled(marker_x, row_y, 4.5, fill)
                arcade.draw_circle_outline(
                    marker_x,
                    row_y,
                    6.5,
                    (
                        self.theme.accent
                        if kind == "selected"
                        else self.theme.herbivore_outline
                    ),
                    1.5,
                )
            self._draw_text(
                f"species_tree_legend_node_{index}",
                label,
                left + 25.0,
                row_y,
                self.theme.text_muted,
                9.5,
                anchor_y="center",
            )

        divider_y = y - 180.0
        arcade.draw_line(
            left,
            divider_y,
            right,
            divider_y,
            self.theme.panel_border,
            1.0,
        )
        y = divider_y - 27.0
        self._draw_text(
            "species_tree_legend_lines_title",
            "LINE STYLES",
            left,
            y,
            self.theme.text_primary,
            9,
            bold=True,
        )
        for index, (label, color, width, dashed) in enumerate(
            (
                (
                    "Curved lineage",
                    self._brain_color_alpha(self.theme.accent, 230),
                    2.0,
                    False,
                ),
                (
                    "Extinct lineage",
                    self._brain_color_alpha(self.theme.text_muted, 150),
                    1.5,
                    True,
                ),
                (
                    "Selected ancestry",
                    self._brain_color_alpha(self.theme.accent, 245),
                    4.0,
                    False,
                ),
                (
                    "Descendant weight",
                    self.theme.herbivore_outline,
                    3.5,
                    False,
                ),
            )
        ):
            row_y = y - 30.0 - index * 29.0
            if dashed:
                for dash_start in (0.0, 10.0, 20.0):
                    arcade.draw_line(
                        left + dash_start,
                        row_y,
                        left + min(25.0, dash_start + 6.0),
                        row_y,
                        color,
                        width,
                    )
            else:
                self._draw_curve(
                    self._quadratic_bezier_points(
                        (left, row_y - 2.0),
                        (left + 10.0, row_y + 4.0),
                        (left + 25.0, row_y),
                        steps=5,
                    ),
                    color,
                    width,
                )
            self._draw_text(
                f"species_tree_legend_line_{index}",
                label,
                left + 35.0,
                row_y,
                self.theme.text_muted,
                9.5,
                anchor_y="center",
            )

        divider_y = y - 144.0
        arcade.draw_line(
            left,
            divider_y,
            right,
            divider_y,
            self.theme.panel_border,
            1.0,
        )
        y = divider_y - 27.0
        self._draw_text(
            "species_tree_legend_controls_title",
            "CONTROLS",
            left,
            y,
            self.theme.text_primary,
            9,
            bold=True,
        )
        for index, line in enumerate(
            (
                "Click a species to inspect",
                "Drag the canvas to pan",
                "Scroll or use header zoom",
                "Click the timeline to jump",
            )
        ):
            self._draw_text(
                f"species_tree_legend_control_{index}",
                line,
                left,
                y - 29.0 - index * 25.0,
                self.theme.text_muted,
                9,
                width=max(20.0, right - left),
                multiline=True,
            )
    def _draw_species_tree_empty_timeline(self, timeline: arcade.Rect) -> None:
        """Draw species tree empty timeline.

        Parameters
        ----------
        timeline
            Rectangle defining the relevant UI area.
        """
        self._draw_text(
            "species_tree_timeline_title",
            "TIME",
            timeline.center_x,
            timeline.top - 18.0,
            self.theme.text_primary,
            10,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )
        arcade.draw_line(
            timeline.left + 12.0,
            timeline.top - 38.0,
            timeline.right - 12.0,
            timeline.top - 38.0,
            self.theme.panel_border,
            1.0,
        )
        self._draw_text(
            "species_tree_timeline_empty",
            "00:00",
            timeline.center_x,
            timeline.center_y,
            self.theme.text_muted,
            10,
            anchor_x="center",
            anchor_y="center",
        )
    def _draw_species_tree_timeline(
        self,
        layout: SpeciesTreeLayout,
        timeline: arcade.Rect,
        canvas: arcade.Rect,
    ) -> None:
        """Draw species tree timeline.

        Parameters
        ----------
        layout
            Value used by the operation.
        timeline
            Rectangle defining the relevant UI area.
        canvas
            Rectangle defining the relevant UI area.
        """
        self._species_tree_timeline_bucket_bounds.clear()
        self._draw_text(
            "species_tree_timeline_title",
            "TIME",
            timeline.center_x,
            timeline.top - 18.0,
            self.theme.text_primary,
            10,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )
        arcade.draw_line(
            timeline.left + 12.0,
            timeline.top - 38.0,
            timeline.right - 12.0,
            timeline.top - 38.0,
            self.theme.panel_border,
            1.0,
        )
        ruler = arcade.LBWH(
            timeline.left + 8.0,
            timeline.bottom + 12.0,
            max(0.0, timeline.width - 16.0),
            max(0.0, timeline.height - 42.0),
        )
        start = layout.timeline_start
        end = max(start, layout.timeline_end)
        duration = end - start
        axis_x = ruler.right - 12.0
        arcade.draw_line(
            axis_x,
            ruler.bottom,
            axis_x,
            ruler.top,
            self.theme.panel_border,
            1.5,
        )

        interval = self._species_tree_timeline_tick_interval(duration)
        tick = 0.0 if interval <= 0.0 else ceil(start / interval) * interval
        final_tick = end if duration > 0.0 else start
        while tick <= final_tick + 1e-9:
            y = self._species_tree_timeline_y(tick, start, end, ruler)
            arcade.draw_line(
                axis_x - 7.0,
                y,
                axis_x,
                y,
                self.theme.text_muted,
                1.0,
            )
            self._draw_text(
                f"species_tree_timeline_tick_{tick:.9g}",
                self._format_species_tree_time(tick),
                ruler.left,
                y,
                self.theme.text_muted,
                9,
                anchor_y="center",
            )
            if interval <= 0.0:
                break
            tick += interval

        summaries = self._species_tree_layout_manager.bucket_summaries()
        max_bucket_count = max(
            (summary.node_count for summary in summaries),
            default=1,
        )
        for summary in summaries:
            bucket_time = min(
                end,
                summary.start_time
                + self._species_tree_layout_manager.bucket_seconds * 0.5,
            )
            y = self._species_tree_timeline_y(
                bucket_time,
                start,
                end,
                ruler,
            )
            radius = 3.0 + 4.0 * (
                summary.node_count / max_bucket_count
            ) ** 0.5
            marker = arcade.LBWH(
                axis_x - radius - 3.0,
                y - radius - 3.0,
                (radius + 3.0) * 2.0,
                (radius + 3.0) * 2.0,
            )
            self._species_tree_timeline_bucket_bounds[
                summary.bucket_id
            ] = marker
            arcade.draw_circle_filled(
                axis_x,
                y,
                radius,
                self.theme.accent_soft,
            )
            arcade.draw_circle_outline(
                axis_x,
                y,
                radius,
                self.theme.herbivore_outline,
                1.0,
            )
            self._draw_text(
                f"species_tree_bucket_count_{summary.bucket_id}",
                str(summary.node_count),
                axis_x - radius - 5.0,
                y,
                self.theme.text_muted,
                8,
                anchor_x="right",
                anchor_y="center",
            )

        visible_start, visible_end = self._species_tree_visible_time_range(
            layout,
            canvas,
        )
        top_y = self._species_tree_timeline_y(visible_start, start, end, ruler)
        bottom_y = self._species_tree_timeline_y(visible_end, start, end, ruler)
        indicator_bottom = min(top_y, bottom_y)
        indicator_height = max(6.0, abs(top_y - bottom_y))
        indicator = arcade.LBWH(
            axis_x - 10.0,
            indicator_bottom,
            20.0,
            indicator_height,
        )
        arcade.draw_lrbt_rectangle_filled(
            indicator.left,
            indicator.right,
            indicator.bottom,
            indicator.top,
            self._brain_color_alpha(self.theme.accent_soft, 80),
        )
        for line_start, line_end in (
            ((indicator.left, indicator.bottom), (indicator.right, indicator.bottom)),
            ((indicator.right, indicator.bottom), (indicator.right, indicator.top)),
            ((indicator.right, indicator.top), (indicator.left, indicator.top)),
            ((indicator.left, indicator.top), (indicator.left, indicator.bottom)),
        ):
            arcade.draw_line(
                line_start[0],
                line_start[1],
                line_end[0],
                line_end[1],
                self.theme.accent,
                2.0,
            )
    @staticmethod
    def _species_tree_timeline_tick_interval(duration: float) -> float:
        """Return species tree timeline tick interval.

        Parameters
        ----------
        duration
            Value used by the operation.

        Returns
        -------
        float
            Computed result.
        """
        if duration <= 0.0:
            return 0.0
        raw_interval = duration / 6.0
        magnitude = 10.0 ** floor(log10(raw_interval))
        normalized = raw_interval / magnitude
        if normalized <= 1.0:
            step = 1.0
        elif normalized <= 2.0:
            step = 2.0
        elif normalized <= 5.0:
            step = 5.0
        else:
            step = 10.0
        return step * magnitude
    @staticmethod
    def _species_tree_timeline_y(
        value: float,
        start: float,
        end: float,
        ruler: arcade.Rect,
    ) -> float:
        """Return species tree timeline y.

        Parameters
        ----------
        value
            Value used by the operation.
        start
            Value used by the operation.
        end
            Value used by the operation.
        ruler
            Value used by the operation.

        Returns
        -------
        float
            Computed result.
        """
        if end <= start:
            return ruler.top
        ratio = max(0.0, min(1.0, (value - start) / (end - start)))
        return ruler.top - ratio * ruler.height
    def _jump_species_tree_from_timeline(
        self,
        timeline: arcade.Rect,
        y: float,
    ) -> None:
        """Jump species tree from timeline.

        Parameters
        ----------
        timeline
            Rectangle defining the relevant UI area.
        y
            Logical screen coordinate.
        """
        layout = self._species_tree_last_layout
        if layout is None:
            return
        ruler_top = timeline.top - 30.0
        ruler_bottom = timeline.bottom + 12.0
        if ruler_top <= ruler_bottom or layout.timeline_end <= layout.timeline_start:
            self._jump_species_tree_to_time(layout.timeline_start)
            return
        ratio = max(0.0, min(1.0, (ruler_top - y) / (ruler_top - ruler_bottom)))
        time_value = layout.timeline_start + ratio * (
            layout.timeline_end - layout.timeline_start
        )
        self._jump_species_tree_to_time(time_value)
    def _species_tree_screen_positions(
        self,
        layout: SpeciesTreeLayout,
        canvas: arcade.Rect,
        species_ids: tuple[int, ...] | None = None,
    ) -> dict[int, tuple[float, float]]:
        """Return species tree screen positions.

        Parameters
        ----------
        layout
            Value used by the operation.
        canvas
            Rectangle defining the relevant UI area.
        species_ids
            Value used by the operation.

        Returns
        -------
        dict[int, tuple[float, float]]
            Computed collection.
        """
        horizontal_inset, vertical_inset = self._species_tree_content_insets(
            layout,
            canvas,
            self._species_tree_zoom,
        )
        visible_ids = (
            tuple(layout.positions)
            if species_ids is None
            else species_ids
        )
        return {
            species_id: (
                canvas.left
                + horizontal_inset
                + (position[0] - layout.content_left)
                * self._species_tree_zoom
                - self._species_tree_horizontal_offset,
                canvas.top
                - vertical_inset
                - position[1] * self._species_tree_zoom
                + self._species_tree_vertical_offset,
            )
            for species_id in visible_ids
            if (position := layout.positions.get(species_id)) is not None
        }
    def _species_tree_screen_point(
        self,
        point: tuple[float, float],
        layout: SpeciesTreeLayout,
        canvas: arcade.Rect,
    ) -> tuple[float, float]:
        """Return species tree screen point.

        Parameters
        ----------
        point
            Value used by the operation.
        layout
            Value used by the operation.
        canvas
            Rectangle defining the relevant UI area.

        Returns
        -------
        tuple[float, float]
            Computed collection.
        """
        horizontal_inset, vertical_inset = self._species_tree_content_insets(
            layout,
            canvas,
            self._species_tree_zoom,
        )
        return (
            canvas.left
            + horizontal_inset
            + (point[0] - layout.content_left) * self._species_tree_zoom
            - self._species_tree_horizontal_offset,
            canvas.top
            - vertical_inset
            - point[1] * self._species_tree_zoom
            + self._species_tree_vertical_offset,
        )
    def _species_tree_highlighted_path(
        self,
        layout: SpeciesTreeLayout,
    ) -> tuple[set[int], set[tuple[int, int]]]:
        """Return species tree highlighted path.

        Parameters
        ----------
        layout
            Value used by the operation.

        Returns
        -------
        tuple[set[int], set[tuple[int, int]]]
            Computed collection.
        """
        selected_id = self._species_tree_selected_id
        if selected_id is None or selected_id not in layout.positions:
            return set(), set()
        if self._species_tree_highlight_cache_id == selected_id:
            return (
                self._species_tree_highlight_nodes,
                self._species_tree_highlight_edges,
            )
        parents = self._species_tree_layout_manager.parents
        nodes = {selected_id}
        edges: set[tuple[int, int]] = set()
        current = selected_id
        while (parent := parents.get(current)) is not None:
            edge = (parent, current)
            if edge in edges:
                break
            edges.add(edge)
            nodes.add(parent)
            current = parent
        self._species_tree_highlight_cache_id = selected_id
        self._species_tree_highlight_nodes = nodes
        self._species_tree_highlight_edges = edges
        return nodes, edges
    def _species_tree_founder_color(
        self,
        record: SpeciesRecord,
    ) -> tuple[int, int, int]:
        """Return species tree founder color.

        Parameters
        ----------
        record
            Species history data to inspect.

        Returns
        -------
        tuple[int, int, int]
            Computed collection.
        """
        color = record.founder_color or self.theme.herbivore_fill
        return int(color[0]), int(color[1]), int(color[2])
    def _species_tree_refined_color(
        self,
        record: SpeciesRecord,
        strength: float,
        *,
        alpha: int | None = None,
    ) -> tuple[int, ...]:
        """Return species tree refined color.

        Parameters
        ----------
        record
            Species history data to inspect.
        strength
            Value used by the operation.
        alpha
            Value used by the operation.

        Returns
        -------
        tuple[int, ...]
            Computed collection.
        """
        color = self._brain_blend_color(
            self.theme.card_background,
            self._species_tree_founder_color(record),
            strength,
        )
        return color if alpha is None else self._brain_color_alpha(color, alpha)
    def _species_tree_line_color(
        self,
        record: SpeciesRecord,
        *,
        alpha: int = 230,
        muted: bool = False,
    ) -> tuple[int, int, int, int]:
        """Return species tree line color.

        Parameters
        ----------
        record
            Species history data to inspect.
        alpha
            Value used by the operation.
        muted
            Value used by the operation.

        Returns
        -------
        tuple[int, int, int, int]
            Computed collection.
        """
        founder = self._species_tree_founder_color(record)
        luminance = (
            founder[0] * 0.2126
            + founder[1] * 0.7152
            + founder[2] * 0.0722
        )
        darken = (
            0.18
            if luminance < 145.0
            else 0.32
            if luminance < 200.0
            else 0.45
        )
        color = self._brain_blend_color(
            founder,
            self.theme.text_primary,
            darken,
        )
        if muted:
            color = self._brain_blend_color(color, self.theme.text_muted, 0.45)
        for _ in range(12):
            if self._species_tree_contrast_ratio(
                color,
                self.theme.card_background,
                alpha,
            ) >= 3.5:
                break
            color = self._brain_blend_color(
                color,
                self.theme.text_primary,
                0.14,
            )
        return self._brain_color_alpha(color, alpha)
    @staticmethod
    def _species_tree_contrast_ratio(
        foreground: tuple[int, ...],
        background: tuple[int, ...],
        alpha: int = 255,
    ) -> float:
        """Return species tree contrast ratio.

        Parameters
        ----------
        foreground
            Value used by the operation.
        background
            Value used by the operation.
        alpha
            Value used by the operation.

        Returns
        -------
        float
            Computed result.
        """
        opacity = max(0.0, min(1.0, alpha / 255.0))

        def linear(component: float) -> float:
            """Return linear.

            Parameters
            ----------
            component
                Value used by the operation.

            Returns
            -------
            float
                Computed result.
            """
            value = component / 255.0
            return (
                value / 12.92
                if value <= 0.04045
                else ((value + 0.055) / 1.055) ** 2.4
            )

        composited = tuple(
            foreground[index] * opacity
            + background[index] * (1.0 - opacity)
            for index in range(3)
        )
        foreground_luminance = sum(
            weight * linear(component)
            for weight, component in zip(
                (0.2126, 0.7152, 0.0722),
                composited,
            )
        )
        background_luminance = sum(
            weight * linear(float(background[index]))
            for index, weight in enumerate((0.2126, 0.7152, 0.0722))
        )
        lighter = max(foreground_luminance, background_luminance)
        darker = min(foreground_luminance, background_luminance)
        return (lighter + 0.05) / (darker + 0.05)
    def _draw_species_tree_canvas_grid(
        self,
        layout: SpeciesTreeLayout,
        canvas: arcade.Rect,
    ) -> None:
        """Draw species tree canvas grid.

        Parameters
        ----------
        layout
            Value used by the operation.
        canvas
            Rectangle defining the relevant UI area.
        """
        duration = max(0.0, layout.timeline_end - layout.timeline_start)
        interval = self._species_tree_timeline_tick_interval(duration)
        tick = (
            layout.timeline_start
            if interval <= 0.0
            else ceil(layout.timeline_start / interval) * interval
        )
        final_tick = max(layout.timeline_start, layout.timeline_end)
        grid_color = self._brain_color_alpha(self.theme.panel_border, 48)
        while tick <= final_tick + 1e-9:
            point = (
                self._species_tree_layout_manager.padding,
                self._species_tree_layout_manager.padding
                + tick * self._species_tree_layout_manager.time_scale,
            )
            y = self._species_tree_screen_point(point, layout, canvas)[1]
            if canvas.bottom <= y <= canvas.top:
                arcade.draw_line(
                    canvas.left,
                    y,
                    canvas.right,
                    y,
                    grid_color,
                    0.75,
                )
            if interval <= 0.0:
                break
            tick += interval

        selected_id = self._species_tree_selected_id
        selected_position = (
            None if selected_id is None else layout.positions.get(selected_id)
        )
        if selected_position is None:
            return
        focus_y = self._species_tree_screen_point(
            selected_position,
            layout,
            canvas,
        )[1]
        if canvas.bottom <= focus_y <= canvas.top:
            arcade.draw_lrbt_rectangle_filled(
                canvas.left,
                canvas.right,
                focus_y - 8.0,
                focus_y + 8.0,
                self._brain_color_alpha(self.theme.accent, 16),
            )
            arcade.draw_line(
                canvas.left,
                focus_y,
                canvas.right,
                focus_y,
                self._brain_color_alpha(self.theme.accent, 52),
                1.0,
            )
    def _species_tree_rounded_route_points(
        self,
        route: SpeciesTreeRoute,
        *,
        radius: float = 7.0,
    ) -> tuple[tuple[float, float], ...]:
        """Return species tree rounded route points.

        Parameters
        ----------
        route
            Value used by the operation.
        radius
            Requested logical size.

        Returns
        -------
        tuple[tuple[float, float], ...]
            Computed collection.
        """
        if len(route) < 3:
            return tuple(route)
        result: list[tuple[float, float]] = [route[0]]
        for previous, corner, following in zip(route, route[1:], route[2:]):
            incoming = (corner[0] - previous[0], corner[1] - previous[1])
            outgoing = (following[0] - corner[0], following[1] - corner[1])
            incoming_length = hypot(*incoming)
            outgoing_length = hypot(*outgoing)
            if incoming_length <= 0.0 or outgoing_length <= 0.0:
                result.append(corner)
                continue
            trim = min(radius, incoming_length * 0.35, outgoing_length * 0.35)
            start = (
                corner[0] - incoming[0] / incoming_length * trim,
                corner[1] - incoming[1] / incoming_length * trim,
            )
            end = (
                corner[0] + outgoing[0] / outgoing_length * trim,
                corner[1] + outgoing[1] / outgoing_length * trim,
            )
            result.append(start)
            result.extend(
                self._quadratic_bezier_points(
                    start,
                    corner,
                    end,
                    steps=5,
                )[1:]
            )
        result.append(route[-1])
        return tuple(result)
    def _species_tree_soft_route_points(
        self,
        route: SpeciesTreeRoute,
    ) -> SpeciesTreeRoute:
        """Return species tree soft route points.

        Parameters
        ----------
        route
            Value used by the operation.

        Returns
        -------
        SpeciesTreeRoute
            Computed result.
        """
        if len(route) != 2:
            return self._species_tree_rounded_route_points(route, radius=8.0)
        junction, end = route
        horizontal_length = abs(end[0] - junction[0])
        if horizontal_length <= 0.0:
            return tuple(route)
        lead = min(10.0, max(4.0, horizontal_length * 0.18))
        direction = 1.0 if end[0] >= junction[0] else -1.0
        first_control = (
            junction[0],
            junction[1] + lead,
        )
        handle = min(horizontal_length * 0.46, max(18.0, horizontal_length * 0.30))
        second_control = (
            junction[0] + direction * handle,
            junction[1],
        )
        steps = max(8, min(16, ceil(horizontal_length / 18.0)))
        return tuple(
            self._cubic_bezier_points(
                junction,
                first_control,
                second_control,
                end,
                steps=steps,
            )
        )
    def _draw_species_tree_edges(
        self,
        records: dict[int, SpeciesRecord],
        layout: SpeciesTreeLayout,
        edges: tuple[tuple[int, int], ...],
        routes: dict[tuple[int, int], SpeciesTreeRoute],
        highlighted_edges: set[tuple[int, int]],
    ) -> None:
        """Draw species tree edges.

        Parameters
        ----------
        records
            Species history data to inspect.
        layout
            Value used by the operation.
        edges
            Value used by the operation.
        routes
            Value used by the operation.
        highlighted_edges
            Value used by the operation.
        """
        use_dashes = len(edges) <= 900 and self._species_tree_zoom >= 0.35
        base_width = max(1.15, min(2.6, 1.45 * self._species_tree_zoom))
        path_active = self._species_tree_selected_id is not None
        for highlighted in (False, True):
            for edge in edges:
                if (edge in highlighted_edges) != highlighted:
                    continue
                route = routes.get(edge)
                child = records.get(edge[1])
                if route is None or child is None:
                    continue
                dimmed = path_active and not highlighted
                extinct = isfinite(layout.end_times.get(edge[1], float("inf")))
                points = self._species_tree_soft_route_points(route)
                core_width = base_width * (0.76 if dimmed else 1.0)
                if highlighted:
                    self._draw_species_tree_path(
                        points,
                        self._brain_color_alpha(self.theme.accent, 82),
                        base_width + 4.0,
                    )
                self._draw_species_tree_path(
                    points,
                    self._brain_color_alpha(
                        self.theme.text_primary,
                        14 if dimmed else 20 if extinct else 32,
                    ),
                    core_width + 1.8,
                    dashed=extinct and not highlighted and use_dashes,
                )
                color = self._species_tree_line_color(
                    child,
                    alpha=(
                        175
                        if dimmed
                        else 245
                        if highlighted
                        else 150
                        if extinct
                        else 230
                    ),
                    muted=extinct or dimmed,
                )
                self._draw_species_tree_path(
                    points,
                    color,
                    core_width + (0.7 if highlighted else 0.0),
                    dashed=extinct and not highlighted and use_dashes,
                )
                junction = route[0]
                arcade.draw_circle_filled(
                    junction[0],
                    junction[1],
                    max(2.25, min(3.75, 2.7 * self._species_tree_zoom)),
                    self._brain_color_alpha(self.theme.card_background, 245),
                )
                arcade.draw_circle_filled(
                    junction[0],
                    junction[1],
                    max(
                        1.25 if dimmed else 1.5,
                        min(2.75, 1.9 * self._species_tree_zoom),
                    ),
                    color,
                )
    def _draw_species_tree_lifelines(
        self,
        records: dict[int, SpeciesRecord],
        layout: SpeciesTreeLayout,
        species_ids: tuple[int, ...],
        canvas: arcade.Rect,
        highlighted_nodes: set[int],
    ) -> None:
        """Draw species tree lifelines.

        Parameters
        ----------
        records
            Species history data to inspect.
        layout
            Value used by the operation.
        species_ids
            Value used by the operation.
        canvas
            Rectangle defining the relevant UI area.
        highlighted_nodes
            Value used by the operation.
        """
        path_active = self._species_tree_selected_id is not None
        for species_id in sorted(
            species_ids,
            key=lambda candidate: (
                candidate in highlighted_nodes,
                candidate == self._species_tree_selected_id,
                candidate,
            ),
        ):
            record = records[species_id]
            start = layout.positions[species_id]
            end_time = min(
                layout.end_times.get(species_id, float("inf")),
                layout.timeline_end,
            )
            end = (
                start[0],
                self._species_tree_layout_manager.padding
                + max(layout.effective_times[species_id], end_time)
                * self._species_tree_layout_manager.time_scale,
            )
            screen_start = self._species_tree_screen_point(
                start,
                layout,
                canvas,
            )
            screen_end = self._species_tree_screen_point(
                end,
                layout,
                canvas,
            )
            founder_color = self._species_tree_founder_color(record)
            weight = species_tree_line_width(
                layout.descendant_counts.get(species_id, 0)
            )
            scaled_width = max(
                1.15,
                min(4.25, (1.15 + weight * 0.28) * self._species_tree_zoom),
            )
            in_path = species_id in highlighted_nodes
            dimmed = path_active and not in_path
            if dimmed:
                scaled_width *= 0.76
            extinct = isfinite(layout.end_times.get(species_id, float("inf")))
            line_color = self._species_tree_line_color(
                record,
                alpha=(
                    175
                    if dimmed
                    else 245
                    if in_path
                    else 150
                    if extinct
                    else 235
                ),
                muted=extinct or dimmed,
            )
            clipped_start = (screen_start[0], min(canvas.top, screen_start[1]))
            clipped_end = (screen_end[0], max(canvas.bottom, screen_end[1]))
            line = (clipped_start, clipped_end)
            if (
                in_path
                or species_id == self._species_tree_selected_id
            ):
                self._draw_species_tree_path(
                    line,
                    self._brain_color_alpha(self.theme.accent, 82),
                    scaled_width + max(3.0, 4.0 * self._species_tree_zoom),
                )
            self._draw_species_tree_path(
                line,
                self._brain_color_alpha(
                    self.theme.text_primary,
                    14 if dimmed else 18 if extinct else 30,
                ),
                scaled_width + 1.6,
                dashed=(
                    extinct
                    and not in_path
                    and len(species_ids) <= 900
                    and self._species_tree_zoom >= 0.35
                ),
            )
            self._draw_species_tree_path(
                line,
                line_color,
                scaled_width,
                dashed=(
                    extinct
                    and not in_path
                    and len(species_ids) <= 900
                    and self._species_tree_zoom >= 0.35
                ),
            )
            if extinct:
                self._draw_species_tree_extinct_marker(screen_end)
            else:
                self._draw_species_tree_extant_marker(
                    screen_end,
                    founder_color,
                    dimmed=dimmed,
                )
    def _draw_species_tree_extinct_marker(
        self,
        position: tuple[float, float],
    ) -> None:
        """Draw species tree extinct marker.

        Parameters
        ----------
        position
            Value used by the operation.
        """
        half_size = max(3.0, 5.0 * self._species_tree_zoom)
        width = max(1.0, 1.5 * self._species_tree_zoom)
        color = self.theme.text_muted
        arcade.draw_circle_outline(
            position[0],
            position[1],
            half_size + 1.5,
            self._brain_color_alpha(color, 105),
            max(0.75, width * 0.7),
        )
        arcade.draw_line(
            position[0] - half_size,
            position[1],
            position[0] + half_size,
            position[1],
            color,
            width,
        )
        arcade.draw_line(
            position[0],
            position[1] - half_size,
            position[0],
            position[1] + half_size,
            color,
            width,
        )
    def _draw_species_tree_extant_marker(
        self,
        position: tuple[float, float],
        color: tuple[int, ...],
        *,
        dimmed: bool = False,
    ) -> None:
        """Draw species tree extant marker.

        Parameters
        ----------
        position
            Value used by the operation.
        color
            Arcade-compatible color.
        dimmed
            Value used by the operation.
        """
        red, green, blue = color[:3]
        zoom = self._species_tree_zoom
        arcade.draw_circle_filled(
            position[0],
            position[1],
            max(6.0, 12.0 * zoom),
            (red, green, blue, 18 if dimmed else 35),
        )
        arcade.draw_circle_filled(
            position[0],
            position[1],
            max(4.5, 8.0 * zoom),
            (red, green, blue, 38 if dimmed else 75),
        )
        arcade.draw_circle_filled(
            position[0],
            position[1],
            max(3.0, 4.5 * zoom),
            (red, green, blue, 120 if dimmed else 255),
        )
    def _draw_species_tree_nodes(
        self,
        records: dict[int, SpeciesRecord],
        layout: SpeciesTreeLayout,
        species_ids: tuple[int, ...],
        visible_edges: tuple[tuple[int, int], ...],
        positions: dict[int, tuple[float, float]],
        highlighted_nodes: set[int],
        canvas: arcade.Rect,
    ) -> None:
        """Draw species tree nodes.

        Parameters
        ----------
        records
            Species history data to inspect.
        layout
            Value used by the operation.
        species_ids
            Value used by the operation.
        visible_edges
            Value used by the operation.
        positions
            Value used by the operation.
        highlighted_nodes
            Value used by the operation.
        canvas
            Rectangle defining the relevant UI area.
        """
        self._species_tree_node_bounds.clear()
        ordered_ids = sorted(
            species_ids,
            key=lambda species_id: (
                species_id in highlighted_nodes,
                species_id == self._species_tree_selected_id,
                species_id,
            ),
        )
        visual_radii: dict[int, float] = {}
        path_active = self._species_tree_selected_id is not None
        for species_id in ordered_ids:
            record = records[species_id]
            position = positions[species_id]
            radius = self._species_tree_node_visual_radius(layout, species_id)
            visual_radii[species_id] = radius
            hit_radius = max(12.0, radius + 5.0)
            self._species_tree_node_bounds[species_id] = arcade.LBWH(
                position[0] - hit_radius,
                position[1] - hit_radius,
                hit_radius * 2.0,
                hit_radius * 2.0,
            )
            in_path = species_id in highlighted_nodes
            dimmed = (
                path_active
                and not in_path
                and species_id != self._species_tree_hovered_id
            )
            founder_fill = self._species_tree_founder_color(record)
            fill = (
                self._brain_color_alpha(founder_fill, 145)
                if dimmed
                else founder_fill
            )
            if species_id == self._species_tree_selected_id:
                outline = self.theme.accent
                outline_width = 2.5
            elif in_path:
                outline = self.theme.accent
                outline_width = 1.8
            elif species_id == self._species_tree_hovered_id:
                outline = self.theme.accent
                outline_width = 2.0
            elif dimmed:
                outline = self._species_tree_line_color(
                    record,
                    alpha=175,
                    muted=True,
                )
                outline_width = 1.1
            else:
                outline = self._species_tree_line_color(record, alpha=235)
                outline_width = 1.5
            mutation_intensity = (
                0.0
                if dimmed
                else self._species_tree_mutation_intensity(record)
            )
            if mutation_intensity > 0.0:
                arcade.draw_circle_filled(
                    position[0],
                    position[1],
                    radius + 4.0 + mutation_intensity * 3.0,
                    self._brain_color_alpha(self.theme.accent, int(10 + mutation_intensity * 18)),
                )
            if species_id == self._species_tree_selected_id:
                arcade.draw_circle_filled(
                    position[0],
                    position[1],
                    radius + 9.0,
                    self._brain_color_alpha(self.theme.accent, 24),
                )
                arcade.draw_circle_outline(
                    position[0],
                    position[1],
                    radius + 5.0,
                    self._brain_color_alpha(self.theme.accent, 105),
                    2.0,
                )
            elif species_id == self._species_tree_hovered_id:
                arcade.draw_circle_filled(
                    position[0],
                    position[1],
                    radius + 7.0,
                    self._brain_color_alpha(self.theme.accent, 28),
                )
            arcade.draw_circle_filled(position[0], position[1], radius, fill)
            arcade.draw_circle_outline(
                position[0],
                position[1],
                radius,
                outline,
                max(0.75, outline_width * self._species_tree_zoom),
            )
            if mutation_intensity > 0.0:
                self._draw_species_tree_mutation_ticks(
                    position,
                    radius,
                    mutation_intensity,
                )

        labels = self._species_tree_context_labels(
            layout,
            visible_edges,
            positions,
            visual_radii,
            highlighted_nodes,
            canvas,
        )
        self._draw_species_tree_context_labels(labels)
    def _species_tree_mutation_intensity(self, record: SpeciesRecord) -> float:
        """Return species tree mutation intensity.

        Parameters
        ----------
        record
            Species history data to inspect.

        Returns
        -------
        float
            Computed result.
        """
        if (
            record.parent_species_id is None
            or record.data_quality.lower() != "exact"
        ):
            return 0.0
        changes = record.neat_changes
        traits = record.trait_deltas
        if changes is None or traits is None:
            return 0.0
        structural = (
            changes.nodes_added
            + changes.nodes_removed
            + changes.connections_added
            + changes.connections_removed
            + changes.connections_enabled
            + changes.connections_disabled
        )
        parameters = changes.weights_changed + changes.node_parameters_changed
        trait_changes = sum(
            abs(value) > 1e-12
            for value in (
                traits.radius,
                traits.vision_range,
                traits.vision_angle,
                traits.movement_cost_multiplier,
                traits.separation_gene,
                traits.alignment_gene,
                traits.cohesion_gene,
            )
        )
        score = 2 * structural + parameters + trait_changes
        if score <= 0:
            return 0.0
        return min(1.0, log1p(score) / log1p(24.0))
    @staticmethod
    def _species_tree_mutation_tick_count(intensity: float) -> int:
        """Return species tree mutation tick count.

        Parameters
        ----------
        intensity
            Value used by the operation.

        Returns
        -------
        int
            Computed result.
        """
        if intensity <= 0.0:
            return 0
        return max(1, min(3, ceil(intensity * 3.0)))
    def _draw_species_tree_mutation_ticks(
        self,
        position: tuple[float, float],
        radius: float,
        intensity: float,
    ) -> None:
        """Draw species tree mutation ticks.

        Parameters
        ----------
        position
            Value used by the operation.
        radius
            Requested logical size.
        intensity
            Value used by the operation.
        """
        count = self._species_tree_mutation_tick_count(intensity)
        color = self._brain_color_alpha(self.theme.accent, 145)
        for index in range(count):
            angle = pi * (0.18 + index * 0.32)
            inner = radius + 2.5
            outer = inner + 2.5 + intensity * 2.0
            arcade.draw_line(
                position[0] + cos(angle) * inner,
                position[1] + sin(angle) * inner,
                position[0] + cos(angle) * outer,
                position[1] + sin(angle) * outer,
                color,
                1.25,
            )
    @staticmethod
    def _species_tree_rects_overlap(first: arcade.Rect, second: arcade.Rect) -> bool:
        """Return species tree rects overlap.

        Parameters
        ----------
        first
            Value used by the operation.
        second
            Value used by the operation.

        Returns
        -------
        bool
            Whether the operation succeeded or consumed the input.
        """
        return not (
            first.right + 3.0 <= second.left
            or second.right + 3.0 <= first.left
            or first.top + 3.0 <= second.bottom
            or second.top + 3.0 <= first.bottom
        )
    def _species_tree_context_labels(
        self,
        layout: SpeciesTreeLayout,
        visible_edges: tuple[tuple[int, int], ...],
        positions: dict[int, tuple[float, float]],
        radii: dict[int, float],
        highlighted_nodes: set[int],
        canvas: arcade.Rect,
    ) -> tuple[_SpeciesTreeLabel, ...]:
        """Return species tree context labels.

        Parameters
        ----------
        layout
            Value used by the operation.
        visible_edges
            Value used by the operation.
        positions
            Value used by the operation.
        radii
            Value used by the operation.
        highlighted_nodes
            Value used by the operation.
        canvas
            Rectangle defining the relevant UI area.

        Returns
        -------
        tuple[_SpeciesTreeLabel, ...]
            Computed collection.
        """
        selected_id = self._species_tree_selected_id
        hovered_id = self._species_tree_hovered_id
        candidates: list[tuple[int, str, bool]] = []

        def add(species_id: int | None, full: bool = False) -> None:
            """Return add.

            Parameters
            ----------
            species_id
                Value used by the operation.
            full
                Value used by the operation.
            """
            if species_id is None or species_id not in positions:
                return
            if any(candidate[0] == species_id for candidate in candidates):
                return
            text = f"Species {species_id}" if full else f"S{species_id}"
            candidates.append((species_id, text, full))

        add(selected_id, True)
        add(hovered_id, True)
        if selected_id is not None and self._species_tree_zoom >= 0.8:
            add(self._species_tree_layout_manager.parents.get(selected_id))
            for parent, child in visible_edges:
                if parent == selected_id:
                    add(child)
        if self._species_tree_zoom >= 1.15:
            for species_id in sorted(highlighted_nodes, reverse=True):
                if len(candidates) >= 12:
                    break
                add(species_id)

        placed: list[_SpeciesTreeLabel] = []
        for species_id, text, emphasized in candidates[:24]:
            x, y = positions[species_id]
            radius = radii.get(species_id, 5.0)
            width = max(40.0, min(126.0, 20.0 + len(text) * 6.2))
            height = 22.0
            placements = (
                (x - width / 2.0, y - radius - height - 7.0),
                (x + radius + 7.0, y - height / 2.0),
                (x - radius - width - 7.0, y - height / 2.0),
                (x - width / 2.0, y + radius + 7.0),
            )
            chosen: arcade.Rect | None = None
            for left, bottom in placements:
                bounds = arcade.LBWH(left, bottom, width, height)
                contained = (
                    bounds.left >= canvas.left + 3.0
                    and bounds.right <= canvas.right - 3.0
                    and bounds.bottom >= canvas.bottom + 3.0
                    and bounds.top <= canvas.top - 3.0
                )
                if contained and not any(
                    self._species_tree_rects_overlap(bounds, label.bounds)
                    for label in placed
                ):
                    chosen = bounds
                    break
            if chosen is None and emphasized:
                chosen = arcade.LBWH(
                    max(canvas.left + 3.0, min(canvas.right - width - 3.0, x - width / 2.0)),
                    max(canvas.bottom + 3.0, min(canvas.top - height - 3.0, y - radius - height - 7.0)),
                    width,
                    height,
                )
            if chosen is not None:
                placed.append(_SpeciesTreeLabel(species_id, text, chosen, emphasized))
        return tuple(placed)
    def _draw_species_tree_context_labels(
        self,
        labels: tuple[_SpeciesTreeLabel, ...],
    ) -> None:
        """Draw species tree context labels.

        Parameters
        ----------
        labels
            Value used by the operation.
        """
        for label in labels:
            self._draw_rounded_rect(
                label.bounds,
                self.theme.card_background,
                self.theme.accent if label.emphasized else self.theme.panel_border,
                6.0,
                1.0,
            )
            key = (
                "species_tree_selected_label"
                if label.species_id == self._species_tree_selected_id
                else f"species_tree_context_label_{label.species_id}"
            )
            self._draw_text(
                key,
                label.text,
                label.bounds.center_x,
                label.bounds.center_y,
                self.theme.text_primary if label.emphasized else self.theme.text_muted,
                9,
                bold=label.emphasized,
                anchor_x="center",
                anchor_y="center",
            )
    def _species_tree_node_at(self, x: float, y: float) -> int | None:
        """Return species tree node at.

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
        canvas = self._control_hitboxes.get("species_tree_canvas")
        if canvas is None or not self._contains_bounds(canvas, x, y):
            return None
        for species_id in sorted(self._species_tree_node_bounds, reverse=True):
            bounds = self._species_tree_node_bounds[species_id]
            radius = bounds.width * 0.5
            if (x - bounds.center_x) ** 2 + (y - bounds.center_y) ** 2 <= radius**2:
                return species_id
        return None
    def _draw_species_tree_scrollbars(self, canvas: arcade.Rect) -> None:
        """Draw species tree scrollbars.

        Parameters
        ----------
        canvas
            Rectangle defining the relevant UI area.
        """
        horizontal_track = arcade.LBWH(
            canvas.left,
            canvas.bottom - 14.0,
            canvas.width,
            8.0,
        )
        vertical_track = arcade.LBWH(
            canvas.right + 6.0,
            canvas.bottom,
            8.0,
            canvas.height,
        )
        self._control_hitboxes["species_tree_horizontal_track"] = horizontal_track
        self._control_hitboxes["species_tree_vertical_track"] = vertical_track
        self._draw_rounded_rect_fill(horizontal_track, self.theme.panel_border, 4.0)
        self._draw_rounded_rect_fill(vertical_track, self.theme.panel_border, 4.0)

        horizontal_thumb = self._species_tree_scroll_thumb(
            horizontal_track,
            self._species_tree_horizontal_offset,
            self._species_tree_horizontal_limit,
            horizontal=True,
        )
        vertical_thumb = self._species_tree_scroll_thumb(
            vertical_track,
            self._species_tree_vertical_offset,
            self._species_tree_vertical_limit,
            horizontal=False,
        )
        self._control_hitboxes["species_tree_horizontal_thumb"] = horizontal_thumb
        self._control_hitboxes["species_tree_vertical_thumb"] = vertical_thumb
        self._draw_rounded_rect_fill(horizontal_thumb, self.theme.accent, 4.0)
        self._draw_rounded_rect_fill(vertical_thumb, self.theme.accent, 4.0)
    def _species_tree_scroll_thumb(
        self,
        track: arcade.Rect,
        offset: float,
        limit: float,
        *,
        horizontal: bool,
    ) -> arcade.Rect:
        """Return species tree scroll thumb.

        Parameters
        ----------
        track
            Value used by the operation.
        offset
            Value used by the operation.
        limit
            Value used by the operation.
        horizontal
            Value used by the operation.

        Returns
        -------
        arcade.Rect
            Computed UI rectangle.
        """
        track_length = track.width if horizontal else track.height
        content_length = track_length + limit
        thumb_length = (
            track_length
            if limit <= 0.0
            else max(24.0, track_length * track_length / content_length)
        )
        travel = max(0.0, track_length - thumb_length)
        position = 0.0 if limit <= 0.0 else travel * offset / limit
        if horizontal:
            return arcade.LBWH(
                track.left + position,
                track.bottom,
                thumb_length,
                track.height,
            )
        return arcade.LBWH(
            track.left,
            track.top - position - thumb_length,
            track.width,
            thumb_length,
        )
    def _draw_species_tree_tooltip(
        self,
        window_bounds: arcade.Rect,
        record: SpeciesRecord,
        parent_record: SpeciesRecord | None = None,
    ) -> None:
        """Draw species tree tooltip.

        Parameters
        ----------
        window_bounds
            Value used by the operation.
        record
            Species history data to inspect.
        parent_record
            Value used by the operation.
        """
        lines = self._species_tree_tooltip_lines(record, parent_record)
        width = max(160.0, min(350.0, window_bounds.width - 28.0))
        height = 42.0 + len(lines) * 17.0
        mouse_x, mouse_y = self._species_tree_mouse
        left = min(
            window_bounds.right - width - 14.0,
            max(window_bounds.left + 14.0, mouse_x + 18.0),
        )
        bottom = min(
            window_bounds.top - height - 14.0,
            max(window_bounds.bottom + 14.0, mouse_y - height * 0.5),
        )
        bounds = arcade.LBWH(left, bottom, width, height)
        self._draw_rounded_rect(
            bounds,
            self.theme.panel_background,
            self.theme.panel_border,
            self.config.layout.card_radius,
            1.0,
        )
        self._draw_text(
            "species_tree_tooltip_title",
            f"Species {record.species_id}",
            bounds.left + 16.0,
            bounds.top - 22.0,
            self.theme.text_primary,
            15,
            bold=True,
            anchor_y="center",
        )
        for index, line in enumerate(lines):
            self._draw_text(
                f"species_tree_tooltip_{index}",
                line,
                bounds.left + 16.0,
                bounds.top - 45.0 - index * 17.0,
                self.theme.text_muted,
                10,
            )
    def _species_tree_tooltip_lines(
        self,
        record: SpeciesRecord,
        parent_record: SpeciesRecord | None = None,
    ) -> list[str]:
        """Return species tree tooltip lines.

        Parameters
        ----------
        record
            Species history data to inspect.
        parent_record
            Value used by the operation.

        Returns
        -------
        list[str]
            Computed collection.
        """
        lines = [
            f"Species ID: {record.species_id}",
            f"Parent ID: {record.parent_species_id if record.parent_species_id is not None else 'None'}",
            f"Emergence: {self._format_optional_number(record.emerged_at, 2)} s",
        ]
        trait_labels = {
            "radius": "Radius",
            "vision_range": "Vision range",
            "vision_angle": "Vision angle",
            "movement_cost_multiplier": "Move cost",
            "stomach_capacity": "Stomach capacity",
            "digestion_rate": "Digestion rate",
            "digestion_efficiency": "Digestion efficiency",
        }
        summaries = [
            f"{insight.percent_change:+.0f}% {trait_labels[insight.trait]}"
            for insight in profile_morphology(record, parent_record)
        ]
        lines.extend(
            ", ".join(summaries[index : index + 2])
            for index in range(0, min(4, len(summaries)), 2)
        )
        if not summaries:
            lines.append("Morphology: baseline or unavailable")
        return lines
