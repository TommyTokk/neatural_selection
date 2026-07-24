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

class SpeciesTreeWindowComponent:
    """Group related behavior extracted from ``UiRenderer``."""

    def _sync_species_tree_layout(self, world: World) -> SpeciesTreeLayout:
        """Synchronize species tree layout.

        Parameters
        ----------
        world
            Simulation world providing current state.

        Returns
        -------
        SpeciesTreeLayout
            Computed result.
        """
        records = getattr(world, "species_history", {})
        elapsed_time = max(
            0.0,
            float(getattr(world, "elapsed_time", 0.0)),
        )
        living_species_ids = {
            int(creature.lineage.species_id)
            for creature in getattr(world, "creatures", ())
            if getattr(creature, "lineage", None) is not None
        }
        sync_signature = (
            id(records),
            len(records),
            elapsed_time,
            frozenset(living_species_ids),
        )
        if (
            sync_signature == self._species_tree_sync_signature
            and self._species_tree_cached_layout is not None
        ):
            # Layout construction is expensive for long histories; reuse it
            # until records, elapsed time, or living membership changes.
            return self._species_tree_cached_layout
        record_ids = {int(species_id) for species_id in records}
        telemetry_end_times: dict[int, float] = {}
        load_species_end_times = getattr(
            getattr(world, "telemetry", None),
            "load_species_end_times",
            None,
        )
        if load_species_end_times is not None:
            telemetry_end_times = {
                int(species_id): float(end_time)
                for species_id, end_time in load_species_end_times(
                    up_to_time=elapsed_time
                ).items()
            }
        self._species_tree_extinction_times = {
            species_id: extinction_time
            for species_id, extinction_time
            in self._species_tree_extinction_times.items()
            if species_id in record_ids
        }
        for species_id in record_ids:
            if species_id in living_species_ids:
                self._species_tree_extinction_times.pop(species_id, None)
            else:
                telemetry_end_time = telemetry_end_times.get(species_id)
                if (
                    telemetry_end_time is not None
                    and isfinite(telemetry_end_time)
                    and telemetry_end_time <= elapsed_time
                ):
                    self._species_tree_extinction_times[
                        species_id
                    ] = telemetry_end_time
                    continue
                self._species_tree_extinction_times.setdefault(
                    species_id,
                    elapsed_time,
                )
        species_end_times = {
            species_id: self._species_tree_extinction_times.get(
                species_id,
                float("inf"),
            )
            for species_id in record_ids
        }
        self._species_tree_cached_layout = (
            self._species_tree_layout_manager.sync(
                records,
                timeline_end=elapsed_time,
                species_end_times=species_end_times,
            )
        )
        self._species_tree_sync_signature = sync_signature
        return self._species_tree_cached_layout
    @property
    def species_tree_open(self) -> bool:
        """Return species tree open.

        Returns
        -------
        bool
            Whether the operation succeeded or consumed the input.
        """
        return self._species_tree_open
    def open_species_tree(self, world: World) -> None:
        """Open species tree.

        Parameters
        ----------
        world
            Simulation world providing current state.
        """
        if self._species_tree_open:
            return
        # The modal owns simulation focus and restores the caller's exact pause
        # state when it closes.
        self._species_tree_previous_pause = bool(world.is_paused)
        world.is_paused = True
        self._species_tree_open = True
        self._species_tree_hovered_id = None
        self._species_tree_selected_id = None
        self._species_tree_pending_selection_id = None
        self._species_tree_report = None
        self._species_tree_report_species_id = None
        self._clear_species_radar_state()
        self._scroll_offsets["species_tree_inspector"] = 0.0
        self._species_tree_scroll_drag = None
        self._species_tree_canvas_drag = False
        self._species_tree_canvas_drag_started = False
        self._species_tree_inspector_width = None
        self._species_tree_inspector_resize_drag = False
        self._species_tree_zoom = 1.0
        self._species_tree_fit_mode = False
        self._species_tree_fit_requested = False
        self._species_tree_horizontal_offset = 0.0
        self._species_tree_vertical_offset = 0.0
        self._species_tree_focus_latest_pending = True
        self._species_tree_highlight_cache_id = None
        self._species_tree_highlight_nodes.clear()
        self._species_tree_highlight_edges.clear()
        self._control_hitboxes.pop("species_tree_parent_button", None)
    def close_species_tree(self, world: World) -> None:
        """Close species tree.

        Parameters
        ----------
        world
            Simulation world providing current state.
        """
        if not self._species_tree_open:
            return
        previous_pause = self._species_tree_previous_pause
        self._species_tree_open = False
        self._species_tree_previous_pause = None
        self._species_tree_hovered_id = None
        self._species_tree_selected_id = None
        self._species_tree_pending_selection_id = None
        self._species_tree_report = None
        self._species_tree_report_species_id = None
        self._clear_species_radar_state()
        self._scroll_offsets.pop("species_tree_inspector", None)
        self._species_tree_scroll_drag = None
        self._species_tree_canvas_drag = False
        self._species_tree_canvas_drag_started = False
        self._species_tree_inspector_width = None
        self._species_tree_inspector_resize_drag = False
        self._control_hitboxes.pop("species_tree_inspector_resize", None)
        self._control_hitboxes.pop("species_tree_parent_button", None)
        self._species_tree_timeline_bucket_bounds.clear()
        self._species_tree_visible_slice = TreeViewportSlice((), (), {})
        self._species_tree_highlight_cache_id = None
        self._species_tree_highlight_nodes.clear()
        self._species_tree_highlight_edges.clear()
        if previous_pause is not None:
            # Restore rather than toggle so an already-paused world stays paused.
            world.is_paused = previous_pause
    def _draw_species_tree_window(self, world: World) -> None:
        """Draw species tree window.

        Parameters
        ----------
        world
            Simulation world providing current state.
        """
        if not self._species_tree_open:
            return
        layout = self._sync_species_tree_layout(world)

        margin = float(self.config.layout.outer_padding)
        window = world.layout.window
        bounds = arcade.LBWH(
            window.left + margin,
            window.bottom + margin,
            max(0.0, window.width - margin * 2.0),
            max(0.0, window.height - margin * 2.0),
        )
        records = getattr(world, "species_history", {})
        selected_id = self._species_tree_selected_id
        show_inspector = selected_id is not None and selected_id in records
        title = "Species Evolution Tree"
        if show_inspector:
            title += f"  /  Selected: Species {selected_id}"
        content = self._draw_floating_panel(
            bounds,
            title,
            "species_tree",
            icon_name="species",
            body_top_padding=10.0,
            panel_fill=self.theme.panel_background,
            title_icon_size=28.0,
        )
        self._control_hitboxes.pop("species_tree_drag", None)
        self._control_hitboxes["species_tree_window"] = bounds
        inspector = None
        legend = None
        content_right = content.right - 20.0
        if show_inspector:
            inspector_width = self._species_tree_inspector_clamped_width(
                content
            )
            inspector = arcade.LBWH(
                content.right - inspector_width,
                content.bottom + 20.0,
                inspector_width,
                max(0.0, content.height - 20.0),
            )
            content_right = inspector.left - self.SPECIES_TREE_TIMELINE_GAP

        timeline_width = min(
            self.SPECIES_TREE_TIMELINE_WIDTH,
            max(0.0, content.width * 0.24),
        )
        legend_width = min(190.0, max(168.0, content.width * 0.14))
        legend_canvas_width = (
            content_right
            - content.left
            - timeline_width
            - legend_width
            - self.SPECIES_TREE_TIMELINE_GAP * 2.0
        )
        if content.width >= 1200.0 and legend_canvas_width >= 300.0:
            legend = arcade.LBWH(
                content_right - legend_width,
                content.bottom + 20.0,
                legend_width,
                max(0.0, content.height - 20.0),
            )
            content_right = legend.left - self.SPECIES_TREE_TIMELINE_GAP
            self._control_hitboxes["species_tree_legend"] = legend
        else:
            self._control_hitboxes.pop("species_tree_legend", None)

        timeline = arcade.LBWH(
            content.left,
            content.bottom + 20.0,
            timeline_width,
            max(0.0, content.height - 20.0),
        )
        canvas = arcade.LBWH(
            timeline.right + self.SPECIES_TREE_TIMELINE_GAP,
            content.bottom + 20.0,
            max(
                0.0,
                content_right
                - timeline.right
                - self.SPECIES_TREE_TIMELINE_GAP
            ),
            max(0.0, content.height - 20.0),
        )
        self._control_hitboxes["species_tree_timeline"] = timeline
        self._control_hitboxes["species_tree_canvas"] = canvas
        self._draw_rounded_rect(
            timeline,
            self.theme.card_background,
            self.theme.panel_border,
            self.config.layout.card_radius,
            1.0,
        )
        self._draw_rounded_rect(
            canvas,
            self.theme.card_background,
            self.theme.panel_border,
            self.config.layout.card_radius,
            1.0,
        )
        if legend is not None:
            self._draw_species_tree_legend(legend)

        if not records:
            self._species_tree_zoom = 1.0
            self._species_tree_fit_mode = True
            self._species_tree_fit_requested = False
            self._species_tree_last_layout = None
            self._species_tree_last_canvas = canvas
            self._species_tree_node_bounds.clear()
            self._species_tree_timeline_bucket_bounds.clear()
            self._species_tree_visible_slice = TreeViewportSlice((), (), {})
            self._species_tree_horizontal_limit = 0.0
            self._species_tree_vertical_limit = 0.0
            self._species_tree_horizontal_offset_min = 0.0
            self._species_tree_horizontal_offset_max = 0.0
            self._species_tree_vertical_offset_min = 0.0
            self._species_tree_vertical_offset_max = 0.0
            self._draw_species_tree_zoom_controls(bounds)
            self._draw_species_tree_empty_timeline(timeline)
            self._draw_text(
                "species_tree_empty_title",
                "No species history available",
                canvas.center_x,
                canvas.center_y + 12.0,
                self.theme.text_primary,
                18,
                bold=True,
                anchor_x="center",
            )
            self._draw_text(
                "species_tree_empty_body",
                "New speciation events will appear here as the simulation evolves.",
                canvas.center_x,
                canvas.center_y - 20.0,
                self.theme.text_muted,
                12,
                anchor_x="center",
            )
            return

        previous_layout = self._species_tree_last_layout
        previous_canvas = self._species_tree_last_canvas
        canvas_changed = (
            previous_canvas is not None
            and (
                previous_canvas.width != canvas.width
                or previous_canvas.height != canvas.height
            )
        )
        if canvas_changed and not self._species_tree_fit_mode:
            if layout.content_width * self._species_tree_zoom <= canvas.width:
                self._species_tree_horizontal_offset = 0.0
            if layout.content_height * self._species_tree_zoom <= canvas.height:
                self._species_tree_vertical_offset = 0.0
        if previous_layout is not None and not self._species_tree_fit_mode:
            old_horizontal_inset = self._species_tree_content_insets(
                previous_layout,
                canvas,
                self._species_tree_zoom,
            )[0]
            new_horizontal_inset = self._species_tree_content_insets(
                layout,
                canvas,
                self._species_tree_zoom,
            )[0]
            self._species_tree_horizontal_offset += (
                new_horizontal_inset
                - old_horizontal_inset
                + (previous_layout.content_left - layout.content_left)
                * self._species_tree_zoom
            )
        self._species_tree_last_layout = layout
        self._species_tree_last_canvas = canvas
        self._update_species_tree_zoom_and_limits(layout, canvas)
        if self._species_tree_focus_latest_pending:
            self._focus_species_tree_latest(layout, canvas)
        self._draw_species_tree_zoom_controls(bounds)
        content_bounds = self._species_tree_visible_content_bounds(
            layout,
            canvas,
        )
        visible = self._species_tree_layout_manager.viewport_slice(
            left=content_bounds[0],
            right=content_bounds[1],
            top=content_bounds[2],
            bottom=content_bounds[3],
            node_padding=24.0,
        )
        self._species_tree_visible_slice = visible
        positions = self._species_tree_screen_positions(
            layout,
            canvas,
            visible.node_ids,
        )
        screen_routes = {
            edge: tuple(
                self._species_tree_screen_point(point, layout, canvas)
                for point in route
            )
            for edge, route in visible.routes.items()
        }
        highlighted_nodes, highlighted_edges = (
            self._species_tree_highlighted_path(layout)
        )

        with self._ui_clip(canvas):
            self._draw_species_tree_canvas_grid(layout, canvas)
            self._draw_species_tree_lifelines(
                records,
                layout,
                visible.lifeline_ids,
                canvas,
                highlighted_nodes,
            )
            self._draw_species_tree_edges(
                records,
                layout,
                visible.edges,
                screen_routes,
                highlighted_edges,
            )
            self._draw_species_tree_nodes(
                records,
                layout,
                visible.node_ids,
                visible.edges,
                positions,
                highlighted_nodes,
                canvas,
            )

        self._draw_species_tree_timeline(layout, timeline, canvas)
        self._species_tree_hovered_id = self._species_tree_node_at(
            *self._species_tree_mouse
        )
        self._draw_species_tree_scrollbars(canvas)
        if inspector is not None:
            self._ensure_species_inspector_report(world, records)
            selected_record = (
                records.get(self._species_tree_selected_id)
                if self._species_tree_selected_id is not None
                else None
            )
            self._draw_species_inspector(
                inspector,
                self._species_tree_report,
                selected_record,
            )
        else:
            self._control_hitboxes.pop("species_tree_inspector_resize", None)
            self._control_hitboxes.pop("species_tree_parent_button", None)
        hovered = self._species_tree_hovered_id
        if hovered is not None and hovered in records:
            parent_id = records[hovered].parent_species_id
            self._draw_species_tree_tooltip(
                bounds,
                records[hovered],
                records.get(parent_id) if parent_id is not None else None,
            )
    @staticmethod
    def _format_species_tree_time(value: float) -> str:
        """Format species tree time.

        Parameters
        ----------
        value
            Value used by the operation.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        seconds = max(0, int(round(value)))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"
    @staticmethod
    def _valid_species_tree_time(value: object) -> float | None:
        """Return whether valid species tree time.

        Parameters
        ----------
        value
            Value used by the operation.

        Returns
        -------
        float | None
            Computed result.
        """
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not isfinite(parsed) or parsed < 0.0:
            return None
        return parsed
    def _species_tree_visible_time_range(
        self,
        layout: SpeciesTreeLayout,
        canvas: arcade.Rect,
    ) -> tuple[float, float]:
        """Return species tree visible time range.

        Parameters
        ----------
        layout
            Value used by the operation.
        canvas
            Rectangle defining the relevant UI area.

        Returns
        -------
        tuple[float, float]
            Computed collection.
        """
        _, vertical_inset = self._species_tree_content_insets(
            layout,
            canvas,
            self._species_tree_zoom,
        )

        def time_at(screen_y: float) -> float:
            """Return time at.

            Parameters
            ----------
            screen_y
                Value used by the operation.

            Returns
            -------
            float
                Computed result.
            """
            content_y = (
                canvas.top
                - vertical_inset
                + self._species_tree_vertical_offset
                - screen_y
            ) / max(0.0001, self._species_tree_zoom)
            return (
                content_y - self.SPECIES_TREE_CONTENT_PADDING
            ) / self.SPECIES_TREE_TIME_SCALE

        start = max(layout.timeline_start, time_at(canvas.top))
        end = min(layout.timeline_end, time_at(canvas.bottom))
        return min(start, end), max(start, end)
    def _species_tree_visible_content_bounds(
        self,
        layout: SpeciesTreeLayout,
        canvas: arcade.Rect,
    ) -> tuple[float, float, float, float]:
        """Return species tree visible content bounds.

        Parameters
        ----------
        layout
            Value used by the operation.
        canvas
            Rectangle defining the relevant UI area.

        Returns
        -------
        tuple[float, float, float, float]
            Computed collection.
        """
        horizontal_inset, vertical_inset = (
            self._species_tree_content_insets(
                layout,
                canvas,
                self._species_tree_zoom,
            )
        )
        zoom = max(0.0001, self._species_tree_zoom)
        left = (
            canvas.left
            - canvas.left
            - horizontal_inset
            + self._species_tree_horizontal_offset
        ) / zoom + layout.content_left
        right = (
            canvas.right
            - canvas.left
            - horizontal_inset
            + self._species_tree_horizontal_offset
        ) / zoom + layout.content_left
        top = (
            canvas.top
            - vertical_inset
            + self._species_tree_vertical_offset
            - canvas.top
        ) / zoom
        bottom = (
            canvas.top
            - vertical_inset
            + self._species_tree_vertical_offset
            - canvas.bottom
        ) / zoom
        return left, right, top, bottom
    def _focus_species_tree_latest(
        self,
        layout: SpeciesTreeLayout,
        canvas: arcade.Rect,
    ) -> None:
        """Focus species tree latest.

        Parameters
        ----------
        layout
            Value used by the operation.
        canvas
            Rectangle defining the relevant UI area.
        """
        species_id = self._species_tree_layout_manager.latest_species_id
        if species_id is None or species_id not in layout.positions:
            self._species_tree_focus_latest_pending = False
            return
        horizontal_inset, _ = self._species_tree_content_insets(
            layout,
            canvas,
            self._species_tree_zoom,
        )
        latest_x, _ = layout.positions[species_id]
        self._species_tree_horizontal_offset = (
            canvas.left
            + horizontal_inset
            + (latest_x - layout.content_left) * self._species_tree_zoom
            - canvas.center_x
        )
        self._species_tree_vertical_offset = (
            self._species_tree_vertical_offset_max
        )
        self._clamp_species_tree_offsets()
        self._species_tree_focus_latest_pending = False
    def _jump_species_tree_to_time(
        self,
        time_value: float,
        *,
        species_id: int | None = None,
    ) -> None:
        """Jump species tree to time.

        Parameters
        ----------
        time_value
            Value used by the operation.
        species_id
            Value used by the operation.
        """
        layout = self._species_tree_last_layout
        canvas = self._species_tree_last_canvas
        if layout is None or canvas is None:
            return

        anchor = (canvas.center_x, canvas.center_y)
        if self._species_tree_zoom != 1.0 or self._species_tree_fit_mode:
            self._adjust_species_tree_zoom(
                1.0 / max(0.0001, self._species_tree_zoom),
                anchor=anchor,
            )
        self._species_tree_fit_mode = False
        self._species_tree_fit_requested = False
        self._species_tree_zoom = 1.0
        self._update_species_tree_zoom_and_limits(layout, canvas)
        horizontal_inset, vertical_inset = self._species_tree_content_insets(
            layout,
            canvas,
            self._species_tree_zoom,
        )
        if species_id is not None and species_id in layout.positions:
            position_x, position_y = layout.positions[species_id]
            self._species_tree_horizontal_offset = (
                canvas.left
                + horizontal_inset
                + position_x
                - layout.content_left
                - canvas.center_x
            )
        else:
            position_y = (
                self.SPECIES_TREE_CONTENT_PADDING
                + max(layout.timeline_start, min(layout.timeline_end, time_value))
                * self.SPECIES_TREE_TIME_SCALE
            )
        self._species_tree_vertical_offset = (
            canvas.center_y
            - canvas.top
            + vertical_inset
            + position_y
        )
        self._clamp_species_tree_offsets()
    def _select_species_tree_species(
        self,
        species_id: int,
        *,
        focus: bool = False,
    ) -> None:
        """Select species tree species.

        Parameters
        ----------
        species_id
            Value used by the operation.
        focus
            Value used by the operation.
        """
        layout = self._species_tree_last_layout
        if layout is None or species_id not in layout.positions:
            return
        changed = self._species_tree_selected_id != species_id
        self._species_tree_selected_id = species_id
        self._species_tree_pending_selection_id = None
        self._species_tree_highlight_cache_id = None
        self._species_tree_highlight_nodes.clear()
        self._species_tree_highlight_edges.clear()
        if changed:
            self._species_tree_report = None
            self._species_tree_report_species_id = None
            self._clear_species_radar_state()
            self._scroll_offsets["species_tree_inspector"] = 0.0
        if focus:
            self._jump_species_tree_to_time(
                layout.effective_times.get(species_id, layout.timeline_start),
                species_id=species_id,
            )
    def _update_species_tree_zoom_and_limits(
        self,
        layout: SpeciesTreeLayout,
        canvas: arcade.Rect,
    ) -> None:
        """Update species tree zoom and limits.

        Parameters
        ----------
        layout
            Value used by the operation.
        canvas
            Rectangle defining the relevant UI area.
        """
        if self._species_tree_fit_mode:
            width_zoom = (
                canvas.width / layout.content_width
                if layout.content_width > 0.0
                else 1.0
            )
            height_zoom = (
                canvas.height / layout.content_height
                if layout.content_height > 0.0
                else 1.0
            )
            self._species_tree_zoom = max(
                0.0001,
                min(1.0, width_zoom, height_zoom),
            )
            self._species_tree_horizontal_offset = 0.0
            self._species_tree_vertical_offset = 0.0
            self._species_tree_fit_requested = False

        scaled_width = layout.content_width * self._species_tree_zoom
        scaled_height = layout.content_height * self._species_tree_zoom
        self._species_tree_horizontal_limit = max(
            0.0, scaled_width - canvas.width
        )
        self._species_tree_vertical_limit = max(
            0.0, scaled_height - canvas.height
        )
        horizontal_inset = max(0.0, (canvas.width - scaled_width) * 0.5)
        vertical_inset = max(0.0, (canvas.height - scaled_height) * 0.5)
        self._species_tree_horizontal_offset_min = (
            -horizontal_inset
            if self._species_tree_horizontal_limit <= 0.0
            else 0.0
        )
        self._species_tree_horizontal_offset_max = (
            horizontal_inset
            if self._species_tree_horizontal_limit <= 0.0
            else self._species_tree_horizontal_limit
        )
        self._species_tree_vertical_offset_min = (
            -vertical_inset
            if self._species_tree_vertical_limit <= 0.0
            else 0.0
        )
        self._species_tree_vertical_offset_max = (
            vertical_inset
            if self._species_tree_vertical_limit <= 0.0
            else self._species_tree_vertical_limit
        )
        self._clamp_species_tree_offsets()
    def _species_tree_content_insets(
        self,
        layout: SpeciesTreeLayout,
        canvas: arcade.Rect,
        zoom: float,
    ) -> tuple[float, float]:
        """Return species tree content insets.

        Parameters
        ----------
        layout
            Value used by the operation.
        canvas
            Rectangle defining the relevant UI area.
        zoom
            Value used by the operation.

        Returns
        -------
        tuple[float, float]
            Computed collection.
        """
        return (
            max(0.0, (canvas.width - layout.content_width * zoom) * 0.5),
            max(0.0, (canvas.height - layout.content_height * zoom) * 0.5),
        )
    def _draw_species_tree_zoom_controls(self, bounds: arcade.Rect) -> None:
        """Draw species tree zoom controls.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        """
        close_bounds = self._control_hitboxes.get("species_tree_close")
        group_right = (
            close_bounds.left - 12.0
            if close_bounds is not None
            else bounds.right - 60.0
        )
        button_height = 30.0
        gap = 6.0
        fit_width = 46.0
        button_width = 30.0
        label_width = 58.0
        bottom = bounds.top - 43.0

        fit_button = arcade.LBWH(
            group_right - fit_width,
            bottom,
            fit_width,
            button_height,
        )
        plus_button = arcade.LBWH(
            fit_button.left - gap - button_width,
            bottom,
            button_width,
            button_height,
        )
        zoom_label = arcade.LBWH(
            plus_button.left - gap - label_width,
            bottom,
            label_width,
            button_height,
        )
        minus_button = arcade.LBWH(
            zoom_label.left - gap - button_width,
            bottom,
            button_width,
            button_height,
        )
        self._control_hitboxes["species_tree_zoom_out"] = minus_button
        self._control_hitboxes["species_tree_zoom_label"] = zoom_label
        self._control_hitboxes["species_tree_zoom_in"] = plus_button
        self._control_hitboxes["species_tree_zoom_fit"] = fit_button

        self._draw_species_tree_zoom_icon_button(
            minus_button,
            "zoom_out",
            "species_tree_zoom_out",
        )
        self._draw_rounded_rect(
            zoom_label,
            self.theme.card_background,
            self.theme.panel_border,
            6.0,
            1.0,
        )
        self._draw_text(
            "species_tree_zoom_percentage",
            f"{self._species_tree_zoom * 100.0:.0f}%",
            zoom_label.center_x,
            zoom_label.center_y,
            self.theme.text_primary,
            10,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )
        self._draw_species_tree_zoom_icon_button(
            plus_button,
            "zoom_in",
            "species_tree_zoom_in",
        )
        self._draw_button(
            fit_button,
            "Fit",
            "species_tree_zoom_fit",
        )
    def _draw_species_tree_zoom_icon_button(
        self,
        bounds: arcade.Rect,
        icon_name: str,
        key: str,
    ) -> None:
        """Draw species tree zoom icon button.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        icon_name
            Stable identifier used by the UI.
        key
            Stable identifier used by the UI.
        """
        self._draw_rounded_rect(
            bounds,
            self.theme.panel_background,
            self.theme.panel_border,
            8,
            1.5,
        )
        icon_size = min(20.0, bounds.width - 8.0, bounds.height - 8.0)
        self._draw_icon(
            arcade.LBWH(
                bounds.center_x - icon_size / 2.0,
                bounds.center_y - icon_size / 2.0,
                icon_size,
                icon_size,
            ),
            icon_name,
            key,
        )
    def _activate_species_tree_fit(self) -> None:
        """Return activate species tree fit.
        """
        self._species_tree_fit_mode = True
        self._species_tree_fit_requested = True
        self._species_tree_horizontal_offset = 0.0
        self._species_tree_vertical_offset = 0.0
        layout = self._species_tree_last_layout
        canvas = self._species_tree_last_canvas
        if layout is not None and canvas is not None:
            self._update_species_tree_zoom_and_limits(layout, canvas)
    def _adjust_species_tree_zoom(
        self,
        factor: float,
        *,
        anchor: tuple[float, float] | None = None,
    ) -> None:
        """Adjust species tree zoom.

        Parameters
        ----------
        factor
            Value used by the operation.
        anchor
            Rectangle defining the relevant UI area.
        """
        layout = self._species_tree_last_layout
        canvas = self._species_tree_last_canvas
        if layout is None or canvas is None or factor <= 0.0:
            return

        old_zoom = max(0.0001, self._species_tree_zoom)
        if old_zoom < self.SPECIES_TREE_MIN_ZOOM and factor < 1.0:
            return
        old_horizontal_inset, old_vertical_inset = (
            self._species_tree_content_insets(layout, canvas, old_zoom)
        )
        anchor_x, anchor_y = anchor or (canvas.center_x, canvas.center_y)
        anchor_x = max(canvas.left, min(canvas.right, anchor_x))
        anchor_y = max(canvas.bottom, min(canvas.top, anchor_y))
        content_center_x = (
            anchor_x
            - canvas.left
            - old_horizontal_inset
            + self._species_tree_horizontal_offset
        ) / old_zoom
        content_center_y = (
            canvas.top
            - old_vertical_inset
            + self._species_tree_vertical_offset
            - anchor_y
        ) / old_zoom

        # Convert the anchor to content coordinates before changing zoom, then
        # solve the new offsets so the same content remains beneath the cursor.
        self._species_tree_fit_mode = False
        self._species_tree_fit_requested = False
        self._species_tree_zoom = max(
            self.SPECIES_TREE_MIN_ZOOM,
            min(self.SPECIES_TREE_MAX_ZOOM, old_zoom * factor),
        )
        new_horizontal_inset, new_vertical_inset = (
            self._species_tree_content_insets(
                layout,
                canvas,
                self._species_tree_zoom,
            )
        )
        self._species_tree_horizontal_offset = (
            canvas.left
            + new_horizontal_inset
            + content_center_x * self._species_tree_zoom
            - anchor_x
        )
        self._species_tree_vertical_offset = (
            anchor_y
            - canvas.top
            + new_vertical_inset
            + content_center_y * self._species_tree_zoom
        )
        self._update_species_tree_zoom_and_limits(layout, canvas)
    def _draw_species_tree_path(
        self,
        points: SpeciesTreeRoute,
        color: arcade.Color | tuple[int, ...],
        width: float,
        *,
        dashed: bool = False,
    ) -> None:
        """Draw species tree path.

        Parameters
        ----------
        points
            Value used by the operation.
        color
            Arcade-compatible color.
        width
            Requested logical size.
        dashed
            Value used by the operation.
        """
        if not dashed:
            self._draw_curve(list(points), color, width)
            return
        dash_length = 6.0
        gap_length = 4.0
        for start, end in zip(points, points[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = hypot(dx, dy)
            if length <= 0.0:
                continue
            offset = 0.0
            while offset < length:
                dash_end = min(length, offset + dash_length)
                arcade.draw_line(
                    start[0] + dx * offset / length,
                    start[1] + dy * offset / length,
                    start[0] + dx * dash_end / length,
                    start[1] + dy * dash_end / length,
                    color,
                    width,
                )
                offset += dash_length + gap_length
    def _species_tree_node_visual_radius(
        self,
        layout: SpeciesTreeLayout,
        species_id: int,
    ) -> float:
        """Return species tree node visual radius.

        Parameters
        ----------
        layout
            Value used by the operation.
        species_id
            Value used by the operation.

        Returns
        -------
        float
            Computed result.
        """
        descendants = max(0, layout.descendant_counts.get(species_id, 0))
        base = 6.25 + min(2.25, log1p(descendants) * 0.5)
        return max(4.25, min(11.0, base * self._species_tree_zoom))
    def _clamp_species_tree_offsets(self) -> None:
        """Return clamp species tree offsets.
        """
        self._species_tree_horizontal_offset = max(
            self._species_tree_horizontal_offset_min,
            min(
                self._species_tree_horizontal_offset_max,
                self._species_tree_horizontal_offset,
            ),
        )
        self._species_tree_vertical_offset = max(
            self._species_tree_vertical_offset_min,
            min(
                self._species_tree_vertical_offset_max,
                self._species_tree_vertical_offset,
            ),
        )
    def _set_species_tree_scroll_from_pointer(
        self,
        axis: str,
        x: float,
        y: float,
        *,
        preserve_grab_offset: bool = False,
    ) -> None:
        """Set species tree scroll from pointer.

        Parameters
        ----------
        axis
            Value used by the operation.
        x
            Logical screen coordinate.
        y
            Logical screen coordinate.
        preserve_grab_offset
            Value used by the operation.
        """
        horizontal = axis == "horizontal"
        track_key = (
            "species_tree_horizontal_track"
            if horizontal
            else "species_tree_vertical_track"
        )
        thumb_key = (
            "species_tree_horizontal_thumb"
            if horizontal
            else "species_tree_vertical_thumb"
        )
        track = self._control_hitboxes.get(track_key)
        thumb = self._control_hitboxes.get(thumb_key)
        if track is None or thumb is None:
            return

        track_length = track.width if horizontal else track.height
        thumb_length = thumb.width if horizontal else thumb.height
        travel = max(0.0, track_length - thumb_length)
        limit = (
            self._species_tree_horizontal_limit
            if horizontal
            else self._species_tree_vertical_limit
        )
        if travel <= 0.0 or limit <= 0.0:
            position = 0.0
        elif horizontal:
            grab = (
                self._species_tree_scroll_drag_offset
                if preserve_grab_offset
                else thumb_length * 0.5
            )
            position = x - track.left - grab
        else:
            grab = (
                self._species_tree_scroll_drag_offset
                if preserve_grab_offset
                else thumb_length * 0.5
            )
            position = track.top - y - grab
        offset = 0.0 if travel <= 0.0 else limit * position / travel
        if horizontal:
            self._species_tree_horizontal_offset = offset
        else:
            self._species_tree_vertical_offset = offset
        self._clamp_species_tree_offsets()
