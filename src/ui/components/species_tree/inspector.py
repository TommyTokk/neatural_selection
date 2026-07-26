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

class SpeciesTreeInspectorComponent:
    """Group related behavior extracted from ``UiRenderer``."""

    SPECIES_RADAR_MIN_SIZE = 220.0
    SPECIES_RADAR_MAX_SIZE = 440.0
    SPECIES_RADAR_WIDTH_RATIO = 0.62

    def _species_tree_inspector_width_limits(
        self,
        content: arcade.Rect,
    ) -> tuple[float, float]:
        """Return species tree inspector width limits.

        Parameters
        ----------
        content
            Value used by the operation.

        Returns
        -------
        tuple[float, float]
            Computed collection.
        """
        max_width = max(0.0, content.width * 2.0 / 3.0)
        min_width = min(300.0, max_width)
        return min_width, max_width
    def _species_tree_inspector_default_width(
        self,
        content: arcade.Rect,
    ) -> float:
        """Return species tree inspector default width.

        Parameters
        ----------
        content
            Value used by the operation.

        Returns
        -------
        float
            Computed result.
        """
        min_width, max_width = self._species_tree_inspector_width_limits(
            content
        )
        preferred = max(340.0, content.width * 0.38)
        return max(min_width, min(max_width, preferred))
    def _species_tree_inspector_clamped_width(
        self,
        content: arcade.Rect,
    ) -> float:
        """Return species tree inspector clamped width.

        Parameters
        ----------
        content
            Value used by the operation.

        Returns
        -------
        float
            Computed result.
        """
        min_width, max_width = self._species_tree_inspector_width_limits(
            content
        )
        requested = (
            self._species_tree_inspector_default_width(content)
            if self._species_tree_inspector_width is None
            else self._species_tree_inspector_width
        )
        width = max(min_width, min(max_width, requested))
        self._species_tree_inspector_width = width
        return width
    def _radar_executor(self) -> ThreadPoolExecutor:
        """Return radar executor.

        Returns
        -------
        ThreadPoolExecutor
            Computed result.
        """
        executor = self._species_tree_radar_executor
        if executor is None:
            # A single worker serializes Matplotlib access and avoids creating a
            # new thread for each selected species.
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="species-radar",
            )
            self._species_tree_radar_executor = executor
        return executor
    def _clear_species_radar_state(self) -> None:
        """Clear species radar state.
        """
        future = self._species_tree_radar_future
        if future is not None:
            future.cancel()
        self._species_tree_radar_future = None
        self._species_tree_radar_texture = None
        self._species_tree_radar_species_id = None
        self._species_tree_radar_error = None
    def _consume_species_radar_result(self) -> None:
        """Return consume species radar result.
        """
        future = self._species_tree_radar_future
        if future is None or not future.done():
            return
        self._species_tree_radar_future = None
        if (
            not self._species_tree_open
            or self._species_tree_radar_species_id
            != self._species_tree_selected_id
        ):
            # Ignore results from a selection that was closed or superseded.
            return
        try:
            radar_image = future.result()
        except Exception:
            self._species_tree_radar_error = "render_failed"
            return

        hitbox_algorithm = getattr(
            getattr(arcade, "hitbox", None),
            "algo_bounding_box",
            None,
        )
        texture_options = (
            {}
            if hitbox_algorithm is None
            else {"hit_box_algorithm": hitbox_algorithm}
        )
        self._species_tree_radar_texture = arcade.Texture(
            radar_image,
            **texture_options,
        )
    @staticmethod
    def _pressure_label(value: float) -> str:
        """Return pressure label.

        Parameters
        ----------
        value
            Value used by the operation.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        if value < 1.0 / 3.0:
            return "Low"
        if value < 2.0 / 3.0:
            return "Moderate"
        return "High"
    @staticmethod
    def _format_percent_change(value: float | None) -> str:
        """Format percent change.

        Parameters
        ----------
        value
            Value used by the operation.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        return "change unavailable" if value is None else f"{value:+.2f}% vs parent"
    def _species_tree_neat_node_labels(self, world: World) -> dict[int, str]:
        """Return species tree neat node labels.

        Parameters
        ----------
        world
            Simulation world providing current state.

        Returns
        -------
        dict[int, str]
            Computed collection.
        """
        controller = getattr(world, "neat_controller", None)
        config = getattr(controller, "config", None)
        genome_config = getattr(config, "genome_config", None)
        if genome_config is None:
            self._species_tree_neat_label_signature = None
            self._species_tree_neat_labels = _EMPTY_NEAT_NODE_LABELS
            return _EMPTY_NEAT_NODE_LABELS
        input_keys = tuple(getattr(genome_config, "input_keys", ()) or ())
        output_keys = tuple(getattr(genome_config, "output_keys", ()) or ())
        signature = (input_keys, output_keys)
        if signature == self._species_tree_neat_label_signature:
            return self._species_tree_neat_labels
        labels = {
            int(key): SENSOR_INPUT_NAMES[index]
            for index, key in enumerate(input_keys)
            if index < len(SENSOR_INPUT_NAMES)
        }
        labels.update(
            {
                int(key): ACTION_OUTPUT_NAMES[index]
                for index, key in enumerate(output_keys)
                if index < len(ACTION_OUTPUT_NAMES)
            }
        )
        self._species_tree_neat_label_signature = signature
        self._species_tree_neat_labels = labels
        return labels
    @staticmethod
    def _format_species_tree_neat_change(
        line: str,
        node_labels: dict[int, str],
    ) -> str:
        """Format species tree neat change.

        Parameters
        ----------
        line
            Value used by the operation.
        node_labels
            Value used by the operation.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        node_match = re.fullmatch(
            r"Node (-?\d+) (added|removed)",
            line,
        )
        if node_match is not None:
            key = int(node_match.group(1))
            label = node_labels.get(key, str(key))
            return f"Node {label} {node_match.group(2)}"

        parameter_match = re.fullmatch(
            r"Node (-?\d+) (bias|response|activation|aggregation) (.+) -> (.+)",
            line,
        )
        if parameter_match is not None:
            key = int(parameter_match.group(1))
            label = node_labels.get(key, str(key))
            return (
                f"Node {label} {parameter_match.group(2)} "
                f"{parameter_match.group(3)} -> {parameter_match.group(4)}"
            )

        connection_match = re.fullmatch(
            r"(Connection|Weight) (-?\d+)->(-?\d+)( .+)",
            line,
        )
        if connection_match is not None:
            source_key = int(connection_match.group(2))
            target_key = int(connection_match.group(3))
            source = node_labels.get(source_key, str(source_key))
            target = node_labels.get(target_key, str(target_key))
            return (
                f"{connection_match.group(1)} {source} -> {target}"
                f"{connection_match.group(4)}"
            )
        return line
    def _format_trait_value(
        self,
        snapshot: object | None,
        attribute: str,
        *,
        signed: bool = False,
    ) -> str:
        """Format trait value.

        Parameters
        ----------
        snapshot
            Value used by the operation.
        attribute
            Value used by the operation.
        signed
            Value used by the operation.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        value = None if snapshot is None else getattr(snapshot, attribute, None)
        return self._format_optional_number(value, signed=signed)
    def _format_optional_number(
        self,
        value: object,
        digits: int = 3,
        *,
        signed: bool = False,
    ) -> str:
        """Format optional number.

        Parameters
        ----------
        value
            Value used by the operation.
        digits
            Value used by the operation.
        signed
            Value used by the operation.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        if value is None:
            return "Unavailable"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "Unavailable"
        return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"

    def _ensure_species_inspector_report(
        self,
        world: World,
        records: dict[int, SpeciesRecord],
    ) -> None:
        """Ensure species inspector report.

        Parameters
        ----------
        world
            Simulation world providing current state.
        records
            Species history data to inspect.
        """
        species_id = self._species_tree_selected_id
        if (
            species_id is None
            or species_id not in records
            or self._species_tree_report_species_id == species_id
        ):
            return
        record = records[species_id]
        parent = (
            records.get(record.parent_species_id)
            if record.parent_species_id is not None
            else None
        )
        telemetry = getattr(world, "telemetry", None)
        connection = getattr(telemetry, "connection", None)
        controller_config = getattr(
            getattr(world, "neat_controller", None),
            "config",
            None,
        )
        genome_config = getattr(controller_config, "genome_config", None)
        input_keys_source = getattr(genome_config, "input_keys", None)
        input_keys = (
            None if input_keys_source is None else tuple(input_keys_source)
        )
        output_keys = tuple(
            getattr(genome_config, "output_keys", ()) or ()
        )
        self._species_tree_report = generate_inspector_report(
            record,
            parent,
            connection,
            world.config,
            output_keys,
            input_keys,
        )
        self._clear_species_radar_state()
        self._species_tree_radar_species_id = species_id
        controller = getattr(world, "neat_controller", None)
        species_manager = getattr(controller, "species_manager", None)
        representatives = getattr(species_manager, "representatives", {}) or {}
        child_representative = representatives.get(species_id)
        parent_representative = (
            representatives.get(record.parent_species_id)
            if record.parent_species_id is not None
            else None
        )
        child_genome = self._species_representative_genome(
            child_representative
        )
        parent_genome = self._species_representative_genome(
            parent_representative
        )
        if child_genome is not None:
            child_scores = calculate_behavior_scores(
                child_genome,
                output_keys,
            )
            parent_scores = (
                None
                if parent_genome is None
                else calculate_behavior_scores(parent_genome, output_keys)
            )
            # Radar image generation uses Matplotlib and must not stall Arcade's
            # render loop; the finished image is consumed on a later frame.
            self._species_tree_radar_future = self._radar_executor().submit(
                generate_radar_chart_image,
                child_scores,
                parent_scores,
                BEHAVIOR_RADAR_LABELS,
            )
        else:
            self._species_tree_radar_error = "representative_unavailable"
        self._species_tree_report_species_id = species_id
        self._scroll_offsets["species_tree_inspector"] = 0.0
    @staticmethod
    def _species_representative_genome(representative: object) -> object | None:
        """Return species representative genome.

        Parameters
        ----------
        representative
            Value used by the operation.

        Returns
        -------
        object | None
            Computed result.
        """
        if not isinstance(representative, tuple) or len(representative) != 4:
            return None
        genome = representative[0]
        if not hasattr(genome, "nodes") or not hasattr(genome, "connections"):
            return None
        return genome
    def _draw_species_inspector(
        self,
        bounds: arcade.Rect,
        report: InspectorReport | None,
        record: SpeciesRecord | None = None,
    ) -> None:
        """Draw species inspector.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        report
            Value used by the operation.
        record
            Species history data to inspect.
        """
        self._draw_rounded_rect(
            bounds,
            self.theme.card_background,
            self.theme.panel_border,
            self.config.layout.card_radius,
            1.0,
        )
        resize_hitbox = arcade.LBWH(
            bounds.left - 5.0,
            bounds.bottom,
            10.0,
            bounds.height,
        )
        self._control_hitboxes["species_tree_inspector_resize"] = resize_hitbox

        header = arcade.LBWH(bounds.left, bounds.top - 44.0, bounds.width, 44.0)
        self._draw_text(
            "species_tree_inspector_heading",
            "SPECIES INSPECTOR",
            bounds.left + 16.0,
            bounds.top - 24.0,
            self.theme.text_primary,
            10,
            bold=True,
            anchor_y="center",
        )
        arcade.draw_line(
            bounds.left + 12.0,
            header.bottom,
            bounds.right - 12.0,
            header.bottom,
            self.theme.panel_border,
            1.0,
        )

        summary = arcade.LBWH(
            bounds.left + 14.0,
            header.bottom - 68.0,
            bounds.width - 28.0,
            58.0,
        )
        marker_radius = 10.0
        marker_x = summary.left + marker_radius
        marker_y = summary.center_y
        if record is not None:
            arcade.draw_circle_filled(
                marker_x,
                marker_y,
                marker_radius,
                record.founder_color or self.theme.herbivore_fill,
            )
            arcade.draw_circle_outline(
                marker_x,
                marker_y,
                marker_radius,
                self.theme.selected_outline,
                3.0,
            )
        else:
            arcade.draw_circle_filled(
                marker_x,
                marker_y,
                marker_radius,
                self.theme.herbivore_fill,
            )
            arcade.draw_circle_outline(
                marker_x,
                marker_y,
                marker_radius,
                self.theme.panel_border,
                2.0,
            )

        species_id = (
            report.species_id
            if report is not None
            else (None if record is None else record.species_id)
        )
        quality_label = None
        badge = None
        if record is not None:
            quality_label = (
                "Exact"
                if record.data_quality.lower() == "exact"
                else "Reconstructed"
            )
            badge_width = 58.0 if quality_label == "Exact" else 96.0
            badge = arcade.LBWH(
                summary.right - badge_width,
                summary.center_y - 12.0,
                badge_width,
                24.0,
            )
            self._draw_rounded_rect(
                badge,
                self._brain_blend_color(
                    self.theme.card_background,
                    self.theme.accent,
                    0.10,
                ),
                self._brain_blend_color(
                    self.theme.card_background,
                    self.theme.accent,
                    0.42,
                ),
                6.0,
                1.0,
            )
            self._draw_text(
                "species_tree_inspector_quality",
                quality_label,
                badge.center_x,
                badge.center_y,
                self.theme.accent,
                8.5,
                bold=True,
                anchor_x="center",
                anchor_y="center",
            )

        title_left = marker_x + marker_radius + 10.0
        title_right = summary.right if badge is None else badge.left - 10.0
        self._draw_text(
            "species_tree_inspector_title",
            (
                "Select a species"
                if species_id is None
                else f"Species {species_id} Inspector"
            ),
            title_left,
            summary.center_y,
            self.theme.text_primary,
            14,
            bold=True,
            width=max(20.0, title_right - title_left),
            multiline=True,
            anchor_y="center",
        )
        navigation = arcade.LBWH(
            bounds.left + 16.0,
            summary.bottom - 38.0,
            max(0.0, bounds.width - 32.0),
            30.0,
        )
        self._draw_species_parent_navigation(navigation, record)
        viewport = arcade.LBWH(
            bounds.left + 16.0,
            bounds.bottom + 14.0,
            max(0.0, bounds.width - 32.0),
            max(0.0, navigation.bottom - bounds.bottom - 22.0),
        )
        self._consume_species_radar_result()
        self._draw_species_inspector_content(viewport, report, record)
    def _draw_species_parent_navigation(
        self,
        bounds: arcade.Rect,
        record: SpeciesRecord | None,
    ) -> None:
        """Draw species parent navigation.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        record
            Species history data to inspect.
        """
        parent_id = None if record is None else record.parent_species_id
        layout = self._species_tree_last_layout
        parent_available = (
            parent_id is not None
            and layout is not None
            and parent_id in layout.positions
        )
        if parent_available:
            self._control_hitboxes["species_tree_parent_button"] = bounds
            fill = self._brain_blend_color(
                self.theme.card_background,
                self.theme.accent_soft,
                0.18,
            )
            border = self._brain_blend_color(
                self.theme.card_background,
                self.theme.accent,
                0.62,
            )
            label = f"Go to Parent · Species {parent_id}"
            text_color = self.theme.accent
        else:
            self._control_hitboxes.pop("species_tree_parent_button", None)
            fill = self.theme.card_background
            border = self.theme.panel_border
            label = (
                "Select a species"
                if record is None
                else "Founder species · No parent"
                if parent_id is None
                else "Parent species unavailable"
            )
            text_color = self.theme.text_muted
        self._draw_rounded_rect(bounds, fill, border, 7.0, 1.25)
        self._draw_text(
            "species_tree_parent_navigation",
            self._fit_line(label, max(20.0, bounds.width - 16.0)),
            bounds.center_x,
            bounds.center_y,
            text_color,
            10.5,
            bold=parent_available,
            anchor_x="center",
            anchor_y="center",
        )
    def _draw_species_inspector_content(
        self,
        viewport: arcade.Rect,
        report: InspectorReport | None,
        record: SpeciesRecord | None,
    ) -> None:
        """Draw species inspector content.

        Parameters
        ----------
        viewport
            Rectangle defining the relevant UI area.
        report
            Value used by the operation.
        record
            Species history data to inspect.
        """
        sections = self._species_inspector_sections(report, record)
        radar_size = (
            self._species_radar_chart_size(viewport.width)
            if self._species_tree_radar_species_id is not None
            else 0.0
        )
        content_width = max(24.0, viewport.width - 12.0)
        total_height = 12.0
        for section_index, section in enumerate(sections):
            total_height += 34.0
            total_height += sum(
                self._species_inspector_row_height(row, content_width)
                for row in section.rows
            )
            total_height += 8.0
            if section_index == 0 and radar_size > 0.0:
                total_height += 34.0 + radar_size + 12.0

        scroll_limit = max(0.0, total_height - viewport.height)
        scroll_offset = max(
            0.0,
            min(
                scroll_limit,
                self._scroll_offsets.get("species_tree_inspector", 0.0),
            ),
        )
        self._scroll_offsets["species_tree_inspector"] = scroll_offset
        self._scroll_limits["species_tree_inspector"] = scroll_limit
        self._scroll_regions["species_tree_inspector"] = viewport

        cursor = viewport.top - 8.0 + scroll_offset
        with self._ui_clip(viewport):
            for section_index, section in enumerate(sections):
                self._draw_species_inspector_section_header(
                    viewport,
                    section_index,
                    section.title,
                    cursor,
                )
                cursor -= 34.0
                for row_index, row in enumerate(section.rows):
                    row_height = self._species_inspector_row_height(
                        row,
                        content_width,
                    )
                    self._draw_species_inspector_row(
                        viewport,
                        section_index,
                        row_index,
                        row,
                        cursor,
                        content_width,
                    )
                    cursor -= row_height
                cursor -= 8.0

                if section_index == 0 and radar_size > 0.0:
                    self._draw_species_inspector_section_header(
                        viewport,
                        -1,
                        "BEHAVIORAL PROFILE",
                        cursor,
                    )
                    cursor -= 34.0
                    chart_bounds = arcade.LBWH(
                        viewport.center_x - radar_size / 2.0,
                        cursor - radar_size,
                        radar_size,
                        radar_size,
                    )
                    if self._rect_intersects(chart_bounds, viewport):
                        self._draw_species_radar_chart_in_bounds(chart_bounds)
                    cursor -= radar_size + 12.0

        if scroll_limit > 0.0:
            self._draw_scrollbar(viewport, scroll_offset, scroll_limit)
    def _draw_species_inspector_section_header(
        self,
        viewport: arcade.Rect,
        section_index: int,
        title: str,
        top: float,
    ) -> None:
        """Draw species inspector section header.

        Parameters
        ----------
        viewport
            Rectangle defining the relevant UI area.
        section_index
            Value used by the operation.
        title
            Text displayed by the UI.
        top
            Logical screen coordinate.
        """
        baseline = top - 14.0
        divider_y = top - 27.0
        if viewport.bottom <= baseline <= viewport.top:
            key_suffix = "radar" if section_index < 0 else str(section_index)
            self._draw_text(
                f"species_tree_inspector_section_{key_suffix}",
                title,
                viewport.left,
                baseline,
                self.theme.text_primary,
                9.5,
                bold=True,
            )
        if viewport.bottom <= divider_y <= viewport.top:
            arcade.draw_line(
                viewport.left,
                divider_y,
                viewport.right - 8.0,
                divider_y,
                self.theme.panel_border,
                1.0,
            )
    def _species_inspector_row_height(
        self,
        row: _SpeciesInspectorRow,
        width: float,
    ) -> float:
        """Return species inspector row height.

        Parameters
        ----------
        row
            Value used by the operation.
        width
            Requested logical size.

        Returns
        -------
        float
            Computed result.
        """
        if row.label is None:
            indent = 18.0 if row.marker_color is not None else 0.0
            line_count = len(
                self._wrap_line(row.value, max(24.0, width - indent))
            )
            return max(1, line_count) * 15.0 + 10.0
        gap = 12.0
        label_width = max(72.0, width * 0.38)
        value_width = max(24.0, width - label_width - gap)
        line_count = max(
            len(self._wrap_line(row.label, label_width)),
            len(self._wrap_line(row.value, value_width)),
        )
        return max(1, line_count) * 15.0 + 10.0
    def _draw_species_inspector_row(
        self,
        viewport: arcade.Rect,
        section_index: int,
        row_index: int,
        row: _SpeciesInspectorRow,
        top: float,
        width: float,
    ) -> None:
        """Draw species inspector row.

        Parameters
        ----------
        viewport
            Rectangle defining the relevant UI area.
        section_index
            Value used by the operation.
        row_index
            Value used by the operation.
        row
            Value used by the operation.
        top
            Logical screen coordinate.
        width
            Requested logical size.
        """
        color = self._species_inspector_tone_color(row.tone)
        if row.label is None:
            indent = 18.0 if row.marker_color is not None else 0.0
            lines = self._wrap_line(row.value, max(24.0, width - indent))
            first_y = top - 13.0
            if row.marker_color is not None and viewport.bottom <= first_y <= viewport.top:
                arcade.draw_circle_filled(
                    viewport.left + 6.0,
                    first_y + 4.0,
                    4.5,
                    row.marker_color,
                )
            for line_index, line in enumerate(lines):
                y = first_y - line_index * 15.0
                if viewport.bottom <= y <= viewport.top:
                    self._draw_text(
                        f"species_tree_inspector_row_{section_index}_{row_index}_{line_index}",
                        line,
                        viewport.left + indent,
                        y,
                        color,
                        10.5,
                        bold=row.tone == "primary",
                    )
            return

        gap = 12.0
        label_width = max(72.0, width * 0.38)
        value_width = max(24.0, width - label_width - gap)
        label_lines = self._wrap_line(row.label, label_width)
        value_lines = self._wrap_line(row.value, value_width)
        first_y = top - 13.0
        for line_index, line in enumerate(label_lines):
            y = first_y - line_index * 15.0
            if viewport.bottom <= y <= viewport.top:
                self._draw_text(
                    f"species_tree_inspector_row_{section_index}_{row_index}_label_{line_index}",
                    line,
                    viewport.left,
                    y,
                    self.theme.text_muted,
                    10,
                )
        value_left = viewport.left + label_width + gap
        for line_index, line in enumerate(value_lines):
            y = first_y - line_index * 15.0
            if viewport.bottom <= y <= viewport.top:
                self._draw_text(
                    f"species_tree_inspector_row_{section_index}_{row_index}_value_{line_index}",
                    line,
                    value_left,
                    y,
                    color,
                    10.5,
                    bold=row.tone in {"positive", "negative", "primary"},
                )
    def _species_inspector_tone_color(
        self,
        tone: str,
    ) -> arcade.Color | tuple[int, ...]:
        """Return species inspector tone color.

        Parameters
        ----------
        tone
            Value used by the operation.

        Returns
        -------
        arcade.Color | tuple[int, ...]
            Computed result.
        """
        if tone == "positive":
            return (20, 139, 70)
        if tone == "negative":
            return self.theme.selected_outline
        if tone == "muted":
            return self.theme.text_muted
        return self.theme.text_primary
    def _draw_species_radar_chart(
        self,
        viewport: arcade.Rect,
    ) -> arcade.Rect:
        """Draw species radar chart.

        Parameters
        ----------
        viewport
            Rectangle defining the relevant UI area.

        Returns
        -------
        arcade.Rect
            Computed UI rectangle.
        """
        if self._species_tree_radar_species_id is None:
            return viewport
        chart_size = min(
            self._species_radar_chart_size(viewport.width),
            max(0.0, viewport.height - 96.0),
        )
        if chart_size <= 0.0:
            return viewport
        chart_bounds = arcade.LBWH(
            viewport.center_x - chart_size / 2.0,
            viewport.top - chart_size,
            chart_size,
            chart_size,
        )
        self._draw_species_radar_chart_in_bounds(chart_bounds)
        gap = 8.0
        return arcade.LBWH(
            viewport.left,
            viewport.bottom,
            viewport.width,
            max(0.0, chart_bounds.bottom - gap - viewport.bottom),
        )
    def _species_radar_chart_size(self, viewport_width: float) -> float:
        """Return a responsive radar size for the inspector width."""
        available_width = max(0.0, viewport_width - 24.0)
        preferred_width = max(
            self.SPECIES_RADAR_MIN_SIZE,
            viewport_width * self.SPECIES_RADAR_WIDTH_RATIO,
        )
        return min(
            self.SPECIES_RADAR_MAX_SIZE,
            available_width,
            preferred_width,
        )
    def _draw_species_radar_chart_in_bounds(
        self,
        chart_bounds: arcade.Rect,
    ) -> None:
        """Draw species radar chart in bounds.

        Parameters
        ----------
        chart_bounds
            Value used by the operation.
        """
        texture = self._species_tree_radar_texture
        if texture is not None:
            arcade.draw_texture_rect(texture, chart_bounds)
        else:
            message = (
                "Loading behavioral profile..."
                if self._species_tree_radar_future is not None
                else "Behavioral profile unavailable"
            )
            self._draw_text(
                "species_tree_radar_status",
                message,
                chart_bounds.center_x,
                chart_bounds.center_y,
                self.theme.text_muted,
                11,
                anchor_x="center",
                anchor_y="center",
            )
    def _species_inspector_sections(
        self,
        report: InspectorReport | None,
        record: SpeciesRecord | None,
    ) -> tuple[_SpeciesInspectorSection, ...]:
        """Return species inspector sections.

        Parameters
        ----------
        report
            Value used by the operation.
        record
            Species history data to inspect.

        Returns
        -------
        tuple[_SpeciesInspectorSection, ...]
            Computed collection.
        """
        if record is None:
            overview_rows = (
                _SpeciesInspectorRow(
                    None,
                    "Select a species node to generate its report.",
                    "muted",
                ),
            )
        else:
            emerged_at = self._valid_species_tree_time(record.emerged_at)
            overview_rows = (
                _SpeciesInspectorRow(
                    "Parent species",
                    (
                        "None"
                        if record.parent_species_id is None
                        else f"Species {record.parent_species_id}"
                    ),
                ),
                _SpeciesInspectorRow(
                    "Emergence",
                    (
                        "Unavailable"
                        if emerged_at is None
                        else self._format_species_tree_time(emerged_at)
                    ),
                ),
                _SpeciesInspectorRow(
                    "Founder creature",
                    (
                        "Unavailable"
                        if record.founder_creature_id is None
                        else str(record.founder_creature_id)
                    ),
                ),
                _SpeciesInspectorRow(
                    "Founder genome",
                    (
                        "Unavailable"
                        if record.founder_genome_id is None
                        else str(record.founder_genome_id)
                    ),
                ),
            )
        sections: list[_SpeciesInspectorSection] = [
            _SpeciesInspectorSection("OVERVIEW", overview_rows)
        ]
        if report is None:
            return tuple(sections)

        sections.append(
            _SpeciesInspectorSection(
                "ECOLOGICAL CONTEXT",
                (
                    _SpeciesInspectorRow(
                        "Food scarcity",
                        (
                            "Unavailable"
                            if report.food_scarcity is None
                            else (
                                f"{report.food_scarcity:.0%} "
                                f"({self._pressure_label(report.food_scarcity)})"
                            )
                        ),
                    ),
                    _SpeciesInspectorRow(
                        "Population density",
                        (
                            "Unavailable"
                            if report.population_density is None
                            else (
                                f"{report.population_density:.0%} "
                                f"({self._pressure_label(report.population_density)})"
                            )
                        ),
                    ),
                ),
            )
        )

        traits = report.species_traits
        if traits is None:
            anatomy_rows = (
                _SpeciesInspectorRow(None, "Species traits unavailable", "muted"),
            )
        else:
            anatomy_rows = (
                _SpeciesInspectorRow("Radius", f"{traits.radius:.2f} px"),
                _SpeciesInspectorRow(
                    "Vision range",
                    f"{traits.vision_range:.2f} px",
                ),
                _SpeciesInspectorRow(
                    "Vision angle",
                    f"{traits.vision_angle:.3f} rad",
                ),
                _SpeciesInspectorRow(
                    "Movement cost",
                    f"{traits.movement_cost_multiplier:.3f}x",
                ),
                _SpeciesInspectorRow(
                    "Separation gene",
                    f"{traits.separation_gene:.3f}",
                ),
                _SpeciesInspectorRow(
                    "Alignment gene",
                    f"{traits.alignment_gene:.3f}",
                ),
                _SpeciesInspectorRow(
                    "Cohesion gene",
                    f"{traits.cohesion_gene:.3f}",
                ),
            )
        sections.append(
            _SpeciesInspectorSection("ANATOMY & MORPHOLOGY", anatomy_rows)
        )

        if report.morphology:
            parent_rows = tuple(
                _SpeciesInspectorRow(
                    insight.description,
                    f"{insight.percent_change:+.1f}%",
                    (
                        "positive"
                        if insight.percent_change > 0.0
                        else "negative"
                    ),
                )
                for insight in report.morphology
            )
        else:
            parent_rows = (
                _SpeciesInspectorRow(
                    None,
                    (
                        "No parent comparison"
                        if report.parent_species_id is None
                        else "No measurable morphology change"
                    ),
                    "muted",
                ),
            )
        sections.append(
            _SpeciesInspectorSection("PARENT COMPARISON", parent_rows)
        )

        metabolism = report.metabolism
        if metabolism.child_idle_cost is None:
            metabolism_rows = (
                _SpeciesInspectorRow(
                    None,
                    "Species metabolism unavailable",
                    "muted",
                ),
            )
        else:
            metabolism_rows_list = [
                _SpeciesInspectorRow(
                    "Basal metabolic BMR",
                    (
                        f"{metabolism.child_idle_cost:.5f} energy/s "
                        f"({self._format_percent_change(metabolism.idle_percent_change)})"
                    ),
                    self._species_inspector_change_tone(
                        metabolism.idle_percent_change
                    ),
                ),
                _SpeciesInspectorRow(
                    "Active foraging cost",
                    (
                        f"{metabolism.child_active_cost:.5f} energy/s "
                        f"({self._format_percent_change(metabolism.active_percent_change)})"
                    ),
                    self._species_inspector_change_tone(
                        metabolism.active_percent_change
                    ),
                ),
            ]
            if (
                metabolism.parent_idle_cost is not None
                and metabolism.parent_active_cost is not None
            ):
                metabolism_rows_list.append(
                    _SpeciesInspectorRow(
                        "Parent BMR / active",
                        (
                            f"{metabolism.parent_idle_cost:.5f} / "
                            f"{metabolism.parent_active_cost:.5f}"
                        ),
                    )
                )
            metabolism_rows = tuple(metabolism_rows_list)
        sections.append(
            _SpeciesInspectorSection("METABOLIC PROFILE", metabolism_rows)
        )

        hub_rows: list[_SpeciesInspectorRow] = []
        if not report.neuro_integration_hubs:
            hub_rows.append(
                _SpeciesInspectorRow(
                    None,
                    "No evolving interneuron hubs detected",
                    "muted",
                )
            )
        for hub in report.neuro_integration_hubs:
            hub_rows.append(
                _SpeciesInspectorRow(
                    None,
                    f"Integration Hub {hub.hub_id}",
                    "primary",
                )
            )
            for description in hub.sensory_integrations:
                hub_rows.append(_SpeciesInspectorRow(None, description))
            for description in hub.behavioral_modulations:
                hub_rows.append(_SpeciesInspectorRow(None, description))
        sections.append(
            _SpeciesInspectorSection(
                "NEURO-INTEGRATION HUBS",
                tuple(hub_rows),
            )
        )

        ethogram_rows: list[_SpeciesInspectorRow] = []
        if not report.behavioral_ethogram:
            ethogram_rows.append(
                _SpeciesInspectorRow(
                    None,
                    "No direct stimulus-response reflex shifts",
                    "muted",
                )
            )
        for reflex in report.behavioral_ethogram:
            description = reflex.description
            marker_color = (
                self._ethogram_marker_color(description[0])
                if description
                else None
            )
            if marker_color is not None:
                description = description[1:].lstrip()
            ethogram_rows.append(
                _SpeciesInspectorRow(
                    None,
                    description,
                    (
                        "positive"
                        if reflex.weight_delta > 0.0
                        else (
                            "negative"
                            if reflex.weight_delta < 0.0
                            else "muted"
                        )
                    ),
                    marker_color,
                )
            )
        sections.append(
            _SpeciesInspectorSection(
                "BEHAVIORAL ETHOGRAM",
                tuple(ethogram_rows),
            )
        )

        sections.append(
            _SpeciesInspectorSection(
                "SPECIES LEGACY",
                (
                    _SpeciesInspectorRow(
                        "Descendants",
                        (
                            "Unavailable"
                            if report.legacy.descendant_count is None
                            else str(report.legacy.descendant_count)
                        ),
                    ),
                    _SpeciesInspectorRow(
                        "Average lifespan",
                        (
                            "Unavailable"
                            if report.legacy.average_lifespan is None
                            else f"{report.legacy.average_lifespan:.2f} s"
                        ),
                    ),
                ),
            )
        )
        return tuple(sections)
    @staticmethod
    def _species_inspector_change_tone(value: float | None) -> str:
        """Return species inspector change tone.

        Parameters
        ----------
        value
            Value used by the operation.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        if value is None or value == 0.0:
            return "default"
        return "positive" if value > 0.0 else "negative"
