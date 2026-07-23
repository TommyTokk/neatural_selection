from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
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
from src.brain_graph import (
    BrainEdgeKind,
    BrainGraphEdge,
    BrainGraphLayout,
    BrainGraphNode,
    BrainNodeKind,
    build_brain_graph_layout,
    highlighted_path_through_node,
)
from src.speciation import SpeciesRecord
from src.species_tree import (
    SpeciesTreeLayout,
    SpeciesTreeRoute,
    TreeLayoutManager,
    TreeViewportSlice,
    species_tree_line_width,
)
from src.vision import SENSOR_INPUT_NAMES
from src.world import World

_EMPTY_NEAT_NODE_LABELS: dict[int, str] = {}


@dataclass(frozen=True, slots=True)
class _SpeciesInspectorRow:
    label: str | None
    value: str
    tone: str = "default"
    marker_color: tuple[int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class _SpeciesInspectorSection:
    title: str
    rows: tuple[_SpeciesInspectorRow, ...]


@dataclass(frozen=True, slots=True)
class _SpeciesTreeLabel:
    species_id: int
    text: str
    bounds: arcade.Rect
    emphasized: bool


class UiRenderer:
    ICON_BUTTON_SIZE = 58.0
    ICON_BUTTON_GAP = 20.0
    RAIL_VERTICAL_PADDING = 32.0
    PANEL_KEYS = ("inspector", "stats", "settings")
    SPECIES_TREE_MIN_ZOOM = 0.1
    SPECIES_TREE_MAX_ZOOM = 2.0
    SPECIES_TREE_ZOOM_FACTOR = 1.2
    SPECIES_TREE_TIME_SCALE = 2.0
    SPECIES_TREE_CONTENT_PADDING = 48.0
    SPECIES_TREE_TIMELINE_WIDTH = 118.0
    SPECIES_TREE_TIMELINE_GAP = 12.0

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.theme = config.theme
        self._text_cache: dict[str, arcade.Text] = {}
        self._texture_cache: dict[str, object | None] = {}
        self._sprite_cache: dict[str, object | None] = {}
        self._control_hitboxes: dict[str, arcade.Rect] = {}
        self._scroll_regions: dict[str, arcade.Rect] = {}
        self._scroll_offsets: dict[str, float] = {}
        self._scroll_limits: dict[str, float] = {}
        self._active_slider = False
        self._panel_open = {
            "inspector": False,
            "stats": False,
            "settings": False,
        }
        self._map_submenu_open = False
        self._panel_bounds: dict[str, arcade.Rect] = {}
        self._active_panel_drag: str | None = None
        self._panel_drag_offset = (0.0, 0.0)
        self._brain_window_open = False
        self._brain_window_bounds: arcade.Rect | None = None
        self._brain_graph_zoom = 1.0
        self._brain_node_bounds: dict[int, arcade.Rect] = {}
        self._brain_selected_node_key: int | None = None
        self._brain_node_inspector_open = True
        self._brain_selection_identity: tuple[int, int] | None = None
        self._species_tree_open = False
        self._species_tree_previous_pause: bool | None = None
        self._species_tree_mouse = (0.0, 0.0)
        self._species_tree_hovered_id: int | None = None
        self._species_tree_selected_id: int | None = None
        self._species_tree_pending_selection_id: int | None = None
        self._species_tree_report: InspectorReport | None = None
        self._species_tree_report_species_id: int | None = None
        self._species_tree_radar_texture: arcade.Texture | None = None
        self._species_tree_radar_species_id: int | None = None
        self._species_tree_radar_future: Future[object] | None = None
        self._species_tree_radar_executor: ThreadPoolExecutor | None = None
        self._species_tree_radar_error: str | None = None
        self._species_tree_horizontal_offset = 0.0
        self._species_tree_vertical_offset = 0.0
        self._species_tree_horizontal_limit = 0.0
        self._species_tree_vertical_limit = 0.0
        self._species_tree_horizontal_offset_min = 0.0
        self._species_tree_horizontal_offset_max = 0.0
        self._species_tree_vertical_offset_min = 0.0
        self._species_tree_vertical_offset_max = 0.0
        self._species_tree_scroll_drag: str | None = None
        self._species_tree_scroll_drag_offset = 0.0
        self._species_tree_canvas_drag = False
        self._species_tree_canvas_drag_started = False
        self._species_tree_canvas_drag_last = (0.0, 0.0)
        self._species_tree_inspector_width: float | None = None
        self._species_tree_inspector_resize_drag = False
        self._species_tree_timeline_bucket_bounds: dict[int, arcade.Rect] = {}
        self._species_tree_node_bounds: dict[int, arcade.Rect] = {}
        self._species_tree_zoom = 1.0
        self._species_tree_fit_mode = True
        self._species_tree_fit_requested = True
        self._species_tree_last_layout: SpeciesTreeLayout | None = None
        self._species_tree_last_canvas: arcade.Rect | None = None
        self._species_tree_layout_manager = TreeLayoutManager(
            horizontal_gap=92.0,
            time_scale=self.SPECIES_TREE_TIME_SCALE,
            padding=self.SPECIES_TREE_CONTENT_PADDING,
        )
        self._species_tree_extinction_times: dict[int, float] = {}
        self._species_tree_cached_layout: SpeciesTreeLayout | None = None
        self._species_tree_visible_slice = TreeViewportSlice((), (), {})
        self._species_tree_focus_latest_pending = False
        self._species_tree_highlight_cache_id: int | None = None
        self._species_tree_highlight_nodes: set[int] = set()
        self._species_tree_highlight_edges: set[tuple[int, int]] = set()
        self._species_tree_neat_label_signature: tuple[
            tuple[int, ...],
            tuple[int, ...],
        ] | None = None
        self._species_tree_neat_labels = _EMPTY_NEAT_NODE_LABELS
        self._species_tree_sync_signature: tuple[
            int,
            int,
            float,
            frozenset[int],
        ] | None = None

    def draw(self, world: World) -> None:
        self._control_hitboxes.clear()
        self._brain_node_bounds.clear()
        self._scroll_regions.clear()
        self._scroll_limits.clear()
        self._draw_icon_rail(world)
        self._draw_floating_panels(world)
        self._draw_brain_window(world)
        self._draw_species_tree_window(world)

    def _sync_species_tree_layout(self, world: World) -> SpeciesTreeLayout:
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

    def _draw_icon_rail(self, world: World) -> None:
        bounds = world.layout.left_sidebar
        self._draw_rounded_rect(
            bounds,
            self.theme.panel_background_alt,
            self.theme.panel_border,
            14,
            1.5,
        )
        self._control_hitboxes["icon_rail"] = bounds

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

    def _draw_floating_panels(self, world: World) -> None:
        if self._panel_open["stats"]:
            self._draw_stats_panel(world)
        if self._panel_open["inspector"]:
            self._draw_inspector_panel(world)
        if self._panel_open["settings"]:
            self._draw_settings_panel(world)

    def _stats_panel_bounds(self, world: World) -> arcade.Rect:
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
        bounds = self._panel_bounds.get(key, default_bounds)
        bounds = self._clamp_panel_bounds(world, bounds)
        self._panel_bounds[key] = bounds
        return bounds

    def _clamp_panel_bounds(self, world: World, bounds: arcade.Rect) -> arcade.Rect:
        margin = float(self.config.layout.outer_padding)
        max_left = max(margin, world.layout.window.width - margin - bounds.width)
        max_bottom = max(margin, world.layout.window.height - margin - bounds.height)
        return arcade.LBWH(
            max(margin, min(max_left, bounds.left)),
            max(margin, min(max_bottom, bounds.bottom)),
            bounds.width,
            bounds.height,
        )

    def _draw_inspector_panel(self, world: World) -> None:
        bounds = self._inspector_panel_bounds(world)
        content = self._draw_floating_panel(bounds, "Inspector", "inspector")
        selected = world.selected_creature

        if selected is None:
            self._draw_rounded_rect(
                content,
                self.theme.card_background,
                self.theme.panel_border,
                8,
                1.0,
            )
            self._draw_text(
                "inspector_empty_title",
                "No creature selected",
                content.left + 18,
                content.top - 38,
                self.theme.text_primary,
                17,
                bold=True,
            )
            self._draw_text(
                "inspector_empty_body",
                "Click a herbivore in the simulation to inspect it.",
                content.left + 18,
                content.top - 68,
                self.theme.text_muted,
                12,
                width=content.width - 36,
                multiline=True,
            )
            return

        card = arcade.LBWH(
            content.left + 8,
            content.bottom + 10,
            content.width - 16,
            content.height - 18,
        )
        self._draw_rounded_rect(
            card,
            self.theme.card_background,
            self.theme.panel_border,
            8,
            1.0,
        )
        self._draw_inspector_page_marker(card)

        inner_viewport = arcade.LBWH(
            card.left + 12,
            card.bottom + 10,
            card.width - 24,
            card.height - 20,
        )
        with self._ui_clip(inner_viewport):
            self._draw_inspector_content(world, inner_viewport)

    def _draw_inspector_page_marker(self, card: arcade.Rect) -> None:
        marker = arcade.LBWH(card.left, card.bottom, 7.0, card.height)
        self._draw_left_rounded_rect_fill(marker, self.theme.accent, 8.0)

    def _draw_left_rounded_rect_fill(
        self,
        bounds: arcade.Rect,
        color: arcade.Color | tuple[int, ...],
        radius: float,
    ) -> None:
        if bounds.width <= 0 or bounds.height <= 0:
            return
        radius = min(radius, bounds.width / 2, bounds.height / 2)
        if radius <= 0:
            arcade.draw_lrbt_rectangle_filled(
                bounds.left,
                bounds.right,
                bounds.bottom,
                bounds.top,
                color,
            )
            return
        arcade.draw_lrbt_rectangle_filled(
            bounds.left + radius,
            bounds.right,
            bounds.bottom,
            bounds.top,
            color,
        )
        arcade.draw_lrbt_rectangle_filled(
            bounds.left,
            bounds.right,
            bounds.bottom + radius,
            bounds.top - radius,
            color,
        )
        arcade.draw_circle_filled(
            bounds.left + radius,
            bounds.bottom + radius,
            radius,
            color,
        )
        arcade.draw_circle_filled(
            bounds.left + radius,
            bounds.top - radius,
            radius,
            color,
        )

    def _draw_inspector_content(self, world: World, viewport: arcade.Rect) -> None:
        selected = world.selected_creature
        if selected is None:
            return

        snapshot = world.sensor_snapshot_for(selected)
        fitness = world.fitness_for(selected)
        genome_id = world.neat_controller.genome_id_for(selected.creature_id)
        vision_cost = world.vision.energy_cost_per_second(selected)
        physical_traits = getattr(selected, "physical_traits", None)
        lineage = getattr(selected, "lineage", None)
        mutation_delta = getattr(lineage, "mutation_delta", None)
        radius = (
            getattr(physical_traits, "radius", None)
            if physical_traits is not None
            else getattr(selected, "radius", 0.0)
        )
        movement_cost_multiplier = (
            getattr(physical_traits, "movement_cost_multiplier", 1.0)
            if physical_traits is not None
            else 1.0
        )
        flocking_traits = getattr(selected, "flocking_traits", None)
        current_action = getattr(selected, "last_action", None)
        herding = float(getattr(current_action, "herding", 0.0))
        parent_id = getattr(lineage, "parent_id", None)
        generation = getattr(lineage, "generation", 0)
        fitness_score = (
            fitness.score(world.config.fitness) if fitness is not None else None
        )
        species_id, species_color = self._selected_species_identity(
            world,
            selected,
        )
        energy_ratio = self._inspector_energy_ratio(world)
        stomach_ratio = max(
            0.0,
            min(1.0, float(getattr(snapshot, "stomach_fullness", 0.0))),
        )
        flock_snapshot = getattr(snapshot, "flock", None)
        effective_flockmate_count = max(
            0.0,
            float(getattr(flock_snapshot, "flockmate_count", 0.0)),
        )
        normalized_flockmate_count = effective_flockmate_count / (
            effective_flockmate_count + 3.0
        )
        padding = 18.0
        section_gap = 18.0
        species_row_height = 28.0 if species_id is not None else 0.0
        total_height = (
            (824.0 if fitness_score is not None else 792.0)
            + species_row_height
        )
        scroll_limit = max(0.0, total_height - viewport.height)
        scroll_offset = max(
            0.0,
            min(scroll_limit, self._scroll_offsets.get("inspector", 0.0)),
        )
        self._scroll_offsets["inspector"] = scroll_offset
        self._scroll_limits["inspector"] = scroll_limit
        self._scroll_regions["inspector"] = viewport

        left = viewport.left + padding
        right = viewport.right - padding
        width = max(0.0, right - left)
        y = viewport.top - 28.0 + scroll_offset

        self._draw_text_in_viewport(
            viewport,
            "inspector_label",
            "SELECTED CREATURE",
            left,
            y,
            self.theme.text_primary,
            10,
            bold=True,
        )
        self._draw_status_chip_in_viewport(
            viewport, arcade.LBWH(right - 50, y - 14, 50, 25), "LIVE"
        )
        y -= 26.0
        self._draw_text_in_viewport(
            viewport,
            "inspector_name",
            self._fit_line(selected.name, width - 64),
            left,
            y,
            self.theme.text_primary,
            17,
            bold=True,
        )

        if species_id is not None:
            y -= 29.0
            marker_x = left + 9.0
            marker_y = y + 5.0
            marker_radius = 8.0
            marker_bounds = arcade.LBWH(
                marker_x - marker_radius,
                marker_y - marker_radius,
                marker_radius * 2.0,
                marker_radius * 2.0,
            )
            if self._rect_intersects(marker_bounds, viewport):
                arcade.draw_circle_filled(
                    marker_x,
                    marker_y,
                    marker_radius,
                    species_color,
                )
                arcade.draw_circle_outline(
                    marker_x,
                    marker_y,
                    marker_radius,
                    self.theme.selected_outline,
                    2.5,
                )
            self._draw_text_in_viewport(
                viewport,
                "inspector_species",
                f"Species #{species_id}",
                left + 25.0,
                y,
                self.theme.text_muted,
                11,
                bold=True,
            )
            y -= 15.0

        y -= 44.0
        self._draw_inspector_section_label(
            viewport, "inspector_energy_section", "ENERGY", left, y
        )
        self._draw_text_in_viewport(
            viewport,
            "inspector_energy_value",
            f"{energy_ratio:.0%}",
            right,
            y,
            self._inspector_energy_color(energy_ratio),
            13,
            bold=True,
            anchor_x="right",
        )
        y -= 18.0
        energy_bar = arcade.LBWH(left, y - 4, width, 8)
        if self._rect_intersects(energy_bar, viewport):
            self._draw_progress_bar(
                energy_bar,
                energy_ratio,
                fill_color=self._inspector_energy_color(energy_ratio),
            )

        y -= 24.0
        self._draw_inspector_section_label(
            viewport, "inspector_stomach_section", "STOMACH", left, y
        )
        self._draw_text_in_viewport(
            viewport,
            "inspector_stomach_value",
            f"{stomach_ratio:.0%}",
            right,
            y,
            (236, 153, 45),
            13,
            bold=True,
            anchor_x="right",
        )
        y -= 18.0
        stomach_bar = arcade.LBWH(left, y - 4, width, 8)
        if self._rect_intersects(stomach_bar, viewport):
            self._draw_progress_bar(
                stomach_bar,
                stomach_ratio,
                fill_color=(236, 153, 45),
            )

        y -= 22.0 + section_gap
        self._draw_inspector_section_label(
            viewport, "inspector_stats_section", "STATS", left, y
        )
        y -= 26.0
        tile_gap = 10.0
        tile_width = (width - tile_gap) / 2.0
        self._draw_inspector_stat_tile_in_viewport(
            viewport,
            "inspector_speed",
            "Speed",
            f"{selected.speed:.1f} px/s",
            arcade.LBWH(left, y - 45.0, tile_width, 46.0),
        )
        self._draw_inspector_stat_tile_in_viewport(
            viewport,
            "inspector_heading",
            "Heading",
            f"{selected.heading:.2f} rad",
            arcade.LBWH(left + tile_width + tile_gap, y - 45.0, tile_width, 46.0),
        )
        y -= 56.0
        self._draw_inspector_stat_tile_in_viewport(
            viewport,
            "inspector_genome",
            "Genome",
            f"#{genome_id}" if genome_id is not None else "None",
            arcade.LBWH(left, y - 45.0, tile_width, 46.0),
        )
        fitness_label = (
            self._format_decimal(fitness_score) if fitness_score is not None else "None"
        )
        self._draw_inspector_stat_tile_in_viewport(
            viewport,
            "inspector_fitness",
            "Fitness",
            fitness_label,
            arcade.LBWH(left + tile_width + tile_gap, y - 45.0, tile_width, 46.0),
        )

        y -= 64.0 + section_gap
        self._draw_inspector_section_label(
            viewport, "inspector_senses_section", "SENSES", left, y
        )
        y -= 24.0
        self._draw_metric_row_in_viewport(
            viewport,
            "inspector_vision_range",
            "Vision",
            f"{selected.vision.range:.0f}px / {selected.vision.angle:.2f} rad",
            left,
            y,
            width,
        )
        y -= 25.0
        self._draw_metric_row_in_viewport(
            viewport,
            "inspector_flockmate_count",
            "Flockmates (eff/net)",
            f"{effective_flockmate_count:.2f} / {normalized_flockmate_count:.2f}",
            left,
            y,
            width,
        )
        y -= 25.0
        self._draw_metric_row_in_viewport(
            viewport,
            "inspector_vision_cost",
            "Cost",
            f"{vision_cost:.3f}/s",
            left,
            y,
            width,
        )
        y -= 25.0
        self._draw_metric_row_in_viewport(
            viewport,
            "inspector_body",
            "Body",
            f"{radius:.1f}px / {movement_cost_multiplier:.2f}x move",
            left,
            y,
            width,
        )
        y -= 25.0
        self._draw_metric_row_in_viewport(
            viewport,
            "inspector_flocking_genes",
            "Flocking genes (inherited)",
            (
                "Unavailable"
                if flocking_traits is None
                else (
                    f"S {flocking_traits.separation_gene:.2f} / "
                    f"A {flocking_traits.alignment_gene:.2f} / "
                    f"C {flocking_traits.cohesion_gene:.2f}"
                )
            ),
            left,
            y,
            width,
        )
        y -= 25.0
        self._draw_metric_row_in_viewport(
            viewport,
            "inspector_herding",
            "Herding (current)",
            f"{herding:.2f}",
            left,
            y,
            width,
        )
        y -= 25.0
        self._draw_metric_row_in_viewport(
            viewport,
            "inspector_collision_avoidance",
            "Collision avoidance",
            "Universal / automatic",
            left,
            y,
            width,
        )
        y -= 25.0
        self._draw_metric_row_in_viewport(
            viewport,
            "inspector_lineage",
            "Lineage",
            f"Parent {parent_id if parent_id is not None else 'None'} / Gen {generation}",
            left,
            y,
            width,
        )
        y -= 25.0
        self._draw_metric_row_in_viewport(
            viewport,
            "inspector_mutations",
            "Mutations",
            self._format_mutation_delta(mutation_delta),
            left,
            y,
            width,
        )
        y -= 25.0
        self._draw_metric_row_in_viewport(
            viewport,
            "inspector_food",
            "Food",
            f"{snapshot.food.visible:.0f} seen / {snapshot.food.density:.2f}",
            left,
            y,
            width,
        )
        y -= 25.0
        self._draw_metric_row_in_viewport(
            viewport,
            "inspector_near",
            "Near",
            f"{snapshot.creatures.visible:.0f} seen / {snapshot.creatures.density:.2f}",
            left,
            y,
            width,
        )

        if fitness_score is not None:
            y -= 25.0
            self._draw_metric_row_in_viewport(
                viewport,
                "inspector_age",
                "Age",
                f"{fitness.age_seconds:.1f}s",
                left,
                y,
                width,
            )

        y -= 56.0
        button_gap = 10
        button_width = (width - button_gap) / 2.0
        brain_button = arcade.LBWH(left, y - 36.0, button_width, 36)
        kill_button = arcade.LBWH(
            brain_button.right + button_gap, y - 36.0, button_width, 36
        )
        self._control_hitboxes.pop("open_brain_window", None)
        self._control_hitboxes.pop("kill_selected_creature", None)
        if self._rect_intersects(brain_button, viewport):
            self._control_hitboxes["open_brain_window"] = brain_button
            self._draw_action_button(
                brain_button,
                "Brain",
                "brain",
                "open_brain_window",
                fill_color=self.theme.accent_soft,
                text_color=self.theme.accent,
            )
        if self._rect_intersects(kill_button, viewport):
            self._control_hitboxes["kill_selected_creature"] = kill_button
            self._draw_action_button(
                kill_button,
                "Kill",
                "kill",
                "kill_selected_creature",
                fill_color=(255, 218, 214),
                text_color=self.theme.selected_outline,
            )
        if scroll_limit > 0.0:
            self._draw_scrollbar(viewport, scroll_offset, scroll_limit)

    def _draw_stats_panel(self, world: World) -> None:
        bounds = self._stats_panel_bounds(world)
        content = self._draw_floating_panel(
            bounds, "Stats", "stats", icon_name="analytics"
        )
        metrics = [
            (
                "POPULATION",
                f"{world.stats.herbivore_count}/{world.config.population.max_creatures}",
                self.theme.text_primary,
            ),
            ("FOOD", str(world.stats.food_count), self.theme.text_primary),
            (
                "BIOMES",
                self._biome_food_summary(world),
                self.theme.text_primary,
            ),
            (
                "BIOMASS",
                self._format_decimal(world.stats.available_biomass),
                self.theme.text_primary,
            ),
            (
                "PLANT PRESS",
                f"{world.stats.plant_spawn_pressure:.0%}",
                self.theme.accent,
            ),
            ("BIRTHS", str(world.rt_neat.stats.births), self.theme.text_primary),
            ("DEATHS", str(world.rt_neat.stats.deaths), self.theme.selected_outline),
            ("ARCHIVED", str(world.archived_fitness_count()), self.theme.text_primary),
            (
                "BEST FITNESS",
                self._format_decimal(world.rt_neat.stats.best_fitness),
                (58, 104, 96),
            ),
            (
                "AVG FITNESS",
                self._format_decimal(world.rt_neat.stats.average_fitness),
                self.theme.accent,
            ),
            (
                "WORST FITNESS",
                self._format_decimal(world.rt_neat.stats.worst_fitness),
                self.theme.selected_outline,
            ),
            (
                "AVG SPEED",
                f"{world.rt_neat.stats.average_speed:.1f}",
                self.theme.text_primary,
            ),
            ("LIVE BRAINS", str(world.live_brain_count()), self.theme.text_primary),
            ("SPEED", f"{world.simulation_speed:.2f}x", self.theme.accent),
            (
                "STATE",
                "Paused" if world.is_paused else "Running",
                self.theme.text_primary,
            ),
        ]
        row_spacing = 28.0
        total_height = len(metrics) * row_spacing
        scroll_limit = max(0.0, total_height - content.height + 10.0)
        scroll_offset = max(
            0.0,
            min(scroll_limit, self._scroll_offsets.get("stats", 0.0)),
        )
        self._scroll_offsets["stats"] = scroll_offset
        self._scroll_limits["stats"] = scroll_limit
        self._scroll_regions["stats"] = content
        for index, (label, value, color) in enumerate(metrics):
            y = content.top - 24.0 - index * row_spacing + scroll_offset
            if y < content.bottom + 4 or y > content.top:
                continue
            self._draw_metric_row(
                f"stats_metric_{index}",
                label,
                value,
                content.left + 10,
                y,
                content.width - 20,
                value_color=color,
            )
        if scroll_limit > 0.0:
            self._draw_scrollbar(content, scroll_offset, scroll_limit)

    def _draw_settings_panel(self, world: World) -> None:
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

    def _draw_selected_creature(self, world: World, bounds: arcade.Rect) -> None:
        selected = world.selected_creature

        lines: list[str]
        if selected is None:
            lines = [
                "No creature selected.",
                "Click inside the environment",
                "to inspect a herbivore.",
            ]
        else:
            snapshot = world.sensor_snapshot_for(selected)
            fitness = world.fitness_for(selected)
            lines = [
                selected.name,
                f"Energy: {selected.energy:.0%}",
                f"Speed: {selected.speed:.1f} px/s",
                f"Heading: {selected.heading:.2f} rad",
                f"Vision: {selected.vision.range:.0f}px / {selected.vision.angle:.2f} rad",
                f"Food: {snapshot.food.visible:.0f} seen / {snapshot.food.density:.2f} density",
                f"Creatures: {snapshot.creatures.visible:.0f} seen / {snapshot.creatures.density:.2f} density",
                (
                    "Compatible flockmates: "
                    f"{float(getattr(getattr(snapshot, 'flock', None), 'flockmate_count', 0.0)):.2f}"
                ),
                f"Vision cost: {world.vision.energy_cost_per_second(selected):.3f}/s",
            ]
            if fitness is not None:
                can_reproduce = world.rt_neat.is_reproduction_eligible(
                    selected,
                    fitness,
                    world.config.population,
                )
                genome_id = world.neat_controller.genome_id_for(selected.creature_id)
                cooldown_remaining = max(
                    0.0,
                    world.config.population.reproduction_cooldown
                    - fitness.seconds_since_reproduction(),
                )
                lines.extend(
                    [
                        f"Genome: {genome_id if genome_id is not None else 'None'}",
                        f"Fitness: {fitness.score(world.config.fitness):.2f}",
                        f"Age: {fitness.age_seconds:.1f}s",
                        f"Food discovered: {fitness.food_discovered}",
                        f"Food eaten: {fitness.food_eaten}",
                        f"Energy gained: {fitness.energy_gained:.3f}",
                        f"Can reproduce: {'Yes' if can_reproduce else 'No'}",
                        f"Cooldown: {cooldown_remaining:.1f}s",
                        f"Offspring: {fitness.offspring_count}",
                    ]
                )

        if selected is None:
            self._draw_scrollable_lines(
                "selected",
                bounds,
                lines,
                line_spacing=22,
                first_line_color=self.theme.text_primary,
                body_color=self.theme.text_muted,
                first_line_bold=True,
            )
            return

        content = self._card_content_bounds(bounds)
        button_height = 32
        button_gap = 10
        button_width = (content.width - button_gap) / 2
        open_button = arcade.LBWH(
            content.left,
            content.bottom,
            button_width,
            button_height,
        )
        kill_button = arcade.LBWH(
            open_button.right + button_gap,
            content.bottom,
            button_width,
            button_height,
        )
        lines_bounds = arcade.LBWH(
            content.left,
            content.bottom + button_height + button_gap,
            content.width,
            max(0.0, content.height - button_height - button_gap),
        )
        self._draw_scrollable_lines_in_bounds(
            "selected",
            lines_bounds,
            lines,
            line_spacing=22,
            first_line_color=self.theme.text_primary,
            body_color=self.theme.text_muted,
            first_line_bold=True,
        )
        self._control_hitboxes["open_brain_window"] = open_button
        self._control_hitboxes["kill_selected_creature"] = kill_button
        self._draw_button(open_button, "Open Brain", "open_brain_window")
        self._draw_button(kill_button, "Kill", "kill_selected_creature")

    def _draw_selected_brain(self, world: World, bounds: arcade.Rect) -> None:
        selected = world.selected_creature
        content = self._card_content_bounds(bounds)
        if selected is None:
            return

        brain = world.neat_controller.brain_for(selected.creature_id)
        if brain is None:
            self._draw_text(
                "selected_brain_empty",
                "No brain assigned.",
                content.left,
                content.top - 12,
                self.theme.text_muted,
                12,
            )
            return

        input_keys = list(world.neat_controller.config.genome_config.input_keys)
        output_keys = list(world.neat_controller.config.genome_config.output_keys)
        sensor_usage = brain.sensor_usage(input_keys, output_keys)
        hidden_keys = sorted(
            key for key in brain.genome.nodes if key not in output_keys
        )
        graph_bounds = arcade.LBWH(
            content.left,
            content.bottom + 82,
            content.width,
            max(96.0, content.height - 82),
        )
        details_bounds = arcade.LBWH(
            content.left,
            content.bottom,
            content.width,
            70,
        )

        input_positions = self._node_column_positions(
            input_keys,
            graph_bounds.left + 18,
            graph_bounds.bottom + 18,
            graph_bounds.top - 22,
        )
        output_positions = self._node_column_positions(
            output_keys,
            graph_bounds.right - 18,
            graph_bounds.bottom + 24,
            graph_bounds.top - 30,
        )
        hidden_positions = self._node_column_positions(
            hidden_keys,
            graph_bounds.center_x,
            graph_bounds.bottom + 22,
            graph_bounds.top - 26,
        )
        positions = {**input_positions, **hidden_positions, **output_positions}

        for connection in brain.genome.connections.values():
            if not connection.enabled:
                continue
            start = positions.get(connection.key[0])
            end = positions.get(connection.key[1])
            if start is None or end is None:
                continue
            color = (
                self.theme.accent
                if connection.weight >= 0.0
                else self.theme.selected_outline
            )
            width = max(1.0, min(4.0, abs(connection.weight) * 0.7))
            arcade.draw_line(start[0], start[1], end[0], end[1], color, width)

        for index, key in enumerate(input_keys):
            position = positions[key]
            value = brain.last_inputs[index] if index < len(brain.last_inputs) else 0.0
            usage = sensor_usage[index]
            self._draw_brain_node(
                position,
                self._brain_activity_color(value),
                self.theme.accent if usage.has_enabled_path else self.theme.panel_border,
                radius=4.0 + min(1.0, abs(value)) * 3.0,
            )
            label = (
                SENSOR_INPUT_NAMES[index]
                if index < len(SENSOR_INPUT_NAMES)
                else str(key)
            )
            self._draw_brain_node_label(
                f"brain_input_{index}",
                (
                    f"{'on' if usage.has_enabled_path else 'off'} "
                    f"{self._short_brain_label(label)} {value:.2f}"
                ),
                position,
                graph_bounds,
                side="right",
            )

        for key in hidden_keys:
            self._draw_brain_node(
                positions[key], self.theme.panel_background, self.theme.panel_border
            )

        for index, key in enumerate(output_keys):
            position = positions[key]
            value = (
                brain.last_outputs[index] if index < len(brain.last_outputs) else 0.0
            )
            self._draw_brain_node(
                position,
                self._brain_activity_color(value),
                self.theme.herbivore_outline,
                radius=4.0 + min(1.0, abs(value)) * 3.0,
            )
            label = (
                ACTION_OUTPUT_NAMES[index]
                if index < len(ACTION_OUTPUT_NAMES)
                else str(key)
            )
            self._draw_brain_node_label(
                f"brain_output_{index}",
                f"{self._short_brain_label(label)} {value:.2f}",
                position,
                graph_bounds,
                side="left",
            )

        action = brain.last_action
        action_label = (
            f"acc {action.accelerate:.2f} rot {action.rotate:.2f}"
            if action is not None
            else "waiting"
        )
        enabled_connections = sum(
            1 for connection in brain.genome.connections.values() if connection.enabled
        )
        biome_path_count = sum(
            usage.has_enabled_path for usage in sensor_usage[17:21]
        )
        detail_lines = [
            f"Genome: {brain.genome_id}",
            f"Signed action: {action_label}",
            f"Speed: {selected.speed:.1f} px/s",
            self._brain_input_readout(brain.last_inputs),
            self._brain_output_readout(brain.last_outputs),
            f"Nodes: {len(brain.genome.nodes)}",
            f"Connections: {enabled_connections}/{len(brain.genome.connections)} enabled",
            f"Biome sensing paths: {biome_path_count}/4",
            f"Fitness: {self._selected_fitness_label(world, selected)}",
        ]
        self._draw_scrollable_lines_in_bounds(
            "brain_details",
            details_bounds,
            detail_lines,
            line_spacing=17,
            first_line_color=self.theme.text_primary,
            body_color=self.theme.text_muted,
            first_line_bold=True,
        )

    def _draw_brain_window(self, world: World) -> None:
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
            desired_width = max(280.0, min(360.0, bounds.width * 0.24))
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
            self._draw_brain_node_inspector(brain, layout, inspector_bounds)
        if brain is not None:
            self._draw_brain_footer(world, selected, brain, footer_bounds)

    def _draw_brain_graph(
        self,
        world: World,
        bounds: arcade.Rect,
    ) -> BrainGraphLayout | None:
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
        layout = build_brain_graph_layout(
            brain.genome,
            input_keys,
            output_keys,
            layout_bounds,
            SENSOR_INPUT_NAMES,
            ACTION_OUTPUT_NAMES,
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
            highlighted_path_through_node(layout, selected_key)
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
                self._draw_curve(points, color, width)
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
            self._draw_curve(points, color, width)
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

    def _draw_brain_node_inspector(
        self,
        brain: object | None,
        layout: BrainGraphLayout | None,
        bounds: arcade.Rect,
    ) -> None:
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
        return (
            f"{relation}: {self._brain_node_display_name(source)} "
            f"[ID {source.key}] -> {self._brain_node_display_name(target)} "
            f"[ID {target.key}] | "
            f"{self._brain_connection_inspector_details(connection)}"
        )

    def _brain_connection_inspector_details(self, connection: object) -> str:
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

    def _sync_brain_graph_selection(
        self,
        selected: object,
        brain: object,
        layout: BrainGraphLayout,
    ) -> None:
        identity = (int(selected.creature_id), int(brain.genome_id))
        if identity != self._brain_selection_identity:
            self._brain_selection_identity = identity
            self._brain_selected_node_key = None
            self._scroll_offsets["brain_node_inspector"] = 0.0
        elif (
            self._brain_selected_node_key is not None
            and self._brain_selected_node_key not in layout.nodes
        ):
            self._brain_selected_node_key = None
            self._scroll_offsets["brain_node_inspector"] = 0.0

    def _close_brain_window(self) -> None:
        self._brain_window_open = False
        self._brain_selected_node_key = None
        self._brain_selection_identity = None
        self._brain_node_bounds.clear()
        self._scroll_offsets["brain_node_inspector"] = 0.0

    def _brain_node_at(self, x: float, y: float) -> int | None:
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

    def _brain_node_display_name(self, node: BrainGraphNode) -> str:
        if node.kind == BrainNodeKind.HIDDEN:
            return f"Hidden {node.key}"
        return node.label

    def _brain_node_kind_color(
        self,
        kind: BrainNodeKind,
    ) -> arcade.Color | tuple[int, ...]:
        if kind == BrainNodeKind.INPUT:
            return (39, 110, 241)
        if kind == BrainNodeKind.OUTPUT:
            return (31, 168, 82)
        return (130, 54, 224)

    def _brain_color_alpha(
        self,
        color: arcade.Color | tuple[int, ...],
        alpha: int,
    ) -> tuple[int, int, int, int]:
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
        background_components = tuple(background)
        foreground_components = tuple(foreground)
        mix = max(0.0, min(1.0, amount))
        return tuple(
            int(background_components[index] * (1.0 - mix) + foreground_components[index] * mix)
            for index in range(3)
        )

    def _ensure_brain_window_bounds(self, world: World) -> None:
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
        return (
            bounds.center_x + (position[0] - bounds.center_x) * self._brain_graph_zoom,
            bounds.center_y + (position[1] - bounds.center_y) * self._brain_graph_zoom,
        )

    @property
    def species_tree_open(self) -> bool:
        return self._species_tree_open

    def open_species_tree(self, world: World) -> None:
        if self._species_tree_open:
            return
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
            world.is_paused = previous_pause

    def close(self) -> None:
        """Release asynchronous UI resources owned by this renderer."""
        self._clear_species_radar_state()
        executor = self._species_tree_radar_executor
        self._species_tree_radar_executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def handle_key_press(self, world: World, symbol: int, modifiers: int) -> bool:
        del world, symbol, modifiers
        return self._species_tree_open

    def handle_mouse_motion(
        self,
        world: World,
        x: float,
        y: float,
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> bool:
        del world, dx, dy
        self._species_tree_mouse = (x, y)
        if not self._species_tree_open:
            self._species_tree_hovered_id = None
            return False
        self._species_tree_hovered_id = self._species_tree_node_at(x, y)
        return True

    def _draw_species_tree_window(self, world: World) -> None:
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

    def _species_tree_inspector_width_limits(
        self,
        content: arcade.Rect,
    ) -> tuple[float, float]:
        max_width = max(0.0, content.width * 2.0 / 3.0)
        min_width = min(300.0, max_width)
        return min_width, max_width

    def _species_tree_inspector_default_width(
        self,
        content: arcade.Rect,
    ) -> float:
        min_width, max_width = self._species_tree_inspector_width_limits(
            content
        )
        preferred = max(340.0, content.width * 0.38)
        return max(min_width, min(max_width, preferred))

    def _species_tree_inspector_clamped_width(
        self,
        content: arcade.Rect,
    ) -> float:
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

    def _draw_species_tree_legend(self, bounds: arcade.Rect) -> None:
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
    def _format_species_tree_time(value: float) -> str:
        seconds = max(0, int(round(value)))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _valid_species_tree_time(value: object) -> float | None:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not isfinite(parsed) or parsed < 0.0:
            return None
        return parsed

    @staticmethod
    def _species_tree_timeline_y(
        value: float,
        start: float,
        end: float,
        ruler: arcade.Rect,
    ) -> float:
        if end <= start:
            return ruler.top
        ratio = max(0.0, min(1.0, (value - start) / (end - start)))
        return ruler.top - ratio * ruler.height

    def _species_tree_visible_time_range(
        self,
        layout: SpeciesTreeLayout,
        canvas: arcade.Rect,
    ) -> tuple[float, float]:
        _, vertical_inset = self._species_tree_content_insets(
            layout,
            canvas,
            self._species_tree_zoom,
        )

        def time_at(screen_y: float) -> float:
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

    def _jump_species_tree_from_timeline(
        self,
        timeline: arcade.Rect,
        y: float,
    ) -> None:
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

    def _update_species_tree_zoom_and_limits(
        self,
        layout: SpeciesTreeLayout,
        canvas: arcade.Rect,
    ) -> None:
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
        return (
            max(0.0, (canvas.width - layout.content_width * zoom) * 0.5),
            max(0.0, (canvas.height - layout.content_height * zoom) * 0.5),
        )

    def _draw_species_tree_zoom_controls(self, bounds: arcade.Rect) -> None:
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

    def _species_tree_screen_positions(
        self,
        layout: SpeciesTreeLayout,
        canvas: arcade.Rect,
        species_ids: tuple[int, ...] | None = None,
    ) -> dict[int, tuple[float, float]]:
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
        color = record.founder_color or self.theme.herbivore_fill
        return int(color[0]), int(color[1]), int(color[2])

    def _species_tree_refined_color(
        self,
        record: SpeciesRecord,
        strength: float,
        *,
        alpha: int | None = None,
    ) -> tuple[int, ...]:
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
        opacity = max(0.0, min(1.0, alpha / 255.0))

        def linear(component: float) -> float:
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

    def _draw_species_tree_path(
        self,
        points: SpeciesTreeRoute,
        color: arcade.Color | tuple[int, ...],
        width: float,
        *,
        dashed: bool = False,
    ) -> None:
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

    def _draw_species_tree_edges(
        self,
        records: dict[int, SpeciesRecord],
        layout: SpeciesTreeLayout,
        edges: tuple[tuple[int, int], ...],
        routes: dict[tuple[int, int], SpeciesTreeRoute],
        highlighted_edges: set[tuple[int, int]],
    ) -> None:
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

    def _species_tree_node_visual_radius(
        self,
        layout: SpeciesTreeLayout,
        species_id: int,
    ) -> float:
        descendants = max(0, layout.descendant_counts.get(species_id, 0))
        base = 6.25 + min(2.25, log1p(descendants) * 0.5)
        return max(4.25, min(11.0, base * self._species_tree_zoom))

    def _species_tree_mutation_intensity(self, record: SpeciesRecord) -> float:
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
        if intensity <= 0.0:
            return 0
        return max(1, min(3, ceil(intensity * 3.0)))

    def _draw_species_tree_mutation_ticks(
        self,
        position: tuple[float, float],
        radius: float,
        intensity: float,
    ) -> None:
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
        selected_id = self._species_tree_selected_id
        hovered_id = self._species_tree_hovered_id
        candidates: list[tuple[int, str, bool]] = []

        def add(species_id: int | None, full: bool = False) -> None:
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
        canvas = self._control_hitboxes.get("species_tree_canvas")
        if canvas is None or not self._contains_bounds(canvas, x, y):
            return None
        for species_id in sorted(self._species_tree_node_bounds, reverse=True):
            bounds = self._species_tree_node_bounds[species_id]
            radius = bounds.width * 0.5
            if (x - bounds.center_x) ** 2 + (y - bounds.center_y) ** 2 <= radius**2:
                return species_id
        return None

    def _clamp_species_tree_offsets(self) -> None:
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

    def _draw_species_tree_scrollbars(self, canvas: arcade.Rect) -> None:
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

    def _ensure_species_inspector_report(
        self,
        world: World,
        records: dict[int, SpeciesRecord],
    ) -> None:
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

    def _radar_executor(self) -> ThreadPoolExecutor:
        executor = self._species_tree_radar_executor
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="species-radar",
            )
            self._species_tree_radar_executor = executor
        return executor

    def _clear_species_radar_state(self) -> None:
        future = self._species_tree_radar_future
        if future is not None:
            future.cancel()
        self._species_tree_radar_future = None
        self._species_tree_radar_texture = None
        self._species_tree_radar_species_id = None
        self._species_tree_radar_error = None

    def _consume_species_radar_result(self) -> None:
        future = self._species_tree_radar_future
        if future is None or not future.done():
            return
        self._species_tree_radar_future = None
        if (
            not self._species_tree_open
            or self._species_tree_radar_species_id
            != self._species_tree_selected_id
        ):
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
    def _species_representative_genome(representative: object) -> object | None:
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
        sections = self._species_inspector_sections(report, record)
        radar_size = (
            min(220.0, max(0.0, viewport.width - 24.0))
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
        if self._species_tree_radar_species_id is None:
            return viewport
        chart_size = min(
            300.0,
            viewport.width,
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

    def _draw_species_radar_chart_in_bounds(
        self,
        chart_bounds: arcade.Rect,
    ) -> None:
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
        if value is None or value == 0.0:
            return "default"
        return "positive" if value > 0.0 else "negative"

    @staticmethod
    def _pressure_label(value: float) -> str:
        if value < 1.0 / 3.0:
            return "Low"
        if value < 2.0 / 3.0:
            return "Moderate"
        return "High"

    @staticmethod
    def _format_percent_change(value: float | None) -> str:
        return "change unavailable" if value is None else f"{value:+.2f}% vs parent"

    def _draw_species_tree_tooltip(
        self,
        window_bounds: arcade.Rect,
        record: SpeciesRecord,
        parent_record: SpeciesRecord | None = None,
    ) -> None:
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

    def _species_tree_neat_node_labels(self, world: World) -> dict[int, str]:
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
        value = None if snapshot is None else getattr(snapshot, attribute, None)
        return self._format_optional_number(value, signed=signed)

    def _format_optional_number(
        self,
        value: object,
        digits: int = 3,
        *,
        signed: bool = False,
    ) -> str:
        if value is None:
            return "Unavailable"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "Unavailable"
        return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"

    def _brain_edge_color(self, weight: float) -> arcade.Color | tuple[int, ...]:
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
        x, y = position
        points = [
            (x + 8.0, y + 2.0),
            (x + 32.0, y + 26.0),
            (x + 22.0, y - 24.0),
            (x + 8.0, y - 2.0),
        ]
        final_curve: list[tuple[float, float]] = []
        for start, control, end in (
            (points[0], points[1], points[2]),
            (points[2], points[3], points[0]),
        ):
            curve = self._quadratic_bezier_points(start, control, end, steps=10)
            if dashed:
                self._draw_dashed_curve(curve, color, width)
            else:
                self._draw_curve(curve, color, width)
            final_curve = curve
        if draw_arrow:
            self._draw_brain_arrowhead(final_curve, color, width)

    def _draw_curve(
        self,
        points: list[tuple[float, float]],
        color: arcade.Color | tuple[int, ...],
        width: float,
    ) -> None:
        for start, end in zip(points, points[1:]):
            arcade.draw_line(start[0], start[1], end[0], end[1], color, width)

    def _draw_dashed_curve(
        self,
        points: list[tuple[float, float]],
        color: arcade.Color | tuple[int, ...],
        width: float,
        *,
        dash_length: float = 6.0,
        gap_length: float = 4.0,
    ) -> None:
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
    ) -> list[tuple[float, float]]:
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
        return points

    def _cubic_bezier_points(
        self,
        start: tuple[float, float],
        first_control: tuple[float, float],
        second_control: tuple[float, float],
        end: tuple[float, float],
        *,
        steps: int = 28,
    ) -> list[tuple[float, float]]:
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
        return points

    def _draw_brain_arrowhead(
        self,
        points: list[tuple[float, float]],
        color: arcade.Color | tuple[int, ...],
        width: float,
    ) -> None:
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

    def _draw_environment_stats(self, world: World, bounds: arcade.Rect) -> None:
        lines = [
            f"Population: {world.stats.herbivore_count}/{world.config.population.max_creatures}",
            f"Food nodes: {world.stats.food_count}",
            f"Biome food: {self._biome_food_summary(world)}",
            f"Biome area: {self._biome_area_summary(world)}",
            f"Biomass: {world.stats.available_biomass:.1f} available",
            f"Plant pressure: {world.stats.plant_spawn_pressure:.0%}",
            f"Plants: {world.stats.plant_energy:.1f} energy",
            f"Creatures: {world.stats.creature_energy:.1f} energy",
            f"Elapsed time: {world.elapsed_time:0.1f}s",
            "State: Paused" if world.is_paused else "State: Running",
            f"Controller: {'NEAT' if world.use_neat_brains else 'Baseline'}",
            f"Simulation speed: {world.simulation_speed:.2f}x",
            f"Zoom: {world.environment_zoom:.2f}x",
            f"Births: {world.rt_neat.stats.births}",
            f"Deaths: {world.rt_neat.stats.deaths}",
            f"Births/min: {world.rt_neat.stats.births_per_minute:.2f}",
            f"Deaths/min: {world.rt_neat.stats.deaths_per_minute:.2f}",
            f"Normal replacements: {world.rt_neat.stats.normal_replacements}",
            f"Extinction replacements: {world.rt_neat.stats.extinction_replacements}",
            f"Avg lifespan: {world.rt_neat.stats.average_lifespan_at_death:.1f}s",
            f"Avg speed: {world.rt_neat.stats.average_speed:.1f} px/s",
            f"Avg distance: {world.rt_neat.stats.average_distance_traveled:.0f}px",
            f"Brain size: {world.rt_neat.stats.average_brain_nodes:.1f} nodes / "
            f"{world.rt_neat.stats.average_brain_enabled_connections:.1f}"
            f"/{world.rt_neat.stats.average_brain_connections:.1f} conns",
            f"Live brains: {world.live_brain_count()}",
            f"Archived: {world.archived_fitness_count()}",
            f"Best fitness: {world.rt_neat.stats.best_fitness:.2f}",
            f"Avg fitness: {world.rt_neat.stats.average_fitness:.2f}",
            f"Worst fitness: {world.rt_neat.stats.worst_fitness:.2f}",
            f"Eligible parents: {world.rt_neat.stats.eligible_parent_count}",
            world.stats.generation_label,
        ]
        self._draw_scrollable_lines(
            "stats",
            bounds,
            lines,
            line_spacing=24,
            first_line_color=self.theme.text_muted,
            body_color=self.theme.text_muted,
        )

    def _draw_controls(self, world: World, bounds: arcade.Rect) -> None:
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

    def handle_mouse_press(self, world: World, x: float, y: float) -> bool:
        if self._species_tree_open:
            if self._contains_hitbox("species_tree_close", x, y):
                self.close_species_tree(world)
                return True
            if self._contains_hitbox("species_tree_zoom_out", x, y):
                self._adjust_species_tree_zoom(
                    1.0 / self.SPECIES_TREE_ZOOM_FACTOR
                )
                return True
            if self._contains_hitbox("species_tree_zoom_in", x, y):
                self._adjust_species_tree_zoom(self.SPECIES_TREE_ZOOM_FACTOR)
                return True
            if self._contains_hitbox("species_tree_zoom_fit", x, y):
                self._activate_species_tree_fit()
                return True
            if self._contains_hitbox("species_tree_parent_button", x, y):
                records = getattr(world, "species_history", {})
                selected = records.get(self._species_tree_selected_id)
                parent_id = (
                    None
                    if selected is None
                    else selected.parent_species_id
                )
                if parent_id is not None and parent_id in records:
                    self._select_species_tree_species(parent_id, focus=True)
                return True
            if self._contains_hitbox("species_tree_inspector_resize", x, y):
                self._species_tree_inspector_resize_drag = True
                return True
            if self._contains_hitbox("species_tree_horizontal_thumb", x, y):
                thumb = self._control_hitboxes["species_tree_horizontal_thumb"]
                self._species_tree_scroll_drag = "horizontal"
                self._species_tree_scroll_drag_offset = x - thumb.left
                return True
            if self._contains_hitbox("species_tree_vertical_thumb", x, y):
                thumb = self._control_hitboxes["species_tree_vertical_thumb"]
                self._species_tree_scroll_drag = "vertical"
                self._species_tree_scroll_drag_offset = thumb.top - y
                return True
            if self._contains_hitbox("species_tree_horizontal_track", x, y):
                self._set_species_tree_scroll_from_pointer("horizontal", x, y)
                return True
            if self._contains_hitbox("species_tree_vertical_track", x, y):
                self._set_species_tree_scroll_from_pointer("vertical", x, y)
                return True
            for bucket_id in sorted(
                self._species_tree_timeline_bucket_bounds,
                reverse=True,
            ):
                marker = self._species_tree_timeline_bucket_bounds[bucket_id]
                if self._contains_bounds(marker, x, y):
                    bucket_seconds = (
                        self._species_tree_layout_manager.bucket_seconds
                    )
                    self._jump_species_tree_to_time(
                        (bucket_id + 0.5) * bucket_seconds
                    )
                    return True
            if self._contains_hitbox("species_tree_timeline", x, y):
                timeline = self._control_hitboxes["species_tree_timeline"]
                self._jump_species_tree_from_timeline(timeline, y)
                return True
            if self._contains_hitbox("species_tree_canvas", x, y):
                self._species_tree_pending_selection_id = (
                    self._species_tree_node_at(x, y)
                )
                self._species_tree_canvas_drag = True
                self._species_tree_canvas_drag_started = False
                self._species_tree_canvas_drag_last = (x, y)
                return True
            return True

        if (
            self._brain_window_open
            and self._brain_window_bounds is not None
            and self._contains_bounds(self._brain_window_bounds, x, y)
        ):
            if self._contains_hitbox("brain_window_close", x, y):
                self._close_brain_window()
                return True
            if self._contains_hitbox("brain_node_inspector_toggle", x, y):
                self._brain_node_inspector_open = not self._brain_node_inspector_open
                return True
            node_key = self._brain_node_at(x, y)
            if node_key is not None:
                self._brain_selected_node_key = node_key
                self._scroll_offsets["brain_node_inspector"] = 0.0
                return True
            if self._contains_hitbox("brain_window_graph", x, y):
                self._brain_selected_node_key = None
                self._scroll_offsets["brain_node_inspector"] = 0.0
                return True
            return True

        if self._contains_hitbox("open_map_submenu", x, y):
            self._map_submenu_open = not self._map_submenu_open
            return True
        if self._map_submenu_open:
            if self._contains_hitbox("map_layer_biome", x, y):
                world.select_environment_map("biome")
                self._map_submenu_open = False
                return True
            if self._contains_hitbox("map_layer_pheromones", x, y):
                world.select_environment_map("pheromones")
                self._map_submenu_open = False
                return True
            if self._contains_hitbox("map_submenu", x, y):
                return True
            self._map_submenu_open = False

        for panel_name in self.PANEL_KEYS:
            if self._contains_hitbox(f"{panel_name}_close", x, y):
                self._panel_open[panel_name] = False
                self._active_panel_drag = None
                return True
        if self._contains_hitbox("panel_toggle_inspector", x, y):
            self._panel_open["inspector"] = not self._panel_open["inspector"]
            return True
        if self._contains_hitbox("panel_toggle_stats", x, y):
            self._panel_open["stats"] = not self._panel_open["stats"]
            return True
        if self._contains_hitbox("panel_toggle_settings", x, y):
            self._panel_open["settings"] = not self._panel_open["settings"]
            return True
        if self._contains_hitbox("save_simulation", x, y):
            world.save_now()
            return True
        if self._contains_hitbox("open_species_tree", x, y):
            self.open_species_tree(world)
            return True
        if self._contains_hitbox("open_brain_window", x, y):
            if world.selected_creature is not None:
                self._brain_window_open = True
                self._ensure_brain_window_bounds(world)
            return True
        if self._contains_hitbox("kill_selected_creature", x, y):
            if world.kill_selected_creature():
                self._close_brain_window()
            return True
        if self._contains_hitbox("pause", x, y):
            world.toggle_pause()
            return True
        if self._contains_hitbox("reset_speed", x, y):
            world.reset_simulation_speed()
            return True
        if self._contains_hitbox("speed_down", x, y):
            world.decrease_simulation_speed()
            return True
        if self._contains_hitbox("speed_up", x, y):
            world.increase_simulation_speed()
            return True
        if self._contains_hitbox("speed_min", x, y):
            world.set_simulation_speed(world.MIN_SIMULATION_SPEED)
            return True
        if self._contains_hitbox("speed_max", x, y):
            world.set_simulation_speed(world.MAX_SIMULATION_SPEED)
            return True
        if self._contains_hitbox("speed_slider", x, y):
            self._active_slider = True
            self._set_speed_from_slider(world, x)
            return True
        for panel_name in self.PANEL_KEYS:
            if self._contains_hitbox(f"{panel_name}_drag", x, y):
                bounds = self._panel_bounds.get(panel_name)
                if bounds is not None:
                    self._active_panel_drag = panel_name
                    self._panel_drag_offset = (x - bounds.left, y - bounds.bottom)
                    return True
        for key in (
            "icon_rail",
            "inspector_panel",
            "inspector_body",
            "stats_panel",
            "stats_body",
            "settings_panel",
            "settings_body",
        ):
            if self._contains_hitbox(key, x, y):
                return True
        return False

    def handle_mouse_drag(self, world: World, x: float, y: float) -> bool:
        if self._species_tree_open:
            if self._species_tree_inspector_resize_drag:
                content = self._control_hitboxes.get("species_tree_body")
                if content is not None:
                    min_width, max_width = (
                        self._species_tree_inspector_width_limits(content)
                    )
                    requested_width = content.right - x
                    self._species_tree_inspector_width = max(
                        min_width,
                        min(max_width, requested_width),
                    )
                return True
            if self._species_tree_scroll_drag is not None:
                self._set_species_tree_scroll_from_pointer(
                    self._species_tree_scroll_drag,
                    x,
                    y,
                    preserve_grab_offset=True,
                )
            elif self._species_tree_canvas_drag:
                previous_x, previous_y = self._species_tree_canvas_drag_last
                if not self._species_tree_canvas_drag_started:
                    self._species_tree_fit_mode = False
                    self._species_tree_fit_requested = False
                    self._species_tree_canvas_drag_started = True
                    self._species_tree_pending_selection_id = None
                self._species_tree_horizontal_offset -= x - previous_x
                self._species_tree_vertical_offset += y - previous_y
                self._species_tree_canvas_drag_last = (x, y)
                self._clamp_species_tree_offsets()
            return True
        if self._active_panel_drag is not None:
            bounds = self._panel_bounds.get(self._active_panel_drag)
            if bounds is None:
                return False
            offset_x, offset_y = self._panel_drag_offset
            self._panel_bounds[self._active_panel_drag] = self._clamp_panel_bounds(
                world,
                arcade.LBWH(
                    x - offset_x,
                    y - offset_y,
                    bounds.width,
                    bounds.height,
                ),
            )
            return True
        if not self._active_slider:
            return False
        self._set_speed_from_slider(world, x)
        return True

    def handle_mouse_release(self) -> None:
        self._active_slider = False
        self._active_panel_drag = None
        self._species_tree_scroll_drag = None
        self._species_tree_inspector_resize_drag = False
        if (
            self._species_tree_canvas_drag
            and not self._species_tree_canvas_drag_started
            and self._species_tree_pending_selection_id is not None
        ):
            self._select_species_tree_species(
                self._species_tree_pending_selection_id,
            )
        self._species_tree_pending_selection_id = None
        self._species_tree_canvas_drag = False
        self._species_tree_canvas_drag_started = False

    def handle_mouse_scroll(
        self,
        x: float,
        y: float,
        scroll_y: float,
        scroll_x: float = 0.0,
        command_down: bool = False,
    ) -> bool:
        if self._species_tree_open:
            inspector_region = self._scroll_regions.get(
                "species_tree_inspector"
            )
            if (
                inspector_region is not None
                and self._contains_bounds(inspector_region, x, y)
            ):
                limit = self._scroll_limits.get(
                    "species_tree_inspector",
                    0.0,
                )
                current = self._scroll_offsets.get(
                    "species_tree_inspector",
                    0.0,
                )
                self._scroll_offsets["species_tree_inspector"] = max(
                    0.0,
                    min(limit, current - scroll_y * 24.0),
                )
                return True
            if self._contains_hitbox("species_tree_canvas", x, y):
                if command_down and scroll_y != 0.0:
                    zoom_steps = max(-8.0, min(8.0, scroll_y))
                    self._adjust_species_tree_zoom(
                        self.SPECIES_TREE_ZOOM_FACTOR**zoom_steps,
                        anchor=(x, y),
                    )
                else:
                    if (
                        scroll_x != 0.0
                        and self._species_tree_horizontal_limit > 0.0
                    ):
                        self._species_tree_horizontal_offset -= scroll_x * 36.0
                    if (
                        scroll_y != 0.0
                        and self._species_tree_vertical_limit > 0.0
                    ):
                        self._species_tree_vertical_offset -= scroll_y * 36.0
                    self._clamp_species_tree_offsets()
            return True
        if (
            self._brain_window_open
            and self._brain_window_bounds is not None
            and self._contains_bounds(self._brain_window_bounds, x, y)
        ):
            inspector_region = self._scroll_regions.get("brain_node_inspector")
            if (
                inspector_region is not None
                and self._contains_bounds(inspector_region, x, y)
            ):
                limit = self._scroll_limits.get("brain_node_inspector", 0.0)
                current = self._scroll_offsets.get("brain_node_inspector", 0.0)
                self._scroll_offsets["brain_node_inspector"] = max(
                    0.0,
                    min(limit, current - scroll_y * 24.0),
                )
            return True

        for key, bounds in self._scroll_regions.items():
            if not (
                bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top
            ):
                continue

            limit = self._scroll_limits.get(key, 0.0)
            current = self._scroll_offsets.get(key, 0.0)
            self._scroll_offsets[key] = max(
                0.0,
                min(limit, current - scroll_y * 24.0),
            )
            return True

        return False

    def _set_species_tree_scroll_from_pointer(
        self,
        axis: str,
        x: float,
        y: float,
        *,
        preserve_grab_offset: bool = False,
    ) -> None:
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

    def _draw_panel(
        self,
        bounds: arcade.Rect,
        fill_color: arcade.Color | tuple[int, ...] | None = None,
    ) -> None:
        self._draw_rounded_rect(
            bounds,
            fill_color or self.theme.panel_background,
            self.theme.panel_border,
            self.config.layout.panel_radius,
            2,
        )

    @contextmanager
    def _ui_clip(self, bounds: arcade.Rect):
        try:
            arcade.get_window()
        except (AttributeError, RuntimeError):
            yield
            return

        try:
            from pyglet import gl
        except ImportError:
            yield
            return

        scale_x, scale_y = self._framebuffer_scale()
        x = round(bounds.left * scale_x)
        y = round(bounds.bottom * scale_y)
        width = round(bounds.width * scale_x)
        height = round(bounds.height * scale_y)
        previous_box = (gl.GLint * 4)()
        was_enabled = bool(gl.glIsEnabled(gl.GL_SCISSOR_TEST))
        gl.glGetIntegerv(gl.GL_SCISSOR_BOX, previous_box)

        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(x, y, width, height)
        try:
            yield
        finally:
            gl.glScissor(
                previous_box[0],
                previous_box[1],
                previous_box[2],
                previous_box[3],
            )
            if not was_enabled:
                gl.glDisable(gl.GL_SCISSOR_TEST)

    def _framebuffer_scale(self) -> tuple[float, float]:
        try:
            window = arcade.get_window()
            window_width, window_height = window.get_size()
            framebuffer_width, framebuffer_height = window.get_framebuffer_size()
        except (AttributeError, RuntimeError):
            return 1.0, 1.0

        if window_width <= 0 or window_height <= 0:
            return 1.0, 1.0
        return framebuffer_width / window_width, framebuffer_height / window_height

    def _draw_floating_panel(
        self,
        bounds: arcade.Rect,
        title: str,
        key: str,
        *,
        icon_name: str | None = None,
        show_close: bool = True,
        body_top_padding: float = 24.0,
        panel_fill: arcade.Color | tuple[int, ...] | None = None,
        title_icon_size: float = 20.0,
    ) -> arcade.Rect:
        self._draw_rounded_rect(
            bounds,
            panel_fill or self.theme.panel_background_alt,
            self.theme.panel_border,
            14,
            1.5,
        )
        self._control_hitboxes[f"{key}_panel"] = bounds

        header_height = 58.0 if title else 0.0
        if header_height > 0.0:
            header = arcade.LBWH(
                bounds.left + 1.5,
                bounds.top - header_height,
                bounds.width - 3.0,
                header_height - 1.5,
            )
            self._draw_rounded_rect_fill(
                header,
                self.theme.panel_background,
                max(0.0, 14.0 - 1.5),
            )
            title_x = bounds.left + 28.0
            if icon_name is not None:
                icon_bounds = arcade.LBWH(
                    bounds.left + 26.0,
                    bounds.top - 28.0 - title_icon_size / 2.0,
                    title_icon_size,
                    title_icon_size,
                )
                self._draw_icon(icon_bounds, icon_name, f"{key}_title_icon")
                title_x = icon_bounds.right + 12.0
            self._draw_text(
                f"{key}_panel_title",
                title,
                title_x,
                bounds.top - 33.0,
                self.theme.text_primary,
                19,
                bold=True,
                anchor_y="center",
            )
            self._control_hitboxes[f"{key}_drag"] = header
        else:
            self._control_hitboxes[f"{key}_drag"] = arcade.LBWH(
                bounds.left,
                bounds.top - 44.0,
                bounds.width,
                44.0,
            )

        if show_close:
            close_bounds = arcade.LBWH(bounds.right - 48, bounds.top - 42, 28, 28)
            self._control_hitboxes[f"{key}_close"] = close_bounds
            self._draw_panel_close_button(close_bounds, key)

        body_top = bounds.top - header_height - body_top_padding
        body = arcade.LBWH(
            bounds.left + 22.0,
            bounds.bottom + 18.0,
            max(0.0, bounds.width - 44.0),
            max(0.0, body_top - bounds.bottom - 18.0),
        )
        self._control_hitboxes[f"{key}_body"] = body
        return body

    def _draw_icon_button(
        self,
        bounds: arcade.Rect,
        icon_name: str,
        key: str,
        *,
        active: bool,
    ) -> None:
        fill = self.theme.accent_soft if active else self.theme.panel_background_alt
        border = self.theme.accent_soft if active else self.theme.panel_background_alt
        self._draw_rounded_rect(bounds, fill, border, 8, 1.0)
        icon_size = 26.0
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

    def _draw_panel_close_button(self, bounds: arcade.Rect, key: str) -> None:
        self._draw_icon(bounds, "kill", f"{key}_close_icon")

    def _draw_icon(
        self,
        bounds: arcade.Rect,
        icon_name: str,
        key: str,
    ) -> None:
        texture = self._icon_texture(icon_name)
        if texture is not None and self._draw_icon_texture(bounds, texture):
            return
        if self._draw_icon_sprite(bounds, icon_name, texture):
            return
        fallback = {
            "search": "?",
            "analytics": "#",
            "tune": "=",
            "brain": "@",
            "kill": "x",
            "globe": "O",
        }.get(icon_name, "*")
        self._draw_text(
            f"icon_fallback_{key}",
            fallback,
            bounds.center_x,
            bounds.center_y,
            self.theme.text_primary,
            max(11.0, bounds.height * 0.58),
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )

    def _draw_icon_texture(self, bounds: arcade.Rect, texture: object) -> bool:
        draw_texture_rectangle = getattr(arcade, "draw_texture_rectangle", None)
        if draw_texture_rectangle is not None:
            try:
                draw_texture_rectangle(
                    bounds.center_x,
                    bounds.center_y,
                    bounds.width,
                    bounds.height,
                    texture,
                )
                return True
            except TypeError:
                pass

        draw_texture_rect = getattr(arcade, "draw_texture_rect", None)
        if draw_texture_rect is not None:
            try:
                draw_texture_rect(texture, bounds)
                return True
            except TypeError:
                pass

        return False

    def _draw_icon_sprite(
        self,
        bounds: arcade.Rect,
        icon_name: str,
        texture: object | None,
    ) -> bool:
        sprite = self._icon_sprite(icon_name, texture)
        if sprite is None:
            return False
        try:
            sprite.center_x = bounds.center_x
            sprite.center_y = bounds.center_y
            sprite.width = bounds.width
            sprite.height = bounds.height
            sprite.draw()
        except (AttributeError, TypeError):
            return False
        return True

    def _icon_sprite(self, icon_name: str, texture: object | None) -> object | None:
        if icon_name in self._sprite_cache:
            return self._sprite_cache[icon_name]
        sprite_cls = getattr(arcade, "Sprite", None)
        if sprite_cls is None:
            self._sprite_cache[icon_name] = None
            return None

        sprite = None
        if texture is not None:
            try:
                sprite = sprite_cls(texture=texture)
            except TypeError:
                sprite = None
        if sprite is None:
            try:
                sprite = sprite_cls(str(self._icon_path(icon_name)))
            except (TypeError, FileNotFoundError):
                sprite = None

        self._sprite_cache[icon_name] = sprite
        return sprite

    def _icon_texture(self, icon_name: str) -> object | None:
        if icon_name in self._texture_cache:
            return self._texture_cache[icon_name]
        load_texture = getattr(arcade, "load_texture", None)
        if load_texture is None:
            self._texture_cache[icon_name] = None
            return None
        try:
            texture = load_texture(str(self._icon_path(icon_name)))
        except Exception:
            texture = None
        self._texture_cache[icon_name] = texture
        return texture

    def _icon_path(self, icon_name: str) -> Path:
        return Path(__file__).resolve().parents[1] / "assets" / f"{icon_name}.png"

    def _draw_status_chip(self, bounds: arcade.Rect, label: str) -> None:
        self._draw_rounded_rect(bounds, (188, 237, 220), (188, 237, 220), 999, 1)
        self._draw_text(
            f"status_chip_{label}",
            label,
            bounds.center_x,
            bounds.center_y,
            self.theme.accent,
            9,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )

    def _inspector_energy_ratio(self, world: World) -> float:
        selected = world.selected_creature
        if selected is None:
            return 0.0
        max_energy = max(0.0001, float(world.config.metabolism.max_energy))
        return max(0.0, min(1.0, selected.energy / max_energy))

    def _selected_species_identity(
        self,
        world: World,
        selected: object,
    ) -> tuple[int | None, arcade.Color | tuple[int, ...]]:
        lineage = getattr(selected, "lineage", None)
        raw_species_id = getattr(lineage, "species_id", None)
        try:
            species_id = int(raw_species_id)
        except (TypeError, ValueError):
            return None, self.theme.herbivore_fill

        color = getattr(selected, "color", None)
        if (
            not isinstance(color, (tuple, list))
            or len(color) < 3
        ):
            records = getattr(world, "species_history", {}) or {}
            record = records.get(species_id)
            color = (
                getattr(record, "founder_color", None)
                if record is not None
                else None
            )
        if not isinstance(color, (tuple, list)) or len(color) < 3:
            color = self.theme.herbivore_fill
        return species_id, tuple(color[:3])

    def _biome_food_summary(self, world: World) -> str:
        counts = world.stats.biome_food_counts
        return (
            f"F:{counts.get('Forest', 0)} "
            f"B:{counts.get('Bushes', 0)} "
            f"P:{counts.get('Prairie', 0)}"
        )

    def _biome_area_summary(self, world: World) -> str:
        shares = world.stats.biome_area_shares
        return (
            f"F:{shares.get('Forest', 0.0):.0%} "
            f"B:{shares.get('Bushes', 0.0):.0%} "
            f"P:{shares.get('Prairie', 0.0):.0%}"
        )

    def _inspector_energy_color(
        self,
        ratio: float,
    ) -> arcade.Color | tuple[int, ...]:
        if ratio < 0.25:
            return self.theme.selected_outline
        if ratio < 0.55:
            return (186, 112, 42)
        return self.theme.accent

    def _draw_text_in_viewport(
        self,
        viewport: arcade.Rect,
        key: str,
        text: str,
        x: float,
        y: float,
        color: arcade.Color | tuple[int, ...],
        size: float,
        **kwargs: object,
    ) -> None:
        if viewport.bottom <= y <= viewport.top:
            self._draw_text(key, text, x, y, color, size, **kwargs)

    def _draw_status_chip_in_viewport(
        self,
        viewport: arcade.Rect,
        bounds: arcade.Rect,
        label: str,
    ) -> None:
        if self._rect_intersects(bounds, viewport):
            self._draw_status_chip(bounds, label)

    def _draw_inspector_section_label(
        self,
        viewport: arcade.Rect,
        key: str,
        label: str,
        x: float,
        y: float,
    ) -> None:
        self._draw_text_in_viewport(
            viewport,
            key,
            label,
            x,
            y,
            self.theme.text_muted,
            9,
            bold=True,
        )

    def _draw_inspector_stat_tile_in_viewport(
        self,
        viewport: arcade.Rect,
        key: str,
        label: str,
        value: str,
        bounds: arcade.Rect,
    ) -> None:
        if not self._rect_intersects(bounds, viewport):
            return
        self._draw_rounded_rect(
            bounds,
            self.theme.panel_background,
            self.theme.panel_border,
            7,
            1,
        )
        self._draw_text(
            f"{key}_tile_label",
            label,
            bounds.left + 10,
            bounds.top - 15,
            self.theme.text_muted,
            9,
        )
        self._draw_text(
            f"{key}_tile_value",
            self._fit_line(value, bounds.width - 20),
            bounds.left + 10,
            bounds.bottom + 12,
            self.theme.text_primary,
            12,
            bold=True,
        )

    def _draw_metric_row_in_viewport(
        self,
        viewport: arcade.Rect,
        key: str,
        label: str,
        value: str,
        x: float,
        y: float,
        width: float,
        *,
        value_color: arcade.Color | tuple[int, ...] | None = None,
    ) -> None:
        if viewport.bottom <= y <= viewport.top:
            self._draw_metric_row(
                key,
                label,
                value,
                x,
                y,
                width,
                value_color=value_color,
            )

    def _rect_intersects(self, first: arcade.Rect, second: arcade.Rect) -> bool:
        return not (
            first.right < second.left
            or first.left > second.right
            or first.top < second.bottom
            or first.bottom > second.top
        )

    def _draw_compact_value(
        self,
        key: str,
        label: str,
        value: str,
        x: float,
        y: float,
        width: float,
    ) -> None:
        self._draw_text(
            f"{key}_label",
            label,
            x,
            y,
            self.theme.text_muted,
            9,
        )
        self._draw_text(
            f"{key}_value",
            self._fit_line(value, width),
            x,
            y - 16,
            self.theme.text_primary,
            12,
        )

    def _draw_metric_row(
        self,
        key: str,
        label: str,
        value: str,
        x: float,
        y: float,
        width: float,
        *,
        value_color: arcade.Color | tuple[int, ...] | None = None,
    ) -> None:
        self._draw_text(
            f"{key}_label",
            label,
            x,
            y,
            self.theme.text_muted,
            10,
        )
        self._draw_text(
            f"{key}_value",
            self._fit_line(value, width * 0.46),
            x + width,
            y,
            value_color or self.theme.text_primary,
            12,
            anchor_x="right",
        )

    def _draw_progress_bar(
        self,
        bounds: arcade.Rect,
        ratio: float,
        *,
        fill_color: arcade.Color | tuple[int, ...] | None = None,
    ) -> None:
        epsilon = 1e-6
        ratio = max(0.0, min(1.0, ratio))
        radius = bounds.height / 2.0
        self._draw_rounded_rect_fill(bounds, (222, 224, 255), radius)
        fill_width = bounds.width * ratio
        if fill_width <= epsilon:
            return
        fill = arcade.LBWH(bounds.left, bounds.bottom, fill_width, bounds.height)
        fill_radius = min(radius, fill.width / 2.0, fill.height / 2.0)
        self._draw_rounded_rect_fill(fill, fill_color or self.theme.accent, fill_radius)

    def _draw_action_button(
        self,
        bounds: arcade.Rect,
        label: str,
        icon_name: str,
        key: str,
        *,
        fill_color: arcade.Color | tuple[int, ...],
        text_color: arcade.Color | tuple[int, ...],
    ) -> None:
        self._draw_rounded_rect(bounds, fill_color, fill_color, 7, 1)
        icon_size = 20.0
        icon_text_gap = 12.0
        fitted_label = self._fit_line(
            label, bounds.width - icon_size - icon_text_gap - 18
        )
        text_width = min(
            max(0.0, bounds.width - icon_size - icon_text_gap - 18),
            len(fitted_label) * 7.0,
        )
        group_width = icon_size + icon_text_gap + text_width
        group_left = bounds.center_x - group_width / 2.0
        icon_bounds = arcade.LBWH(
            group_left,
            bounds.center_y - icon_size / 2.0,
            icon_size,
            icon_size,
        )
        self._draw_icon(icon_bounds, icon_name, f"{key}_icon")
        self._draw_text(
            f"action_button_{key}",
            fitted_label,
            icon_bounds.right + icon_text_gap,
            bounds.center_y,
            text_color,
            12,
            bold=True,
            anchor_x="left",
            anchor_y="center",
        )

    def _draw_play_pause_button(self, bounds: arcade.Rect, *, is_paused: bool) -> None:
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

    def _draw_icon_text_button(
        self,
        bounds: arcade.Rect,
        label: str,
        key: str,
        *,
        fill_color: arcade.Color | tuple[int, ...] | None,
        size: float,
        y_offset: float = 0.0,
    ) -> None:
        if fill_color is not None:
            self._draw_rounded_rect(bounds, fill_color, fill_color, 8, 1)
        self._draw_text(
            f"icon_text_button_{key}",
            label,
            bounds.center_x,
            bounds.center_y + y_offset,
            self.theme.accent,
            size,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )

    def _draw_keycap(
        self,
        key: str,
        key_label: str,
        label: str,
        x: float,
        y: float,
    ) -> None:
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

    def _format_decimal(self, value: float) -> str:
        if abs(value) >= 1000.0:
            return f"{value:.1f}"
        if abs(value) >= 100.0:
            return f"{value:.2f}"
        return f"{value:.2f}".rstrip("0").rstrip(".")

    def _format_mutation_delta(self, mutation_delta: object | None) -> str:
        if mutation_delta is None:
            return "None"

        vision_range = getattr(mutation_delta, "vision_range", 0.0)
        vision_angle = getattr(mutation_delta, "vision_angle", 0.0)
        radius = getattr(mutation_delta, "radius", 0.0)
        movement_cost = getattr(mutation_delta, "movement_cost_multiplier", 0.0)
        separation = getattr(mutation_delta, "separation_gene", 0.0)
        alignment = getattr(mutation_delta, "alignment_gene", 0.0)
        cohesion = getattr(mutation_delta, "cohesion_gene", 0.0)
        return (
            f"R {radius:+.1f}, V {vision_range:+.1f}/"
            f"{vision_angle:+.2f}, M {movement_cost:+.2f}, "
            f"F {separation:+.2f}/{alignment:+.2f}/{cohesion:+.2f}"
        )

    def _draw_card(self, bounds: arcade.Rect, title: str) -> None:
        self._draw_rounded_rect(
            bounds,
            self.theme.card_background,
            self.theme.panel_border,
            self.config.layout.card_radius,
            2,
        )
        self._draw_text(
            f"card_title_{title}",
            title,
            bounds.left + 16,
            bounds.top - 24,
            self.theme.text_primary,
            14,
            bold=True,
        )

    def _draw_button(self, bounds: arcade.Rect, label: str, key: str) -> None:
        self._draw_rounded_rect(
            bounds,
            self.theme.panel_background,
            self.theme.panel_border,
            8,
            1.5,
        )
        self._draw_text(
            f"button_{key}",
            self._fit_line(label, bounds.width - 8),
            bounds.center_x,
            bounds.center_y,
            self.theme.text_primary,
            14,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )

    def _draw_speed_slider(self, bounds: arcade.Rect, world: World) -> None:
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

    def _node_column_positions(
        self,
        node_keys: list[int],
        x: float,
        bottom: float,
        top: float,
    ) -> dict[int, tuple[float, float]]:
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
        arcade.draw_circle_filled(position[0], position[1], radius, fill_color)
        arcade.draw_circle_outline(position[0], position[1], radius, outline_color, 1.5)

    def _brain_activity_color(self, value: float) -> arcade.Color | tuple[int, ...]:
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

    def _format_genome_fitness(self, fitness: object) -> str:
        if fitness is None:
            return "None"
        try:
            return f"{float(fitness):.2f}"
        except (TypeError, ValueError):
            return str(fitness)

    def _selected_fitness_label(self, world: World, selected: object) -> str:
        fitness = world.fitness_for(selected)
        if fitness is None:
            return "None"
        return self._format_genome_fitness(fitness.score(world.config.fitness))

    def _brain_value_readout(
        self,
        inputs: list[float],
        outputs: list[float],
    ) -> str:
        return f"{self._brain_input_readout(inputs)}\n{self._brain_output_readout(outputs)}"

    def _brain_input_readout(self, inputs: list[float]) -> str:
        def value(index: int, values: list[float]) -> float:
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
        def value(index: int, values: list[float]) -> float:
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

    def _contains_hitbox(self, key: str, x: float, y: float) -> bool:
        bounds = self._control_hitboxes.get(key)
        if bounds is None:
            return False
        return self._contains_bounds(bounds, x, y)

    def _contains_bounds(self, bounds: arcade.Rect, x: float, y: float) -> bool:
        return bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top

    def _set_speed_from_slider(self, world: World, x: float) -> None:
        bounds = self._control_hitboxes["speed_slider"]
        ratio = (x - bounds.left) / bounds.width
        ratio = max(0.0, min(1.0, ratio))
        speed = world.MIN_SIMULATION_SPEED + ratio * (
            world.MAX_SIMULATION_SPEED - world.MIN_SIMULATION_SPEED
        )
        world.set_simulation_speed(speed)

    def _draw_text(
        self,
        key: str,
        text: str,
        x: float,
        y: float,
        color: arcade.Color | tuple[int, ...],
        size: float,
        *,
        bold: bool = False,
        width: float | None = None,
        multiline: bool = False,
        align: str = "left",
        anchor_x: str = "left",
        anchor_y: str = "baseline",
    ) -> None:
        rx = round(x)
        ry = round(y)
        cached = self._text_cache.get(key)
        if cached is None:
            cached = arcade.Text(
                text,
                rx,
                ry,
                color,
                size,
                font_name=("Hanken Grotesk", "Manrope", "JetBrains Mono", "Arial"),
                bold=bold,
                width=width,
                align=align,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                multiline=multiline,
            )
            self._text_cache[key] = cached
        else:
            cached.text = text
            cached.x = rx
            cached.y = ry
            cached.color = color
            cached.font_size = size
            cached.bold = bold
            cached.width = width
            cached.multiline = multiline
            cached.align = align
            cached.anchor_x = anchor_x
            cached.anchor_y = anchor_y
        cached.draw()

    def _draw_scrollable_lines(
        self,
        key: str,
        card_bounds: arcade.Rect,
        lines: list[str],
        *,
        line_spacing: float,
        first_line_color: arcade.Color | tuple[int, ...],
        body_color: arcade.Color | tuple[int, ...],
        first_line_bold: bool = False,
    ) -> None:
        content = self._card_content_bounds(card_bounds)
        self._draw_scrollable_lines_in_bounds(
            key,
            content,
            lines,
            line_spacing=line_spacing,
            first_line_color=first_line_color,
            body_color=body_color,
            first_line_bold=first_line_bold,
        )

    def _draw_scrollable_lines_in_bounds(
        self,
        key: str,
        content: arcade.Rect,
        lines: list[str],
        *,
        line_spacing: float,
        first_line_color: arcade.Color | tuple[int, ...],
        body_color: arcade.Color | tuple[int, ...],
        first_line_bold: bool = False,
        wrap_lines: bool = False,
        draw_ethogram_markers: bool = False,
    ) -> None:
        visual_lines = (
            self._wrapped_scrollable_lines(
                lines,
                content.width - 12.0,
                draw_ethogram_markers=draw_ethogram_markers,
            )
            if wrap_lines
            else [
                (
                    line,
                    line_index == 0,
                    None,
                    0.0,
                )
                for line_index, line in enumerate(lines)
            ]
        )
        total_height = max(0.0, len(visual_lines) * line_spacing)
        scroll_limit = max(0.0, total_height - content.height)
        scroll_offset = max(
            0.0,
            min(scroll_limit, self._scroll_offsets.get(key, 0.0)),
        )
        self._scroll_offsets[key] = scroll_offset
        self._scroll_limits[key] = scroll_limit
        self._scroll_regions[key] = content

        for line_index, (line, is_first_line, marker_color, x) in enumerate(
            visual_lines
        ):
            y = content.top - 12 - line_index * line_spacing + scroll_offset
            if y < content.bottom or y > content.top:
                continue
            if marker_color is not None:
                arcade.draw_circle_filled(
                    content.left + 8.0,
                    y + 4.0,
                    6.0,
                    marker_color,
                )
            self._draw_text(
                f"{key}_line_{line_index}",
                (
                    line
                    if wrap_lines
                    else self._fit_line(
                        line,
                        content.width - (12 if scroll_limit > 0 else 0),
                    )
                ),
                content.left + x,
                y,
                first_line_color if is_first_line else body_color,
                12,
                bold=first_line_bold and is_first_line,
            )

        if scroll_limit > 0.0:
            self._draw_scrollbar(content, scroll_offset, scroll_limit)

    def _wrapped_scrollable_lines(
        self,
        lines: list[str],
        width: float,
        *,
        draw_ethogram_markers: bool,
    ) -> list[
        tuple[
            str,
            bool,
            tuple[int, int, int] | None,
            float,
        ]
    ]:
        visual_lines: list[
            tuple[str, bool, tuple[int, int, int] | None, float]
        ] = []
        base_x = 0.0
        for logical_index, raw_line in enumerate(lines):
            marker_color: tuple[int, int, int] | None = None
            line = raw_line
            marker_indent = 0.0
            if draw_ethogram_markers and line:
                marker_color = self._ethogram_marker_color(line[0])
                if marker_color is not None:
                    line = line[1:].lstrip()
                    marker_indent = 20.0

            available_width = max(24.0, width - marker_indent)
            wrapped = self._wrap_line(line, available_width)
            if not wrapped:
                wrapped = [""]
            for wrapped_index, wrapped_line in enumerate(wrapped):
                visual_lines.append(
                    (
                        wrapped_line,
                        logical_index == 0 and wrapped_index == 0,
                        marker_color if wrapped_index == 0 else None,
                        base_x + marker_indent,
                    )
                )
        return visual_lines

    @staticmethod
    def _ethogram_marker_color(
        marker: str,
    ) -> tuple[int, int, int] | None:
        return {
            "🟢": (0, 210, 72),
            "🔴": (255, 55, 65),
            "⚪": (150, 160, 170),
        }.get(marker)

    def _wrap_line(self, text: str, width: float) -> list[str]:
        max_chars = max(4, int(width / 7.0))
        if len(text) <= max_chars:
            return [text]
        leading = len(text) - len(text.lstrip(" "))
        indent = text[:leading]
        continuation_indent = indent
        words = text.strip().split()
        if not words:
            return [text]

        lines: list[str] = []
        current = indent
        chunk_size = max(1, max_chars - len(continuation_indent))
        for word in words:
            if len(word) > chunk_size:
                if current.strip():
                    lines.append(current)
                    current = continuation_indent
                for start in range(0, len(word), chunk_size):
                    chunk = word[start : start + chunk_size]
                    if start + chunk_size < len(word):
                        lines.append(f"{continuation_indent}{chunk}")
                    else:
                        current = f"{continuation_indent}{chunk}"
                continue
            candidate = (
                f"{current} {word}"
                if current.strip()
                else f"{continuation_indent}{word}"
            )
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current.strip():
                lines.append(current)
            current = f"{continuation_indent}{word}"
        if current.strip() or not lines:
            lines.append(current)
        return lines

    def _card_content_bounds(self, bounds: arcade.Rect) -> arcade.Rect:
        bottom = bounds.bottom + 12
        top = bounds.top - 42
        return arcade.LBWH(
            bounds.left + 16,
            bottom,
            max(0.0, bounds.width - 32),
            max(0.0, top - bottom),
        )

    def _draw_scrollbar(
        self, bounds: arcade.Rect, scroll_offset: float, scroll_limit: float
    ) -> None:
        track_width = 3
        track_left = bounds.right - track_width
        arcade.draw_lrbt_rectangle_filled(
            track_left,
            bounds.right,
            bounds.bottom,
            bounds.top,
            self.theme.panel_border,
        )
        visible_ratio = bounds.height / (bounds.height + scroll_limit)
        thumb_height = max(18.0, bounds.height * visible_ratio)
        travel = max(0.0, bounds.height - thumb_height)
        thumb_top = bounds.top - travel * (scroll_offset / scroll_limit)
        arcade.draw_lrbt_rectangle_filled(
            track_left,
            bounds.right,
            thumb_top - thumb_height,
            thumb_top,
            self.theme.accent,
        )

    def _fit_line(self, text: str, width: float) -> str:
        max_chars = max(4, int(width / 7.0))
        if len(text) <= max_chars:
            return text
        return f"{text[: max_chars - 3]}..."

    def _draw_rounded_rect(
        self,
        bounds: arcade.Rect,
        fill_color: arcade.Color | tuple[int, ...],
        border_color: arcade.Color | tuple[int, ...],
        radius: float,
        border_width: float,
    ) -> None:
        self._draw_rounded_rect_fill(bounds, border_color, radius)
        inner = arcade.LBWH(
            bounds.left + border_width,
            bounds.bottom + border_width,
            max(0, bounds.width - border_width * 2),
            max(0, bounds.height - border_width * 2),
        )
        self._draw_rounded_rect_fill(inner, fill_color, max(0, radius - border_width))

    def _draw_rounded_rect_fill(
        self,
        bounds: arcade.Rect,
        color: arcade.Color | tuple[int, ...],
        radius: float,
    ) -> None:
        if bounds.width <= 0 or bounds.height <= 0:
            return
        radius = min(radius, bounds.width / 2, bounds.height / 2)
        horizontal_left = bounds.left + radius
        horizontal_right = bounds.right - radius
        vertical_bottom = bounds.bottom + radius
        vertical_top = bounds.top - radius
        if horizontal_left <= horizontal_right:
            arcade.draw_lrbt_rectangle_filled(
                horizontal_left,
                horizontal_right,
                bounds.bottom,
                bounds.top,
                color,
            )
        if vertical_bottom <= vertical_top:
            arcade.draw_lrbt_rectangle_filled(
                bounds.left,
                bounds.right,
                vertical_bottom,
                vertical_top,
                color,
            )
        if radius <= 0:
            return
        arcade.draw_circle_filled(
            bounds.left + radius, bounds.bottom + radius, radius, color
        )
        arcade.draw_circle_filled(
            bounds.right - radius, bounds.bottom + radius, radius, color
        )
        arcade.draw_circle_filled(
            bounds.left + radius, bounds.top - radius, radius, color
        )
        arcade.draw_circle_filled(
            bounds.right - radius, bounds.top - radius, radius, color
        )
