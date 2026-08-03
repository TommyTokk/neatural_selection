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

class StatsPanelComponent:
    """Group related behavior extracted from ``UiRenderer``."""

    def _draw_stats_panel(self, world: World) -> None:
        """Draw stats panel.

        Parameters
        ----------
        world
            Simulation world providing current state.
        """
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
        action_height = 44.0
        action = arcade.LBWH(
            content.left + 8.0,
            content.bottom + 6.0,
            content.width - 16.0,
            action_height,
        )
        self._control_hitboxes["open_behavior_report"] = action
        self._draw_rounded_rect(
            action,
            self.theme.accent_soft,
            self.theme.accent,
            8.0,
            1.0,
        )
        self._draw_text(
            "stats_open_behavior_report",
            "View Behaviour History",
            action.center_x,
            action.center_y,
            self.theme.accent,
            12.0,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )

        metrics_viewport = arcade.LBWH(
            content.left,
            action.top + 12.0,
            content.width,
            max(0.0, content.top - action.top - 12.0),
        )
        row_spacing = 25.0
        total_height = len(metrics) * row_spacing
        scroll_limit = max(
            0.0,
            total_height - metrics_viewport.height + 10.0,
        )
        scroll_offset = max(
            0.0,
            min(scroll_limit, self._scroll_offsets.get("stats", 0.0)),
        )
        self._scroll_offsets["stats"] = scroll_offset
        self._scroll_limits["stats"] = scroll_limit
        self._scroll_regions["stats"] = metrics_viewport
        for index, (label, value, color) in enumerate(metrics):
            y = metrics_viewport.top - 20.0 - index * row_spacing + scroll_offset
            if y < metrics_viewport.bottom + 4 or y > metrics_viewport.top:
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
            self._draw_scrollbar(
                metrics_viewport,
                scroll_offset,
                scroll_limit,
            )
    def _draw_environment_stats(self, world: World, bounds: arcade.Rect) -> None:
        """Draw environment stats.

        Parameters
        ----------
        world
            Simulation world providing current state.
        bounds
            Rectangle defining the relevant UI area.
        """
        lag = world.simulation_lag_metrics
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
            "Controller: NEAT",
            f"Simulation speed: {world.simulation_speed:.2f}x",
            f"Effective speed: {lag.effective_speed_multiplier:.2f}x",
            f"Pending simulation: {lag.pending_seconds:.3f}s",
            f"Dropped this session: {lag.session_dropped_seconds:.3f}s",
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
