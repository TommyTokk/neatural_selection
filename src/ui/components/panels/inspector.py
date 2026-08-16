from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from math import ceil, cos, floor, hypot, isfinite, log1p, log10, pi, sin
from pathlib import Path
import re
from types import SimpleNamespace

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

class InspectorPanelComponent:
    """Group related behavior extracted from ``UiRenderer``."""

    def _draw_inspector_panel(self, world: World) -> None:
        """Draw inspector panel.

        Parameters
        ----------
        world
            Simulation world providing current state.
        """
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
        """Draw inspector page marker.

        Parameters
        ----------
        card
            Rectangle defining the relevant UI area.
        """
        marker = arcade.LBWH(card.left, card.bottom, 7.0, card.height)
        self._draw_left_rounded_rect_fill(marker, self.theme.accent, 8.0)
    def _draw_left_rounded_rect_fill(
        self,
        bounds: arcade.Rect,
        color: arcade.Color | tuple[int, ...],
        radius: float,
    ) -> None:
        """Draw left rounded rect fill.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        color
            Arcade-compatible color.
        radius
            Requested logical size.
        """
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
        """Draw inspector content.

        Parameters
        ----------
        world
            Simulation world providing current state.
        viewport
            Rectangle defining the relevant UI area.
        """
        selected = world.selected_creature
        if selected is None:
            return

        snapshot = self._cached_inspector_snapshot(world, selected)
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
        stomach_capacity = max(
            0.0,
            float(
                getattr(
                    physical_traits,
                    "stomach_capacity",
                    float(radius or 0.0)
                    * float(
                        getattr(
                            world.config.metabolism,
                            "stomach_capacity_per_radius",
                            0.1,
                        )
                    ),
                )
            ),
        )
        stomach_energy = max(
            0.0,
            float(getattr(selected, "stomach_energy", 0.0)),
        )
        trait_config = getattr(world.config, "trait", None)
        digestion_rate = float(
            getattr(
                physical_traits,
                "digestion_rate",
                getattr(trait_config, "default_digestion_rate", 0.2),
            )
        )
        digestion_efficiency = float(
            getattr(
                physical_traits,
                "digestion_efficiency",
                getattr(trait_config, "default_digestion_efficiency", 0.9),
            )
        )
        metabolism_model = getattr(world, "metabolism", None)
        upkeep_calculator = getattr(
            metabolism_model,
            "digestive_upkeep_energy_cost_per_second",
            None,
        )
        digestive_upkeep = (
            float(upkeep_calculator(selected))
            if callable(upkeep_calculator)
            else 0.0
        )
        processing_cost = float(
            getattr(
                world,
                "_last_digestion_processing_costs_per_second",
                {},
            ).get(selected.creature_id, 0.0)
        )
        flocking_traits = getattr(selected, "flocking_traits", None)
        current_action = getattr(selected, "last_action", None)
        effective_herding = float(getattr(current_action, "herding", 0.0))
        panic = float(
            getattr(current_action, "flee_panic_intensity", 0.0)
        )
        flock_runtime = getattr(
            world,
            "_last_flocking_runtime",
            {},
        ).get(selected.creature_id)
        raw_neural_herding = float(
            getattr(
                flock_runtime,
                "raw_neural_herding",
                effective_herding,
            )
        )
        effective_herding = float(
            getattr(
                flock_runtime,
                "effective_herding",
                effective_herding,
            )
        )
        parent_id = getattr(lineage, "parent_id", None)
        generation = getattr(lineage, "generation", 0)
        fitness_score = (
            fitness.score(selected) if fitness is not None else None
        )
        species_id, species_color = self._selected_species_identity(
            world,
            selected,
        )
        energy_ratio = self._inspector_energy_ratio(world)
        max_life = max(
            1e-12,
            float(getattr(world.config.metabolism, "max_life", 1.0)),
        )
        life_ratio = max(
            0.0,
            min(1.0, float(getattr(selected, "life", max_life)) / max_life),
        )
        ledger = getattr(selected, "ledger_diagnostics", None)
        activity_diagnostics = getattr(ledger, "activity", None)
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
        estimated_total_height = (
            (1199.0 if fitness_score is not None else 1167.0)
            + species_row_height
        )
        total_height = max(
            estimated_total_height,
            self._inspector_content_height,
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
        content_top = y

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

        y -= 22.0
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_stomach_storage",
            "Stored / capacity",
            f"{stomach_energy:.3f} / {stomach_capacity:.3f}",
            left,
            y,
            width,
        )
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_stomach_fullness",
            "Fullness",
            f"{stomach_ratio:.1%}",
            left,
            y,
            width,
        )
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_digestion_traits",
            "Rate / efficiency",
            f"{digestion_rate:.3f}/s / {digestion_efficiency:.1%}",
            left,
            y,
            width,
        )
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_digestive_upkeep",
            "Digestive upkeep",
            f"{digestive_upkeep:.4f}/s",
            left,
            y,
            width,
        )
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_processing_cost",
            "Recent processing cost",
            f"{processing_cost:.4f}/s",
            left,
            y,
            width,
        )

        y -= section_gap
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
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_vision_range",
            "Vision",
            f"{selected.vision.range:.0f}px / {selected.vision.angle:.2f} rad",
            left,
            y,
            width,
        )
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_flockmate_count",
            "Flockmates (eff/net)",
            f"{effective_flockmate_count:.2f} / {normalized_flockmate_count:.2f}",
            left,
            y,
            width,
        )
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_vision_cost",
            "Cost",
            f"{vision_cost:.3f}/s",
            left,
            y,
            width,
        )
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_body",
            "Body",
            f"{radius:.1f}px / {movement_cost_multiplier:.2f}x move",
            left,
            y,
            width,
        )
        y -= self._draw_metric_row_in_viewport(
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
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_herding",
            "Herding raw / effective",
            f"{raw_neural_herding:.2f} / {effective_herding:.2f}",
            left,
            y,
            width,
        )
        observation = (
            None if flock_runtime is None else flock_runtime.observation
        )
        intent = None if flock_runtime is None else flock_runtime.intent
        weights = None if intent is None else intent.weights
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_social_presence",
            "Presence (personal/social)",
            (
                "Unavailable"
                if observation is None
                else (
                    f"{observation.personal_space_presence:.2f} / "
                    f"{observation.social_presence:.2f}"
                )
            ),
            left,
            y,
            width,
        )
        flocking_config = getattr(world.config, "flocking", None)
        compatibility_config = getattr(
            flocking_config,
            "compatibility",
            None,
        )
        raw_compatibility_mode = getattr(
            compatibility_config,
            "mode",
            "legacy",
        )
        compatibility_mode = getattr(
            raw_compatibility_mode,
            "value",
            raw_compatibility_mode,
        )
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_social_compatibility",
            "Social compatibility",
            (
                f"{compatibility_mode} / tag "
                f"({getattr(flocking_traits, 'social_tag_x', 0.5):.2f}, "
                f"{getattr(flocking_traits, 'social_tag_y', 0.5):.2f})"
            ),
            left,
            y,
            width,
        )
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_effective_flocking_weights",
            "Effective weights",
            (
                "Unavailable"
                if weights is None
                else (
                    f"S {weights.separation:.2f} / "
                    f"A {weights.alignment:.2f} / "
                    f"C {weights.cohesion:.2f}"
                )
            ),
            left,
            y,
            width,
        )
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_social_engagement",
            "Engagement / panic attenuation",
            (
                f"{effective_herding:.2f} / {1.0 - panic:.2f}"
                if weights is None
                else (
                    f"{weights.engagement:.2f} / "
                    f"{weights.panic_attenuation:.2f}"
                )
            ),
            left,
            y,
            width,
        )
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_desired_velocities",
            "Desired velocity N / S / blend",
            (
                "Recomputing"
                if flock_runtime is None
                else (
                    f"{hypot(*flock_runtime.neural_desired_velocity):.1f} / "
                    f"{hypot(*intent.desired_velocity):.1f} / "
                    f"{hypot(*flock_runtime.blended_desired_velocity):.1f}"
                )
            ),
            left,
            y,
            width,
        )
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_social_force_contribution",
            "Counterfactual Δ requested / accepted",
            (
                "Recomputing"
                if flock_runtime is None
                else (
                    f"{hypot(*flock_runtime.requested_social_contribution):.2f} / "
                    f"{hypot(*flock_runtime.accepted_counterfactual_delta):.2f}"
                )
            ),
            left,
            y,
            width,
        )
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_social_blend",
            "Social blend",
            (
                "Recomputing"
                if flock_runtime is None
                else f"{flock_runtime.social_influence:.1%}"
            ),
            left,
            y,
            width,
        )
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_local_group",
            "Persistent local group",
            (
                "None"
                if flock_runtime is None
                or flock_runtime.local_group_id is None
                else (
                    f"#{flock_runtime.local_group_id} / "
                    f"{flock_runtime.local_group_size} members"
                )
            ),
            left,
            y,
            width,
        )
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_flocking_benchmark",
            "Benchmark fitness",
            (
                "0.000"
                if fitness is None
                else f"{fitness.flocking_benchmark_reward:.3f}"
            ),
            left,
            y,
            width,
        )
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_collision_avoidance",
            "Collision avoidance",
            "Universal / automatic",
            left,
            y,
            width,
        )
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_lineage",
            "Lineage",
            f"Parent {parent_id if parent_id is not None else 'None'} / Gen {generation}",
            left,
            y,
            width,
        )
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_mutations",
            "Mutations",
            self._format_mutation_delta(mutation_delta),
            left,
            y,
            width,
        )
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_food",
            "Food",
            f"{snapshot.food.visible:.0f} seen / {snapshot.food.density:.2f}",
            left,
            y,
            width,
        )
        y -= self._draw_metric_row_in_viewport(
            viewport,
            "inspector_near",
            "Near",
            f"{snapshot.creatures.visible:.0f} seen / {snapshot.creatures.density:.2f}",
            left,
            y,
            width,
        )

        y -= section_gap
        self._draw_inspector_section_label(
            viewport, "inspector_ledger_section", "REST & LEDGER", left, y
        )
        y -= 22.0
        for key, label, value in (
            ("life", "Life reserve", f"{life_ratio:.1%}"),
            (
                "rest_stages",
                "Rest intent / smooth / effective",
                f"{getattr(selected, 'rest_intent', 0.0):.2f} / "
                f"{getattr(selected, 'smoothed_rest', 0.0):.2f} / "
                f"{getattr(selected, 'effective_rest', 0.0):.2f}",
            ),
            (
                "activity_components",
                "Activity M/S/T/C/R/N",
                f"{getattr(activity_diagnostics, 'voluntary_motor_effort', 0.0):.2f}/"
                f"{getattr(activity_diagnostics, 'normalized_speed', 0.0):.2f}/"
                f"{getattr(activity_diagnostics, 'turn', 0.0):.2f}/"
                f"{getattr(activity_diagnostics, 'communication', 0.0):.2f}/"
                f"{getattr(activity_diagnostics, 'reproduction', 0.0):.0f}/"
                f"{getattr(activity_diagnostics, 'nursing', 0.0):.0f}",
            ),
            (
                "activity_total",
                "Weighted activity",
                f"{getattr(selected, 'activity', 0.0):.3f}",
            ),
            (
                "digestion_ledger",
                "Consumed / gross / net",
                f"{getattr(ledger, 'stomach_consumed', 0.0):.4f} / "
                f"{getattr(ledger, 'gross_energy', 0.0):.4f} / "
                f"{getattr(ledger, 'net_energy', 0.0):.4f}",
            ),
            (
                "energy_ledger",
                "Demand / deficit",
                f"{getattr(ledger, 'total_energy_demand', 0.0):.4f} / "
                f"{getattr(ledger, 'unmet_energy_demand', 0.0):.4f}",
            ),
            (
                "rest_recovery",
                "Rest gain / healing spend",
                f"{getattr(ledger, 'rest_energy_recovered', 0.0):.4f} / "
                f"{getattr(ledger, 'healing_energy_spent', 0.0):.4f}",
            ),
            (
                "life_damage",
                "Deficit / direct / healed",
                f"{getattr(ledger, 'life_damage_from_deficit', 0.0):.4f} / "
                f"{getattr(ledger, 'direct_life_damage', 0.0):.4f} / "
                f"{getattr(ledger, 'life_healed', 0.0):.4f}",
            ),
            (
                "transaction_status",
                "Final transaction",
                str(getattr(ledger, "transaction_status", "not_evaluated")),
            ),
        ):
            y -= self._draw_metric_row_in_viewport(
                viewport,
                f"inspector_{key}",
                label,
                value,
                left,
                y,
                width,
            )

        if fitness_score is not None:
            y -= self._draw_metric_row_in_viewport(
                viewport,
                "inspector_age",
                "Age",
                f"{fitness.age_seconds:.1f}s",
                left,
                y,
                width,
            )

        y -= 56.0
        button_gap = 10.0
        button_height = 40.0
        brain_button = arcade.LBWH(
            left,
            y - button_height,
            width,
            button_height,
        )
        report_button = arcade.LBWH(
            left,
            brain_button.bottom - button_gap - button_height,
            width,
            button_height,
        )
        kill_button = arcade.LBWH(
            left,
            report_button.bottom - button_gap - button_height,
            width,
            button_height,
        )
        measured_content_height = max(
            0.0,
            content_top - kill_button.bottom + 28.0,
        )
        self._inspector_content_height = measured_content_height
        measured_scroll_limit = max(
            0.0,
            measured_content_height - viewport.height,
        )
        self._scroll_limits["inspector"] = measured_scroll_limit
        self._scroll_offsets["inspector"] = min(
            scroll_offset,
            measured_scroll_limit,
        )
        self._control_hitboxes.pop("open_brain_window", None)
        self._control_hitboxes.pop("open_behavior_report_selected", None)
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
        if self._rect_intersects(report_button, viewport):
            self._control_hitboxes[
                "open_behavior_report_selected"
            ] = report_button
            self._draw_action_button(
                report_button,
                "Report",
                "analytics",
                "open_behavior_report_selected",
                fill_color=self.theme.panel_background_alt,
                text_color=self.theme.text_primary,
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
        if measured_scroll_limit > 0.0:
            self._draw_scrollbar(
                viewport,
                self._scroll_offsets["inspector"],
                measured_scroll_limit,
            )
    def _draw_selected_creature(self, world: World, bounds: arcade.Rect) -> None:
        """Draw selected creature.

        Parameters
        ----------
        world
            Simulation world providing current state.
        bounds
            Rectangle defining the relevant UI area.
        """
        selected = world.selected_creature

        lines: list[str]
        if selected is None:
            lines = [
                "No creature selected.",
                "Click inside the environment",
                "to inspect a herbivore.",
            ]
        else:
            snapshot = self._cached_inspector_snapshot(world, selected)
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
                        f"Fitness: {fitness.score(selected):.2f}",
                        f"Age: {fitness.age_seconds:.1f}s",
                        f"Food eaten: {fitness.food_eaten}",
                        "Lifetime energy gathered: "
                        f"{selected.total_energy_gathered:.3f}",
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
        button_height = 40.0
        button_gap = 10.0
        kill_button = arcade.LBWH(
            content.left,
            content.bottom,
            content.width,
            button_height,
        )
        report_button = arcade.LBWH(
            content.left,
            kill_button.top + button_gap,
            content.width,
            button_height,
        )
        open_button = arcade.LBWH(
            content.left,
            report_button.top + button_gap,
            content.width,
            button_height,
        )
        lines_bounds = arcade.LBWH(
            content.left,
            open_button.top + button_gap,
            content.width,
            max(
                0.0,
                content.height - button_height * 3.0 - button_gap * 3.0,
            ),
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
        self._control_hitboxes["open_behavior_report_selected"] = (
            report_button
        )
        self._control_hitboxes["kill_selected_creature"] = kill_button
        self._draw_button(open_button, "Brain", "open_brain_window")
        self._draw_button(
            report_button,
            "Report",
            "open_behavior_report_selected",
        )
        self._draw_button(kill_button, "Kill", "kill_selected_creature")

    @staticmethod
    def _cached_inspector_snapshot(
        world: World,
        selected: object,
    ) -> object:
        """Return cached sensing data without triggering simulation work."""
        cached = getattr(world, "_last_sensor_snapshots", None)
        if cached is None:
            # Compatibility for lightweight UI hosts and existing test doubles;
            # production World instances always own the cache.
            return world.sensor_snapshot_for(selected)
        snapshot = cached.get(getattr(selected, "creature_id", None))
        if snapshot is not None:
            return snapshot
        empty_target = SimpleNamespace(
            visible=0.0,
            density=0.0,
            proximity=0.0,
            angle=0.0,
        )
        return SimpleNamespace(
            food=empty_target,
            creatures=empty_target,
            stomach_fullness=0.0,
            flock=SimpleNamespace(flockmate_count=0.0),
        )

    def _draw_selected_brain(self, world: World, bounds: arcade.Rect) -> None:
        """Draw selected brain.

        Parameters
        ----------
        world
            Simulation world providing current state.
        bounds
            Rectangle defining the relevant UI area.
        """
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
            usage.has_enabled_path for usage in sensor_usage[17:20]
        )
        detail_lines = [
            f"Genome: {brain.genome_id}",
            f"Signed action: {action_label}",
            f"Speed: {selected.speed:.1f} px/s",
            self._brain_input_readout(brain.last_inputs),
            self._brain_output_readout(brain.last_outputs),
            f"Nodes: {len(brain.genome.nodes)}",
            f"Connections: {enabled_connections}/{len(brain.genome.connections)} enabled",
            f"Biome sensing paths: {biome_path_count}/3",
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
    def _inspector_energy_ratio(self, world: World) -> float:
        """Return inspector energy ratio.

        Parameters
        ----------
        world
            Simulation world providing current state.

        Returns
        -------
        float
            Computed result.
        """
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
        """Return selected species identity.

        Parameters
        ----------
        world
            Simulation world providing current state.
        selected
            Value used by the operation.

        Returns
        -------
        tuple[int | None, arcade.Color | tuple[int, ...]]
            Computed collection.
        """
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
        """Return biome food summary.

        Parameters
        ----------
        world
            Simulation world providing current state.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        counts = world.stats.biome_food_counts
        return (
            f"F:{counts.get('Forest', 0)} "
            f"B:{counts.get('Bushes', 0)} "
            f"P:{counts.get('Prairie', 0)}"
        )
    def _biome_area_summary(self, world: World) -> str:
        """Return biome area summary.

        Parameters
        ----------
        world
            Simulation world providing current state.

        Returns
        -------
        str
            Formatted or resolved value.
        """
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
        """Return inspector energy color.

        Parameters
        ----------
        ratio
            Value used by the operation.

        Returns
        -------
        arcade.Color | tuple[int, ...]
            Computed result.
        """
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
        """Draw text in viewport.

        Parameters
        ----------
        viewport
            Rectangle defining the relevant UI area.
        key
            Stable identifier used by the UI.
        text
            Text displayed by the UI.
        x
            Logical screen coordinate.
        y
            Logical screen coordinate.
        color
            Arcade-compatible color.
        size
            Requested logical size.
        kwargs
            Value used by the operation.
        """
        if viewport.bottom <= y <= viewport.top:
            self._draw_text(key, text, x, y, color, size, **kwargs)
    def _draw_status_chip_in_viewport(
        self,
        viewport: arcade.Rect,
        bounds: arcade.Rect,
        label: str,
    ) -> None:
        """Draw status chip in viewport.

        Parameters
        ----------
        viewport
            Rectangle defining the relevant UI area.
        bounds
            Rectangle defining the relevant UI area.
        label
            Text displayed by the UI.
        """
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
        """Draw inspector section label.

        Parameters
        ----------
        viewport
            Rectangle defining the relevant UI area.
        key
            Stable identifier used by the UI.
        label
            Text displayed by the UI.
        x
            Logical screen coordinate.
        y
            Logical screen coordinate.
        """
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
        """Draw inspector stat tile in viewport.

        Parameters
        ----------
        viewport
            Rectangle defining the relevant UI area.
        key
            Stable identifier used by the UI.
        label
            Text displayed by the UI.
        value
            Value used by the operation.
        bounds
            Rectangle defining the relevant UI area.
        """
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
    ) -> float:
        """Draw metric row in viewport.

        Parameters
        ----------
        viewport
            Rectangle defining the relevant UI area.
        key
            Stable identifier used by the UI.
        label
            Text displayed by the UI.
        value
            Value used by the operation.
        x
            Logical screen coordinate.
        y
            Logical screen coordinate.
        width
            Requested logical size.
        value_color
            Value used by the operation.

        Returns
        -------
        float
            Vertical space consumed by the responsive row.
        """
        row_height = self._metric_row_layout(
            label,
            value,
            x,
            y,
            width,
        )[-1]
        row_bounds = arcade.LBWH(
            x,
            y - row_height,
            width,
            row_height,
        )
        if self._rect_intersects(row_bounds, viewport):
            return self._draw_metric_row(
                key,
                label,
                value,
                x,
                y,
                width,
                value_color=value_color,
            )
        return row_height
    def _draw_compact_value(
        self,
        key: str,
        label: str,
        value: str,
        x: float,
        y: float,
        width: float,
    ) -> None:
        """Draw compact value.

        Parameters
        ----------
        key
            Stable identifier used by the UI.
        label
            Text displayed by the UI.
        value
            Value used by the operation.
        x
            Logical screen coordinate.
        y
            Logical screen coordinate.
        width
            Requested logical size.
        """
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
    def _format_decimal(self, value: float) -> str:
        """Format decimal.

        Parameters
        ----------
        value
            Value used by the operation.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        if abs(value) >= 1000.0:
            return f"{value:.1f}"
        if abs(value) >= 100.0:
            return f"{value:.2f}"
        return f"{value:.2f}".rstrip("0").rstrip(".")
    def _format_mutation_delta(self, mutation_delta: object | None) -> str:
        """Format mutation delta.

        Parameters
        ----------
        mutation_delta
            Value used by the operation.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        if mutation_delta is None:
            return "None"

        vision_range = getattr(mutation_delta, "vision_range", 0.0)
        vision_angle = getattr(mutation_delta, "vision_angle", 0.0)
        radius = getattr(mutation_delta, "radius", 0.0)
        movement_cost = getattr(mutation_delta, "movement_cost_multiplier", 0.0)
        separation = getattr(mutation_delta, "separation_gene", 0.0)
        alignment = getattr(mutation_delta, "alignment_gene", 0.0)
        cohesion = getattr(mutation_delta, "cohesion_gene", 0.0)
        stomach_capacity = getattr(mutation_delta, "stomach_capacity", 0.0)
        digestion_rate = getattr(mutation_delta, "digestion_rate", 0.0)
        digestion_efficiency = getattr(
            mutation_delta,
            "digestion_efficiency",
            0.0,
        )
        digestive_delta = (
            f"D {stomach_capacity:+.2f}/{digestion_rate:+.2f}/"
            f"{digestion_efficiency:+.2f}, "
            if any(
                hasattr(mutation_delta, name)
                for name in (
                    "stomach_capacity",
                    "digestion_rate",
                    "digestion_efficiency",
                )
            )
            else ""
        )
        return (
            f"R {radius:+.1f}, V {vision_range:+.1f}/"
            f"{vision_angle:+.2f}, M {movement_cost:+.2f}, "
            f"{digestive_delta}"
            f"F {separation:+.2f}/{alignment:+.2f}/{cohesion:+.2f}"
        )
    def _format_genome_fitness(self, fitness: object) -> str:
        """Format genome fitness.

        Parameters
        ----------
        fitness
            Value used by the operation.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        if fitness is None:
            return "None"
        try:
            return f"{float(fitness):.2f}"
        except (TypeError, ValueError):
            return str(fitness)
    def _selected_fitness_label(self, world: World, selected: object) -> str:
        """Return selected fitness label.

        Parameters
        ----------
        world
            Simulation world providing current state.
        selected
            Value used by the operation.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        fitness = world.fitness_for(selected)
        if fitness is None:
            return "None"
        return self._format_genome_fitness(fitness.score(selected))
