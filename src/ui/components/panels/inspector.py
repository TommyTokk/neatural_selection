from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil, cos, floor, hypot, isfinite, log1p, log10, pi, sin
from pathlib import Path
import re
from types import SimpleNamespace

import arcade

from configs.sim_config import SimConfig
from src.creature.action import ACTION_OUTPUT_NAMES
from src.analysis import (
    BEHAVIOR_RADAR_LABELS,
    InspectorReport,
    calculate_behavior_scores,
    generate_inspector_report,
    generate_radar_chart_image,
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


@dataclass(frozen=True, slots=True)
class _InspectorCardField:
    """One vertically stacked, fully labelled inspector value."""

    key: str
    label: str
    value: str
    detail: str | None = None
    detail_tone: str = "muted"
    progress_ratio: float | None = None
    progress_color: arcade.Color | tuple[int, ...] | None = None
    value_color: arcade.Color | tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class _InspectorCardSection:
    """A measured group of stacked inspector fields."""

    key: str
    title: str
    fields: tuple[_InspectorCardField, ...]

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
        """Draw the selected creature as a measured, scrollable ID card."""
        selected = world.selected_creature
        if selected is None:
            return

        sections, species_id, species_color = self._inspector_card_sections(
            world,
            selected,
        )
        horizontal_padding = 16.0
        content_left = viewport.left + horizontal_padding
        content_width = max(24.0, viewport.width - horizontal_padding * 2.0)
        header_height = 86.0 if species_id is not None else 62.0
        section_gap = 14.0
        action_gap = 10.0
        action_height = 40.0
        actions_height = action_height * 3.0 + action_gap * 2.0
        section_heights = tuple(
            self._inspector_card_section_height(section, content_width)
            for section in sections
        )
        total_height = (
            12.0
            + header_height
            + section_gap
            + sum(section_heights)
            + section_gap * len(sections)
            + 18.0
            + actions_height
            + 16.0
        )
        scroll_limit = max(0.0, total_height - viewport.height)
        scroll_offset = max(
            0.0,
            min(scroll_limit, self._scroll_offsets.get("inspector", 0.0)),
        )
        self._inspector_content_height = total_height
        self._scroll_offsets["inspector"] = scroll_offset
        self._scroll_limits["inspector"] = scroll_limit
        self._scroll_regions["inspector"] = viewport

        cursor = viewport.top - 12.0 + scroll_offset
        self._draw_inspector_identity_header(
            viewport,
            selected,
            species_id,
            species_color,
            content_left,
            cursor,
            content_width,
        )
        cursor -= header_height + section_gap

        for section, height in zip(sections, section_heights):
            bounds = arcade.LBWH(
                content_left,
                cursor - height,
                content_width,
                height,
            )
            self._draw_inspector_card_section(viewport, section, bounds)
            cursor -= height + section_gap

        cursor -= 18.0
        brain_button = arcade.LBWH(
            content_left,
            cursor - action_height,
            content_width,
            action_height,
        )
        report_button = arcade.LBWH(
            content_left,
            brain_button.bottom - action_gap - action_height,
            content_width,
            action_height,
        )
        kill_button = arcade.LBWH(
            content_left,
            report_button.bottom - action_gap - action_height,
            content_width,
            action_height,
        )
        self._draw_inspector_actions(
            viewport,
            brain_button,
            report_button,
            kill_button,
        )
        if scroll_limit > 0.0:
            self._draw_scrollbar(viewport, scroll_offset, scroll_limit)

    def _inspector_card_sections(
        self,
        world: World,
        selected: object,
    ) -> tuple[
        tuple[_InspectorCardSection, ...],
        int | None,
        arcade.Color | tuple[int, ...],
    ]:
        """Build the complete, presentation-only creature ID-card model."""
        snapshot = self._cached_inspector_snapshot(world, selected)
        fitness = world.fitness_for(selected)
        creature_id = int(getattr(selected, "creature_id", 0))
        genome_id = world.neat_controller.genome_id_for(creature_id)
        lineage = getattr(selected, "lineage", None)
        parent_id = getattr(lineage, "parent_id", None)
        generation = int(getattr(lineage, "generation", 0))
        mutation_delta = getattr(lineage, "mutation_delta", None)
        species_id, species_color = self._selected_species_identity(
            world,
            selected,
        )

        physical_traits = getattr(selected, "physical_traits", None)
        radius = float(
            getattr(
                physical_traits,
                "radius",
                getattr(selected, "radius", 0.0),
            )
        )
        movement_cost = float(
            getattr(physical_traits, "movement_cost_multiplier", 1.0)
        )
        trait_config = getattr(world.config, "trait", None)
        stomach_capacity = max(
            0.0,
            float(
                getattr(
                    physical_traits,
                    "stomach_capacity",
                    radius
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
        flocking_traits = getattr(selected, "flocking_traits", None)

        energy_ratio = self._inspector_energy_ratio(world)
        stomach_ratio = max(
            0.0,
            min(1.0, float(getattr(snapshot, "stomach_fullness", 0.0))),
        )
        max_life = max(
            1e-12,
            float(getattr(world.config.metabolism, "max_life", 1.0)),
        )
        life_ratio = max(
            0.0,
            min(1.0, float(getattr(selected, "life", max_life)) / max_life),
        )
        stomach_energy = max(
            0.0,
            float(getattr(selected, "stomach_energy", 0.0)),
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
            ).get(creature_id, 0.0)
        )

        current_action = getattr(selected, "last_action", None)
        effective_herding = float(getattr(current_action, "herding", 0.0))
        panic = float(getattr(current_action, "flee_panic_intensity", 0.0))
        flock_runtime = getattr(world, "_last_flocking_runtime", {}).get(
            creature_id
        )
        raw_neural_herding = float(
            getattr(flock_runtime, "raw_neural_herding", effective_herding)
        )
        effective_herding = float(
            getattr(flock_runtime, "effective_herding", effective_herding)
        )
        observation = getattr(flock_runtime, "observation", None)
        intent = getattr(flock_runtime, "intent", None)
        weights = getattr(intent, "weights", None)

        flock_snapshot = getattr(snapshot, "flock", None)
        effective_flockmate_count = max(
            0.0,
            float(getattr(flock_snapshot, "flockmate_count", 0.0)),
        )
        normalized_flockmate_count = effective_flockmate_count / (
            effective_flockmate_count + 3.0
        )
        compatibility_config = getattr(
            getattr(world.config, "flocking", None),
            "compatibility",
            None,
        )
        raw_compatibility_mode = getattr(
            compatibility_config,
            "mode",
            "legacy",
        )
        compatibility_mode = str(
            getattr(raw_compatibility_mode, "value", raw_compatibility_mode)
        )
        ledger = getattr(selected, "ledger_diagnostics", None)
        activity = getattr(ledger, "activity", None)
        net_energy_balance = (
            fitness.net_energy_balance if fitness is not None else None
        )

        def trait_field(
            key: str,
            label: str,
            value: str,
            delta_name: str,
            delta_format: str,
            *,
            delta_scale: float = 1.0,
            delta_suffix: str = "",
        ) -> _InspectorCardField:
            detail, tone = self._inspector_trait_delta_detail(
                mutation_delta,
                parent_id,
                delta_name,
                delta_format,
                delta_scale=delta_scale,
                delta_suffix=delta_suffix,
            )
            return _InspectorCardField(
                key,
                label,
                value,
                detail,
                tone,
            )

        def runtime_value(value: object | None, formatter: str) -> str:
            if value is None:
                return "Recomputing"
            return format(float(value), formatter)

        identity = _InspectorCardSection(
            "identity",
            "IDENTITY & LINEAGE",
            (
                _InspectorCardField(
                    "inspector_creature_id",
                    "Creature ID",
                    f"#{creature_id}",
                ),
                _InspectorCardField(
                    "inspector_species_identity",
                    "Species",
                    "Unassigned" if species_id is None else f"Species #{species_id}",
                ),
                _InspectorCardField(
                    "inspector_parent",
                    "Parent creature",
                    "None (founder)" if parent_id is None else f"#{parent_id}",
                ),
                _InspectorCardField(
                    "inspector_generation",
                    "Generation",
                    str(generation),
                ),
                _InspectorCardField(
                    "inspector_neat_genome",
                    "NEAT genome ID",
                    "None" if genome_id is None else f"#{genome_id}",
                ),
            ),
        )
        vital = _InspectorCardSection(
            "vital",
            "VITAL STATUS",
            (
                _InspectorCardField(
                    "inspector_energy",
                    "Energy reserve",
                    f"{energy_ratio:.1%}",
                    progress_ratio=energy_ratio,
                    progress_color=self._inspector_energy_color(energy_ratio),
                    value_color=self._inspector_energy_color(energy_ratio),
                ),
                _InspectorCardField(
                    "inspector_stomach",
                    "Stomach fullness",
                    f"{stomach_ratio:.1%}",
                    progress_ratio=stomach_ratio,
                    progress_color=(236, 153, 45),
                    value_color=(236, 153, 45),
                ),
                _InspectorCardField(
                    "inspector_stomach_storage",
                    "Stored stomach energy",
                    f"{stomach_energy:.3f} energy",
                ),
                _InspectorCardField(
                    "inspector_life",
                    "Life reserve",
                    f"{life_ratio:.1%}",
                    progress_ratio=life_ratio,
                    progress_color=self.theme.accent,
                ),
            ),
        )
        inherited_anatomy = _InspectorCardSection(
            "inherited_anatomy",
            "NON-NEAT GENOME · ANATOMY",
            (
                trait_field(
                    "inspector_radius",
                    "Body radius",
                    f"{radius:.1f} px",
                    "radius",
                    "+.1f",
                    delta_suffix=" px",
                ),
                trait_field(
                    "inspector_movement_cost",
                    "Movement cost multiplier",
                    f"{movement_cost:.3f}×",
                    "movement_cost_multiplier",
                    "+.3f",
                    delta_suffix="×",
                ),
            ),
        )
        inherited_vision = _InspectorCardSection(
            "inherited_vision",
            "NON-NEAT GENOME · VISION",
            (
                trait_field(
                    "inspector_vision_range",
                    "Vision range",
                    f"{selected.vision.range:.1f} px",
                    "vision_range",
                    "+.1f",
                    delta_suffix=" px",
                ),
                trait_field(
                    "inspector_vision_angle",
                    "Vision angle",
                    f"{selected.vision.angle:.3f} rad",
                    "vision_angle",
                    "+.3f",
                    delta_suffix=" rad",
                ),
            ),
        )
        inherited_digestion = _InspectorCardSection(
            "inherited_digestion",
            "NON-NEAT GENOME · DIGESTION",
            (
                trait_field(
                    "inspector_stomach_capacity",
                    "Stomach capacity",
                    f"{stomach_capacity:.3f} energy",
                    "stomach_capacity",
                    "+.3f",
                    delta_suffix=" energy",
                ),
                trait_field(
                    "inspector_digestion_rate",
                    "Digestion rate",
                    f"{digestion_rate:.3f} energy/s",
                    "digestion_rate",
                    "+.3f",
                    delta_suffix=" energy/s",
                ),
                trait_field(
                    "inspector_digestion_efficiency",
                    "Digestion efficiency",
                    f"{digestion_efficiency:.1%}",
                    "digestion_efficiency",
                    "+.1f",
                    delta_scale=100.0,
                    delta_suffix=" percentage points",
                ),
            ),
        )

        def flock_value(name: str) -> str:
            if flocking_traits is None:
                return "Unavailable"
            return f"{float(getattr(flocking_traits, name, 0.0)):.3f}"

        inherited_social = _InspectorCardSection(
            "inherited_social",
            "NON-NEAT GENOME · SOCIAL TRAITS",
            (
                trait_field(
                    "inspector_separation_gene",
                    "Separation gene",
                    flock_value("separation_gene"),
                    "separation_gene",
                    "+.3f",
                ),
                trait_field(
                    "inspector_alignment_gene",
                    "Alignment gene",
                    flock_value("alignment_gene"),
                    "alignment_gene",
                    "+.3f",
                ),
                trait_field(
                    "inspector_cohesion_gene",
                    "Cohesion gene",
                    flock_value("cohesion_gene"),
                    "cohesion_gene",
                    "+.3f",
                ),
                trait_field(
                    "inspector_social_tag_x",
                    "Social identity tag X",
                    flock_value("social_tag_x"),
                    "social_tag_x",
                    "+.3f",
                ),
                trait_field(
                    "inspector_social_tag_y",
                    "Social identity tag Y",
                    flock_value("social_tag_y"),
                    "social_tag_y",
                    "+.3f",
                ),
            ),
        )
        movement_fitness_fields = [
            _InspectorCardField(
                "inspector_speed",
                "Current speed",
                f"{float(getattr(selected, 'speed', 0.0)):.1f} px/s",
            ),
            _InspectorCardField(
                "inspector_heading",
                "Heading",
                f"{float(getattr(selected, 'heading', 0.0)):.3f} rad",
            ),
            _InspectorCardField(
                "inspector_net_energy",
                "Net energy balance",
                "Unavailable"
                if net_energy_balance is None
                else self._format_decimal(net_energy_balance),
            ),
            _InspectorCardField(
                "inspector_flocking_benchmark",
                "Flocking benchmark fitness",
                "Unavailable"
                if fitness is None
                else f"{fitness.flocking_benchmark_reward:.3f}",
            ),
            _InspectorCardField(
                "inspector_collision_avoidance",
                "Collision avoidance",
                "Universal and automatic",
            ),
        ]
        if fitness is not None:
            movement_fitness_fields.append(
                _InspectorCardField(
                    "inspector_age",
                    "Age",
                    f"{fitness.age_seconds:.1f} s",
                )
            )
        movement_fitness = _InspectorCardSection(
            "movement_fitness",
            "MOVEMENT & FITNESS",
            tuple(movement_fitness_fields),
        )
        perception = _InspectorCardSection(
            "perception",
            "PERCEPTION",
            (
                _InspectorCardField(
                    "inspector_vision_cost",
                    "Vision energy cost",
                    f"{world.vision.energy_cost_per_second(selected):.3f} energy/s",
                ),
                _InspectorCardField(
                    "inspector_food_visible",
                    "Visible food items",
                    f"{snapshot.food.visible:.0f}",
                ),
                _InspectorCardField(
                    "inspector_food_density",
                    "Food density",
                    f"{snapshot.food.density:.2f}",
                ),
                _InspectorCardField(
                    "inspector_creatures_visible",
                    "Visible creatures",
                    f"{snapshot.creatures.visible:.0f}",
                ),
                _InspectorCardField(
                    "inspector_creature_density",
                    "Creature density",
                    f"{snapshot.creatures.density:.2f}",
                ),
                _InspectorCardField(
                    "inspector_flockmate_count",
                    "Effective flockmate count",
                    f"{effective_flockmate_count:.2f}",
                ),
                _InspectorCardField(
                    "inspector_normalized_flockmate_count",
                    "Normalized flockmate count",
                    f"{normalized_flockmate_count:.2f}",
                ),
            ),
        )
        social_fields = [
            _InspectorCardField(
                "inspector_social_compatibility",
                "Compatibility mode",
                compatibility_mode,
            ),
            _InspectorCardField(
                "inspector_raw_herding",
                "Raw neural herding",
                f"{raw_neural_herding:.2f}",
            ),
            _InspectorCardField(
                "inspector_herding",
                "Effective herding",
                f"{effective_herding:.2f}",
            ),
            _InspectorCardField(
                "inspector_personal_presence",
                "Personal-space presence",
                runtime_value(
                    getattr(observation, "personal_space_presence", None),
                    ".2f",
                ),
            ),
            _InspectorCardField(
                "inspector_social_presence",
                "Social presence",
                runtime_value(
                    getattr(observation, "social_presence", None),
                    ".2f",
                ),
            ),
            _InspectorCardField(
                "inspector_effective_separation",
                "Effective separation weight",
                runtime_value(getattr(weights, "separation", None), ".2f"),
            ),
            _InspectorCardField(
                "inspector_effective_alignment",
                "Effective alignment weight",
                runtime_value(getattr(weights, "alignment", None), ".2f"),
            ),
            _InspectorCardField(
                "inspector_effective_cohesion",
                "Effective cohesion weight",
                runtime_value(getattr(weights, "cohesion", None), ".2f"),
            ),
            _InspectorCardField(
                "inspector_social_engagement",
                "Social engagement",
                (
                    f"{effective_herding:.2f}"
                    if weights is None
                    else f"{weights.engagement:.2f}"
                ),
            ),
            _InspectorCardField(
                "inspector_panic_attenuation",
                "Panic attenuation",
                (
                    f"{1.0 - panic:.2f}"
                    if weights is None
                    else f"{weights.panic_attenuation:.2f}"
                ),
            ),
        ]
        if flock_runtime is None:
            social_fields.extend(
                (
                    _InspectorCardField(
                        "inspector_neural_desired_speed",
                        "Neural desired speed",
                        "Recomputing",
                    ),
                    _InspectorCardField(
                        "inspector_social_desired_speed",
                        "Social desired speed",
                        "Recomputing",
                    ),
                    _InspectorCardField(
                        "inspector_blended_desired_speed",
                        "Blended desired speed",
                        "Recomputing",
                    ),
                    _InspectorCardField(
                        "inspector_requested_social_force",
                        "Requested social contribution",
                        "Recomputing",
                    ),
                    _InspectorCardField(
                        "inspector_accepted_social_force",
                        "Accepted counterfactual contribution",
                        "Recomputing",
                    ),
                    _InspectorCardField(
                        "inspector_social_blend",
                        "Social influence",
                        "Recomputing",
                    ),
                    _InspectorCardField(
                        "inspector_local_group",
                        "Persistent local group",
                        "None",
                    ),
                )
            )
        else:
            social_fields.extend(
                (
                    _InspectorCardField(
                        "inspector_neural_desired_speed",
                        "Neural desired speed",
                        f"{hypot(*flock_runtime.neural_desired_velocity):.1f} px/s",
                    ),
                    _InspectorCardField(
                        "inspector_social_desired_speed",
                        "Social desired speed",
                        (
                            "Recomputing"
                            if intent is None
                            else f"{hypot(*intent.desired_velocity):.1f} px/s"
                        ),
                    ),
                    _InspectorCardField(
                        "inspector_blended_desired_speed",
                        "Blended desired speed",
                        f"{hypot(*flock_runtime.blended_desired_velocity):.1f} px/s",
                    ),
                    _InspectorCardField(
                        "inspector_requested_social_force",
                        "Requested social contribution",
                        f"{hypot(*flock_runtime.requested_social_contribution):.2f}",
                    ),
                    _InspectorCardField(
                        "inspector_accepted_social_force",
                        "Accepted counterfactual contribution",
                        f"{hypot(*flock_runtime.accepted_counterfactual_delta):.2f}",
                    ),
                    _InspectorCardField(
                        "inspector_social_blend",
                        "Social influence",
                        f"{flock_runtime.social_influence:.1%}",
                    ),
                    _InspectorCardField(
                        "inspector_local_group",
                        "Persistent local group",
                        (
                            "None"
                            if flock_runtime.local_group_id is None
                            else f"Group #{flock_runtime.local_group_id}"
                        ),
                        (
                            None
                            if flock_runtime.local_group_id is None
                            else f"{flock_runtime.local_group_size} members"
                        ),
                    ),
                )
            )
        social_runtime = _InspectorCardSection(
            "social_runtime",
            "SOCIAL & FLOCKING RUNTIME",
            tuple(social_fields),
        )
        rest_ledger = _InspectorCardSection(
            "rest_ledger",
            "REST, METABOLISM & LEDGER",
            (
                _InspectorCardField(
                    "inspector_digestive_upkeep",
                    "Digestive upkeep",
                    f"{digestive_upkeep:.4f} energy/s",
                ),
                _InspectorCardField(
                    "inspector_processing_cost",
                    "Recent digestion processing cost",
                    f"{processing_cost:.4f} energy/s",
                ),
                _InspectorCardField(
                    "inspector_rest_intent",
                    "Rest intent",
                    f"{float(getattr(selected, 'rest_intent', 0.0)):.2f}",
                ),
                _InspectorCardField(
                    "inspector_smoothed_rest",
                    "Smoothed rest",
                    f"{float(getattr(selected, 'smoothed_rest', 0.0)):.2f}",
                ),
                _InspectorCardField(
                    "inspector_effective_rest",
                    "Effective rest",
                    f"{float(getattr(selected, 'effective_rest', 0.0)):.2f}",
                ),
                _InspectorCardField(
                    "inspector_activity_motor",
                    "Voluntary motor effort",
                    f"{float(getattr(activity, 'voluntary_motor_effort', 0.0)):.2f}",
                ),
                _InspectorCardField(
                    "inspector_activity_speed",
                    "Normalized speed activity",
                    f"{float(getattr(activity, 'normalized_speed', 0.0)):.2f}",
                ),
                _InspectorCardField(
                    "inspector_activity_turn",
                    "Turning activity",
                    f"{float(getattr(activity, 'turn', 0.0)):.2f}",
                ),
                _InspectorCardField(
                    "inspector_activity_communication",
                    "Communication activity",
                    f"{float(getattr(activity, 'communication', 0.0)):.2f}",
                ),
                _InspectorCardField(
                    "inspector_activity_reproduction",
                    "Reproduction activity",
                    f"{float(getattr(activity, 'reproduction', 0.0)):.0f}",
                ),
                _InspectorCardField(
                    "inspector_activity_nursing",
                    "Nursing activity",
                    f"{float(getattr(activity, 'nursing', 0.0)):.0f}",
                ),
                _InspectorCardField(
                    "inspector_activity_total",
                    "Weighted activity",
                    f"{float(getattr(selected, 'activity', 0.0)):.3f}",
                ),
                _InspectorCardField(
                    "inspector_stomach_consumed",
                    "Stomach energy consumed",
                    f"{float(getattr(ledger, 'stomach_consumed', 0.0)):.4f}",
                ),
                _InspectorCardField(
                    "inspector_gross_energy",
                    "Gross digested energy",
                    f"{float(getattr(ledger, 'gross_energy', 0.0)):.4f}",
                ),
                _InspectorCardField(
                    "inspector_net_energy",
                    "Net digested energy",
                    f"{float(getattr(ledger, 'net_energy', 0.0)):.4f}",
                ),
                _InspectorCardField(
                    "inspector_energy_demand",
                    "Total energy demand",
                    f"{float(getattr(ledger, 'total_energy_demand', 0.0)):.4f}",
                ),
                _InspectorCardField(
                    "inspector_energy_deficit",
                    "Unmet energy demand",
                    f"{float(getattr(ledger, 'unmet_energy_demand', 0.0)):.4f}",
                ),
                _InspectorCardField(
                    "inspector_rest_recovery",
                    "Energy recovered through rest",
                    f"{float(getattr(ledger, 'rest_energy_recovered', 0.0)):.4f}",
                ),
                _InspectorCardField(
                    "inspector_healing_spend",
                    "Energy spent on healing",
                    f"{float(getattr(ledger, 'healing_energy_spent', 0.0)):.4f}",
                ),
                _InspectorCardField(
                    "inspector_deficit_damage",
                    "Life damage from energy deficit",
                    f"{float(getattr(ledger, 'life_damage_from_deficit', 0.0)):.4f}",
                ),
                _InspectorCardField(
                    "inspector_direct_damage",
                    "Direct life damage",
                    f"{float(getattr(ledger, 'direct_life_damage', 0.0)):.4f}",
                ),
                _InspectorCardField(
                    "inspector_life_damage",
                    "Life healed",
                    f"{float(getattr(ledger, 'life_healed', 0.0)):.4f}",
                ),
                _InspectorCardField(
                    "inspector_transaction_status",
                    "Final transaction status",
                    str(getattr(ledger, "transaction_status", "not_evaluated")),
                ),
            ),
        )
        return (
            (
                identity,
                vital,
                inherited_anatomy,
                inherited_vision,
                inherited_digestion,
                inherited_social,
                movement_fitness,
                perception,
                social_runtime,
                rest_ledger,
            ),
            species_id,
            species_color,
        )

    @staticmethod
    def _inspector_trait_delta_detail(
        mutation_delta: object | None,
        parent_id: int | None,
        attribute: str,
        number_format: str,
        *,
        delta_scale: float = 1.0,
        delta_suffix: str = "",
    ) -> tuple[str, str]:
        """Format one explicit parent delta without compact abbreviations."""
        if parent_id is None:
            return "Change from parent: not applicable (founder)", "muted"
        if mutation_delta is None or not hasattr(mutation_delta, attribute):
            return "Change from parent: unavailable", "muted"
        try:
            delta = float(getattr(mutation_delta, attribute)) * delta_scale
        except (TypeError, ValueError):
            return "Change from parent: unavailable", "muted"
        if not isfinite(delta):
            return "Change from parent: unavailable", "muted"
        tone = "positive" if delta > 0.0 else "negative" if delta < 0.0 else "muted"
        return (
            f"Change from parent: {format(delta, number_format)}{delta_suffix}",
            tone,
        )

    def _inspector_card_field_height(
        self,
        field: _InspectorCardField,
        width: float,
    ) -> float:
        """Measure one nested field, including wrapping and progress bars."""
        text_width = max(24.0, width - 22.0)
        label_lines = self._wrap_line(field.label, text_width, font_size=9.0)
        value_lines = self._wrap_line(field.value, text_width, font_size=12.0)
        height = (
            11.0
            + max(1, len(label_lines)) * 12.0
            + 4.0
            + max(1, len(value_lines)) * 15.0
        )
        if field.detail is not None:
            detail_lines = self._wrap_line(
                field.detail,
                text_width,
                font_size=10.0,
            )
            height += 4.0 + max(1, len(detail_lines)) * 13.0
        if field.progress_ratio is not None:
            height += 16.0
        return height + 11.0

    def _inspector_card_section_height(
        self,
        section: _InspectorCardSection,
        width: float,
    ) -> float:
        """Measure a complete section before scroll limits are calculated."""
        field_width = max(24.0, width - 28.0)
        field_heights = (
            self._inspector_card_field_height(field, field_width)
            for field in section.fields
        )
        return 42.0 + sum(field_heights) + max(0, len(section.fields) - 1) * 8.0 + 14.0

    def _draw_inspector_identity_header(
        self,
        viewport: arcade.Rect,
        selected: object,
        species_id: int | None,
        species_color: arcade.Color | tuple[int, ...],
        left: float,
        top: float,
        width: float,
    ) -> None:
        """Draw the padded name, status, and species marker."""
        self._draw_text_in_viewport(
            viewport,
            "inspector_label",
            "SELECTED CREATURE",
            left,
            top - 10.0,
            self.theme.text_muted,
            9,
            bold=True,
        )
        self._draw_status_chip_in_viewport(
            viewport,
            arcade.LBWH(left + width - 52.0, top - 27.0, 52.0, 24.0),
            "LIVE",
        )
        self._draw_text_in_viewport(
            viewport,
            "inspector_name",
            self._fit_line(str(getattr(selected, "name", "Creature")), width),
            left,
            top - 39.0,
            self.theme.text_primary,
            17,
            bold=True,
        )
        if species_id is None:
            return
        marker_x = left + 8.0
        marker_y = top - 72.0
        marker_bounds = arcade.LBWH(marker_x - 8.0, marker_y - 8.0, 16.0, 16.0)
        if self._rect_intersects(marker_bounds, viewport):
            arcade.draw_circle_filled(marker_x, marker_y, 8.0, species_color)
            arcade.draw_circle_outline(
                marker_x,
                marker_y,
                8.0,
                self.theme.selected_outline,
                2.5,
            )
        self._draw_text_in_viewport(
            viewport,
            "inspector_species",
            f"Species #{species_id}",
            left + 24.0,
            top - 77.0,
            self.theme.text_muted,
            11,
            bold=True,
        )

    def _draw_inspector_card_section(
        self,
        viewport: arcade.Rect,
        section: _InspectorCardSection,
        bounds: arcade.Rect,
    ) -> None:
        """Draw one padded section and its vertically stacked field cards."""
        if self._rect_intersects(bounds, viewport):
            self._draw_rounded_rect(
                bounds,
                self.theme.panel_background,
                self.theme.panel_border,
                9.0,
                1.0,
            )
            self._draw_text(
                f"inspector_section_{section.key}",
                section.title,
                bounds.left + 14.0,
                bounds.top - 14.0,
                self.theme.text_muted,
                9,
                bold=True,
                anchor_y="top",
                width=max(24.0, bounds.width - 28.0),
            )
        field_width = max(24.0, bounds.width - 28.0)
        cursor = bounds.top - 42.0
        for field in section.fields:
            height = self._inspector_card_field_height(field, field_width)
            field_bounds = arcade.LBWH(
                bounds.left + 14.0,
                cursor - height,
                field_width,
                height,
            )
            self._draw_inspector_card_field(viewport, field, field_bounds)
            cursor -= height + 8.0

    def _draw_inspector_card_field(
        self,
        viewport: arcade.Rect,
        field: _InspectorCardField,
        bounds: arcade.Rect,
    ) -> None:
        """Draw a field with guaranteed padding on every text edge."""
        if not self._rect_intersects(bounds, viewport):
            return
        self._draw_rounded_rect(
            bounds,
            self.theme.card_background,
            self.theme.panel_border,
            7.0,
            0.8,
        )
        text_left = bounds.left + 11.0
        text_width = max(24.0, bounds.width - 22.0)
        label_lines = self._wrap_line(field.label, text_width, font_size=9.0)
        value_lines = self._wrap_line(field.value, text_width, font_size=12.0)
        cursor = bounds.top - 11.0
        self._draw_text(
            f"{field.key}_label",
            "\n".join(label_lines),
            text_left,
            cursor,
            self.theme.text_muted,
            9,
            width=text_width,
            multiline=True,
            anchor_y="top",
        )
        cursor -= max(1, len(label_lines)) * 12.0 + 4.0
        self._draw_text(
            f"{field.key}_value",
            "\n".join(value_lines),
            text_left,
            cursor,
            field.value_color or self.theme.text_primary,
            12,
            bold=True,
            width=text_width,
            multiline=True,
            anchor_y="top",
        )
        cursor -= max(1, len(value_lines)) * 15.0
        if field.detail is not None:
            detail_lines = self._wrap_line(
                field.detail,
                text_width,
                font_size=10.0,
            )
            cursor -= 4.0
            self._draw_text(
                f"{field.key}_detail",
                "\n".join(detail_lines),
                text_left,
                cursor,
                self._inspector_detail_tone_color(field.detail_tone),
                10,
                width=text_width,
                multiline=True,
                anchor_y="top",
            )
            cursor -= max(1, len(detail_lines)) * 13.0
        if field.progress_ratio is not None:
            bar = arcade.LBWH(text_left, cursor - 12.0, text_width, 8.0)
            self._draw_progress_bar(
                bar,
                field.progress_ratio,
                fill_color=field.progress_color,
            )

    def _inspector_detail_tone_color(
        self,
        tone: str,
    ) -> arcade.Color | tuple[int, ...]:
        """Resolve mutation-detail emphasis without changing value colours."""
        if tone == "positive":
            return self.theme.accent
        if tone == "negative":
            return self.theme.selected_outline
        return self.theme.text_muted

    def _draw_inspector_actions(
        self,
        viewport: arcade.Rect,
        brain_button: arcade.Rect,
        report_button: arcade.Rect,
        kill_button: arcade.Rect,
    ) -> None:
        """Draw and register only the action buttons currently in view."""
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
            self._control_hitboxes["open_behavior_report_selected"] = report_button
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
                    world.config.population.birth_cooldown_seconds
                    - (selected.age_seconds - selected.last_birth_time),
                )
                lines.extend(
                    [
                        f"Genome: {genome_id if genome_id is not None else 'None'}",
                        f"Net energy: {fitness.net_energy_balance:.3f}",
                        f"Net metabolic rate: {fitness.net_metabolic_rate:.4f}/s",
                        f"Age: {selected.age_seconds:.1f}s",
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
        return self._format_genome_fitness(fitness.net_energy_balance)
