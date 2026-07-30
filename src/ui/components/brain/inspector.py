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
from src.behavior_observer import (
    BehaviorKind,
    BehaviorObserverDiagnostics,
    BehaviorStateSnapshot,
    BoutStatus,
)
from src.counterfactual_neat import (
    BEHAVIOR_EXPLANATION_SPECS,
    MINIMAL_INFLUENCE_THRESHOLD,
    MODERATE_INFLUENCE_THRESHOLD,
    WEAK_INFLUENCE_THRESHOLD,
    CounterfactualDiagnostics,
    SemanticEffectSnapshot,
    SemanticIntervention,
    WhySnapshot,
)

_EMPTY_NEAT_NODE_LABELS: dict[int, str] = {}

class BrainInspectorComponent:
    """Group related behavior extracted from ``UiRenderer``."""

    BRAIN_NODE_SUMMARY_MIN_HEIGHT = 56.0
    BRAIN_NODE_SUMMARY_LINE_HEIGHT = 17.0
    BRAIN_NODE_SUMMARY_VERTICAL_PADDING = 20.0
    BRAIN_BEHAVIOR_CARD_HEIGHT = 88.0
    BRAIN_BEHAVIOR_CARD_GAP = 10.0
    BRAIN_BEHAVIOR_HEADER_HEIGHT = 56.0
    BRAIN_BEHAVIOR_NOTICE_HEIGHT = 82.0
    BRAIN_BEHAVIOR_NOTICE_LINE_HEIGHT = 15.0
    BRAIN_BEHAVIOR_DIAGNOSTICS_HEIGHT = 138.0
    BRAIN_BEHAVIOR_DETAIL_LINE_HEIGHT = 17.0
    BRAIN_WHY_HEADER_HEIGHT = BRAIN_BEHAVIOR_HEADER_HEIGHT
    BRAIN_WHY_NOTICE_HEIGHT = BRAIN_BEHAVIOR_NOTICE_HEIGHT
    BRAIN_WHY_CARD_HEIGHT = BRAIN_BEHAVIOR_CARD_HEIGHT
    BRAIN_WHY_CARD_GAP = BRAIN_BEHAVIOR_CARD_GAP
    BRAIN_WHY_EFFECT_GAP = 8.0
    BRAIN_WHY_DETAIL_PADDING = 20.0
    BRAIN_WHY_CALCULATION_LINE_HEIGHT = 15.0
    BRAIN_BEHAVIOR_ACCENTS = {
        BehaviorKind.FOOD_ORIENTATION: (58, 125, 225),
        BehaviorKind.FOOD_APPROACH: (17, 158, 145),
        BehaviorKind.FEEDING: (35, 168, 89),
        BehaviorKind.RESTING: (132, 91, 205),
        BehaviorKind.COHESION: (215, 72, 128),
        BehaviorKind.ALARM_RETREAT: (228, 91, 49),
    }
    BRAIN_BEHAVIOR_ACTIVATION_COPY = {
        BehaviorKind.FOOD_ORIENTATION: (
            "The same visible food target persists, heading error decreases, "
            "and the creature turns toward it."
        ),
        BehaviorKind.FOOD_APPROACH: (
            "The same visible food target gets consistently closer while "
            "realized movement points toward it."
        ),
        BehaviorKind.FEEDING: (
            "An explicit food-consumption event occurs and swallowed energy "
            "increases. Proximity or intent alone is not enough."
        ),
        BehaviorKind.RESTING: (
            "Current realized speed is low and most recent samples remain "
            "below the configured rest-speed threshold."
        ),
        BehaviorKind.COHESION: (
            "A compatible group remains visible outside separation range, "
            "then the creature approaches its center or aligns its velocity."
        ),
        BehaviorKind.ALARM_RETREAT: (
            "Local alarm is present and falls ahead and over time while the "
            "creature moves forward down the alarm gradient."
        ),
    }

    def _draw_brain_side_inspector(
        self,
        world: World,
        brain: object | None,
        layout: BrainGraphLayout | None,
        bounds: arcade.Rect,
    ) -> None:
        """Draw the switchable node/observed-behaviour side panel."""
        self._draw_rounded_rect(
            bounds,
            self.theme.panel_background,
            self.theme.panel_border,
            self.config.layout.card_radius,
            1.0,
        )
        selector_height = 42.0
        selector = arcade.LBWH(
            bounds.left + 8.0,
            bounds.top - selector_height - 6.0,
            bounds.width - 16.0,
            selector_height,
        )
        close_button = arcade.LBWH(
            selector.right - 26.0,
            selector.center_y - 11.0,
            22.0,
            22.0,
        )
        available_width = max(80.0, close_button.left - selector.left - 8.0)
        gap = 5.0
        tab_width = max(36.0, (available_width - 2.0 * gap) / 3.0)
        node_tab = arcade.LBWH(
            selector.left,
            selector.bottom + 5.0,
            tab_width,
            selector.height - 10.0,
        )
        behavior_tab = arcade.LBWH(
            node_tab.right + gap,
            node_tab.bottom,
            tab_width,
            node_tab.height,
        )
        why_tab = arcade.LBWH(
            behavior_tab.right + gap,
            behavior_tab.bottom,
            tab_width,
            behavior_tab.height,
        )
        self._control_hitboxes["brain_inspector_page_node"] = node_tab
        self._control_hitboxes["brain_inspector_page_behaviors"] = behavior_tab
        self._control_hitboxes["brain_inspector_page_why"] = why_tab
        self._control_hitboxes["brain_node_inspector_toggle"] = close_button
        self._draw_brain_inspector_tab(
            node_tab,
            "NODE",
            "node",
            self._brain_inspector_page == "node",
        )
        self._draw_brain_inspector_tab(
            behavior_tab,
            "BEHAVIOURS",
            "behaviors",
            self._brain_inspector_page == "behaviors",
        )
        self._draw_brain_inspector_tab(
            why_tab,
            "WHY",
            "why",
            self._brain_inspector_page == "why",
        )
        self._draw_panel_close_button(
            close_button,
            "brain_node_inspector",
        )
        content = arcade.LBWH(
            bounds.left + 8.0,
            bounds.bottom + 8.0,
            bounds.width - 16.0,
            max(1.0, selector.bottom - bounds.bottom - 14.0),
        )
        if self._brain_inspector_page == "behaviors":
            self._draw_brain_behavior_inspector(world, content)
        elif self._brain_inspector_page == "why":
            self._draw_brain_why_inspector(world, content)
        else:
            self._draw_brain_node_inspector(
                brain,
                layout,
                content,
                show_close=False,
            )

    def _draw_brain_why_inspector(
        self,
        world: World,
        bounds: arcade.Rect,
    ) -> None:
        """Draw stable, expandable counterfactual behavior cards."""
        self._draw_rounded_rect(
            bounds,
            self.theme.card_background,
            self.theme.panel_border,
            self.config.layout.card_radius,
            1.0,
        )
        why_config = getattr(
            getattr(world, "config", None),
            "counterfactual_why",
            None,
        )
        behavior_snapshot = getattr(
            world,
            "selected_behavior_snapshot",
            None,
        )
        snapshots = tuple(
            getattr(world, "selected_why_snapshots", ()) or ()
        )
        diagnostics = getattr(
            world,
            "counterfactual_diagnostics",
            CounterfactualDiagnostics(worker_health="unavailable"),
        )
        snapshot_by_behavior: dict[BehaviorKind, WhySnapshot] = {
            snapshot.behavior: snapshot for snapshot in snapshots
        }
        states_by_behavior: dict[BehaviorKind, BehaviorStateSnapshot] = (
            {}
            if behavior_snapshot is None
            else {
                state.behavior: state
                for state in behavior_snapshot.behaviors
            }
        )
        for behavior in BehaviorKind:
            self._control_hitboxes.pop(
                self._brain_why_card_hitbox_key(behavior),
                None,
            )
        expanded_behavior = next(
            (
                behavior
                for behavior in BehaviorKind
                if behavior.value == self._brain_expanded_why_behavior
            ),
            None,
        )

        status_title, status_message, status_is_error = (
            self._brain_why_status(
                world,
                why_config,
                behavior_snapshot,
                states_by_behavior,
                snapshots,
                diagnostics,
            )
        )
        status_lines = tuple(
            self._wrap_line(
                status_message,
                max(24.0, bounds.width - 66.0),
                font_size=10.0,
            )
        )[:2]
        diagnostics_visible = bool(
            getattr(world, "debug_vision_enabled", False)
        )
        diagnostics_height = 184.0 if diagnostics_visible else 0.0
        content = arcade.LBWH(
            bounds.left + 14.0,
            bounds.bottom + 14.0,
            bounds.width - 28.0,
            bounds.height - 28.0,
        )
        drawing_width = max(1.0, content.width - 12.0)
        calculation_lines = self._brain_why_calculation_lines(drawing_width)
        calculation_height = (
            44.0
            + len(calculation_lines)
            * self.BRAIN_WHY_CALCULATION_LINE_HEIGHT
        )
        cards_height = (
            len(BehaviorKind) * self.BRAIN_WHY_CARD_HEIGHT
            + (len(BehaviorKind) - 1) * self.BRAIN_WHY_CARD_GAP
            + (
                self._brain_why_detail_height(expanded_behavior)
                if expanded_behavior is not None
                else 0.0
            )
        )
        total_height = (
            self.BRAIN_WHY_HEADER_HEIGHT
            + calculation_height
            + self.BRAIN_WHY_CARD_GAP
            + self.BRAIN_WHY_NOTICE_HEIGHT
            + self.BRAIN_WHY_CARD_GAP
            + cards_height
            + (
                self.BRAIN_WHY_CARD_GAP + diagnostics_height
                if diagnostics_visible
                else 0.0
            )
        )
        scroll_key = "brain_why_inspector"
        self._scroll_offsets[scroll_key] = self._brain_why_scroll_offset
        scroll_limit = max(0.0, total_height - content.height)
        scroll_offset = max(
            0.0,
            min(scroll_limit, self._scroll_offsets.get(scroll_key, 0.0)),
        )
        self._scroll_offsets[scroll_key] = scroll_offset
        self._scroll_limits[scroll_key] = scroll_limit
        self._scroll_regions[scroll_key] = content
        self._brain_why_scroll_offset = scroll_offset
        cursor_top = content.top + scroll_offset

        with self._ui_clip(content):
            self._draw_text(
                "brain_why_title",
                "WHY THIS BEHAVIOUR?",
                content.left,
                cursor_top - 16.0,
                self.theme.text_primary,
                11.5,
                bold=True,
            )
            self._draw_text(
                "brain_why_subtitle",
                "Counterfactual effect on the current NEAT decision",
                content.left,
                cursor_top - 41.0,
                self.theme.text_muted,
                10.0,
            )
            cursor_top -= self.BRAIN_WHY_HEADER_HEIGHT

            calculation_bounds = arcade.LBWH(
                content.left,
                cursor_top - calculation_height,
                drawing_width,
                calculation_height,
            )
            self._draw_brain_why_calculation_section(
                calculation_bounds,
                calculation_lines,
            )
            cursor_top = (
                calculation_bounds.bottom - self.BRAIN_WHY_CARD_GAP
            )

            notice_bounds = arcade.LBWH(
                content.left,
                cursor_top - self.BRAIN_WHY_NOTICE_HEIGHT,
                drawing_width,
                self.BRAIN_WHY_NOTICE_HEIGHT,
            )
            self._draw_brain_behavior_notice(
                notice_bounds,
                1,
                status_title,
                status_lines,
                is_error=status_is_error,
            )
            cursor_top = notice_bounds.bottom - self.BRAIN_WHY_CARD_GAP

            for behavior_index, behavior in enumerate(BehaviorKind):
                expanded = behavior is expanded_behavior
                card_height = (
                    self.BRAIN_WHY_CARD_HEIGHT
                    + (
                        self._brain_why_detail_height(behavior)
                        if expanded
                        else 0.0
                    )
                )
                card_bounds = arcade.LBWH(
                    content.left,
                    cursor_top - card_height,
                    drawing_width,
                    card_height,
                )
                self._draw_brain_why_card(
                    card_bounds,
                    behavior,
                    states_by_behavior.get(behavior),
                    snapshot_by_behavior.get(behavior),
                    expanded=expanded,
                )
                visible_bottom = max(card_bounds.bottom, content.bottom)
                visible_top = min(card_bounds.top, content.top)
                if visible_top > visible_bottom:
                    self._control_hitboxes[
                        self._brain_why_card_hitbox_key(behavior)
                    ] = arcade.LBWH(
                        card_bounds.left,
                        visible_bottom,
                        card_bounds.width,
                        visible_top - visible_bottom,
                    )
                cursor_top = card_bounds.bottom
                if behavior_index < len(BehaviorKind) - 1:
                    cursor_top -= self.BRAIN_WHY_CARD_GAP

            if diagnostics_visible:
                cursor_top -= self.BRAIN_WHY_CARD_GAP
                self._draw_brain_why_diagnostics(
                    arcade.LBWH(
                        content.left,
                        cursor_top - diagnostics_height,
                        drawing_width,
                        diagnostics_height,
                    ),
                    diagnostics,
                )
        if scroll_limit > 0.0:
            self._draw_scrollbar(content, scroll_offset, scroll_limit)

    def _brain_why_status(
        self,
        world: World,
        why_config: object | None,
        behavior_snapshot: object | None,
        states_by_behavior: dict[BehaviorKind, BehaviorStateSnapshot],
        snapshots: tuple[WhySnapshot, ...],
        diagnostics: CounterfactualDiagnostics,
    ) -> tuple[str, str, bool]:
        """Return the stable notice shown above the WHY behavior cards."""
        if why_config is not None and not getattr(
            why_config,
            "enabled",
            False,
        ):
            return (
                "WHY disabled",
                "Enable counterfactual WHY in simulation configuration.",
                False,
            )
        if diagnostics.last_error:
            return ("WHY worker unavailable", diagnostics.last_error, True)
        if behavior_snapshot is None:
            return (
                "Collecting temporal evidence",
                "Cards remain fixed while the behavior observer initializes.",
                False,
            )
        mapped_states = tuple(
            behavior
            for behavior, state in states_by_behavior.items()
            if (
                behavior in BEHAVIOR_EXPLANATION_SPECS
                and (
                    behavior
                    not in {
                        BehaviorKind.FOOD_ORIENTATION,
                        BehaviorKind.FOOD_APPROACH,
                    }
                    or state.target_id is not None
                )
            )
        )
        if not mapped_states:
            return (
                "Waiting for a neural WHY bout",
                (
                    "No mapped emerging or active behavior currently requires "
                    "a counterfactual probe."
                ),
                False,
            )
        if not snapshots:
            return (
                "Calculating explanations",
                "Showing stable cards while the latest WHY probe completes.",
                False,
            )
        delayed = (
            not bool(getattr(world, "is_paused", False))
            and diagnostics.latest_result_age_ms is not None
            and diagnostics.latest_result_age_ms > 500.0
        )
        if delayed:
            return (
                "WHY updating",
                "Values are from the latest completed probe and may briefly lag.",
                False,
            )
        return (
            "Live counterfactual explanations",
            (
                f"{len(snapshots)} current behavior explanation"
                f"{'' if len(snapshots) == 1 else 's'}. "
                "Click a card to inspect fixed-position details."
            ),
            False,
        )

    def _draw_brain_why_calculation_section(
        self,
        bounds: arcade.Rect,
        lines: tuple[str, ...],
    ) -> None:
        """Explain the displayed counterfactual values and thresholds."""
        self._draw_rounded_rect(
            bounds,
            self.theme.panel_background,
            self._brain_blend_color(
                self.theme.panel_border,
                self.theme.accent,
                0.42,
            ),
            self.config.layout.card_radius,
            1.0,
        )
        self._draw_text(
            "brain_why_calculation_title",
            "HOW THE VALUES ARE CALCULATED",
            bounds.left + 14.0,
            bounds.top - 22.0,
            self.theme.accent,
            10.0,
            bold=True,
            anchor_y="center",
        )
        for line_index, line in enumerate(lines):
            self._draw_text(
                f"brain_why_calculation_{line_index}",
                line,
                bounds.left + 14.0,
                (
                    bounds.top
                    - 48.0
                    - line_index * self.BRAIN_WHY_CALCULATION_LINE_HEIGHT
                ),
                self.theme.text_muted,
                9.0,
                anchor_y="center",
            )

    def _brain_why_calculation_lines(
        self,
        width: float,
    ) -> tuple[str, ...]:
        """Return wrapped, scientifically explicit WHY calculation copy."""
        target_dead_zone = (
            self.config.counterfactual_why.target_center_dead_zone_radians
        )
        paragraphs = (
            (
                "Actual is the centered output from the completed live NEAT "
                "decision. Counterfactual uses the same frozen brain with one "
                "semantic sensor group replaced."
            ),
            (
                "Output influence = |actual − counterfactual| ÷ its natural "
                "output span. Behavior influence is the mean across scored "
                "outputs only; it is not a percentage allocation."
            ),
            (
                "For food orientation and approach, rotate influence still "
                "uses that raw delta, but its direction compares steering "
                "toward the same factual food heading. Within "
                f"{target_dead_zone:.2f} "
                "rad, a smaller turn "
                "is the better stabilizing response. Without a matching "
                "visible target, food WHY waits instead of using magnitude."
            ),
            (
                f"Labels: <{MINIMAL_INFLUENCE_THRESHOLD:.2f} minimal, "
                f"<{WEAK_INFLUENCE_THRESHOLD:.2f} weak, "
                f"<{MODERATE_INFLUENCE_THRESHOLD:.2f} moderate, otherwise "
                "strong."
            ),
            (
                "Supportive means the factual response weakens; suppressive "
                "means it strengthens; reversing crosses zero; mixed means "
                "scored outputs disagree."
            ),
            (
                "These are local mechanistic influences, not definitive "
                "biological causation, and they do not sum to 100%."
            ),
        )
        lines: list[str] = []
        for paragraph in paragraphs:
            lines.extend(
                self._wrap_line(
                    paragraph,
                    max(24.0, width - 28.0),
                    font_size=9.0,
                )
            )
        return tuple(lines)

    def _draw_brain_why_card(
        self,
        bounds: arcade.Rect,
        behavior: BehaviorKind,
        state: BehaviorStateSnapshot | None,
        snapshot: WhySnapshot | None,
        *,
        expanded: bool,
    ) -> None:
        """Draw one stable WHY card and its optional detail area."""
        accent = self.BRAIN_BEHAVIOR_ACCENTS[behavior]
        intensity = self._behavior_display_intensity(state)
        fill = self._brain_blend_color(
            self.theme.card_background,
            accent,
            0.04 + 0.18 * intensity,
        )
        border = self._brain_blend_color(
            self.theme.panel_border,
            accent,
            0.24 + 0.70 * intensity,
        )
        title_color = self._brain_blend_color(
            self.theme.text_muted,
            accent,
            0.38 + 0.62 * intensity,
        )
        self._draw_rounded_rect(
            bounds,
            fill,
            border,
            self.config.layout.card_radius,
            1.0 + intensity,
        )
        arcade.draw_circle_filled(
            bounds.left + 16.0,
            bounds.top - 21.0,
            6.0,
            self._brain_blend_color(
                self.theme.card_background,
                accent,
                0.28 + 0.72 * intensity,
            ),
        )
        self._draw_text(
            f"brain_why_{behavior.value}_name",
            self._behavior_display_name(behavior),
            bounds.left + 31.0,
            bounds.top - 21.0,
            title_color,
            13.0,
            bold=True,
            anchor_y="center",
        )
        self._draw_text(
            f"brain_why_{behavior.value}_expand",
            "−" if expanded else "+",
            bounds.right - 15.0,
            bounds.top - 21.0,
            title_color,
            15.0,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )
        state_text = "INACTIVE"
        if state is not None:
            state_text = (
                f"{state.status.value.upper()} · "
                f"{state.duration_seconds:.1f} s"
            )
        self._draw_text(
            f"brain_why_{behavior.value}_state",
            state_text,
            bounds.left + 16.0,
            bounds.top - 50.0,
            title_color,
            10.0,
            bold=state is not None,
            anchor_y="center",
        )
        strongest = (
            max(snapshot.effects, key=lambda effect: effect.influence_score)
            if snapshot is not None and snapshot.effects
            else None
        )
        summary = "NO DIRECT NEURAL WHY" if (
            behavior is BehaviorKind.RESTING
        ) else "WAITING"
        score = 0.0
        if strongest is not None:
            score = max(0.0, min(1.0, strongest.influence_score))
            summary = f"INFLUENCE {score:.2f}"
        elif (
            state is not None
            and behavior
            in {
                BehaviorKind.FOOD_ORIENTATION,
                BehaviorKind.FOOD_APPROACH,
            }
            and state.target_id is None
        ):
            summary = "WAITING FOR TARGET"
        elif state is not None and behavior in BEHAVIOR_EXPLANATION_SPECS:
            summary = "CALCULATING"
        self._draw_text(
            f"brain_why_{behavior.value}_summary",
            summary,
            bounds.right - 16.0,
            bounds.top - 50.0,
            title_color,
            9.0,
            bold=True,
            anchor_x="right",
            anchor_y="center",
        )
        track = arcade.LBWH(
            bounds.left + 16.0,
            bounds.top - self.BRAIN_WHY_CARD_HEIGHT + 11.0,
            max(1.0, bounds.width - 32.0),
            6.0,
        )
        self._draw_rounded_rect_fill(
            track,
            self._brain_blend_color(
                self.theme.card_background,
                self.theme.panel_border,
                0.72,
            ),
            3.0,
        )
        if score > 0.0:
            self._draw_rounded_rect_fill(
                arcade.LBWH(
                    track.left,
                    track.bottom,
                    max(1.0, track.width * score),
                    track.height,
                ),
                self._brain_blend_color(
                    self.theme.panel_border,
                    accent,
                    0.35 + 0.65 * intensity,
                ),
                3.0,
            )
        if expanded:
            detail_top = bounds.top - self.BRAIN_WHY_CARD_HEIGHT
            arcade.draw_line(
                bounds.left + 12.0,
                detail_top,
                bounds.right - 12.0,
                detail_top,
                self._brain_blend_color(
                    self.theme.panel_border,
                    accent,
                    0.34 + 0.46 * intensity,
                ),
                1.0,
            )
            self._draw_brain_why_details(
                arcade.LBWH(
                    bounds.left,
                    bounds.bottom,
                    bounds.width,
                    detail_top - bounds.bottom,
                ),
                behavior,
                snapshot,
                accent,
            )

    def _draw_brain_why_details(
        self,
        bounds: arcade.Rect,
        behavior: BehaviorKind,
        snapshot: WhySnapshot | None,
        accent: tuple[int, int, int],
    ) -> None:
        """Draw fixed-order intervention details inside an expanded card."""
        self._draw_text(
            f"brain_why_{behavior.value}_detail_title",
            "COUNTERFACTUAL DETAILS",
            bounds.left + 16.0,
            bounds.top - 22.0,
            accent,
            10.0,
            bold=True,
            anchor_y="center",
        )
        if behavior is BehaviorKind.RESTING:
            lines = self._wrap_line(
                (
                    "No direct neural WHY available. Resting is defined from "
                    "realized locomotion and has no dedicated neural action "
                    "output."
                ),
                max(24.0, bounds.width - 32.0),
                font_size=10.0,
            )
            for line_index, line in enumerate(lines[:4]):
                self._draw_text(
                    f"brain_why_resting_detail_{line_index}",
                    line,
                    bounds.left + 16.0,
                    bounds.top - 49.0 - line_index * 16.0,
                    self.theme.text_muted,
                    10.0,
                )
            return

        spec = BEHAVIOR_EXPLANATION_SPECS[behavior]
        effect_by_intervention = (
            {}
            if snapshot is None
            else {
                effect.intervention: effect
                for effect in snapshot.effects
            }
        )
        cursor_top = bounds.top - 43.0
        for effect_index, intervention in enumerate(spec.interventions):
            if effect_index:
                cursor_top -= self.BRAIN_WHY_EFFECT_GAP
            height = self._brain_why_effect_height(behavior, intervention)
            card = arcade.LBWH(
                bounds.left + 10.0,
                cursor_top - height,
                bounds.width - 20.0,
                height,
            )
            self._draw_brain_why_effect_card(
                card,
                behavior,
                effect_index,
                intervention,
                effect_by_intervention.get(intervention),
                spec.displayed_outputs,
                accent,
            )
            cursor_top = card.bottom

    def _draw_brain_why_effect_card(
        self,
        bounds: arcade.Rect,
        behavior: BehaviorKind,
        effect_index: int,
        intervention: SemanticIntervention,
        effect: SemanticEffectSnapshot | None,
        output_names: tuple[str, ...],
        accent: tuple[int, int, int],
    ) -> None:
        """Draw one fixed-position semantic intervention result."""
        self._draw_rounded_rect(
            bounds,
            self.theme.card_background,
            self.theme.panel_border,
            max(4.0, self.config.layout.card_radius - 2.0),
            1.0,
        )
        key_prefix = f"brain_why_{behavior.value}_effect_{effect_index}"
        self._draw_text(
            key_prefix,
            self._why_intervention_label(intervention),
            bounds.left + 10.0,
            bounds.top - 18.0,
            self.theme.text_primary,
            10.0,
            bold=True,
        )
        meta = "WAITING FOR CURRENT BOUT"
        if effect is not None:
            meta = (
                f"INFLUENCE {effect.influence_score:.2f} · "
                f"{effect.influence_label.value.upper()} · "
                f"{effect.effect_direction.value.upper()} · "
                f"n={effect.sample_count}"
            )
        self._draw_text(
            f"{key_prefix}_meta",
            meta,
            bounds.left + 10.0,
            bounds.top - 38.0,
            accent,
            8.5,
            bold=True,
        )
        output_top_offset = 58.0
        if intervention is SemanticIntervention.SATIATED_STATE:
            self._draw_text(
                f"{key_prefix}_satiated",
                "If this creature were satiated",
                bounds.left + 10.0,
                bounds.top - 56.0,
                self.theme.text_muted,
                8.5,
            )
            output_top_offset = 74.0
        outputs_by_name = (
            {}
            if effect is None
            else {
                output.output_name: output
                for output in effect.output_effects
            }
        )
        for output_index, output_name in enumerate(output_names):
            output = outputs_by_name.get(output_name)
            secondary = (
                " · SECONDARY CONTEXT"
                if output is not None and output.secondary_context
                else ""
            )
            label_y = (
                bounds.top - output_top_offset - output_index * 34.0
            )
            self._draw_text(
                f"{key_prefix}_output_{output_index}_name",
                f"{output_name}{secondary}",
                bounds.left + 10.0,
                label_y,
                self.theme.text_muted,
                8.5,
                bold=True,
            )
            values = "actual —  →  counterfactual —"
            if output is not None:
                values = (
                    f"actual {output.actual:+.2f}  →  "
                    f"counterfactual {output.counterfactual:+.2f}  · "
                    f"{output.direction.value.upper()}"
                )
            self._draw_text(
                f"{key_prefix}_output_{output_index}_values",
                values,
                bounds.left + 18.0,
                label_y - 16.0,
                self.theme.text_muted,
                8.5,
            )

    @classmethod
    def _brain_why_effect_height(
        cls,
        behavior: BehaviorKind,
        intervention: SemanticIntervention,
    ) -> float:
        """Return stable height for one semantic-effect detail card."""
        output_count = len(
            BEHAVIOR_EXPLANATION_SPECS[behavior].displayed_outputs
        )
        satiated_extra = (
            16.0
            if intervention is SemanticIntervention.SATIATED_STATE
            else 0.0
        )
        return 54.0 + 34.0 * output_count + satiated_extra

    @classmethod
    def _brain_why_detail_height(cls, behavior: BehaviorKind) -> float:
        """Return data-independent expanded height for one WHY card."""
        if behavior is BehaviorKind.RESTING:
            return 112.0
        spec = BEHAVIOR_EXPLANATION_SPECS[behavior]
        effects_height = sum(
            cls._brain_why_effect_height(behavior, intervention)
            for intervention in spec.interventions
        )
        gaps_height = (
            max(0, len(spec.interventions) - 1)
            * cls.BRAIN_WHY_EFFECT_GAP
        )
        return (
            43.0
            + effects_height
            + gaps_height
            + cls.BRAIN_WHY_DETAIL_PADDING
        )

    @staticmethod
    def _brain_why_card_hitbox_key(behavior: BehaviorKind) -> str:
        """Return the stable interaction key for one WHY behavior card."""
        return f"brain_why_card_{behavior.value}"

    @staticmethod
    def _why_intervention_label(
        intervention: SemanticIntervention,
    ) -> str:
        """Return the concise scientific UI label for an intervention."""
        return {
            SemanticIntervention.VISIBLE_FOOD_CUES: "Visible food cues",
            SemanticIntervention.RESOURCE_GRADIENT_CUES: "Resource gradient",
            SemanticIntervention.SATIATED_STATE: "Satiated state",
            SemanticIntervention.SOCIAL_CUES: "Social cues",
            SemanticIntervention.OFFSPRING_CUES: "Offspring cues",
            SemanticIntervention.ACOUSTIC_CUES: "Acoustic cues",
            SemanticIntervention.TRAIL_PHEROMONE_CUES: "Trail pheromone cues",
            SemanticIntervention.ALARM_PHEROMONE_CUES: "Alarm pheromone cues",
            SemanticIntervention.WALL_CUES: "Wall cues",
        }[intervention]

    def _draw_brain_why_diagnostics(
        self,
        bounds: arcade.Rect,
        diagnostics: CounterfactualDiagnostics,
    ) -> None:
        """Draw debug-only counterfactual queue and worker diagnostics."""
        self._draw_rounded_rect(
            bounds,
            self.theme.panel_background,
            self.theme.panel_border,
            self.config.layout.card_radius,
            1.0,
        )
        queue_size = (
            diagnostics.probe_queue_size
            if diagnostics.probe_queue_size is not None
            else "n/a"
        )
        lines = (
            "WHY DIAGNOSTICS",
            (
                f"Requests {diagnostics.probe_requests} · dropped "
                f"{diagnostics.probe_requests_dropped}"
            ),
            f"Superseded {diagnostics.probes_superseded}",
            f"Evaluations {diagnostics.evaluations_performed}",
            f"Results dropped {diagnostics.result_drops}",
            (
                "Latency unavailable"
                if diagnostics.result_latency_ms is None
                else f"Latency {diagnostics.result_latency_ms:.1f} ms"
            ),
            (
                "Result age unavailable"
                if diagnostics.latest_result_age_ms is None
                else f"Result age {diagnostics.latest_result_age_ms:.1f} ms"
            ),
            (
                f"Worker {diagnostics.worker_health} · queue {queue_size} · "
                f"{diagnostics.evaluations_per_second:.1f} eval/s"
            ),
        )
        for index, line in enumerate(lines):
            self._draw_text(
                f"brain_why_diagnostics_{index}",
                line,
                bounds.left + 12.0,
                bounds.top - 20.0 - index * 21.0,
                (
                    self.theme.text_primary
                    if index == 0
                    else self.theme.text_muted
                ),
                9.5 if index == 0 else 8.5,
                bold=index == 0,
            )

    def _draw_brain_inspector_tab(
        self,
        bounds: arcade.Rect,
        label: str,
        key: str,
        active: bool,
    ) -> None:
        """Draw one page selector and preserve its interaction key."""
        fill = self.theme.accent_soft if active else self.theme.card_background
        border = self.theme.accent if active else self.theme.panel_border
        text_color = self.theme.accent if active else self.theme.text_muted
        self._draw_rounded_rect(bounds, fill, border, 6.0, 1.0)
        self._draw_text(
            f"brain_inspector_tab_{key}",
            label,
            bounds.center_x,
            bounds.center_y,
            text_color,
            9,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )

    def _draw_brain_behavior_inspector(
        self,
        world: World,
        bounds: arcade.Rect,
    ) -> None:
        """Draw persistent focal-behaviour cards and observer diagnostics."""
        self._draw_rounded_rect(
            bounds,
            self.theme.card_background,
            self.theme.panel_border,
            self.config.layout.card_radius,
            1.0,
        )
        behavior_config = getattr(
            getattr(world, "config", None),
            "behavior",
            None,
        )
        snapshot = getattr(world, "selected_behavior_snapshot", None)
        diagnostics = getattr(
            world,
            "behavior_observer_diagnostics",
            None,
        )
        states_by_behavior: dict[BehaviorKind, BehaviorStateSnapshot] = {}
        if behavior_config is not None and not behavior_config.enabled:
            status = (
                "Observer disabled",
                "Enable the behaviour observer in simulation configuration.",
                False,
            )
        elif diagnostics is not None and diagnostics.last_error:
            status = (
                "Observer unavailable",
                diagnostics.last_error,
                True,
            )
        elif snapshot is None:
            status = (
                "Collecting temporal evidence",
                "Cards brighten after their start persistence is satisfied.",
                False,
            )
        else:
            delayed = (
                not bool(getattr(world, "is_paused", False))
                and behavior_config is not None
                and float(getattr(world, "elapsed_time", 0.0))
                - snapshot.simulation_time
                > max(0.5, 3.0 / behavior_config.sample_hz)
            )
            states_by_behavior = {
                state.behavior: state
                for state in snapshot.behaviors
            }
            if delayed:
                status = (
                    "Observer updating",
                    "Showing the latest Evidence while results catch up.",
                    False,
                )
            elif not states_by_behavior:
                status = (
                    "No sustained bout detected",
                    "Current motion does not satisfy an operational rule.",
                    False,
                )
            else:
                active_count = sum(
                    state.status is BoutStatus.ACTIVE
                    for state in states_by_behavior.values()
                )
                emerging_count = len(states_by_behavior) - active_count
                status = (
                    "Live bouts detected",
                    (
                        f"{active_count} active · {emerging_count} emerging. "
                        "Click a card to inspect its activation conditions."
                    ),
                    False,
                )

        diagnostics_visible = (
            bool(getattr(world, "debug_vision_enabled", False))
            and diagnostics is not None
        )
        content = arcade.LBWH(
            bounds.left + 14.0,
            bounds.bottom + 14.0,
            bounds.width - 28.0,
            bounds.height - 28.0,
        )
        drawing_width = max(1.0, content.width - 12.0)
        for behavior in BehaviorKind:
            self._control_hitboxes.pop(
                self._brain_behavior_card_hitbox_key(behavior),
                None,
            )
        expanded_behavior = next(
            (
                behavior
                for behavior in BehaviorKind
                if behavior.value == self._brain_expanded_behavior
            ),
            None,
        )
        expanded_detail_lines = (
            tuple(
                self._wrap_line(
                    self.BRAIN_BEHAVIOR_ACTIVATION_COPY[expanded_behavior],
                    max(24.0, drawing_width - 28.0),
                    font_size=10.5,
                )
            )
            if expanded_behavior is not None
            else ()
        )
        expanded_detail_height = (
            self._brain_behavior_detail_height(expanded_detail_lines)
            if expanded_behavior is not None
            else 0.0
        )
        status_title, status_message, status_is_error = status
        status_lines = tuple(
            self._wrap_line(
                status_message,
                max(24.0, drawing_width - 24.0),
                font_size=10.0,
            )
        )
        if len(status_lines) > 2:
            status_lines = (
                status_lines[0],
                self._fit_line(f"{status_lines[1]}…", drawing_width - 24.0),
            )
        cards_height = (
            len(BehaviorKind) * self.BRAIN_BEHAVIOR_CARD_HEIGHT
            + (len(BehaviorKind) - 1) * self.BRAIN_BEHAVIOR_CARD_GAP
            + expanded_detail_height
        )
        diagnostics_height = (
            self.BRAIN_BEHAVIOR_CARD_GAP
            + self.BRAIN_BEHAVIOR_DIAGNOSTICS_HEIGHT
            if diagnostics_visible
            else 0.0
        )
        total_height = (
            self.BRAIN_BEHAVIOR_HEADER_HEIGHT
            + self.BRAIN_BEHAVIOR_NOTICE_HEIGHT
            + self.BRAIN_BEHAVIOR_CARD_GAP
            + cards_height
            + diagnostics_height
        )
        scroll_key = "brain_behavior_inspector"
        self._scroll_offsets[scroll_key] = self._brain_behavior_scroll_offset
        scroll_limit = max(0.0, total_height - content.height)
        scroll_offset = max(
            0.0,
            min(scroll_limit, self._scroll_offsets.get(scroll_key, 0.0)),
        )
        self._scroll_offsets[scroll_key] = scroll_offset
        self._scroll_limits[scroll_key] = scroll_limit
        self._scroll_regions[scroll_key] = content
        self._brain_behavior_scroll_offset = scroll_offset

        cursor_top = content.top + scroll_offset
        with self._ui_clip(content):
            self._draw_text(
                "brain_behavior_title",
                "OBSERVED BEHAVIOURS",
                content.left,
                cursor_top - 16.0,
                self.theme.text_primary,
                11.5,
                bold=True,
            )
            self._draw_text(
                "brain_behavior_subtitle",
                "World/action history, not neural intent",
                content.left,
                cursor_top - 41.0,
                self.theme.text_muted,
                10,
            )
            cursor_top -= self.BRAIN_BEHAVIOR_HEADER_HEIGHT

            notice_bounds = arcade.LBWH(
                content.left,
                cursor_top - self.BRAIN_BEHAVIOR_NOTICE_HEIGHT,
                drawing_width,
                self.BRAIN_BEHAVIOR_NOTICE_HEIGHT,
            )
            self._draw_brain_behavior_notice(
                notice_bounds,
                0,
                status_title,
                status_lines,
                is_error=status_is_error,
            )
            cursor_top = notice_bounds.bottom - self.BRAIN_BEHAVIOR_CARD_GAP

            for behavior_index, behavior in enumerate(BehaviorKind):
                detail_lines = (
                    expanded_detail_lines
                    if behavior is expanded_behavior
                    else ()
                )
                card_height = (
                    self.BRAIN_BEHAVIOR_CARD_HEIGHT
                    + (
                        expanded_detail_height
                        if behavior is expanded_behavior
                        else 0.0
                    )
                )
                card_bounds = arcade.LBWH(
                    content.left,
                    cursor_top - card_height,
                    drawing_width,
                    card_height,
                )
                if detail_lines:
                    self._draw_brain_behavior_card(
                        card_bounds,
                        behavior,
                        states_by_behavior.get(behavior),
                        detail_lines=detail_lines,
                    )
                else:
                    self._draw_brain_behavior_card(
                        card_bounds,
                        behavior,
                        states_by_behavior.get(behavior),
                    )
                visible_bottom = max(card_bounds.bottom, content.bottom)
                visible_top = min(card_bounds.top, content.top)
                if visible_top > visible_bottom:
                    self._control_hitboxes[
                        self._brain_behavior_card_hitbox_key(behavior)
                    ] = arcade.LBWH(
                        card_bounds.left,
                        visible_bottom,
                        card_bounds.width,
                        visible_top - visible_bottom,
                    )
                cursor_top = card_bounds.bottom
                if behavior_index < len(BehaviorKind) - 1:
                    cursor_top -= self.BRAIN_BEHAVIOR_CARD_GAP

            if diagnostics_visible:
                cursor_top -= self.BRAIN_BEHAVIOR_CARD_GAP
                diagnostic_bounds = arcade.LBWH(
                    content.left,
                    cursor_top - self.BRAIN_BEHAVIOR_DIAGNOSTICS_HEIGHT,
                    drawing_width,
                    self.BRAIN_BEHAVIOR_DIAGNOSTICS_HEIGHT,
                )
                self._draw_brain_behavior_diagnostics(
                    diagnostic_bounds,
                    diagnostics,
                )

        if scroll_limit > 0.0:
            self._draw_scrollbar(content, scroll_offset, scroll_limit)

    def _draw_brain_behavior_card(
        self,
        bounds: arcade.Rect,
        behavior: BehaviorKind,
        state: BehaviorStateSnapshot | None,
        *,
        detail_lines: tuple[str, ...] = (),
    ) -> None:
        """Draw one behaviour card with optional activation details."""
        accent = self.BRAIN_BEHAVIOR_ACCENTS[behavior]
        intensity = self._behavior_display_intensity(state)
        expanded = bool(detail_lines)
        score = (
            self._clamped_behavior_evidence(state.evidence_score)
            if state is not None
            else None
        )
        fill = self._brain_blend_color(
            self.theme.card_background,
            accent,
            0.04 + 0.18 * intensity,
        )
        border = self._brain_blend_color(
            self.theme.panel_border,
            accent,
            0.24 + 0.70 * intensity,
        )
        title_color = self._brain_blend_color(
            self.theme.text_muted,
            accent,
            0.38 + 0.62 * intensity,
        )
        self._draw_rounded_rect(
            bounds,
            fill,
            border,
            self.config.layout.card_radius,
            1.0 + intensity,
        )
        arcade.draw_circle_filled(
            bounds.left + 16.0,
            bounds.top - 21.0,
            6.0,
            self._brain_blend_color(
                self.theme.card_background,
                accent,
                0.28 + 0.72 * intensity,
            ),
        )
        self._draw_text(
            f"brain_behavior_{behavior.value}_name",
            self._behavior_display_name(behavior),
            bounds.left + 31.0,
            bounds.top - 21.0,
            title_color,
            13,
            bold=True,
            anchor_y="center",
        )
        evidence_text = "—" if score is None else f"{score:.2f}"
        narrow_card = bounds.width < 230.0
        evidence_right = bounds.right - (14.0 if narrow_card else 36.0)
        self._draw_text(
            f"brain_behavior_{behavior.value}_evidence",
            f"EVIDENCE {evidence_text}",
            evidence_right,
            bounds.top - (65.0 if narrow_card else 21.0),
            title_color,
            10.5,
            bold=True,
            anchor_x="right",
            anchor_y="center",
        )
        self._draw_text(
            f"brain_behavior_{behavior.value}_expand",
            "−" if expanded else "+",
            bounds.right - 15.0,
            bounds.top - 21.0,
            title_color,
            15,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )
        state_text = "INACTIVE"
        if state is not None:
            state_text = (
                f"{state.status.value.upper()} · "
                f"{state.duration_seconds:.1f} s"
            )
        self._draw_text(
            f"brain_behavior_{behavior.value}_state",
            state_text,
            bounds.left + 16.0,
            bounds.top - (43.0 if narrow_card else 50.0),
            self._brain_blend_color(
                self.theme.text_muted,
                accent,
                0.12 + 0.68 * intensity,
            ),
            10.5,
            bold=state is not None,
            anchor_y="center",
        )
        track = arcade.LBWH(
            bounds.left + 16.0,
            bounds.top - self.BRAIN_BEHAVIOR_CARD_HEIGHT + 11.0,
            max(1.0, bounds.width - 32.0),
            6.0,
        )
        self._draw_rounded_rect_fill(
            track,
            self._brain_blend_color(
                self.theme.card_background,
                self.theme.panel_border,
                0.72,
            ),
            3.0,
        )
        if score is not None and score > 0.0:
            fill_bounds = arcade.LBWH(
                track.left,
                track.bottom,
                max(1.0, track.width * score),
                track.height,
            )
            self._draw_rounded_rect_fill(
                fill_bounds,
                self._brain_blend_color(
                    self.theme.panel_border,
                    accent,
                    0.35 + 0.65 * intensity,
                ),
                3.0,
            )
        if expanded:
            detail_top = bounds.top - self.BRAIN_BEHAVIOR_CARD_HEIGHT
            arcade.draw_line(
                bounds.left + 12.0,
                detail_top,
                bounds.right - 12.0,
                detail_top,
                self._brain_blend_color(
                    self.theme.panel_border,
                    accent,
                    0.34 + 0.46 * intensity,
                ),
                1.0,
            )
            self._draw_text(
                f"brain_behavior_{behavior.value}_detail_title",
                "ACTIVATES WHEN",
                bounds.left + 16.0,
                detail_top - 22.0,
                title_color,
                10,
                bold=True,
                anchor_y="center",
            )
            for line_index, line in enumerate(detail_lines):
                self._draw_text(
                    (
                        f"brain_behavior_{behavior.value}_detail_"
                        f"{line_index}"
                    ),
                    line,
                    bounds.left + 16.0,
                    (
                        detail_top
                        - 48.0
                        - line_index
                        * self.BRAIN_BEHAVIOR_DETAIL_LINE_HEIGHT
                    ),
                    self.theme.text_muted,
                    10.5,
                    anchor_y="center",
                )

    def _draw_brain_behavior_notice(
        self,
        bounds: arcade.Rect,
        index: int,
        title: str,
        message_lines: tuple[str, ...],
        *,
        is_error: bool,
    ) -> None:
        """Draw one compact observer-state notice."""
        accent = self.theme.selected_outline if is_error else self.theme.accent
        self._draw_rounded_rect(
            bounds,
            self._brain_blend_color(
                self.theme.card_background,
                accent,
                0.08,
            ),
            self._brain_blend_color(
                self.theme.panel_border,
                accent,
                0.42,
            ),
            self.config.layout.card_radius,
            1.0,
        )
        self._draw_text(
            f"brain_behavior_notice_{index}_title",
            title,
            bounds.left + 14.0,
            bounds.top - 22.0,
            accent,
            11,
            bold=True,
            anchor_y="center",
        )
        for line_index, line in enumerate(message_lines):
            self._draw_text(
                f"brain_behavior_notice_{index}_message_{line_index}",
                line,
                bounds.left + 14.0,
                (
                    bounds.top
                    - 50.0
                    - line_index * self.BRAIN_BEHAVIOR_NOTICE_LINE_HEIGHT
                ),
                self.theme.text_muted,
                10,
                anchor_y="center",
            )

    @classmethod
    def _brain_behavior_detail_height(
        cls,
        detail_lines: tuple[str, ...],
    ) -> float:
        """Return the extra height required by expanded activation details."""
        line_count = max(1, len(detail_lines))
        return (
            65.0
            + (line_count - 1) * cls.BRAIN_BEHAVIOR_DETAIL_LINE_HEIGHT
        )

    @staticmethod
    def _brain_behavior_card_hitbox_key(behavior: BehaviorKind) -> str:
        """Return the stable interaction key for one behaviour card."""
        return f"brain_behavior_card_{behavior.value}"

    def _draw_brain_behavior_diagnostics(
        self,
        bounds: arcade.Rect,
        diagnostics: BehaviorObserverDiagnostics,
    ) -> None:
        """Draw debug-only behavior-observer diagnostics."""
        self._draw_rounded_rect(
            bounds,
            self.theme.panel_background,
            self.theme.panel_border,
            self.config.layout.card_radius,
            1.0,
        )
        queue_size = (
            diagnostics.input_queue_size
            if diagnostics.input_queue_size is not None
            else "n/a"
        )
        lines = (
            "OBSERVER DIAGNOSTICS",
            (
                f"Samples {diagnostics.samples_produced} · "
                f"dropped {diagnostics.samples_dropped}"
            ),
            (
                f"Processed {diagnostics.observations_processed} · "
                f"results dropped {diagnostics.results_dropped}"
            ),
            (
                "Result latency unavailable"
                if diagnostics.result_latency_ms is None
                else f"Result latency {diagnostics.result_latency_ms:.1f} ms"
            ),
            (
                f"Worker {diagnostics.worker_health} · queue "
                f"{queue_size}"
            ),
        )
        for line_index, line in enumerate(lines):
            self._draw_text(
                f"brain_behavior_diagnostics_{line_index}",
                line,
                bounds.left + 14.0,
                bounds.top - 21.0 - line_index * 24.0,
                (
                    self.theme.text_primary
                    if line_index == 0
                    else self.theme.text_muted
                ),
                10.5 if line_index == 0 else 9.5,
                bold=line_index == 0,
            )

    @staticmethod
    def _clamped_behavior_evidence(value: float) -> float:
        """Clamp an Evidence value for presentation."""
        if not isfinite(value):
            return 0.0
        return max(0.0, min(1.0, value))

    @classmethod
    def _behavior_display_intensity(
        cls,
        state: BehaviorStateSnapshot | None,
    ) -> float:
        """Map an existing bout state and Evidence score to UI intensity."""
        if state is None:
            return 0.0
        score = cls._clamped_behavior_evidence(state.evidence_score)
        if state.status is BoutStatus.ACTIVE:
            return 0.55 + 0.45 * score
        return 0.35 + 0.45 * score

    @staticmethod
    def _behavior_display_name(behavior: BehaviorKind) -> str:
        """Return concise user-facing copy for an observational label."""
        labels = {
            BehaviorKind.FOOD_ORIENTATION: "Food orientation",
            BehaviorKind.FOOD_APPROACH: "Food approach",
            BehaviorKind.FEEDING: "Feeding",
            BehaviorKind.RESTING: "Resting",
            BehaviorKind.COHESION: "Cohesion",
            BehaviorKind.ALARM_RETREAT: "Alarm retreat",
        }
        return labels[behavior]

    def _draw_brain_node_inspector(
        self,
        brain: object | None,
        layout: BrainGraphLayout | None,
        bounds: arcade.Rect,
        *,
        show_close: bool = True,
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
        if show_close:
            self._control_hitboxes["brain_node_inspector_toggle"] = close_button
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

        kind_label = f"{node.kind.value.title()} Node"
        summary, badge, name_bounds, name_text = (
            self._brain_node_summary_layout(
                header,
                self._brain_node_display_name(node),
                kind_label,
            )
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
            name_text,
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
            self._cached_brain_node_inspector_lines(brain, layout, node),
            line_spacing=20,
            first_line_color=self.theme.text_primary,
            body_color=self.theme.text_muted,
            first_line_bold=True,
            wrap_lines=True,
        )
    def _brain_node_summary_layout(
        self,
        header: arcade.Rect,
        name: str,
        kind_label: str,
    ) -> tuple[arcade.Rect, arcade.Rect, arcade.Rect, str]:
        """Return responsive bounds and text for a selected-node summary.

        Parameters
        ----------
        header
            Inspector header used to anchor the summary.
        name
            Node display name.
        kind_label
            Node-kind badge label.

        Returns
        -------
        tuple[arcade.Rect, arcade.Rect, arcade.Rect, str]
            Summary, badge, name bounds, and wrapped name text.
        """
        summary_top = header.bottom - 12.0
        summary_width = max(1.0, header.width - 28.0)
        summary = arcade.LBWH(
            header.left + 14.0,
            summary_top - self.BRAIN_NODE_SUMMARY_MIN_HEIGHT,
            summary_width,
            self.BRAIN_NODE_SUMMARY_MIN_HEIGHT,
        )
        _, name_bounds = self._brain_node_badge_layout(summary, kind_label)
        name_text = self._brain_node_name_text(name, name_bounds.width)
        line_count = max(1, len(name_text.splitlines()))
        summary_height = max(
            self.BRAIN_NODE_SUMMARY_MIN_HEIGHT,
            self.BRAIN_NODE_SUMMARY_VERTICAL_PADDING
            + line_count * self.BRAIN_NODE_SUMMARY_LINE_HEIGHT,
        )
        summary = arcade.LBWH(
            summary.left,
            summary_top - summary_height,
            summary.width,
            summary_height,
        )
        badge, name_bounds = self._brain_node_badge_layout(summary, kind_label)
        return summary, badge, name_bounds, name_text

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
        return "\n".join(
            self._wrap_line(
                name,
                width,
                font_size=14.0,
                bold=True,
            )
        )
    def _cached_brain_node_inspector_lines(
        self,
        brain: object,
        layout: BrainGraphLayout,
        node: BrainGraphNode,
    ) -> tuple[str, ...]:
        """Return cached inspector content for the stable node selection."""
        state = self._brain_state
        if (
            state.inspector_brain is brain
            and state.inspector_layout is layout
            and state.inspector_node_key == node.key
        ):
            return state.inspector_lines

        lines = tuple(self._brain_node_inspector_lines(brain, layout, node))
        state.inspector_brain = brain
        state.inspector_layout = layout
        state.inspector_node_key = node.key
        state.inspector_lines = lines
        return lines
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
        highlight = self._brain_highlight_for_node(layout, node.key)
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
