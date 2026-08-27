from __future__ import annotations

from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, replace
from time import perf_counter
import sys
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest.mock import ANY, Mock, patch

try:
    import arcade
except ModuleNotFoundError:
    arcade = ModuleType("arcade")
    sys.modules["arcade"] = arcade

if not hasattr(arcade, "LBWH"):

    @dataclass(slots=True)
    class FakeRect:
        left: float
        bottom: float
        width: float
        height: float

        @property
        def right(self) -> float:
            return self.left + self.width

        @property
        def top(self) -> float:
            return self.bottom + self.height

        @property
        def center_x(self) -> float:
            return self.left + self.width / 2

        @property
        def center_y(self) -> float:
            return self.bottom + self.height / 2

    def fake_lbwh(left: float, bottom: float, width: float, height: float) -> FakeRect:
        return FakeRect(left, bottom, width, height)

    arcade.LBWH = fake_lbwh
    arcade.Rect = FakeRect

class FakeText:
    def __init__(
        self,
        text: str,
        x: float,
        y: float,
        color: object,
        size: float,
        **kwargs: object,
    ) -> None:
        self.text = text
        self.x = x
        self.y = y
        self.color = color
        self.font_size = size
        self.bold = kwargs.get("bold", False)
        self.width = kwargs.get("width")
        self.multiline = kwargs.get("multiline", False)
        self.align = kwargs.get("align", "left")
        self.anchor_x = kwargs.get("anchor_x", "left")
        self.anchor_y = kwargs.get("anchor_y", "baseline")

    def draw(self) -> None:
        return None


arcade.Text = FakeText

for draw_name in (
    "draw_lrbt_rectangle_filled",
    "draw_circle_filled",
    "draw_circle_outline",
    "draw_line",
    "draw_line_strip",
    "draw_polygon_filled",
    "draw_texture_rectangle",
):
    setattr(arcade, draw_name, lambda *args, **kwargs: None)

for optional_module in ("neat", "pymunk"):
    try:
        __import__(optional_module)
    except ModuleNotFoundError:
        sys.modules[optional_module] = ModuleType(optional_module)

from configs.sim_config import LiveFoodConfig, build_sim_config
from src.analysis import (
    BEHAVIOR_RADAR_LABELS,
    generate_inspector_report,
    generate_radar_chart_image,
)
from src.behavior_observer import (
    BehaviorKind,
    BehaviorObserverDiagnostics,
    BehaviorSnapshot,
    BehaviorStateSnapshot,
    BoutStatus,
)
from src.flocking import FlockingRuntimeSnapshot
from src.counterfactual_neat import (
    CounterfactualDiagnostics,
    EffectDirection,
    InfluenceLabel,
    OutputEffect,
    SemanticEffectSnapshot,
    SemanticIntervention,
    WhySnapshot,
)
from src.ui.layouts.brain_graph import (
    BrainEdgeKind,
    BrainGraphEdge,
    BrainNodeKind,
    build_brain_graph_layout,
    highlighted_path_through_node,
)
from src.creature import (
    FlockingTraits,
    LedgerDiagnostics,
    LineageInfo,
    PhysicalTraits,
    TraitMutationDelta,
)
from src.ui.layouts.screen import build_screen_layout
from src.speciation import (
    NeatChangeSummary,
    NeuralShift,
    SpeciesDistanceBreakdown,
    SpeciesRecord,
    SpeciesTraitSnapshot,
)
from src.ui.renderer import UiRenderer
from src.vision import SENSOR_INPUT_NAMES


class UiRendererBrainWindowScrollTest(unittest.TestCase):
    def setUp(self) -> None:
        self.renderer = UiRenderer(build_sim_config())
        self.renderer._brain_window_open = True
        self.renderer._brain_window_bounds = arcade.LBWH(100, 100, 400, 300)
        self.renderer._control_hitboxes["brain_window_graph"] = arcade.LBWH(
            120,
            150,
            360,
            180,
        )

    def test_scroll_over_graph_is_consumed_without_zooming_brain_graph(self) -> None:
        handled = self.renderer.handle_mouse_scroll(200, 200, 1)

        self.assertTrue(handled)
        self.assertAlmostEqual(self.renderer._brain_graph_zoom, 1.0)

    def test_brain_inspector_page_buttons_switch_pages(self) -> None:
        self.renderer._control_hitboxes["brain_inspector_page_node"] = arcade.LBWH(
            120,
            340,
            80,
            24,
        )
        self.renderer._control_hitboxes[
            "brain_inspector_page_behaviors"
        ] = arcade.LBWH(210, 340, 100, 24)

        self.assertTrue(
            self.renderer.handle_mouse_press(SimpleNamespace(), 250, 350)
        )
        self.assertEqual(self.renderer._brain_inspector_page, "behaviors")
        self.assertTrue(
            self.renderer.handle_mouse_press(SimpleNamespace(), 150, 350)
        )
        self.assertEqual(self.renderer._brain_inspector_page, "node")

    def test_why_page_button_and_scroll_state_are_independent(self) -> None:
        self.renderer._control_hitboxes[
            "brain_inspector_page_why"
        ] = arcade.LBWH(320, 340, 70, 24)

        self.assertTrue(
            self.renderer.handle_mouse_press(SimpleNamespace(), 350, 350)
        )
        self.assertEqual(self.renderer._brain_inspector_page, "why")

        self.renderer._scroll_regions["brain_why_inspector"] = arcade.LBWH(
            200,
            180,
            200,
            120,
        )
        self.renderer._scroll_limits["brain_why_inspector"] = 100
        self.assertTrue(self.renderer.handle_mouse_scroll(250, 220, -1))
        self.assertEqual(self.renderer._brain_why_scroll_offset, 24)
        self.assertEqual(self.renderer._brain_behavior_scroll_offset, 0.0)

    def test_why_labels_separate_food_gradient_and_satiation(self) -> None:
        self.assertEqual(
            self.renderer._why_intervention_label(
                SemanticIntervention.VISIBLE_FOOD_CUES
            ),
            "Visible food cues",
        )
        self.assertEqual(
            self.renderer._why_intervention_label(
                SemanticIntervention.RESOURCE_GRADIENT_CUES
            ),
            "Resource gradient",
        )
        self.assertEqual(
            self.renderer._why_intervention_label(
                SemanticIntervention.SATIATED_STATE
            ),
            "Satiated state",
        )

    def test_why_card_click_toggles_one_expanded_card(self) -> None:
        self.renderer._brain_inspector_page = "why"
        food_key = self.renderer._brain_why_card_hitbox_key(
            BehaviorKind.FOOD_APPROACH
        )
        rest_key = self.renderer._brain_why_card_hitbox_key(
            BehaviorKind.RESTING
        )
        self.renderer._control_hitboxes[food_key] = arcade.LBWH(
            310,
            180,
            100,
            50,
        )
        self.renderer._control_hitboxes[rest_key] = arcade.LBWH(
            310,
            240,
            100,
            50,
        )

        self.assertTrue(
            self.renderer.handle_mouse_press(
                SimpleNamespace(),
                350,
                200,
            )
        )
        self.assertEqual(
            self.renderer._brain_expanded_why_behavior,
            BehaviorKind.FOOD_APPROACH.value,
        )

        self.assertTrue(
            self.renderer.handle_mouse_press(
                SimpleNamespace(),
                350,
                260,
            )
        )
        self.assertEqual(
            self.renderer._brain_expanded_why_behavior,
            BehaviorKind.RESTING.value,
        )

        self.assertTrue(
            self.renderer.handle_mouse_press(
                SimpleNamespace(),
                350,
                260,
            )
        )
        self.assertIsNone(self.renderer._brain_expanded_why_behavior)

    def test_why_page_keeps_all_behavior_cards_in_fixed_positions(self) -> None:
        captured = []
        original_card = self.renderer._draw_brain_why_card

        def capture_card(
            bounds,
            behavior,
            state,
            snapshot,
            *,
            expanded,
        ):
            captured.append(
                (bounds, behavior, state, snapshot, expanded)
            )

        self.renderer._draw_brain_why_card = capture_card
        config = build_sim_config()
        empty_snapshot = BehaviorSnapshot(
            creature_id=1,
            selection_generation=1,
            simulation_time=1.0,
            behaviors=(),
            observations_processed=10,
            produced_monotonic=0.0,
        )
        world = SimpleNamespace(
            config=config,
            selected_behavior_snapshot=empty_snapshot,
            selected_why_snapshots=(),
            counterfactual_diagnostics=CounterfactualDiagnostics(
                worker_health="running"
            ),
            debug_vision_enabled=False,
            elapsed_time=1.0,
            is_paused=False,
        )
        bounds = arcade.LBWH(100, 100, 280, 500)
        try:
            self.renderer._draw_brain_why_inspector(world, bounds)
            empty_cards = tuple(captured)
            empty_limit = self.renderer._scroll_limits[
                "brain_why_inspector"
            ]

            captured.clear()
            active_snapshot = replace(
                empty_snapshot,
                behaviors=(
                    BehaviorStateSnapshot(
                        behavior=BehaviorKind.FOOD_APPROACH,
                        status=BoutStatus.ACTIVE,
                        evidence_score=0.8,
                        duration_seconds=1.0,
                        evidence=(),
                    ),
                ),
            )
            active_world = SimpleNamespace(
                **{
                    **vars(world),
                    "selected_behavior_snapshot": active_snapshot,
                }
            )
            self.renderer._draw_brain_why_inspector(
                active_world,
                bounds,
            )
            active_cards = tuple(captured)
            active_limit = self.renderer._scroll_limits[
                "brain_why_inspector"
            ]
        finally:
            self.renderer._draw_brain_why_card = original_card

        self.assertEqual(
            [item[1] for item in empty_cards],
            list(BehaviorKind),
        )
        self.assertEqual(
            [item[1] for item in active_cards],
            list(BehaviorKind),
        )
        self.assertEqual(
            [(item[0].top, item[0].bottom) for item in active_cards],
            [(item[0].top, item[0].bottom) for item in empty_cards],
        )
        self.assertEqual(active_limit, empty_limit)
        active_states = {
            behavior: state
            for _bounds, behavior, state, _snapshot, _expanded
            in active_cards
        }
        self.assertIsNotNone(
            active_states[BehaviorKind.FOOD_APPROACH]
        )
        self.assertIsNone(active_states[BehaviorKind.FEEDING])

    def test_why_expansion_reserves_data_independent_detail_height(self) -> None:
        captured = []
        original_card = self.renderer._draw_brain_why_card

        def capture_card(
            bounds,
            behavior,
            state,
            snapshot,
            *,
            expanded,
        ):
            captured.append((bounds, behavior, expanded))

        self.renderer._draw_brain_why_card = capture_card
        world = SimpleNamespace(
            config=build_sim_config(),
            selected_behavior_snapshot=BehaviorSnapshot(
                creature_id=1,
                selection_generation=1,
                simulation_time=1.0,
                behaviors=(),
                observations_processed=10,
                produced_monotonic=0.0,
            ),
            selected_why_snapshots=(),
            counterfactual_diagnostics=CounterfactualDiagnostics(
                worker_health="running"
            ),
            debug_vision_enabled=False,
            elapsed_time=1.0,
            is_paused=False,
        )
        bounds = arcade.LBWH(100, 100, 280, 500)
        try:
            self.renderer._brain_expanded_why_behavior = None
            self.renderer._draw_brain_why_inspector(world, bounds)
            collapsed_limit = self.renderer._scroll_limits[
                "brain_why_inspector"
            ]

            captured.clear()
            self.renderer._brain_expanded_why_behavior = (
                BehaviorKind.FEEDING.value
            )
            self.renderer._draw_brain_why_inspector(world, bounds)
            expanded_limit = self.renderer._scroll_limits[
                "brain_why_inspector"
            ]
        finally:
            self.renderer._draw_brain_why_card = original_card

        expanded_cards = [
            (card_bounds, behavior)
            for card_bounds, behavior, expanded in captured
            if expanded
        ]
        self.assertEqual(len(expanded_cards), 1)
        card_bounds, behavior = expanded_cards[0]
        detail_height = self.renderer._brain_why_detail_height(behavior)
        self.assertIs(behavior, BehaviorKind.FEEDING)
        self.assertAlmostEqual(
            card_bounds.height,
            self.renderer.BRAIN_WHY_CARD_HEIGHT + detail_height,
        )
        self.assertAlmostEqual(
            expanded_limit,
            collapsed_limit + detail_height,
        )

    def test_why_calculation_section_explains_formula_and_meaning(self) -> None:
        copy = " ".join(
            self.renderer._brain_why_calculation_lines(260.0)
        ).lower()

        self.assertIn("|actual − counterfactual|", copy)
        self.assertIn("natural output span", copy)
        self.assertIn("mean across scored outputs", copy)
        self.assertIn("<0.10 minimal", copy)
        self.assertIn("<0.30 weak", copy)
        self.assertIn("<0.60 moderate", copy)
        self.assertIn("mixed", copy)
        self.assertIn("do not sum to 100%", copy)
        self.assertIn("factual food heading", copy)
        self.assertIn("0.05 rad", copy)
        self.assertIn("smaller turn", copy)
        self.assertIn("factual cues strengthen", copy)
        self.assertIn("real paired probe nearest the median", copy)
        self.assertIn("factual flock center", copy)
        self.assertIn("independently masked rgb cues", copy)
        self.assertNotIn("factual response weakens", copy)

    def test_expanded_why_card_draws_result_and_waiting_placeholders(
        self,
    ) -> None:
        texts = []
        original_text = self.renderer._draw_text
        self.renderer._draw_text = (
            lambda key, text, *args, **kwargs: texts.append((key, text))
        )
        state = BehaviorStateSnapshot(
            behavior=BehaviorKind.FEEDING,
            status=BoutStatus.ACTIVE,
            evidence_score=1.0,
            duration_seconds=0.8,
            evidence=(),
            bout_id=4,
        )
        output = OutputEffect(
            output_name="want_eat",
            actual=0.8,
            counterfactual=0.2,
            delta=0.6,
            influence_score=0.6,
            direction=EffectDirection.SUPPORTIVE,
        )
        effect = SemanticEffectSnapshot(
            intervention=SemanticIntervention.VISIBLE_FOOD_CUES,
            influence_score=0.6,
            influence_label=InfluenceLabel.STRONG,
            effect_direction=EffectDirection.SUPPORTIVE,
            output_effects=(output,),
            sample_count=3,
        )
        snapshot = WhySnapshot(
            creature_id=1,
            selection_generation=1,
            brain_revision=2,
            simulation_time=3.0,
            behavior=BehaviorKind.FEEDING,
            status=BoutStatus.ACTIVE,
            bout_id=4,
            behavior_duration=0.8,
            effects=(effect,),
            produced_monotonic=0.0,
        )
        detail_height = self.renderer._brain_why_detail_height(
            BehaviorKind.FEEDING
        )
        try:
            self.renderer._draw_brain_why_card(
                arcade.LBWH(
                    100,
                    100,
                    280,
                    self.renderer.BRAIN_WHY_CARD_HEIGHT + detail_height,
                ),
                BehaviorKind.FEEDING,
                state,
                snapshot,
                expanded=True,
            )
        finally:
            self.renderer._draw_text = original_text

        copy = " ".join(text for _key, text in texts)
        self.assertIn("INFLUENCE 0.60 · STRONG · SUPPORTIVE · n=3", copy)
        self.assertIn(
            "actual +0.80  →  counterfactual +0.20  · SUPPORTIVE",
            copy,
        )
        self.assertIn("If this creature were satiated", copy)
        self.assertIn("WAITING FOR CURRENT BOUT", copy)

    def test_food_rotate_displays_raw_values_with_target_direction(
        self,
    ) -> None:
        texts = []
        original_text = self.renderer._draw_text
        self.renderer._draw_text = (
            lambda key, text, *args, **kwargs: texts.append(text)
        )
        effect = SemanticEffectSnapshot(
            intervention=SemanticIntervention.VISIBLE_FOOD_CUES,
            influence_score=0.45,
            influence_label=InfluenceLabel.MODERATE,
            effect_direction=EffectDirection.SUPPORTIVE,
            output_effects=(
                OutputEffect(
                    output_name="rotate",
                    actual=-0.04,
                    counterfactual=0.86,
                    delta=-0.90,
                    influence_score=0.45,
                    direction=EffectDirection.SUPPORTIVE,
                    actual_target_alignment=-0.04,
                    counterfactual_target_alignment=-0.86,
                ),
            ),
            sample_count=2,
        )
        try:
            self.renderer._draw_brain_why_effect_card(
                arcade.LBWH(100, 100, 280, 130),
                BehaviorKind.FOOD_ORIENTATION,
                0,
                SemanticIntervention.VISIBLE_FOOD_CUES,
                effect,
                ("rotate",),
                self.renderer.BRAIN_BEHAVIOR_ACCENTS[
                    BehaviorKind.FOOD_ORIENTATION
                ],
            )
        finally:
            self.renderer._draw_text = original_text

        copy = " ".join(texts)
        self.assertIn(
            "actual -0.04  →  counterfactual +0.86  · SUPPORTIVE",
            copy,
        )
        self.assertNotIn("target_alignment", copy)

    def test_food_why_card_reports_missing_target_context(self) -> None:
        texts = []
        original_text = self.renderer._draw_text
        self.renderer._draw_text = (
            lambda key, text, *args, **kwargs: texts.append(text)
        )
        state = BehaviorStateSnapshot(
            behavior=BehaviorKind.FOOD_APPROACH,
            status=BoutStatus.ACTIVE,
            evidence_score=0.8,
            duration_seconds=1.0,
            evidence=(),
            bout_id=2,
            target_id=None,
        )
        try:
            self.renderer._draw_brain_why_card(
                arcade.LBWH(
                    100,
                    100,
                    280,
                    self.renderer.BRAIN_WHY_CARD_HEIGHT,
                ),
                BehaviorKind.FOOD_APPROACH,
                state,
                None,
                expanded=False,
            )
        finally:
            self.renderer._draw_text = original_text

        self.assertIn("WAITING FOR TARGET", texts)

    def test_clicking_neural_node_returns_to_node_page(self) -> None:
        self.renderer._brain_inspector_page = "behaviors"
        self.renderer._brain_node_bounds[7] = arcade.LBWH(180, 180, 24, 24)

        handled = self.renderer.handle_mouse_press(
            SimpleNamespace(),
            190,
            190,
        )

        self.assertTrue(handled)
        self.assertEqual(self.renderer._brain_selected_node_key, 7)
        self.assertEqual(self.renderer._brain_inspector_page, "node")

    def test_behavior_card_click_toggles_one_expanded_card(self) -> None:
        self.renderer._brain_inspector_page = "behaviors"
        food_key = self.renderer._brain_behavior_card_hitbox_key(
            BehaviorKind.FOOD_APPROACH
        )
        rest_key = self.renderer._brain_behavior_card_hitbox_key(
            BehaviorKind.RESTING
        )
        self.renderer._control_hitboxes[food_key] = arcade.LBWH(
            310,
            180,
            100,
            50,
        )
        self.renderer._control_hitboxes[rest_key] = arcade.LBWH(
            310,
            240,
            100,
            50,
        )

        self.assertTrue(
            self.renderer.handle_mouse_press(
                SimpleNamespace(),
                350,
                200,
            )
        )
        self.assertEqual(
            self.renderer._brain_expanded_behavior,
            BehaviorKind.FOOD_APPROACH.value,
        )

        self.assertTrue(
            self.renderer.handle_mouse_press(
                SimpleNamespace(),
                350,
                260,
            )
        )
        self.assertEqual(
            self.renderer._brain_expanded_behavior,
            BehaviorKind.RESTING.value,
        )

        self.assertTrue(
            self.renderer.handle_mouse_press(
                SimpleNamespace(),
                350,
                260,
            )
        )
        self.assertIsNone(self.renderer._brain_expanded_behavior)

    def test_behavior_page_renders_all_cards_in_stable_enum_order(self) -> None:
        captured = []
        original = self.renderer._draw_brain_behavior_card
        self.renderer._draw_brain_behavior_card = (
            lambda bounds, behavior, state: captured.append((behavior, state))
        )
        snapshot = BehaviorSnapshot(
            creature_id=1,
            selection_generation=1,
            simulation_time=2.0,
            behaviors=(
                BehaviorStateSnapshot(
                    behavior=BehaviorKind.RESTING,
                    status=BoutStatus.EMERGING,
                    evidence_score=0.40,
                    duration_seconds=0.2,
                    evidence=(),
                ),
                BehaviorStateSnapshot(
                    behavior=BehaviorKind.FOOD_APPROACH,
                    status=BoutStatus.ACTIVE,
                    evidence_score=0.84,
                    duration_seconds=1.8,
                    evidence=(),
                ),
            ),
            observations_processed=20,
            produced_monotonic=0.0,
        )
        config = build_sim_config()
        world = SimpleNamespace(
            config=config,
            selected_behavior_snapshot=snapshot,
            behavior_observer_diagnostics=BehaviorObserverDiagnostics(
                worker_health="running"
            ),
            debug_vision_enabled=False,
            elapsed_time=2.0,
            is_paused=False,
        )
        try:
            self.renderer._draw_brain_behavior_inspector(
                world,
                arcade.LBWH(100, 100, 280, 500),
            )
        finally:
            self.renderer._draw_brain_behavior_card = original

        self.assertEqual(
            [behavior for behavior, _state in captured],
            list(BehaviorKind),
        )
        states = {
            behavior: state
            for behavior, state in captured
        }
        self.assertIs(states[BehaviorKind.FOOD_APPROACH].status, BoutStatus.ACTIVE)
        self.assertIs(states[BehaviorKind.RESTING].status, BoutStatus.EMERGING)
        self.assertIsNone(states[BehaviorKind.FEEDING])
        self.assertEqual(
            set(self.renderer.BRAIN_BEHAVIOR_ACCENTS),
            set(BehaviorKind),
        )
        self.assertEqual(
            len(set(self.renderer.BRAIN_BEHAVIOR_ACCENTS.values())),
            len(BehaviorKind),
        )
        self.assertEqual(
            set(self.renderer.BRAIN_BEHAVIOR_ACTIVATION_COPY),
            set(BehaviorKind),
        )
        self.assertGreater(
            self.renderer._scroll_limits["brain_behavior_inspector"],
            0.0,
        )
        self.assertIn(
            "brain_behavior_inspector",
            self.renderer._scroll_regions,
        )

    def test_behavior_page_expands_one_card_with_activation_copy(self) -> None:
        captured = []
        wrap_calls = []
        original = self.renderer._draw_brain_behavior_card
        original_wrap = self.renderer._wrap_line

        def capture_card(bounds, behavior, state, **kwargs):
            captured.append(
                (
                    bounds,
                    behavior,
                    kwargs.get("detail_lines", ()),
                )
            )

        def capture_wrap(text, width, **kwargs):
            wrap_calls.append(text)
            return original_wrap(text, width, **kwargs)

        self.renderer._draw_brain_behavior_card = capture_card
        self.renderer._wrap_line = capture_wrap
        snapshot = BehaviorSnapshot(
            creature_id=1,
            selection_generation=1,
            simulation_time=2.0,
            behaviors=(
                BehaviorStateSnapshot(
                    behavior=BehaviorKind.FEEDING,
                    status=BoutStatus.ACTIVE,
                    evidence_score=1.0,
                    duration_seconds=0.5,
                    evidence=(),
                ),
            ),
            observations_processed=20,
            produced_monotonic=0.0,
        )
        world = SimpleNamespace(
            config=build_sim_config(),
            selected_behavior_snapshot=snapshot,
            behavior_observer_diagnostics=BehaviorObserverDiagnostics(
                worker_health="running"
            ),
            debug_vision_enabled=False,
            elapsed_time=2.0,
            is_paused=False,
        )
        bounds = arcade.LBWH(100, 100, 280, 500)
        try:
            self.renderer._brain_expanded_behavior = None
            self.renderer._draw_brain_behavior_inspector(world, bounds)
            collapsed_limit = self.renderer._scroll_limits[
                "brain_behavior_inspector"
            ]

            captured.clear()
            self.renderer._brain_expanded_behavior = (
                BehaviorKind.FEEDING.value
            )
            self.renderer._draw_brain_behavior_inspector(world, bounds)
            expanded_limit = self.renderer._scroll_limits[
                "brain_behavior_inspector"
            ]
        finally:
            self.renderer._draw_brain_behavior_card = original
            self.renderer._wrap_line = original_wrap

        detailed = [
            (card_bounds, behavior, detail_lines)
            for card_bounds, behavior, detail_lines in captured
            if detail_lines
        ]
        self.assertEqual(len(detailed), 1)
        card_bounds, behavior, detail_lines = detailed[0]
        self.assertIs(behavior, BehaviorKind.FEEDING)
        self.assertIn("consumption", " ".join(detail_lines).lower())
        detail_wraps = [
            text
            for text in wrap_calls
            if text in self.renderer.BRAIN_BEHAVIOR_ACTIVATION_COPY.values()
        ]
        self.assertEqual(
            detail_wraps,
            [
                self.renderer.BRAIN_BEHAVIOR_ACTIVATION_COPY[
                    BehaviorKind.FEEDING
                ],
            ],
        )
        detail_height = self.renderer._brain_behavior_detail_height(
            detail_lines
        )
        self.assertAlmostEqual(
            card_bounds.height,
            self.renderer.BRAIN_BEHAVIOR_CARD_HEIGHT + detail_height,
        )
        self.assertAlmostEqual(
            expanded_limit,
            collapsed_limit + detail_height,
        )
        self.assertIn(
            self.renderer._brain_behavior_card_hitbox_key(
                BehaviorKind.FEEDING
            ),
            self.renderer._control_hitboxes,
        )

    def test_behavior_card_uses_state_intensity_and_clamped_evidence(self) -> None:
        active = BehaviorStateSnapshot(
            behavior=BehaviorKind.FEEDING,
            status=BoutStatus.ACTIVE,
            evidence_score=1.4,
            duration_seconds=2.25,
            evidence=(),
        )
        emerging = replace(
            active,
            status=BoutStatus.EMERGING,
            evidence_score=0.5,
        )

        self.assertEqual(self.renderer._behavior_display_intensity(None), 0.0)
        self.assertAlmostEqual(
            self.renderer._behavior_display_intensity(emerging),
            0.575,
        )
        self.assertAlmostEqual(
            self.renderer._behavior_display_intensity(active),
            1.0,
        )
        self.assertEqual(
            self.renderer._clamped_behavior_evidence(float("nan")),
            0.0,
        )

        texts = []
        text_sizes = {}
        fills = []
        original_text = self.renderer._draw_text
        original_fill = self.renderer._draw_rounded_rect_fill

        def capture_text(key, text, *args, **kwargs):
            texts.append(text)
            text_sizes[key] = args[3]

        self.renderer._draw_text = capture_text
        self.renderer._draw_rounded_rect_fill = (
            lambda bounds, color, radius: fills.append(bounds)
        )
        bounds = arcade.LBWH(
            100,
            100,
            300,
            self.renderer.BRAIN_BEHAVIOR_CARD_HEIGHT,
        )
        try:
            self.renderer._draw_brain_behavior_card(
                bounds,
                BehaviorKind.FEEDING,
                active,
            )
            self.assertIn("EVIDENCE 1.00", texts)
            self.assertIn("ACTIVE · 2.2 s", texts)
            self.assertEqual(len(fills), 2)
            self.assertAlmostEqual(fills[0].width, fills[1].width)
            self.assertGreaterEqual(
                text_sizes["brain_behavior_feeding_name"],
                13.0,
            )
            self.assertGreaterEqual(
                text_sizes["brain_behavior_feeding_evidence"],
                10.0,
            )
            self.assertGreaterEqual(
                text_sizes["brain_behavior_feeding_state"],
                10.0,
            )
            self.assertGreaterEqual(
                self.renderer.BRAIN_BEHAVIOR_CARD_HEIGHT,
                88.0,
            )
            self.assertGreaterEqual(
                self.renderer.BRAIN_BEHAVIOR_CARD_GAP,
                10.0,
            )
            self.assertGreaterEqual(
                self.renderer.BRAIN_INSPECTOR_MIN_WIDTH,
                320.0,
            )

            texts.clear()
            fills.clear()
            self.renderer._draw_brain_behavior_card(
                bounds,
                BehaviorKind.FEEDING,
                None,
            )
            self.assertIn("EVIDENCE —", texts)
            self.assertIn("INACTIVE", texts)
            self.assertEqual(len(fills), 1)

            texts.clear()
            detail_lines = ("Explicit food consumption is recorded.",)
            expanded_bounds = arcade.LBWH(
                bounds.left,
                bounds.bottom,
                bounds.width,
                (
                    self.renderer.BRAIN_BEHAVIOR_CARD_HEIGHT
                    + self.renderer._brain_behavior_detail_height(
                        detail_lines
                    )
                ),
            )
            self.renderer._draw_brain_behavior_card(
                expanded_bounds,
                BehaviorKind.FEEDING,
                active,
                detail_lines=detail_lines,
            )
            self.assertIn("ACTIVATES WHEN", texts)
            self.assertIn(detail_lines[0], texts)
            self.assertIn("−", texts)
        finally:
            self.renderer._draw_text = original_text
            self.renderer._draw_rounded_rect_fill = original_fill

    def test_behavior_page_renders_empty_error_delayed_and_debug_states(
        self,
    ) -> None:
        notices = []
        cards = []
        diagnostics_drawn = []
        original_notice = self.renderer._draw_brain_behavior_notice
        original_card = self.renderer._draw_brain_behavior_card
        original_diagnostics = self.renderer._draw_brain_behavior_diagnostics
        self.renderer._draw_brain_behavior_notice = (
            lambda bounds, index, title, message, **kwargs: notices.append(title)
        )
        self.renderer._draw_brain_behavior_card = (
            lambda bounds, behavior, state: cards.append((behavior, state))
        )
        self.renderer._draw_brain_behavior_diagnostics = (
            lambda bounds, diagnostics: diagnostics_drawn.append(diagnostics)
        )
        config = build_sim_config()
        bounds = arcade.LBWH(100, 100, 280, 500)
        try:
            collecting = SimpleNamespace(
                config=config,
                selected_behavior_snapshot=None,
                behavior_observer_diagnostics=BehaviorObserverDiagnostics(
                    worker_health="running"
                ),
                debug_vision_enabled=False,
                elapsed_time=1.0,
                is_paused=False,
            )
            self.renderer._draw_brain_behavior_inspector(
                collecting,
                bounds,
            )
            self.assertIn("Collecting temporal evidence", notices)
            self.assertEqual(len(cards), len(BehaviorKind))
            self.assertTrue(all(state is None for _behavior, state in cards))

            notices.clear()
            cards.clear()
            disabled = SimpleNamespace(
                **{
                    **vars(collecting),
                    "config": SimpleNamespace(
                        behavior=SimpleNamespace(enabled=False)
                    ),
                }
            )
            self.renderer._draw_brain_behavior_inspector(disabled, bounds)
            self.assertIn("Observer disabled", notices)
            self.assertEqual(len(cards), len(BehaviorKind))
            self.assertTrue(all(state is None for _behavior, state in cards))

            notices.clear()
            cards.clear()
            error = SimpleNamespace(
                **{
                    **vars(collecting),
                    "behavior_observer_diagnostics": (
                        BehaviorObserverDiagnostics(
                            worker_health="error",
                            last_error="worker failed",
                        )
                    ),
                },
            )
            self.renderer._draw_brain_behavior_inspector(error, bounds)
            self.assertIn("Observer unavailable", notices)
            self.assertEqual(len(cards), len(BehaviorKind))
            self.assertTrue(all(state is None for _behavior, state in cards))

            notices.clear()
            cards.clear()
            snapshot = BehaviorSnapshot(
                creature_id=1,
                selection_generation=1,
                simulation_time=0.0,
                behaviors=(),
                observations_processed=1,
                produced_monotonic=0.0,
            )
            delayed = SimpleNamespace(
                **{
                    **vars(collecting),
                    "selected_behavior_snapshot": snapshot,
                    "debug_vision_enabled": True,
                    "elapsed_time": 1.0,
                }
            )
            self.renderer._draw_brain_behavior_inspector(delayed, bounds)
            self.assertIn("Observer updating", notices)
            self.assertNotIn("No sustained bout detected", notices)
            self.assertEqual(len(cards), len(BehaviorKind))
            self.assertEqual(len(diagnostics_drawn), 1)
        finally:
            self.renderer._draw_brain_behavior_notice = original_notice
            self.renderer._draw_brain_behavior_card = original_card
            self.renderer._draw_brain_behavior_diagnostics = original_diagnostics

    def test_behavior_status_slot_keeps_cards_stable_when_bout_appears(
        self,
    ) -> None:
        notices = []
        cards = []
        original_notice = self.renderer._draw_brain_behavior_notice
        original_card = self.renderer._draw_brain_behavior_card
        self.renderer._draw_brain_behavior_notice = (
            lambda bounds, index, title, message_lines, **kwargs: notices.append(
                (bounds, title, message_lines)
            )
        )
        self.renderer._draw_brain_behavior_card = (
            lambda bounds, behavior, state: cards.append(bounds)
        )
        config = build_sim_config()
        snapshot = BehaviorSnapshot(
            creature_id=1,
            selection_generation=1,
            simulation_time=1.0,
            behaviors=(),
            observations_processed=10,
            produced_monotonic=0.0,
        )
        world = SimpleNamespace(
            config=config,
            selected_behavior_snapshot=snapshot,
            behavior_observer_diagnostics=BehaviorObserverDiagnostics(
                worker_health="running"
            ),
            debug_vision_enabled=False,
            elapsed_time=1.0,
            is_paused=False,
        )
        try:
            self.renderer._draw_brain_behavior_inspector(
                world,
                arcade.LBWH(100, 100, 200, 500),
            )
            empty_notice = notices[0]
            empty_cards = tuple(cards)
            empty_scroll_limit = self.renderer._scroll_limits[
                "brain_behavior_inspector"
            ]

            notices.clear()
            cards.clear()
            active_snapshot = replace(
                snapshot,
                behaviors=(
                    BehaviorStateSnapshot(
                        behavior=BehaviorKind.FEEDING,
                        status=BoutStatus.ACTIVE,
                        evidence_score=1.0,
                        duration_seconds=0.5,
                        evidence=(),
                    ),
                ),
            )
            active_world = SimpleNamespace(
                **{
                    **vars(world),
                    "selected_behavior_snapshot": active_snapshot,
                }
            )
            self.renderer._draw_brain_behavior_inspector(
                active_world,
                arcade.LBWH(100, 100, 200, 500),
            )
            active_notice = notices[0]
            active_cards = tuple(cards)
            active_scroll_limit = self.renderer._scroll_limits[
                "brain_behavior_inspector"
            ]
        finally:
            self.renderer._draw_brain_behavior_notice = original_notice
            self.renderer._draw_brain_behavior_card = original_card

        empty_bounds, empty_title, empty_lines = empty_notice
        active_bounds, active_title, active_lines = active_notice
        self.assertEqual(empty_title, "No sustained bout detected")
        self.assertEqual(active_title, "Live bouts detected")
        self.assertLessEqual(len(empty_lines), 2)
        self.assertLessEqual(len(active_lines), 2)
        self.assertEqual(
            empty_bounds.height,
            self.renderer.BRAIN_BEHAVIOR_NOTICE_HEIGHT,
        )
        self.assertEqual(active_bounds.height, empty_bounds.height)
        self.assertAlmostEqual(
            empty_bounds.bottom - self.renderer.BRAIN_BEHAVIOR_CARD_GAP,
            empty_cards[0].top,
        )
        self.assertEqual(
            [(card.top, card.bottom) for card in active_cards],
            [(card.top, card.bottom) for card in empty_cards],
        )
        self.assertEqual(active_scroll_limit, empty_scroll_limit)

    def test_behavior_page_has_independent_scroll_offset(self) -> None:
        self.renderer._brain_inspector_page = "behaviors"
        self.renderer._scroll_regions[
            "brain_behavior_inspector"
        ] = arcade.LBWH(200, 180, 200, 120)
        self.renderer._scroll_limits["brain_behavior_inspector"] = 100

        handled = self.renderer.handle_mouse_scroll(250, 220, -1)

        self.assertTrue(handled)
        self.assertEqual(
            self.renderer._scroll_offsets["brain_behavior_inspector"],
            24,
        )
        self.assertEqual(
            self.renderer._brain_behavior_scroll_offset,
            24,
        )
        self.assertEqual(
            self.renderer._scroll_offsets.get("brain_node_inspector", 0.0),
            0.0,
        )

    def test_brain_output_readout_labels_centered_values(self) -> None:
        readout = self.renderer._brain_output_readout([-0.25, 0.5])

        self.assertIn("Centered outputs: -0.25/0.50", readout)
        self.assertNotIn("Raw outputs", readout)

    def test_scroll_over_non_graph_brain_window_is_consumed(self) -> None:
        handled = self.renderer.handle_mouse_scroll(200, 380, 1)

        self.assertTrue(handled)
        self.assertAlmostEqual(self.renderer._brain_graph_zoom, 1.0)

    def test_scroll_outside_ui_is_not_consumed(self) -> None:
        handled = self.renderer.handle_mouse_scroll(20, 20, 1)

        self.assertFalse(handled)
        self.assertAlmostEqual(self.renderer._brain_graph_zoom, 1.0)

    def test_drag_inside_graph_is_consumed_without_moving_graph(self) -> None:
        world = SimpleNamespace()
        graph_bounds = self.renderer._control_hitboxes["brain_window_graph"]
        graph_position = (graph_bounds.center_x + 40, graph_bounds.center_y - 25)
        before = self.renderer._brain_graph_screen_position(
            graph_position,
            graph_bounds,
        )

        pressed = self.renderer.handle_mouse_press(world, 200, 200)
        dragged = self.renderer.handle_mouse_drag(world, 260, 240)
        after = self.renderer._brain_graph_screen_position(
            graph_position,
            graph_bounds,
        )

        self.assertTrue(pressed)
        self.assertFalse(dragged)
        self.assertEqual(after, before)

    def test_closed_brain_window_bounds_do_not_consume_scroll(self) -> None:
        self.renderer._brain_window_open = False
        self.renderer._control_hitboxes.clear()

        handled = self.renderer.handle_mouse_scroll(200, 380, 1)

        self.assertFalse(handled)

    def test_default_brain_window_uses_large_centered_workspace(self) -> None:
        world = SimpleNamespace(
            layout=SimpleNamespace(
                environment=arcade.LBWH(340, 20, 1000, 700),
                window=arcade.LBWH(0, 0, 1440, 900),
            )
        )

        self.renderer._brain_window_bounds = None
        self.renderer._ensure_brain_window_bounds(world)

        bounds = self.renderer._brain_window_bounds
        self.assertIsNotNone(bounds)
        assert bounds is not None
        self.assertEqual(bounds.left, 40)
        self.assertEqual(bounds.bottom, 40)
        self.assertEqual(bounds.width, 1360)
        self.assertEqual(bounds.height, 820)

    def test_clicking_node_selects_it_and_empty_graph_clears_it(self) -> None:
        self.renderer._brain_node_bounds[7] = arcade.LBWH(180, 180, 24, 24)
        world = SimpleNamespace()

        selected = self.renderer.handle_mouse_press(world, 190, 190)
        cleared = self.renderer.handle_mouse_press(world, 300, 200)

        self.assertTrue(selected)
        self.assertTrue(cleared)
        self.assertIsNone(self.renderer._brain_selected_node_key)

    def test_inspector_toggle_preserves_selected_node(self) -> None:
        self.renderer._brain_selected_node_key = 7
        self.renderer._control_hitboxes["brain_node_inspector_toggle"] = arcade.LBWH(
            450,
            350,
            24,
            24,
        )
        world = SimpleNamespace()

        handled = self.renderer.handle_mouse_press(world, 460, 360)

        self.assertTrue(handled)
        self.assertFalse(self.renderer._brain_node_inspector_open)
        self.assertEqual(self.renderer._brain_selected_node_key, 7)

        reopened = self.renderer.handle_mouse_press(world, 460, 360)

        self.assertTrue(reopened)
        self.assertTrue(self.renderer._brain_node_inspector_open)
        self.assertEqual(self.renderer._brain_selected_node_key, 7)

    def test_closing_brain_window_clears_selection_but_preserves_panel_preference(
        self,
    ) -> None:
        self.renderer._brain_selected_node_key = 7
        self.renderer._brain_selection_identity = (5, 10)
        self.renderer._brain_node_inspector_open = False
        self.renderer._brain_inspector_page = "behaviors"
        self.renderer._brain_behavior_scroll_offset = 48.0
        self.renderer._brain_expanded_behavior = BehaviorKind.COHESION.value
        self.renderer._brain_expanded_why_behavior = (
            BehaviorKind.FEEDING.value
        )
        self.renderer._brain_connection_direction = "incoming"
        self.renderer._brain_connection_filter = "active"
        self.renderer._brain_connection_sort_descending = False
        self.renderer._control_hitboxes["brain_window_close"] = arcade.LBWH(
            450,
            350,
            24,
            24,
        )

        handled = self.renderer.handle_mouse_press(SimpleNamespace(), 460, 360)

        self.assertTrue(handled)
        self.assertFalse(self.renderer._brain_window_open)
        self.assertIsNone(self.renderer._brain_selected_node_key)
        self.assertIsNone(self.renderer._brain_selection_identity)
        self.assertFalse(self.renderer._brain_node_inspector_open)
        self.assertEqual(self.renderer._brain_inspector_page, "node")
        self.assertEqual(self.renderer._brain_connection_direction, "both")
        self.assertEqual(self.renderer._brain_connection_filter, "all")
        self.assertTrue(self.renderer._brain_connection_sort_descending)
        self.assertEqual(self.renderer._brain_behavior_scroll_offset, 0.0)
        self.assertIsNone(self.renderer._brain_expanded_behavior)
        self.assertIsNone(self.renderer._brain_expanded_why_behavior)

    def test_scroll_inside_node_inspector_updates_its_offset(self) -> None:
        self.renderer._scroll_regions["brain_node_inspector"] = arcade.LBWH(
            200,
            180,
            200,
            120,
        )
        self.renderer._scroll_limits["brain_node_inspector"] = 100

        handled = self.renderer.handle_mouse_scroll(250, 220, -1)

        self.assertTrue(handled)
        self.assertEqual(self.renderer._scroll_offsets["brain_node_inspector"], 24)

    def make_brain_world(self, *, branched: bool = False) -> SimpleNamespace:
        gene = lambda: SimpleNamespace(
            activation="tanh",
            aggregation="sum",
            bias=0.25,
            response=1.0,
        )
        nodes = {0: gene(), 1: gene()}
        connection_specs = [(-1, 1, 0.8, True), (1, 0, -0.6, True)]
        if branched:
            nodes[2] = gene()
            connection_specs.extend([(-1, 2, 0.4, True), (2, 0, 0.3, True)])
        connections = {
            (source, target): SimpleNamespace(
                key=(source, target),
                weight=weight,
                enabled=enabled,
            )
            for source, target, weight, enabled in connection_specs
        }
        genome = SimpleNamespace(nodes=nodes, connections=connections)
        brain = SimpleNamespace(
            genome_id=10,
            genome=genome,
            last_inputs=[0.5],
            last_outputs=[0.25],
            last_action=None,
            sensor_usage=lambda input_keys, output_keys: tuple(
                SimpleNamespace(
                    current_value=0.5,
                    has_enabled_path=True,
                    reachable_action_outputs=("accelerate",),
                )
                for _ in input_keys
            ),
        )
        selected = SimpleNamespace(creature_id=5, name="Herbivore 5")
        world = SimpleNamespace(
            selected_creature=selected,
            neat_controller=SimpleNamespace(
                brain_for=lambda creature_id: brain,
                config=SimpleNamespace(
                    genome_config=SimpleNamespace(input_keys=[-1], output_keys=[0])
                ),
            ),
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, 1440, 900),
                environment=arcade.LBWH(0, 0, 1440, 900),
            ),
            config=self.renderer.config,
            fitness_for=lambda creature: None,
        )
        return SimpleNamespace(world=world, selected=selected, brain=brain)

    def test_inspector_lines_show_static_metadata_and_disabled_connections(self) -> None:
        fixture = self.make_brain_world()
        fixture.brain.genome.connections[(-1, 1)].enabled = False
        layout = build_brain_graph_layout(
            fixture.brain.genome,
            [-1],
            [0],
            arcade.LBWH(0, 0, 600, 300),
            ["sensor"],
            ["accelerate"],
        )

        lines = self.renderer._brain_node_inspector_lines(
            fixture.brain,
            layout,
            layout.nodes[1],
        )
        text = "\n".join(lines)

        self.assertIn("Layer: Hidden 1", text)
        self.assertIn("Activation: tanh", text)
        self.assertIn("sensor [ID -1] | +0.800 | Disabled", text)
        self.assertIn("accelerate [ID 0] | -0.600 | Enabled", text)
        self.assertIn("ADDITIONAL ENABLED SIGNAL ROUTE (0)", text)
        self.assertNotIn("Current value", text)
        self.assertNotIn("Contrib", text)

    def test_inspector_separates_direct_genes_from_additional_signal_route(
        self,
    ) -> None:
        fixture = self.make_brain_world()
        fixture.brain.genome.nodes[2] = SimpleNamespace(
            activation="tanh",
            aggregation="sum",
            bias=0.25,
            response=1.0,
        )
        fixture.brain.genome.connections = {
            key: SimpleNamespace(key=key, weight=weight, enabled=True)
            for key, weight in (
                ((-1, 1), 0.8),
                ((1, 2), -0.6),
                ((2, 0), 0.3),
            )
        }
        layout = build_brain_graph_layout(
            fixture.brain.genome,
            [-1],
            [0],
            arcade.LBWH(0, 0, 600, 300),
            ["sensor"],
            ["accelerate"],
        )

        lines = self.renderer._brain_node_inspector_lines(
            fixture.brain,
            layout,
            layout.nodes[1],
        )
        text = "\n".join(lines)

        self.assertIn("sensor [ID -1] | +0.800 | Enabled", text)
        self.assertIn("Hidden 2 [ID 2] | -0.600 | Enabled", text)
        self.assertIn("ADDITIONAL ENABLED SIGNAL ROUTE (1)", text)
        self.assertIn(
            "Downstream: Hidden 2 [ID 2] -> accelerate [ID 0] "
            "| +0.300 | Enabled",
            text,
        )

    def test_structured_connection_view_filters_and_sorts_enabled_genes(self) -> None:
        fixture = self.make_brain_world()
        fixture.brain.genome.nodes[2] = SimpleNamespace(
            activation="tanh",
            aggregation="sum",
            bias=0.0,
            response=1.0,
        )
        fixture.brain.genome.connections[(2, 1)] = SimpleNamespace(
            key=(2, 1),
            weight=-0.5,
            enabled=False,
        )
        layout = build_brain_graph_layout(
            fixture.brain.genome,
            [-1],
            [0],
            arcade.LBWH(0, 0, 600, 300),
            ["sensor"],
            ["accelerate"],
        )
        view = self.renderer._brain_node_inspector_view(
            fixture.brain,
            layout,
            layout.nodes[1],
        )

        self.assertEqual(len(view.incoming_rows), 2)
        self.renderer._brain_connection_filter = "active"
        active = self.renderer._brain_visible_connection_rows(view.incoming_rows)
        self.assertEqual([row.endpoint_key for row in active], [-1])

        self.renderer._brain_connection_filter = "all"
        descending = self.renderer._brain_visible_connection_rows(view.incoming_rows)
        self.assertEqual([row.weight for row in descending], [0.8, -0.5])
        self.renderer._brain_connection_sort_descending = False
        ascending = self.renderer._brain_visible_connection_rows(view.incoming_rows)
        self.assertEqual([row.weight for row in ascending], [-0.5, 0.8])

    def test_structured_connection_renderer_honors_direction_and_counts(self) -> None:
        fixture = self.make_brain_world()
        fixture.brain.genome.connections[(-1, 1)].enabled = False
        layout = build_brain_graph_layout(
            fixture.brain.genome,
            [-1],
            [0],
            arcade.LBWH(0, 0, 600, 300),
            ["sensor_with_a_name_that_needs_truncation"],
            ["accelerate"],
        )
        view = self.renderer._brain_node_inspector_view(
            fixture.brain,
            layout,
            layout.nodes[1],
        )
        self.renderer._brain_connection_direction = "incoming"
        self.renderer._brain_connection_filter = "active"

        self.renderer._draw_brain_connection_inspector_content(
            arcade.LBWH(100, 100, 288, 420),
            view,
            weight_scale=5.0,
        )

        self.assertEqual(
            self.renderer._text_cache["brain_connection_incoming_title"].text,
            "INCOMING CONNECTIONS (0 / 1)",
        )
        self.assertEqual(
            self.renderer._text_cache["brain_connection_incoming_empty"].text,
            "No active incoming connections",
        )
        self.assertNotIn("brain_connection_outgoing_title", self.renderer._text_cache)
        self.assertIn("brain_connection_route_title", self.renderer._text_cache)
        self.assertIn(
            "brain_connection_direction_incoming",
            self.renderer._control_hitboxes,
        )

    def test_connection_weight_colors_use_fixed_signed_strength_scale(self) -> None:
        neutral = (180, 188, 200)

        self.assertEqual(
            self.renderer._brain_connection_weight_color(None, 5.0),
            neutral,
        )
        self.assertEqual(
            self.renderer._brain_connection_weight_color(0.1, 5.0),
            neutral,
        )
        weak_positive = self.renderer._brain_connection_weight_color(0.5, 5.0)
        self.assertNotEqual(weak_positive, neutral)
        self.assertEqual(
            self.renderer._brain_connection_weight_color(5.0, 5.0),
            (43, 108, 246),
        )
        self.assertEqual(
            self.renderer._brain_connection_weight_color(-5.0, 5.0),
            (245, 62, 62),
        )
        self.assertEqual(
            self.renderer._brain_connection_weight_color(float("nan"), 0.0),
            neutral,
        )

    def test_connection_rows_use_larger_node_font_and_inner_padding(self) -> None:
        fixture = self.make_brain_world()
        layout = build_brain_graph_layout(
            fixture.brain.genome,
            [-1],
            [0],
            arcade.LBWH(0, 0, 600, 300),
            ["sensor"],
            ["accelerate"],
        )
        view = self.renderer._brain_node_inspector_view(
            fixture.brain,
            layout,
            layout.nodes[1],
        )
        bounds = arcade.LBWH(100, 100, 288, 520)

        self.renderer._draw_brain_connection_inspector_content(
            bounds,
            view,
            weight_scale=5.0,
        )

        endpoint = self.renderer._text_cache[
            "brain_connection_incoming_0_endpoint"
        ]
        status = self.renderer._text_cache[
            "brain_connection_incoming_0_status"
        ]
        self.assertEqual(endpoint.font_size, 11.5)
        self.assertFalse(endpoint.bold)
        self.assertEqual(status.font_size, 8.5)
        self.assertFalse(status.bold)
        self.assertEqual(
            status.color,
            self.renderer._brain_connection_weight_color(0.8, 5.0),
        )
        self.assertGreaterEqual(
            endpoint.x,
            bounds.left + self.renderer.BRAIN_CONNECTION_HORIZONTAL_PADDING,
        )
        self.assertEqual(self.renderer.BRAIN_CONNECTION_ROW_HEIGHT, 40.0)

        positive_background = (
            self.renderer._brain_connection_row_background_color(
                2.0,
                5.0,
                enabled=True,
            )
        )
        negative_background = (
            self.renderer._brain_connection_row_background_color(
                -2.0,
                5.0,
                enabled=True,
            )
        )
        self.assertNotEqual(positive_background, negative_background)

    def test_connection_controls_update_state_and_reset_scroll(self) -> None:
        self.renderer._brain_inspector_page = "node"
        self.renderer._scroll_offsets["brain_node_inspector"] = 72.0
        self.renderer._control_hitboxes[
            "brain_connection_direction_outgoing"
        ] = arcade.LBWH(180, 180, 50, 24)

        handled = self.renderer.handle_mouse_press(SimpleNamespace(), 200, 190)

        self.assertTrue(handled)
        self.assertEqual(self.renderer._brain_connection_direction, "outgoing")
        self.assertEqual(self.renderer._scroll_offsets["brain_node_inspector"], 0.0)

        self.renderer._control_hitboxes[
            "brain_connection_filter_active"
        ] = arcade.LBWH(240, 180, 50, 24)
        self.renderer.handle_mouse_press(SimpleNamespace(), 260, 190)
        self.assertEqual(self.renderer._brain_connection_filter, "active")

        self.renderer._control_hitboxes[
            "brain_connection_sort_weight"
        ] = arcade.LBWH(300, 180, 70, 24)
        self.renderer.handle_mouse_press(SimpleNamespace(), 330, 190)
        self.assertFalse(self.renderer._brain_connection_sort_descending)

    def test_workspace_registers_nodes_and_expands_when_inspector_collapses(self) -> None:
        fixture = self.make_brain_world()

        self.renderer._draw_brain_window(fixture.world)
        open_width = self.renderer._control_hitboxes["brain_window_graph"].width
        self.assertEqual(set(self.renderer._brain_node_bounds), {-1, 0, 1})
        self.assertTrue(
            all(bounds.width >= 20 for bounds in self.renderer._brain_node_bounds.values())
        )
        input_bounds = self.renderer._brain_node_bounds[-1]
        hidden_bounds = self.renderer._brain_node_bounds[1]
        self.assertGreater(input_bounds.width, hidden_bounds.width)
        self.assertTrue(
            self.renderer.handle_mouse_press(
                fixture.world,
                input_bounds.left + 2.0,
                input_bounds.center_y,
            )
        )
        self.assertEqual(self.renderer._brain_selected_node_key, -1)

        self.renderer._brain_node_inspector_open = False
        self.renderer._draw_brain_window(fixture.world)
        collapsed_width = self.renderer._control_hitboxes["brain_window_graph"].width

        self.assertGreater(collapsed_width, open_width)

    def test_brain_graph_layout_is_reused_for_stable_genome_and_bounds(
        self,
    ) -> None:
        fixture = self.make_brain_world()
        bounds = arcade.LBWH(20, 30, 600, 300)

        with patch(
            "src.ui.components.brain.graph.build_brain_graph_layout",
            wraps=build_brain_graph_layout,
        ) as build:
            first = self.renderer._brain_graph_layout(
                fixture.selected,
                fixture.brain,
                [-1],
                [0],
                bounds,
            )
            second = self.renderer._brain_graph_layout(
                fixture.selected,
                fixture.brain,
                [-1],
                [0],
                bounds,
            )
            resized = self.renderer._brain_graph_layout(
                fixture.selected,
                fixture.brain,
                [-1],
                [0],
                arcade.LBWH(20, 30, 640, 300),
            )

        self.assertIs(first, second)
        self.assertIsNot(first, resized)
        self.assertEqual(build.call_count, 2)

    def test_brain_highlight_is_shared_for_stable_layout_and_node(self) -> None:
        fixture = self.make_brain_world()
        layout = build_brain_graph_layout(
            fixture.brain.genome,
            [-1],
            [0],
            arcade.LBWH(0, 0, 600, 300),
            ["sensor"],
            ["accelerate"],
        )

        with patch(
            "src.ui.components.brain.graph.highlighted_path_through_node",
            wraps=highlighted_path_through_node,
        ) as build:
            first = self.renderer._brain_highlight_for_node(layout, 1)
            second = self.renderer._brain_highlight_for_node(layout, 1)

        self.assertIs(first, second)
        build.assert_called_once_with(layout, 1)

    def test_overlapping_node_hitboxes_select_nearest_node_center(self) -> None:
        self.renderer._brain_node_bounds = {
            1: arcade.LBWH(100, 90, 24, 24),
            2: arcade.LBWH(100, 100, 24, 24),
        }

        self.assertEqual(self.renderer._brain_node_at(112, 110), 2)

    def test_selected_node_draws_all_five_connection_detail_tiers(self) -> None:
        fixture = self.make_brain_world()
        node_gene = fixture.brain.genome.nodes[1]
        fixture.brain.genome.nodes.update(
            {
                key: SimpleNamespace(
                    activation=node_gene.activation,
                    aggregation=node_gene.aggregation,
                    bias=node_gene.bias,
                    response=node_gene.response,
                )
                for key in range(2, 6)
            }
        )
        connection_specs = (
            (-1, 1, 0.8, True),
            (1, 2, 0.7, True),
            (2, 0, 0.6, True),
            (-1, 3, 0.5, True),
            (3, 0, 0.4, True),
            (1, 4, 0.3, False),
            (-1, 5, 0.2, False),
        )
        fixture.brain.genome.connections = {
            (source, target): SimpleNamespace(
                key=(source, target),
                weight=weight,
                enabled=enabled,
            )
            for source, target, weight, enabled in connection_specs
        }
        self.renderer._brain_selected_node_key = 1
        self.renderer._brain_selection_identity = (5, 10)
        calls: dict[tuple[int, int], dict[str, object]] = {}
        original = self.renderer._draw_brain_graph_edge

        def capture_edge(*args: object, **kwargs: object) -> None:
            edge = args[0]
            calls[(edge.source, edge.target)] = kwargs

        self.renderer._draw_brain_graph_edge = capture_edge
        try:
            self.renderer._draw_brain_graph(
                fixture.world,
                arcade.LBWH(0, 0, 800, 500),
            )
        finally:
            self.renderer._draw_brain_graph_edge = original

        self.assertTrue(calls[(-1, 5)]["disabled"])
        self.assertFalse(calls[(-1, 5)]["direct"])
        self.assertTrue(calls[(-1, 3)]["dimmed"])
        self.assertTrue(calls[(2, 0)]["highlighted"])
        self.assertFalse(calls[(2, 0)]["direct"])
        self.assertTrue(calls[(1, 4)]["disabled"])
        self.assertTrue(calls[(1, 4)]["direct"])
        self.assertTrue(calls[(-1, 1)]["highlighted"])
        self.assertTrue(calls[(-1, 1)]["direct"])

    def test_changing_genome_clears_node_selection(self) -> None:
        fixture = self.make_brain_world()
        layout = build_brain_graph_layout(
            fixture.brain.genome,
            [-1],
            [0],
            arcade.LBWH(0, 0, 600, 300),
            ["sensor"],
            ["accelerate"],
        )
        self.renderer._brain_selected_node_key = 1
        self.renderer._brain_selection_identity = (5, 10)
        replacement_brain = SimpleNamespace(genome_id=11)

        self.renderer._sync_brain_graph_selection(
            fixture.selected,
            replacement_brain,
            layout,
        )

        self.assertIsNone(self.renderer._brain_selected_node_key)

    def test_reference_lane_geometry_leaves_wide_routing_gaps(self) -> None:
        bounds = arcade.LBWH(0, 0, 1000, 500)

        lanes = self.renderer._brain_graph_lane_bounds(bounds)

        input_lane = lanes[BrainNodeKind.INPUT]
        hidden_lane = lanes[BrainNodeKind.HIDDEN]
        output_lane = lanes[BrainNodeKind.OUTPUT]
        self.assertAlmostEqual(input_lane.width, 210)
        self.assertAlmostEqual(hidden_lane.width, 250)
        self.assertAlmostEqual(output_lane.width, 180)
        self.assertAlmostEqual(hidden_lane.left - input_lane.right, 180)
        self.assertAlmostEqual(output_lane.left - hidden_lane.right, 180)

    def test_reference_positions_put_nodes_on_facing_card_edges(self) -> None:
        fixture = self.make_brain_world()
        bounds = arcade.LBWH(0, 0, 800, 500)
        layout = build_brain_graph_layout(
            fixture.brain.genome,
            [-1],
            [0],
            bounds,
            ["sensor"],
            ["accelerate"],
        )
        lanes = self.renderer._brain_graph_lane_bounds(bounds)

        positions = self.renderer._brain_graph_node_positions(layout, lanes)

        self.assertGreater(positions[-1][0], lanes[BrainNodeKind.INPUT].center_x)
        self.assertLess(positions[0][0], lanes[BrainNodeKind.OUTPUT].center_x)
        self.assertAlmostEqual(
            positions[1][0],
            lanes[BrainNodeKind.HIDDEN].center_x,
        )

    def test_graph_labels_face_away_from_routing_gaps(self) -> None:
        lane = arcade.LBWH(20, 20, 180, 300)

        input_bounds = self.renderer._draw_brain_graph_label(
            -1,
            "sensor",
            BrainNodeKind.INPUT,
            (170, 160),
            lane,
            radius=8,
            font_size=9,
        )
        input_text = self.renderer._text_cache["brain_window_node_label_-1"]
        output_bounds = self.renderer._draw_brain_graph_label(
            0,
            "accelerate",
            BrainNodeKind.OUTPUT,
            (50, 160),
            lane,
            radius=8,
            font_size=9,
        )
        output_text = self.renderer._text_cache["brain_window_node_label_0"]

        self.assertEqual(input_text.anchor_x, "right")
        self.assertLess(input_text.x, 170)
        self.assertEqual(output_text.anchor_x, "left")
        self.assertGreater(output_text.x, 50)
        self.assertIsNotNone(input_bounds)
        self.assertIsNotNone(output_bounds)
        assert input_bounds is not None and output_bounds is not None
        self.assertTrue(self.renderer._contains_bounds(input_bounds, input_text.x, 160))
        self.assertTrue(
            self.renderer._contains_bounds(output_bounds, output_text.x, 160)
        )

    def test_dense_input_layout_reduces_node_and_label_size_without_hiding(self) -> None:
        dense_input_keys = list(range(-1, -39, -1))
        genome = SimpleNamespace(
            nodes={0: SimpleNamespace()},
            connections={},
        )
        bounds = arcade.LBWH(0, 0, 800, 500)
        sparse_layout = build_brain_graph_layout(
            genome,
            [-1],
            [0],
            bounds,
            ["sensor"],
            ["action"],
        )
        dense_layout = build_brain_graph_layout(
            genome,
            dense_input_keys,
            [0],
            bounds,
            [str(key) for key in dense_input_keys],
            ["action"],
        )

        sparse_radius, sparse_font = self.renderer._brain_graph_node_metrics(
            sparse_layout,
            bounds,
        )
        dense_radius, dense_font = self.renderer._brain_graph_node_metrics(
            dense_layout,
            bounds,
        )

        self.assertEqual(len(dense_layout.nodes), 39)
        self.assertLess(dense_radius, sparse_radius)
        self.assertLess(dense_font, sparse_font)
        self.assertEqual(sparse_radius, 13.0)
        self.assertGreaterEqual(dense_radius, 6.0)
        self.assertGreaterEqual(dense_font, 9.5)

    def test_graph_nodes_use_static_white_fill_and_hide_hidden_canvas_labels(
        self,
    ) -> None:
        fixture = self.make_brain_world()
        fills: list[object] = []
        radii: list[float] = []
        original = self.renderer._draw_brain_node
        self.renderer._draw_brain_node = (
            lambda position, fill, outline, **kwargs: (
                fills.append(fill),
                radii.append(kwargs["radius"]),
            )
        )
        try:
            self.renderer._draw_brain_graph(
                fixture.world,
                arcade.LBWH(0, 0, 800, 500),
            )
        finally:
            self.renderer._draw_brain_node = original

        self.assertEqual(fills, [self.renderer.theme.panel_background] * 3)
        self.assertEqual(min(radii), 13.0)
        self.assertEqual(max(radii), 15.0)
        self.assertNotIn("brain_window_node_label_1", self.renderer._text_cache)

    def test_forward_edges_use_curves_and_enabled_arrowheads(self) -> None:
        edge = BrainGraphEdge(
            source=-1,
            target=0,
            weight=0.8,
            enabled=True,
            kind=BrainEdgeKind.FORWARD,
        )
        curves: list[list[tuple[float, float]]] = []
        arrows: list[list[tuple[float, float]]] = []
        original_curve = self.renderer._draw_brain_solid_curve
        original_arrow = self.renderer._draw_brain_arrowhead
        self.renderer._draw_brain_solid_curve = (
            lambda points, color, width: curves.append(points)
        )
        self.renderer._draw_brain_arrowhead = (
            lambda points, color, width: arrows.append(points)
        )
        try:
            self.renderer._draw_brain_graph_edge(
                edge,
                {-1: (100, 100), 0: (500, 240)},
                arcade.LBWH(0, 0, 600, 300),
            )
        finally:
            self.renderer._draw_brain_solid_curve = original_curve
            self.renderer._draw_brain_arrowhead = original_arrow

        self.assertEqual(len(curves), 1)
        self.assertEqual(len(arrows), 1)
        self.assertGreater(len(curves[0]), 20)
        self.assertEqual(curves[0][0], (100, 100))
        self.assertEqual(curves[0][-1], (500, 240))

    def test_recurrent_and_self_loop_edges_render_without_dag_assumptions(self) -> None:
        recurrent = BrainGraphEdge(
            source=0,
            target=1,
            weight=0.8,
            enabled=True,
            kind=BrainEdgeKind.RECURRENT,
        )
        self_loop = BrainGraphEdge(
            source=0,
            target=0,
            weight=-0.4,
            enabled=True,
            kind=BrainEdgeKind.SELF_LOOP,
        )
        curves: list[object] = []
        loops: list[object] = []
        original_curve = self.renderer._draw_brain_solid_curve
        original_loop = self.renderer._draw_self_loop
        self.renderer._draw_brain_solid_curve = (
            lambda *args, **kwargs: curves.append(args)
        )
        self.renderer._draw_self_loop = (
            lambda *args, **kwargs: loops.append(args)
        )
        try:
            self.renderer._draw_brain_graph_edge(
                recurrent,
                {0: (500, 240), 1: (300, 100)},
                arcade.LBWH(0, 0, 600, 300),
            )
            self.renderer._draw_brain_graph_edge(
                self_loop,
                {0: (500, 240)},
                arcade.LBWH(0, 0, 600, 300),
            )
        finally:
            self.renderer._draw_brain_solid_curve = original_curve
            self.renderer._draw_self_loop = original_loop

        self.assertEqual(len(curves), 1)
        self.assertEqual(len(loops), 1)

    def test_brain_curve_geometry_is_cached_and_drawn_as_one_strip(
        self,
    ) -> None:
        start = (10.0, 20.0)
        first_control = (30.0, 20.0)
        second_control = (50.0, 60.0)
        end = (70.0, 60.0)

        first = self.renderer._cubic_bezier_points(
            start,
            first_control,
            second_control,
            end,
        )
        second = self.renderer._cubic_bezier_points(
            start,
            first_control,
            second_control,
            end,
        )
        with patch("src.ui.renderer.arcade.draw_line_strip") as strip:
            self.renderer._draw_brain_solid_curve(
                first,
                self.renderer.theme.accent,
                2.0,
            )

        self.assertIs(first, second)
        strip.assert_called_once_with(
            first,
            self.renderer.theme.accent,
            2.0,
        )

    def test_disabled_edges_are_curved_without_arrowheads(self) -> None:
        edge = BrainGraphEdge(
            source=-1,
            target=0,
            weight=0.8,
            enabled=False,
            kind=BrainEdgeKind.FORWARD,
        )
        curves: list[object] = []
        arrows: list[object] = []
        original_curve = self.renderer._draw_brain_solid_curve
        original_arrow = self.renderer._draw_brain_arrowhead
        self.renderer._draw_brain_solid_curve = lambda *args: curves.append(args)
        self.renderer._draw_brain_arrowhead = lambda *args: arrows.append(args)
        try:
            self.renderer._draw_brain_graph_edge(
                edge,
                {-1: (100, 100), 0: (500, 240)},
                arcade.LBWH(0, 0, 600, 300),
                disabled=True,
            )
        finally:
            self.renderer._draw_brain_solid_curve = original_curve
            self.renderer._draw_brain_arrowhead = original_arrow

        self.assertEqual(len(curves), 1)
        self.assertEqual(arrows, [])

    def test_selected_disabled_direct_edge_is_dashed_and_directional(self) -> None:
        edge = BrainGraphEdge(
            source=-1,
            target=0,
            weight=-0.8,
            enabled=False,
            kind=BrainEdgeKind.FORWARD,
        )
        dashed_curves: list[object] = []
        arrows: list[object] = []
        original_dashed = self.renderer._draw_dashed_curve
        original_arrow = self.renderer._draw_brain_arrowhead
        self.renderer._draw_dashed_curve = (
            lambda *args, **kwargs: dashed_curves.append(args)
        )
        self.renderer._draw_brain_arrowhead = (
            lambda *args, **kwargs: arrows.append(args)
        )
        try:
            self.renderer._draw_brain_graph_edge(
                edge,
                {-1: (100, 100), 0: (500, 240)},
                arcade.LBWH(0, 0, 600, 300),
                disabled=True,
                direct=True,
            )
        finally:
            self.renderer._draw_dashed_curve = original_dashed
            self.renderer._draw_brain_arrowhead = original_arrow

        self.assertEqual(len(dashed_curves), 1)
        self.assertEqual(len(arrows), 1)

    def test_graph_sends_disabled_connections_to_background_group(self) -> None:
        fixture = self.make_brain_world()
        fixture.brain.genome.connections[(-1, 1)].enabled = False
        calls: list[dict[str, object]] = []
        original = self.renderer._draw_brain_graph_edge
        self.renderer._draw_brain_graph_edge = (
            lambda *args, **kwargs: calls.append(kwargs)
        )
        try:
            self.renderer._draw_brain_graph(
                fixture.world,
                arcade.LBWH(0, 0, 800, 500),
            )
        finally:
            self.renderer._draw_brain_graph_edge = original

        self.assertTrue(any(call.get("disabled") for call in calls))
        self.assertTrue(any(not call.get("disabled") for call in calls))

    def test_brain_legend_explains_layered_selection_detail(self) -> None:
        self.renderer._draw_brain_legend(arcade.LBWH(0, 0, 220, 620))

        labels = [
            self.renderer._text_cache[f"brain_legend_strength_{index}"].text
            for index in range(4)
        ]

        self.assertEqual(
            labels,
            [
                "Direct gene",
                "Enabled signal route",
                "Unrelated while selected",
                "Disabled direct gene",
            ],
        )

    def test_node_badge_stays_inside_summary_without_overlapping_name(self) -> None:
        labels = ("Input Node", "Hidden Node", "Output Node")
        for inspector_width in (220, 280, 360):
            summary = arcade.LBWH(14, 100, inspector_width - 28, 56)
            for label in labels:
                with self.subTest(width=inspector_width, label=label):
                    badge, name = self.renderer._brain_node_badge_layout(
                        summary,
                        label,
                    )
                    self.assertGreaterEqual(badge.left, summary.left)
                    self.assertLessEqual(badge.right, summary.right)
                    self.assertGreaterEqual(badge.width, len(label) * 5.0 + 16)
                    self.assertLessEqual(name.right + 10, badge.left)
                    self.assertGreater(name.width, 0)

    def test_short_node_name_preserves_compact_summary_height(self) -> None:
        header = arcade.LBWH(100.0, 400.0, 260.0, 44.0)

        summary, _, _, name_text = self.renderer._brain_node_summary_layout(
            header,
            "Accelerate",
            "Output Node",
        )

        self.assertEqual(
            summary.height,
            self.renderer.BRAIN_NODE_SUMMARY_MIN_HEIGHT,
        )
        self.assertEqual(name_text, "Accelerate")

    def test_long_node_name_expands_summary_without_overlapping_content(
        self,
    ) -> None:
        fixture = self.make_brain_world()
        long_name = "temperature_gradient_sensor_with_extended_range"
        layout = build_brain_graph_layout(
            fixture.brain.genome,
            [-1],
            [0],
            arcade.LBWH(0, 0, 600, 300),
            [long_name],
            ["accelerate"],
        )
        bounds = arcade.LBWH(100, 100, 260, 420)
        self.renderer._brain_selected_node_key = -1

        self.renderer._draw_brain_node_inspector(fixture.brain, layout, bounds)

        rendered = self.renderer._text_cache["brain_node_inspector_name"]
        header = arcade.LBWH(
            bounds.left,
            bounds.top - 44,
            bounds.width,
            44,
        )
        summary, badge, name, expected_text = (
            self.renderer._brain_node_summary_layout(
                header,
                long_name,
                "Input Node",
            )
        )
        self.assertTrue(rendered.multiline)
        self.assertEqual(rendered.width, name.width)
        self.assertEqual(rendered.text, expected_text)
        self.assertEqual(rendered.text.replace("\n", ""), long_name)
        self.assertNotIn("...", rendered.text)
        self.assertGreater(len(rendered.text.splitlines()), 2)
        self.assertGreater(
            summary.height,
            self.renderer.BRAIN_NODE_SUMMARY_MIN_HEIGHT,
        )
        self.assertLessEqual(rendered.x + rendered.width + 10, badge.left)
        self.assertLessEqual(
            self.renderer._scroll_regions["brain_node_inspector"].top,
            summary.bottom,
        )

    def test_node_summary_uses_measured_wide_glyph_layout(self) -> None:
        fixture = self.make_brain_world()
        long_name = "WWWWWWWWWWWWWWWWWWWWWWWW"
        layout = build_brain_graph_layout(
            fixture.brain.genome,
            [-1],
            [0],
            arcade.LBWH(0, 0, 600, 300),
            [long_name],
            ["accelerate"],
        )
        bounds = arcade.LBWH(100, 100, 260, 420)
        self.renderer._brain_selected_node_key = -1

        def measured_width(
            text: str,
            _font_size: float,
            *,
            bold: bool = False,
        ) -> float:
            """Return deliberately wide metrics for the regression case."""
            weight = 1.05 if bold else 1.0
            return sum(
                12.0 if character == "W" else 4.0
                for character in text
            ) * weight

        with patch.object(
            self.renderer._painter,
            "measure_text_width",
            side_effect=measured_width,
        ):
            self.renderer._draw_brain_node_inspector(
                fixture.brain,
                layout,
                bounds,
            )

        rendered = self.renderer._text_cache["brain_node_inspector_name"]
        header = arcade.LBWH(
            bounds.left,
            bounds.top - 44,
            bounds.width,
            44,
        )
        summary, badge, name, _ = self.renderer._brain_node_summary_layout(
            header,
            long_name,
            "Input Node",
        )
        self.assertTrue(all(
            measured_width(line, 14.0, bold=True) <= name.width
            for line in rendered.text.splitlines()
        ))
        self.assertLessEqual(name.right + 10.0, badge.left)
        self.assertLessEqual(
            self.renderer._scroll_regions["brain_node_inspector"].top,
            summary.bottom,
        )

    def test_node_inspector_reuses_wrapped_layout_between_frames(self) -> None:
        fixture = self.make_brain_world()
        layout = build_brain_graph_layout(
            fixture.brain.genome,
            [-1],
            [0],
            arcade.LBWH(0, 0, 600, 300),
            ["temperature_gradient_sensor_with_extended_range"],
            ["accelerate"],
        )
        bounds = arcade.LBWH(100, 100, 260, 420)
        self.renderer._brain_selected_node_key = -1
        painter = self.renderer._painter

        with patch.object(
            painter,
            "measure_text_width",
            wraps=painter.measure_text_width,
        ) as measure:
            self.renderer._draw_brain_node_inspector(
                fixture.brain,
                layout,
                bounds,
            )
            first_frame_measurements = measure.call_count
            self.renderer._draw_brain_node_inspector(
                fixture.brain,
                layout,
                bounds,
            )

        self.assertGreater(first_frame_measurements, 0)
        self.assertEqual(measure.call_count, first_frame_measurements)

    def test_node_inspector_lines_are_built_once_per_stable_selection(
        self,
    ) -> None:
        fixture = self.make_brain_world()
        layout = build_brain_graph_layout(
            fixture.brain.genome,
            [-1],
            [0],
            arcade.LBWH(0, 0, 600, 300),
            ["sensor"],
            ["accelerate"],
        )
        node = layout.nodes[1]

        with patch.object(
            self.renderer,
            "_brain_node_inspector_lines",
            wraps=self.renderer._brain_node_inspector_lines,
        ) as build:
            first = self.renderer._cached_brain_node_inspector_lines(
                fixture.brain,
                layout,
                node,
            )
            second = self.renderer._cached_brain_node_inspector_lines(
                fixture.brain,
                layout,
                node,
            )

        self.assertIs(first, second)
        build.assert_called_once_with(fixture.brain, layout, node)

    def test_scrollable_blocks_wrap_when_requested_and_count_visual_lines(
        self,
    ) -> None:
        viewport = arcade.LBWH(100.0, 100.0, 100.0, 50.0)
        value = "long_connection_identifier_that_must_wrap"

        self.renderer._draw_scrollable_lines_in_bounds(
            "responsive_block",
            viewport,
            [value],
            line_spacing=20.0,
            first_line_color=self.renderer.theme.text_primary,
            body_color=self.renderer.theme.text_muted,
            wrap_lines=True,
        )

        rendered = [
            text.text
            for key, text in self.renderer._text_cache.items()
            if key.startswith("responsive_block_line_")
        ]
        visual_lines = self.renderer._wrapped_scrollable_lines(
            [value],
            viewport.width - 12.0,
        )
        self.assertGreater(len(rendered), 1)
        self.assertEqual("".join(line for line, *_ in visual_lines), value)
        self.assertFalse(any("..." in line for line, *_ in visual_lines))
        self.assertTrue(
            all(is_first_line for _, is_first_line, *_ in visual_lines)
        )
        self.assertEqual(
            self.renderer._scroll_limits["responsive_block"],
            len(visual_lines) * 20.0 - viewport.height,
        )

    def test_scrollable_block_draws_only_visible_wrapped_rows(self) -> None:
        viewport = arcade.LBWH(100.0, 100.0, 180.0, 100.0)
        lines = tuple(
            f"Connection {index}: long_endpoint_identifier_{index}"
            for index in range(5000)
        )

        with patch.object(self.renderer, "_draw_text") as draw_text:
            self.renderer._draw_scrollable_lines_in_bounds(
                "large_responsive_block",
                viewport,
                lines,
                line_spacing=20.0,
                first_line_color=self.renderer.theme.text_primary,
                body_color=self.renderer.theme.text_muted,
                wrap_lines=True,
            )

        self.assertLessEqual(draw_text.call_count, 5)

    def test_brain_footer_cells_and_text_are_centered_with_colored_titles(
        self,
    ) -> None:
        fixture = self.make_brain_world()
        bounds = arcade.LBWH(40, 30, 1000, 72)

        self.renderer._draw_brain_footer(
            fixture.world,
            fixture.selected,
            fixture.brain,
            bounds,
        )

        cell_width = bounds.width / 5
        title_colors: list[object] = []
        for index in range(5):
            expected_x = bounds.left + (index + 0.5) * cell_width
            label = self.renderer._text_cache[f"brain_footer_label_{index}"]
            value = self.renderer._text_cache[f"brain_footer_value_{index}"]
            with self.subTest(index=index):
                self.assertEqual(label.x, round(expected_x))
                self.assertEqual(value.x, round(expected_x))
                self.assertEqual(label.anchor_x, "center")
                self.assertEqual(value.anchor_x, "center")
                self.assertEqual(label.anchor_y, "center")
                self.assertEqual(value.anchor_y, "center")
                self.assertEqual(label.font_size, 11)
                self.assertTrue(label.bold)
            title_colors.append(label.color)

        self.assertEqual(len(set(title_colors)), 5)

    def test_node_badge_uses_opaque_soft_color(self) -> None:
        color = self.renderer._brain_blend_color(
            self.renderer.theme.panel_background,
            self.renderer._brain_node_kind_color(BrainNodeKind.HIDDEN),
            0.12,
        )

        self.assertEqual(len(color), 3)


class FloatingSimulationUiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.renderer = UiRenderer(build_sim_config())

    def make_world_shell(self) -> SimpleNamespace:
        return SimpleNamespace(
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, 1440, 900),
                environment=arcade.LBWH(0, 0, 1440, 900),
            )
        )

    def make_inspector_world(
        self,
        *,
        energy: float = 0.4,
        max_energy: float = 2.0,
        vision_range: float = 160.0,
        panel_height: float = 414.0,
        selected: object | None = None,
    ) -> SimpleNamespace:
        creature = selected or SimpleNamespace(
            creature_id=938,
            name="Herbivore 938",
            energy=energy,
            speed=170.0,
            heading=1.25,
            vision=SimpleNamespace(range=vision_range, angle=2.17),
        )
        return SimpleNamespace(
            selected_creature=creature,
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, 1440, 900),
                environment=arcade.LBWH(0, 0, 1440, 900),
            ),
            config=SimpleNamespace(
                metabolism=SimpleNamespace(max_energy=max_energy),
                fitness=SimpleNamespace(),
            ),
            sensor_snapshot_for=lambda selected_creature: SimpleNamespace(
                food=SimpleNamespace(visible=2.0, density=0.25),
                creatures=SimpleNamespace(visible=1.0, density=0.1),
                stomach_fullness=0.6,
                flock=SimpleNamespace(flockmate_count=1.5),
            ),
            fitness_for=lambda selected_creature: None,
            neat_controller=SimpleNamespace(
                genome_id_for=lambda creature_id: creature_id,
            ),
            vision=SimpleNamespace(
                energy_cost_per_second=lambda selected_creature: 0.01,
            ),
        )

    def test_layout_uses_full_window_environment_and_slim_rail(self) -> None:
        config = build_sim_config()

        layout = build_screen_layout(1440, 900, config.layout)
        expected_rail_height = (
            UiRenderer.ICON_BUTTON_SIZE * 4
            + UiRenderer.ICON_BUTTON_GAP * 3
            + UiRenderer.RAIL_VERTICAL_PADDING
        )

        self.assertEqual(layout.environment.left, 0)
        self.assertEqual(layout.environment.bottom, 0)
        self.assertEqual(layout.environment.width, 1440)
        self.assertEqual(layout.environment.height, 900)
        self.assertEqual(layout.left_sidebar.width, config.layout.left_panel_width)
        self.assertLess(layout.left_sidebar.width, 120)
        self.assertEqual(layout.left_sidebar.height, expected_rail_height)
        self.assertEqual(
            layout.left_sidebar.center_y,
            layout.window.center_y,
        )

    def test_left_rail_stays_inside_small_window_padding(self) -> None:
        config = build_sim_config()

        layout = build_screen_layout(360, 300, config.layout)

        self.assertGreaterEqual(layout.left_sidebar.left, config.layout.outer_padding)
        self.assertGreaterEqual(layout.left_sidebar.bottom, config.layout.outer_padding)
        self.assertLessEqual(
            layout.left_sidebar.top,
            layout.window.top - config.layout.outer_padding,
        )

    def test_icon_helper_uses_loaded_asset_texture_path(self) -> None:
        calls = []
        previous_load_texture = getattr(arcade, "load_texture", None)
        previous_draw_texture_rectangle = getattr(
            arcade,
            "draw_texture_rectangle",
            None,
        )

        def fake_load_texture(path: str) -> object:
            calls.append(("load", path))
            return "texture"

        def fake_draw_texture_rectangle(*args: object) -> None:
            calls.append(("draw", args))

        arcade.load_texture = fake_load_texture
        arcade.draw_texture_rectangle = fake_draw_texture_rectangle
        try:
            self.renderer._draw_icon(
                arcade.LBWH(10, 20, 30, 40),
                "search",
                "test_search",
            )
        finally:
            if previous_load_texture is None:
                delattr(arcade, "load_texture")
            else:
                arcade.load_texture = previous_load_texture
            if previous_draw_texture_rectangle is None:
                delattr(arcade, "draw_texture_rectangle")
            else:
                arcade.draw_texture_rectangle = previous_draw_texture_rectangle

        self.assertEqual(calls[0][0], "load")
        self.assertTrue(str(calls[0][1]).endswith("assets/search.png"))
        self.assertEqual(calls[1], ("draw", (25.0, 40.0, 30, 40, "texture")))

    def test_panel_default_bounds_are_created_once_and_preserved(self) -> None:
        world = self.make_world_shell()

        first = self.renderer._stats_panel_bounds(world)
        self.renderer._panel_bounds["stats"] = arcade.LBWH(
            first.left - 80,
            first.bottom - 40,
            first.width,
            first.height,
        )
        second = self.renderer._stats_panel_bounds(world)

        self.assertNotEqual(second.left, first.left)
        self.assertNotEqual(second.bottom, first.bottom)

    def test_panel_icon_buttons_toggle_independently(self) -> None:
        self.renderer._control_hitboxes["panel_toggle_inspector"] = arcade.LBWH(0, 0, 20, 20)
        self.renderer._control_hitboxes["panel_toggle_stats"] = arcade.LBWH(30, 0, 20, 20)
        world = SimpleNamespace()

        self.assertTrue(self.renderer.handle_mouse_press(world, 10, 10))
        self.assertTrue(self.renderer.handle_mouse_press(world, 40, 10))

        self.assertTrue(self.renderer._panel_open["inspector"])
        self.assertTrue(self.renderer._panel_open["stats"])
        self.assertFalse(self.renderer._panel_open["settings"])

    def test_left_rail_registers_map_submenu_button(self) -> None:
        world = SimpleNamespace(
            layout=build_screen_layout(1440, 900, build_sim_config().layout),
            environment_map_mode="none",
        )

        self.renderer._draw_icon_rail(world)

        self.assertIn("open_map_submenu", self.renderer._control_hitboxes)

    def test_left_rail_registers_save_button_with_save_sim_icon(self) -> None:
        world = SimpleNamespace(
            layout=build_screen_layout(1440, 900, build_sim_config().layout),
            show_biome_background=False,
            save_in_progress=False,
        )
        calls: list[tuple[str, str, bool]] = []
        original_draw_icon_button = self.renderer._draw_icon_button
        self.renderer._draw_icon_button = (
            lambda bounds, icon_name, key, active: calls.append(
                (key, icon_name, active)
            )
        )
        try:
            self.renderer._draw_icon_rail(world)
        finally:
            self.renderer._draw_icon_button = original_draw_icon_button

        self.assertIn("save_simulation", self.renderer._control_hitboxes)
        self.assertIn(("save_simulation", "save_sim", False), calls)
        self.assertTrue(self.renderer._icon_path("save_sim").is_file())

    def test_save_button_active_state_follows_world(self) -> None:
        world = SimpleNamespace(
            layout=build_screen_layout(1440, 900, build_sim_config().layout),
            show_biome_background=False,
            save_in_progress=True,
        )
        calls: list[tuple[str, str, bool]] = []
        original_draw_icon_button = self.renderer._draw_icon_button
        self.renderer._draw_icon_button = (
            lambda bounds, icon_name, key, active: calls.append(
                (key, icon_name, active)
            )
        )
        try:
            self.renderer._draw_icon_rail(world)
        finally:
            self.renderer._draw_icon_button = original_draw_icon_button

        self.assertIn(("save_simulation", "save_sim", True), calls)

    def test_save_button_calls_world_save_now(self) -> None:
        calls: list[str] = []
        world = SimpleNamespace(save_now=lambda: calls.append("save"))
        self.renderer._control_hitboxes["save_simulation"] = arcade.LBWH(
            0,
            0,
            20,
            20,
        )

        handled = self.renderer.handle_mouse_press(world, 10, 10)

        self.assertTrue(handled)
        self.assertEqual(calls, ["save"])

    def test_six_left_rail_buttons_fit_without_overlap(self) -> None:
        world = SimpleNamespace(
            layout=build_screen_layout(800, 600, build_sim_config().layout),
            show_biome_background=False,
            save_in_progress=False,
        )

        self.renderer._draw_icon_rail(world)

        keys = (
            "panel_toggle_inspector",
            "panel_toggle_stats",
            "panel_toggle_settings",
            "open_map_submenu",
            "save_simulation",
            "open_species_tree",
        )
        buttons = [self.renderer._control_hitboxes[key] for key in keys]
        rail = world.layout.left_sidebar
        for button in buttons:
            self.assertGreaterEqual(button.bottom, rail.bottom)
            self.assertLessEqual(button.top, rail.top)
        for upper, lower in zip(buttons, buttons[1:]):
            self.assertGreaterEqual(upper.bottom, lower.top)

    def test_left_rail_uses_supplied_speciation_icon(self) -> None:
        world = SimpleNamespace(
            layout=build_screen_layout(1440, 900, build_sim_config().layout),
            show_biome_background=False,
            save_in_progress=False,
        )
        calls: list[tuple[str, str, bool]] = []
        original_draw_icon_button = self.renderer._draw_icon_button
        self.renderer._draw_icon_button = (
            lambda bounds, icon_name, key, active: calls.append(
                (key, icon_name, active)
            )
        )
        try:
            self.renderer._draw_icon_rail(world)
        finally:
            self.renderer._draw_icon_button = original_draw_icon_button

        self.assertIn(("open_species_tree", "species", False), calls)
        self.assertTrue(self.renderer._icon_path("species").is_file())

    def test_globe_button_opens_and_closes_map_submenu(self) -> None:
        world = SimpleNamespace()
        self.renderer._control_hitboxes["open_map_submenu"] = arcade.LBWH(
            0,
            0,
            20,
            20,
        )

        self.assertTrue(self.renderer.handle_mouse_press(world, 10, 10))
        self.assertTrue(self.renderer._map_submenu_open)
        self.assertTrue(self.renderer.handle_mouse_press(world, 10, 10))
        self.assertFalse(self.renderer._map_submenu_open)

    def test_globe_button_active_state_follows_selected_map(self) -> None:
        world = SimpleNamespace(
            layout=build_screen_layout(1440, 900, build_sim_config().layout),
            environment_map_mode="pheromones",
        )
        calls: list[tuple[str, str, bool]] = []
        original_draw_icon_button = self.renderer._draw_icon_button
        self.renderer._draw_icon_button = (
            lambda bounds, icon_name, key, active: calls.append(
                (key, icon_name, active)
            )
        )

        try:
            self.renderer._draw_icon_rail(world)
        finally:
            self.renderer._draw_icon_button = original_draw_icon_button

        self.assertIn(("open_map_submenu", "globe", True), calls)

    def test_globe_button_is_active_while_map_submenu_is_open(self) -> None:
        world = SimpleNamespace(
            layout=build_screen_layout(1440, 900, build_sim_config().layout),
            environment_map_mode="none",
        )
        calls: list[tuple[str, str, bool]] = []
        original_draw_icon_button = self.renderer._draw_icon_button
        original_draw_map_submenu = self.renderer._draw_map_submenu
        self.renderer._map_submenu_open = True
        self.renderer._draw_icon_button = (
            lambda bounds, icon_name, key, active: calls.append(
                (key, icon_name, active)
            )
        )
        self.renderer._draw_map_submenu = lambda active_world, anchor: None
        try:
            self.renderer._draw_icon_rail(world)
        finally:
            self.renderer._draw_icon_button = original_draw_icon_button
            self.renderer._draw_map_submenu = original_draw_map_submenu

        self.assertIn(("open_map_submenu", "globe", True), calls)

    def test_map_submenu_uses_supplied_icons_and_stays_inside_window(self) -> None:
        world = SimpleNamespace(
            layout=SimpleNamespace(window=arcade.LBWH(0, 0, 240, 180)),
            environment_map_mode="none",
        )
        icons: list[tuple[str, str]] = []
        original_draw_icon = self.renderer._draw_icon
        self.renderer._draw_icon = (
            lambda bounds, icon_name, key: icons.append((key, icon_name))
        )
        try:
            self.renderer._draw_map_submenu(
                world,
                arcade.LBWH(8, 8, 58, 58),
            )
        finally:
            self.renderer._draw_icon = original_draw_icon

        card = self.renderer._control_hitboxes["map_submenu"]
        self.assertGreaterEqual(card.left, world.layout.window.left)
        self.assertLessEqual(card.right, world.layout.window.right)
        self.assertGreaterEqual(card.bottom, world.layout.window.bottom)
        self.assertLessEqual(card.top, world.layout.window.top)
        self.assertIn(("map_layer_biome", "biome_map"), icons)
        self.assertIn(("map_layer_pheromones", "pheromone_map"), icons)
        self.assertTrue(self.renderer._icon_path("biome_map").is_file())
        self.assertTrue(self.renderer._icon_path("pheromone_map").is_file())

    def test_map_submenu_highlights_only_active_layer(self) -> None:
        world = SimpleNamespace(
            layout=SimpleNamespace(window=arcade.LBWH(0, 0, 800, 600)),
            environment_map_mode="biome",
        )
        fills: list[tuple[object, object]] = []
        original_draw_rounded_rect = self.renderer._draw_rounded_rect
        original_draw_icon = self.renderer._draw_icon
        original_draw_text = self.renderer._draw_text
        self.renderer._draw_rounded_rect = (
            lambda bounds, fill, border, radius, width: fills.append(
                (bounds, fill)
            )
        )
        self.renderer._draw_icon = lambda *args, **kwargs: None
        self.renderer._draw_text = lambda *args, **kwargs: None
        try:
            self.renderer._draw_map_submenu(
                world,
                arcade.LBWH(20, 250, 58, 58),
            )
        finally:
            self.renderer._draw_rounded_rect = original_draw_rounded_rect
            self.renderer._draw_icon = original_draw_icon
            self.renderer._draw_text = original_draw_text

        biome_row = self.renderer._control_hitboxes["map_layer_biome"]
        pheromone_row = self.renderer._control_hitboxes["map_layer_pheromones"]
        fill_by_id = {id(bounds): fill for bounds, fill in fills}
        self.assertEqual(
            fill_by_id[id(biome_row)],
            self.renderer.theme.accent_soft,
        )
        self.assertEqual(
            fill_by_id[id(pheromone_row)],
            self.renderer.theme.panel_background_alt,
        )

    def test_map_selection_is_exclusive_toggles_off_and_closes_menu(self) -> None:
        world = SimpleNamespace(environment_map_mode="none")

        def select_environment_map(mode: str) -> None:
            world.environment_map_mode = (
                "none" if world.environment_map_mode == mode else mode
            )

        world.select_environment_map = select_environment_map
        self.renderer._map_submenu_open = True
        self.renderer._control_hitboxes["map_submenu"] = arcade.LBWH(
            0, 0, 80, 80
        )
        self.renderer._control_hitboxes["map_layer_biome"] = arcade.LBWH(
            0, 40, 80, 40
        )
        self.renderer._control_hitboxes["map_layer_pheromones"] = arcade.LBWH(
            0, 0, 80, 40
        )

        self.assertTrue(self.renderer.handle_mouse_press(world, 20, 60))
        self.assertEqual(world.environment_map_mode, "biome")
        self.assertFalse(self.renderer._map_submenu_open)

        self.renderer._map_submenu_open = True
        self.assertTrue(self.renderer.handle_mouse_press(world, 20, 20))
        self.assertEqual(world.environment_map_mode, "pheromones")

        self.renderer._map_submenu_open = True
        self.assertTrue(self.renderer.handle_mouse_press(world, 20, 20))
        self.assertEqual(world.environment_map_mode, "none")

    def test_outside_submenu_click_closes_it_and_reaches_control(self) -> None:
        self.renderer._map_submenu_open = True
        self.renderer._control_hitboxes["map_submenu"] = arcade.LBWH(
            100, 100, 100, 100
        )
        self.renderer._control_hitboxes["panel_toggle_stats"] = arcade.LBWH(
            0, 0, 20, 20
        )

        handled = self.renderer.handle_mouse_press(SimpleNamespace(), 10, 10)

        self.assertTrue(handled)
        self.assertFalse(self.renderer._map_submenu_open)
        self.assertTrue(self.renderer._panel_open["stats"])

    def test_floating_panel_hitbox_consumes_click(self) -> None:
        self.renderer._control_hitboxes["stats_panel"] = arcade.LBWH(100, 100, 200, 120)

        handled = self.renderer.handle_mouse_press(SimpleNamespace(), 150, 150)

        self.assertTrue(handled)

    def test_dragging_inspector_updates_panel_bounds(self) -> None:
        world = self.make_world_shell()
        self.renderer._panel_bounds["inspector"] = arcade.LBWH(100, 100, 300, 240)
        self.renderer._control_hitboxes["inspector_drag"] = arcade.LBWH(100, 300, 300, 40)

        pressed = self.renderer.handle_mouse_press(world, 130, 320)
        dragged = self.renderer.handle_mouse_drag(world, 190, 360)

        self.assertTrue(pressed)
        self.assertTrue(dragged)
        self.assertEqual(self.renderer._panel_bounds["inspector"].left, 160)
        self.assertEqual(self.renderer._panel_bounds["inspector"].bottom, 140)

    def test_dragging_stats_updates_panel_bounds(self) -> None:
        world = self.make_world_shell()
        self.renderer._panel_bounds["stats"] = arcade.LBWH(500, 500, 300, 200)
        self.renderer._control_hitboxes["stats_drag"] = arcade.LBWH(500, 660, 300, 40)

        self.assertTrue(self.renderer.handle_mouse_press(world, 520, 680))
        self.assertTrue(self.renderer.handle_mouse_drag(world, 470, 640))

        self.assertEqual(self.renderer._panel_bounds["stats"].left, 450)
        self.assertEqual(self.renderer._panel_bounds["stats"].bottom, 460)

    def test_dragging_settings_updates_panel_bounds(self) -> None:
        world = self.make_world_shell()
        self.renderer._panel_bounds["settings"] = arcade.LBWH(420, 40, 500, 148)
        self.renderer._control_hitboxes["settings_drag"] = arcade.LBWH(420, 144, 500, 44)

        self.assertTrue(self.renderer.handle_mouse_press(world, 500, 166))
        self.assertTrue(self.renderer.handle_mouse_drag(world, 540, 206))

        self.assertEqual(self.renderer._panel_bounds["settings"].left, 460)
        self.assertEqual(self.renderer._panel_bounds["settings"].bottom, 80)

    def test_panel_controls_do_not_start_dragging(self) -> None:
        world = SimpleNamespace(toggle_pause=lambda: None)
        self.renderer._panel_bounds["settings"] = arcade.LBWH(420, 40, 500, 148)
        self.renderer._control_hitboxes["settings_drag"] = arcade.LBWH(420, 144, 500, 44)
        self.renderer._control_hitboxes["pause"] = arcade.LBWH(480, 140, 50, 50)

        handled = self.renderer.handle_mouse_press(world, 500, 166)

        self.assertTrue(handled)
        self.assertIsNone(self.renderer._active_panel_drag)

    def test_panel_close_does_not_start_dragging(self) -> None:
        world = self.make_world_shell()
        self.renderer._panel_open["stats"] = True
        self.renderer._panel_bounds["stats"] = arcade.LBWH(500, 500, 300, 200)
        self.renderer._control_hitboxes["stats_drag"] = arcade.LBWH(500, 660, 300, 40)
        self.renderer._control_hitboxes["stats_close"] = arcade.LBWH(760, 660, 28, 28)

        handled = self.renderer.handle_mouse_press(world, 770, 670)

        self.assertTrue(handled)
        self.assertFalse(self.renderer._panel_open["stats"])
        self.assertIsNone(self.renderer._active_panel_drag)

    def test_dragged_panel_clamps_to_window(self) -> None:
        world = self.make_world_shell()
        self.renderer._panel_bounds["inspector"] = arcade.LBWH(100, 100, 300, 240)
        self.renderer._control_hitboxes["inspector_drag"] = arcade.LBWH(100, 300, 300, 40)

        self.assertTrue(self.renderer.handle_mouse_press(world, 130, 320))
        self.assertTrue(self.renderer.handle_mouse_drag(world, -500, 2000))

        bounds = self.renderer._panel_bounds["inspector"]
        self.assertEqual(bounds.left, self.renderer.config.layout.outer_padding)
        self.assertEqual(
            bounds.bottom,
            world.layout.window.height
            - self.renderer.config.layout.outer_padding
            - bounds.height,
        )

    def test_inspector_defaults_to_large_window_safe_bounds(self) -> None:
        world = self.make_world_shell()

        bounds = self.renderer._inspector_panel_bounds(world)

        self.assertEqual(bounds.width, 440.0)
        self.assertEqual(bounds.height, 600.0)

        small_world = self.make_world_shell()
        small_world.layout.window = arcade.LBWH(0, 0, 360, 300)
        self.renderer._panel_bounds.pop("inspector", None)

        small_bounds = self.renderer._inspector_panel_bounds(small_world)
        margin = self.renderer.config.layout.outer_padding
        self.assertGreaterEqual(small_bounds.left, margin)
        self.assertGreaterEqual(small_bounds.bottom, margin)
        self.assertLessEqual(small_bounds.right, 360 - margin)
        self.assertLessEqual(small_bounds.top, 300 - margin)

    def test_inspector_energy_ratio_uses_creature_energy_not_vision(self) -> None:
        world = self.make_inspector_world(energy=0.4, max_energy=2.0, vision_range=999.0)

        self.assertEqual(self.renderer._inspector_energy_ratio(world), 0.2)

    def test_inspector_energy_ratio_clamps(self) -> None:
        high = self.make_inspector_world(energy=3.0, max_energy=2.0)
        low = self.make_inspector_world(energy=-0.25, max_energy=2.0)

        self.assertEqual(self.renderer._inspector_energy_ratio(high), 1.0)
        self.assertEqual(self.renderer._inspector_energy_ratio(low), 0.0)

    def test_creature_radar_uses_individual_genome_and_traits_once(self) -> None:
        world = self.make_inspector_world()
        world.config = self.renderer.config
        selected = world.selected_creature
        selected.physical_traits = PhysicalTraits(
            radius=18.0,
            movement_cost_multiplier=1.1,
        )
        selected.flocking_traits = FlockingTraits(0.2, 0.7, 0.8)
        genome = SimpleNamespace(
            nodes={key: SimpleNamespace(bias=0.0) for key in range(12)},
            connections={},
        )
        brain = SimpleNamespace(genome_id=44, genome=genome)
        world.neat_controller = SimpleNamespace(
            brain_for=lambda creature_id: brain,
            genome_id_for=lambda creature_id: brain.genome_id,
            config=SimpleNamespace(
                genome_config=SimpleNamespace(output_keys=tuple(range(12)))
            ),
        )
        future: Future[object] = Future()
        executor = SimpleNamespace(submit=Mock(return_value=future))
        self.renderer._species_tree_radar_executor = executor

        with patch(
            "src.ui.components.panels.inspector.calculate_genotypic_behavior_scores",
            return_value=(0.5,) * 6,
        ) as calculate:
            self.renderer._sync_creature_radar(world, selected)
            self.renderer._sync_creature_radar(world, selected)

        calculate.assert_called_once_with(
            genome,
            tuple(range(12)),
            physical_traits=selected.physical_traits,
            vision_traits=selected.vision,
            flocking_traits=selected.flocking_traits,
            trait_config=world.config.trait,
            vision_config=world.config.vision,
        )
        executor.submit.assert_called_once_with(
            generate_radar_chart_image,
            (0.5,) * 6,
            None,
            BEHAVIOR_RADAR_LABELS,
            primary_label="Selected creature",
        )
        self.assertEqual(self.renderer._creature_radar_identity, (938, 44))

    def test_creature_radar_replaces_selection_and_ignores_stale_result(self) -> None:
        world = self.make_inspector_world()
        world.config = self.renderer.config
        first = world.selected_creature
        genome = SimpleNamespace(
            nodes={key: SimpleNamespace(bias=0.0) for key in range(12)},
            connections={},
        )
        brain = SimpleNamespace(genome_id=44, genome=genome)
        world.neat_controller = SimpleNamespace(
            brain_for=lambda creature_id: brain,
            config=SimpleNamespace(
                genome_config=SimpleNamespace(output_keys=tuple(range(12)))
            ),
        )
        first_future: Future[object] = Future()
        second_future: Future[object] = Future()
        executor = SimpleNamespace(
            submit=Mock(side_effect=(first_future, second_future))
        )
        self.renderer._species_tree_radar_executor = executor

        self.renderer._sync_creature_radar(world, first)
        world.selected_creature = SimpleNamespace(
            creature_id=939,
            name="Herbivore 939",
            vision=first.vision,
        )
        self.renderer._sync_creature_radar(world, world.selected_creature)

        self.assertTrue(first_future.cancelled())
        self.assertEqual(self.renderer._creature_radar_identity, (939, 44))
        self.assertEqual(executor.submit.call_count, 2)

        stale: Future[object] = Future()
        stale.set_result(object())
        self.renderer._creature_radar_future = stale
        self.renderer._creature_radar_identity = (938, 44)
        self.renderer._panel_open["inspector"] = True
        with patch("src.ui.renderer.arcade.Texture", create=True) as texture:
            self.renderer._consume_creature_radar_result(world)
        texture.assert_not_called()
        self.assertIsNone(self.renderer._creature_radar_texture)

    def test_creature_radar_is_above_all_inspector_cards_and_responsive(self) -> None:
        world = self.make_inspector_world()
        viewport = arcade.LBWH(100.0, 100.0, 360.0, 2400.0)
        radar_bounds: list[arcade.Rect] = []
        card_bounds: list[arcade.Rect] = []

        with (
            patch.object(
                self.renderer,
                "_draw_creature_radar_chart_in_bounds",
                side_effect=lambda bounds: radar_bounds.append(bounds),
            ),
            patch.object(
                self.renderer,
                "_draw_inspector_card_section",
                side_effect=lambda _viewport, _section, bounds: card_bounds.append(bounds),
            ),
        ):
            self.renderer._draw_inspector_content(world, viewport)

        self.assertEqual(len(radar_bounds), 1)
        self.assertTrue(card_bounds)
        self.assertGreater(radar_bounds[0].bottom, card_bounds[0].top)
        self.assertLessEqual(radar_bounds[0].width, viewport.width - 32.0)
        self.assertEqual(
            self.renderer._creature_radar_chart_size(180.0),
            180.0,
        )
        self.assertEqual(
            self.renderer._creature_radar_chart_size(1000.0),
            self.renderer.CREATURE_RADAR_MAX_SIZE,
        )

    def test_creature_radar_unavailable_placeholder_and_panel_close_cleanup(self) -> None:
        world = self.make_inspector_world()
        self.renderer._sync_creature_radar(world, world.selected_creature)
        self.renderer._draw_creature_radar_chart_in_bounds(
            arcade.LBWH(100.0, 100.0, 220.0, 220.0)
        )
        self.assertEqual(
            self.renderer._text_cache["creature_radar_status"].text,
            "Behavioral profile unavailable",
        )

        future: Future[object] = Future()
        self.renderer._creature_radar_future = future
        self.renderer._creature_radar_texture = object()
        self.renderer._creature_radar_identity = (938, 44)
        self.renderer._panel_open["inspector"] = True
        self.renderer._control_hitboxes["inspector_close"] = arcade.LBWH(
            10.0, 10.0, 20.0, 20.0
        )
        self.renderer.handle_mouse_press(world, 20.0, 20.0)

        self.assertTrue(future.cancelled())
        self.assertIsNone(self.renderer._creature_radar_identity)
        self.assertIsNone(self.renderer._creature_radar_texture)

    def test_creature_radar_loading_success_and_render_failure_states(self) -> None:
        world = self.make_inspector_world()
        self.renderer._panel_open["inspector"] = True
        self.renderer._creature_radar_identity = (938, 44)
        pending: Future[object] = Future()
        self.renderer._creature_radar_future = pending
        bounds = arcade.LBWH(100.0, 100.0, 220.0, 220.0)

        self.renderer._draw_creature_radar_chart_in_bounds(bounds)
        self.assertEqual(
            self.renderer._text_cache["creature_radar_status"].text,
            "Loading behavioral profile...",
        )

        pending.set_result(object())
        with patch(
            "src.ui.renderer.arcade.Texture",
            return_value="creature-texture",
            create=True,
        ):
            self.renderer._consume_creature_radar_result(world)
        self.assertEqual(
            self.renderer._creature_radar_texture,
            "creature-texture",
        )

        failed: Future[object] = Future()
        failed.set_exception(RuntimeError("render failed"))
        self.renderer._creature_radar_texture = None
        self.renderer._creature_radar_future = failed
        self.renderer._consume_creature_radar_result(world)
        self.renderer._draw_creature_radar_chart_in_bounds(bounds)
        self.assertEqual(self.renderer._creature_radar_error, "render_failed")
        self.assertEqual(
            self.renderer._text_cache["creature_radar_status"].text,
            "Behavioral profile unavailable",
        )

    def test_inspector_draw_registers_scroll_region_and_action_hitboxes(self) -> None:
        world = self.make_inspector_world()
        self.renderer._panel_bounds["inspector"] = arcade.LBWH(100, 100, 368, 330)

        self.renderer._draw_inspector_panel(world)

        self.assertIn("inspector", self.renderer._scroll_regions)
        self.assertGreater(self.renderer._scroll_limits["inspector"], 0)
        self.renderer._scroll_offsets["inspector"] = self.renderer._scroll_limits["inspector"]
        self.renderer._draw_inspector_panel(world)
        self.assertIn("open_brain_window", self.renderer._control_hitboxes)
        self.assertIn(
            "open_behavior_report_selected",
            self.renderer._control_hitboxes,
        )
        self.assertIn("kill_selected_creature", self.renderer._control_hitboxes)

    def test_inspector_action_buttons_are_full_width_and_stacked(self) -> None:
        world = self.make_inspector_world()
        self.renderer._panel_bounds["inspector"] = arcade.LBWH(
            100,
            100,
            440,
            600,
        )

        self.renderer._draw_inspector_panel(world)
        self.renderer._scroll_offsets["inspector"] = (
            self.renderer._scroll_limits["inspector"]
        )
        self.renderer._draw_inspector_panel(world)

        brain = self.renderer._control_hitboxes["open_brain_window"]
        report = self.renderer._control_hitboxes[
            "open_behavior_report_selected"
        ]
        kill = self.renderer._control_hitboxes["kill_selected_creature"]
        for button in (brain, report, kill):
            self.assertEqual(button.height, 40.0)
            self.assertEqual(button.left, brain.left)
            self.assertEqual(button.width, brain.width)
        self.assertEqual(brain.bottom - report.top, 10.0)
        self.assertEqual(report.bottom - kill.top, 10.0)

    def test_alternative_inspector_actions_use_the_same_stacked_layout(
        self,
    ) -> None:
        world = self.make_inspector_world()

        self.renderer._draw_selected_creature(
            world,
            arcade.LBWH(100, 100, 440, 600),
        )

        brain = self.renderer._control_hitboxes["open_brain_window"]
        report = self.renderer._control_hitboxes[
            "open_behavior_report_selected"
        ]
        kill = self.renderer._control_hitboxes["kill_selected_creature"]
        for button in (brain, report, kill):
            self.assertEqual(button.height, 40.0)
            self.assertEqual(button.left, brain.left)
            self.assertEqual(button.width, brain.width)
        self.assertEqual(brain.bottom - report.top, 10.0)
        self.assertEqual(report.bottom - kill.top, 10.0)

    def test_inspector_snapshot_fallback_does_not_run_production_sensing(
        self,
    ) -> None:
        selected = SimpleNamespace(creature_id=7)
        world = SimpleNamespace(
            _last_sensor_snapshots={},
            sensor_snapshot_for=lambda _selected: self.fail(
                "renderer triggered a sensing pass"
            ),
        )
        snapshot = self.renderer._cached_inspector_snapshot(world, selected)
        self.assertEqual(snapshot.food.visible, 0.0)
        self.assertEqual(snapshot.creatures.visible, 0.0)

    def test_inspector_shows_live_effective_and_normalized_flockmate_count(
        self,
    ) -> None:
        world = self.make_inspector_world()
        sections, _species_id, _species_color = (
            self.renderer._inspector_card_sections(
                world,
                world.selected_creature,
            )
        )
        fields = {
            field.key: field
            for section in sections
            for field in section.fields
        }

        self.assertEqual(
            fields["inspector_flockmate_count"].value,
            "1.50",
        )
        self.assertEqual(
            fields["inspector_normalized_flockmate_count"].value,
            "0.33",
        )

    def test_inspector_scroll_region_uses_inner_card_viewport(self) -> None:
        world = self.make_inspector_world()
        self.renderer._panel_bounds["inspector"] = arcade.LBWH(100, 100, 368, 330)

        self.renderer._draw_inspector_panel(world)

        body = self.renderer._control_hitboxes["inspector_body"]
        scroll_region = self.renderer._scroll_regions["inspector"]
        self.assertGreater(scroll_region.left, body.left)
        self.assertGreater(scroll_region.bottom, body.bottom)
        self.assertLess(scroll_region.width, body.width)
        self.assertLess(scroll_region.height, body.height)

    def test_inspector_page_marker_follows_inner_card_edge(self) -> None:
        world = self.make_inspector_world()
        self.renderer._panel_bounds["inspector"] = arcade.LBWH(100, 100, 368, 330)
        marker_rectangles = []
        marker_circles = []
        original_rectangle = arcade.draw_lrbt_rectangle_filled
        original_circle = arcade.draw_circle_filled

        def capture_rectangle(
            left: float,
            right: float,
            bottom: float,
            top: float,
            color: object,
        ) -> None:
            if color == self.renderer.theme.accent:
                marker_rectangles.append((left, right, bottom, top))

        def capture_circle(
            x: float,
            y: float,
            radius: float,
            color: object,
        ) -> None:
            if color == self.renderer.theme.accent:
                marker_circles.append((x, y, radius))

        arcade.draw_lrbt_rectangle_filled = capture_rectangle
        arcade.draw_circle_filled = capture_circle
        try:
            self.renderer._draw_inspector_panel(world)
        finally:
            arcade.draw_lrbt_rectangle_filled = original_rectangle
            arcade.draw_circle_filled = original_circle

        body = self.renderer._control_hitboxes["inspector_body"]
        marker_left = body.left + 8
        marker_bottom = body.bottom + 10
        marker_width = 7.0
        marker_height = body.height - 18
        marker_right = marker_left + marker_width
        marker_top = marker_bottom + marker_height
        radius = marker_width / 2.0

        self.assertIn(
            (marker_left + radius, marker_right, marker_bottom, marker_top),
            marker_rectangles,
        )
        self.assertIn(
            (marker_left, marker_right, marker_bottom + radius, marker_top - radius),
            marker_rectangles,
        )
        self.assertEqual(
            marker_circles,
            [
                (marker_left + radius, marker_bottom + radius, radius),
                (marker_left + radius, marker_top - radius, radius),
            ],
        )

    def test_inspector_content_uses_clip_for_inner_viewport(self) -> None:
        world = self.make_inspector_world()
        self.renderer._panel_bounds["inspector"] = arcade.LBWH(100, 100, 368, 330)
        clips = []
        original_ui_clip = self.renderer._ui_clip

        @contextmanager
        def capture_clip(bounds: object):
            clips.append(bounds)
            yield

        self.renderer._ui_clip = capture_clip
        try:
            self.renderer._draw_inspector_panel(world)
        finally:
            self.renderer._ui_clip = original_ui_clip

        self.assertEqual(clips, [self.renderer._scroll_regions["inspector"]])

    def test_inspector_top_scroll_does_not_leave_action_hitboxes(self) -> None:
        world = self.make_inspector_world()
        self.renderer._panel_bounds["inspector"] = arcade.LBWH(100, 100, 368, 330)

        self.renderer._draw_inspector_panel(world)

        self.assertNotIn("open_brain_window", self.renderer._control_hitboxes)
        self.assertNotIn(
            "open_behavior_report_selected",
            self.renderer._control_hitboxes,
        )
        self.assertNotIn("kill_selected_creature", self.renderer._control_hitboxes)

    def test_inspector_progress_bars_use_vital_status_ratios(self) -> None:
        world = self.make_inspector_world(energy=0.5, max_energy=2.0, vision_range=160.0)
        ratios = []
        original_draw_progress_bar = self.renderer._draw_progress_bar

        def capture_progress_bar(bounds: object, ratio: float, **kwargs: object) -> None:
            del bounds, kwargs
            ratios.append(ratio)

        self.renderer._draw_progress_bar = capture_progress_bar
        try:
            sections, _species_id, _species_color = (
                self.renderer._inspector_card_sections(
                    world,
                    world.selected_creature,
                )
            )
            vital = next(section for section in sections if section.key == "vital")
            bounds = arcade.LBWH(
                100.0,
                100.0,
                300.0,
                self.renderer._inspector_card_section_height(vital, 300.0),
            )
            self.renderer._draw_inspector_card_section(bounds, vital, bounds)
        finally:
            self.renderer._draw_progress_bar = original_draw_progress_bar

        self.assertEqual(ratios, [0.25, 0.6, 1.0])

    def test_creature_inspector_marker_matches_selected_creature(self) -> None:
        selected = SimpleNamespace(
            creature_id=938,
            name="Herbivore 938",
            energy=0.5,
            speed=10.0,
            heading=0.0,
            color=(9, 9, 9),
            vision=SimpleNamespace(range=120.0, angle=1.0),
            lineage=LineageInfo(species_id=7),
        )
        world = self.make_inspector_world(selected=selected)
        world.species_history = {
            7: SimpleNamespace(founder_color=(210, 40, 90))
        }

        with (
            patch("src.ui.renderer.arcade.draw_circle_filled") as filled,
            patch("src.ui.renderer.arcade.draw_circle_outline") as outlined,
        ):
            self.renderer._draw_inspector_panel(world)

        filled.assert_any_call(
            ANY,
            ANY,
            8.0,
            (9, 9, 9),
        )
        outlined.assert_any_call(
            ANY,
            ANY,
            8.0,
            self.renderer.theme.selected_outline,
            2.5,
        )
        self.assertEqual(
            self.renderer._text_cache["inspector_species"].text,
            "Species #7",
        )

    def test_creature_inspector_marker_uses_theme_fallback(self) -> None:
        selected = SimpleNamespace(
            creature_id=938,
            name="Herbivore 938",
            energy=0.5,
            speed=10.0,
            heading=0.0,
            vision=SimpleNamespace(range=120.0, angle=1.0),
            lineage=LineageInfo(species_id=7),
        )
        world = self.make_inspector_world(selected=selected)
        world.species_history = {
            7: SimpleNamespace(founder_color=None)
        }

        with patch("src.ui.renderer.arcade.draw_circle_filled") as filled:
            self.renderer._draw_inspector_panel(world)

        filled.assert_any_call(
            ANY,
            ANY,
            8.0,
            self.renderer.theme.herbivore_fill,
        )

    def test_creature_inspector_marker_falls_back_to_species_color(self) -> None:
        selected = SimpleNamespace(
            creature_id=938,
            name="Herbivore 938",
            energy=0.5,
            speed=10.0,
            heading=0.0,
            vision=SimpleNamespace(range=120.0, angle=1.0),
            lineage=LineageInfo(species_id=7),
        )
        world = self.make_inspector_world(selected=selected)
        world.species_history = {
            7: SimpleNamespace(founder_color=(210, 40, 90))
        }

        with patch("src.ui.renderer.arcade.draw_circle_filled") as filled:
            self.renderer._draw_inspector_panel(world)

        filled.assert_any_call(
            ANY,
            ANY,
            8.0,
            (210, 40, 90),
        )

    def test_inspector_builds_explicit_identity_and_non_neat_genome_fields(
        self,
    ) -> None:
        selected = SimpleNamespace(
            creature_id=938,
            name="Herbivore 938",
            energy=0.5,
            speed=170.0,
            heading=1.25,
            vision=SimpleNamespace(range=120.0, angle=1.7),
            physical_traits=PhysicalTraits(
                radius=18.0,
                movement_cost_multiplier=1.12,
            ),
            flocking_traits=FlockingTraits(0.8, 0.3, 0.6),
            last_action=SimpleNamespace(herding=0.7),
            ledger_diagnostics=LedgerDiagnostics(
                rest_energy_recovered=0.04,
                healing_energy_spent=0.01,
                life_healed=0.01,
            ),
            lineage=LineageInfo(
                parent_id=12,
                generation=3,
                mutation_delta=TraitMutationDelta(
                    vision_range=2.0,
                    vision_angle=-0.03,
                    radius=1.0,
                    movement_cost_multiplier=0.04,
                ),
            ),
        )
        world = self.make_inspector_world(selected=selected)
        world._last_flocking_runtime = {
            selected.creature_id: FlockingRuntimeSnapshot(
                raw_neural_herding=0.90,
                effective_herding=0.70,
            )
        }
        sections, species_id, _species_color = (
            self.renderer._inspector_card_sections(world, selected)
        )
        self.assertEqual(species_id, 1)
        fields = {
            field.key: field
            for section in sections
            for field in section.fields
        }

        self.assertEqual(fields["inspector_creature_id"].value, "#938")
        self.assertEqual(fields["inspector_parent"].value, "#12")
        self.assertEqual(fields["inspector_generation"].value, "3")
        self.assertEqual(fields["inspector_neat_genome"].value, "#938")
        self.assertEqual(fields["inspector_radius"].label, "Body radius")
        self.assertEqual(fields["inspector_radius"].value, "18.0 px")
        self.assertEqual(
            fields["inspector_radius"].detail,
            "Change from parent: +1.0 px",
        )
        self.assertEqual(
            fields["inspector_vision_angle"].detail,
            "Change from parent: -0.030 rad",
        )
        self.assertEqual(
            fields["inspector_digestion_efficiency"].detail,
            "Change from parent: +0.0 percentage points",
        )
        self.assertEqual(fields["inspector_separation_gene"].value, "0.800")
        self.assertEqual(fields["inspector_alignment_gene"].value, "0.300")
        self.assertEqual(fields["inspector_cohesion_gene"].value, "0.600")
        self.assertEqual(fields["inspector_social_tag_x"].value, "0.500")
        self.assertEqual(fields["inspector_social_tag_y"].value, "0.500")
        inherited_keys = {
            field.key
            for section in sections
            if section.key.startswith("inherited_")
            for field in section.fields
        }
        self.assertEqual(len(inherited_keys), 12)
        for key in inherited_keys:
            self.assertIsNotNone(fields[key].detail)
            self.assertIn("Change from parent:", fields[key].detail)
        self.assertEqual(
            fields["inspector_social_tag_x"].detail,
            "Change from parent: +0.000",
        )
        self.assertEqual(fields["inspector_raw_herding"].value, "0.90")
        self.assertEqual(fields["inspector_herding"].value, "0.70")
        self.assertEqual(
            fields["inspector_rest_recovery"].value,
            "0.0400",
        )
        self.assertEqual(
            fields["inspector_healing_spend"].value,
            "0.0100",
        )
        self.assertEqual(
            fields["inspector_life_damage"].value,
            "0.0100",
        )
        self.assertEqual(
            fields["inspector_collision_avoidance"].value,
            "Universal and automatic",
        )
        for field in fields.values():
            self.assertNotIn("R +", field.value)
            self.assertNotIn("S ", field.value)

    def test_shared_card_metric_row_wraps_within_available_width(self) -> None:
        width = 260.0
        consumed = self.renderer._draw_metric_row(
            "responsive_metric",
            "A deliberately long inherited morphology label",
            "R +1.00 / V +2.00 / D +0.10 / F +0.20 / C +0.30",
            20.0,
            240.0,
            width,
        )

        label = self.renderer._text_cache["responsive_metric_label"]
        value = self.renderer._text_cache["responsive_metric_value"]
        self.assertTrue(label.multiline)
        self.assertTrue(value.multiline)
        self.assertIn("\n", label.text)
        self.assertIn("\n", value.text)
        self.assertGreater(consumed, 25.0)
        self.assertGreaterEqual(label.x, 20.0)
        self.assertLessEqual(value.x + value.width, 20.0 + width)
        for rendered, font_size in ((label, 10.0), (value, 12.0)):
            for line in rendered.text.splitlines():
                self.assertLessEqual(
                    self.renderer._painter.measure_text_width(
                        line,
                        font_size,
                    ),
                    rendered.width,
                )

    def test_shared_metric_rows_align_labels_and_values_to_columns(
        self,
    ) -> None:
        x = 20.0
        width = 360.0

        self.renderer._draw_metric_row(
            "aligned_metric",
            "Persistent local group",
            "#12 / 8 members",
            x,
            240.0,
            width,
        )

        label = self.renderer._text_cache["aligned_metric_label"]
        value = self.renderer._text_cache["aligned_metric_value"]
        self.assertEqual(label.align, "left")
        self.assertEqual(value.align, "right")
        self.assertEqual(label.x, x)
        self.assertEqual(value.x + value.width, x + width)
        self.assertEqual(label.y, value.y)

    def test_responsive_card_rows_stack_without_overlap_at_all_widths(
        self,
    ) -> None:
        for width in (140.0, 260.0, 360.0):
            with self.subTest(width=width):
                first = self.renderer._metric_row_layout(
                    "Inherited digestive morphology",
                    "capacity 2.600 / rate 0.400 / efficiency 98.0%",
                    20.0,
                    240.0,
                    width,
                )
                first_label, first_value = first[2], first[3]
                second_y = 240.0 - first[4]
                second = self.renderer._metric_row_layout(
                    "Mutation delta",
                    "+0.120 / +0.020 / +0.015",
                    20.0,
                    second_y,
                    width,
                )
                second_label, second_value = second[2], second[3]

                self.assertLessEqual(
                    max(second_label.top, second_value.top),
                    min(first_label.bottom, first_value.bottom),
                )
                for bounds in (
                    first_label,
                    first_value,
                    second_label,
                    second_value,
                ):
                    self.assertGreaterEqual(bounds.left, 20.0)
                    self.assertLessEqual(bounds.right, 20.0 + width)

    def test_creature_inspector_stacked_fields_keep_text_inside_padding(
        self,
    ) -> None:
        world = self.make_inspector_world()
        world.selected_creature.lineage = LineageInfo(
            mutation_delta=TraitMutationDelta(
                radius=1.0,
                vision_range=2.0,
                vision_angle=-0.03,
                movement_cost_multiplier=0.04,
                stomach_capacity=0.12,
                digestion_rate=0.02,
                digestion_efficiency=0.01,
                separation_gene=0.1,
                alignment_gene=-0.1,
                cohesion_gene=0.2,
            )
        )
        sections, _species_id, _species_color = (
            self.renderer._inspector_card_sections(
                world,
                world.selected_creature,
            )
        )
        social = next(
            section for section in sections if section.key == "inherited_social"
        )
        bounds = arcade.LBWH(
            100.0,
            100.0,
            230.0,
            self.renderer._inspector_card_section_height(social, 230.0),
        )
        self.renderer._draw_inspector_card_section(bounds, social, bounds)

        rendered = [
            text
            for key, text in self.renderer._text_cache.items()
            if key.startswith("inspector_social_tag")
        ]
        self.assertTrue(rendered)
        for text in rendered:
            self.assertGreaterEqual(text.x, bounds.left + 25.0)
            self.assertLessEqual(text.x + text.width, bounds.right - 25.0)

    def test_progress_bar_zero_ratio_skips_fill(self) -> None:
        fills = []
        original_fill = self.renderer._draw_rounded_rect_fill

        def capture_fill(bounds: object, color: object, radius: float) -> None:
            del color, radius
            fills.append(bounds)

        self.renderer._draw_rounded_rect_fill = capture_fill
        try:
            self.renderer._draw_progress_bar(arcade.LBWH(10, 20, 100, 8), 0.0)
        finally:
            self.renderer._draw_rounded_rect_fill = original_fill

        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].width, 100)

    def test_progress_bar_tiny_ratio_does_not_raise(self) -> None:
        self.renderer._draw_progress_bar(arcade.LBWH(10, 20, 100, 8), 1e-9)
        self.renderer._draw_progress_bar(arcade.LBWH(10, 20, 100, 8), 1e-5)

    def test_rounded_rect_fill_tolerates_zero_and_tiny_widths(self) -> None:
        self.renderer._draw_rounded_rect_fill(arcade.LBWH(10, 20, 0, 8), (1, 2, 3), 4)
        self.renderer._draw_rounded_rect_fill(
            arcade.LBWH(10, 20, 0.0000000001, 8),
            (1, 2, 3),
            4,
        )

    def test_inspector_zero_energy_render_does_not_raise(self) -> None:
        world = self.make_inspector_world(energy=0.0, max_energy=1.0)

        self.renderer._draw_inspector_panel(world)

    def test_inspector_empty_state_draws_without_selection(self) -> None:
        labels = []
        original_draw_text = self.renderer._draw_text
        world = self.make_inspector_world(selected=None)
        world.selected_creature = None

        def capture_text(key: str, text: str, *args: object, **kwargs: object) -> None:
            del key, args, kwargs
            labels.append(text)

        self.renderer._draw_text = capture_text
        try:
            self.renderer._draw_inspector_panel(world)
        finally:
            self.renderer._draw_text = original_draw_text

        self.assertIn("No creature selected", labels)

    def test_action_button_icon_and_text_have_padding_and_centered_group(self) -> None:
        icon_bounds = []
        texts = []
        original_draw_icon = self.renderer._draw_icon
        original_draw_text = self.renderer._draw_text

        def capture_icon(bounds: object, icon_name: str, key: str) -> None:
            del icon_name, key
            icon_bounds.append(bounds)

        def capture_text(
            key: str,
            text: str,
            x: float,
            y: float,
            *args: object,
            **kwargs: object,
        ) -> None:
            del key, args, kwargs
            texts.append((text, x, y))

        self.renderer._draw_icon = capture_icon
        self.renderer._draw_text = capture_text
        try:
            button = arcade.LBWH(100, 200, 150, 36)
            self.renderer._draw_action_button(
                button,
                "Open Brain",
                "brain",
                "open_brain_window",
                fill_color=(1, 2, 3),
                text_color=(4, 5, 6),
            )
        finally:
            self.renderer._draw_icon = original_draw_icon
            self.renderer._draw_text = original_draw_text

        icon = icon_bounds[0]
        text, text_x, text_y = texts[-1]
        text_width = len(text) * 7.0
        group_left = icon.left
        group_right = text_x + text_width
        self.assertGreaterEqual(text_x - icon.right, 12.0)
        self.assertAlmostEqual((group_left + group_right) / 2.0, button.center_x)
        self.assertEqual(text_y, button.center_y)

    def test_open_brain_button_requires_selected_creature(self) -> None:
        self.renderer._control_hitboxes["open_brain_window"] = arcade.LBWH(0, 0, 40, 20)
        world = SimpleNamespace(selected_creature=None)

        handled = self.renderer.handle_mouse_press(world, 10, 10)

        self.assertTrue(handled)
        self.assertFalse(self.renderer._brain_window_open)

    def test_open_brain_button_opens_when_creature_selected(self) -> None:
        self.renderer._control_hitboxes["open_brain_window"] = arcade.LBWH(0, 0, 40, 20)
        world = SimpleNamespace(
            selected_creature=object(),
            layout=SimpleNamespace(
                environment=arcade.LBWH(0, 0, 1440, 900),
                window=arcade.LBWH(0, 0, 1440, 900),
            ),
        )

        handled = self.renderer.handle_mouse_press(world, 10, 10)

        self.assertTrue(handled)
        self.assertTrue(self.renderer._brain_window_open)

    def test_kill_button_calls_world_and_closes_brain_window(self) -> None:
        calls = []
        self.renderer._brain_window_open = True
        self.renderer._control_hitboxes["kill_selected_creature"] = arcade.LBWH(0, 0, 40, 20)
        world = SimpleNamespace(
            kill_selected_creature=lambda: calls.append("kill") or True
        )

        handled = self.renderer.handle_mouse_press(world, 10, 10)

        self.assertTrue(handled)
        self.assertEqual(calls, ["kill"])
        self.assertFalse(self.renderer._brain_window_open)

    def test_speed_buttons_preserve_world_speed_api(self) -> None:
        calls = []
        self.renderer._control_hitboxes["speed_down"] = arcade.LBWH(0, 0, 20, 20)
        self.renderer._control_hitboxes["speed_up"] = arcade.LBWH(30, 0, 20, 20)
        self.renderer._control_hitboxes["reset_speed"] = arcade.LBWH(60, 0, 20, 20)
        world = SimpleNamespace(
            decrease_simulation_speed=lambda: calls.append("down"),
            increase_simulation_speed=lambda: calls.append("up"),
            reset_simulation_speed=lambda: calls.append("reset"),
        )

        self.assertTrue(self.renderer.handle_mouse_press(world, 10, 10))
        self.assertTrue(self.renderer.handle_mouse_press(world, 40, 10))
        self.assertTrue(self.renderer.handle_mouse_press(world, 70, 10))

        self.assertEqual(calls, ["down", "up", "reset"])

    def test_speed_min_button_sets_minimum_speed(self) -> None:
        calls = []
        self.renderer._control_hitboxes["speed_min"] = arcade.LBWH(0, 0, 20, 20)
        world = SimpleNamespace(
            MIN_SIMULATION_SPEED=0.25,
            set_simulation_speed=lambda speed: calls.append(speed),
        )

        self.assertTrue(self.renderer.handle_mouse_press(world, 10, 10))

        self.assertEqual(calls, [0.25])

    def test_speed_max_button_sets_maximum_speed(self) -> None:
        calls = []
        self.renderer._control_hitboxes["speed_max"] = arcade.LBWH(0, 0, 20, 20)
        world = SimpleNamespace(
            MAX_SIMULATION_SPEED=5.0,
            set_simulation_speed=lambda speed: calls.append(speed),
        )

        self.assertTrue(self.renderer.handle_mouse_press(world, 10, 10))

        self.assertEqual(calls, [5.0])

    def test_settings_panel_draws_all_speed_controls_inside_panel(self) -> None:
        world = SimpleNamespace(
            is_paused=False,
            simulation_speed=1.0,
            MIN_SIMULATION_SPEED=0.25,
            MAX_SIMULATION_SPEED=5.0,
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, 1440, 900),
                environment=arcade.LBWH(0, 0, 1440, 900),
            ),
        )

        self.renderer._draw_settings_panel(world)

        panel = self.renderer._control_hitboxes["settings_panel"]
        for key in (
            "speed_slider",
            "speed_min",
            "speed_down",
            "pause",
            "speed_up",
            "speed_max",
        ):
            bounds = self.renderer._control_hitboxes[key]
            self.assertGreaterEqual(bounds.left, panel.left)
            self.assertLessEqual(bounds.right, panel.right)
        self.assertNotIn("reset_speed", self.renderer._control_hitboxes)

    def test_settings_speed_control_order_places_pause_between_arrows(self) -> None:
        world = SimpleNamespace(
            is_paused=False,
            simulation_speed=1.0,
            MIN_SIMULATION_SPEED=0.25,
            MAX_SIMULATION_SPEED=5.0,
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, 1440, 900),
                environment=arcade.LBWH(0, 0, 1440, 900),
            ),
        )

        self.renderer._draw_settings_panel(world)

        ordered_keys = (
            "speed_min",
            "speed_down",
            "pause",
            "speed_up",
            "speed_max",
        )
        for left_key, right_key in zip(ordered_keys, ordered_keys[1:]):
            self.assertLess(
                self.renderer._control_hitboxes[left_key].left,
                self.renderer._control_hitboxes[right_key].left,
            )

    def test_settings_speed_controls_are_compact_and_padded(self) -> None:
        world = SimpleNamespace(
            is_paused=False,
            simulation_speed=1.0,
            MIN_SIMULATION_SPEED=0.25,
            MAX_SIMULATION_SPEED=5.0,
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, 1440, 900),
                environment=arcade.LBWH(0, 0, 1440, 900),
            ),
        )

        self.renderer._draw_settings_panel(world)

        slider = self.renderer._control_hitboxes["speed_slider"]
        self.assertGreater(slider.width, 72.0)

        ordered_keys = (
            "speed_min",
            "speed_down",
            "pause",
            "speed_up",
            "speed_max",
        )
        previous = slider
        for key in ordered_keys:
            bounds = self.renderer._control_hitboxes[key]
            if key == "pause":
                self.assertGreater(bounds.width, 32.0)
                self.assertGreater(bounds.height, 32.0)
            else:
                self.assertLess(bounds.width, 32.0)
                self.assertLess(bounds.height, 32.0)
            self.assertGreaterEqual(bounds.left - previous.right, 8.0)
            previous = bounds

    def test_settings_speed_controls_do_not_overlap_at_minimum_panel_width(self) -> None:
        world = SimpleNamespace(
            is_paused=False,
            simulation_speed=1.0,
            MIN_SIMULATION_SPEED=0.25,
            MAX_SIMULATION_SPEED=5.0,
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, 520, 720),
                environment=arcade.LBWH(0, 0, 520, 720),
            ),
        )

        self.renderer._draw_settings_panel(world)

        ordered_keys = (
            "speed_slider",
            "speed_min",
            "speed_down",
            "pause",
            "speed_up",
            "speed_max",
        )
        for left_key, right_key in zip(ordered_keys, ordered_keys[1:]):
            left_bounds = self.renderer._control_hitboxes[left_key]
            right_bounds = self.renderer._control_hitboxes[right_key]
            self.assertGreaterEqual(right_bounds.left - left_bounds.right, 8.0)

    def test_settings_panel_removes_stale_reset_speed_hitbox(self) -> None:
        world = SimpleNamespace(
            is_paused=False,
            simulation_speed=1.0,
            MIN_SIMULATION_SPEED=0.25,
            MAX_SIMULATION_SPEED=5.0,
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, 1440, 900),
                environment=arcade.LBWH(0, 0, 1440, 900),
            ),
        )
        self.renderer._control_hitboxes["reset_speed"] = arcade.LBWH(0, 0, 20, 20)

        self.renderer._draw_settings_panel(world)

        self.assertNotIn("reset_speed", self.renderer._control_hitboxes)

    def test_settings_food_controls_expand_upward_and_scroll(self) -> None:
        config = build_sim_config()
        world = SimpleNamespace(
            is_paused=False,
            simulation_speed=1.0,
            MIN_SIMULATION_SPEED=0.25,
            MAX_SIMULATION_SPEED=5.0,
            live_food_config=LiveFoodConfig.from_configs(
                config.biome,
                config.food,
            ),
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, 1440, 900),
                environment=arcade.LBWH(0, 0, 1440, 900),
            ),
        )

        self.renderer._draw_settings_panel(world)
        collapsed = self.renderer._control_hitboxes["settings_panel"]
        toggle = self.renderer._control_hitboxes["settings_food_toggle"]
        self.assertTrue(
            self.renderer.handle_mouse_press(
                world,
                toggle.center_x,
                toggle.center_y,
            )
        )

        self.renderer._draw_settings_panel(world)
        expanded = self.renderer._control_hitboxes["settings_panel"]
        viewport = self.renderer._scroll_regions["settings_food"]
        self.assertEqual(expanded.bottom, collapsed.bottom)
        self.assertEqual(expanded.height, 680.0)
        self.assertGreater(
            self.renderer._scroll_limits["settings_food"],
            0.0,
        )
        for field_name in self.renderer.LIVE_FOOD_SLIDER_FIELDS:
            key = self.renderer._live_food_slider_key(field_name)
            slider = self.renderer._control_hitboxes.get(key)
            if slider is None:
                continue
            self.assertGreaterEqual(slider.bottom, viewport.bottom)
            self.assertLessEqual(slider.top, viewport.top)

        self.assertNotIn(
            "food_slider_low_food_burst_interval",
            self.renderer._control_hitboxes,
        )
        self.assertTrue(
            self.renderer.handle_mouse_scroll(
                viewport.center_x,
                viewport.center_y,
                -10.0,
            )
        )
        self.renderer._draw_settings_panel(world)
        self.assertIn(
            "food_slider_low_food_burst_interval",
            self.renderer._control_hitboxes,
        )

    def test_settings_food_toggle_has_its_own_padded_footer_row(self) -> None:
        world = SimpleNamespace(
            is_paused=False,
            simulation_speed=1.0,
            MIN_SIMULATION_SPEED=0.25,
            MAX_SIMULATION_SPEED=5.0,
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, 1440, 900),
                environment=arcade.LBWH(0, 0, 1440, 900),
            ),
        )

        self.renderer._draw_settings_panel(world)

        content = self.renderer._control_hitboxes["settings_body"]
        toggle = self.renderer._control_hitboxes["settings_food_toggle"]
        first_hint = self.renderer._text_cache["settings_hint_0_key"]
        divider_y = content.bottom + 100.0
        self.assertGreaterEqual(divider_y - toggle.top, 12.0)
        self.assertGreaterEqual(toggle.left - content.left, 16.0)
        self.assertGreaterEqual(content.right - toggle.right, 16.0)
        self.assertGreaterEqual(toggle.bottom - first_hint.y, 12.0)

    def test_expanded_settings_panel_clamps_to_small_window(self) -> None:
        config = build_sim_config()
        self.renderer._settings_expanded = True
        world = SimpleNamespace(
            is_paused=False,
            simulation_speed=1.0,
            MIN_SIMULATION_SPEED=0.25,
            MAX_SIMULATION_SPEED=5.0,
            live_food_config=LiveFoodConfig.from_configs(
                config.biome,
                config.food,
            ),
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, 520, 420),
                environment=arcade.LBWH(0, 0, 520, 420),
            ),
        )

        self.renderer._draw_settings_panel(world)

        panel = self.renderer._control_hitboxes["settings_panel"]
        self.assertEqual(panel.height, 380.0)
        self.assertGreaterEqual(panel.left, config.layout.outer_padding)
        self.assertLessEqual(
            panel.right,
            world.layout.window.right - config.layout.outer_padding,
        )

    def test_live_food_slider_click_and_drag_snap_integer_values(self) -> None:
        calls = []
        key = "food_slider_max_food_items"
        self.renderer._control_hitboxes[key] = arcade.LBWH(0, 0, 200, 18)
        world = SimpleNamespace(
            set_live_food_config_value=lambda name, value: calls.append(
                (name, value)
            )
        )

        self.assertTrue(self.renderer.handle_mouse_press(world, 100, 9))
        self.assertEqual(self.renderer._active_slider, key)
        self.assertEqual(calls[-1], ("max_food_items", 1000))

        self.assertTrue(self.renderer.handle_mouse_drag(world, 200, 9))
        self.assertEqual(calls[-1], ("max_food_items", 2000))
        self.renderer.handle_mouse_release()
        self.assertIsNone(self.renderer._active_slider)

    def test_pause_button_text_is_drawn_at_button_center(self) -> None:
        texts = []
        original_draw_text = self.renderer._draw_text

        def capture_text(
            key: str,
            text: str,
            x: float,
            y: float,
            *args: object,
            **kwargs: object,
        ) -> None:
            del text, args, kwargs
            texts.append((key, x, y))

        self.renderer._draw_text = capture_text
        try:
            pause_button = arcade.LBWH(100, 200, 50, 50)
            self.renderer._draw_icon_text_button(
                pause_button,
                "||",
                "pause",
                fill_color=(1, 2, 3),
                size=19,
            )
        finally:
            self.renderer._draw_text = original_draw_text

        _, x, y = texts[-1]
        self.assertEqual(x, pause_button.center_x)
        self.assertEqual(y, pause_button.center_y)

    def test_pause_state_bars_are_centered_in_button(self) -> None:
        fills = []
        original_fill = self.renderer._draw_rounded_rect_fill

        def capture_fill(bounds: object, color: object, radius: float) -> None:
            del color, radius
            fills.append(bounds)

        self.renderer._draw_rounded_rect_fill = capture_fill
        try:
            button = arcade.LBWH(100, 200, 50, 50)
            self.renderer._draw_play_pause_button(button, is_paused=False)
        finally:
            self.renderer._draw_rounded_rect_fill = original_fill

        bars = fills[-2:]
        center = (bars[0].center_x + bars[1].center_x) / 2.0
        self.assertEqual(center, button.center_x)
        self.assertEqual(bars[0].center_y, button.center_y)
        self.assertEqual(bars[1].center_y, button.center_y)
        self.assertLess(bars[0].height, 22)

    def test_play_state_icon_is_centered_in_button(self) -> None:
        polygons = []
        previous_draw_polygon = getattr(arcade, "draw_polygon_filled", None)

        def capture_polygon(points: object, color: object) -> None:
            del color
            polygons.append(points)

        arcade.draw_polygon_filled = capture_polygon
        try:
            button = arcade.LBWH(100, 200, 50, 50)
            self.renderer._draw_play_pause_button(button, is_paused=True)
        finally:
            if previous_draw_polygon is None:
                delattr(arcade, "draw_polygon_filled")
            else:
                arcade.draw_polygon_filled = previous_draw_polygon

        points = polygons[0]
        center_x = sum(point[0] for point in points) / len(points)
        center_y = sum(point[1] for point in points) / len(points)
        self.assertAlmostEqual(center_x, button.center_x - 2 / 3)
        self.assertEqual(center_y, button.center_y)

    def test_keycap_label_has_padding_after_keycap(self) -> None:
        texts = []
        original_draw_text = self.renderer._draw_text

        def capture_text(
            key: str,
            text: str,
            x: float,
            y: float,
            *args: object,
            **kwargs: object,
        ) -> None:
            del text, y, args, kwargs
            texts.append((key, x))

        self.renderer._draw_text = capture_text
        try:
            self.renderer._draw_keycap("settings_hint_test", "SPACE", "PLAY/PAUSE", 100, 50)
        finally:
            self.renderer._draw_text = original_draw_text

        key_x = next(x for key, x in texts if key == "settings_hint_test_key")
        label_x = next(x for key, x in texts if key == "settings_hint_test_label")
        self.assertEqual(label_x - key_x, 33)

    def test_keyboard_hints_have_space_after_play_pause_label(self) -> None:
        texts = []
        original_draw_text = self.renderer._draw_text
        world = SimpleNamespace(
            is_paused=False,
            simulation_speed=1.0,
            MIN_SIMULATION_SPEED=0.25,
            MAX_SIMULATION_SPEED=5.0,
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, 1440, 900),
                environment=arcade.LBWH(0, 0, 1440, 900),
            ),
        )

        def capture_text(
            key: str,
            text: str,
            x: float,
            y: float,
            *args: object,
            **kwargs: object,
        ) -> None:
            del y, args, kwargs
            texts.append((key, text, x))

        self.renderer._draw_text = capture_text
        try:
            self.renderer._draw_settings_panel(world)
        finally:
            self.renderer._draw_text = original_draw_text

        play_pause = next(
            item for item in texts if item[0] == "settings_hint_0_label"
        )
        a_key = next(item for item in texts if item[0] == "settings_hint_1_key")
        play_pause_right = play_pause[2] + len(play_pause[1]) * 7.0
        self.assertGreaterEqual(a_key[2] - play_pause_right, 20.0)

    def test_selected_fitness_label_uses_live_creature_fitness(self) -> None:
        selected = SimpleNamespace(creature_id=938)
        live_fitness = SimpleNamespace(score=lambda creature: 7.25)
        world = SimpleNamespace(
            fitness_for=lambda creature: live_fitness
            if creature is selected
            else None,
        )

        self.assertEqual(
            self.renderer._selected_fitness_label(world, selected),
            "7.25",
        )


class SpeciesTreeWindowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.renderer = UiRenderer(build_sim_config())

    def make_record(
        self,
        species_id: int,
        parent_species_id: int | None,
        *,
        exact: bool = True,
        emerged_at: float | None = None,
        neural_shifts: tuple[NeuralShift, ...] = (),
    ) -> SpeciesRecord:
        traits = (
            SpeciesTraitSnapshot(16.0 + species_id, 100.0, 0.9, 1.0)
            if exact
            else None
        )
        deltas = (
            SpeciesTraitSnapshot(1.0, 2.0, 0.1, 0.02)
            if exact
            else None
        )
        return SpeciesRecord(
            species_id=species_id,
            parent_species_id=parent_species_id,
            founder_creature_id=species_id * 10 if exact else None,
            founder_genome_id=species_id * 100 if exact else None,
            emerged_at=(
                (float(species_id) * 60.0 if emerged_at is None else emerged_at)
                if exact
                else None
            ),
            founder_color=(100, 120, 140) if exact else None,
            data_quality="exact" if exact else "reconstructed",
            founder_traits=traits,
            trait_deltas=deltas,
            distances=SpeciesDistanceBreakdown(
                neat_distance=2.0 if exact else None,
                phenotypic_distance=0.6 if exact else None,
                weighted_phenotypic_distance=1.2 if exact else None,
                composite_distance=3.2 if exact else None,
                compatibility_threshold=3.0 if exact else None,
                phenotypic_weight=2.0 if exact else None,
                radius_component=0.2 if exact else None,
                vision_range_component=0.2 if exact else None,
                vision_angle_component=0.1 if exact else None,
                movement_cost_component=0.1 if exact else None,
            ),
            neural_shifts=neural_shifts,
        )

    def make_world(
        self,
        records: dict[int, SpeciesRecord] | None = None,
        *,
        paused: bool = False,
        width: float = 1440.0,
        height: float = 900.0,
        representatives: dict[int, object] | None = None,
    ) -> SimpleNamespace:
        active_records = records or {}
        emerged_times = [
            float(record.emerged_at)
            for record in active_records.values()
            if record.emerged_at is not None
        ]
        return SimpleNamespace(
            is_paused=paused,
            species_history=active_records,
            elapsed_time=max([0.0, *emerged_times]),
            config=self.renderer.config,
            telemetry=None,
            neat_controller=SimpleNamespace(
                config=SimpleNamespace(
                    genome_config=SimpleNamespace(
                        input_keys=tuple(range(-1, -39, -1)),
                        output_keys=tuple(range(12)),
                    )
                ),
                species_manager=SimpleNamespace(
                    representatives=representatives or {}
                ),
            ),
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, width, height),
                environment=arcade.LBWH(0, 0, width, height),
            ),
        )

    def test_open_and_close_restore_running_state(self) -> None:
        world = self.make_world()
        self.renderer._control_hitboxes["open_species_tree"] = arcade.LBWH(
            0, 0, 30, 30
        )

        self.assertTrue(self.renderer.handle_mouse_press(world, 10, 10))
        self.assertTrue(world.is_paused)
        self.assertTrue(self.renderer.species_tree_open)
        self.renderer._control_hitboxes["species_tree_close"] = arcade.LBWH(
            40, 40, 30, 30
        )
        self.assertTrue(self.renderer.handle_mouse_press(world, 50, 50))
        self.assertFalse(world.is_paused)
        self.assertFalse(self.renderer.species_tree_open)

    def test_close_preserves_preexisting_pause(self) -> None:
        world = self.make_world(paused=True)

        self.renderer.open_species_tree(world)
        self.renderer.close_species_tree(world)

        self.assertTrue(world.is_paused)

    def test_regular_draw_skips_species_history_query_when_tree_is_closed(
        self,
    ) -> None:
        world = self.make_world({1: self.make_record(1, None)})
        load_end_times = Mock(return_value={})
        world.telemetry = SimpleNamespace(
            load_species_end_times=load_end_times,
        )

        with (
            patch.object(self.renderer, "_draw_icon_rail"),
            patch.object(self.renderer, "_draw_floating_panels"),
            patch.object(self.renderer, "_draw_brain_window"),
        ):
            self.renderer.draw(world)

        load_end_times.assert_not_called()

    def test_species_layout_reuses_unchanged_telemetry_snapshot(self) -> None:
        world = self.make_world({1: self.make_record(1, None)})
        load_end_times = Mock(return_value={1: float("inf")})
        world.telemetry = SimpleNamespace(
            load_species_end_times=load_end_times,
        )

        first = self.renderer._sync_species_tree_layout(world)
        second = self.renderer._sync_species_tree_layout(world)

        self.assertIs(second, first)
        load_end_times.assert_called_once_with(up_to_time=world.elapsed_time)

    def test_modal_captures_keyboard_and_underlying_controls(self) -> None:
        calls = []
        world = self.make_world()
        world.toggle_pause = lambda: calls.append("pause")
        self.renderer.open_species_tree(world)
        self.renderer._control_hitboxes["pause"] = arcade.LBWH(0, 0, 40, 40)

        self.assertTrue(self.renderer.handle_key_press(world, 1, 0))
        self.assertTrue(self.renderer.handle_mouse_press(world, 10, 10))
        self.assertEqual(calls, [])
        self.assertTrue(world.is_paused)

    def test_cancelled_pointer_capture_does_not_commit_species_click(
        self,
    ) -> None:
        records = {
            1: self.make_record(1, None),
            2: self.make_record(2, 1),
        }
        world = self.make_world(records)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        previous_selection = self.renderer._species_tree_selected_id
        target_id = 1 if previous_selection == 2 else 2
        node = self.renderer._species_tree_node_bounds[target_id]

        self.renderer.handle_mouse_press(
            world,
            node.center_x,
            node.center_y,
        )
        self.renderer.cancel_pointer_interaction()

        self.assertEqual(
            self.renderer._species_tree_selected_id,
            previous_selection,
        )
        self.assertIsNone(self.renderer._species_tree_pending_selection_id)
        self.assertFalse(self.renderer._species_tree_canvas_drag)

    def test_window_uses_full_inset_bounds_and_registers_nodes(self) -> None:
        records = {
            1: self.make_record(1, None),
            2: self.make_record(2, 1),
        }
        world = self.make_world(records)
        self.renderer.open_species_tree(world)

        self.renderer._draw_species_tree_window(world)

        bounds = self.renderer._control_hitboxes["species_tree_window"]
        self.assertEqual((bounds.left, bounds.bottom), (20, 20))
        self.assertEqual((bounds.width, bounds.height), (1400, 860))
        self.assertEqual(set(self.renderer._species_tree_node_bounds), {1, 2})

    def test_species_tree_header_uses_menu_icon_and_selected_context(self) -> None:
        records = {
            1: self.make_record(1, None),
            2: self.make_record(2, 1),
        }
        world = self.make_world(records)
        icons: list[tuple[str, str]] = []
        original_draw_icon = self.renderer._draw_icon
        self.renderer._draw_icon = (
            lambda bounds, icon_name, key: icons.append((icon_name, key))
        )
        try:
            self.renderer.open_species_tree(world)
            self.renderer._draw_species_tree_window(world)
            node = self.renderer._species_tree_node_bounds[2]
            self.renderer.handle_mouse_press(
                world,
                node.center_x,
                node.center_y,
            )
            self.renderer.handle_mouse_release()
            self.renderer._draw_species_tree_window(world)
        finally:
            self.renderer._draw_icon = original_draw_icon

        self.assertIn(("species", "species_tree_title_icon"), icons)
        self.assertEqual(
            self.renderer._text_cache["species_tree_panel_title"].text,
            "Species Evolution Tree  /  Selected: Species 2",
        )
        self.assertEqual(
            self.renderer._text_cache["species_tree_selected_label"].text,
            "Species 2",
        )

    def test_species_tree_wide_layout_places_legend_between_canvas_and_inspector(
        self,
    ) -> None:
        records = {
            1: self.make_record(1, None),
            2: self.make_record(2, 1),
        }
        world = self.make_world(records, width=1440, height=900)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        node = self.renderer._species_tree_node_bounds[2]
        self.renderer.handle_mouse_press(world, node.center_x, node.center_y)
        self.renderer.handle_mouse_release()

        self.renderer._draw_species_tree_window(world)

        timeline = self.renderer._control_hitboxes["species_tree_timeline"]
        canvas = self.renderer._control_hitboxes["species_tree_canvas"]
        legend = self.renderer._control_hitboxes["species_tree_legend"]
        resize = self.renderer._control_hitboxes["species_tree_inspector_resize"]
        inspector_left = resize.left + 5.0
        self.assertLessEqual(timeline.right, canvas.left)
        self.assertLessEqual(canvas.right, legend.left)
        self.assertLessEqual(legend.right, inspector_left)
        self.assertGreater(canvas.width, 0.0)

    def test_species_tree_compact_layout_hides_legend_and_preserves_canvas(
        self,
    ) -> None:
        records = {
            1: self.make_record(1, None),
            2: self.make_record(2, 1),
        }
        world = self.make_world(records, width=1000, height=700)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        node = self.renderer._species_tree_node_bounds[2]
        self.renderer.handle_mouse_press(world, node.center_x, node.center_y)
        self.renderer.handle_mouse_release()

        self.renderer._draw_species_tree_window(world)

        self.assertNotIn("species_tree_legend", self.renderer._control_hitboxes)
        timeline = self.renderer._control_hitboxes["species_tree_timeline"]
        canvas = self.renderer._control_hitboxes["species_tree_canvas"]
        resize = self.renderer._control_hitboxes["species_tree_inspector_resize"]
        self.assertLessEqual(timeline.right, canvas.left)
        self.assertLessEqual(canvas.right, resize.left + 5.0)
        self.assertGreater(canvas.width, 180.0)

    def test_runtime_species_lifecycle_tracks_extinction_and_revival(self) -> None:
        world = self.make_world(
            {1: self.make_record(1, None, emerged_at=0.0)}
        )
        world.elapsed_time = 5.0
        world.creatures = [
            SimpleNamespace(lineage=SimpleNamespace(species_id=1))
        ]

        living = self.renderer._sync_species_tree_layout(world)
        self.assertEqual(living.end_times[1], float("inf"))

        world.elapsed_time = 12.0
        world.creatures = []
        extinct = self.renderer._sync_species_tree_layout(world)
        self.assertEqual(extinct.end_times[1], 12.0)

        world.elapsed_time = 15.0
        world.creatures = [
            SimpleNamespace(lineage=SimpleNamespace(species_id=1))
        ]
        revived = self.renderer._sync_species_tree_layout(world)
        self.assertEqual(revived.end_times[1], float("inf"))

    def test_loaded_species_tree_uses_telemetry_extinction_times(self) -> None:
        world = self.make_world(
            {
                1: self.make_record(1, None, emerged_at=0.0),
                2: self.make_record(2, 1, emerged_at=15.0),
            }
        )
        world.elapsed_time = 20.0
        world.creatures = [
            SimpleNamespace(lineage=SimpleNamespace(species_id=2))
        ]
        world.telemetry = SimpleNamespace(
            load_species_end_times=lambda *, up_to_time: {
                1: 10.0,
                2: float("inf"),
            }
        )

        layout = self.renderer._sync_species_tree_layout(world)

        self.assertEqual(layout.end_times[1], 10.0)
        self.assertEqual(layout.end_times[2], float("inf"))
        self.assertEqual(layout.lanes[1], layout.lanes[2])

    def test_species_endpoint_markers_distinguish_extant_and_extinct(self) -> None:
        with patch("src.ui.renderer.arcade.draw_circle_filled") as filled:
            self.renderer._draw_species_tree_extant_marker(
                (10.0, 20.0),
                (100, 120, 140),
            )

        self.assertEqual(filled.call_count, 3)
        self.assertEqual(filled.call_args.args[-1], (100, 120, 140, 255))

        with patch("src.ui.renderer.arcade.draw_line") as line:
            self.renderer._draw_species_tree_extinct_marker((10.0, 20.0))

        self.assertEqual(line.call_count, 2)
        self.assertTrue(
            all(
                call.args[-2] == self.renderer.theme.text_muted
                for call in line.call_args_list
            )
        )

    def test_species_tree_refines_founder_color_toward_card_background(self) -> None:
        record = self.make_record(2, 1)

        softened = self.renderer._species_tree_refined_color(record, 0.5)

        self.assertEqual(
            softened,
            self.renderer._brain_blend_color(
                self.renderer.theme.card_background,
                record.founder_color,
                0.5,
            ),
        )
        self.assertNotEqual(softened, record.founder_color)

    def test_species_tree_line_color_has_more_contrast_than_soft_tint(self) -> None:
        record = replace(
            self.make_record(2, 1),
            founder_color=(245, 220, 80),
        )
        background = self.renderer.theme.card_background
        softened = self.renderer._species_tree_refined_color(record, 0.43)

        visible = self.renderer._species_tree_line_color(record)

        softened_distance = sum(
            abs(background[index] - softened[index]) for index in range(3)
        )
        visible_distance = sum(
            abs(background[index] - visible[index]) for index in range(3)
        )
        self.assertGreater(visible_distance, softened_distance)
        self.assertEqual(visible[3], 230)
        self.assertGreaterEqual(
            self.renderer._species_tree_contrast_ratio(
                visible,
                background,
                visible[3],
            ),
            3.5,
        )
        extinct = self.renderer._species_tree_line_color(
            record,
            alpha=150,
            muted=True,
        )
        self.assertGreaterEqual(
            self.renderer._species_tree_contrast_ratio(
                extinct,
                background,
                extinct[3],
            ),
            3.5,
        )
        for founder_color in (
            (255, 255, 255),
            (80, 220, 220),
            (255, 180, 200),
            (40, 50, 70),
        ):
            candidate = replace(record, founder_color=founder_color)
            for alpha, muted in ((230, False), (150, True)):
                line_color = self.renderer._species_tree_line_color(
                    candidate,
                    alpha=alpha,
                    muted=muted,
                )
                self.assertGreaterEqual(
                    self.renderer._species_tree_contrast_ratio(
                        line_color,
                        background,
                        line_color[3],
                    ),
                    3.5,
                )

    def test_species_tree_lifeline_uses_visible_core_and_understroke(self) -> None:
        record = self.make_record(1, None, emerged_at=0.0)
        world = self.make_world({1: record})
        world.elapsed_time = 20.0
        world.creatures = [
            SimpleNamespace(lineage=SimpleNamespace(species_id=1))
        ]
        layout = self.renderer._sync_species_tree_layout(world)
        canvas = arcade.LBWH(0.0, 0.0, 400.0, 300.0)

        with patch.object(
            self.renderer,
            "_draw_species_tree_path",
        ) as draw_path:
            self.renderer._draw_species_tree_lifelines(
                {1: record},
                layout,
                (1,),
                canvas,
                set(),
            )

        colors = [call.args[1] for call in draw_path.call_args_list]
        self.assertIn(
            self.renderer._brain_color_alpha(
                self.renderer.theme.text_primary,
                30,
            ),
            colors,
        )
        self.assertIn(
            self.renderer._species_tree_line_color(record, alpha=235),
            colors,
        )

    def test_species_tree_legend_describes_curved_high_contrast_lineage(self) -> None:
        self.renderer._draw_species_tree_legend(
            arcade.LBWH(0.0, 0.0, 220.0, 700.0)
        )

        self.assertEqual(
            self.renderer._text_cache["species_tree_legend_line_0"].text,
            "Curved lineage",
        )

    def test_species_tree_dashed_paths_and_selected_edges_have_hierarchy(self) -> None:
        records = {
            1: self.make_record(1, None, emerged_at=0.0),
            2: self.make_record(2, 1, emerged_at=10.0),
        }
        world = self.make_world(records)
        layout = self.renderer._sync_species_tree_layout(world)
        layout = replace(layout, end_times={1: float("inf"), 2: 20.0})
        route = {(1, 2): ((10.0, 20.0), (110.0, 20.0))}

        with (
            patch("src.ui.renderer.arcade.draw_line") as line,
            patch("src.ui.renderer.arcade.draw_circle_filled"),
        ):
            self.renderer._draw_species_tree_edges(
                records,
                layout,
                ((1, 2),),
                route,
                {(1, 2)},
            )

        widths = [call.args[-1] for call in line.call_args_list]
        colors = [call.args[-2] for call in line.call_args_list]
        self.assertGreater(len(line.call_args_list), 3)
        self.assertGreater(max(widths), min(widths))
        self.assertIn(
            self.renderer._brain_color_alpha(self.renderer.theme.accent, 82),
            colors,
        )

    def test_selected_species_path_brightens_ancestry_and_dims_other_edges(
        self,
    ) -> None:
        records = {
            1: self.make_record(1, None, emerged_at=0.0),
            2: self.make_record(2, 1, emerged_at=10.0),
            3: self.make_record(3, 1, emerged_at=20.0),
        }
        world = self.make_world(records)
        world.creatures = [
            SimpleNamespace(lineage=SimpleNamespace(species_id=species_id))
            for species_id in records
        ]
        layout = self.renderer._sync_species_tree_layout(world)
        routes = {
            (1, 2): ((20.0, 80.0), (120.0, 80.0)),
            (1, 3): ((20.0, 40.0), (120.0, 40.0)),
        }
        self.renderer._species_tree_selected_id = 2

        with (
            patch.object(self.renderer, "_draw_species_tree_path") as path,
            patch("src.ui.renderer.arcade.draw_circle_filled"),
        ):
            self.renderer._draw_species_tree_edges(
                records,
                layout,
                ((1, 2), (1, 3)),
                routes,
                {(1, 2)},
            )

        colors = [call.args[1] for call in path.call_args_list]
        widths = [call.args[2] for call in path.call_args_list]
        self.assertIn(
            self.renderer._species_tree_line_color(records[2], alpha=245),
            colors,
        )
        self.assertIn(
            self.renderer._species_tree_line_color(
                records[3],
                alpha=175,
                muted=True,
            ),
            colors,
        )
        self.assertIn(
            self.renderer._brain_color_alpha(self.renderer.theme.accent, 82),
            colors,
        )
        self.assertGreater(max(widths), min(widths))

    def test_selected_species_path_dims_unrelated_node_core(self) -> None:
        records = {
            1: self.make_record(1, None, emerged_at=0.0),
            2: self.make_record(2, 1, emerged_at=10.0),
            3: self.make_record(3, 1, emerged_at=20.0),
        }
        layout = self.renderer._sync_species_tree_layout(self.make_world(records))
        positions = {1: (80.0, 180.0), 2: (150.0, 110.0), 3: (220.0, 70.0)}
        self.renderer._species_tree_selected_id = 2

        with patch("src.ui.renderer.arcade.draw_circle_filled") as filled:
            self.renderer._draw_species_tree_nodes(
                records,
                layout,
                (1, 2, 3),
                ((1, 2), (1, 3)),
                positions,
                {1, 2},
                arcade.LBWH(0.0, 0.0, 300.0, 240.0),
            )

        filled.assert_any_call(
            positions[3][0],
            positions[3][1],
            self.renderer._species_tree_node_visual_radius(layout, 3),
            self.renderer._brain_color_alpha(records[3].founder_color, 145),
        )

    def test_selected_species_path_brightens_lifelines(self) -> None:
        records = {
            1: self.make_record(1, None, emerged_at=0.0),
            2: self.make_record(2, 1, emerged_at=10.0),
            3: self.make_record(3, 1, emerged_at=20.0),
        }
        world = self.make_world(records)
        world.elapsed_time = 60.0
        world.creatures = [
            SimpleNamespace(lineage=SimpleNamespace(species_id=species_id))
            for species_id in records
        ]
        layout = self.renderer._sync_species_tree_layout(world)
        self.renderer._species_tree_selected_id = 2

        with (
            patch.object(self.renderer, "_draw_species_tree_path") as path,
            patch.object(self.renderer, "_draw_species_tree_extant_marker"),
        ):
            self.renderer._draw_species_tree_lifelines(
                records,
                layout,
                (1, 2, 3),
                arcade.LBWH(0.0, 0.0, 500.0, 400.0),
                {1, 2},
            )

        colors = [call.args[1] for call in path.call_args_list]
        self.assertIn(
            self.renderer._species_tree_line_color(records[2], alpha=245),
            colors,
        )
        self.assertIn(
            self.renderer._species_tree_line_color(
                records[3],
                alpha=175,
                muted=True,
            ),
            colors,
        )

    def test_species_tree_mutation_intensity_uses_exact_available_changes(self) -> None:
        root = self.make_record(1, None)
        child = self.make_record(2, 1)
        zero_traits = SpeciesTraitSnapshot(0.0, 0.0, 0.0, 0.0)
        empty_changes = NeatChangeSummary.empty()

        self.assertEqual(self.renderer._species_tree_mutation_intensity(root), 0.0)
        self.assertEqual(
            self.renderer._species_tree_mutation_intensity(
                replace(child, data_quality="reconstructed")
            ),
            0.0,
        )
        self.assertEqual(
            self.renderer._species_tree_mutation_intensity(
                replace(child, trait_deltas=None, neat_changes=None)
            ),
            0.0,
        )
        self.assertEqual(
            self.renderer._species_tree_mutation_intensity(
                replace(child, trait_deltas=None, neat_changes=empty_changes)
            ),
            0.0,
        )
        self.assertEqual(
            self.renderer._species_tree_mutation_intensity(
                replace(child, neat_changes=None)
            ),
            0.0,
        )
        self.assertEqual(
            self.renderer._species_tree_mutation_intensity(
                replace(child, trait_deltas=zero_traits, neat_changes=empty_changes)
            ),
            0.0,
        )
        trait_only = self.renderer._species_tree_mutation_intensity(
            replace(
                child,
                trait_deltas=SpeciesTraitSnapshot(1.0, 0.0, 0.0, 0.0),
                neat_changes=empty_changes,
            )
        )
        structural = self.renderer._species_tree_mutation_intensity(
            replace(
                child,
                trait_deltas=zero_traits,
                neat_changes=replace(empty_changes, nodes_added=2),
            )
        )
        high = self.renderer._species_tree_mutation_intensity(
            replace(
                child,
                neat_changes=replace(
                    empty_changes,
                    nodes_added=100,
                    weights_changed=100,
                ),
            )
        )
        self.assertGreater(trait_only, 0.0)
        self.assertGreater(structural, 0.0)
        self.assertEqual(high, 1.0)
        self.assertEqual(
            [
                self.renderer._species_tree_mutation_tick_count(value)
                for value in (trait_only, structural, high)
            ],
            [1, 2, 3],
        )

    def test_species_tree_route_corners_are_rounded(self) -> None:
        route = ((20.0, 100.0), (100.0, 100.0), (100.0, 20.0))

        rounded = self.renderer._species_tree_rounded_route_points(route)

        self.assertEqual(rounded[0], route[0])
        self.assertEqual(rounded[-1], route[-1])
        self.assertGreater(len(rounded), len(route))
        self.assertNotIn(route[1], rounded)

    def test_species_tree_horizontal_connector_gets_a_soft_elbow(self) -> None:
        junction = (20.0, 100.0)
        endpoint = (120.0, 100.0)

        softened = self.renderer._species_tree_soft_route_points(
            (junction, endpoint)
        )

        self.assertEqual(softened[0], junction)
        self.assertEqual(softened[-1], endpoint)
        self.assertGreater(len(softened), 3)
        self.assertGreater(max(point[1] for point in softened), junction[1])
        self.assertTrue(
            all(
                first[0] <= second[0]
                for first, second in zip(softened, softened[1:])
            )
        )

    def test_species_tree_nodes_use_compact_visuals_and_large_hit_targets(self) -> None:
        records = {1: self.make_record(1, None, emerged_at=0.0)}
        world = self.make_world(records)
        world.creatures = [
            SimpleNamespace(lineage=SimpleNamespace(species_id=1))
        ]
        self.renderer.open_species_tree(world)

        self.renderer._draw_species_tree_window(world)

        layout = self.renderer._species_tree_last_layout
        visual_radius = self.renderer._species_tree_node_visual_radius(layout, 1)
        hitbox = self.renderer._species_tree_node_bounds[1]
        self.assertAlmostEqual(visual_radius, 6.25)
        self.assertLessEqual(visual_radius, 11.0)
        self.assertGreaterEqual(hitbox.width, 24.0)
        self.assertGreater(hitbox.width, visual_radius * 2.0)

    def test_species_tree_signed_content_origin_maps_into_canvas_padding(self) -> None:
        records = {
            1: self.make_record(1, None, emerged_at=0.0),
            2: self.make_record(2, 1, emerged_at=10.0),
            3: self.make_record(3, 1, emerged_at=20.0),
        }
        layout = self.renderer._sync_species_tree_layout(self.make_world(records))
        canvas = arcade.LBWH(100.0, 80.0, layout.content_width, 400.0)
        self.renderer._species_tree_fit_mode = False
        self.renderer._species_tree_zoom = 1.0
        self.renderer._species_tree_horizontal_offset = 0.0

        positions = self.renderer._species_tree_screen_positions(layout, canvas)
        visible = self.renderer._species_tree_visible_content_bounds(
            layout,
            canvas,
        )

        leftmost_id = min(layout.positions, key=lambda key: layout.positions[key][0])
        self.assertAlmostEqual(
            positions[leftmost_id][0],
            canvas.left + self.renderer.SPECIES_TREE_CONTENT_PADDING,
        )
        self.assertAlmostEqual(visible[0], layout.content_left)
        self.assertAlmostEqual(
            visible[1],
            layout.content_left + layout.content_width,
        )

    def test_species_tree_left_growth_keeps_existing_node_fixed_when_not_fitted(
        self,
    ) -> None:
        records = {1: self.make_record(1, None, emerged_at=0.0)}
        world = self.make_world(records)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        self.renderer._species_tree_fit_mode = False
        self.renderer._species_tree_zoom = 1.0
        self.renderer._draw_species_tree_window(world)
        original_x = self.renderer._species_tree_node_bounds[1].center_x

        records[2] = self.make_record(2, 1, emerged_at=10.0)
        world.elapsed_time = 10.0
        world.creatures = [
            SimpleNamespace(lineage=SimpleNamespace(species_id=species_id))
            for species_id in (1, 2)
        ]
        self.renderer._draw_species_tree_window(world)

        self.assertAlmostEqual(
            self.renderer._species_tree_node_bounds[1].center_x,
            original_x,
        )
        self.assertLess(
            self.renderer._species_tree_last_layout.content_left,
            -self.renderer.SPECIES_TREE_CONTENT_PADDING,
        )

    def test_species_tree_context_labels_respect_zoom_priority_and_bounds(self) -> None:
        records = {
            1: self.make_record(1, None, emerged_at=0.0),
            2: self.make_record(2, 1, emerged_at=10.0),
            3: self.make_record(3, 2, emerged_at=20.0),
            4: self.make_record(4, 3, emerged_at=30.0),
        }
        layout = self.renderer._sync_species_tree_layout(self.make_world(records))
        canvas = arcade.LBWH(0.0, 0.0, 420.0, 300.0)
        positions = {
            1: (80.0, 230.0),
            2: (150.0, 175.0),
            3: (220.0, 120.0),
            4: (290.0, 65.0),
        }
        radii = {species_id: 5.0 for species_id in records}
        self.renderer._species_tree_selected_id = 3

        self.renderer._species_tree_zoom = 0.79
        compact = self.renderer._species_tree_context_labels(
            layout,
            layout.edges,
            positions,
            radii,
            {1, 2, 3},
            canvas,
        )
        self.assertEqual([(label.species_id, label.text) for label in compact], [(3, "Species 3")])

        self.renderer._species_tree_zoom = 1.15
        expanded = self.renderer._species_tree_context_labels(
            layout,
            layout.edges,
            positions,
            radii,
            {1, 2, 3},
            canvas,
        )
        labels_by_id = {label.species_id: label for label in expanded}
        self.assertEqual(labels_by_id[3].text, "Species 3")
        self.assertEqual(labels_by_id[2].text, "S2")
        self.assertEqual(labels_by_id[4].text, "S4")
        for label in expanded:
            self.assertGreaterEqual(label.bounds.left, canvas.left)
            self.assertLessEqual(label.bounds.right, canvas.right)
            self.assertGreaterEqual(label.bounds.bottom, canvas.bottom)
            self.assertLessEqual(label.bounds.top, canvas.top)
        for index, first in enumerate(expanded):
            for second in expanded[index + 1 :]:
                self.assertFalse(
                    self.renderer._species_tree_rects_overlap(
                        first.bounds,
                        second.bounds,
                    )
                )

    def test_species_tree_focus_band_aligns_with_selected_node_time(self) -> None:
        records = {
            1: self.make_record(1, None, emerged_at=0.0),
            2: self.make_record(2, 1, emerged_at=10.0),
        }
        layout = self.renderer._sync_species_tree_layout(self.make_world(records))
        canvas = arcade.LBWH(100.0, 80.0, 500.0, 400.0)
        self.renderer._species_tree_selected_id = 2
        expected_y = self.renderer._species_tree_screen_point(
            layout.positions[2],
            layout,
            canvas,
        )[1]

        with (
            patch("src.ui.renderer.arcade.draw_line") as line,
            patch("src.ui.renderer.arcade.draw_lrbt_rectangle_filled") as focus,
        ):
            self.renderer._draw_species_tree_canvas_grid(layout, canvas)

        focus.assert_called_once_with(
            canvas.left,
            canvas.right,
            expected_y - 8.0,
            expected_y + 8.0,
            self.renderer._brain_color_alpha(self.renderer.theme.accent, 16),
        )
        self.assertTrue(
            any(
                call.args[1] == expected_y and call.args[3] == expected_y
                for call in line.call_args_list
            )
        )

    def test_hover_finds_node_and_legacy_tooltip_marks_unavailable_data(self) -> None:
        record = self.make_record(1, None, exact=False)
        world = self.make_world({1: record})
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        node = self.renderer._species_tree_node_bounds[1]

        self.assertTrue(
            self.renderer.handle_mouse_motion(
                world, node.center_x, node.center_y
            )
        )
        self.assertEqual(self.renderer._species_tree_hovered_id, 1)
        self.assertIn(
            "Unavailable",
            " ".join(self.renderer._species_tree_tooltip_lines(record)),
        )

    def test_click_selects_node_and_highlights_its_ancestry(self) -> None:
        records = {
            1: self.make_record(1, None),
            2: self.make_record(2, 1),
            3: self.make_record(3, 2),
        }
        world = self.make_world(records)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        node = self.renderer._species_tree_node_bounds[3]

        self.renderer.handle_mouse_press(world, node.center_x, node.center_y)
        self.renderer.handle_mouse_release()
        nodes, edges = self.renderer._species_tree_highlighted_path(
            self.renderer._species_tree_last_layout
        )

        self.assertEqual(self.renderer._species_tree_selected_id, 3)
        self.assertEqual(nodes, {1, 2, 3})
        self.assertEqual(edges, {(1, 2), (2, 3)})

    def test_parent_button_selects_and_focuses_the_direct_parent(self) -> None:
        records = {
            1: self.make_record(1, None, emerged_at=0.0),
            2: self.make_record(2, 1, emerged_at=10.0),
            3: self.make_record(3, 2, emerged_at=20.0),
        }
        world = self.make_world(records)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        self.renderer._select_species_tree_species(3)
        self.renderer._draw_species_tree_window(world)
        self.assertIsNotNone(self.renderer._species_tree_state.neuro_integration_view)
        parent_button = self.renderer._control_hitboxes[
            "species_tree_parent_button"
        ]

        handled = self.renderer.handle_mouse_press(
            world,
            parent_button.center_x,
            parent_button.center_y,
        )

        self.assertTrue(handled)
        self.assertEqual(self.renderer._species_tree_selected_id, 2)
        self.assertEqual(self.renderer._species_tree_zoom, 1.0)
        self.assertFalse(self.renderer._species_tree_fit_mode)
        self.assertIsNone(self.renderer._species_tree_report_species_id)
        self.assertIsNone(self.renderer._species_tree_state.neuro_integration_view)
        self.assertEqual(
            self.renderer._scroll_offsets["species_tree_inspector"],
            0.0,
        )
        nodes, edges = self.renderer._species_tree_highlighted_path(
            self.renderer._species_tree_last_layout
        )
        self.assertEqual(nodes, {1, 2})
        self.assertEqual(edges, {(1, 2)})

    def test_parent_button_label_updates_and_founder_state_is_disabled(self) -> None:
        records = {
            1: self.make_record(1, None, emerged_at=0.0),
            2: self.make_record(2, 1, emerged_at=10.0),
        }
        world = self.make_world(records)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)

        self.renderer._select_species_tree_species(2)
        self.renderer._draw_species_tree_window(world)
        self.assertIn("species_tree_parent_button", self.renderer._control_hitboxes)
        self.assertEqual(
            self.renderer._text_cache["species_tree_parent_navigation"].text,
            "View parent",
        )
        self.assertEqual(
            self.renderer._text_cache["species_tree_parent_lineage"].text,
            "Descended from Species 1",
        )

        self.renderer._select_species_tree_species(1)
        self.renderer._draw_species_tree_window(world)
        self.assertNotIn(
            "species_tree_parent_button",
            self.renderer._control_hitboxes,
        )
        self.assertEqual(
            self.renderer._text_cache["species_tree_parent_lineage"].text,
            "Founder species · No parent",
        )

    def test_dragging_from_node_pans_without_changing_selection(self) -> None:
        records = {
            1: self.make_record(1, None),
            2: self.make_record(2, 1),
        }
        world = self.make_world(records)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        self.renderer._species_tree_selected_id = 1
        node = self.renderer._species_tree_node_bounds[2]

        self.renderer.handle_mouse_press(world, node.center_x, node.center_y)
        self.renderer.handle_mouse_drag(
            world,
            node.center_x + 8.0,
            node.center_y + 8.0,
        )
        self.renderer.handle_mouse_release()

        self.assertEqual(self.renderer._species_tree_selected_id, 1)

    def test_inspector_report_is_generated_only_after_click(self) -> None:
        records = {
            1: self.make_record(1, None),
            2: self.make_record(2, 1),
        }
        world = self.make_world(records)
        self.renderer.open_species_tree(world)

        with patch(
            "src.ui.components.species_tree.inspector.generate_inspector_report",
            wraps=generate_inspector_report,
        ) as generate:
            self.renderer._draw_species_tree_window(world)
            node = self.renderer._species_tree_node_bounds[2]
            self.renderer.handle_mouse_motion(
                world,
                node.center_x,
                node.center_y,
            )
            self.renderer._draw_species_tree_window(world)
            self.assertEqual(generate.call_count, 0)

            self.renderer.handle_mouse_press(
                world,
                node.center_x,
                node.center_y,
            )
            self.renderer.handle_mouse_release()
            self.renderer._draw_species_tree_window(world)
            self.renderer._draw_species_tree_window(world)

            self.assertEqual(generate.call_count, 1)
            self.assertEqual(
                self.renderer._species_tree_report_species_id,
                2,
            )

    def test_radar_texture_is_lazy_cached_replaced_and_cleared(self) -> None:
        records = {
            1: self.make_record(1, None),
            2: self.make_record(2, 1),
        }
        genome = SimpleNamespace(
            nodes={
                key: SimpleNamespace(bias=0.0)
                for key in range(12)
            },
            connections={},
        )
        representative = (genome, object(), object(), object())
        world = self.make_world(
            records,
            representatives={1: representative, 2: representative},
        )
        self.renderer.open_species_tree(world)
        textures = [object(), object()]
        futures: list[Future[object]] = [Future(), Future()]
        executor = SimpleNamespace(
            submit=Mock(side_effect=futures),
        )
        self.renderer._species_tree_radar_executor = executor

        with (
            patch(
                "src.ui.renderer.arcade.Texture",
                side_effect=textures,
                create=True,
            ) as texture,
            patch("src.ui.renderer.arcade.draw_texture_rect", create=True),
        ):
            self.renderer._draw_species_tree_window(world)
            self.assertEqual(executor.submit.call_count, 0)

            child_node = self.renderer._species_tree_node_bounds[2]
            self.renderer.handle_mouse_press(
                world, child_node.center_x, child_node.center_y
            )
            self.renderer.handle_mouse_release()
            self.renderer._draw_species_tree_window(world)
            self.renderer._draw_species_tree_window(world)

            self.assertEqual(executor.submit.call_count, 1)
            self.assertEqual(texture.call_count, 0)
            self.assertIsNone(self.renderer._species_tree_radar_texture)
            self.assertEqual(
                self.renderer._text_cache["species_tree_radar_status"].text,
                "Loading behavioral profile...",
            )

            futures[0].set_result(object())
            self.renderer._draw_species_tree_window(world)

            self.assertEqual(texture.call_count, 1)
            self.assertIs(self.renderer._species_tree_radar_texture, textures[0])

            parent_node = self.renderer._species_tree_node_bounds[1]
            self.renderer.handle_mouse_press(
                world, parent_node.center_x, parent_node.center_y
            )
            self.renderer.handle_mouse_release()
            self.renderer._draw_species_tree_window(world)

            self.assertEqual(executor.submit.call_count, 2)
            self.assertEqual(texture.call_count, 1)
            self.assertIsNone(self.renderer._species_tree_radar_texture)

            futures[1].set_result(object())
            self.renderer._draw_species_tree_window(world)

            self.assertEqual(texture.call_count, 2)
            self.assertIs(self.renderer._species_tree_radar_texture, textures[1])

        self.renderer.close_species_tree(world)
        self.assertIsNone(self.renderer._species_tree_radar_texture)
        self.assertIsNone(self.renderer._species_tree_radar_species_id)
        self.renderer.open_species_tree(world)
        self.assertIsNone(self.renderer._species_tree_radar_texture)

    def test_species_radar_scores_complete_child_and_parent_representatives(
        self,
    ) -> None:
        records = {
            1: self.make_record(1, None),
            2: self.make_record(2, 1),
        }
        genomes = [
            SimpleNamespace(
                nodes={key: SimpleNamespace(bias=0.0) for key in range(12)},
                connections={},
            )
            for _ in range(2)
        ]
        physical = [
            PhysicalTraits(radius=16.0),
            PhysicalTraits(radius=18.0),
        ]
        vision = [
            SimpleNamespace(range=140.0, angle=0.8),
            SimpleNamespace(range=175.0, angle=1.4),
        ]
        flocking = [
            FlockingTraits(0.6, 0.4, 0.3),
            FlockingTraits(0.2, 0.8, 0.7),
        ]
        representatives = {
            1: (genomes[0], physical[0], vision[0], flocking[0]),
            2: (genomes[1], physical[1], vision[1], flocking[1]),
        }
        world = self.make_world(records, representatives=representatives)
        future: Future[object] = Future()
        executor = SimpleNamespace(submit=Mock(return_value=future))
        self.renderer._species_tree_radar_executor = executor
        self.renderer._species_tree_selected_id = 2

        with patch(
            "src.ui.components.species_tree.inspector.calculate_genotypic_behavior_scores",
            side_effect=((0.6,) * 6, (0.4,) * 6),
        ) as calculate:
            self.renderer._ensure_species_inspector_report(world, records)

        self.assertEqual(calculate.call_count, 2)
        child_call, parent_call = calculate.call_args_list
        self.assertIs(child_call.args[0], genomes[1])
        self.assertIs(child_call.kwargs["physical_traits"], physical[1])
        self.assertIs(child_call.kwargs["vision_traits"], vision[1])
        self.assertIs(child_call.kwargs["flocking_traits"], flocking[1])
        self.assertIs(parent_call.args[0], genomes[0])
        self.assertIs(parent_call.kwargs["physical_traits"], physical[0])
        self.assertIs(parent_call.kwargs["vision_traits"], vision[0])
        self.assertIs(parent_call.kwargs["flocking_traits"], flocking[0])
        executor.submit.assert_called_once_with(
            generate_radar_chart_image,
            (0.6,) * 6,
            (0.4,) * 6,
            BEHAVIOR_RADAR_LABELS,
            primary_label="Selected species",
        )

    def test_radar_draw_bounds_are_above_text_and_inside_viewport(self) -> None:
        viewport = arcade.LBWH(100.0, 100.0, 320.0, 500.0)
        self.renderer._species_tree_radar_texture = object()
        self.renderer._species_tree_radar_species_id = 2

        with patch("src.ui.renderer.arcade.draw_texture_rect", create=True) as draw:
            text_viewport = self.renderer._draw_species_radar_chart(viewport)

        draw.assert_called_once()
        texture, chart_bounds = draw.call_args.args
        self.assertIs(texture, self.renderer._species_tree_radar_texture)
        self.assertGreaterEqual(chart_bounds.left, viewport.left)
        self.assertLessEqual(chart_bounds.right, viewport.right)
        self.assertGreaterEqual(chart_bounds.bottom, viewport.bottom)
        self.assertLessEqual(chart_bounds.top, viewport.top)
        self.assertLess(text_viewport.top, chart_bounds.bottom)

    def test_radar_chart_grows_with_large_inspector_width(self) -> None:
        compact = self.renderer._species_radar_chart_size(320.0)
        medium = self.renderer._species_radar_chart_size(600.0)
        large = self.renderer._species_radar_chart_size(1200.0)

        self.assertEqual(compact, 220.0)
        self.assertEqual(medium, 372.0)
        self.assertEqual(
            large,
            self.renderer.SPECIES_RADAR_MAX_SIZE,
        )
        self.assertLess(compact, medium)
        self.assertLess(medium, large)

    def test_large_radar_draw_uses_responsive_maximum_size(self) -> None:
        viewport = arcade.LBWH(100.0, 100.0, 1200.0, 800.0)
        self.renderer._species_tree_radar_texture = object()
        self.renderer._species_tree_radar_species_id = 2

        with patch("src.ui.renderer.arcade.draw_texture_rect", create=True) as draw:
            self.renderer._draw_species_radar_chart(viewport)

        _, chart_bounds = draw.call_args.args
        self.assertEqual(
            chart_bounds.width,
            self.renderer.SPECIES_RADAR_MAX_SIZE,
        )
        self.assertEqual(chart_bounds.height, chart_bounds.width)

    def test_stale_radar_result_is_not_converted_to_texture(self) -> None:
        old_future: Future[object] = Future()
        old_future.set_result(object())
        new_future: Future[object] = Future()
        self.renderer._species_tree_open = True
        self.renderer._species_tree_selected_id = 2
        self.renderer._species_tree_radar_species_id = 1
        self.renderer._species_tree_radar_future = old_future

        with patch("src.ui.renderer.arcade.Texture", create=True) as texture:
            self.renderer._consume_species_radar_result()

        texture.assert_not_called()
        self.assertIsNone(self.renderer._species_tree_radar_texture)

        self.renderer._species_tree_radar_species_id = 2
        self.renderer._species_tree_radar_future = new_future
        new_future.set_result(object())
        with patch(
            "src.ui.renderer.arcade.Texture",
            return_value="texture",
            create=True,
        ) as texture:
            self.renderer._consume_species_radar_result()

        texture.assert_called_once()
        self.assertEqual(self.renderer._species_tree_radar_texture, "texture")

    def test_radar_render_failure_uses_unavailable_placeholder(self) -> None:
        future: Future[object] = Future()
        future.set_exception(RuntimeError("render failed"))
        self.renderer._species_tree_open = True
        self.renderer._species_tree_selected_id = 2
        self.renderer._species_tree_radar_species_id = 2
        self.renderer._species_tree_radar_future = future

        self.renderer._consume_species_radar_result()
        viewport = arcade.LBWH(100.0, 100.0, 320.0, 500.0)
        text_viewport = self.renderer._draw_species_radar_chart(viewport)

        self.assertEqual(self.renderer._species_tree_radar_error, "render_failed")
        self.assertEqual(
            self.renderer._text_cache["species_tree_radar_status"].text,
            "Behavioral profile unavailable",
        )
        self.assertLess(text_viewport.height, viewport.height)

    def test_close_cancels_radar_work_and_shuts_down_executor(self) -> None:
        future: Future[object] = Future()
        creature_future: Future[object] = Future()
        executor = Mock()
        self.renderer._species_tree_radar_future = future
        self.renderer._creature_radar_future = creature_future
        self.renderer._creature_radar_identity = (938, 44)
        self.renderer._species_tree_radar_executor = executor
        self.renderer._species_tree_radar_species_id = 2

        self.renderer.close()

        self.assertTrue(future.cancelled())
        self.assertTrue(creature_future.cancelled())
        self.assertIsNone(self.renderer._species_tree_radar_future)
        self.assertIsNone(self.renderer._creature_radar_future)
        self.assertIsNone(self.renderer._creature_radar_identity)
        self.assertIsNone(self.renderer._species_tree_radar_executor)
        executor.shutdown.assert_called_once_with(
            wait=False,
            cancel_futures=True,
        )

    def test_species_inspector_header_marker_matches_selected_node(self) -> None:
        parent = self.make_record(1, None)
        record = self.make_record(2, 1)
        report = generate_inspector_report(
            record,
            parent,
            None,
            self.renderer.config,
            range(12),
        )
        bounds = arcade.LBWH(100.0, 100.0, 340.0, 600.0)

        with (
            patch("src.ui.renderer.arcade.draw_circle_filled") as filled,
            patch("src.ui.renderer.arcade.draw_circle_outline") as outlined,
        ):
            self.renderer._draw_species_inspector(
                bounds,
                report,
                record,
            )

        summary = arcade.LBWH(
            bounds.left + 14.0,
            bounds.top - 44.0 - 72.0,
            bounds.width - 28.0,
            64.0,
        )
        marker_x = summary.left + 10.0
        marker_y = summary.center_y
        filled.assert_any_call(marker_x, marker_y, 10.0, record.founder_color)
        outlined.assert_any_call(
            marker_x,
            marker_y,
            10.0,
            self.renderer.theme.selected_outline,
            3.0,
        )
        self.assertEqual(
            self.renderer._text_cache[
                "species_tree_inspector_title"
            ].text,
            "Species 2",
        )
        self.assertEqual(
            self.renderer._text_cache["species_tree_inspector_quality"].text,
            "Data: Exact",
        )
        sections = self.renderer._species_inspector_sections(report, record)
        section_titles = {section.title for section in sections}
        self.assertIn("ANATOMY & MORPHOLOGY", section_titles)
        self.assertNotIn("PARENT COMPARISON", section_titles)
        self.assertIn("METABOLIC PROFILE", section_titles)
        self.assertIn("NEURO-INTEGRATION HUBS", section_titles)
        self.assertIn("BRAIN CHANGES FROM PARENT", section_titles)
        rows = {
            row.label: row.value
            for section in sections
            for row in section.rows
            if row.label is not None
        }
        self.assertEqual(rows["Radius"], "18.00 px · +5.9% vs parent")
        self.assertEqual(rows["Vision range"], "100.00 px · +2.0% vs parent")
        self.assertEqual(rows["Vision angle"], "0.900 rad · +12.5% vs parent")
        self.assertIn("energy/s", rows["Basal metabolic BMR"])
        self.assertIn(" / ", rows["Parent BMR / active"])

    def test_species_anatomy_rows_cover_every_parent_relative_trait(self) -> None:
        parent = self.make_record(1, None)
        record = replace(
            self.make_record(2, 1),
            founder_traits=SpeciesTraitSnapshot(
                radius=20.0,
                vision_range=120.0,
                vision_angle=1.2,
                movement_cost_multiplier=1.2,
                separation_gene=0.6,
                alignment_gene=0.4,
                cohesion_gene=0.8,
                stomach_capacity=2.0,
                digestion_rate=0.3,
                digestion_efficiency=0.9,
            ),
            trait_deltas=SpeciesTraitSnapshot(
                radius=2.0,
                vision_range=-10.0,
                vision_angle=0.0,
                movement_cost_multiplier=0.2,
                separation_gene=0.1,
                alignment_gene=-0.1,
                cohesion_gene=0.2,
                stomach_capacity=0.5,
                digestion_rate=0.05,
                digestion_efficiency=-0.05,
            ),
        )
        report = generate_inspector_report(
            record,
            parent,
            None,
            self.renderer.config,
            range(12),
        )

        anatomy = next(
            section
            for section in self.renderer._species_inspector_sections(
                report,
                record,
            )
            if section.title == "ANATOMY & MORPHOLOGY"
        )
        rows = {row.label: row for row in anatomy.rows}

        self.assertEqual(len(rows), 10)
        self.assertIn("+11.1% vs parent", rows["Radius"].value)
        self.assertIn("-7.7% vs parent", rows["Vision range"].value)
        self.assertIn("+0.0% vs parent", rows["Vision angle"].value)
        self.assertIn("+33.3% vs parent", rows["Stomach capacity"].value)
        self.assertIn("-5.3% vs parent", rows["Digestion efficiency"].value)
        self.assertIn("+20.0% vs parent", rows["Separation gene"].value)
        self.assertIn("-20.0% vs parent", rows["Alignment gene"].value)
        self.assertEqual(rows["Vision range"].tone, "negative")
        self.assertEqual(rows["Vision angle"].tone, "default")

    def test_species_anatomy_parent_comparison_fallbacks_are_explicit(self) -> None:
        root = self.make_record(1, None)
        root_report = generate_inspector_report(
            root,
            None,
            None,
            self.renderer.config,
            range(12),
        )
        root_anatomy = next(
            section
            for section in self.renderer._species_inspector_sections(
                root_report,
                root,
            )
            if section.title == "ANATOMY & MORPHOLOGY"
        )
        self.assertTrue(
            all("Root species" in row.value for row in root_anatomy.rows)
        )

        reconstructed = replace(
            self.make_record(2, 1),
            data_quality="reconstructed",
        )
        reconstructed_report = generate_inspector_report(
            reconstructed,
            root,
            None,
            self.renderer.config,
            range(12),
        )
        reconstructed_anatomy = next(
            section
            for section in self.renderer._species_inspector_sections(
                reconstructed_report,
                reconstructed,
            )
            if section.title == "ANATOMY & MORPHOLOGY"
        )
        self.assertTrue(
            all(
                "change unavailable" in row.value
                for row in reconstructed_anatomy.rows
            )
        )

        zero_parent = replace(
            self.make_record(2, 1),
            founder_traits=replace(
                self.make_record(2, 1).founder_traits,
                radius=1.0,
            ),
            trait_deltas=replace(
                self.make_record(2, 1).trait_deltas,
                radius=1.0,
            ),
        )
        zero_parent_report = generate_inspector_report(
            zero_parent,
            root,
            None,
            self.renderer.config,
            range(12),
        )
        zero_parent_anatomy = next(
            section
            for section in self.renderer._species_inspector_sections(
                zero_parent_report,
                zero_parent,
            )
            if section.title == "ANATOMY & MORPHOLOGY"
        )
        radius_row = next(
            row for row in zero_parent_anatomy.rows if row.label == "Radius"
        )
        self.assertIn("change unavailable", radius_row.value)

    def test_species_inspector_uses_neutral_frame_and_structured_scroll(self) -> None:
        parent = self.make_record(1, None)
        record = self.make_record(2, 1)
        report = generate_inspector_report(
            record,
            parent,
            None,
            self.renderer.config,
            range(12),
        )
        bounds = arcade.LBWH(100.0, 100.0, 340.0, 280.0)

        with patch.object(self.renderer, "_draw_rounded_rect") as rounded:
            self.renderer._draw_species_inspector(bounds, report, record)

        rounded.assert_any_call(
            bounds,
            self.renderer.theme.card_background,
            self.renderer.theme.panel_border,
            self.renderer.config.layout.card_radius,
            1.0,
        )
        self.assertIn("species_tree_inspector", self.renderer._scroll_regions)
        scroll_limit = self.renderer._scroll_limits["species_tree_inspector"]
        self.assertGreater(scroll_limit, 0.0)

        self.renderer._scroll_offsets["species_tree_inspector"] = scroll_limit
        self.renderer._draw_species_inspector(bounds, report, record)

        self.assertEqual(
            self.renderer._scroll_offsets["species_tree_inspector"],
            scroll_limit,
        )

    def test_neuro_integration_view_groups_canonical_inputs_and_outputs(self) -> None:
        parent = self.make_record(1, None)
        record = self.make_record(
            2,
            1,
            neural_shifts=(
                NeuralShift(-17, 53, "added", None, 0.41),
                NeuralShift(53, 8, "changed", 0.5, 2.74, 2.24),
                NeuralShift(-35, 26, "added", None, 0.36),
                NeuralShift(53, 0, "removed", -0.8, None),
                NeuralShift(53, 1, "added", None, 0.2),
                NeuralShift(-2, 53, "removed", 1.1, None),
            ),
        )
        input_keys = tuple(range(-1, -44, -1))
        output_keys = tuple(range(15))
        report = generate_inspector_report(
            record,
            parent,
            None,
            self.renderer.config,
            output_keys,
            input_keys,
        )

        view = self.renderer._build_neuro_integration_view(
            report,
            input_keys,
            output_keys,
        )

        self.assertEqual(
            (view.hub_count, view.incoming_count, view.outgoing_count),
            (2, 3, 3),
        )
        self.assertEqual([hub.hub_id for hub in view.hubs], [26, 53])
        hub = view.hubs[1]
        self.assertEqual(
            [row.source_node_id for row in hub.incoming_rows],
            [-2, -17],
        )
        self.assertEqual(
            [row.target_node_id for row in hub.outgoing_rows],
            [0, 1, 8],
        )
        self.assertEqual(
            [row.endpoint_primary for row in hub.outgoing_rows],
            ["Accelerate", "Turn", "Panic intensity"],
        )
        self.assertEqual(
            hub.outgoing_rows[0].classification,
            "Negative influence removed",
        )
        self.assertEqual(hub.outgoing_rows[0].child_sign, "No connection")
        self.assertEqual(hub.incoming_rows[-1].endpoint_primary, "Carrying something")

    def test_neuro_integration_view_uses_safe_unknown_node_fallbacks(self) -> None:
        parent = self.make_record(1, None)
        input_keys = (*tuple(range(-1, -47, -1)), -99)
        output_keys = (*tuple(range(16)), 99)
        record = self.make_record(
            2,
            1,
            neural_shifts=(
                NeuralShift(-99, 53, "added", None, 0.2),
                NeuralShift(53, 99, "added", None, 0.3),
            ),
        )
        report = generate_inspector_report(
            record,
            parent,
            None,
            self.renderer.config,
            output_keys,
            input_keys,
        )

        view = self.renderer._build_neuro_integration_view(
            report,
            input_keys,
            output_keys,
        )

        self.assertEqual(view.hubs[0].incoming_rows[0].endpoint_primary, "Input -99")
        self.assertEqual(view.hubs[0].outgoing_rows[0].endpoint_primary, "Output 99")

    def test_neuro_integration_incomplete_rows_do_not_guess_weights(self) -> None:
        parent = self.make_record(1, None)
        record = self.make_record(
            2,
            1,
            neural_shifts=(
                (53, -17, "weight", 0.6),
                (8, 53, "weight", -0.4),
            ),
        )
        input_keys = tuple(range(-1, -44, -1))
        output_keys = tuple(range(15))
        report = generate_inspector_report(
            record,
            parent,
            None,
            self.renderer.config,
            output_keys,
            input_keys,
        )

        view = self.renderer._build_neuro_integration_view(
            report,
            input_keys,
            output_keys,
        )

        rows = view.hubs[0].incoming_rows + view.hubs[0].outgoing_rows
        self.assertTrue(all(not row.weights_complete for row in rows))
        self.assertTrue(
            all(row.classification == "Historical weights unavailable" for row in rows)
        )
        self.assertEqual([row.transition for row in rows], ["None → None"] * 2)
        self.assertEqual([row.delta for row in rows], ["Δ +0.60", "Δ -0.40"])

    def test_neuro_integration_cards_use_shared_rows_and_standard_frame(self) -> None:
        parent = self.make_record(1, None)
        record = self.make_record(
            2,
            1,
            neural_shifts=(
                NeuralShift(-35, 26, "added", None, 0.36),
                NeuralShift(26, 8, "changed", 0.5, 2.74, 2.24),
            ),
        )
        input_keys = tuple(range(-1, -44, -1))
        output_keys = tuple(range(15))
        report = generate_inspector_report(
            record,
            parent,
            None,
            self.renderer.config,
            output_keys,
            input_keys,
        )
        view = self.renderer._build_neuro_integration_view(
            report,
            input_keys,
            output_keys,
        )
        bounds = arcade.LBWH(100.0, 100.0, 340.0, 900.0)

        with patch.object(self.renderer, "_draw_rounded_rect") as rounded:
            consumed = self.renderer._draw_neuro_integration_section(
                bounds,
                0,
                view,
                bounds.top,
                bounds.width,
            )

        self.assertEqual(
            consumed,
            self.renderer._neuro_integration_section_height(view, bounds.width),
        )
        self.assertTrue(
            any(
                call.args[1:5]
                == (
                    self.renderer.theme.card_background,
                    self.renderer.theme.panel_border,
                    self.renderer.config.layout.card_radius,
                    1.0,
                )
                for call in rounded.call_args_list
            )
        )
        texts = {text.text for text in self.renderer._text_cache.values()}
        self.assertIn("Integration Hub 26", texts)
        self.assertIn("Hidden neural node 26", texts)
        self.assertIn("INPUTS INTO HUB", texts)
        self.assertIn("OUTPUTS FROM HUB", texts)
        self.assertIn("Sound direction (cosine)", texts)
        self.assertIn("Panic intensity", texts)
        self.assertIn("ADDED", texts)
        self.assertIn("CHANGED", texts)
        self.assertIn("None → +0.36", texts)
        self.assertIn("+0.50 → +2.74", texts)
        self.assertIn("Δ +2.24", texts)

    def test_neuro_integration_minimum_width_stacks_shared_metadata(self) -> None:
        parent = self.make_record(1, None)
        record = self.make_record(
            2,
            1,
            neural_shifts=(
                NeuralShift(53, 13, "changed", -0.65, 0.75, 1.4),
            ),
        )
        input_keys = tuple(range(-1, -44, -1))
        output_keys = tuple(range(15))
        report = generate_inspector_report(
            record,
            parent,
            None,
            self.renderer.config,
            output_keys,
            input_keys,
        )
        view = self.renderer._build_neuro_integration_view(
            report,
            input_keys,
            output_keys,
        )
        width = 256.0
        bounds = arcade.LBWH(100.0, 100.0, width, 900.0)

        consumed = self.renderer._draw_neuro_integration_section(
            bounds,
            0,
            view,
            bounds.top,
            width,
        )

        self.assertEqual(
            consumed,
            self.renderer._neuro_integration_section_height(view, width),
        )
        expected_body_left = bounds.left + 94.0
        self.assertEqual(
            self.renderer._text_cache[
                "species_tree_neuro_hub_0_0_outgoing_row_0_delta"
            ].x,
            expected_body_left,
        )
        self.assertEqual(
            self.renderer._text_cache[
                "species_tree_neuro_hub_0_0_outgoing_row_0_movement"
            ].x,
            expected_body_left,
        )

    def test_neuro_integration_section_has_root_and_empty_states(self) -> None:
        input_keys = tuple(range(-1, -44, -1))
        output_keys = tuple(range(15))
        root_report = generate_inspector_report(
            self.make_record(1, None),
            None,
            None,
            self.renderer.config,
            output_keys,
            input_keys,
        )
        root_view = self.renderer._build_neuro_integration_view(
            root_report,
            input_keys,
            output_keys,
        )
        self.assertEqual(
            self.renderer._neuro_integration_intro_lines(root_view, 300.0),
            ("Founder species has no parent hub comparison.",),
        )

        empty_report = generate_inspector_report(
            self.make_record(2, 1),
            self.make_record(1, None),
            None,
            self.renderer.config,
            output_keys,
            input_keys,
        )
        empty_view = self.renderer._build_neuro_integration_view(
            empty_report,
            input_keys,
            output_keys,
        )
        self.assertTrue(
            any(
                "No hidden-node connection changes" in line
                for line in self.renderer._neuro_integration_intro_lines(
                    empty_view,
                    300.0,
                )
            )
        )

    def test_brain_change_view_groups_sources_and_keeps_exact_outputs(self) -> None:
        parent = self.make_record(1, None)
        record = self.make_record(
            2,
            1,
            neural_shifts=(
                NeuralShift(-17, 0, "added", None, 0.72),
                NeuralShift(-17, 1, "changed", -1.2, -0.6, 0.6),
                NeuralShift(-17, 8, "removed", -0.8, None),
            ),
        )
        input_keys = tuple(range(-1, -39, -1))
        output_keys = tuple(range(12))
        report = generate_inspector_report(
            record,
            parent,
            None,
            self.renderer.config,
            output_keys,
            input_keys,
        )

        view = self.renderer._build_brain_changes_view(
            report,
            input_keys,
            output_keys,
        )

        self.assertEqual(
            (view.total_count, view.added_count, view.changed_count, view.removed_count),
            (3, 1, 1, 1),
        )
        self.assertEqual(len(view.groups), 1)
        self.assertEqual(view.groups[0].source_primary, "Carrying something")
        self.assertEqual(
            [row.endpoint_primary for row in view.groups[0].rows],
            ["Accelerate", "Turn", "Panic intensity"],
        )
        removed = view.groups[0].rows[-1]
        self.assertEqual(removed.classification, "Negative influence removed")
        self.assertEqual(removed.child_sign, "No connection")
        self.assertEqual(removed.transition, "-0.80 → None")

    def test_brain_change_groups_and_rows_use_canonical_node_order(self) -> None:
        parent = self.make_record(1, None)
        record = self.make_record(
            2,
            1,
            neural_shifts=(
                NeuralShift(-17, 8, "added", None, 0.2),
                NeuralShift(-2, 8, "added", None, 0.3),
                NeuralShift(-2, 0, "added", None, 0.4),
            ),
        )
        input_keys = tuple(range(-1, -39, -1))
        output_keys = tuple(range(12))
        report = generate_inspector_report(
            record,
            parent,
            None,
            self.renderer.config,
            output_keys,
            input_keys,
        )

        view = self.renderer._build_brain_changes_view(
            report,
            input_keys,
            output_keys,
        )

        self.assertEqual([group.source_node_id for group in view.groups], [-2, -17])
        self.assertEqual(view.groups[0].connection_count, 2)
        self.assertEqual(
            [row.target_node_id for row in view.groups[0].rows],
            [0, 8],
        )

    def test_incomplete_historical_row_preserves_delta_without_guessing(self) -> None:
        parent = self.make_record(1, None)
        record = self.make_record(
            2,
            1,
            neural_shifts=((8, -17, "weight", 0.6),),
        )
        input_keys = tuple(range(-1, -39, -1))
        output_keys = tuple(range(12))
        report = generate_inspector_report(
            record,
            parent,
            None,
            self.renderer.config,
            output_keys,
            input_keys,
        )

        view = self.renderer._build_brain_changes_view(
            report,
            input_keys,
            output_keys,
        )
        row = view.groups[0].rows[0]

        self.assertFalse(row.weights_complete)
        self.assertEqual(row.classification, "Historical weights unavailable")
        self.assertEqual(row.transition, "None → None")
        self.assertEqual(row.delta, "Δ +0.60")

    def test_brain_change_cards_use_standard_card_frame_and_text_badges(self) -> None:
        parent = self.make_record(1, None)
        record = self.make_record(
            2,
            1,
            neural_shifts=(NeuralShift(-17, 8, "added", None, -0.8),),
        )
        input_keys = tuple(range(-1, -39, -1))
        output_keys = tuple(range(12))
        report = generate_inspector_report(
            record,
            parent,
            None,
            self.renderer.config,
            output_keys,
            input_keys,
        )
        view = self.renderer._build_brain_changes_view(report, input_keys, output_keys)
        bounds = arcade.LBWH(100.0, 100.0, 340.0, 500.0)

        with patch.object(self.renderer, "_draw_rounded_rect") as rounded:
            self.renderer._draw_brain_changes_section(
                bounds,
                0,
                view,
                bounds.top,
                bounds.width,
            )

        self.assertTrue(
            any(
                call.args[1:5]
                == (
                    self.renderer.theme.card_background,
                    self.renderer.theme.panel_border,
                    self.renderer.config.layout.card_radius,
                    1.0,
                )
                for call in rounded.call_args_list
            )
        )
        texts = {text.text for text in self.renderer._text_cache.values()}
        self.assertIn("ADDED", texts)
        self.assertIn("Negative influence added", texts)
        self.assertIn("None → -0.80", texts)

    def test_brain_change_card_height_matches_minimum_width_rendering(self) -> None:
        parent = self.make_record(1, None)
        record = self.make_record(
            2,
            1,
            neural_shifts=(
                NeuralShift(-30, 10, "changed", -0.65, 0.75, 1.4),
                NeuralShift(-30, 13, "removed", -0.8, None),
            ),
        )
        input_keys = tuple(range(-1, -44, -1))
        output_keys = tuple(range(15))
        report = generate_inspector_report(
            record,
            parent,
            None,
            self.renderer.config,
            output_keys,
            input_keys,
        )
        view = self.renderer._build_brain_changes_view(
            report,
            input_keys,
            output_keys,
        )
        width = 256.0
        bounds = arcade.LBWH(100.0, 100.0, width, 900.0)

        consumed = self.renderer._draw_brain_changes_section(
            bounds,
            0,
            view,
            bounds.top,
            width,
        )

        self.assertEqual(
            consumed,
            self.renderer._brain_changes_section_height(view, width),
        )
        self.assertFalse(
            any(
                "..." in text.text
                for key, text in self.renderer._text_cache.items()
                if key.startswith("species_tree_brain_change_")
            )
        )
        expected_body_left = bounds.left + 94.0
        self.assertEqual(
            self.renderer._text_cache[
                "species_tree_brain_change_0_0_row_0_delta"
            ].x,
            expected_body_left,
        )
        self.assertEqual(
            self.renderer._text_cache[
                "species_tree_brain_change_0_0_row_0_movement"
            ].x,
            expected_body_left,
        )
        self.assertEqual(
            self.renderer._connection_change_badge_color("removed"),
            (108, 117, 125),
        )

    def test_brain_change_section_has_root_and_empty_states(self) -> None:
        root_report = generate_inspector_report(
            self.make_record(1, None),
            None,
            None,
            self.renderer.config,
            tuple(range(12)),
            tuple(range(-1, -39, -1)),
        )
        root_view = self.renderer._build_brain_changes_view(
            root_report,
            tuple(range(-1, -39, -1)),
            tuple(range(12)),
        )
        self.assertEqual(
            self.renderer._brain_change_intro_lines(root_view, 300.0),
            ("Founder species has no parent comparison.",),
        )

        empty_report = generate_inspector_report(
            self.make_record(2, 1),
            self.make_record(1, None),
            None,
            self.renderer.config,
            tuple(range(12)),
            tuple(range(-1, -39, -1)),
        )
        empty_view = self.renderer._build_brain_changes_view(
            empty_report,
            tuple(range(-1, -39, -1)),
            tuple(range(12)),
        )
        self.assertTrue(
            any(
                "No direct input-to-output" in line
                for line in self.renderer._brain_change_intro_lines(empty_view, 300.0)
            )
        )

    def test_species_inspector_marker_uses_legacy_color_fallback(self) -> None:
        record = replace(
            self.make_record(7, None),
            founder_color=None,
        )
        bounds = arcade.LBWH(100.0, 100.0, 340.0, 600.0)

        with patch("src.ui.renderer.arcade.draw_circle_filled") as filled:
            self.renderer._draw_species_inspector(bounds, None, record)

        self.assertIn(
            (
                bounds.left + 24.0,
                bounds.top - 84.0,
                10.0,
                self.renderer.theme.herbivore_fill,
            ),
            [call.args for call in filled.call_args_list],
        )

    def test_species_inspector_marks_reconstructed_data_quality(self) -> None:
        record = self.make_record(7, None, exact=False)
        bounds = arcade.LBWH(100.0, 100.0, 340.0, 600.0)

        self.renderer._draw_species_inspector(bounds, None, record)

        self.assertEqual(
            self.renderer._text_cache["species_tree_inspector_quality"].text,
            "Data: Reconstructed",
        )
        self.assertEqual(
            self.renderer._text_cache["species_tree_inspector_title"].text,
            "Species 7",
        )

    def test_species_inspector_marker_stays_fixed_while_scrolling(self) -> None:
        record = self.make_record(2, 1)
        bounds = arcade.LBWH(100.0, 100.0, 340.0, 220.0)
        marker_positions: list[tuple[float, float]] = []

        with patch("src.ui.renderer.arcade.draw_circle_filled") as filled:
            self.renderer._scroll_offsets["species_tree_inspector"] = 0.0
            self.renderer._draw_species_inspector(bounds, None, record)
            marker = next(
                call.args
                for call in filled.call_args_list
                if call.args[2:] == (10.0, record.founder_color)
            )
            marker_positions.append(marker[:2])
            filled.reset_mock()
            self.renderer._scroll_offsets["species_tree_inspector"] = 100.0
            self.renderer._draw_species_inspector(bounds, None, record)
            marker = next(
                call.args
                for call in filled.call_args_list
                if call.args[2:] == (10.0, record.founder_color)
            )
            marker_positions.append(marker[:2])

        self.assertEqual(marker_positions[0], marker_positions[1])

    def test_species_inspector_defaults_wider_than_legacy_cap(self) -> None:
        records = {
            1: self.make_record(1, None),
            2: self.make_record(2, 1),
        }
        world = self.make_world(records, width=1440, height=900)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        node = self.renderer._species_tree_node_bounds[2]
        self.renderer.handle_mouse_press(world, node.center_x, node.center_y)
        self.renderer.handle_mouse_release()

        self.renderer._draw_species_tree_window(world)

        content = self.renderer._control_hitboxes["species_tree_body"]
        resize = self.renderer._control_hitboxes["species_tree_inspector_resize"]
        inspector_left = resize.left + 5.0
        inspector_width = content.right - inspector_left
        self.assertGreater(inspector_width, 380.0)

    def test_species_inspector_resize_clamps_to_two_thirds_width(self) -> None:
        records = {
            1: self.make_record(1, None),
            2: self.make_record(2, 1),
        }
        world = self.make_world(records, width=1440, height=900)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        node = self.renderer._species_tree_node_bounds[2]
        self.renderer.handle_mouse_press(world, node.center_x, node.center_y)
        self.renderer.handle_mouse_release()
        self.renderer._draw_species_tree_window(world)
        content = self.renderer._control_hitboxes["species_tree_body"]
        resize = self.renderer._control_hitboxes["species_tree_inspector_resize"]

        self.assertTrue(
            self.renderer.handle_mouse_press(
                world,
                resize.center_x,
                resize.center_y,
            )
        )
        self.assertTrue(
            self.renderer.handle_mouse_drag(
                world,
                content.left,
                resize.center_y,
            )
        )
        self.renderer.handle_mouse_release()
        self.renderer._draw_species_tree_window(world)

        self.assertAlmostEqual(
            self.renderer._species_tree_inspector_width or 0.0,
            content.width * 2.0 / 3.0,
        )

    def test_species_inspector_resize_drag_does_not_select_or_pan(self) -> None:
        records = {
            1: self.make_record(1, None),
            2: self.make_record(2, 1),
        }
        world = self.make_world(records, width=1440, height=900)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        node = self.renderer._species_tree_node_bounds[2]
        self.renderer.handle_mouse_press(world, node.center_x, node.center_y)
        self.renderer.handle_mouse_release()
        self.renderer._draw_species_tree_window(world)
        resize = self.renderer._control_hitboxes["species_tree_inspector_resize"]
        selected = self.renderer._species_tree_selected_id
        horizontal_offset = self.renderer._species_tree_horizontal_offset
        vertical_offset = self.renderer._species_tree_vertical_offset

        self.renderer.handle_mouse_press(world, resize.center_x, resize.center_y)
        self.renderer.handle_mouse_drag(
            world,
            resize.center_x - 80.0,
            resize.center_y,
        )
        self.renderer.handle_mouse_release()

        self.assertEqual(self.renderer._species_tree_selected_id, selected)
        self.assertEqual(
            self.renderer._species_tree_horizontal_offset,
            horizontal_offset,
        )
        self.assertEqual(
            self.renderer._species_tree_vertical_offset,
            vertical_offset,
        )
        self.assertFalse(self.renderer._species_tree_canvas_drag)

    def test_tooltip_is_compact_and_uses_parent_percentages(self) -> None:
        parent = self.make_record(1, None)
        record = self.make_record(2, 1)

        tooltip = self.renderer._species_tree_tooltip_lines(record, parent)

        self.assertLessEqual(len(tooltip), 5)
        self.assertIn("Species ID: 2", tooltip)
        self.assertTrue(any("% Radius" in line for line in tooltip))
        self.assertNotIn("NEAT CHANGES FROM PARENT", " ".join(tooltip))

    def test_species_tree_builds_neat_labels_from_active_config(self) -> None:
        world = SimpleNamespace(
            neat_controller=SimpleNamespace(
                config=SimpleNamespace(
                    genome_config=SimpleNamespace(
                        input_keys=list(range(-1, -47, -1)),
                        output_keys=list(range(16)),
                    )
                )
            )
        )

        labels = self.renderer._species_tree_neat_node_labels(world)

        self.assertEqual(
            [labels[key] for key in range(-1, -47, -1)],
            list(SENSOR_INPUT_NAMES),
        )
        self.assertEqual(labels[-11], "food_proximity")
        self.assertEqual(labels[-24], "flock_effective_count")
        self.assertEqual(labels[-32], "stomach_fullness")
        self.assertEqual(labels[-33], "sound_strength")
        self.assertEqual(labels[-46], "life_normalized")
        self.assertEqual(self.renderer._short_brain_label(labels[-24]), "flock_n")
        self.assertEqual(labels[0], "accelerate")
        self.assertEqual(labels[3], "want_eat")
        self.assertEqual(labels[10], "emit_sound")

    def test_species_tree_reuses_and_invalidates_neat_label_cache(self) -> None:
        genome_config = SimpleNamespace(
            input_keys=[-1],
            output_keys=[0],
        )
        world = SimpleNamespace(
            neat_controller=SimpleNamespace(
                config=SimpleNamespace(genome_config=genome_config)
            )
        )

        first = self.renderer._species_tree_neat_node_labels(world)
        second = self.renderer._species_tree_neat_node_labels(world)
        genome_config.output_keys = [10]
        changed = self.renderer._species_tree_neat_node_labels(world)

        self.assertIs(first, second)
        self.assertIsNot(first, changed)
        self.assertEqual(changed[10], "accelerate")

    def test_species_tree_missing_config_reuses_empty_neat_labels(self) -> None:
        world = SimpleNamespace()

        first = self.renderer._species_tree_neat_node_labels(world)
        second = self.renderer._species_tree_neat_node_labels(world)

        self.assertIs(first, second)
        self.assertEqual(first, {})

    def test_species_tree_formats_named_neat_nodes_in_every_change_kind(
        self,
    ) -> None:
        labels = {-11: "food_proximity", 0: "accelerate", 3: "want_eat"}
        cases = {
            "Node 0 added": "Node accelerate added",
            "Node 0 removed": "Node accelerate removed",
            "Node 0 bias +0.100 -> +0.200": (
                "Node accelerate bias +0.100 -> +0.200"
            ),
            "Node 0 activation sigmoid -> tanh": (
                "Node accelerate activation sigmoid -> tanh"
            ),
            "Connection -11->3 added": (
                "Connection food_proximity -> want_eat added"
            ),
            "Connection -11->3 removed": (
                "Connection food_proximity -> want_eat removed"
            ),
            "Connection -11->3 enabled": (
                "Connection food_proximity -> want_eat enabled"
            ),
            "Connection -11->3 disabled": (
                "Connection food_proximity -> want_eat disabled"
            ),
            "Weight -11->3 +0.100 -> +0.900": (
                "Weight food_proximity -> want_eat +0.100 -> +0.900"
            ),
        }

        for original, expected in cases.items():
            with self.subTest(original=original):
                self.assertEqual(
                    self.renderer._format_species_tree_neat_change(
                        original,
                        labels,
                    ),
                    expected,
                )

    def test_species_tree_neat_names_keep_hidden_ids_and_custom_text(self) -> None:
        labels = {0: "accelerate"}

        self.assertEqual(
            self.renderer._format_species_tree_neat_change(
                "Connection 42->0 added",
                labels,
            ),
            "Connection 42 -> accelerate added",
        )
        self.assertEqual(
            self.renderer._format_species_tree_neat_change(
                "A legacy custom description",
                labels,
            ),
            "A legacy custom description",
        )

    def test_wheel_scrolls_vertically_and_clamps(self) -> None:
        records = {1: self.make_record(1, None)}
        for species_id in range(2, 12):
            records[species_id] = self.make_record(species_id, species_id - 1)
        world = self.make_world(records, width=700, height=500)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        self.renderer._species_tree_fit_mode = False
        self.renderer._species_tree_zoom = 1.0
        self.renderer._draw_species_tree_window(world)
        canvas = self.renderer._control_hitboxes["species_tree_canvas"]

        self.assertGreater(self.renderer._species_tree_vertical_limit, 0.0)
        self.assertTrue(
            self.renderer.handle_mouse_scroll(
                canvas.center_x, canvas.center_y, -3
            )
        )
        self.assertGreater(self.renderer._species_tree_vertical_offset, 0.0)
        self.renderer.handle_mouse_scroll(
            canvas.center_x, canvas.center_y, 1000
        )
        self.assertEqual(self.renderer._species_tree_vertical_offset, 0.0)

    def test_many_branches_expose_horizontal_overflow(self) -> None:
        records = {1: self.make_record(1, None)}
        for species_id in range(2, 14):
            records[species_id] = self.make_record(
                species_id,
                1,
                emerged_at=120.0,
            )
        world = self.make_world(records, width=700, height=420)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        self.renderer._species_tree_fit_mode = False
        self.renderer._species_tree_zoom = 1.0
        self.renderer._draw_species_tree_window(world)

        self.assertGreater(self.renderer._species_tree_horizontal_limit, 0.0)
        self.assertIn(
            "species_tree_horizontal_thumb",
            self.renderer._control_hitboxes,
        )

    def test_horizontal_scrollbar_drag_and_resize_clamp_offset(self) -> None:
        records = {1: self.make_record(1, None)}
        for species_id in range(2, 12):
            records[species_id] = self.make_record(
                species_id,
                1,
                emerged_at=120.0,
            )
        world = self.make_world(records, width=700, height=500)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        self.renderer._species_tree_fit_mode = False
        self.renderer._species_tree_zoom = 1.0
        self.renderer._draw_species_tree_window(world)
        thumb = self.renderer._control_hitboxes["species_tree_horizontal_thumb"]
        track = self.renderer._control_hitboxes["species_tree_horizontal_track"]

        self.assertTrue(
            self.renderer.handle_mouse_press(
                world, thumb.center_x, thumb.center_y
            )
        )
        self.assertTrue(
            self.renderer.handle_mouse_drag(
                world, track.right - 1.0, track.center_y
            )
        )
        self.assertGreater(self.renderer._species_tree_horizontal_offset, 0.0)
        self.renderer.handle_mouse_release()

        world.layout.window = arcade.LBWH(0, 0, 2400, 900)
        self.renderer._draw_species_tree_window(world)
        self.assertEqual(self.renderer._species_tree_horizontal_limit, 0.0)
        self.assertEqual(self.renderer._species_tree_horizontal_offset, 0.0)

    def test_trackpad_horizontal_axis_scrolls_tree(self) -> None:
        records = {1: self.make_record(1, None)}
        for species_id in range(2, 10):
            records[species_id] = self.make_record(
                species_id,
                1,
                emerged_at=120.0,
            )
        world = self.make_world(records, width=700, height=500)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        self.renderer._species_tree_fit_mode = False
        self.renderer._species_tree_zoom = 1.0
        self.renderer._draw_species_tree_window(world)
        canvas = self.renderer._control_hitboxes["species_tree_canvas"]

        self.renderer.handle_mouse_scroll(
            canvas.center_x,
            canvas.center_y,
            0.0,
            -2.0,
        )

        self.assertGreater(self.renderer._species_tree_horizontal_offset, 0.0)

    def test_opening_focuses_latest_chunk_without_fit(self) -> None:
        records = {1: self.make_record(1, None)}
        for species_id in range(2, 16):
            records[species_id] = self.make_record(species_id, species_id - 1)
        world = self.make_world(records, width=700, height=500)

        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)

        canvas = self.renderer._control_hitboxes["species_tree_canvas"]
        self.assertFalse(self.renderer._species_tree_fit_mode)
        self.assertEqual(self.renderer._species_tree_zoom, 1.0)
        self.assertEqual(
            self.renderer._species_tree_vertical_offset,
            self.renderer._species_tree_vertical_offset_max,
        )
        self.assertLess(
            len(self.renderer._species_tree_node_bounds),
            len(records),
        )

    def test_opening_culls_offscreen_branched_nodes(self) -> None:
        records = {1: self.make_record(1, None)}
        for species_id in range(2, 18):
            records[species_id] = self.make_record(species_id, 1)
        world = self.make_world(records, width=720, height=420)

        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)

        self.assertLess(
            len(self.renderer._species_tree_node_bounds),
            len(records),
        )
        self.assertEqual(
            set(self.renderer._species_tree_node_bounds),
            set(self.renderer._species_tree_visible_slice.node_ids),
        )

    def test_zoom_controls_switch_to_manual_and_fit_again(self) -> None:
        records = {1: self.make_record(1, None)}
        for species_id in range(2, 10):
            records[species_id] = self.make_record(species_id, species_id - 1)
        world = self.make_world(records, width=700, height=500)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        initial_zoom = self.renderer._species_tree_zoom
        for key in (
            "species_tree_zoom_out",
            "species_tree_zoom_label",
            "species_tree_zoom_in",
            "species_tree_zoom_fit",
        ):
            self.assertIn(key, self.renderer._control_hitboxes)
        self.assertEqual(
            self.renderer._text_cache["species_tree_zoom_percentage"].text,
            f"{initial_zoom * 100.0:.0f}%",
        )
        self.assertTrue(self.renderer._icon_path("zoom_in").is_file())
        self.assertTrue(self.renderer._icon_path("zoom_out").is_file())
        self.assertIn("zoom_in", self.renderer._texture_cache)
        self.assertIn("zoom_out", self.renderer._texture_cache)
        self.assertNotIn(
            "button_species_tree_zoom_in",
            self.renderer._text_cache,
        )
        self.assertNotIn(
            "button_species_tree_zoom_out",
            self.renderer._text_cache,
        )
        plus = self.renderer._control_hitboxes["species_tree_zoom_in"]

        self.renderer.handle_mouse_press(world, plus.center_x, plus.center_y)

        self.assertFalse(self.renderer._species_tree_fit_mode)
        self.assertGreater(self.renderer._species_tree_zoom, initial_zoom)
        self.assertGreater(self.renderer._species_tree_vertical_limit, 0.0)
        fit = self.renderer._control_hitboxes["species_tree_zoom_fit"]
        self.renderer.handle_mouse_press(world, fit.center_x, fit.center_y)
        self.assertTrue(self.renderer._species_tree_fit_mode)
        self.assertLess(self.renderer._species_tree_zoom, initial_zoom)
        self.assertEqual(self.renderer._species_tree_horizontal_offset, 0.0)
        self.assertEqual(self.renderer._species_tree_vertical_offset, 0.0)

    def test_manual_zoom_preserves_content_at_canvas_center(self) -> None:
        records = {1: self.make_record(1, None)}
        for species_id in range(2, 12):
            records[species_id] = self.make_record(species_id, species_id - 1)
        world = self.make_world(records, width=700, height=500)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        layout = self.renderer._species_tree_last_layout
        canvas = self.renderer._species_tree_last_canvas
        assert layout is not None
        assert canvas is not None

        old_zoom = self.renderer._species_tree_zoom
        old_inset_x, old_inset_y = self.renderer._species_tree_content_insets(
            layout, canvas, old_zoom
        )
        old_content_center = (
            (
                canvas.center_x
                - canvas.left
                - old_inset_x
                + self.renderer._species_tree_horizontal_offset
            )
            / old_zoom,
            (
                canvas.top
                - old_inset_y
                + self.renderer._species_tree_vertical_offset
                - canvas.center_y
            )
            / old_zoom,
        )

        self.renderer._adjust_species_tree_zoom(
            self.renderer.SPECIES_TREE_ZOOM_FACTOR
        )

        new_zoom = self.renderer._species_tree_zoom
        new_inset_x, new_inset_y = self.renderer._species_tree_content_insets(
            layout, canvas, new_zoom
        )
        new_content_center = (
            (
                canvas.center_x
                - canvas.left
                - new_inset_x
                + self.renderer._species_tree_horizontal_offset
            )
            / new_zoom,
            (
                canvas.top
                - new_inset_y
                + self.renderer._species_tree_vertical_offset
                - canvas.center_y
            )
            / new_zoom,
        )
        self.assertAlmostEqual(new_content_center[0], old_content_center[0])
        self.assertAlmostEqual(new_content_center[1], old_content_center[1])

    def test_manual_zoom_limits_are_ten_to_two_hundred_percent(self) -> None:
        world = self.make_world({1: self.make_record(1, None)})
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)

        for _ in range(30):
            self.renderer._adjust_species_tree_zoom(
                self.renderer.SPECIES_TREE_ZOOM_FACTOR
            )
        self.assertEqual(
            self.renderer._species_tree_zoom,
            self.renderer.SPECIES_TREE_MAX_ZOOM,
        )
        for _ in range(60):
            self.renderer._adjust_species_tree_zoom(
                1.0 / self.renderer.SPECIES_TREE_ZOOM_FACTOR
            )
        self.assertEqual(
            self.renderer._species_tree_zoom,
            self.renderer.SPECIES_TREE_MIN_ZOOM,
        )

    def test_fit_recalculates_after_resize_and_history_change(self) -> None:
        records = {1: self.make_record(1, None)}
        for species_id in range(2, 10):
            records[species_id] = self.make_record(species_id, species_id - 1)
        world = self.make_world(records, width=700, height=500)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        self.renderer._activate_species_tree_fit()
        self.renderer._draw_species_tree_window(world)
        small_window_zoom = self.renderer._species_tree_zoom

        world.layout.window = arcade.LBWH(0, 0, 1600, 900)
        self.renderer._draw_species_tree_window(world)
        self.assertGreater(self.renderer._species_tree_zoom, small_window_zoom)

        for species_id in range(10, 18):
            records[species_id] = self.make_record(species_id, species_id - 1)
        self.renderer._draw_species_tree_window(world)
        self.assertLess(self.renderer._species_tree_zoom, 1.0)

    def test_command_wheel_zoom_preserves_content_beneath_cursor(self) -> None:
        records = {1: self.make_record(1, None)}
        for species_id in range(2, 12):
            records[species_id] = self.make_record(species_id, species_id - 1)
        world = self.make_world(records, width=700, height=500)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        layout = self.renderer._species_tree_last_layout
        canvas = self.renderer._species_tree_last_canvas
        assert layout is not None
        assert canvas is not None
        cursor_x = canvas.left + canvas.width * 0.72
        cursor_y = canvas.bottom + canvas.height * 0.38

        old_zoom = self.renderer._species_tree_zoom
        old_inset_x, old_inset_y = self.renderer._species_tree_content_insets(
            layout, canvas, old_zoom
        )
        old_content_point = (
            (
                cursor_x
                - canvas.left
                - old_inset_x
                + self.renderer._species_tree_horizontal_offset
            )
            / old_zoom,
            (
                canvas.top
                - old_inset_y
                + self.renderer._species_tree_vertical_offset
                - cursor_y
            )
            / old_zoom,
        )

        handled = self.renderer.handle_mouse_scroll(
            cursor_x,
            cursor_y,
            1.0,
            command_down=True,
        )

        self.assertTrue(handled)
        self.assertFalse(self.renderer._species_tree_fit_mode)
        self.assertGreater(self.renderer._species_tree_zoom, old_zoom)
        new_zoom = self.renderer._species_tree_zoom
        new_inset_x, new_inset_y = self.renderer._species_tree_content_insets(
            layout, canvas, new_zoom
        )
        new_content_point = (
            (
                cursor_x
                - canvas.left
                - new_inset_x
                + self.renderer._species_tree_horizontal_offset
            )
            / new_zoom,
            (
                canvas.top
                - new_inset_y
                + self.renderer._species_tree_vertical_offset
                - cursor_y
            )
            / new_zoom,
        )
        self.assertAlmostEqual(new_content_point[0], old_content_point[0])
        self.assertAlmostEqual(new_content_point[1], old_content_point[1])

        self.renderer.handle_mouse_scroll(
            cursor_x,
            cursor_y,
            -1.0,
            command_down=True,
        )
        self.assertAlmostEqual(self.renderer._species_tree_zoom, old_zoom)
        self.assertFalse(self.renderer._species_tree_fit_mode)

    def test_command_wheel_outside_canvas_is_consumed_without_zooming(self) -> None:
        world = self.make_world({1: self.make_record(1, None)})
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        canvas = self.renderer._control_hitboxes["species_tree_canvas"]
        old_zoom = self.renderer._species_tree_zoom

        handled = self.renderer.handle_mouse_scroll(
            canvas.center_x,
            canvas.top + 12.0,
            1.0,
            command_down=True,
        )

        self.assertTrue(handled)
        self.assertEqual(self.renderer._species_tree_zoom, old_zoom)
        self.assertFalse(self.renderer._species_tree_fit_mode)

    def test_timeline_formats_ticks_and_registers_only_timed_events(self) -> None:
        records = {
            1: self.make_record(1, None, emerged_at=0.0),
            2: self.make_record(2, 1, emerged_at=3661.0),
            3: self.make_record(3, 2, exact=False),
        }
        world = self.make_world(records)
        world.elapsed_time = 4000.0
        self.renderer.open_species_tree(world)

        self.renderer._draw_species_tree_window(world)

        self.assertEqual(
            self.renderer._format_species_tree_time(61.0),
            "01:01",
        )
        self.assertEqual(
            self.renderer._format_species_tree_time(3661.0),
            "1:01:01",
        )
        self.assertEqual(
            set(self.renderer._species_tree_timeline_bucket_bounds),
            {0, 2},
        )
        self.assertIn("species_tree_timeline", self.renderer._control_hitboxes)

    def test_timeline_bucket_jumps_to_its_time_range(self) -> None:
        records = {
            1: self.make_record(1, None, emerged_at=0.0),
            2: self.make_record(2, 1, emerged_at=300.0),
            3: self.make_record(3, 1, emerged_at=600.0),
        }
        world = self.make_world(records, width=700, height=420)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        marker = self.renderer._species_tree_timeline_bucket_bounds[0]

        handled = self.renderer.handle_mouse_press(
            world,
            marker.center_x,
            marker.center_y,
        )

        self.assertTrue(handled)
        self.assertFalse(self.renderer._species_tree_fit_mode)
        self.assertEqual(self.renderer._species_tree_zoom, 1.0)
        self.assertGreaterEqual(self.renderer._species_tree_vertical_offset, 0.0)

    def test_timeline_ruler_jump_preserves_horizontal_offset(self) -> None:
        records = {1: self.make_record(1, None, emerged_at=0.0)}
        for species_id in range(2, 12):
            records[species_id] = self.make_record(
                species_id,
                1,
                emerged_at=600.0,
            )
        world = self.make_world(records, width=700, height=420)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        self.renderer._species_tree_fit_mode = False
        self.renderer._species_tree_zoom = 1.0
        self.renderer._draw_species_tree_window(world)
        self.renderer._species_tree_horizontal_offset = 30.0
        timeline = self.renderer._control_hitboxes["species_tree_timeline"]

        self.renderer.handle_mouse_press(
            world,
            timeline.left + 5.0,
            timeline.bottom + 13.0,
        )

        self.assertAlmostEqual(
            self.renderer._species_tree_horizontal_offset,
            30.0,
        )
        self.assertGreater(self.renderer._species_tree_vertical_offset, 0.0)

    def test_canvas_drag_preserves_fitted_zoom_and_pans(self) -> None:
        records = {1: self.make_record(1, None, emerged_at=0.0)}
        for species_id in range(2, 12):
            records[species_id] = self.make_record(
                species_id,
                1,
                emerged_at=600.0,
            )
        world = self.make_world(records, width=700, height=420)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        canvas = self.renderer._control_hitboxes["species_tree_canvas"]
        self.assertFalse(self.renderer._species_tree_fit_mode)
        fitted_zoom = self.renderer._species_tree_zoom
        initial_horizontal_offset = self.renderer._species_tree_horizontal_offset

        self.renderer.handle_mouse_press(
            world,
            canvas.center_x,
            canvas.center_y,
        )
        self.renderer.handle_mouse_drag(
            world,
            canvas.center_x + 24.0,
            canvas.center_y + 18.0,
        )

        self.assertFalse(self.renderer._species_tree_fit_mode)
        self.assertEqual(self.renderer._species_tree_zoom, fitted_zoom)
        self.assertTrue(self.renderer._species_tree_canvas_drag_started)
        self.assertNotEqual(
            self.renderer._species_tree_horizontal_offset,
            initial_horizontal_offset,
        )
        self.renderer.handle_mouse_release()
        self.assertFalse(self.renderer._species_tree_canvas_drag)

    def test_canvas_drag_preserves_manual_zoom_and_pans_both_axes(self) -> None:
        records = {1: self.make_record(1, None, emerged_at=0.0)}
        for species_id in range(2, 12):
            records[species_id] = self.make_record(
                species_id,
                1,
                emerged_at=600.0,
            )
        world = self.make_world(records, width=700, height=420)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        self.renderer._species_tree_fit_mode = False
        self.renderer._species_tree_zoom = 1.25
        self.renderer._draw_species_tree_window(world)
        canvas = self.renderer._control_hitboxes["species_tree_canvas"]

        self.renderer.handle_mouse_press(
            world,
            canvas.center_x,
            canvas.center_y,
        )
        self.renderer.handle_mouse_drag(
            world,
            canvas.center_x - 20.0,
            canvas.center_y + 15.0,
        )

        self.assertEqual(self.renderer._species_tree_zoom, 1.25)
        self.assertGreater(self.renderer._species_tree_horizontal_offset, 0.0)
        self.assertGreater(self.renderer._species_tree_vertical_offset, 0.0)
        self.renderer.handle_mouse_release()
        self.assertFalse(self.renderer._species_tree_canvas_drag)

    def test_cached_five_thousand_node_modal_opens_under_fifty_ms(self) -> None:
        records = {
            species_id: self.make_record(
                species_id,
                None if species_id == 1 else species_id - 1,
                emerged_at=(species_id - 1) * (36000.0 / 4999.0),
            )
            for species_id in range(1, 5001)
        }
        world = self.make_world(records, width=1440, height=900)
        world.elapsed_time = 36000.0
        self.renderer._sync_species_tree_layout(world)
        placements = self.renderer._species_tree_layout_manager.placement_count

        started = perf_counter()
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        elapsed = perf_counter() - started

        self.assertLess(elapsed, 0.05)
        self.assertEqual(
            self.renderer._species_tree_layout_manager.placement_count,
            placements,
        )
        self.assertLess(len(self.renderer._species_tree_node_bounds), 500)
        self.assertLess(
            len(self.renderer._species_tree_visible_slice.routes),
            500,
        )
        self.assertLessEqual(
            len(self.renderer._species_tree_timeline_bucket_bounds),
            21,
        )


if __name__ == "__main__":
    unittest.main()
