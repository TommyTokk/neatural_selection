from __future__ import annotations

from types import SimpleNamespace
import unittest

import arcade

from configs.sim_config import build_sim_config
from src.behavior_history import (
    BehaviorEvidenceSummary,
    BehaviorLifetimeSummary,
    BehaviorLifetimeWhySummary,
    BehaviorOutcome,
    BehaviorTermination,
    CompletedBehaviorBout,
    CompletedSemanticEffect,
    CompletedWhyExplanation,
    CreatureBehaviorReport,
    CreatureBehaviorSummary,
    CreatureHistoryIndexEntry,
    EffectDirectionCounts,
)
from src.behavior_observer import BehaviorKind
from src.counterfactual_neat import (
    EffectDirection,
    InfluenceLabel,
    SemanticIntervention,
)
from src.ui.renderer import UiRenderer


class BehaviorReportUiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.renderer = UiRenderer(build_sim_config())

    @staticmethod
    def _index_entry(
        creature_id: int = 8,
        *,
        deceased: bool = False,
        bouts: int = 2,
    ) -> CreatureHistoryIndexEntry:
        return CreatureHistoryIndexEntry(
            creature_id=creature_id,
            creature_name=f"Herbivore {creature_id}",
            deceased=deceased,
            last_observed_time=12.0,
            completed_bout_count=bouts,
        )

    @staticmethod
    def _empty_report(creature_id: int = 8) -> CreatureBehaviorReport:
        return CreatureBehaviorReport(
            creature_id=creature_id,
            creature_name=f"Herbivore {creature_id}",
            deceased=False,
            completed_bouts=(),
            summary=CreatureBehaviorSummary(
                creature_id=creature_id,
                completed_bout_count=0,
                behaviors=(),
                stable_pattern_threshold=3,
            ),
            history_incomplete=False,
            history_completions_not_recorded=0,
            detailed_bouts_dropped=0,
        )

    def test_left_rail_removes_the_report_action(self) -> None:
        self.renderer._control_hitboxes["open_behavior_report"] = arcade.LBWH(
            0,
            0,
            20,
            20,
        )
        world = SimpleNamespace(
            layout=SimpleNamespace(
                left_sidebar=arcade.LBWH(20, 100, 92, 324),
            ),
            show_biome_background=False,
            save_in_progress=False,
        )
        original_icon_button = self.renderer._draw_icon_button
        self.renderer._draw_icon_button = lambda *_args, **_kwargs: None
        try:
            self.renderer._draw_icon_rail(world)
        finally:
            self.renderer._draw_icon_button = original_icon_button

        self.assertNotIn("open_behavior_report", self.renderer._control_hitboxes)
        for key in (
            "panel_toggle_inspector",
            "panel_toggle_stats",
            "panel_toggle_settings",
            "open_map_submenu",
            "save_simulation",
            "open_species_tree",
        ):
            self.assertIn(key, self.renderer._control_hitboxes)

    def test_stats_card_has_fixed_report_launcher_without_scroll(self) -> None:
        world = SimpleNamespace(
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, 1440, 900),
            ),
            stats=SimpleNamespace(
                herbivore_count=8,
                food_count=20,
                biome_food_counts={},
                available_biomass=42.0,
                plant_spawn_pressure=0.25,
            ),
            config=SimpleNamespace(
                population=SimpleNamespace(max_creatures=64),
            ),
            rt_neat=SimpleNamespace(
                stats=SimpleNamespace(
                    births=3,
                    deaths=1,
                    best_fitness=4.0,
                    average_fitness=2.0,
                    worst_fitness=1.0,
                    average_speed=12.0,
                )
            ),
            archived_fitness_count=lambda: 5,
            live_brain_count=lambda: 8,
            simulation_speed=1.0,
            is_paused=False,
        )
        original_icon = self.renderer._draw_icon
        self.renderer._draw_icon = lambda *_args, **_kwargs: None
        try:
            self.renderer._draw_stats_panel(world)
        finally:
            self.renderer._draw_icon = original_icon

        panel = self.renderer._control_hitboxes["stats_panel"]
        action = self.renderer._control_hitboxes["open_behavior_report"]
        self.assertEqual(panel.width, 400.0)
        self.assertEqual(panel.height, 600.0)
        self.assertEqual(self.renderer._scroll_limits["stats"], 0.0)
        self.assertGreaterEqual(action.left, panel.left)
        self.assertLessEqual(action.right, panel.right)
        self.assertGreaterEqual(action.bottom, panel.bottom)

    def test_report_uses_large_window_safe_bounds_and_no_icons(self) -> None:
        report = self._empty_report()
        entry = self._index_entry()
        world = SimpleNamespace(
            layout=SimpleNamespace(window=arcade.LBWH(0, 0, 1440, 900)),
            behavior_history_index=(entry,),
            selected_creature_id=8,
            behavior_report_for=lambda _creature_id: report,
        )
        self.renderer._behavior_report_open = True
        icon_keys: list[str] = []
        original_icon = self.renderer._draw_icon
        self.renderer._draw_icon = (
            lambda _bounds, _name, key: icon_keys.append(key)
        )
        try:
            self.renderer._draw_behavior_report_window(world)
        finally:
            self.renderer._draw_icon = original_icon

        bounds = self.renderer._behavior_report_bounds
        self.assertIsNotNone(bounds)
        self.assertEqual(bounds.width, 1320.0)
        self.assertEqual(bounds.height, 840.0)
        self.assertEqual(icon_keys, ["behavior_report_close_icon"])
        self.assertEqual(
            self.renderer._text_cache["behavior_report_subtitle"].text,
            "Inspect completed behaviour history for observed creatures",
        )

        world.layout.window = arcade.LBWH(0, 0, 640, 480)
        self.renderer._draw_behavior_report_window(world)
        bounds = self.renderer._behavior_report_bounds
        self.assertLessEqual(bounds.right, 640 - 24)
        self.assertLessEqual(bounds.top, 480 - 24)

    def test_creature_rows_are_spaced_and_scroll_only_on_overflow(self) -> None:
        entries = tuple(self._index_entry(index) for index in range(10))
        world = SimpleNamespace(
            behavior_history_index=entries,
            selected_creature_id=0,
        )
        bounds = arcade.LBWH(0, 0, 270, 700)

        self.renderer._draw_report_creature_index(world, bounds)

        first = self.renderer._control_hitboxes["behavior_report_creature_0"]
        second = self.renderer._control_hitboxes["behavior_report_creature_1"]
        self.assertEqual(first.height, 50.0)
        self.assertEqual(first.bottom - second.top, 4.0)
        self.assertEqual(
            self.renderer._scroll_limits["behavior_report_creatures"],
            0.0,
        )

        world.behavior_history_index = tuple(
            self._index_entry(index) for index in range(16)
        )
        self.renderer._draw_report_creature_index(world, bounds)
        self.assertGreater(
            self.renderer._scroll_limits["behavior_report_creatures"],
            0.0,
        )

    def test_creature_sidebar_scroll_is_independent_from_page_scroll(self) -> None:
        sidebar = arcade.LBWH(0, 0, 270, 300)
        page = arcade.LBWH(300, 0, 600, 300)
        self.renderer._behavior_report_open = True
        self.renderer._scroll_regions["behavior_report_creatures"] = sidebar
        self.renderer._scroll_regions["behavior_report"] = page
        self.renderer._scroll_limits["behavior_report_creatures"] = 200.0
        self.renderer._scroll_limits["behavior_report"] = 300.0
        self.renderer._scroll_offsets["behavior_report_creatures"] = 0.0
        self.renderer._scroll_offsets["behavior_report"] = 40.0

        handled = self.renderer.handle_mouse_scroll(
            sidebar.center_x,
            sidebar.center_y,
            -2.0,
        )

        self.assertTrue(handled)
        self.assertEqual(
            self.renderer._scroll_offsets["behavior_report_creatures"],
            48.0,
        )
        self.assertEqual(
            self.renderer._scroll_offsets["behavior_report"],
            40.0,
        )

    def test_global_action_opens_most_recent_retained_creature(self) -> None:
        entry = CreatureHistoryIndexEntry(
            creature_id=8,
            creature_name="Eight",
            deceased=True,
            last_observed_time=12.0,
            completed_bout_count=4,
        )
        world = SimpleNamespace(
            selected_creature_id=None,
            behavior_history_index=(entry,),
        )
        self.renderer._control_hitboxes["open_behavior_report"] = arcade.LBWH(
            0,
            0,
            40,
            20,
        )

        handled = self.renderer.handle_mouse_press(world, 10, 10)

        self.assertTrue(handled)
        self.assertTrue(self.renderer._behavior_report_open)
        self.assertEqual(self.renderer._behavior_report_creature_id, 8)

    def test_selected_report_action_does_not_replace_global_rail_hitbox(
        self,
    ) -> None:
        entry = CreatureHistoryIndexEntry(
            creature_id=8,
            creature_name="Eight",
            deceased=False,
            last_observed_time=12.0,
            completed_bout_count=1,
        )
        world = SimpleNamespace(
            selected_creature_id=8,
            behavior_history_index=(entry,),
        )
        global_bounds = arcade.LBWH(0, 0, 20, 20)
        selected_bounds = arcade.LBWH(100, 0, 40, 20)
        self.renderer._control_hitboxes["open_behavior_report"] = (
            global_bounds
        )
        self.renderer._control_hitboxes[
            "open_behavior_report_selected"
        ] = selected_bounds

        handled = self.renderer.handle_mouse_press(world, 110, 10)

        self.assertTrue(handled)
        self.assertTrue(self.renderer._behavior_report_open)
        self.assertIs(
            self.renderer._control_hitboxes["open_behavior_report"],
            global_bounds,
        )

    def test_incomplete_history_warning_is_visible(self) -> None:
        report = CreatureBehaviorReport(
            creature_id=8,
            creature_name="Eight",
            deceased=True,
            completed_bouts=(),
            summary=CreatureBehaviorSummary(
                creature_id=8,
                completed_bout_count=0,
                behaviors=(),
                stable_pattern_threshold=3,
            ),
            history_incomplete=True,
            history_completions_not_recorded=5,
            detailed_bouts_dropped=0,
        )
        captured: list[str] = []
        original_text = self.renderer._draw_text
        original_rounded = self.renderer._draw_rounded_rect
        self.renderer._draw_text = (
            lambda _key, text, *_args, **_kwargs: captured.append(text)
        )
        self.renderer._draw_rounded_rect = (
            lambda *_args, **_kwargs: None
        )
        try:
            self.renderer._draw_report_content(
                report,
                arcade.LBWH(0, 0, 700, 500),
            )
        finally:
            self.renderer._draw_text = original_text
            self.renderer._draw_rounded_rect = original_rounded

        self.assertTrue(
            any("History incomplete" in text for text in captured)
        )
        self.assertTrue(
            any("5 completed bouts" in text for text in captured)
        )

    @staticmethod
    def _why_report(
        completed_bout_count: int = 2,
        effect_count: int = 1,
    ) -> CreatureBehaviorReport:
        interventions = tuple(SemanticIntervention)
        why = tuple(
            BehaviorLifetimeWhySummary(
                intervention=interventions[index % len(interventions)],
                behavior_bout_count=completed_bout_count,
                contributing_bout_count=completed_bout_count,
                median_bout_influence=0.66,
                p25=0.5,
                p75=0.8,
                influence_label=InfluenceLabel.STRONG,
                direction_counts=EffectDirectionCounts(
                    suppressive=1,
                    mixed=max(0, completed_bout_count - 1),
                ),
            )
            for index in range(effect_count)
        )
        behavior = BehaviorLifetimeSummary(
            behavior=BehaviorKind.FOOD_APPROACH,
            completed_bout_count=completed_bout_count,
            total_duration=4.0,
            median_duration=2.0,
            outcome_counts=(),
            why_summaries=why,
        )
        return CreatureBehaviorReport(
            creature_id=8,
            creature_name="Eight",
            deceased=False,
            completed_bouts=(),
            summary=CreatureBehaviorSummary(
                creature_id=8,
                completed_bout_count=completed_bout_count,
                behaviors=(behavior,),
                stable_pattern_threshold=3,
            ),
            history_incomplete=False,
            history_completions_not_recorded=0,
            detailed_bouts_dropped=0,
        )

    def test_why_uses_threshold_wording_and_all_direction_counts(self) -> None:
        captured: list[str] = []
        original_text = self.renderer._draw_text
        self.renderer._draw_text = (
            lambda _key, text, *_args, **_kwargs: captured.append(text)
        )
        try:
            report = self._why_report()
            self.renderer._draw_report_why(
                report,
                arcade.LBWH(0, 0, 700, 500),
            )
        finally:
            self.renderer._draw_text = original_text

        self.assertTrue(any("insufficient history" in text for text in captured))
        self.assertFalse(any("Typical" in text for text in captured))
        self.assertTrue(
            any("Median bout influence: 0.66 · STRONG" in text for text in captured)
        )
        self.assertTrue(
            any("WHY available in 2 / 2 bouts" in text for text in captured)
        )
        self.assertTrue(
            any("SUPPRESSIVE in 1 / 2 bouts" in text for text in captured)
        )
        self.assertTrue(
            any("MIXED in 1 / 2 bouts" in text for text in captured)
        )

        captured.clear()
        stable = self._why_report(completed_bout_count=3)
        self.renderer._draw_text = (
            lambda _key, text, *_args, **_kwargs: captured.append(text)
        )
        try:
            self.renderer._draw_report_why(
                stable,
                arcade.LBWH(0, 0, 700, 500),
            )
        finally:
            self.renderer._draw_text = original_text
        self.assertTrue(any("Typical" in text for text in captured))

    def test_why_page_scroll_has_a_real_bounded_offset(self) -> None:
        report = self._why_report(effect_count=18)
        bounds = arcade.LBWH(0, 0, 500, 180)
        self.renderer._draw_text = lambda *_args, **_kwargs: None
        self.renderer._draw_scrollbar = lambda *_args, **_kwargs: None
        self.renderer._draw_report_why(report, bounds)
        self.renderer._scroll_regions["behavior_report"] = bounds
        limit = self.renderer._scroll_limits["behavior_report"]
        self.assertGreater(limit, 0.0)

        self.renderer._behavior_report_open = True
        handled = self.renderer.handle_mouse_scroll(
            bounds.center_x,
            bounds.center_y,
            -100.0,
        )

        self.assertTrue(handled)
        self.assertEqual(
            self.renderer._scroll_offsets["behavior_report"],
            limit,
        )

    def test_standard_why_card_grid_does_not_scroll(self) -> None:
        report = self._why_report(effect_count=3)

        self.renderer._draw_report_why(
            report,
            arcade.LBWH(0, 0, 980, 560),
        )

        self.assertEqual(self.renderer._scroll_limits["behavior_report"], 0.0)
        self.assertEqual(
            self.renderer._behavior_report_why_behavior,
            BehaviorKind.FOOD_APPROACH.value,
        )
        self.assertIn(
            "behavior_report_why_behavior_food_approach",
            self.renderer._control_hitboxes,
        )

    def test_why_behavior_selector_updates_ui_only_selection(self) -> None:
        self.renderer._behavior_report_open = True
        self.renderer._behavior_report_bounds = arcade.LBWH(0, 0, 900, 700)
        target = arcade.LBWH(100, 100, 120, 36)
        self.renderer._control_hitboxes[
            "behavior_report_why_behavior_feeding"
        ] = target
        world = SimpleNamespace(
            behavior_history_index=(),
        )

        handled = self.renderer.handle_mouse_press(
            world,
            target.center_x,
            target.center_y,
        )

        self.assertTrue(handled)
        self.assertEqual(
            self.renderer._behavior_report_why_behavior,
            BehaviorKind.FEEDING.value,
        )

    def test_summary_cards_preserve_supplied_aggregate_values(self) -> None:
        behaviors = tuple(
            BehaviorLifetimeSummary(
                behavior=behavior,
                completed_bout_count=index + 1,
                total_duration=10.0 + index,
                median_duration=2.0 + index,
                outcome_counts=(),
                why_summaries=(),
            )
            for index, behavior in enumerate(BehaviorKind)
        )
        report = CreatureBehaviorReport(
            creature_id=8,
            creature_name="Eight",
            deceased=False,
            completed_bouts=(),
            summary=CreatureBehaviorSummary(
                creature_id=8,
                completed_bout_count=21,
                behaviors=behaviors,
                stable_pattern_threshold=3,
            ),
            history_incomplete=False,
            history_completions_not_recorded=0,
            detailed_bouts_dropped=0,
        )

        self.renderer._draw_report_summary(
            report,
            arcade.LBWH(0, 0, 980, 560),
        )

        self.assertEqual(self.renderer._scroll_limits["behavior_report"], 0.0)
        feeding = BehaviorKind.FEEDING.value
        self.assertEqual(
            self.renderer._text_cache[
                f"behavior_report_summary_count_{feeding}_value"
            ].text,
            "3",
        )
        self.assertEqual(
            self.renderer._text_cache[
                f"behavior_report_summary_total_{feeding}_value"
            ].text,
            "12.0s",
        )
        self.assertEqual(
            self.renderer._text_cache[
                f"behavior_report_summary_median_{feeding}_value"
            ].text,
            "4.0s",
        )

    def test_summary_card_uses_larger_title_and_measured_text_height(
        self,
    ) -> None:
        short = BehaviorLifetimeSummary(
            behavior=BehaviorKind.FEEDING,
            completed_bout_count=2,
            total_duration=3.0,
            median_duration=1.5,
            outcome_counts=(),
            why_summaries=(),
        )
        long = BehaviorLifetimeSummary(
            behavior=BehaviorKind.FOOD_ORIENTATION,
            completed_bout_count=8,
            total_duration=12.0,
            median_duration=1.5,
            outcome_counts=tuple((outcome, 1) for outcome in BehaviorOutcome),
            why_summaries=(),
        )
        card = arcade.LBWH(20, 30, 190, 260)

        self.renderer._draw_summary_card(short, card)

        title = self.renderer._text_cache["behavior_report_summary_feeding"]
        outcome = self.renderer._text_cache["behavior_report_outcomes_feeding"]
        self.assertEqual(title.font_size, 12.0)
        self.assertEqual(title.x, round(card.left + 30.0))
        self.assertEqual(title.y, round(card.top - 14.0))
        self.assertEqual(title.anchor_y, "top")
        self.assertEqual(outcome.x, round(card.left + 16.0))
        self.assertEqual(outcome.anchor_y, "top")
        self.assertGreater(
            self.renderer._summary_card_height(long, 190.0),
            self.renderer._summary_card_height(short, 190.0),
        )
        self.assertGreater(
            self.renderer._summary_card_height(long, 190.0),
            self.renderer._summary_card_height(long, 360.0),
        )

    def test_summary_grid_sizes_each_row_from_its_tallest_card(self) -> None:
        behaviors = (
            BehaviorLifetimeSummary(
                behavior=BehaviorKind.FOOD_ORIENTATION,
                completed_bout_count=8,
                total_duration=12.0,
                median_duration=1.5,
                outcome_counts=tuple(
                    (outcome, 1) for outcome in BehaviorOutcome
                ),
                why_summaries=(),
            ),
            BehaviorLifetimeSummary(
                behavior=BehaviorKind.FOOD_APPROACH,
                completed_bout_count=1,
                total_duration=1.0,
                median_duration=1.0,
                outcome_counts=(),
                why_summaries=(),
            ),
            BehaviorLifetimeSummary(
                behavior=BehaviorKind.FEEDING,
                completed_bout_count=1,
                total_duration=1.0,
                median_duration=1.0,
                outcome_counts=(),
                why_summaries=(),
            ),
            BehaviorLifetimeSummary(
                behavior=BehaviorKind.RESTING,
                completed_bout_count=1,
                total_duration=1.0,
                median_duration=1.0,
                outcome_counts=(),
                why_summaries=(),
            ),
        )
        report = CreatureBehaviorReport(
            creature_id=8,
            creature_name="Eight",
            deceased=False,
            completed_bouts=(),
            summary=CreatureBehaviorSummary(
                creature_id=8,
                completed_bout_count=11,
                behaviors=behaviors,
                stable_pattern_threshold=3,
            ),
            history_incomplete=False,
            history_completions_not_recorded=0,
            detailed_bouts_dropped=0,
        )
        cards: list[arcade.Rect] = []
        original_card = self.renderer._draw_summary_card
        self.renderer._draw_summary_card = (
            lambda _item, bounds: cards.append(bounds)
        )
        try:
            self.renderer._draw_report_summary(
                report,
                arcade.LBWH(0, 0, 700, 600),
            )
        finally:
            self.renderer._draw_summary_card = original_card

        self.assertEqual(len(cards), 4)
        self.assertEqual(cards[0].height, cards[1].height)
        self.assertEqual(cards[2].height, cards[3].height)
        self.assertGreater(cards[0].height, cards[2].height)
        self.assertEqual(cards[0].bottom - cards[2].top, 12.0)

    def test_why_card_wraps_complete_text_and_expands_for_narrow_width(
        self,
    ) -> None:
        report = self._why_report(completed_bout_count=5)
        behavior = report.summary.behaviors[0]
        effect = BehaviorLifetimeWhySummary(
            intervention=SemanticIntervention.RESOURCE_GRADIENT_CUES,
            behavior_bout_count=5,
            contributing_bout_count=5,
            median_bout_influence=0.66,
            p25=0.5,
            p75=0.8,
            influence_label=InfluenceLabel.STRONG,
            direction_counts=EffectDirectionCounts(
                supportive=1,
                suppressive=1,
                reversing=1,
                mixed=1,
                minimal=1,
            ),
        )
        narrow_height = self.renderer._why_card_height(report, effect, 180.0)
        wide_height = self.renderer._why_card_height(report, effect, 360.0)
        card = arcade.LBWH(10, 20, 180, narrow_height)

        self.renderer._draw_why_card(report, behavior, effect, card)

        prefix = "food_approach_resource_gradient_cues"
        title = self.renderer._text_cache[
            f"behavior_report_why_intervention_{prefix}"
        ]
        pattern = self.renderer._text_cache[
            f"behavior_report_why_pattern_{prefix}"
        ]
        self.assertEqual(title.text, "Resource Gradient Cues")
        self.assertNotIn("...", title.text)
        self.assertEqual(title.font_size, 12.0)
        self.assertTrue(title.multiline)
        self.assertEqual(title.anchor_y, "top")
        self.assertEqual(title.x, round(card.left + 30.0))
        self.assertEqual(pattern.x, round(card.left + 16.0))
        self.assertTrue(pattern.multiline)
        self.assertGreater(narrow_height, wide_height)

    def test_why_grid_sizes_rows_independently_without_overlap(self) -> None:
        long_effect = BehaviorLifetimeWhySummary(
            intervention=SemanticIntervention.RESOURCE_GRADIENT_CUES,
            behavior_bout_count=5,
            contributing_bout_count=5,
            median_bout_influence=0.66,
            p25=0.5,
            p75=0.8,
            influence_label=InfluenceLabel.STRONG,
            direction_counts=EffectDirectionCounts(
                supportive=1,
                suppressive=1,
                reversing=1,
                mixed=1,
                minimal=1,
            ),
        )
        short_effects = tuple(
            BehaviorLifetimeWhySummary(
                intervention=intervention,
                behavior_bout_count=5,
                contributing_bout_count=5,
                median_bout_influence=0.1,
                p25=None,
                p75=None,
                influence_label=InfluenceLabel.MINIMAL,
                direction_counts=EffectDirectionCounts(),
            )
            for intervention in (
                SemanticIntervention.SATIATED_STATE,
                SemanticIntervention.ALARM_PHEROMONE_CUES,
            )
        )
        behavior = BehaviorLifetimeSummary(
            behavior=BehaviorKind.FOOD_APPROACH,
            completed_bout_count=5,
            total_duration=5.0,
            median_duration=1.0,
            outcome_counts=(),
            why_summaries=(long_effect, *short_effects),
        )
        report = CreatureBehaviorReport(
            creature_id=8,
            creature_name="Eight",
            deceased=False,
            completed_bouts=(),
            summary=CreatureBehaviorSummary(
                creature_id=8,
                completed_bout_count=5,
                behaviors=(behavior,),
                stable_pattern_threshold=3,
            ),
            history_incomplete=False,
            history_completions_not_recorded=0,
            detailed_bouts_dropped=0,
        )
        cards: list[arcade.Rect] = []
        original_card = self.renderer._draw_why_card
        self.renderer._draw_why_card = (
            lambda _report, _behavior, _effect, bounds: cards.append(bounds)
        )
        try:
            self.renderer._draw_report_why(
                report,
                arcade.LBWH(0, 0, 500, 720),
            )
        finally:
            self.renderer._draw_why_card = original_card

        self.assertEqual(len(cards), 3)
        self.assertEqual(cards[0].height, cards[1].height)
        self.assertGreater(cards[0].height, cards[2].height)
        self.assertEqual(cards[0].bottom - cards[2].top, 12.0)

    def test_overlapping_timeline_bouts_have_stable_click_targets(self) -> None:
        bouts = (
            CompletedBehaviorBout(
                creature_id=8,
                behavior=BehaviorKind.FOOD_ORIENTATION,
                bout_id=1,
                start_time=1.0,
                end_time=3.0,
                duration=2.0,
                evidence_summary=(),
                outcome=None,
                termination=BehaviorTermination.NATURAL,
            ),
            CompletedBehaviorBout(
                creature_id=8,
                behavior=BehaviorKind.FOOD_APPROACH,
                bout_id=2,
                start_time=2.0,
                end_time=4.0,
                duration=2.0,
                evidence_summary=(),
                outcome=None,
                termination=BehaviorTermination.NATURAL,
            ),
        )
        report = CreatureBehaviorReport(
            creature_id=8,
            creature_name="Eight",
            deceased=False,
            completed_bouts=bouts,
            summary=CreatureBehaviorSummary(
                creature_id=8,
                completed_bout_count=2,
                behaviors=(),
                stable_pattern_threshold=3,
            ),
            history_incomplete=False,
            history_completions_not_recorded=0,
            detailed_bouts_dropped=0,
        )
        original_text = self.renderer._draw_text
        original_rounded = self.renderer._draw_rounded_rect
        original_line = arcade.draw_line
        self.renderer._draw_text = lambda *_args, **_kwargs: None
        self.renderer._draw_rounded_rect = lambda *_args, **_kwargs: None
        arcade.draw_line = lambda *_args, **_kwargs: None
        try:
            self.renderer._draw_report_timeline(
                report,
                arcade.LBWH(0, 0, 700, 420),
            )
        finally:
            self.renderer._draw_text = original_text
            self.renderer._draw_rounded_rect = original_rounded
            arcade.draw_line = original_line

        first = self.renderer._control_hitboxes["behavior_report_bout_1"]
        second = self.renderer._control_hitboxes["behavior_report_bout_2"]
        self.assertNotEqual(first.center_y, second.center_y)
        self.assertGreater(first.width, 4.0)
        self.assertGreater(second.width, 4.0)

    def test_bout_detail_draws_every_finalized_evidence_and_why_row(
        self,
    ) -> None:
        evidence = tuple(
            BehaviorEvidenceSummary(
                key=f"metric_{index}",
                label=f"Metric {index}",
                unit="px/s",
                sample_count=10,
                passed_count=8,
                median_value=float(index),
                p25=float(index) - 0.5,
                p75=float(index) + 0.5,
                first_value=0.0,
                last_value=float(index),
            )
            for index in range(5)
        )
        interventions = tuple(SemanticIntervention)[:4]
        effects = tuple(
            CompletedSemanticEffect(
                intervention=intervention,
                sample_count=6,
                median_influence=0.5,
                p25=0.4,
                p75=0.6,
                influence_label=InfluenceLabel.MODERATE,
                effect_direction=EffectDirection.SUPPORTIVE,
                direction_counts=EffectDirectionCounts(supportive=6),
                output_summaries=(),
            )
            for intervention in interventions
        )
        bout = CompletedBehaviorBout(
            creature_id=8,
            behavior=BehaviorKind.FOOD_APPROACH,
            bout_id=3,
            start_time=1.0,
            end_time=3.0,
            duration=2.0,
            evidence_summary=evidence,
            outcome=None,
            termination=BehaviorTermination.NATURAL,
            why_summary=CompletedWhyExplanation(
                behavior=BehaviorKind.FOOD_APPROACH,
                bout_id=3,
                effects=effects,
            ),
        )
        keys: list[str] = []
        original_text = self.renderer._draw_text
        original_rounded = self.renderer._draw_rounded_rect
        self.renderer._draw_text = (
            lambda key, *_args, **_kwargs: keys.append(key)
        )
        self.renderer._draw_rounded_rect = lambda *_args, **_kwargs: None
        try:
            self.renderer._draw_bout_detail(
                bout,
                arcade.LBWH(0, 0, 800, 420),
            )
        finally:
            self.renderer._draw_text = original_text
            self.renderer._draw_rounded_rect = original_rounded

        self.assertEqual(
            sum(
                key.startswith("behavior_report_bout_detail_evidence_")
                and key[-1].isdigit()
                for key in keys
            ),
            5,
        )
        self.assertEqual(
            sum(
                key.startswith("behavior_report_bout_detail_why_")
                and key[-1].isdigit()
                for key in keys
            ),
            4,
        )


if __name__ == "__main__":
    unittest.main()
