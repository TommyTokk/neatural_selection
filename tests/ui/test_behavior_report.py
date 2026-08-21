from __future__ import annotations

from types import SimpleNamespace
from contextlib import nullcontext
import unittest
from unittest.mock import patch

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
    SpeciesBehaviorIndexEntry,
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
                    best_net_energy_balance=4.0,
                    average_net_energy_balance=2.0,
                    average_net_metabolic_rate=1.0,
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
        entry = self._index_entry()
        world = SimpleNamespace(
            layout=SimpleNamespace(window=arcade.LBWH(0, 0, 1440, 900)),
            behavior_history_index=(entry,),
            selected_creature_id=8,
            behavior_report_for=lambda _creature_id: None,
        )
        self.renderer._behavior_report_open = True
        icon_keys: list[str] = []
        text_calls: dict[str, SimpleNamespace] = {}

        def record_text(key, text, x, y, _color, _size, **kwargs) -> None:
            text_calls[key] = SimpleNamespace(
                text=text,
                x=x,
                y=y,
                width=kwargs.get("width"),
                multiline=kwargs.get("multiline", False),
            )

        original_rounded = self.renderer._draw_rounded_rect
        original_text = self.renderer._draw_text
        original_clip = self.renderer._ui_clip
        original_close = self.renderer._draw_panel_close_button
        self.renderer._draw_rounded_rect = lambda *_args, **_kwargs: None
        self.renderer._draw_text = record_text
        self.renderer._ui_clip = lambda _bounds: nullcontext()
        self.renderer._draw_panel_close_button = (
            lambda _bounds, prefix: icon_keys.append(f"{prefix}_close_icon")
        )
        try:
            with (
                patch("arcade.draw_line"),
                patch("arcade.draw_circle_filled"),
            ):
                self.renderer._draw_behavior_report_window(world)

                bounds = self.renderer._behavior_report_bounds
                self.assertIsNotNone(bounds)
                self.assertEqual(bounds.width, 1320.0)
                self.assertEqual(bounds.height, 840.0)
                self.assertEqual(icon_keys, ["behavior_report_close_icon"])
                self.assertEqual(
                    text_calls["behavior_report_subtitle"].text,
                    "Inspect completed behaviour history for observed creatures",
                )

                world.layout.window = arcade.LBWH(0, 0, 640, 480)
                self.renderer._draw_behavior_report_window(world)
        finally:
            self.renderer._draw_rounded_rect = original_rounded
            self.renderer._draw_text = original_text
            self.renderer._ui_clip = original_clip
            self.renderer._draw_panel_close_button = original_close

        bounds = self.renderer._behavior_report_bounds
        self.assertLessEqual(bounds.right, 640 - 24)
        self.assertLessEqual(bounds.top, 480 - 24)
        title = text_calls["behavior_report_title"]
        close = self.renderer._control_hitboxes["behavior_report_close"]
        self.assertTrue(title.multiline)
        self.assertLessEqual(title.x + title.width, close.left)

    def test_species_sidebar_header_measures_title_and_paused_badge(
        self,
    ) -> None:
        bounds = arcade.LBWH(0, 0, 180, 420)
        normal_height = self.renderer._behavior_sidebar_header_height(
            bounds,
            paused=False,
        )
        paused_height = self.renderer._behavior_sidebar_header_height(
            bounds,
            paused=True,
        )

        self.assertGreater(paused_height, normal_height)
        self.assertGreaterEqual(
            self.renderer._behavior_sidebar_header_height(
                arcade.LBWH(0, 0, 90, 420),
                paused=True,
            ),
            paused_height,
        )

    def test_every_behavior_report_text_uses_larger_font_scale(self) -> None:
        with patch.object(self.renderer._painter, "draw_text") as draw_text:
            self.renderer._draw_text(
                "behavior_report_test_text",
                "Report",
                0.0,
                0.0,
                self.renderer.theme.text_primary,
                10.0,
            )
            report_size = draw_text.call_args.args[5]
            self.renderer._draw_text(
                "unrelated_test_text",
                "Other",
                0.0,
                0.0,
                self.renderer.theme.text_primary,
                10.0,
            )
            unrelated_size = draw_text.call_args.args[5]

        self.assertEqual(
            report_size,
            10.0 * self.renderer._BEHAVIOR_REPORT_FONT_SCALE,
        )
        self.assertEqual(unrelated_size, 10.0)

    def test_creature_rows_are_spaced_and_scroll_only_on_overflow(self) -> None:
        entries = tuple(self._index_entry(index) for index in range(9))
        world = SimpleNamespace(
            behavior_history_index=entries,
            selected_creature_id=0,
        )
        bounds = arcade.LBWH(0, 0, 270, 700)

        original_rounded = self.renderer._draw_rounded_rect
        original_text = self.renderer._draw_text
        original_clip = self.renderer._ui_clip
        original_scrollbar = self.renderer._draw_scrollbar
        self.renderer._draw_rounded_rect = lambda *_args, **_kwargs: None
        self.renderer._draw_text = lambda *_args, **_kwargs: None
        self.renderer._ui_clip = lambda _bounds: nullcontext()
        self.renderer._draw_scrollbar = lambda *_args, **_kwargs: None
        try:
            with patch("arcade.draw_circle_filled"):
                self.renderer._draw_report_creature_index(world, bounds)
        finally:
            self.renderer._draw_rounded_rect = original_rounded
            self.renderer._draw_text = original_text
            self.renderer._ui_clip = original_clip
            self.renderer._draw_scrollbar = original_scrollbar

        first = self.renderer._control_hitboxes["behavior_report_creature_0"]
        second = self.renderer._control_hitboxes["behavior_report_creature_1"]
        self.assertEqual(first.height, 56.0)
        self.assertEqual(first.bottom - second.top, 8.0)
        self.assertEqual(
            self.renderer._scroll_limits["behavior_report_creatures"],
            0.0,
        )

        world.behavior_history_index = tuple(
            self._index_entry(index) for index in range(16)
        )
        self.renderer._draw_rounded_rect = lambda *_args, **_kwargs: None
        self.renderer._draw_text = lambda *_args, **_kwargs: None
        self.renderer._ui_clip = lambda _bounds: nullcontext()
        self.renderer._draw_scrollbar = lambda *_args, **_kwargs: None
        try:
            with patch("arcade.draw_circle_filled"):
                self.renderer._draw_report_creature_index(world, bounds)
        finally:
            self.renderer._draw_rounded_rect = original_rounded
            self.renderer._draw_text = original_text
            self.renderer._ui_clip = original_clip
            self.renderer._draw_scrollbar = original_scrollbar
        self.assertGreater(
            self.renderer._scroll_limits["behavior_report_creatures"],
            0.0,
        )

    def test_species_hierarchy_shows_active_and_collapses_historical(self) -> None:
        active = SpeciesBehaviorIndexEntry(
            species_id=3,
            alive_population=8,
            monitored_count=3,
            observed_creature_count=1,
            total_observation_seconds=12.0,
            completed_bout_count=2,
            active=True,
        )
        historical = SpeciesBehaviorIndexEntry(
            species_id=4,
            alive_population=0,
            monitored_count=0,
            observed_creature_count=1,
            total_observation_seconds=8.0,
            completed_bout_count=1,
            active=False,
        )
        creature = CreatureHistoryIndexEntry(
            creature_id=8,
            creature_name="Eight",
            deceased=False,
            last_observed_time=12.0,
            completed_bout_count=2,
            species_id=3,
            active=True,
            last_observation_mode="automatic",
        )
        world = SimpleNamespace(
            species_behavior_index=(active, historical),
            behavior_history_index=(creature,),
            selected_creature_id=None,
            automatic_behavior_cohort_ids=frozenset({8}),
        )
        original_rounded = self.renderer._draw_rounded_rect
        original_text = self.renderer._draw_text
        original_clip = self.renderer._ui_clip
        original_scrollbar = self.renderer._draw_scrollbar
        self.renderer._draw_rounded_rect = lambda *_args, **_kwargs: None
        self.renderer._draw_text = lambda *_args, **_kwargs: None
        self.renderer._ui_clip = lambda _bounds: nullcontext()
        self.renderer._draw_scrollbar = lambda *_args, **_kwargs: None
        try:
            self.renderer._draw_report_creature_index(
                world,
                arcade.LBWH(0, 0, 270, 700),
            )
        finally:
            self.renderer._draw_rounded_rect = original_rounded
            self.renderer._draw_text = original_text
            self.renderer._ui_clip = original_clip
            self.renderer._draw_scrollbar = original_scrollbar

        self.assertIn(
            "behavior_report_species_3",
            self.renderer._control_hitboxes,
        )
        self.assertIn(
            "behavior_report_creature_8",
            self.renderer._control_hitboxes,
        )
        self.assertNotIn(
            "behavior_report_species_4",
            self.renderer._control_hitboxes,
        )
        species_card = self.renderer._control_hitboxes[
            "behavior_report_species_3"
        ]
        creature_card = self.renderer._control_hitboxes[
            "behavior_report_creature_8"
        ]
        viewport = self.renderer._scroll_regions[
            "behavior_report_creatures"
        ]
        self.assertEqual(species_card.bottom - creature_card.top, 8.0)
        self.assertEqual(
            species_card.left - viewport.left,
            self.renderer._SIDEBAR_CARD_OUTER_X,
        )
        self.assertGreaterEqual(
            viewport.right - species_card.right,
            self.renderer._REPORT_SCROLLBAR_GUTTER,
        )
        self.assertTrue(self.renderer._behavior_report_species_selected)

    def test_sidebar_rows_measure_wrapped_text_and_preserve_status_copy(
        self,
    ) -> None:
        narrow_width = 150.0
        wide_width = 270.0
        title = "Creature with an exceptionally long inherited display name"
        detail = "OBSERVED · NO SUSTAINED BOUTS"

        narrow_height = self.renderer._behavior_sidebar_row_height(
            "creature",
            title,
            detail,
            narrow_width,
        )
        wide_height = self.renderer._behavior_sidebar_row_height(
            "creature",
            title,
            detail,
            wide_width,
        )

        self.assertGreater(narrow_height, wide_height)
        statuses = []
        cases = (
            SimpleNamespace(
                creature_id=1,
                creature_name="Auto",
                active=True,
                last_observation_mode="automatic",
                deceased=False,
                completed_bout_count=0,
            ),
            SimpleNamespace(
                creature_id=2,
                creature_name="Focal",
                active=True,
                last_observation_mode="focal",
                deceased=False,
                completed_bout_count=0,
            ),
            SimpleNamespace(
                creature_id=3,
                creature_name="Paused",
                active=False,
                last_observation_mode="automatic",
                deceased=False,
                completed_bout_count=0,
            ),
            SimpleNamespace(
                creature_id=4,
                creature_name="Deceased",
                active=False,
                last_observation_mode="automatic",
                deceased=True,
                completed_bout_count=0,
            ),
            SimpleNamespace(
                creature_id=5,
                creature_name="Observed",
                active=False,
                last_observation_mode="automatic",
                deceased=False,
                completed_bout_count=12,
            ),
            SimpleNamespace(
                creature_id=6,
                creature_name="No bouts",
                active=False,
                last_observation_mode="automatic",
                deceased=False,
                completed_bout_count=0,
            ),
        )
        for creature in cases:
            statuses.append(
                self.renderer._behavior_sidebar_row_text(
                    "creature",
                    creature,
                    {3},
                )[1]
            )
        self.assertEqual(
            statuses,
            [
                "AUTO · RECORDING",
                "FOCAL · RECORDING",
                "AUTO PAUSED",
                "DECEASED",
                "OBSERVED · 12 BOUTS",
                "OBSERVED · NO SUSTAINED BOUTS",
            ],
        )

    def test_large_species_metrics_and_many_creatures_scroll_safely(
        self,
    ) -> None:
        species = SpeciesBehaviorIndexEntry(
            species_id=123456,
            alive_population=12345,
            monitored_count=3,
            observed_creature_count=14,
            total_observation_seconds=500.0,
            completed_bout_count=987654,
            active=True,
        )
        creatures = tuple(
            CreatureHistoryIndexEntry(
                creature_id=index,
                creature_name=(
                    f"Creature {index} with a deliberately long display name"
                ),
                deceased=False,
                last_observed_time=12.0,
                completed_bout_count=index,
                species_id=123456,
                active=index < 3,
                last_observation_mode="automatic",
            )
            for index in range(14)
        )
        world = SimpleNamespace(
            species_behavior_index=(species,),
            behavior_history_index=creatures,
            selected_creature_id=None,
            automatic_behavior_cohort_ids=frozenset({0, 1, 2}),
        )
        original_rounded = self.renderer._draw_rounded_rect
        original_text = self.renderer._draw_text
        original_clip = self.renderer._ui_clip
        original_scrollbar = self.renderer._draw_scrollbar
        self.renderer._draw_rounded_rect = lambda *_args, **_kwargs: None
        self.renderer._draw_text = lambda *_args, **_kwargs: None
        self.renderer._ui_clip = lambda _bounds: nullcontext()
        self.renderer._draw_scrollbar = lambda *_args, **_kwargs: None
        try:
            self.renderer._draw_report_creature_index(
                world,
                arcade.LBWH(0, 0, 190, 420),
            )
        finally:
            self.renderer._draw_rounded_rect = original_rounded
            self.renderer._draw_text = original_text
            self.renderer._ui_clip = original_clip
            self.renderer._draw_scrollbar = original_scrollbar

        self.assertGreater(
            self.renderer._scroll_limits["behavior_report_creatures"],
            0.0,
        )
        species_card = self.renderer._control_hitboxes[
            "behavior_report_species_123456"
        ]
        self.assertLessEqual(species_card.width, 174.0)

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

    def test_help_copy_is_complete_and_uses_runtime_thresholds(self) -> None:
        config = build_sim_config()
        config.behavior.background_representatives_per_species = 2
        config.behavior.orientation_min_error_reduction = 0.22
        config.behavior.approach_min_closing_speed = 9.5
        config.behavior_history.active_metric_sample_capacity = 640

        sections = self.renderer._behavior_report_help_sections(
            SimpleNamespace(config=config)
        )
        titles = {title for title, _body in sections}
        copy = " ".join(body for _title, body in sections)

        self.assertTrue(
            {
                "Monitoring and coverage",
                "Food orientation",
                "Food approach",
                "Feeding",
                "Resting",
                "Cohesion",
                "Alarm retreat",
                "Bouts and Evidence",
                "Counterfactual influence",
                "Influence labels",
                "Effect directions",
                "Median, quartiles, and IQR",
            }.issubset(titles)
        )
        for expected in (
            "2 stable living representatives",
            "0.22 rad/s",
            "9.5 px/s",
            "MINIMAL is below 0.10",
            "STRONG is 0.60 or above",
            "Q1–Q3 is the middle-50% interval",
            "IQR is its width",
            "640 probes",
        ):
            self.assertIn(expected, copy)

    def test_quartile_spread_shows_interval_width_and_estimate(self) -> None:
        self.assertEqual(
            self.renderer._quartile_spread_text(
                0.4,
                0.6,
                estimated=True,
            ),
            " · Q1–Q3 0.40–0.60 · IQR 0.20 · estimated",
        )
        self.assertEqual(
            self.renderer._quartile_spread_text(
                None,
                None,
                estimated=True,
            ),
            " · estimated",
        )

    def test_lifetime_why_card_includes_quartiles_iqr_and_estimate(self) -> None:
        report = self._why_report(completed_bout_count=3)
        effect = BehaviorLifetimeWhySummary(
            intervention=SemanticIntervention.VISIBLE_FOOD_CUES,
            behavior_bout_count=3,
            contributing_bout_count=3,
            median_bout_influence=0.5,
            p25=0.4,
            p75=0.7,
            influence_label=InfluenceLabel.MODERATE,
            direction_counts=EffectDirectionCounts(supportive=3),
            quantiles_estimated=True,
        )

        _title, median, _available, _wording = (
            self.renderer._why_card_text(report, effect)
        )

        self.assertIn("Q1–Q3 0.40–0.70", median)
        self.assertIn("IQR 0.30", median)
        self.assertIn("estimated", median)

    def test_help_button_stays_inside_report_bottom_right(self) -> None:
        bounds = arcade.LBWH(20, 30, 640, 460)
        original_rounded = self.renderer._draw_rounded_rect
        original_text = self.renderer._draw_text
        self.renderer._draw_rounded_rect = lambda *_args, **_kwargs: None
        self.renderer._draw_text = lambda *_args, **_kwargs: None
        try:
            self.renderer._draw_behavior_report_help_button(bounds)
        finally:
            self.renderer._draw_rounded_rect = original_rounded
            self.renderer._draw_text = original_text

        button = self.renderer._control_hitboxes["behavior_report_help"]
        self.assertLessEqual(button.right, bounds.right)
        self.assertGreaterEqual(button.bottom, bounds.bottom)
        self.assertGreater(button.center_x, bounds.center_x)
        self.assertLess(button.center_y, bounds.center_y)

    def test_help_overlay_is_responsive_and_scrolls_independently(self) -> None:
        world = SimpleNamespace(config=build_sim_config())
        report_bounds = arcade.LBWH(24, 24, 592, 432)
        original_rounded = self.renderer._draw_rounded_rect
        original_text = self.renderer._draw_text
        original_clip = self.renderer._ui_clip
        original_close = self.renderer._draw_panel_close_button
        original_scrollbar = self.renderer._draw_scrollbar
        text_widths: list[float] = []

        def record_text(key, _text, *_args, **kwargs) -> None:
            if key.startswith("behavior_report_help_section_"):
                text_widths.append(kwargs["width"])

        self.renderer._draw_rounded_rect = lambda *_args, **_kwargs: None
        self.renderer._draw_text = record_text
        self.renderer._ui_clip = lambda _bounds: nullcontext()
        self.renderer._draw_panel_close_button = lambda *_args, **_kwargs: None
        self.renderer._draw_scrollbar = lambda *_args, **_kwargs: None
        try:
            with (
                patch("arcade.draw_lrbt_rectangle_filled"),
                patch("arcade.draw_line"),
            ):
                self.renderer._draw_behavior_report_help_overlay(
                    world,
                    report_bounds,
                )
                self.renderer._scroll_offsets["behavior_report_help"] = 1e9
                self.renderer._draw_behavior_report_help_overlay(
                    world,
                    report_bounds,
                )
        finally:
            self.renderer._draw_rounded_rect = original_rounded
            self.renderer._draw_text = original_text
            self.renderer._ui_clip = original_clip
            self.renderer._draw_panel_close_button = original_close
            self.renderer._draw_scrollbar = original_scrollbar

        overlay = self.renderer._control_hitboxes[
            "behavior_report_help_overlay"
        ]
        self.assertGreaterEqual(overlay.left, report_bounds.left)
        self.assertLessEqual(overlay.right, report_bounds.right)
        self.assertGreaterEqual(overlay.bottom, report_bounds.bottom)
        self.assertLessEqual(overlay.top, report_bounds.top)
        self.assertGreater(
            self.renderer._scroll_limits["behavior_report_help"],
            0.0,
        )
        self.assertEqual(
            self.renderer._scroll_offsets["behavior_report_help"],
            self.renderer._scroll_limits["behavior_report_help"],
        )
        viewport = self.renderer._scroll_regions["behavior_report_help"]
        self.assertTrue(text_widths)
        self.assertTrue(
            all(
                width
                <= viewport.width - self.renderer._REPORT_SCROLLBAR_GUTTER
                for width in text_widths
            )
        )
        self.renderer._behavior_report_open = True
        self.renderer._behavior_report_help_open = True
        self.renderer._behavior_report_bounds = report_bounds
        self.renderer._scroll_offsets["behavior_report_help"] = 20.0
        self.renderer._scroll_offsets["behavior_report"] = 35.0
        self.renderer.handle_mouse_scroll(
            viewport.center_x,
            viewport.center_y,
            -1.0,
        )
        self.assertEqual(
            self.renderer._scroll_offsets["behavior_report_help"],
            44.0,
        )
        self.assertEqual(self.renderer._scroll_offsets["behavior_report"], 35.0)

    def test_help_overlay_blocks_report_actions_and_preserves_selection(
        self,
    ) -> None:
        report = arcade.LBWH(0, 0, 900, 700)
        help_button = arcade.LBWH(840, 20, 36, 36)
        underlying = arcade.LBWH(20, 100, 200, 50)
        help_close = arcade.LBWH(760, 620, 34, 34)
        self.renderer._behavior_report_open = True
        self.renderer._behavior_report_bounds = report
        self.renderer._behavior_report_creature_id = 8
        self.renderer._behavior_report_page = "summary"
        self.renderer._control_hitboxes["behavior_report_help"] = help_button
        self.renderer._control_hitboxes["behavior_report_species_3"] = underlying
        world = SimpleNamespace(species_behavior_index=())

        self.assertTrue(
            self.renderer.handle_mouse_press(
                world,
                help_button.center_x,
                help_button.center_y,
            )
        )
        self.assertTrue(self.renderer._behavior_report_help_open)
        self.renderer._control_hitboxes[
            "behavior_report_help_close"
        ] = help_close
        self.renderer.handle_mouse_press(
            world,
            underlying.center_x,
            underlying.center_y,
        )
        self.assertEqual(self.renderer._behavior_report_creature_id, 8)
        self.assertEqual(self.renderer._behavior_report_page, "summary")

        self.renderer.handle_mouse_press(
            world,
            help_close.center_x,
            help_close.center_y,
        )
        self.assertFalse(self.renderer._behavior_report_help_open)
        self.assertEqual(self.renderer._behavior_report_creature_id, 8)
        self.assertEqual(self.renderer._behavior_report_page, "summary")
        self.renderer._behavior_report_help_open = True
        self.renderer._scroll_offsets["behavior_report_help"] = 48.0
        self.renderer._close_behavior_report()
        self.assertFalse(self.renderer._behavior_report_help_open)
        self.assertEqual(
            self.renderer._scroll_offsets["behavior_report_help"],
            0.0,
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

        original_rounded = self.renderer._draw_rounded_rect
        original_text = self.renderer._draw_text
        text_calls: dict[str, SimpleNamespace] = {}

        def record_text(key, text, x, y, _color, size, **kwargs) -> None:
            text_calls[key] = SimpleNamespace(
                text=text,
                x=round(x),
                y=round(y),
                font_size=size * self.renderer._BEHAVIOR_REPORT_FONT_SCALE,
                anchor_y=kwargs.get("anchor_y", "baseline"),
            )

        self.renderer._draw_rounded_rect = lambda *_args, **_kwargs: None
        self.renderer._draw_text = record_text
        try:
            with patch("arcade.draw_circle_filled"):
                self.renderer._draw_summary_card(short, card)
        finally:
            self.renderer._draw_rounded_rect = original_rounded
            self.renderer._draw_text = original_text

        title = text_calls["behavior_report_summary_feeding"]
        outcome = text_calls["behavior_report_outcomes_feeding"]
        self.assertEqual(
            title.font_size,
            12.0 * self.renderer._BEHAVIOR_REPORT_FONT_SCALE,
        )
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
        original_identity = self.renderer._draw_report_identity_strip
        original_clip = self.renderer._ui_clip
        original_scrollbar = self.renderer._draw_scrollbar
        self.renderer._draw_summary_card = (
            lambda _item, bounds: cards.append(bounds)
        )
        self.renderer._draw_report_identity_strip = (
            lambda *_args, **_kwargs: None
        )
        self.renderer._ui_clip = lambda _bounds: nullcontext()
        self.renderer._draw_scrollbar = lambda *_args, **_kwargs: None
        try:
            self.renderer._draw_report_summary(
                report,
                arcade.LBWH(0, 0, 700, 600),
            )
        finally:
            self.renderer._draw_summary_card = original_card
            self.renderer._draw_report_identity_strip = original_identity
            self.renderer._ui_clip = original_clip
            self.renderer._draw_scrollbar = original_scrollbar

        self.assertEqual(len(cards), 4)
        self.assertEqual(cards[0].height, cards[1].height)
        self.assertEqual(cards[2].height, cards[3].height)
        self.assertGreater(cards[0].height, cards[2].height)
        self.assertEqual(cards[0].bottom - cards[2].top, 12.0)
        self.assertLessEqual(
            max(card.right for card in cards),
            700.0 - self.renderer._REPORT_SCROLLBAR_GUTTER,
        )

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

        original_rounded = self.renderer._draw_rounded_rect
        original_text = self.renderer._draw_text
        text_calls: dict[str, SimpleNamespace] = {}

        def record_text(key, text, x, y, _color, size, **kwargs) -> None:
            text_calls[key] = SimpleNamespace(
                text=text,
                x=round(x),
                font_size=size * self.renderer._BEHAVIOR_REPORT_FONT_SCALE,
                multiline=kwargs.get("multiline", False),
                anchor_y=kwargs.get("anchor_y", "baseline"),
            )

        self.renderer._draw_rounded_rect = lambda *_args, **_kwargs: None
        self.renderer._draw_text = record_text
        try:
            with patch("arcade.draw_circle_filled"):
                self.renderer._draw_why_card(
                    report,
                    behavior,
                    effect,
                    card,
                )
        finally:
            self.renderer._draw_rounded_rect = original_rounded
            self.renderer._draw_text = original_text

        prefix = "food_approach_resource_gradient_cues"
        title = text_calls[
            f"behavior_report_why_intervention_{prefix}"
        ]
        pattern = text_calls[
            f"behavior_report_why_pattern_{prefix}"
        ]
        self.assertEqual(title.text, "Resource Gradient Cues")
        self.assertNotIn("...", title.text)
        self.assertEqual(
            title.font_size,
            12.0 * self.renderer._BEHAVIOR_REPORT_FONT_SCALE,
        )
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
        original_identity = self.renderer._draw_report_identity_strip
        original_rounded = self.renderer._draw_rounded_rect
        original_text = self.renderer._draw_text
        original_clip = self.renderer._ui_clip
        original_scrollbar = self.renderer._draw_scrollbar
        self.renderer._draw_why_card = (
            lambda _report, _behavior, _effect, bounds: cards.append(bounds)
        )
        self.renderer._draw_report_identity_strip = (
            lambda *_args, **_kwargs: None
        )
        self.renderer._draw_rounded_rect = lambda *_args, **_kwargs: None
        self.renderer._draw_text = lambda *_args, **_kwargs: None
        self.renderer._ui_clip = lambda _bounds: nullcontext()
        self.renderer._draw_scrollbar = lambda *_args, **_kwargs: None
        try:
            self.renderer._draw_report_why(
                report,
                arcade.LBWH(0, 0, 500, 720),
            )
        finally:
            self.renderer._draw_why_card = original_card
            self.renderer._draw_report_identity_strip = original_identity
            self.renderer._draw_rounded_rect = original_rounded
            self.renderer._draw_text = original_text
            self.renderer._ui_clip = original_clip
            self.renderer._draw_scrollbar = original_scrollbar

        self.assertEqual(len(cards), 3)
        self.assertEqual(cards[0].height, cards[1].height)
        self.assertGreater(cards[0].height, cards[2].height)
        self.assertEqual(cards[0].bottom - cards[2].top, 12.0)
        self.assertLessEqual(
            max(card.right for card in cards),
            500.0 - self.renderer._REPORT_SCROLLBAR_GUTTER,
        )

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
