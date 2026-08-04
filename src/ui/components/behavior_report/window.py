from __future__ import annotations

import arcade

from src.behavior_history import (
    BehaviorLifetimeSummary,
    BehaviorLifetimeWhySummary,
    CompletedBehaviorBout,
    CreatureBehaviorReport,
    SpeciesBehaviorReport,
)
from src.behavior_observer import BehaviorKind
from src.counterfactual_neat import (
    MINIMAL_INFLUENCE_THRESHOLD,
    MODERATE_INFLUENCE_THRESHOLD,
    WEAK_INFLUENCE_THRESHOLD,
)
from src.world import World


class BehaviorReportWindowComponent:
    """Draw the completed-bout report without changing report calculations."""

    _BEHAVIOR_REPORT_FONT_SCALE = 1.20
    _CARD_PADDING_X = 16.0
    _CARD_PADDING_Y = 14.0
    _CARD_TITLE_SIZE = 12.0
    _CARD_TITLE_LINE_HEIGHT = 16.0
    _CARD_BODY_SIZE = 9.5
    _CARD_BODY_LINE_HEIGHT = 14.0
    _CARD_VALUE_SIZE = 10.0
    _CARD_TITLE_GAP = 12.0
    _SIDEBAR_PADDING_X = 16.0
    _SIDEBAR_HEADER_PADDING_Y = 14.0
    _SIDEBAR_HEADER_GAP = 8.0
    _SIDEBAR_ROW_GAP = 8.0
    _SIDEBAR_ROW_PADDING_X = 12.0
    _SIDEBAR_ROW_PADDING_Y = 12.0
    _SIDEBAR_ROW_TEXT_GAP = 7.0
    _SIDEBAR_TITLE_LINE_HEIGHT = 15.0
    _SIDEBAR_DETAIL_LINE_HEIGHT = 12.0
    _SIDEBAR_CARD_OUTER_X = 6.0
    _REPORT_SCROLLBAR_GUTTER = 16.0
    _REPORT_SCROLL_PADDING_Y = 12.0

    _BEHAVIOR_COLORS = {
        BehaviorKind.FOOD_ORIENTATION: (20, 172, 150),
        BehaviorKind.FOOD_APPROACH: (96, 176, 63),
        BehaviorKind.FEEDING: (239, 139, 24),
        BehaviorKind.RESTING: (145, 105, 218),
        BehaviorKind.COHESION: (66, 111, 181),
        BehaviorKind.ALARM_RETREAT: (224, 83, 108),
    }
    _SECTION_FILL = (248, 248, 252)
    _LIVING = (30, 157, 99)
    _LIVING_SOFT = (224, 247, 236)
    _DECEASED = (99, 107, 121)
    _DECEASED_SOFT = (237, 239, 244)

    @staticmethod
    def _format_report_duration(seconds: float) -> str:
        """Format observed or behavioral duration for compact report copy."""
        value = max(0.0, float(seconds))
        if value >= 3600.0:
            return f"{value / 3600.0:.1f} h"
        if value >= 60.0:
            return f"{value / 60.0:.1f} min"
        return f"{value:.1f} s"

    def _behavior_report_help_sections(
        self,
        world: World,
    ) -> tuple[tuple[str, str], ...]:
        """Return help copy using the active observer configuration."""
        behavior = world.config.behavior
        history = world.config.behavior_history
        return (
            (
                "Monitoring and coverage",
                "With no creature selected, AUTO monitoring records up to "
                f"{behavior.background_representatives_per_species} stable "
                "living representatives per species. Selecting a creature "
                "pauses those cohorts and starts exclusive FOCAL monitoring. "
                "Observed time is worker-processed creature-time; completed "
                "bouts are sustained finalized behaviors. Bouts per observed "
                "creature-hour divides completed bouts by total processed "
                "observation time.",
            ),
            (
                "Pages and creature states",
                "TIMELINE places finalized bouts in time. SUMMARY reports bout "
                "counts and durations. WHY reports focal counterfactual brain "
                "analysis and is intentionally unavailable for background-only "
                "observation. AUTO or FOCAL · RECORDING is live; AUTO PAUSED "
                "is a preserved cohort during focal study; OBSERVED means "
                "history exists; DECEASED retains historical results.",
            ),
            (
                "Food orientation",
                "Uses the same visible food target for at least three samples. "
                f"It requires {behavior.food_visibility_ratio:.0%} target "
                "visibility, heading-error reduction of at least "
                f"{behavior.orientation_min_error_reduction:.2f} rad/s, "
                f"{behavior.trend_consistency_ratio:.0%} improving steps, and "
                "a realized turn toward food of at least "
                f"{behavior.orientation_min_turn_rate:.2f} rad/s.",
            ),
            (
                "Food approach",
                "Uses persistent visible-food distance and realized velocity. "
                f"It requires {behavior.food_visibility_ratio:.0%} visibility, "
                f"closing speed of at least {behavior.approach_min_closing_speed:.1f} "
                f"px/s, {behavior.trend_consistency_ratio:.0%} distance-decreasing "
                "steps, and movement-to-food cosine alignment of at least "
                f"{behavior.movement_alignment_threshold:.2f}.",
            ),
            (
                "Feeding",
                "Uses explicit per-creature consumption events. Feeding is "
                "recorded only when both the cumulative consumption count and "
                "swallowed-energy total increase; proximity or eating intent "
                "alone is not enough.",
            ),
            (
                "Resting",
                "Uses realized movement speed. Current speed must be no more "
                f"than {behavior.rest_speed_threshold:.1f} px/s and at least "
                "80% of samples in the temporal window must be at or below "
                "that threshold.",
            ),
            (
                "Cohesion",
                "Uses compatible-group visibility, group-center distance, "
                "personal-space state, and realized velocities. A group must "
                "be visible for at least 60% of samples and the creature must "
                "remain outside personal space for at least 80%. It must then "
                f"close at {behavior.cohesion_min_closing_speed:.1f} px/s while "
                "moving toward the center, or follow group velocity with "
                f"alignment of at least {behavior.cohesion_min_velocity_alignment:.2f}.",
            ),
            (
                "Alarm retreat",
                "Uses local and forward alarm-pheromone samples plus realized "
                "forward speed. Local alarm must be at least "
                f"{behavior.alarm_min_level:.2f}, forward alarm at least "
                f"{behavior.alarm_min_spatial_gradient:.2f} lower, exposure "
                f"must fall by {behavior.alarm_min_temporal_drop:.2f}/s with "
                f"{behavior.trend_consistency_ratio:.0%} consistency, and "
                f"retreat speed must reach {behavior.alarm_retreat_min_speed:.1f} px/s.",
            ),
            (
                "Bouts and Evidence",
                f"Rules are sampled at {behavior.sample_hz:g} Hz over up to "
                f"{behavior.window_seconds:g} seconds. Ordinary bouts are "
                f"EMERGING for {behavior.bout_start_seconds:g} seconds before "
                f"becoming ACTIVE and tolerate {behavior.bout_end_grace_seconds:g} "
                "seconds without matching evidence. Feeding activates "
                f"immediately and remains displayed for {behavior.feeding_display_seconds:g} "
                "seconds. Evidence is a clamped 0–1 rule-satisfaction score, "
                "not a probability or learned confidence.",
            ),
            (
                "Counterfactual influence",
                "For focal bouts, WHY compares the factual brain outputs with "
                "outputs after a sensor family is removed or replaced. Each "
                "absolute output change is normalized to 0–1 by that output's "
                "range, then behavior-relevant outputs are averaged. Food "
                "orientation scores rotation; approach scores acceleration and "
                "rotation; feeding scores eat intent; cohesion and alarm retreat "
                "score their relevant movement and specialized outputs. Resting "
                "currently has no counterfactual WHY specification. Signed "
                "movement direction is evaluated toward the factual food or "
                "flock-center heading; alarm retreat favors forward movement "
                "with a stable heading.",
            ),
            (
                "Influence labels",
                f"MINIMAL is below {MINIMAL_INFLUENCE_THRESHOLD:.2f}; WEAK is "
                f"{MINIMAL_INFLUENCE_THRESHOLD:.2f} to below "
                f"{WEAK_INFLUENCE_THRESHOLD:.2f}; MODERATE is "
                f"{WEAK_INFLUENCE_THRESHOLD:.2f} to below "
                f"{MODERATE_INFLUENCE_THRESHOLD:.2f}; STRONG is "
                f"{MODERATE_INFLUENCE_THRESHOLD:.2f} or above.",
            ),
            (
                "Effect directions",
                "SUPPORTIVE means the factual cues strengthen the relevant "
                "response relative to the counterfactual. SUPPRESSIVE means "
                "they reduce it. REVERSING means a critical signed response "
                "changes direction. MIXED means scored outputs disagree. "
                "MINIMAL means the normalized change is below the minimum "
                "influence threshold.",
            ),
            (
                "Median, quartiles, and IQR",
                "The median is the middle retained value and resists occasional "
                "extremes. Q1–Q3 is the middle-50% interval; IQR is its width, "
                "Q3 − Q1. Both are unavailable for a single value. WHY first "
                "summarizes "
                "each completed bout by its median, then takes the lifetime "
                "median and IQR across bouts so long bouts receive no extra "
                "weight. Probe histories display one real paired probe nearest "
                "the median influence, using the newest probe to break ties. "
                "Quantiles are exact through "
                f"{history.active_metric_sample_capacity} probes per active "
                "metric and are marked estimated after deterministic bounded "
                "compaction.",
            ),
        )

    def _draw_behavior_report_help_button(
        self,
        bounds: arcade.Rect,
    ) -> None:
        """Draw the fixed bottom-right report help control."""
        button = arcade.LBWH(
            bounds.right - 54.0,
            bounds.bottom + 18.0,
            36.0,
            36.0,
        )
        self._control_hitboxes["behavior_report_help"] = button
        self._draw_rounded_rect(
            button,
            self.theme.panel_background,
            self.theme.accent,
            12.0,
            1.5,
        )
        self._draw_text(
            "behavior_report_help_label",
            "?",
            button.center_x,
            button.center_y + 1.0,
            self.theme.accent,
            16.0,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )

    def _draw_behavior_report_help_overlay(
        self,
        world: World,
        report_bounds: arcade.Rect,
    ) -> None:
        """Draw the modal, independently scrollable report explanation."""
        arcade.draw_lrbt_rectangle_filled(
            report_bounds.left,
            report_bounds.right,
            report_bounds.bottom,
            report_bounds.top,
            (20, 25, 34, 150),
        )
        width = min(820.0, max(1.0, report_bounds.width - 48.0))
        height = min(700.0, max(1.0, report_bounds.height - 48.0))
        overlay = arcade.LBWH(
            report_bounds.center_x - width / 2.0,
            report_bounds.center_y - height / 2.0,
            width,
            height,
        )
        self._control_hitboxes["behavior_report_help_overlay"] = overlay
        self._draw_rounded_rect(
            overlay,
            self.theme.panel_background,
            self.theme.panel_border,
            14.0,
            1.5,
        )
        header_height = 68.0
        close = arcade.LBWH(
            overlay.right - 50.0,
            overlay.top - 51.0,
            34.0,
            34.0,
        )
        self._control_hitboxes["behavior_report_help_close"] = close
        self._draw_panel_close_button(close, "behavior_report_help")
        self._draw_text(
            "behavior_report_help_title",
            "HOW TO READ THIS REPORT",
            overlay.left + 24.0,
            overlay.top - 22.0,
            self.theme.text_primary,
            16.0,
            bold=True,
            width=max(1.0, close.left - overlay.left - 36.0),
            multiline=True,
            anchor_y="top",
        )
        arcade.draw_line(
            overlay.left,
            overlay.top - header_height,
            overlay.right,
            overlay.top - header_height,
            self.theme.panel_border,
            1.0,
        )
        viewport = arcade.LBWH(
            overlay.left + 24.0,
            overlay.bottom + 20.0,
            max(1.0, overlay.width - 48.0),
            max(0.0, overlay.height - header_height - 36.0),
        )
        content_width = max(
            1.0,
            viewport.width - self._REPORT_SCROLLBAR_GUTTER,
        )
        sections = self._behavior_report_help_sections(world)
        measured: list[tuple[str, str, float, float]] = []
        for title, body in sections:
            title_height = self._wrapped_text_height(
                title,
                content_width,
                11.5,
                16.0,
                bold=True,
            )
            body_height = self._wrapped_text_height(
                body,
                content_width,
                9.5,
                14.0,
            )
            measured.append((title, body, title_height, body_height))
        section_gap = 20.0
        content_height = (
            self._REPORT_SCROLL_PADDING_Y * 2.0
            + sum(
                title_height + 7.0 + body_height
                for _title, _body, title_height, body_height in measured
            )
            + section_gap * max(0, len(measured) - 1)
        )
        scroll_limit = max(0.0, content_height - viewport.height)
        scroll_offset = max(
            0.0,
            min(
                scroll_limit,
                self._scroll_offsets.get("behavior_report_help", 0.0),
            ),
        )
        self._scroll_regions["behavior_report_help"] = viewport
        self._scroll_limits["behavior_report_help"] = scroll_limit
        self._scroll_offsets["behavior_report_help"] = scroll_offset
        cursor = (
            viewport.top
            + scroll_offset
            - self._REPORT_SCROLL_PADDING_Y
        )
        with self._ui_clip(viewport):
            for index, (title, body, title_height, body_height) in enumerate(
                measured
            ):
                block_height = title_height + 7.0 + body_height
                block = arcade.LBWH(
                    viewport.left,
                    cursor - block_height,
                    content_width,
                    block_height,
                )
                if self._rect_intersects(block, viewport):
                    self._draw_text(
                        f"behavior_report_help_section_{index}_title",
                        title,
                        block.left,
                        block.top,
                        self.theme.text_primary,
                        11.5,
                        bold=True,
                        width=block.width,
                        multiline=True,
                        anchor_y="top",
                    )
                    self._draw_text(
                        f"behavior_report_help_section_{index}_body",
                        body,
                        block.left,
                        block.top - title_height - 7.0,
                        self.theme.text_muted,
                        9.5,
                        width=block.width,
                        multiline=True,
                        anchor_y="top",
                    )
                cursor -= block_height + section_gap
        if scroll_limit > 0.0:
            self._draw_scrollbar(viewport, scroll_offset, scroll_limit)

    def _draw_behavior_report_window(self, world: World) -> None:
        """Draw the enlarged completed-history modal when it is open."""
        if not self._behavior_report_open:
            return
        window = world.layout.window
        margin = 24.0
        width = min(1320.0, max(1.0, window.width - margin * 2.0))
        height = min(840.0, max(1.0, window.height - margin * 2.0))
        bounds = arcade.LBWH(
            window.center_x - width / 2.0,
            window.center_y - height / 2.0,
            width,
            height,
        )
        self._behavior_report_bounds = bounds
        self._control_hitboxes["behavior_report_window"] = bounds
        self._draw_rounded_rect(
            bounds,
            self.theme.panel_background,
            self.theme.panel_border,
            16.0,
            1.5,
        )

        header_padding_x = 28.0
        header_padding_y = 16.0
        close_reserved_width = 74.0
        header_text_width = max(
            1.0,
            bounds.width - header_padding_x * 2.0 - close_reserved_width,
        )
        title_text = "BEHAVIOUR REPORT"
        subtitle_text = (
            "Inspect completed behaviour history for observed creatures"
        )
        title_line_height = 24.0
        subtitle_line_height = 15.0
        title_height = self._wrapped_text_height(
            title_text,
            header_text_width,
            19.0,
            title_line_height,
            bold=True,
        )
        subtitle_height = self._wrapped_text_height(
            subtitle_text,
            header_text_width,
            10.5,
            subtitle_line_height,
        )
        header_height = max(
            82.0,
            header_padding_y * 2.0
            + title_height
            + 5.0
            + subtitle_height,
        )
        header = arcade.LBWH(
            bounds.left,
            bounds.top - header_height,
            bounds.width,
            header_height,
        )
        arcade.draw_line(
            header.left,
            header.bottom,
            header.right,
            header.bottom,
            self.theme.panel_border,
            1.0,
        )
        self._draw_text(
            "behavior_report_title",
            title_text,
            header.left + header_padding_x,
            header.top - header_padding_y,
            self.theme.text_primary,
            19.0,
            bold=True,
            width=header_text_width,
            multiline=True,
            anchor_y="top",
        )
        self._draw_text(
            "behavior_report_subtitle",
            subtitle_text,
            header.left + header_padding_x,
            header.top - header_padding_y - title_height - 5.0,
            self.theme.text_muted,
            10.5,
            width=header_text_width,
            multiline=True,
            anchor_y="top",
        )
        close = arcade.LBWH(
            header.right - 54.0,
            header.center_y - 17.0,
            34.0,
            34.0,
        )
        self._control_hitboxes["behavior_report_close"] = close
        self._draw_panel_close_button(close, "behavior_report")

        report_footer_height = 48.0
        body_bottom = bounds.bottom + 16.0 + report_footer_height
        body_height = max(0.0, header.bottom - body_bottom - 16.0)
        sidebar_width = (
            270.0
            if bounds.width >= 900.0
            else max(150.0, bounds.width * 0.30)
        )
        sidebar = arcade.LBWH(
            bounds.left + 16.0,
            body_bottom,
            min(sidebar_width, max(1.0, bounds.width - 64.0)),
            body_height,
        )
        content = arcade.LBWH(
            sidebar.right + 16.0,
            body_bottom,
            max(1.0, bounds.right - sidebar.right - 32.0),
            body_height,
        )
        self._draw_report_creature_index(world, sidebar)
        if self._behavior_report_species_selected:
            species_report = world.species_behavior_report_for(
                self._behavior_report_species_id
            )
            self._draw_species_report_overview(species_report, content)
        else:
            report = self._selected_behavior_report(world)
            if report is None:
                self._draw_report_empty_body(
                    content,
                    "No completed focal history is available yet.",
                )
            else:
                self._draw_report_content(report, content)
        self._draw_behavior_report_help_button(bounds)
        if self._behavior_report_help_open:
            self._draw_behavior_report_help_overlay(world, bounds)

    def _draw_report_creature_index(
        self,
        world: World,
        bounds: arcade.Rect,
    ) -> None:
        """Draw species hierarchy when supported, with legacy flat fallback."""
        species_entries = tuple(
            getattr(world, "species_behavior_index", ()) or ()
        )
        if not species_entries:
            self._draw_report_flat_creature_index(world, bounds)
            return
        entries = tuple(world.behavior_history_index)
        selected = getattr(world, "selected_creature_id", None)
        paused_automatic_ids = (
            set(getattr(world, "automatic_behavior_cohort_ids", ()))
            if selected is not None
            else set()
        )
        known_ids = {entry.creature_id for entry in entries}
        if selected in known_ids:
            self._behavior_report_creature_id = selected
            self._behavior_report_species_selected = False
        elif (
            self._behavior_report_creature_id not in known_ids
            and not self._behavior_report_species_selected
        ):
            first_active = next(
                (entry for entry in species_entries if entry.active),
                species_entries[0],
            )
            self._behavior_report_species_id = first_active.species_id
            self._behavior_report_species_selected = True

        self._draw_rounded_rect(
            bounds,
            self._SECTION_FILL,
            self.theme.panel_border,
            11.0,
            1.0,
        )
        header_height = self._behavior_sidebar_header_height(
            bounds,
            paused=selected is not None,
        )
        title_width = max(1.0, bounds.width - self._SIDEBAR_PADDING_X * 2.0)
        title_height = self._wrapped_text_height(
            "SPECIES MONITORING",
            title_width,
            self._CARD_TITLE_SIZE,
            self._CARD_TITLE_LINE_HEIGHT,
            bold=True,
        )
        self._draw_text(
            "behavior_report_creatures_title",
            "SPECIES MONITORING",
            bounds.left + self._SIDEBAR_PADDING_X,
            bounds.top - self._SIDEBAR_HEADER_PADDING_Y,
            self.theme.text_primary,
            self._CARD_TITLE_SIZE,
            bold=True,
            width=title_width,
            multiline=True,
            anchor_y="top",
        )
        if selected is not None:
            paused_badge = arcade.LBWH(
                bounds.left + self._SIDEBAR_PADDING_X,
                bounds.top
                - self._SIDEBAR_HEADER_PADDING_Y
                - title_height
                - self._SIDEBAR_HEADER_GAP
                - 20.0,
                min(96.0, title_width),
                20.0,
            )
            self._draw_rounded_rect(
                paused_badge,
                self._DECEASED_SOFT,
                self._DECEASED,
                7.0,
                0.8,
            )
            self._draw_text(
                "behavior_report_auto_paused",
                "AUTO PAUSED",
                paused_badge.center_x,
                paused_badge.center_y,
                self._DECEASED,
                8.0,
                bold=True,
                anchor_x="center",
                anchor_y="center",
            )
        for key in tuple(self._control_hitboxes):
            if key.startswith("behavior_report_creature_") or key.startswith(
                "behavior_report_species_"
            ):
                self._control_hitboxes.pop(key, None)

        active_species = tuple(entry for entry in species_entries if entry.active)
        historical_species = tuple(
            entry for entry in species_entries if not entry.active
        )
        entry_groups: dict[int | None, list[object]] = {}
        for entry in entries:
            entry_groups.setdefault(entry.species_id, []).append(entry)
        rows: list[tuple[str, object]] = [("section_active", active_species)]
        for species in active_species:
            rows.append(("species", species))
            rows.extend(
                ("creature", creature)
                for creature in entry_groups.get(species.species_id, ())
            )
        rows.append(("section_historical", historical_species))
        if self._behavior_report_historical_expanded:
            for species in historical_species:
                rows.append(("species", species))
                rows.extend(
                    ("creature", creature)
                    for creature in entry_groups.get(species.species_id, ())
                )

        footer_height = 34.0
        viewport = arcade.LBWH(
            bounds.left + 8.0,
            bounds.bottom + footer_height,
            bounds.width - 16.0,
            max(0.0, bounds.height - footer_height - header_height),
        )
        row_left = viewport.left + self._SIDEBAR_CARD_OUTER_X
        row_width = max(
            1.0,
            viewport.width
            - self._REPORT_SCROLLBAR_GUTTER
            - self._SIDEBAR_CARD_OUTER_X * 2.0,
        )
        row_layout = tuple(
            (
                kind,
                value,
                *self._behavior_sidebar_row_text(
                    kind,
                    value,
                    paused_automatic_ids,
                ),
            )
            for kind, value in rows
        )
        measured_rows = tuple(
            (
                kind,
                value,
                title,
                detail,
                self._behavior_sidebar_row_height(
                    kind,
                    title,
                    detail,
                    row_width,
                ),
            )
            for kind, value, title, detail in row_layout
        )
        content_height = sum(row[4] for row in measured_rows)
        scroll_limit = max(0.0, content_height - viewport.height)
        scroll_offset = max(
            0.0,
            min(
                scroll_limit,
                self._scroll_offsets.get("behavior_report_creatures", 0.0),
            ),
        )
        self._scroll_regions["behavior_report_creatures"] = viewport
        self._scroll_limits["behavior_report_creatures"] = scroll_limit
        self._scroll_offsets["behavior_report_creatures"] = scroll_offset
        y = viewport.top + scroll_offset
        with self._ui_clip(viewport):
            for kind, value, title, detail, height in measured_rows:
                row = arcade.LBWH(row_left, y - height, row_width, height)
                if kind == "section_active":
                    self._draw_text(
                        "behavior_report_active_species_label",
                        "ACTIVE SPECIES",
                        row.left + 8.0,
                        row.center_y,
                        self.theme.text_muted,
                        8.5,
                        bold=True,
                        anchor_y="center",
                    )
                elif kind == "section_historical":
                    self._control_hitboxes[
                        "behavior_report_species_historical_toggle"
                    ] = row
                    marker = (
                        "▾"
                        if self._behavior_report_historical_expanded
                        else "▸"
                    )
                    self._draw_text(
                        "behavior_report_historical_species_label",
                        f"{marker} HISTORICAL SPECIES · {len(historical_species)}",
                        row.left + 8.0,
                        row.center_y,
                        self.theme.text_muted,
                        8.5,
                        bold=True,
                        anchor_y="center",
                    )
                elif kind == "species":
                    species = value
                    card = arcade.LBWH(
                        row.left,
                        row.bottom + self._SIDEBAR_ROW_GAP,
                        row.width,
                        row.height - self._SIDEBAR_ROW_GAP,
                    )
                    species_key = (
                        "unknown"
                        if species.species_id is None
                        else str(species.species_id)
                    )
                    key = f"behavior_report_species_{species_key}"
                    self._control_hitboxes[key] = card
                    active = (
                        self._behavior_report_species_selected
                        and species.species_id == self._behavior_report_species_id
                    )
                    self._draw_rounded_rect(
                        card,
                        (
                            self.theme.accent_soft
                            if active
                            else self.theme.panel_background
                        ),
                        self.theme.accent if active else self.theme.panel_border,
                        8.0,
                        1.0,
                    )
                    text_width = max(
                        1.0,
                        card.width - self._SIDEBAR_ROW_PADDING_X * 2.0,
                    )
                    title_height = self._wrapped_text_height(
                        title,
                        text_width,
                        10.5,
                        self._SIDEBAR_TITLE_LINE_HEIGHT,
                        bold=True,
                    )
                    self._draw_text(
                        f"{key}_name",
                        title,
                        card.left + self._SIDEBAR_ROW_PADDING_X,
                        card.top - self._SIDEBAR_ROW_PADDING_Y,
                        self.theme.text_primary,
                        10.5,
                        bold=True,
                        width=text_width,
                        multiline=True,
                        anchor_y="top",
                    )
                    self._draw_text(
                        f"{key}_coverage",
                        detail,
                        card.left + self._SIDEBAR_ROW_PADDING_X,
                        card.top
                        - self._SIDEBAR_ROW_PADDING_Y
                        - title_height
                        - self._SIDEBAR_ROW_TEXT_GAP,
                        self.theme.text_muted,
                        8.0,
                        width=text_width,
                        multiline=True,
                        anchor_y="top",
                    )
                else:
                    creature = value
                    card = arcade.LBWH(
                        row.left + 12.0,
                        row.bottom + self._SIDEBAR_ROW_GAP,
                        row.width - 12.0,
                        row.height - self._SIDEBAR_ROW_GAP,
                    )
                    key = f"behavior_report_creature_{creature.creature_id}"
                    self._control_hitboxes[key] = card
                    active = (
                        not self._behavior_report_species_selected
                        and creature.creature_id == self._behavior_report_creature_id
                    )
                    self._draw_rounded_rect(
                        card,
                        (
                            self.theme.accent_soft
                            if active
                            else self.theme.card_background
                        ),
                        self.theme.accent if active else self.theme.panel_border,
                        7.0,
                        0.8,
                    )
                    text_width = max(
                        1.0,
                        card.width - self._SIDEBAR_ROW_PADDING_X * 2.0,
                    )
                    title_height = self._wrapped_text_height(
                        title,
                        text_width,
                        9.5,
                        self._SIDEBAR_TITLE_LINE_HEIGHT,
                        bold=active,
                    )
                    self._draw_text(
                        f"{key}_name",
                        title,
                        card.left + self._SIDEBAR_ROW_PADDING_X,
                        card.top - self._SIDEBAR_ROW_PADDING_Y,
                        self.theme.text_primary,
                        9.5,
                        bold=active,
                        width=text_width,
                        multiline=True,
                        anchor_y="top",
                    )
                    self._draw_text(
                        f"{key}_status",
                        detail,
                        card.left + self._SIDEBAR_ROW_PADDING_X,
                        card.top
                        - self._SIDEBAR_ROW_PADDING_Y
                        - title_height
                        - self._SIDEBAR_ROW_TEXT_GAP,
                        self.theme.text_muted,
                        7.5,
                        width=text_width,
                        multiline=True,
                        anchor_y="top",
                    )
                y -= height
        self._draw_text(
            "behavior_report_creatures_footer",
            f"{len(active_species)} active · {len(historical_species)} historical",
            bounds.left + 16.0,
            bounds.bottom + 15.0,
            self.theme.text_muted,
            8.5,
            anchor_y="center",
        )
        if scroll_limit > 0.0:
            self._draw_scrollbar(viewport, scroll_offset, scroll_limit)

    def _behavior_sidebar_row_text(
        self,
        kind: str,
        value: object,
        paused_automatic_ids: set[int],
    ) -> tuple[str, str]:
        """Return complete title and detail copy for one hierarchy row."""
        if kind == "species":
            species = value
            title = (
                "Unknown species"
                if species.species_id is None
                else f"Species {species.species_id}"
            )
            detail = (
                f"{species.monitored_count}/{species.alive_population} "
                f"monitored · {species.completed_bout_count} bouts"
            )
            return title, detail
        if kind == "creature":
            creature = value
            if creature.active:
                mode = (
                    "FOCAL"
                    if creature.last_observation_mode == "focal"
                    else "AUTO"
                )
                status = f"{mode} · RECORDING"
            elif creature.creature_id in paused_automatic_ids:
                status = "AUTO PAUSED"
            elif creature.deceased:
                status = "DECEASED"
            elif creature.completed_bout_count:
                status = (
                    f"OBSERVED · {creature.completed_bout_count} BOUTS"
                )
            else:
                status = "OBSERVED · NO SUSTAINED BOUTS"
            return str(creature.creature_name), status
        return "", ""

    def _behavior_sidebar_row_height(
        self,
        kind: str,
        title: str,
        detail: str,
        width: float,
    ) -> float:
        """Measure one hierarchy slot including the following card gap."""
        if kind == "section_active":
            return 30.0
        if kind == "section_historical":
            return 34.0
        card_width = max(1.0, width - (12.0 if kind == "creature" else 0.0))
        text_width = max(
            1.0,
            card_width - self._SIDEBAR_ROW_PADDING_X * 2.0,
        )
        title_height = self._wrapped_text_height(
            title,
            text_width,
            9.5 if kind == "creature" else 10.5,
            self._SIDEBAR_TITLE_LINE_HEIGHT,
            bold=True,
        )
        detail_height = self._wrapped_text_height(
            detail,
            text_width,
            7.5 if kind == "creature" else 8.0,
            self._SIDEBAR_DETAIL_LINE_HEIGHT,
        )
        return (
            self._SIDEBAR_ROW_PADDING_Y * 2.0
            + title_height
            + self._SIDEBAR_ROW_TEXT_GAP
            + detail_height
            + self._SIDEBAR_ROW_GAP
        )

    def _behavior_sidebar_header_height(
        self,
        bounds: arcade.Rect,
        *,
        paused: bool,
    ) -> float:
        """Return the measured sidebar-header height for current bounds."""
        text_width = max(
            1.0,
            bounds.width - self._SIDEBAR_PADDING_X * 2.0,
        )
        title_height = self._wrapped_text_height(
            "SPECIES MONITORING",
            text_width,
            self._CARD_TITLE_SIZE,
            self._CARD_TITLE_LINE_HEIGHT,
            bold=True,
        )
        paused_height = (
            self._SIDEBAR_HEADER_GAP + 20.0 if paused else 0.0
        )
        return (
            self._SIDEBAR_HEADER_PADDING_Y
            + title_height
            + paused_height
            + 10.0
        )

    def _draw_report_flat_creature_index(
        self,
        world: World,
        bounds: arcade.Rect,
    ) -> None:
        """Draw a spacious, independently scrollable creature selector."""
        self._draw_rounded_rect(
            bounds,
            self._SECTION_FILL,
            self.theme.panel_border,
            11.0,
            1.0,
        )
        self._draw_text(
            "behavior_report_creatures_title",
            "OBSERVED CREATURES",
            bounds.left + 16.0,
            bounds.top - 14.0,
            self.theme.text_primary,
            self._CARD_TITLE_SIZE,
            bold=True,
            anchor_y="top",
        )
        entries = world.behavior_history_index
        if self._behavior_report_creature_id is None and entries:
            selected = getattr(world, "selected_creature_id", None)
            ids = {entry.creature_id for entry in entries}
            self._behavior_report_creature_id = (
                selected if selected in ids else entries[0].creature_id
            )

        for key in tuple(self._control_hitboxes):
            if key.startswith("behavior_report_creature_"):
                self._control_hitboxes.pop(key, None)

        footer_height = 34.0
        list_viewport = arcade.LBWH(
            bounds.left + 8.0,
            bounds.bottom + footer_height,
            bounds.width - 16.0,
            max(0.0, bounds.height - footer_height - 46.0),
        )
        row_step = 64.0
        row_height = 56.0
        content_height = (
            self._REPORT_SCROLL_PADDING_Y * 2.0
            + len(entries) * row_step
        )
        scroll_limit = max(0.0, content_height - list_viewport.height)
        scroll_offset = max(
            0.0,
            min(
                scroll_limit,
                self._scroll_offsets.get("behavior_report_creatures", 0.0),
            ),
        )
        self._scroll_regions["behavior_report_creatures"] = list_viewport
        self._scroll_limits["behavior_report_creatures"] = scroll_limit
        self._scroll_offsets["behavior_report_creatures"] = scroll_offset

        with self._ui_clip(list_viewport):
            y = (
                list_viewport.top
                + scroll_offset
                - self._REPORT_SCROLL_PADDING_Y
            )
            for entry in entries:
                row = arcade.LBWH(
                    list_viewport.left + self._SIDEBAR_CARD_OUTER_X,
                    y - row_height,
                    max(
                        1.0,
                        list_viewport.width
                        - self._REPORT_SCROLLBAR_GUTTER
                        - self._SIDEBAR_CARD_OUTER_X * 2.0,
                    ),
                    row_height,
                )
                key = f"behavior_report_creature_{entry.creature_id}"
                active = entry.creature_id == self._behavior_report_creature_id
                if self._rect_intersects(row, list_viewport):
                    self._control_hitboxes[key] = row
                    self._draw_rounded_rect(
                        row,
                        self.theme.accent_soft
                        if active
                        else self.theme.panel_background,
                        self.theme.accent
                        if active
                        else self.theme.panel_border,
                        8.0,
                        1.0,
                    )
                    status_color = (
                        self._DECEASED if entry.deceased else self._LIVING
                    )
                    arcade.draw_circle_filled(
                        row.left + 13.0,
                        row.top - 15.0,
                        4.0,
                        status_color,
                    )
                    self._draw_text(
                        f"{key}_name",
                        self._fit_line(entry.creature_name, row.width - 40.0),
                        row.left + 25.0,
                        row.top - 11.0,
                        self.theme.text_primary,
                        10.5,
                        bold=active,
                    )
                    status = "DECEASED" if entry.deceased else "LIVING"
                    badge_width = 62.0 if entry.deceased else 48.0
                    badge = arcade.LBWH(
                        row.left + 25.0,
                        row.bottom + 8.0,
                        badge_width,
                        20.0,
                    )
                    self._draw_rounded_rect(
                        badge,
                        self._DECEASED_SOFT
                        if entry.deceased
                        else self._LIVING_SOFT,
                        status_color,
                        6.0,
                        0.8,
                    )
                    self._draw_text(
                        f"{key}_status",
                        status,
                        badge.center_x,
                        badge.center_y,
                        status_color,
                        7.5,
                        bold=True,
                        anchor_x="center",
                        anchor_y="center",
                    )
                    self._draw_text(
                        f"{key}_bouts",
                        f"{entry.completed_bout_count} bouts",
                        row.right - 10.0,
                        badge.center_y,
                        self.theme.text_muted,
                        8.5,
                        anchor_x="right",
                        anchor_y="center",
                    )
                y -= row_step

        self._draw_text(
            "behavior_report_creatures_footer",
            f"{len(entries)} retained · 1 selected" if entries else "0 retained",
            bounds.left + 16.0,
            bounds.bottom + 15.0,
            self.theme.text_muted,
            8.5,
            anchor_y="center",
        )
        if scroll_limit > 0.0:
            self._draw_scrollbar(list_viewport, scroll_offset, scroll_limit)

    def _selected_behavior_report(
        self,
        world: World,
    ) -> CreatureBehaviorReport | None:
        """Return the prepared report for the modal's selected creature."""
        creature_id = self._behavior_report_creature_id
        if creature_id is None:
            return None
        return world.behavior_report_for(creature_id)

    def _draw_species_report_overview(
        self,
        report: SpeciesBehaviorReport,
        bounds: arcade.Rect,
    ) -> None:
        """Draw coverage and normalized aggregate behavior for one species."""
        self._scroll_regions["behavior_report"] = bounds
        for page in ("timeline", "summary", "why"):
            self._control_hitboxes.pop(
                f"behavior_report_page_{page}",
                None,
            )
        title = (
            "UNKNOWN SPECIES"
            if report.species_id is None
            else f"SPECIES {report.species_id} OVERVIEW"
        )
        header = arcade.LBWH(bounds.left, bounds.top - 112.0, bounds.width, 108.0)
        self._draw_rounded_rect(
            header,
            self._SECTION_FILL,
            self.theme.panel_border,
            10.0,
            1.0,
        )
        self._draw_text(
            "behavior_report_species_overview_title",
            title,
            header.left + 18.0,
            header.top - 20.0,
            self.theme.text_primary,
            14.0,
            bold=True,
        )
        hours = report.total_observation_seconds / 3600.0
        coverage_lines = (
            f"{report.monitored_count}/{report.alive_population} "
            "currently monitored",
            f"{report.observed_creature_count} observed creatures",
            f"{self._format_report_duration(report.total_observation_seconds)} "
            "observed creature-time",
            f"{report.completed_bout_count} completed bouts",
        )
        for index, line in enumerate(coverage_lines):
            self._draw_text(
                f"behavior_report_species_coverage_{index}",
                line,
                header.left + 18.0 + index * (header.width / 4.0),
                header.bottom + 25.0,
                self.theme.text_muted,
                9.5,
                bold=True,
            )
        if not report.behaviors:
            self._scroll_limits["behavior_report"] = 0.0
            self._scroll_offsets["behavior_report"] = 0.0
            self._draw_report_empty_body(
                arcade.LBWH(
                    bounds.left,
                    bounds.bottom,
                    bounds.width,
                    max(0.0, header.bottom - bounds.bottom - 12.0),
                ),
                "Representatives are being observed; no sustained bouts "
                "have completed yet.",
            )
            return
        grid = arcade.LBWH(
            bounds.left,
            bounds.bottom,
            bounds.width,
            max(0.0, header.bottom - bounds.bottom - 12.0),
        )
        content_grid = self._report_scroll_content_bounds(grid)
        columns = (
            3
            if content_grid.width >= 720.0
            else 2 if content_grid.width >= 460.0 else 1
        )
        gap = 12.0
        card_height = 126.0
        rows = (len(report.behaviors) + columns - 1) // columns
        content_height = (
            self._REPORT_SCROLL_PADDING_Y * 2.0
            + rows * card_height
            + max(0, rows - 1) * gap
        )
        scroll_offset = self._prepare_report_scroll(grid, content_height)
        card_width = (
            content_grid.width - gap * (columns - 1)
        ) / columns
        with self._ui_clip(grid):
            for index, item in enumerate(report.behaviors):
                row = index // columns
                column = index % columns
                card = arcade.LBWH(
                    content_grid.left + column * (card_width + gap),
                    content_grid.top
                    + scroll_offset
                    - self._REPORT_SCROLL_PADDING_Y
                    - row * (card_height + gap)
                    - card_height,
                    card_width,
                    card_height,
                )
                if not self._rect_intersects(card, grid):
                    continue
                color = self._BEHAVIOR_COLORS[item.behavior]
                self._draw_rounded_rect(
                    card,
                    self._SECTION_FILL,
                    self.theme.panel_border,
                    10.0,
                    1.0,
                )
                arcade.draw_circle_filled(
                    card.left + 16.0,
                    card.top - 21.0,
                    4.0,
                    color,
                )
                prefix = f"behavior_report_species_summary_{item.behavior.value}"
                self._draw_text(
                    f"{prefix}_title",
                    item.behavior.value.replace("_", " ").upper(),
                    card.left + 28.0,
                    card.top - 16.0,
                    self.theme.text_primary,
                    10.5,
                    bold=True,
                )
                values = (
                    f"{item.completed_bout_count} bouts",
                    f"{self._format_report_duration(item.total_duration)} "
                    f"total · {self._format_report_duration(item.median_duration)} "
                    "median",
                    (
                        f"{item.bouts_per_creature_hour:.1f} bouts / observed "
                        "creature-hour"
                        if hours > 0.0
                        else "Rate unavailable"
                    ),
                )
                for value_index, value in enumerate(values):
                    self._draw_text(
                        f"{prefix}_{value_index}",
                        self._fit_line(value, card.width - 28.0),
                        card.left + 14.0,
                        card.top - 52.0 - value_index * 24.0,
                        self.theme.text_muted,
                        8.8,
                    )
        if self._scroll_limits["behavior_report"] > 0.0:
            self._draw_scrollbar(
                grid,
                scroll_offset,
                self._scroll_limits["behavior_report"],
            )

    def _draw_report_content(
        self,
        report: CreatureBehaviorReport,
        bounds: arcade.Rect,
    ) -> None:
        """Draw the warning, text tabs, and selected report page."""
        top = bounds.top
        if report.history_incomplete:
            warning = arcade.LBWH(bounds.left, top - 56.0, bounds.width, 52.0)
            self._draw_rounded_rect(
                warning,
                (255, 242, 218),
                (190, 118, 32),
                8.0,
                1.0,
            )
            self._draw_text(
                "behavior_report_incomplete",
                "History incomplete: some completed bouts were not retained",
                warning.left + 14.0,
                warning.top - 15.0,
                (128, 72, 18),
                9.0,
                bold=True,
            )
            self._draw_text(
                "behavior_report_incomplete_reason",
                "The history consumer was unavailable.",
                warning.left + 14.0,
                warning.bottom + 10.0,
                (128, 72, 18),
                8.5,
            )
            self._draw_text(
                "behavior_report_incomplete_count",
                f"{report.history_completions_not_recorded} completed bouts "
                "were not recorded.",
                warning.right - 14.0,
                warning.bottom + 10.0,
                (128, 72, 18),
                8.5,
                anchor_x="right",
            )
            top = warning.bottom - 8.0

        tabs = arcade.LBWH(bounds.left, top - 48.0, bounds.width, 44.0)
        arcade.draw_line(
            tabs.left,
            tabs.bottom,
            tabs.right,
            tabs.bottom,
            self.theme.panel_border,
            1.0,
        )
        tab_width = tabs.width / 3.0
        for index, page in enumerate(("timeline", "summary", "why")):
            tab = arcade.LBWH(
                tabs.left + index * tab_width,
                tabs.bottom,
                tab_width,
                tabs.height,
            )
            key = f"behavior_report_page_{page}"
            self._control_hitboxes[key] = tab
            active = page == self._behavior_report_page
            self._draw_text(
                f"{key}_label",
                page.upper(),
                tab.center_x,
                tab.center_y + 2.0,
                self.theme.accent if active else self.theme.text_muted,
                10.5,
                bold=active,
                anchor_x="center",
                anchor_y="center",
            )
            if active:
                arcade.draw_line(
                    tab.left + 28.0,
                    tab.bottom + 1.5,
                    tab.right - 28.0,
                    tab.bottom + 1.5,
                    self.theme.accent,
                    3.0,
                )

        body = arcade.LBWH(
            bounds.left,
            bounds.bottom,
            bounds.width,
            max(0.0, tabs.bottom - bounds.bottom - 10.0),
        )
        self._scroll_regions["behavior_report"] = body
        if self._behavior_report_page == "summary":
            self._draw_report_summary(report, body)
        elif self._behavior_report_page == "why":
            self._draw_report_why(report, body)
        else:
            self._scroll_limits["behavior_report"] = 0.0
            self._scroll_offsets["behavior_report"] = 0.0
            self._draw_report_timeline(report, body)

    def _draw_report_identity_strip(
        self,
        report: CreatureBehaviorReport,
        bounds: arcade.Rect,
    ) -> None:
        """Draw direct report identity and retention values."""
        self._draw_rounded_rect(
            bounds,
            self._SECTION_FILL,
            self.theme.panel_border,
            10.0,
            1.0,
        )
        status_color = self._DECEASED if report.deceased else self._LIVING
        status_fill = self._DECEASED_SOFT if report.deceased else self._LIVING_SOFT
        status = "DECEASED" if report.deceased else "LIVING"
        self._draw_text(
            "behavior_report_identity_name",
            self._fit_line(report.creature_name, bounds.width * 0.32),
            bounds.left + 16.0,
            bounds.center_y + 10.0,
            self.theme.text_primary,
            13.0,
            bold=True,
            anchor_y="center",
        )
        badge = arcade.LBWH(
            bounds.left + 16.0,
            bounds.center_y - 20.0,
            72.0,
            19.0,
        )
        self._draw_rounded_rect(badge, status_fill, status_color, 7.0, 0.8)
        self._draw_text(
            "behavior_report_identity_status",
            status,
            badge.center_x,
            badge.center_y,
            status_color,
            8.0,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )
        columns = (
            ("COMPLETED BOUTS", str(report.summary.completed_bout_count)),
            ("DETAILED BOUTS", str(len(report.completed_bouts))),
            ("DROPPED DETAILS", str(report.detailed_bouts_dropped)),
        )
        column_left = bounds.left + bounds.width * 0.39
        column_width = (bounds.right - column_left - 12.0) / len(columns)
        for index, (label, value) in enumerate(columns):
            center_x = column_left + column_width * (index + 0.5)
            if index:
                divider_x = column_left + column_width * index
                arcade.draw_line(
                    divider_x,
                    bounds.bottom + 12.0,
                    divider_x,
                    bounds.top - 12.0,
                    self.theme.panel_border,
                    1.0,
                )
            self._draw_text(
                f"behavior_report_identity_label_{index}",
                label,
                center_x,
                bounds.center_y + 10.0,
                self.theme.text_muted,
                8.0,
                anchor_x="center",
                anchor_y="center",
            )
            self._draw_text(
                f"behavior_report_identity_value_{index}",
                value,
                center_x,
                bounds.center_y - 11.0,
                self.theme.text_primary,
                12.0,
                bold=True,
                anchor_x="center",
                anchor_y="center",
            )

    def _draw_report_timeline(
        self,
        report: CreatureBehaviorReport,
        bounds: arcade.Rect,
    ) -> None:
        """Draw overview, completed-bout lanes, and selected-bout details."""
        bouts = report.completed_bouts
        if not bouts:
            self._draw_report_empty_body(bounds)
            return
        identity = arcade.LBWH(
            bounds.left,
            bounds.top - 72.0,
            bounds.width,
            68.0,
        )
        self._draw_report_identity_strip(report, identity)
        detail_height = min(202.0, max(142.0, bounds.height * 0.31))
        detail = arcade.LBWH(
            bounds.left,
            bounds.bottom,
            bounds.width,
            detail_height,
        )
        chart = arcade.LBWH(
            bounds.left,
            detail.top + 12.0,
            bounds.width,
            max(120.0, identity.bottom - detail.top - 24.0),
        )
        self._draw_timeline_chart(report, chart)
        selected = next(
            (
                bout
                for bout in bouts
                if bout.bout_id == self._behavior_report_selected_bout_id
            ),
            None,
        )
        if selected is None:
            self._draw_rounded_rect(
                detail,
                self._SECTION_FILL,
                self.theme.panel_border,
                10.0,
                1.0,
            )
            self._draw_text(
                "behavior_report_bout_detail_placeholder_title",
                "SELECTED BOUT DETAILS",
                detail.left + 16.0,
                detail.top - 14.0,
                self.theme.text_primary,
                self._CARD_TITLE_SIZE,
                bold=True,
                anchor_y="top",
            )
            self._draw_text(
                "behavior_report_bout_detail_placeholder",
                "Select a completed bout on the timeline to inspect its "
                "evidence, outcome, and WHY summary.",
                detail.left + 16.0,
                detail.center_y,
                self.theme.text_muted,
                10.0,
                anchor_y="center",
            )
        else:
            self._draw_bout_detail(selected, detail)

    def _draw_timeline_chart(
        self,
        report: CreatureBehaviorReport,
        bounds: arcade.Rect,
    ) -> None:
        """Draw the sole graph in the report: immutable completed bouts."""
        bouts = report.completed_bouts
        start = min(bout.start_time for bout in bouts)
        end = max(bout.end_time for bout in bouts)
        span = max(1e-9, end - start)
        self._draw_rounded_rect(
            bounds,
            self._SECTION_FILL,
            self.theme.panel_border,
            10.0,
            1.0,
        )
        self._draw_text(
            "behavior_report_timeline_heading",
            "FULL HISTORY",
            bounds.left + 16.0,
            bounds.top - 14.0,
            self.theme.text_primary,
            self._CARD_TITLE_SIZE,
            bold=True,
            anchor_y="top",
        )
        self._draw_text(
            "behavior_report_timeline_range",
            f"{start:.1f}s — {end:.1f}s",
            bounds.right - 16.0,
            bounds.top - 16.0,
            self.theme.text_muted,
            self._CARD_BODY_SIZE,
            anchor_x="right",
            anchor_y="top",
        )
        lane_label_width = min(164.0, max(112.0, bounds.width * 0.17))
        graph_left = bounds.left + lane_label_width
        graph_right = bounds.right - 14.0
        graph_width = max(40.0, graph_right - graph_left)
        ruler_y = bounds.top - 42.0
        for tick_index in range(5):
            fraction = tick_index / 4.0
            tick_x = graph_left + fraction * graph_width
            tick_time = start + fraction * span
            arcade.draw_line(
                tick_x,
                ruler_y - 3.0,
                tick_x,
                ruler_y + 3.0,
                self.theme.panel_border,
                1.0,
            )
            self._draw_text(
                f"behavior_report_timeline_tick_{tick_index}",
                f"{tick_time:.1f}s",
                tick_x,
                ruler_y + 7.0,
                self.theme.text_muted,
                7.5,
                anchor_x=(
                    "left"
                    if tick_index == 0
                    else "right"
                    if tick_index == 4
                    else "center"
                ),
            )
        lane_height = max(20.0, (bounds.height - 56.0) / len(BehaviorKind))
        y = ruler_y - 10.0
        for behavior in BehaviorKind:
            color = self._BEHAVIOR_COLORS[behavior]
            lane = arcade.LBWH(
                graph_left,
                y - lane_height,
                graph_width,
                max(1.0, lane_height - 4.0),
            )
            self._draw_rounded_rect(
                lane,
                self.theme.panel_background,
                self.theme.panel_background,
                5.0,
                0.0,
            )
            arcade.draw_circle_filled(
                bounds.left + 18.0,
                y - lane_height / 2.0,
                4.0,
                color,
            )
            self._draw_text(
                f"behavior_report_lane_{behavior.value}",
                behavior.value.replace("_", " ").title(),
                bounds.left + 30.0,
                y - lane_height / 2.0,
                self.theme.text_muted,
                8.5,
                anchor_y="center",
            )
            for bout in bouts:
                if bout.behavior is not behavior:
                    continue
                left = graph_left + (bout.start_time - start) / span * graph_width
                right = graph_left + (bout.end_time - start) / span * graph_width
                bar = arcade.LBWH(
                    left,
                    lane.bottom + 5.0,
                    max(4.0, right - left),
                    max(8.0, lane.height - 10.0),
                )
                key = f"behavior_report_bout_{bout.bout_id}"
                self._control_hitboxes[key] = bar
                self._draw_rounded_rect(
                    bar,
                    color,
                    self.theme.selected_outline
                    if bout.bout_id == self._behavior_report_selected_bout_id
                    else color,
                    5.0,
                    1.5,
                )
            y -= lane_height

    def _draw_bout_detail(
        self,
        bout: CompletedBehaviorBout,
        bounds: arcade.Rect,
    ) -> None:
        """Draw every immutable evidence, outcome, and WHY detail row."""
        detail = arcade.LBWH(
            bounds.left,
            bounds.bottom,
            bounds.width,
            bounds.height,
        )
        self._draw_rounded_rect(
            detail,
            self._SECTION_FILL,
            self.theme.panel_border,
            10.0,
            1.0,
        )
        color = self._BEHAVIOR_COLORS[bout.behavior]
        arcade.draw_circle_filled(
            detail.left + 18.0,
            detail.top - 20.0,
            4.0,
            color,
        )
        self._draw_text(
            "behavior_report_bout_detail_title",
            f"{bout.behavior.value.replace('_', ' ').title()} #{bout.bout_id} · "
            f"{bout.start_time:.1f}–{bout.end_time:.1f}s · {bout.duration:.1f}s",
            detail.left + 30.0,
            detail.top - 14.0,
            self.theme.text_primary,
            self._CARD_TITLE_SIZE,
            bold=True,
            anchor_y="top",
        )
        left = detail.left + 16.0
        right = detail.center_x + 10.0
        column_width = detail.width / 2.0 - 34.0
        self._draw_text(
            "behavior_report_bout_detail_evidence_title",
            "OBSERVED EVIDENCE",
            left,
            detail.top - 46.0,
            self.theme.text_primary,
            10.0,
            bold=True,
        )
        evidence_y = detail.top - 68.0
        if not bout.evidence_summary:
            self._draw_text(
                "behavior_report_bout_detail_evidence_empty",
                "No summarized evidence",
                left,
                evidence_y,
                self.theme.text_muted,
                8.5,
            )
        for index, item in enumerate(bout.evidence_summary):
            unit = f" {item.unit}" if item.unit else ""
            estimated = " · estimate" if item.quantiles_estimated else ""
            central_value = (
                f"total {item.total_value:.2f}{unit}"
                if item.key in {"food_consumption_event", "energy_swallowed"}
                else f"median {item.median_value:.2f}{unit}"
            )
            line = (
                f"{item.label}: {central_value} · "
                f"{item.passed_count}/{item.sample_count} pass{estimated}"
            )
            self._draw_text(
                f"behavior_report_bout_detail_evidence_{index}",
                self._fit_line(line, column_width),
                left,
                evidence_y - index * 18.0,
                self.theme.text_muted,
                8.0,
            )
        outcome = (
            "—"
            if bout.outcome is None
            else bout.outcome.value.replace("_", " ").title()
        )
        self._draw_text(
            "behavior_report_bout_detail_outcome",
            f"Outcome: {outcome} · Termination: "
            f"{bout.termination.value.replace('_', ' ').title()}",
            right,
            detail.top - 48.0,
            self.theme.text_primary,
            10.0,
            bold=True,
        )
        why_y = detail.top - 68.0
        effects = () if bout.why_summary is None else bout.why_summary.effects
        if not effects:
            self._draw_text(
                "behavior_report_bout_detail_why_empty",
                "WHY unavailable for this bout",
                right,
                why_y,
                self.theme.text_muted,
                8.5,
            )
        for index, effect in enumerate(effects):
            spread = self._quartile_spread_text(
                effect.p25,
                effect.p75,
                estimated=effect.quantiles_estimated,
            )
            line = (
                f"{effect.intervention.value.replace('_', ' ').title()}: "
                f"{effect.median_influence:.2f} · "
                f"{effect.influence_label.value.upper()} · "
                f"{effect.effect_direction.value.upper()} · "
                f"{effect.sample_count} probes{spread}"
            )
            self._draw_text(
                f"behavior_report_bout_detail_why_{index}",
                self._fit_line(line, column_width),
                right,
                why_y - index * 18.0,
                self.theme.text_muted,
                8.0,
            )

    def _draw_report_summary(
        self,
        report: CreatureBehaviorReport,
        bounds: arcade.Rect,
    ) -> None:
        """Draw direct per-behaviour aggregates in a card grid."""
        behaviors = report.summary.behaviors
        if not behaviors:
            self._scroll_limits["behavior_report"] = 0.0
            self._scroll_offsets["behavior_report"] = 0.0
            self._draw_report_empty_body(bounds)
            return
        identity = arcade.LBWH(
            bounds.left,
            bounds.top - 72.0,
            bounds.width,
            68.0,
        )
        self._draw_report_identity_strip(report, identity)
        grid = arcade.LBWH(
            bounds.left,
            bounds.bottom,
            bounds.width,
            max(0.0, identity.bottom - bounds.bottom - 12.0),
        )
        content_grid = self._report_scroll_content_bounds(grid)
        columns = (
            3
            if content_grid.width >= 720.0
            else 2 if content_grid.width >= 460.0 else 1
        )
        gap = 12.0
        card_width = (
            content_grid.width - gap * (columns - 1)
        ) / columns
        row_heights = tuple(
            max(
                self._summary_card_height(item, card_width)
                for item in behaviors[row_start : row_start + columns]
            )
            for row_start in range(0, len(behaviors), columns)
        )
        content_height = (
            self._REPORT_SCROLL_PADDING_Y * 2.0
            + sum(row_heights)
            + max(0, len(row_heights) - 1) * gap
        )
        scroll_offset = self._prepare_report_scroll(grid, content_height)
        row_tops: list[float] = []
        row_top = (
            content_grid.top
            + scroll_offset
            - self._REPORT_SCROLL_PADDING_Y
        )
        for row_height in row_heights:
            row_tops.append(row_top)
            row_top -= row_height + gap
        with self._ui_clip(grid):
            for index, item in enumerate(behaviors):
                row = index // columns
                column = index % columns
                card_height = row_heights[row]
                card = arcade.LBWH(
                    content_grid.left + column * (card_width + gap),
                    row_tops[row] - card_height,
                    card_width,
                    card_height,
                )
                if self._rect_intersects(card, grid):
                    self._draw_summary_card(item, card)
        if self._scroll_limits["behavior_report"] > 0.0:
            self._draw_scrollbar(
                grid,
                scroll_offset,
                self._scroll_limits["behavior_report"],
            )

    def _draw_summary_card(
        self,
        item: BehaviorLifetimeSummary,
        bounds: arcade.Rect,
    ) -> None:
        """Draw one behavior summary without deriving new metrics."""
        color = self._BEHAVIOR_COLORS[item.behavior]
        self._draw_rounded_rect(
            bounds,
            self._SECTION_FILL,
            self.theme.panel_border,
            10.0,
            1.0,
        )
        arcade.draw_circle_filled(
            bounds.left + 16.0,
            bounds.top - 21.0,
            4.0,
            color,
        )
        title = item.behavior.value.replace("_", " ").upper()
        title_width = max(1.0, bounds.width - 46.0)
        title_height = self._wrapped_text_height(
            title,
            title_width,
            self._CARD_TITLE_SIZE,
            self._CARD_TITLE_LINE_HEIGHT,
            bold=True,
        )
        self._draw_text(
            f"behavior_report_summary_{item.behavior.value}",
            title,
            bounds.left + 30.0,
            bounds.top - self._CARD_PADDING_Y,
            self.theme.text_primary,
            self._CARD_TITLE_SIZE,
            bold=True,
            width=title_width,
            multiline=True,
            anchor_y="top",
        )
        cursor = (
            bounds.top
            - self._CARD_PADDING_Y
            - title_height
            - self._CARD_TITLE_GAP
        )
        self._draw_report_value_row(
            f"behavior_report_summary_count_{item.behavior.value}",
            "Completed bouts",
            str(item.completed_bout_count),
            bounds.left + self._CARD_PADDING_X,
            bounds.right - self._CARD_PADDING_X,
            cursor,
        )
        cursor -= 22.0
        self._draw_report_value_row(
            f"behavior_report_summary_total_{item.behavior.value}",
            "Total duration",
            f"{item.total_duration:.1f}s",
            bounds.left + self._CARD_PADDING_X,
            bounds.right - self._CARD_PADDING_X,
            cursor,
        )
        cursor -= 22.0
        self._draw_report_value_row(
            f"behavior_report_summary_median_{item.behavior.value}",
            "Median duration",
            f"{item.median_duration:.1f}s",
            bounds.left + self._CARD_PADDING_X,
            bounds.right - self._CARD_PADDING_X,
            cursor,
        )
        cursor -= self._CARD_BODY_LINE_HEIGHT + 12.0
        outcomes = ", ".join(
            f"{count} {outcome.value.replace('_', ' ')}"
            for outcome, count in item.outcome_counts
        )
        outcome_text = (
            f"Outcomes: {outcomes}" if outcomes else "Outcomes: None"
        )
        self._draw_text(
            f"behavior_report_outcomes_{item.behavior.value}",
            outcome_text,
            bounds.left + self._CARD_PADDING_X,
            cursor,
            self.theme.text_muted,
            self._CARD_BODY_SIZE,
            width=max(1.0, bounds.width - self._CARD_PADDING_X * 2.0),
            multiline=True,
            anchor_y="top",
        )

    def _summary_card_height(
        self,
        item: BehaviorLifetimeSummary,
        width: float,
    ) -> float:
        """Return the measured height needed by one summary card."""
        title = item.behavior.value.replace("_", " ").upper()
        outcomes = ", ".join(
            f"{count} {outcome.value.replace('_', ' ')}"
            for outcome, count in item.outcome_counts
        )
        outcome_text = f"Outcomes: {outcomes}" if outcomes else "Outcomes: None"
        title_height = self._wrapped_text_height(
            title,
            max(1.0, width - 46.0),
            self._CARD_TITLE_SIZE,
            self._CARD_TITLE_LINE_HEIGHT,
            bold=True,
        )
        outcome_height = self._wrapped_text_height(
            outcome_text,
            max(1.0, width - self._CARD_PADDING_X * 2.0),
            self._CARD_BODY_SIZE,
            self._CARD_BODY_LINE_HEIGHT,
        )
        metric_height = self._CARD_BODY_LINE_HEIGHT * 3.0 + 7.0 * 2.0
        return (
            self._CARD_PADDING_Y
            + title_height
            + self._CARD_TITLE_GAP
            + metric_height
            + 12.0
            + outcome_height
            + self._CARD_PADDING_Y
        )

    def _report_why_behaviors(
        self,
        report: CreatureBehaviorReport,
    ) -> tuple[BehaviorLifetimeSummary, ...]:
        """Return existing behavior summaries which contain WHY data."""
        return tuple(
            behavior
            for behavior in report.summary.behaviors
            if behavior.why_summaries
        )

    def _draw_report_why(
        self,
        report: CreatureBehaviorReport,
        bounds: arcade.Rect,
    ) -> None:
        """Draw one selected behavior's lifetime WHY summaries as cards."""
        behaviors = self._report_why_behaviors(report)
        if not behaviors:
            self._scroll_limits["behavior_report"] = 0.0
            self._scroll_offsets["behavior_report"] = 0.0
            self._draw_report_empty_body(
                bounds,
                (
                    "WHY is available only for explicitly selected focal observation."
                    if report.last_observation_mode == "automatic"
                    else "No completed bouts contain valid WHY data."
                ),
            )
            return
        values = {behavior.behavior.value for behavior in behaviors}
        if self._behavior_report_why_behavior not in values:
            self._behavior_report_why_behavior = behaviors[0].behavior.value
        selected = next(
            behavior
            for behavior in behaviors
            if behavior.behavior.value == self._behavior_report_why_behavior
        )

        identity = arcade.LBWH(
            bounds.left,
            bounds.top - 72.0,
            bounds.width,
            68.0,
        )
        self._draw_report_identity_strip(report, identity)
        selector = arcade.LBWH(
            bounds.left,
            identity.bottom - 52.0,
            bounds.width,
            40.0,
        )
        gap = 8.0
        tab_width = (selector.width - gap * (len(behaviors) - 1)) / len(behaviors)
        for key in tuple(self._control_hitboxes):
            if key.startswith("behavior_report_why_behavior_"):
                self._control_hitboxes.pop(key, None)
        for index, behavior in enumerate(behaviors):
            tab = arcade.LBWH(
                selector.left + index * (tab_width + gap),
                selector.bottom,
                tab_width,
                selector.height,
            )
            key = f"behavior_report_why_behavior_{behavior.behavior.value}"
            self._control_hitboxes[key] = tab
            active = behavior is selected
            color = self._BEHAVIOR_COLORS[behavior.behavior]
            self._draw_rounded_rect(
                tab,
                self.theme.accent_soft if active else self._SECTION_FILL,
                color if active else self.theme.panel_border,
                8.0,
                1.0,
            )
            self._draw_text(
                f"{key}_label",
                self._fit_line(
                    behavior.behavior.value.replace("_", " ").title(),
                    tab.width - 12.0,
                ),
                tab.center_x,
                tab.center_y,
                self.theme.text_primary,
                self._CARD_BODY_SIZE,
                bold=active,
                anchor_x="center",
                anchor_y="center",
            )

        why_meta = arcade.LBWH(
            bounds.left,
            selector.bottom - 32.0,
            bounds.width,
            22.0,
        )
        self._draw_text(
            "behavior_report_why_basis",
            f"{selected.completed_bout_count} completed behavior bouts · "
            "stable pattern threshold "
            f"{report.summary.stable_pattern_threshold}",
            why_meta.left + 4.0,
            why_meta.center_y,
            self.theme.text_muted,
            self._CARD_BODY_SIZE,
            anchor_y="center",
        )
        cards = arcade.LBWH(
            bounds.left,
            bounds.bottom,
            bounds.width,
            max(0.0, why_meta.bottom - bounds.bottom - 8.0),
        )
        content_cards = self._report_scroll_content_bounds(cards)
        effects = selected.why_summaries
        columns = (
            3
            if content_cards.width >= 720.0
            else 2 if content_cards.width >= 460.0 else 1
        )
        card_gap = 12.0
        card_width = (
            content_cards.width - card_gap * (columns - 1)
        ) / columns
        row_heights = tuple(
            max(
                self._why_card_height(report, effect, card_width)
                for effect in effects[row_start : row_start + columns]
            )
            for row_start in range(0, len(effects), columns)
        )
        content_height = (
            self._REPORT_SCROLL_PADDING_Y * 2.0
            + sum(row_heights)
            + max(0, len(row_heights) - 1) * card_gap
        )
        scroll_offset = self._prepare_report_scroll(cards, content_height)
        row_tops: list[float] = []
        row_top = (
            content_cards.top
            + scroll_offset
            - self._REPORT_SCROLL_PADDING_Y
        )
        for row_height in row_heights:
            row_tops.append(row_top)
            row_top -= row_height + card_gap
        with self._ui_clip(cards):
            for index, effect in enumerate(effects):
                row = index // columns
                column = index % columns
                card_height = row_heights[row]
                card = arcade.LBWH(
                    content_cards.left + column * (card_width + card_gap),
                    row_tops[row] - card_height,
                    card_width,
                    card_height,
                )
                if self._rect_intersects(card, cards):
                    self._draw_why_card(report, selected, effect, card)
        if self._scroll_limits["behavior_report"] > 0.0:
            self._draw_scrollbar(
                cards,
                scroll_offset,
                self._scroll_limits["behavior_report"],
            )

    def _draw_why_card(
        self,
        report: CreatureBehaviorReport,
        behavior: BehaviorLifetimeSummary,
        effect: BehaviorLifetimeWhySummary,
        bounds: arcade.Rect,
    ) -> None:
        """Draw one existing intervention summary without mini graphs."""
        color = self._BEHAVIOR_COLORS[behavior.behavior]
        prefix = (
            f"{behavior.behavior.value}_{effect.intervention.value}"
        )
        self._draw_rounded_rect(
            bounds,
            self._SECTION_FILL,
            self.theme.panel_border,
            10.0,
            1.0,
        )
        arcade.draw_circle_filled(
            bounds.left + 16.0,
            bounds.top - 21.0,
            4.0,
            color,
        )
        title, median, available, wording = self._why_card_text(
            report,
            effect,
        )
        title_width = max(1.0, bounds.width - 46.0)
        body_width = max(1.0, bounds.width - self._CARD_PADDING_X * 2.0)
        title_height = self._wrapped_text_height(
            title,
            title_width,
            self._CARD_TITLE_SIZE,
            self._CARD_TITLE_LINE_HEIGHT,
            bold=True,
        )
        self._draw_text(
            f"behavior_report_why_intervention_{prefix}",
            title,
            bounds.left + 30.0,
            bounds.top - self._CARD_PADDING_Y,
            self.theme.text_primary,
            self._CARD_TITLE_SIZE,
            bold=True,
            width=title_width,
            multiline=True,
            anchor_y="top",
        )
        cursor = (
            bounds.top
            - self._CARD_PADDING_Y
            - title_height
            - self._CARD_TITLE_GAP
        )
        median_height = self._wrapped_text_height(
            median,
            body_width,
            self._CARD_BODY_SIZE,
            self._CARD_BODY_LINE_HEIGHT,
        )
        self._draw_text(
            f"behavior_report_why_median_{prefix}",
            median,
            bounds.left + self._CARD_PADDING_X,
            cursor,
            self.theme.text_primary,
            self._CARD_BODY_SIZE,
            width=body_width,
            multiline=True,
            anchor_y="top",
        )
        cursor -= median_height + 8.0
        available_height = self._wrapped_text_height(
            available,
            body_width,
            self._CARD_BODY_SIZE,
            self._CARD_BODY_LINE_HEIGHT,
        )
        self._draw_text(
            f"behavior_report_why_available_{prefix}",
            available,
            bounds.left + self._CARD_PADDING_X,
            cursor,
            self.theme.text_muted,
            self._CARD_BODY_SIZE,
            width=body_width,
            multiline=True,
            anchor_y="top",
        )
        cursor -= available_height + 8.0
        pattern_height = self._wrapped_text_height(
            wording,
            body_width,
            self._CARD_BODY_SIZE,
            self._CARD_BODY_LINE_HEIGHT,
        )
        self._draw_text(
            f"behavior_report_why_pattern_{prefix}",
            wording,
            bounds.left + self._CARD_PADDING_X,
            cursor,
            self.theme.text_muted,
            self._CARD_BODY_SIZE,
            width=body_width,
            multiline=True,
            anchor_y="top",
        )
        directions = self._direction_lines(effect)
        cursor -= pattern_height + (10.0 if directions else 0.0)
        for index, direction in enumerate(directions):
            direction_height = self._wrapped_text_height(
                direction,
                body_width,
                self._CARD_BODY_SIZE,
                self._CARD_BODY_LINE_HEIGHT,
            )
            self._draw_text(
                f"behavior_report_why_direction_{prefix}_{index}",
                direction,
                bounds.left + self._CARD_PADDING_X,
                cursor,
                self.theme.text_muted,
                self._CARD_BODY_SIZE,
                width=body_width,
                multiline=True,
                anchor_y="top",
            )
            cursor -= direction_height + 4.0

    def _why_card_height(
        self,
        report: CreatureBehaviorReport,
        effect: BehaviorLifetimeWhySummary,
        width: float,
    ) -> float:
        """Return the measured height needed by one WHY card."""
        title, median, available, wording = self._why_card_text(report, effect)
        title_height = self._wrapped_text_height(
            title,
            max(1.0, width - 46.0),
            self._CARD_TITLE_SIZE,
            self._CARD_TITLE_LINE_HEIGHT,
            bold=True,
        )
        body_width = max(1.0, width - self._CARD_PADDING_X * 2.0)
        body_heights = tuple(
            self._wrapped_text_height(
                text,
                body_width,
                self._CARD_BODY_SIZE,
                self._CARD_BODY_LINE_HEIGHT,
            )
            for text in (median, available, wording)
        )
        directions = self._direction_lines(effect)
        direction_heights = tuple(
            self._wrapped_text_height(
                direction,
                body_width,
                self._CARD_BODY_SIZE,
                self._CARD_BODY_LINE_HEIGHT,
            )
            for direction in directions
        )
        return (
            self._CARD_PADDING_Y
            + title_height
            + self._CARD_TITLE_GAP
            + sum(body_heights)
            + 8.0 * 2.0
            + (10.0 if directions else 0.0)
            + sum(direction_heights)
            + max(0, len(direction_heights) - 1) * 4.0
            + self._CARD_PADDING_Y
        )

    def _why_card_text(
        self,
        report: CreatureBehaviorReport,
        effect: BehaviorLifetimeWhySummary,
    ) -> tuple[str, str, str, str]:
        """Return the unchanged stored-value copy used by a WHY card."""
        title = effect.intervention.value.replace("_", " ").title()
        spread = self._quartile_spread_text(
            effect.p25,
            effect.p75,
            estimated=effect.quantiles_estimated,
        )
        median = (
            "Median bout influence: "
            f"{effect.median_bout_influence:.2f} · "
            f"{effect.influence_label.value.upper()}{spread}"
        )
        available = (
            "WHY available in "
            f"{effect.contributing_bout_count} / "
            f"{effect.behavior_bout_count} bouts"
        )
        stable = (
            effect.contributing_bout_count
            >= report.summary.stable_pattern_threshold
        )
        wording = (
            "Typical neural influence · based on "
            f"{effect.contributing_bout_count} completed bouts"
            if stable
            else (
                "Based on "
                f"{effect.contributing_bout_count} completed bout(s) · "
                "insufficient history for a stable pattern"
            )
        )
        return title, median, available, wording

    @staticmethod
    def _quartile_spread_text(
        p25: float | None,
        p75: float | None,
        *,
        estimated: bool,
    ) -> str:
        """Format quartile endpoints and their conventional IQR width."""
        if p25 is None or p75 is None:
            return " · estimated" if estimated else ""
        estimate = " · estimated" if estimated else ""
        return (
            f" · Q1–Q3 {p25:.2f}–{p75:.2f}"
            f" · IQR {max(0.0, p75 - p25):.2f}{estimate}"
        )

    def _wrapped_text_height(
        self,
        text: str,
        width: float,
        font_size: float,
        line_height: float,
        *,
        bold: bool = False,
    ) -> float:
        """Measure a wrapped block with the same metrics used for drawing."""
        scale = self._BEHAVIOR_REPORT_FONT_SCALE
        lines = self._painter.wrap_text(
            text,
            max(1.0, width),
            font_size * scale,
            bold=bold,
        )
        return max(1, len(lines)) * line_height * scale

    def _draw_report_value_row(
        self,
        key: str,
        label: str,
        value: str,
        left: float,
        right: float,
        y: float,
    ) -> None:
        """Draw a label/value pair with consistent opposing alignment."""
        self._draw_text(
            f"{key}_label",
            label,
            left,
            y,
            self.theme.text_muted,
            self._CARD_BODY_SIZE,
            anchor_y="top",
        )
        self._draw_text(
            f"{key}_value",
            value,
            right,
            y,
            self.theme.text_primary,
            self._CARD_VALUE_SIZE,
            bold=True,
            anchor_x="right",
            anchor_y="top",
        )

    def _prepare_report_scroll(
        self,
        bounds: arcade.Rect,
        content_height: float,
    ) -> float:
        """Clamp and register fallback page scrolling."""
        scroll_limit = max(0.0, content_height - bounds.height)
        scroll_offset = max(
            0.0,
            min(
                scroll_limit,
                self._scroll_offsets.get("behavior_report", 0.0),
            ),
        )
        self._scroll_regions["behavior_report"] = bounds
        self._scroll_limits["behavior_report"] = scroll_limit
        self._scroll_offsets["behavior_report"] = scroll_offset
        return scroll_offset

    def _report_scroll_content_bounds(
        self,
        bounds: arcade.Rect,
    ) -> arcade.Rect:
        """Inset scroll content so text and cards never meet the scrollbar."""
        return arcade.LBWH(
            bounds.left + 4.0,
            bounds.bottom,
            max(
                1.0,
                bounds.width - self._REPORT_SCROLLBAR_GUTTER - 4.0,
            ),
            bounds.height,
        )

    @staticmethod
    def _direction_lines(
        effect: BehaviorLifetimeWhySummary,
    ) -> tuple[str, ...]:
        """Format every nonzero stored bout-direction count."""
        counts = effect.direction_counts
        denominator = effect.contributing_bout_count
        values = (
            ("SUPPORTIVE", counts.supportive),
            ("SUPPRESSIVE", counts.suppressive),
            ("REVERSING", counts.reversing),
            ("MIXED", counts.mixed),
            ("MINIMAL", counts.minimal),
        )
        return tuple(
            f"{label} in {count} / {denominator} bouts"
            for label, count in values
            if count
        )

    def _draw_report_empty_body(
        self,
        bounds: arcade.Rect,
        message: str = "No completed bouts yet.",
    ) -> None:
        """Draw a centered empty-state message for a report page."""
        self._draw_text(
            "behavior_report_body_empty",
            message,
            bounds.center_x,
            bounds.center_y,
            self.theme.text_muted,
            12.0,
            anchor_x="center",
            anchor_y="center",
        )

    def _close_behavior_report(self) -> None:
        """Close the report and clear transient UI-only selection state."""
        self._behavior_report_open = False
        self._behavior_report_help_open = False
        self._behavior_report_selected_bout_id = None
        self._behavior_report_why_behavior = None
        self._scroll_offsets["behavior_report"] = 0.0
        self._scroll_offsets["behavior_report_creatures"] = 0.0
        self._scroll_offsets["behavior_report_help"] = 0.0
