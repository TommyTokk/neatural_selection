from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import arcade

from configs.sim_config import SimConfig
from src.action import ACTION_OUTPUT_NAMES
from src.brain_graph import (
    BrainEdgeKind,
    BrainGraphEdge,
    BrainNodeKind,
    build_brain_graph_layout,
)
from src.vision import SENSOR_INPUT_NAMES
from src.world import World


class UiRenderer:
    ICON_BUTTON_SIZE = 58.0
    ICON_BUTTON_GAP = 20.0
    RAIL_VERTICAL_PADDING = 32.0
    PANEL_KEYS = ("inspector", "stats", "settings")

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
        self._panel_bounds: dict[str, arcade.Rect] = {}
        self._active_panel_drag: str | None = None
        self._panel_drag_offset = (0.0, 0.0)
        self._brain_window_open = False
        self._brain_window_bounds: arcade.Rect | None = None
        self._brain_window_drag_offset = (0.0, 0.0)
        self._brain_graph_zoom = 1.0
        self._active_brain_window_drag = False

    def draw(self, world: World) -> None:
        self._control_hitboxes.clear()
        self._scroll_regions.clear()
        self._scroll_limits.clear()
        self._draw_icon_rail(world)
        self._draw_floating_panels(world)
        self._draw_brain_window(world)

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

        button_size = self.ICON_BUTTON_SIZE
        top = bounds.top - self.RAIL_VERTICAL_PADDING / 2.0 - button_size / 2.0
        icon_buttons = (
            ("panel_toggle_inspector", "search", "inspector", top),
            (
                "panel_toggle_stats",
                "analytics",
                "stats",
                top - button_size - self.ICON_BUTTON_GAP,
            ),
            (
                "panel_toggle_settings",
                "tune",
                "settings",
                top - (button_size + self.ICON_BUTTON_GAP) * 2,
            ),
        )
        for key, icon_name, panel_name, center_y in icon_buttons:
            button = arcade.LBWH(
                bounds.center_x - button_size / 2.0,
                center_y - button_size / 2.0,
                button_size,
                button_size,
            )
            self._control_hitboxes[key] = button
            self._draw_icon_button(
                button,
                icon_name,
                key,
                active=self._panel_open[panel_name],
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
        parent_id = getattr(lineage, "parent_id", None)
        generation = getattr(lineage, "generation", 0)
        fitness_score = (
            fitness.score(world.config.fitness) if fitness is not None else None
        )
        energy_ratio = self._inspector_energy_ratio(world)
        padding = 18.0
        section_gap = 18.0
        total_height = 680.0 if fitness_score is not None else 648.0
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
            self._draw_brain_node(
                position,
                self._brain_activity_color(value),
                self.theme.accent,
                radius=4.0 + min(1.0, abs(value)) * 3.0,
            )
            label = (
                SENSOR_INPUT_NAMES[index]
                if index < len(SENSOR_INPUT_NAMES)
                else str(key)
            )
            self._draw_brain_node_label(
                f"brain_input_{index}",
                f"{self._short_brain_label(label)} {value:.2f}",
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
        detail_lines = [
            f"Genome: {brain.genome_id}",
            f"Signed action: {action_label}",
            f"Speed: {selected.speed:.1f} px/s",
            self._brain_input_readout(brain.last_inputs),
            self._brain_output_readout(brain.last_outputs),
            f"Nodes: {len(brain.genome.nodes)}",
            f"Connections: {enabled_connections}/{len(brain.genome.connections)} enabled",
            f"Fitness: {self._format_genome_fitness(brain.genome.fitness)}",
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
            self._brain_window_open = False
            return

        brain = world.neat_controller.brain_for(selected.creature_id)
        self._ensure_brain_window_bounds(world)
        bounds = self._brain_window_bounds
        if bounds is None:
            return

        self._draw_rounded_rect(
            bounds,
            self.theme.panel_background_alt,
            self.theme.panel_border,
            14,
            1.5,
        )

        title_bar = arcade.LBWH(bounds.left, bounds.top - 54, bounds.width, 54)
        close_button = arcade.LBWH(bounds.right - 46, bounds.top - 40, 28, 28)
        self._control_hitboxes["brain_window_title"] = title_bar
        self._control_hitboxes["brain_window_close"] = close_button
        header = arcade.LBWH(
            bounds.left + 1.5, bounds.top - 54, bounds.width - 3.0, 52.5
        )
        self._draw_rounded_rect_fill(header, self.theme.panel_background, 12.5)

        genome_label = f"Genome {brain.genome_id}" if brain is not None else "No genome"
        self._draw_text(
            "brain_window_title_text",
            f"Brain: {selected.name} / {genome_label}",
            bounds.left + 24,
            bounds.top - 30,
            self.theme.text_primary,
            15,
            bold=True,
            anchor_y="center",
        )
        self._draw_panel_close_button(close_button, "brain_window")

        graph_bounds = arcade.LBWH(
            bounds.left + 22,
            bounds.bottom + 72,
            bounds.width - 44,
            max(120.0, bounds.height - 144),
        )
        self._control_hitboxes["brain_window_graph"] = graph_bounds
        self._draw_rounded_rect(
            graph_bounds,
            self.theme.card_background,
            self.theme.panel_border,
            self.config.layout.card_radius,
            1.5,
        )

        footer_bounds = arcade.LBWH(
            bounds.left + 22,
            bounds.bottom + 14,
            bounds.width - 44,
            42,
        )
        if brain is None:
            self._draw_text(
                "brain_window_empty",
                "No brain assigned.",
                graph_bounds.left + 18,
                graph_bounds.top - 28,
                self.theme.text_muted,
                13,
            )
            return

        self._draw_brain_graph(world, graph_bounds)
        enabled_connections = sum(
            1 for connection in brain.genome.connections.values() if connection.enabled
        )
        action = brain.last_action
        action_label = (
            f"acc {action.accelerate:.2f} rot {action.rotate:.2f}"
            if action is not None
            else "waiting"
        )
        footer_lines = [
            (
                f"Nodes: {len(brain.genome.nodes)}  "
                f"Connections: {enabled_connections}/{len(brain.genome.connections)} enabled"
            ),
            (
                f"Fitness: {self._format_genome_fitness(brain.genome.fitness)}  "
                f"Signed action: {action_label}"
            ),
        ]
        self._draw_scrollable_lines_in_bounds(
            "brain_window_footer",
            footer_bounds,
            footer_lines,
            line_spacing=18,
            first_line_color=self.theme.text_muted,
            body_color=self.theme.text_muted,
        )

    def _draw_brain_graph(self, world: World, bounds: arcade.Rect) -> None:
        selected = world.selected_creature
        if selected is None:
            return

        brain = world.neat_controller.brain_for(selected.creature_id)
        if brain is None:
            return

        input_keys = list(world.neat_controller.config.genome_config.input_keys)
        output_keys = list(world.neat_controller.config.genome_config.output_keys)
        layout = build_brain_graph_layout(
            brain.genome,
            input_keys,
            output_keys,
            bounds,
            SENSOR_INPUT_NAMES,
            ACTION_OUTPUT_NAMES,
        )
        positions = {
            key: self._brain_graph_screen_position(position, bounds)
            for key, position in layout.positions.items()
        }

        for edge in layout.edges:
            if not edge.enabled:
                continue
            self._draw_brain_graph_edge(edge, positions, bounds)

        for key, node in layout.nodes.items():
            position = positions.get(key)
            if position is None:
                continue

            fill_color = self.theme.panel_background
            outline_color = self.theme.panel_border
            radius = 6.0
            if node.kind == BrainNodeKind.INPUT:
                index = input_keys.index(key)
                value = (
                    brain.last_inputs[index] if index < len(brain.last_inputs) else 0.0
                )
                fill_color = self._brain_activity_color(value)
                outline_color = self.theme.accent
                radius = 5.0 + min(1.0, abs(value)) * 3.0
            elif node.kind == BrainNodeKind.OUTPUT:
                index = output_keys.index(key)
                value = (
                    brain.last_outputs[index]
                    if index < len(brain.last_outputs)
                    else 0.0
                )
                fill_color = self._brain_activity_color(value)
                outline_color = self.theme.herbivore_outline
                radius = 5.0 + min(1.0, abs(value)) * 3.0

            self._draw_brain_node(position, fill_color, outline_color, radius=radius)
            self._draw_brain_graph_label(key, node.label, node.kind, position, bounds)

    def _draw_brain_graph_edge(
        self,
        edge: BrainGraphEdge,
        positions: dict[int, tuple[float, float]],
        bounds: arcade.Rect,
    ) -> None:
        start = positions.get(edge.source)
        end = positions.get(edge.target)
        if start is None or end is None:
            return

        color = self._brain_edge_color(edge.weight)
        width = max(1.0, min(5.0, abs(edge.weight) * 0.7))
        if edge.kind == BrainEdgeKind.SELF_LOOP:
            self._draw_self_loop(start, color, width)
            return
        if edge.kind == BrainEdgeKind.RECURRENT:
            control_y = (
                bounds.top - 18.0 if start[1] <= end[1] else bounds.bottom + 18.0
            )
            control = ((start[0] + end[0]) * 0.5, control_y)
            self._draw_curve(
                self._quadratic_bezier_points(start, control, end),
                color,
                width,
            )
            return

        arcade.draw_line(start[0], start[1], end[0], end[1], color, width)

    def _draw_brain_graph_label(
        self,
        node_key: int,
        label: str,
        kind: BrainNodeKind,
        position: tuple[float, float],
        bounds: arcade.Rect,
    ) -> None:
        label_text = self._short_brain_label(label)
        label_width = 62.0
        if kind == BrainNodeKind.INPUT:
            x = max(bounds.left + 8, position[0] + 10)
            anchor_x = "left"
        elif kind == BrainNodeKind.OUTPUT:
            x = min(bounds.right - 8, position[0] - 10)
            anchor_x = "right"
        else:
            x = position[0]
            anchor_x = "center"

        y = max(bounds.bottom + 8, min(bounds.top - 16, position[1] - 15))
        self._draw_text(
            f"brain_window_node_label_{node_key}",
            self._fit_line(label_text, label_width),
            x,
            y,
            self.theme.text_muted,
            9,
            anchor_x=anchor_x,
        )

    def _ensure_brain_window_bounds(self, world: World) -> None:
        if self._brain_window_bounds is not None:
            self._brain_window_bounds = self._clamped_brain_window_bounds(
                world,
                self._brain_window_bounds.left,
                self._brain_window_bounds.bottom,
                self._brain_window_bounds.width,
                self._brain_window_bounds.height,
            )
            return

        environment = world.layout.environment
        width = max(360.0, min(environment.width * 0.62, 720.0))
        height = max(260.0, min(environment.height * 0.58, 500.0))
        left = environment.center_x - width / 2
        bottom = environment.center_y - height / 2
        self._brain_window_bounds = self._clamped_brain_window_bounds(
            world,
            left,
            bottom,
            width,
            height,
        )

    def _clamped_brain_window_bounds(
        self,
        world: World,
        left: float,
        bottom: float,
        width: float,
        height: float,
    ) -> arcade.Rect:
        outer_padding = self.config.layout.outer_padding
        min_left = outer_padding
        min_bottom = outer_padding
        max_left = max(min_left, world.layout.window.width - outer_padding - width)
        max_bottom = max(
            min_bottom, world.layout.window.height - outer_padding - height
        )
        return arcade.LBWH(
            max(min_left, min(max_left, left)),
            max(min_bottom, min(max_bottom, bottom)),
            width,
            height,
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

    def _brain_edge_color(self, weight: float) -> arcade.Color | tuple[int, ...]:
        return self.theme.accent if weight >= 0.0 else self.theme.selected_outline

    def _draw_self_loop(
        self,
        position: tuple[float, float],
        color: arcade.Color | tuple[int, ...],
        width: float,
    ) -> None:
        x, y = position
        points = [
            (x + 8.0, y + 2.0),
            (x + 32.0, y + 26.0),
            (x + 22.0, y - 24.0),
            (x + 8.0, y - 2.0),
        ]
        for start, control, end in (
            (points[0], points[1], points[2]),
            (points[2], points[3], points[0]),
        ):
            self._draw_curve(
                self._quadratic_bezier_points(start, control, end, steps=10),
                color,
                width,
            )

    def _draw_curve(
        self,
        points: list[tuple[float, float]],
        color: arcade.Color | tuple[int, ...],
        width: float,
    ) -> None:
        for start, end in zip(points, points[1:]):
            arcade.draw_line(start[0], start[1], end[0], end[1], color, width)

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

    def _draw_environment_stats(self, world: World, bounds: arcade.Rect) -> None:
        lines = [
            f"Population: {world.stats.herbivore_count}/{world.config.population.max_creatures}",
            f"Food nodes: {world.stats.food_count}",
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
        if self._contains_hitbox("brain_window_close", x, y):
            self._brain_window_open = False
            self._active_brain_window_drag = False
            return True
        if self._contains_hitbox("brain_window_title", x, y):
            bounds = self._brain_window_bounds
            if bounds is not None:
                self._active_brain_window_drag = True
                self._brain_window_drag_offset = (x - bounds.left, y - bounds.bottom)
                return True
        if self._contains_hitbox("brain_window_graph", x, y):
            return True
        if self._contains_hitbox("open_brain_window", x, y):
            if world.selected_creature is not None:
                self._brain_window_open = True
                self._ensure_brain_window_bounds(world)
            return True
        if self._contains_hitbox("kill_selected_creature", x, y):
            if world.kill_selected_creature():
                self._brain_window_open = False
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
        if self._active_brain_window_drag:
            bounds = self._brain_window_bounds
            if bounds is None:
                return False
            offset_x, offset_y = self._brain_window_drag_offset
            self._brain_window_bounds = self._clamped_brain_window_bounds(
                world,
                x - offset_x,
                y - offset_y,
                bounds.width,
                bounds.height,
            )
            return True
        if not self._active_slider:
            return False
        self._set_speed_from_slider(world, x)
        return True

    def handle_mouse_release(self) -> None:
        self._active_slider = False
        self._active_panel_drag = None
        self._active_brain_window_drag = False

    def handle_mouse_scroll(self, x: float, y: float, scroll_y: float) -> bool:
        if (
            self._brain_window_open
            and self._brain_window_bounds is not None
            and self._contains_bounds(self._brain_window_bounds, x, y)
        ):
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
    ) -> arcade.Rect:
        self._draw_rounded_rect(
            bounds,
            self.theme.panel_background_alt,
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
                icon_bounds = arcade.LBWH(bounds.left + 26, bounds.top - 38, 20, 20)
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
        return (
            f"R {radius:+.1f}, V {vision_range:+.1f}/"
            f"{vision_angle:+.2f}, M {movement_cost:+.2f}"
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
            "hungriness": "hungry",
            "maturity": "mat",
            "energy_percent": "energy",
            "creature_count": "near",
            "food_count": "food#",
            "clock_tik_tok": "tik",
            "clock_chronometer": "chrono",
            "clock_time_alive": "age",
            "food_proximity": "food",
            "food_angle": "f_ang",
            "creature_proximity": "near",
            "creature_angle": "n_ang",
            "wall_proximity": "wall",
            "wall_angle": "w_ang",
            "accelerate": "acc",
            "rotate": "rot",
            "want_reproduce": "repr",
            "want_eat": "eat",
            "reset_chronometer": "reset",
        }
        return replacements.get(label, label)

    def _format_genome_fitness(self, fitness: object) -> str:
        if fitness is None:
            return "None"
        try:
            return f"{float(fitness):.2f}"
        except (TypeError, ValueError):
            return str(fitness)

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
            f"F {value(0, inputs):.2f}/{value(1, inputs):.2f}  "
            f"C {value(2, inputs):.2f}/{value(3, inputs):.2f}  "
            f"W {value(4, inputs):.2f}/{value(5, inputs):.2f}  "
            f"E {value(6, inputs):.2f}"
        )

    def _brain_output_readout(self, outputs: list[float]) -> str:
        def value(index: int, values: list[float]) -> float:
            return values[index] if index < len(values) else 0.0

        return (
            f"Raw outputs: {value(0, outputs):.2f}/{value(1, outputs):.2f}  "
            f"Intent: {value(2, outputs):.2f}/{value(3, outputs):.2f}/"
            f"{value(4, outputs):.2f}"
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
    ) -> None:
        total_height = max(0.0, len(lines) * line_spacing)
        scroll_limit = max(0.0, total_height - content.height)
        scroll_offset = max(
            0.0,
            min(scroll_limit, self._scroll_offsets.get(key, 0.0)),
        )
        self._scroll_offsets[key] = scroll_offset
        self._scroll_limits[key] = scroll_limit
        self._scroll_regions[key] = content

        for line_index, line in enumerate(lines):
            y = content.top - 12 - line_index * line_spacing + scroll_offset
            if y < content.bottom or y > content.top:
                continue
            is_first_line = line_index == 0
            self._draw_text(
                f"{key}_line_{line_index}",
                self._fit_line(line, content.width - (12 if scroll_limit > 0 else 0)),
                content.left,
                y,
                first_line_color if is_first_line else body_color,
                12,
                bold=first_line_bold and is_first_line,
            )

        if scroll_limit > 0.0:
            self._draw_scrollbar(content, scroll_offset, scroll_limit)

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
