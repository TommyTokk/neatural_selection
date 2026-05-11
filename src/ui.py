from __future__ import annotations

import arcade

from configs.sim_config import SimConfig
from src.world import World


class UiRenderer:
    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.theme = config.theme
        self._text_cache: dict[str, arcade.Text] = {}
        self._control_hitboxes: dict[str, arcade.Rect] = {}
        self._scroll_regions: dict[str, arcade.Rect] = {}
        self._scroll_offsets: dict[str, float] = {}
        self._scroll_limits: dict[str, float] = {}
        self._active_slider = False

    def draw(self, world: World) -> None:
        self._draw_top_bar(world)
        self._draw_sidebar(world)

    def _draw_top_bar(self, world: World) -> None:
        bounds = world.layout.top_bar
        self._draw_panel(bounds)

        self._draw_text(
            "top_title",
            "Neat Game Of Life",
            bounds.left + 18,
            bounds.top - 34,
            self.theme.text_primary,
            24,            
            bold=True,
        )

        self._draw_text(
            "top_subtitle",
            "Window container with nested environment + UI panels",
            bounds.left + 18,
            bounds.top - 60,
            self.theme.text_muted,
            12,
        )

        status = "Debug vision on" if world.debug_vision_enabled else "Debug vision off"
        self._draw_text(
            "top_status",
            status,
            bounds.right - 180,
            bounds.top - 40,
            self.theme.accent,
            13,
            bold=True,
        )

    def _draw_sidebar(self, world: World) -> None:
        bounds = world.layout.left_sidebar
        self._draw_panel(bounds, fill_color=self.theme.panel_background_alt)
        self._scroll_regions.clear()
        self._scroll_limits.clear()

        title_y = bounds.top - 28
        self._draw_text(
            "sidebar_title",
            "Inspector",
            bounds.left + 18,
            title_y,
            self.theme.text_primary,
            18,
            bold=True,
        )

        card_width = bounds.width - 36
        card_height = 238
        stats_card_height = 170
        controls_card_height = 190
        gap = 16

        first_card = arcade.LBWH(
            bounds.left + 18, bounds.top - 58 - card_height, card_width, card_height
        )
        second_card = arcade.LBWH(
            bounds.left + 18,
            first_card.bottom - gap - stats_card_height,
            card_width,
            stats_card_height,
        )
        third_card = arcade.LBWH(
            bounds.left + 18,
            second_card.bottom - gap - controls_card_height,
            card_width,
            controls_card_height,
        )

        self._draw_card(first_card, "Selected Creature")
        self._draw_selected_creature(world, first_card)

        self._draw_card(second_card, "Environment Stats")
        self._draw_environment_stats(world, second_card)

        self._draw_card(third_card, "Controls")
        self._draw_controls(world, third_card)

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
                lines.extend(
                    [
                        f"Fitness: {fitness.score:.2f}",
                        f"Age: {fitness.age_seconds:.1f}s",
                        f"Food eaten: {fitness.food_eaten}",
                        f"Energy gained: {fitness.energy_gained:.3f}",
                        f"Can reproduce: {'Yes' if can_reproduce else 'No'}",
                        f"Offspring: {fitness.offspring_count}",
                    ]
                )

        self._draw_scrollable_lines(
            "selected",
            bounds,
            lines,
            line_spacing=22,
            first_line_color=self.theme.text_primary,
            body_color=self.theme.text_muted,
            first_line_bold=True,
        )

    def _draw_environment_stats(self, world: World, bounds: arcade.Rect) -> None:
        lines = [
            f"Herbivores: {world.stats.herbivore_count}",
            f"Food nodes: {world.stats.food_count}",
            f"Elapsed time: {world.elapsed_time:0.1f}s",
            "State: Paused" if world.is_paused else "State: Running",
            f"Simulation speed: {world.simulation_speed:.2f}x",
            f"Zoom: {world.environment_zoom:.2f}x",
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
        self._control_hitboxes.clear()

        button_top = bounds.top - 48
        button_height = 30
        button_gap = 8
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
        self._draw_button(pause_button, ">" if world.is_paused else "||", "pause")
        self._draw_button(reset_button, "1x", "reset_speed")

        slider_y = reset_button.bottom - 32
        slider = arcade.LBWH(bounds.left + 16, slider_y, bounds.width - 32, 18)
        self._control_hitboxes["speed_slider"] = slider
        self._draw_speed_slider(slider, world)

        small_button_width = (bounds.width - 32 - button_gap) / 2
        small_button_top = slider.bottom - 16
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
        self._draw_button(slow_button, "<<", "speed_down")
        self._draw_button(fast_button, ">>", "speed_up")

        self._draw_text(
            "controls_help",
            f"Space  A/D  </>  {self.config.debug.vision_toggle_label}",
            bounds.left + 16,
            fast_button.bottom - 18,
            self.theme.text_muted,
            10,
            width=bounds.width - 32,
            multiline=True,
        )

    def handle_mouse_press(self, world: World, x: float, y: float) -> bool:
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
        if self._contains_hitbox("speed_slider", x, y):
            self._active_slider = True
            self._set_speed_from_slider(world, x)
            return True
        return False

    def handle_mouse_drag(self, world: World, x: float, y: float) -> bool:
        if not self._active_slider:
            return False
        self._set_speed_from_slider(world, x)
        return True

    def handle_mouse_release(self) -> None:
        self._active_slider = False

    def handle_mouse_scroll(self, x: float, y: float, scroll_y: float) -> bool:
        for key, bounds in self._scroll_regions.items():
            if not (bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top):
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
        self, bounds: arcade.Rect, fill_color: arcade.Color | tuple[int, ...] | None = None
    ) -> None:
        self._draw_rounded_rect(
            bounds,
            fill_color or self.theme.panel_background,
            self.theme.panel_border,
            self.config.layout.panel_radius,
            2,
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
            label,
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
        ratio = (
            (world.simulation_speed - world.MIN_SIMULATION_SPEED)
            / (world.MAX_SIMULATION_SPEED - world.MIN_SIMULATION_SPEED)
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

    def _contains_hitbox(self, key: str, x: float, y: float) -> bool:
        bounds = self._control_hitboxes.get(key)
        if bounds is None:
            return False
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
                font_name=("Verdana", "DejaVu Sans", "Arial"),
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
        return f"{text[:max_chars - 3]}..."

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
        radius = min(radius, bounds.width / 2, bounds.height / 2)
        arcade.draw_lrbt_rectangle_filled(
            bounds.left + radius,
            bounds.right - radius,
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
        arcade.draw_circle_filled(bounds.left + radius, bounds.bottom + radius, radius, color)
        arcade.draw_circle_filled(bounds.right - radius, bounds.bottom + radius, radius, color)
        arcade.draw_circle_filled(bounds.left + radius, bounds.top - radius, radius, color)
        arcade.draw_circle_filled(bounds.right - radius, bounds.top - radius, radius, color)
