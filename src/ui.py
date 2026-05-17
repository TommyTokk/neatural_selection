from __future__ import annotations

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
    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.theme = config.theme
        self._text_cache: dict[str, arcade.Text] = {}
        self._control_hitboxes: dict[str, arcade.Rect] = {}
        self._scroll_regions: dict[str, arcade.Rect] = {}
        self._scroll_offsets: dict[str, float] = {}
        self._scroll_limits: dict[str, float] = {}
        self._active_slider = False
        self._brain_window_open = False
        self._brain_window_bounds: arcade.Rect | None = None
        self._brain_window_drag_offset = (0.0, 0.0)
        self._brain_graph_pan = (0.0, 0.0)
        self._brain_graph_zoom = 1.0
        self._active_brain_window_drag = False
        self._active_brain_graph_drag = False
        self._brain_graph_drag_last = (0.0, 0.0)

    def draw(self, world: World) -> None:
        self._control_hitboxes.clear()
        self._draw_top_bar(world)
        self._draw_sidebar(world)
        self._draw_brain_window(world)

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
        card_height = 300
        stats_card_height = 150
        controls_card_height = 210
        gap = 14

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
                genome_id = world.neat_controller.genome_id_for(selected.creature_id)
                cooldown_remaining = max(
                    0.0,
                    world.config.population.reproduction_cooldown
                    - fitness.seconds_since_reproduction(),
                )
                lines.extend(
                    [
                        f"Genome: {genome_id if genome_id is not None else 'None'}",
                        f"Fitness: {fitness.score:.2f}",
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
        open_button = arcade.LBWH(
            content.left,
            content.bottom,
            content.width,
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
        self._draw_button(open_button, "Open Brain", "open_brain_window")

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
            key
            for key in brain.genome.nodes
            if key not in output_keys
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
            color = self.theme.accent if connection.weight >= 0.0 else self.theme.selected_outline
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
            label = SENSOR_INPUT_NAMES[index] if index < len(SENSOR_INPUT_NAMES) else str(key)
            self._draw_brain_node_label(
                f"brain_input_{index}",
                f"{self._short_brain_label(label)} {value:.2f}",
                position,
                graph_bounds,
                side="right",
            )

        for key in hidden_keys:
            self._draw_brain_node(positions[key], self.theme.panel_background, self.theme.panel_border)

        for index, key in enumerate(output_keys):
            position = positions[key]
            value = brain.last_outputs[index] if index < len(brain.last_outputs) else 0.0
            self._draw_brain_node(
                position,
                self._brain_activity_color(value),
                self.theme.herbivore_outline,
                radius=4.0 + min(1.0, abs(value)) * 3.0,
            )
            label = ACTION_OUTPUT_NAMES[index] if index < len(ACTION_OUTPUT_NAMES) else str(key)
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
            1
            for connection in brain.genome.connections.values()
            if connection.enabled
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
            self.theme.panel_background,
            self.theme.environment_border,
            self.config.layout.panel_radius,
            2.5,
        )

        title_bar = arcade.LBWH(bounds.left, bounds.top - 44, bounds.width, 44)
        close_button = arcade.LBWH(bounds.right - 42, bounds.top - 34, 24, 24)
        self._control_hitboxes["brain_window_title"] = title_bar
        self._control_hitboxes["brain_window_close"] = close_button

        genome_label = (
            f"Genome {brain.genome_id}"
            if brain is not None
            else "No genome"
        )
        self._draw_text(
            "brain_window_title_text",
            f"Brain: {selected.name} / {genome_label}",
            bounds.left + 18,
            bounds.top - 27,
            self.theme.text_primary,
            15,
            bold=True,
        )
        self._draw_button(close_button, "x", "brain_window_close")

        graph_bounds = arcade.LBWH(
            bounds.left + 18,
            bounds.bottom + 72,
            bounds.width - 36,
            max(120.0, bounds.height - 134),
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
            bounds.left + 18,
            bounds.bottom + 14,
            bounds.width - 36,
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
            1
            for connection in brain.genome.connections.values()
            if connection.enabled
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
                value = brain.last_inputs[index] if index < len(brain.last_inputs) else 0.0
                fill_color = self._brain_activity_color(value)
                outline_color = self.theme.accent
                radius = 5.0 + min(1.0, abs(value)) * 3.0
            elif node.kind == BrainNodeKind.OUTPUT:
                index = output_keys.index(key)
                value = brain.last_outputs[index] if index < len(brain.last_outputs) else 0.0
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
            control_y = bounds.top - 18.0 if start[1] <= end[1] else bounds.bottom + 18.0
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
        max_bottom = max(min_bottom, world.layout.window.height - outer_padding - height)
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
        pan_x, pan_y = self._brain_graph_pan
        return (
            bounds.center_x + (position[0] - bounds.center_x) * self._brain_graph_zoom + pan_x,
            bounds.center_y + (position[1] - bounds.center_y) * self._brain_graph_zoom + pan_y,
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
            f"Simulation speed: {world.simulation_speed:.2f}x",
            f"Zoom: {world.environment_zoom:.2f}x",
            f"Births: {world.rt_neat.stats.births}",
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
        self._draw_button(pause_button, "> Space" if world.is_paused else "|| Space", "pause")
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
        if self._contains_hitbox("brain_window_close", x, y):
            self._brain_window_open = False
            self._active_brain_window_drag = False
            self._active_brain_graph_drag = False
            return True
        if self._contains_hitbox("brain_window_title", x, y):
            bounds = self._brain_window_bounds
            if bounds is not None:
                self._active_brain_window_drag = True
                self._brain_window_drag_offset = (x - bounds.left, y - bounds.bottom)
                return True
        if self._contains_hitbox("brain_window_graph", x, y):
            self._active_brain_graph_drag = True
            self._brain_graph_drag_last = (x, y)
            return True
        if self._contains_hitbox("open_brain_window", x, y):
            if world.selected_creature is not None:
                self._brain_window_open = True
                self._ensure_brain_window_bounds(world)
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
        if self._contains_hitbox("speed_slider", x, y):
            self._active_slider = True
            self._set_speed_from_slider(world, x)
            return True
        return False

    def handle_mouse_drag(self, world: World, x: float, y: float) -> bool:
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
        if self._active_brain_graph_drag:
            last_x, last_y = self._brain_graph_drag_last
            pan_x, pan_y = self._brain_graph_pan
            self._brain_graph_pan = (pan_x + x - last_x, pan_y + y - last_y)
            self._brain_graph_drag_last = (x, y)
            return True
        if not self._active_slider:
            return False
        self._set_speed_from_slider(world, x)
        return True

    def handle_mouse_release(self) -> None:
        self._active_slider = False
        self._active_brain_window_drag = False
        self._active_brain_graph_drag = False

    def handle_mouse_scroll(self, x: float, y: float, scroll_y: float) -> bool:
        if (
            self._brain_window_open
            and self._brain_window_bounds is not None
            and self._contains_bounds(self._brain_window_bounds, x, y)
        ):
            return True

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
        return {
            key: (x, top - index * step)
            for index, key in enumerate(node_keys)
        }

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
            "food_proximity": "food",
            "food_angle": "f_ang",
            "creature_proximity": "near",
            "creature_angle": "n_ang",
            "accelerate": "acc",
            "rotate": "rot",
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

        return f"Raw outputs: {value(0, outputs):.2f}/{value(1, outputs):.2f}"

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
