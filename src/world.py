from __future__ import annotations

from dataclasses import dataclass
from math import sin
from random import Random

import pymunk

from configs.sim_config import SimConfig
import src.utils as ut
from src.creature import Creature
from src.food import Food
from src.food_spawner import FoodSpawner
from src.metabolism import Metabolism

from src.layout import build_screen_layout


@dataclass(slots=True)
class WorldStats:
    environment_name: str = "Herbivore Basin"
    generation_label: str = "Physics Prototype"
    herbivore_count: int = 0
    food_count: int = 0


class World:
    CREATURE_COUNT = 20
    CREATURE_RADIUS = 14.0
    FIXED_TIMESTEP = 1.0 / 60.0
    MAX_FRAME_STEPS = 5
    MAX_SPEED = 170.0
    MAX_ANGULAR_SPEED = 4.0
    MIN_SIMULATION_SPEED = 0.25
    MAX_SIMULATION_SPEED = 2.0
    SIMULATION_SPEED_STEP = 0.25
    SELECTED_CREATURE_ZOOM = 2.25

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.rng = Random(7)
        self.elapsed_time = 0.0
        self.is_paused = False
        self.simulation_speed = 1.0
        self._physics_accumulator = 0.0
        self.debug_vision_enabled = config.debug.show_debug_vision_by_default
        self.layout = build_screen_layout(
            config.display.width, config.display.height, config.layout
        )
        self.environment_zoom = config.zoom.default
        self.environment_pan_x = 0.0
        self.environment_pan_y = 0.0
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = 0.94
        self.space.iterations = 12
        self._boundary_shapes: list[pymunk.Shape] = []
        self._rebuild_boundaries()
        self.creatures = self._spawn_creatures()
        self.food_spawner = FoodSpawner(config.food, self.rng)
        self.foods: list[Food] = []
        self._add_foods(
            self.food_spawner.create_initial_foods(self.environment_world_bounds)
        )
        self.selected_creature_id: int | None = None
        self.stats = WorldStats(
            herbivore_count=len(self.creatures),
            food_count=len(self.foods),
        )

        self.metabolism = Metabolism(config.metabolism)

    def resize(self, width: int, height: int) -> None:
        self.layout = build_screen_layout(width, height, self.config.layout)
        self._rebuild_boundaries()
        self._keep_creatures_inside_bounds()
        self._follow_selected_creature()
        self._clamp_environment_pan()

    def update(self, delta_time: float) -> None:
        if self.is_paused:
            self._refresh_stats()
            return

        scaled_delta_time = delta_time * self.simulation_speed
        self.elapsed_time += scaled_delta_time
        self._physics_accumulator += min(
            scaled_delta_time, self.FIXED_TIMESTEP * self.MAX_FRAME_STEPS
        )

        steps = 0
        while (
            self._physics_accumulator >= self.FIXED_TIMESTEP
            and steps < self.MAX_FRAME_STEPS
        ):
            self._apply_creature_intents()
            self.space.step(self.FIXED_TIMESTEP)
            self._limit_creature_motion()
            self._update_metabolism(self.FIXED_TIMESTEP)
            self._physics_accumulator -= self.FIXED_TIMESTEP
            steps += 1

        self._spawn_foods(scaled_delta_time)
        self._refresh_stats()
        self._follow_selected_creature()

    def adjust_environment_zoom(self, scroll_y: float) -> None:
        if scroll_y == 0:
            return
        direction = 1 if scroll_y > 0 else -1
        factor = 1.0 + direction * self.config.zoom.step
        updated_zoom = self.environment_zoom * factor
        self.environment_zoom = max(
            self.config.zoom.minimum,
            min(self.config.zoom.maximum, updated_zoom),
        )
        if self.selected_creature is None:
            self._clamp_environment_pan()
        else:
            self._follow_selected_creature()

    def pan_environment(self, delta_x: float, delta_y: float) -> None:
        self.environment_pan_x += delta_x
        self.environment_pan_y += delta_y
        self._clamp_environment_pan()

    def reset_environment_view(self) -> None:
        self.environment_pan_x = 0.0
        self.environment_pan_y = 0.0
        self.environment_zoom = self.config.zoom.default
        self._clamp_environment_pan()

    def toggle_pause(self) -> None:
        self.is_paused = not self.is_paused

    def set_simulation_speed(self, speed: float) -> None:
        clamped_speed = max(
            self.MIN_SIMULATION_SPEED, min(self.MAX_SIMULATION_SPEED, speed)
        )
        steps = round(clamped_speed / self.SIMULATION_SPEED_STEP)
        self.simulation_speed = steps * self.SIMULATION_SPEED_STEP

    def increase_simulation_speed(self) -> None:
        self.set_simulation_speed(self.simulation_speed + self.SIMULATION_SPEED_STEP)

    def decrease_simulation_speed(self) -> None:
        self.set_simulation_speed(self.simulation_speed - self.SIMULATION_SPEED_STEP)

    def reset_simulation_speed(self) -> None:
        self.set_simulation_speed(1.0)

    @property
    def environment_world_bounds(self) -> tuple[float, float, float, float]:
        visible_bounds = self.layout.environment
        zoom = self.config.zoom.minimum
        half_width = visible_bounds.width / (2.0 * zoom)
        half_height = visible_bounds.height / (2.0 * zoom)
        return (
            visible_bounds.center_x - half_width,
            visible_bounds.center_y - half_height,
            visible_bounds.center_x + half_width,
            visible_bounds.center_y + half_height,
        )

    def environment_to_screen(self, x: float, y: float) -> tuple[float, float]:
        bounds = self.layout.environment
        center_x = bounds.center_x
        center_y = bounds.center_y
        return (
            center_x + (x - center_x) * self.environment_zoom + self.environment_pan_x,
            center_y + (y - center_y) * self.environment_zoom + self.environment_pan_y,
        )

    def screen_to_environment(self, x: float, y: float) -> tuple[float, float]:
        bounds = self.layout.environment
        center_x = bounds.center_x
        center_y = bounds.center_y
        model_x = center_x + (x - center_x - self.environment_pan_x) / self.environment_zoom
        model_y = center_y + (y - center_y - self.environment_pan_y) / self.environment_zoom
        return model_x, model_y

    @property
    def selected_creature(self) -> Creature | None:
        for creature in self.creatures:
            if creature.creature_id == self.selected_creature_id:
                return creature
        return None

    def toggle_debug_vision(self) -> None:
        self.debug_vision_enabled = not self.debug_vision_enabled

    def select_creature_at(self, x: float, y: float) -> None:
        environment = self.layout.environment
        if not ut.contains(environment, x, y):
            self.selected_creature_id = None
            return
        world_x, world_y = self.screen_to_environment(x, y)
        chosen: Creature | None = None
        for creature in reversed(self.creatures):
            if creature.contains_point(world_x, world_y) and self._creature_is_visible(
                creature
            ):
                chosen = creature
                break
        self.selected_creature_id = None if chosen is None else chosen.creature_id
        self._focus_selected_creature()

    def _spawn_creatures(self) -> list[Creature]:
        creatures: list[Creature] = []
        left, bottom, right, top = self.environment_world_bounds
        margin = self.CREATURE_RADIUS + 10.0
        for index in range(self.CREATURE_COUNT):
            creature_id = index + 1
            mass = 1.0
            moment = pymunk.moment_for_circle(mass, 0.0, self.CREATURE_RADIUS)
            body = pymunk.Body(mass, moment)
            body.position = (
                self.rng.uniform(left + margin, right - margin),
                self.rng.uniform(bottom + margin, top - margin),
            )
            body.angle = self.rng.uniform(0.0, 6.283185307179586)
            body.velocity = (
                self.rng.uniform(-35.0, 35.0),
                self.rng.uniform(-35.0, 35.0),
            )

            shape = pymunk.Circle(body, self.CREATURE_RADIUS)
            shape.elasticity = 0.45
            shape.friction = 0.8
            self.space.add(body, shape)

            creatures.append(
                Creature(
                    creature_id=creature_id,
                    name=f"Herbivore {creature_id:02d}",
                    body=body,
                    shape=shape,
                    radius=self.CREATURE_RADIUS,
                    energy=self.rng.uniform(0.55, 0.95),
                    color=self.config.theme.herbivore_fill,
                )
            )
        return creatures

    def _spawn_foods(self, delta_time: float) -> None:
        spawned_foods = self.food_spawner.update(
            delta_time,
            self.environment_world_bounds,
            len(self.foods),
        )
        self._add_foods(spawned_foods)

    def _add_foods(self, foods: list[Food]) -> None:
        for food in foods:
            self.foods.append(food)
            self.space.add(food.body, food.shape)

    def _rebuild_boundaries(self) -> None:
        if self._boundary_shapes:
            self.space.remove(*self._boundary_shapes)
            self._boundary_shapes.clear()

        left, bottom, right, top = self.environment_world_bounds
        corners = [
            (left, bottom),
            (right, bottom),
            (right, top),
            (left, top),
        ]
        for start, end in zip(corners, [*corners[1:], corners[0]]):
            shape = pymunk.Segment(self.space.static_body, start, end, 1.0)
            shape.elasticity = 0.7
            shape.friction = 0.9
            self._boundary_shapes.append(shape)
        self.space.add(*self._boundary_shapes)

    def _apply_creature_intents(self) -> None:
        for creature in self.creatures:
            phase = self.elapsed_time * 1.25 + creature.creature_id * 1.91
            forward_force = 95.0 + 30.0 * sin(phase * 0.7)
            torque = 420.0 * sin(phase)
            creature.body.apply_force_at_local_point((forward_force, 0.0), (0.0, 0.0))
            creature.body.torque += torque

    def _limit_creature_motion(self) -> None:
        for creature in self.creatures:
            velocity = creature.body.velocity
            if velocity.length > self.MAX_SPEED:
                creature.body.velocity = velocity.normalized() * self.MAX_SPEED
            creature.body.angular_velocity = max(
                -self.MAX_ANGULAR_SPEED,
                min(self.MAX_ANGULAR_SPEED, creature.body.angular_velocity),
            )

    def _keep_creatures_inside_bounds(self) -> None:
        left, bottom, right, top = self.environment_world_bounds
        for creature in self.creatures:
            x, y = creature.position
            radius = creature.radius + 2.0
            creature.body.position = (
                max(left + radius, min(right - radius, x)),
                max(bottom + radius, min(top - radius, y)),
            )

    def _clamp_environment_pan(self) -> None:
        visible_bounds = self.layout.environment
        left, bottom, right, top = self.environment_world_bounds
        world_half_width = (right - left) / 2.0
        world_half_height = (top - bottom) / 2.0
        visible_half_width = visible_bounds.width / (2.0 * self.environment_zoom)
        visible_half_height = visible_bounds.height / (2.0 * self.environment_zoom)
        max_pan_x = max(
            0.0, (world_half_width - visible_half_width) * self.environment_zoom
        )
        max_pan_y = max(
            0.0, (world_half_height - visible_half_height) * self.environment_zoom
        )
        self.environment_pan_x = max(
            -max_pan_x, min(max_pan_x, self.environment_pan_x)
        )
        self.environment_pan_y = max(
            -max_pan_y, min(max_pan_y, self.environment_pan_y)
        )

    def _focus_selected_creature(self) -> None:
        selected = self.selected_creature
        if selected is None:
            return

        self.environment_zoom = max(
            self.config.zoom.minimum,
            min(self.config.zoom.maximum, self.SELECTED_CREATURE_ZOOM),
        )
        self._follow_selected_creature()

    def _follow_selected_creature(self) -> None:
        selected = self.selected_creature
        if selected is None:
            return

        bounds = self.layout.environment
        selected_x, selected_y = selected.position
        self.environment_pan_x = (
            -(selected_x - bounds.center_x) * self.environment_zoom
        )
        self.environment_pan_y = (
            -(selected_y - bounds.center_y) * self.environment_zoom
        )
        self._clamp_environment_pan()

    def _creature_is_visible(self, creature: Creature) -> bool:
        draw_x, draw_y = self.environment_to_screen(*creature.position)
        radius = creature.radius * self.environment_zoom
        bounds = self.layout.environment
        return (
            draw_x - radius >= bounds.left
            and draw_x + radius <= bounds.right
            and draw_y - radius >= bounds.bottom
            and draw_y + radius <= bounds.top
        )

    def _refresh_stats(self) -> None:
        self.stats.herbivore_count = len(self.creatures)
        self.stats.food_count = len(self.foods)

    def _update_metabolism(self, delta_time: float) -> None:
        report = self.metabolism.update(self.creatures, self.foods, delta_time, self.MAX_SPEED)

        for food in report.eaten_foods:
            if food in self.foods:
                self.foods.remove(food)
                self.space.remove(food.body, food.shape)

        for creature in report.dead_creatures:
            if creature in self.creatures:
                self.creatures.remove(creature)
                self.space.remove(creature.body, creature.shape)

        if self.selected_creature_id is not None and self.selected_creature is None:
            self.selected_creature_id = None
