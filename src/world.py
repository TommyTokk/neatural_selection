from __future__ import annotations

from colorsys import hsv_to_rgb, rgb_to_hsv
from dataclasses import dataclass, field
from math import cos, floor, sin
from random import Random

import pymunk

from configs.sim_config import SimConfig
import src.utils as ut
from src.action import Action, acceleration_force_vector
from src.biome import Biome, BiomeGenerationHandler
from src.creature import (
    Color,
    Creature,
    LineageInfo,
    PhysicalTraits,
    TraitMutationDelta,
    VisionTraits,
)
from src.fitness import CreatureFitness
from src.food import Food
from src.food_spawner import FoodSpawner
from src.metabolism import Metabolism
from src.vision import SensorSnapshot, VisionSystem
from src.controller import BaselineFoodController
from src.neat_controller import NeatBrainController
from src.rt_neat import RtNeatManager
from src.collision import BOUNDARY_CATEGORY, CREATURE_CATEGORY, FOOD_CATEGORY

from src.layout import build_screen_layout


@dataclass(slots=True)
class WorldStats:
    environment_name: str = "Herbivore Basin"
    generation_label: str = "Physics Prototype"
    herbivore_count: int = 0
    food_count: int = 0
    total_biomass_energy: float = 0.0
    creature_energy: float = 0.0
    plant_energy: float = 0.0
    available_biomass: float = 0.0
    plant_spawn_pressure: float = 1.0
    biome_area_shares: dict[str, float] = field(default_factory=dict)
    biome_food_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class ChildCreatureTraits:
    vision: VisionTraits
    physical_traits: PhysicalTraits
    color: Color
    lineage: LineageInfo


@dataclass(slots=True)
class ArchivedCreatureTraits:
    creature_id: int
    vision: VisionTraits
    physical_traits: PhysicalTraits
    color: Color
    lineage: LineageInfo


class World:
    CREATURE_COLOR_PALETTE: tuple[Color, ...] = (
        (86, 156, 214),
        (207, 112, 139),
        (236, 178, 84),
        (154, 126, 206),
        (69, 170, 160),
        (224, 126, 74),
        (116, 143, 214),
        (198, 96, 185),
        (98, 188, 196),
        (220, 150, 105),
    )
    CREATURE_RADIUS = 16.0
    FIXED_TIMESTEP = 1.0 / 60.0
    MAX_FRAME_STEPS = 5
    MAX_SPEED = 170.0
    MAX_ANGULAR_SPEED = 4.0
    MIN_SIMULATION_SPEED = 0.25
    MAX_SIMULATION_SPEED = 5.0
    SIMULATION_SPEED_STEP = 0.25
    SELECTED_CREATURE_ZOOM = 2.25
    REPRODUCTION_INTERVAL = 1.0

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.rng = Random(7)
        self.elapsed_time = 0.0
        self.fps = 0.0
        self.is_paused = False
        self.simulation_speed = 1.0
        self.show_biome_background = False
        self._physics_accumulator = 0.0
        self._reproduction_accumulator = 0.0
        self._last_actions: dict[int, Action] = {}
        self.debug_vision_enabled = config.debug.show_debug_vision_by_default
        self.layout = build_screen_layout(
            config.display.width, config.display.height, config.layout
        )
        self.environment_zoom = config.zoom.default
        self.environment_pan_x = 0.0
        self.environment_pan_y = 0.0
        self.vision = VisionSystem(config.vision, config.metabolism.eating_distance)
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = 0.94
        self.space.iterations = 12
        self._boundary_shapes: list[pymunk.Shape] = []
        self._rebuild_boundaries()
        self.creatures = self._spawn_creatures()
        self._chronometers: dict[int, float] = {
            creature.creature_id: 0.0 for creature in self.creatures
        }
        self.fitness: dict[int, CreatureFitness] = {
            creature.creature_id: CreatureFitness() for creature in self.creatures
        }
        self.fitness_archive: dict[int, CreatureFitness] = {}
        self.biome_map = BiomeGenerationHandler(config.biome).generate(
            self.environment_world_bounds
        )
        self.food_spawner = FoodSpawner(config.food, self.rng, self.biome_map)
        self.foods: list[Food] = []
        self._held_food_by_creature_id: dict[int, int] = {}
        self._carrier_by_food_id: dict[int, int] = {}
        self._food_grid: dict[tuple[int, int], list[Food]] = {}
        self._food_grid_cells_by_id: dict[int, tuple[int, int]] = {}
        self._food_grid_dirty = False
        self._food_grid_cell_size = (
            max(
                config.vision.max_range,
                config.metabolism.eating_distance,
            )
            + config.food.max_food_radius
            + config.trait.max_radius
        )
        self._add_foods(
            self.food_spawner.create_initial_foods(self.environment_world_bounds)
        )
        self.total_biomass_energy = self._initial_total_biomass_energy()
        self.selected_creature_id: int | None = None
        self.stats = WorldStats(
            herbivore_count=len(self.creatures),
            food_count=len(self.foods),
            total_biomass_energy=self.total_biomass_energy,
            creature_energy=self._creature_energy(),
            plant_energy=self._plant_energy(),
            available_biomass=self._available_biomass(),
            plant_spawn_pressure=self._plant_spawn_pressure(),
            biome_area_shares=self._biome_area_shares(),
            biome_food_counts=self._biome_food_counts(),
        )

        self.metabolism = Metabolism(config.metabolism, self.vision, config.trait)

        self.baseline_controller = BaselineFoodController(self.config.action)
        self.neat_controller = NeatBrainController("configs/neat_herbivore.ini")
        self.neat_controller.assign_initial_brains(
            [creature.creature_id for creature in self.creatures]
        )
        self.rt_neat = RtNeatManager(self.neat_controller)
        self._trait_archive_by_genome_id: dict[int, ArchivedCreatureTraits] = {}
        self.use_neat_brains = config.controller.use_neat_brains
        self.show_brain_view = False

    def resize(self, width: int, height: int) -> None:
        self.layout = build_screen_layout(width, height, self.config.layout)
        self._clamp_environment_pan()

    def update(self, delta_time: float) -> None:
        if delta_time > 0.0:
            instant_fps = 1.0 / delta_time
            self.fps = (
                instant_fps if self.fps == 0.0 else self.fps * 0.9 + instant_fps * 0.1
            )

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
            self._settle_food_motion()
            self._apply_top_down_motion()
            self._limit_creature_motion()
            self._sync_carried_foods()
            self._update_fitness_survival(self.FIXED_TIMESTEP)
            self._update_chronometers(self.FIXED_TIMESTEP)
            self._update_metabolism(self.FIXED_TIMESTEP)
            self._physics_accumulator -= self.FIXED_TIMESTEP
            steps += 1

        self._spawn_foods(scaled_delta_time)
        self._update_reproduction(scaled_delta_time)
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

    def toggle_brain_view(self) -> None:
        self.show_brain_view = not self.show_brain_view

    def toggle_biome_background(self) -> None:
        self.show_biome_background = not self.show_biome_background

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
        half_width = self.config.environment.world_width / 2.0
        half_height = self.config.environment.world_height / 2.0
        return (
            -half_width,
            -half_height,
            half_width,
            half_height,
        )

    def environment_to_screen(self, x: float, y: float) -> tuple[float, float]:
        bounds = self.layout.environment
        return (
            bounds.center_x + x * self.environment_zoom + self.environment_pan_x,
            bounds.center_y + y * self.environment_zoom + self.environment_pan_y,
        )

    def screen_to_environment(self, x: float, y: float) -> tuple[float, float]:
        bounds = self.layout.environment
        model_x = (x - bounds.center_x - self.environment_pan_x) / self.environment_zoom
        model_y = (y - bounds.center_y - self.environment_pan_y) / self.environment_zoom
        return model_x, model_y

    def visible_world_bounds(self) -> tuple[float, float, float, float]:
        bounds = self.layout.environment
        bottom_left = self.screen_to_environment(bounds.left, bounds.bottom)
        top_right = self.screen_to_environment(bounds.right, bounds.top)
        visible_left = min(bottom_left[0], top_right[0])
        visible_bottom = min(bottom_left[1], top_right[1])
        visible_right = max(bottom_left[0], top_right[0])
        visible_top = max(bottom_left[1], top_right[1])

        world_left, world_bottom, world_right, world_top = self.environment_world_bounds
        return (
            max(world_left, visible_left),
            max(world_bottom, visible_bottom),
            min(world_right, visible_right),
            min(world_top, visible_top),
        )

    def visible_foods_for_viewport(self) -> list[Food]:
        left, bottom, right, top = self.visible_world_bounds()
        margin = self.config.food.max_food_radius
        candidate_foods = self._foods_in_world_bounds(
            left - margin,
            bottom - margin,
            right + margin,
            top + margin,
        )
        return [
            food
            for food in candidate_foods
            if self._circle_intersects_world_bounds(
                food.position[0],
                food.position[1],
                food.radius,
                left,
                bottom,
                right,
                top,
            )
        ]

    def visible_creatures_for_viewport(self) -> list[Creature]:
        left, bottom, right, top = self.visible_world_bounds()
        return [
            creature
            for creature in self.creatures
            if self._circle_intersects_world_bounds(
                creature.position[0],
                creature.position[1],
                creature.radius,
                left,
                bottom,
                right,
                top,
            )
        ]

    @property
    def selected_creature(self) -> Creature | None:
        for creature in self.creatures:
            if creature.creature_id == self.selected_creature_id:
                return creature
        return None

    def sensor_snapshot_for(self, creature: Creature) -> SensorSnapshot:
        return self._sensor_snapshot_for(creature, record_food_discoveries=False)

    def _sensor_snapshot_for(
        self,
        creature: Creature,
        *,
        record_food_discoveries: bool,
    ) -> SensorSnapshot:
        nearby_foods = self._nearby_foods_for(
            creature,
            creature.vision.range + self.config.food.max_food_radius,
        )

        fitness = self.fitness.get(creature.creature_id)
        age_seconds = 0.0 if fitness is None else fitness.age_seconds
        chronometer = self._chronometers.get(creature.creature_id, 0.0)

        maturity = min(
            age_seconds / self.config.population.min_reproduction_age,
            1.0,
        )

        clock_tik_tok = 1.0 if int(age_seconds) % 2 == 0 else 0.0
        clock_chronometer = min(chronometer / 20.0, 1.0)
        clock_time_alive = min(age_seconds / 120.0, 1.0)
        ignored_food_ids = self._ignored_food_ids_for(creature)
        is_grabbing = creature.creature_id in self._held_food_by_creature_id

        if record_food_discoveries:
            result = self.vision.sense_with_visible_food_ids(
                creature,
                nearby_foods,
                self.creatures,
                self.environment_world_bounds,
                self.MAX_SPEED,
                maturity=maturity,
                clock_tik_tok=clock_tik_tok,
                clock_chronometer=clock_chronometer,
                clock_time_alive=clock_time_alive,
                is_grabbing=is_grabbing,
                ignored_food_ids=ignored_food_ids,
            )
            self._record_food_discoveries(creature, result.visible_food_ids)
            return result.snapshot

        return self.vision.sense(
            creature,
            nearby_foods,
            self.creatures,
            self.environment_world_bounds,
            self.MAX_SPEED,
            maturity=maturity,
            clock_tik_tok=clock_tik_tok,
            clock_chronometer=clock_chronometer,
            clock_time_alive=clock_time_alive,
            is_grabbing=is_grabbing,
            ignored_food_ids=ignored_food_ids,
        )

    def visible_foods_for(self, creature: Creature) -> list[Food]:
        nearby_foods = self._nearby_foods_for(
            creature,
            creature.vision.range + self.config.food.max_food_radius,
        )
        return self.vision.visible_foods(
            creature,
            nearby_foods,
            self.creatures,
            ignored_food_ids=self._ignored_food_ids_for(creature),
        )

    def visible_creatures_for(self, creature: Creature) -> list[Creature]:
        nearby_foods = self._nearby_foods_for(
            creature,
            creature.vision.range + self.config.food.max_food_radius,
        )
        return self.vision.visible_creatures(
            creature,
            self.creatures,
            nearby_foods,
            ignored_food_ids=self._ignored_food_ids_for(creature),
        )

    def fitness_for(self, creature: Creature) -> CreatureFitness | None:
        return self.fitness.get(creature.creature_id)

    def live_brain_count(self) -> int:
        return sum(
            1
            for creature in self.creatures
            if creature.creature_id in self.neat_controller.brains
        )

    def archived_fitness_count(self) -> int:
        return len(self.fitness_archive)

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

    def kill_selected_creature(self) -> bool:
        selected = self.selected_creature
        if selected is None:
            return False

        self._remove_creature(selected)
        if not self.creatures:
            self._recover_extinct_population()
        self._refresh_stats()
        return True

    def _update_chronometers(self, delta_time: float) -> None:
        for creature in self.creatures:
            self._chronometers[creature.creature_id] = (
                self._chronometers.get(creature.creature_id, 0.0) + delta_time
            )

    def _spawn_creatures(self) -> list[Creature]:
        return [
            self._spawn_creature(
                index + 1,
                color=self._initial_creature_color(index),
            )
            for index in range(self.config.population.initial_creatures)
        ]

    def _spawn_creature(
        self,
        creature_id: int,
        position: tuple[float, float] | None = None,
        heading: float | None = None,
        energy: float | None = None,
        color: Color | None = None,
        vision: VisionTraits | None = None,
        physical_traits: PhysicalTraits | None = None,
        lineage: LineageInfo | None = None,
    ) -> Creature:
        left, bottom, right, top = self.environment_world_bounds

        if physical_traits is None:
            physical_traits = self._initial_physical_traits()

        if lineage is None:
            lineage = LineageInfo()

        radius = physical_traits.radius
        margin = radius + 10.0

        mass = 1.0
        moment = pymunk.moment_for_circle(mass, 0.0, radius)
        body = pymunk.Body(mass, moment)
        if position is None:
            body.position = (
                self.rng.uniform(left + margin, right - margin),
                self.rng.uniform(bottom + margin, top - margin),
            )
        else:
            body.position = position
        body.angle = (
            self.rng.uniform(0.0, 6.283185307179586) if heading is None else heading
        )
        body.velocity = (0.0, 0.0)

        shape = pymunk.Circle(body, radius)
        shape.filter = pymunk.ShapeFilter(
            categories=CREATURE_CATEGORY,
            mask=CREATURE_CATEGORY | FOOD_CATEGORY | BOUNDARY_CATEGORY,
        )
        shape.elasticity = 0.15
        shape.friction = 0.0
        self.space.add(body, shape)

        if vision is None:
            vision = VisionTraits(
                range=self.rng.uniform(
                    self.config.vision.min_range, self.config.vision.max_range
                ),
                angle=self.rng.uniform(
                    self.config.vision.min_angle, self.config.vision.max_angle
                ),
            )

        return Creature(
            creature_id=creature_id,
            name=f"Herbivore {creature_id:02d}",
            body=body,
            shape=shape,
            energy=(self.rng.uniform(0.55, 0.95) if energy is None else energy),
            vision=vision,
            color=(
                color
                if color is not None
                else self._initial_creature_color(creature_id - 1)
            ),
            physical_traits=physical_traits,
            lineage=lineage,
        )

    def _initial_creature_color(self, index: int) -> Color:
        return self.CREATURE_COLOR_PALETTE[index % len(self.CREATURE_COLOR_PALETTE)]

    def _initial_physical_traits(self) -> PhysicalTraits:
        trait_config = self.config.trait
        radius = self._clamp(
            trait_config.default_radius
            + self.rng.gauss(0.0, trait_config.initial_radius_jitter),
            trait_config.min_radius,
            trait_config.max_radius,
        )
        movement_cost_multiplier = self._clamp(
            trait_config.default_movement_cost_multiplier
            + self.rng.gauss(0.0, trait_config.initial_movement_cost_jitter),
            trait_config.min_movement_cost_multiplier,
            trait_config.max_movement_cost_multiplier,
        )
        return PhysicalTraits(
            radius=radius,
            movement_cost_multiplier=movement_cost_multiplier,
        )

    def _mutated_vision(self, parent_vision: VisionTraits) -> VisionTraits:
        vision, _ = self._mutated_vision_with_delta(parent_vision)
        return vision

    def _mutated_vision_with_delta(
        self,
        parent_vision: VisionTraits,
    ) -> tuple[VisionTraits, TraitMutationDelta]:
        range_mutation = self.rng.gauss(0, 8)
        angle_mutation = self.rng.gauss(0, 0.08)

        child_vision = VisionTraits(
            range=self._clamp(
                parent_vision.range + range_mutation,
                self.config.vision.min_range,
                self.config.vision.max_range,
            ),
            angle=self._clamp(
                parent_vision.angle + angle_mutation,
                self.config.vision.min_angle,
                self.config.vision.max_angle,
            ),
        )
        return (
            child_vision,
            TraitMutationDelta(
                vision_range=child_vision.range - parent_vision.range,
                vision_angle=child_vision.angle - parent_vision.angle,
            ),
        )

    def _mutated_physical_traits(
        self,
        parent_traits: PhysicalTraits,
    ) -> tuple[PhysicalTraits, TraitMutationDelta]:
        trait_config = self.config.trait
        radius_mutation = self.rng.gauss(0.0, trait_config.radius_mutation_stddev)
        movement_mutation = self.rng.gauss(
            0.0,
            trait_config.movement_cost_mutation_stddev,
        )

        child_radius = self._clamp(
            parent_traits.radius + radius_mutation,
            trait_config.min_radius,
            trait_config.max_radius,
        )
        child_movement_cost_multiplier = self._clamp(
            parent_traits.movement_cost_multiplier + movement_mutation,
            trait_config.min_movement_cost_multiplier,
            trait_config.max_movement_cost_multiplier,
        )

        return (
            PhysicalTraits(
                radius=child_radius,
                movement_cost_multiplier=child_movement_cost_multiplier,
            ),
            TraitMutationDelta(
                radius=child_radius - parent_traits.radius,
                movement_cost_multiplier=(
                    child_movement_cost_multiplier
                    - parent_traits.movement_cost_multiplier
                ),
            ),
        )

    def _mutated_child_traits(self, parent: Creature) -> ChildCreatureTraits:
        return self._mutated_child_traits_from_parent_values(
            parent_id=parent.creature_id,
            parent_generation=parent.lineage.generation,
            parent_vision=parent.vision,
            parent_physical_traits=parent.physical_traits,
            parent_color=parent.color,
        )

    def _mutated_child_traits_from_parent_values(
        self,
        parent_id: int | None,
        parent_generation: int,
        parent_vision: VisionTraits,
        parent_physical_traits: PhysicalTraits,
        parent_color: Color,
    ) -> ChildCreatureTraits:
        child_vision, vision_delta = self._mutated_vision_with_delta(parent_vision)
        child_physical_traits, physical_delta = self._mutated_physical_traits(
            parent_physical_traits,
        )
        mutation_delta = TraitMutationDelta(
            vision_range=vision_delta.vision_range,
            vision_angle=vision_delta.vision_angle,
            radius=physical_delta.radius,
            movement_cost_multiplier=physical_delta.movement_cost_multiplier,
        )
        return ChildCreatureTraits(
            vision=child_vision,
            physical_traits=child_physical_traits,
            color=self._mutated_creature_color(parent_color),
            lineage=LineageInfo(
                parent_id=parent_id,
                generation=parent_generation + 1,
                mutation_delta=mutation_delta,
            ),
        )

    def _mutated_creature_color(self, parent_color: Color) -> Color:
        red, green, blue = parent_color[:3]
        hue, saturation, value = rgb_to_hsv(
            red / 255.0,
            green / 255.0,
            blue / 255.0,
        )
        hue = (hue + self.rng.uniform(-0.035, 0.035)) % 1.0
        saturation = max(
            0.48,
            min(0.82, saturation + self.rng.uniform(-0.06, 0.06)),
        )
        value = max(0.62, min(0.92, value + self.rng.uniform(-0.05, 0.05)))
        if self._is_food_like_color(hsv_to_rgb(hue, saturation, value)):
            hue = (hue + 0.22) % 1.0
        red, green, blue = hsv_to_rgb(hue, saturation, value)
        return (int(red * 255), int(green * 255), int(blue * 255))

    def _is_food_like_color(self, color: tuple[float, float, float]) -> bool:
        food_red, food_green, food_blue = self.config.theme.food_fill[:3]
        red, green, blue = (channel * 255.0 for channel in color)
        distance_squared = (
            (red - food_red) ** 2 + (green - food_green) ** 2 + (blue - food_blue) ** 2
        )
        return distance_squared < 70.0**2

    def _child_spawn_position(
        self,
        parent: Creature,
        child_radius: float,
    ) -> tuple[float, float]:
        distance = max(
            self.config.population.child_spawn_distance,
            parent.radius + child_radius + 2.0,
        )
        parent_x, parent_y = parent.position
        raw_x = parent_x - cos(parent.heading) * distance
        raw_y = parent_y - sin(parent.heading) * distance

        left, bottom, right, top = self.environment_world_bounds
        radius = child_radius + 2.0
        spawn_x = max(left + radius, min(right - radius, raw_x))
        spawn_y = max(bottom + radius, min(top - radius, raw_y))
        return spawn_x, spawn_y

    def _next_creature_id(self) -> int:
        known_ids = [
            *(creature.creature_id for creature in self.creatures),
            *self.fitness.keys(),
            *self.fitness_archive.keys(),
        ]
        if not known_ids:
            return 1
        return max(known_ids) + 1

    def _spawn_foods(self, delta_time: float) -> None:
        spawned_foods = self.food_spawner.update(
            delta_time,
            self.environment_world_bounds,
            len(self.foods),
            len(self.creatures),
            self._available_biomass(),
        )
        self._add_foods(spawned_foods)

    def _add_foods(self, foods: list[Food]) -> None:
        for food in foods:
            self.foods.append(food)
            self.space.add(food.body, food.shape)
            self._index_food(food)

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
            shape.filter = pymunk.ShapeFilter(
                categories=BOUNDARY_CATEGORY,
                mask=CREATURE_CATEGORY | FOOD_CATEGORY,
            )
            shape.elasticity = 0.25
            shape.friction = 0.0
            self._boundary_shapes.append(shape)
        self.space.add(*self._boundary_shapes)

    def _apply_creature_intents(self) -> None:
        for creature in self.creatures:
            snapshot = self._sensor_snapshot_for(
                creature,
                record_food_discoveries=True,
            )
            if self.use_neat_brains:
                action = self.neat_controller.decide(creature.creature_id, snapshot)
                self._last_actions[creature.creature_id] = action

                if action.reset_chronometer >= 0.5:
                    self._chronometers[creature.creature_id] = 0.0

                self._apply_carry_intent(creature, action)
                self._apply_action(
                    creature,
                    action,
                    stabilize_velocity=False,
                    apply_stabilizers=False,
                )
            else:
                action = self.baseline_controller.decide(snapshot, creature.creature_id)
                self._last_actions[creature.creature_id] = action

                if action.reset_chronometer >= 0.5:
                    self._chronometers[creature.creature_id] = 0.0

                self._apply_carry_intent(creature, action)
                self._apply_action(
                    creature,
                    action,
                    stabilize_velocity=snapshot.food.visible > 0.0,
                    apply_stabilizers=True,
                )

    def _apply_action(
        self,
        creature: Creature,
        action: Action,
        stabilize_velocity: bool = False,
        apply_stabilizers: bool = True,
    ) -> None:
        thrust = action.accelerate
        turn = action.rotate

        if apply_stabilizers and stabilize_velocity and thrust > 0.0:
            self._stabilize_food_tracking_velocity(creature)

        force_vector = acceleration_force_vector(
            thrust,
            creature.heading,
            self.config.action.max_forward_force,
            self.config.action.max_backward_force,
        )
        creature.body.apply_force_at_world_point(
            force_vector,
            creature.body.position,
        )

        if (
            apply_stabilizers
            and turn == 0.0
            and thrust > 0.0
            and abs(creature.body.angular_velocity) > 0.0
        ):
            creature.body.angular_velocity *= (
                self.config.action.centered_food_angular_velocity_retention
            )
            damping_torque = (
                -creature.body.angular_velocity
                * self.config.action.max_turn_torque
                * self.config.action.centered_food_angular_damping
            )
            creature.body.torque += damping_torque

        self._apply_turn_control(creature, turn)

        if apply_stabilizers and thrust < 0.0:
            creature.body.angular_velocity *= (
                self.config.action.boundary_angular_velocity_retention
            )

        if apply_stabilizers and not stabilize_velocity and thrust > 0.0:
            creature.body.angular_velocity *= (
                self.config.action.search_angular_velocity_retention
            )

    def _apply_top_down_motion(self) -> None:
        for creature in self.creatures:
            self._apply_planar_drag(creature)
            action = self._last_actions.get(creature.creature_id)
            if action is not None:
                self._apply_turn_control(creature, action.rotate)

    def _apply_planar_drag(self, creature: Creature) -> None:
        velocity = creature.body.velocity
        heading = creature.heading
        forward_x = cos(heading)
        forward_y = sin(heading)
        lateral_x = -sin(heading)
        lateral_y = cos(heading)

        forward_speed = velocity.x * forward_x + velocity.y * forward_y
        lateral_speed = velocity.x * lateral_x + velocity.y * lateral_y

        forward_speed *= self.config.action.forward_velocity_retention
        lateral_speed *= self.config.action.lateral_velocity_retention

        if abs(forward_speed) < self.config.action.linear_stop_threshold:
            forward_speed = 0.0
        if abs(lateral_speed) < self.config.action.linear_stop_threshold:
            lateral_speed = 0.0

        creature.body.velocity = (
            forward_x * forward_speed + lateral_x * lateral_speed,
            forward_y * forward_speed + lateral_y * lateral_speed,
        )

    def _apply_turn_control(self, creature: Creature, rotate: float) -> None:
        if abs(rotate) < self.config.action.turn_deadzone:
            rotate = 0.0

        target_angular_velocity = rotate * self.MAX_ANGULAR_SPEED
        current_angular_velocity = creature.body.angular_velocity
        response = (
            self.config.action.turn_damping
            if rotate == 0.0
            else self.config.action.turn_response
        )
        response = max(0.0, min(1.0, response))
        updated_angular_velocity = (
            current_angular_velocity
            + (target_angular_velocity - current_angular_velocity) * response
        )

        if (
            rotate == 0.0
            and abs(updated_angular_velocity)
            < self.config.action.angular_stop_threshold
        ):
            updated_angular_velocity = 0.0

        creature.body.angular_velocity = updated_angular_velocity
        creature.body.torque = 0.0

    def _stabilize_food_tracking_velocity(self, creature: Creature) -> None:
        velocity = creature.body.velocity
        heading = creature.heading
        forward_x = cos(heading)
        forward_y = sin(heading)
        lateral_x = -sin(heading)
        lateral_y = cos(heading)

        forward_speed = velocity.x * forward_x + velocity.y * forward_y
        lateral_speed = velocity.x * lateral_x + velocity.y * lateral_y

        lateral_speed *= self.config.action.food_tracking_lateral_velocity_retention
        if forward_speed < 0.0:
            forward_speed *= (
                self.config.action.food_tracking_backward_velocity_retention
            )

        creature.body.velocity = (
            forward_x * forward_speed + lateral_x * lateral_speed,
            forward_y * forward_speed + lateral_y * lateral_speed,
        )

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
            clamped_x = max(left + radius, min(right - radius, x))
            clamped_y = max(bottom + radius, min(top - radius, y))
            creature.body.position = (
                clamped_x,
                clamped_y,
            )
            velocity = creature.body.velocity
            velocity_x = velocity.x
            velocity_y = velocity.y
            if clamped_x != x and (velocity_x < 0.0) == (x < clamped_x):
                velocity_x = 0.0
            if clamped_y != y and (velocity_y < 0.0) == (y < clamped_y):
                velocity_y = 0.0
            creature.body.velocity = (velocity_x, velocity_y)

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
        self.environment_pan_x = max(-max_pan_x, min(max_pan_x, self.environment_pan_x))
        self.environment_pan_y = max(-max_pan_y, min(max_pan_y, self.environment_pan_y))

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

        selected_x, selected_y = selected.position
        self.environment_pan_x = -selected_x * self.environment_zoom
        self.environment_pan_y = -selected_y * self.environment_zoom
        self._clamp_environment_pan()

    def _creature_is_visible(self, creature: Creature) -> bool:
        draw_x, draw_y = self.environment_to_screen(*creature.position)
        radius = creature.radius * self.environment_zoom
        bounds = self.layout.environment
        return (
            draw_x + radius >= bounds.left
            and draw_x - radius <= bounds.right
            and draw_y + radius >= bounds.bottom
            and draw_y - radius <= bounds.top
        )

    def _refresh_stats(self) -> None:
        self.stats.herbivore_count = len(self.creatures)
        self.stats.food_count = len(self.foods)
        self.stats.total_biomass_energy = self.total_biomass_energy
        self.stats.creature_energy = self._creature_energy()
        self.stats.plant_energy = self._plant_energy()
        self.stats.available_biomass = self._available_biomass()
        self.stats.plant_spawn_pressure = self._plant_spawn_pressure()
        self.stats.biome_area_shares = self._biome_area_shares()
        self.stats.biome_food_counts = self._biome_food_counts()
        self.rt_neat.update_stats(
            self.creatures,
            self.fitness,
            self.config.population,
            self.config.fitness,
            self.elapsed_time,
        )

    def _update_fitness_survival(self, delta_time: float) -> None:
        for creature in self.creatures:
            fitness = self.fitness.get(creature.creature_id)
            if fitness is not None:
                fitness.record_tick(delta_time, creature.speed, self.MAX_SPEED)
                fitness.record_trait_cost(
                    self.metabolism.trait_energy_cost_per_second(
                        creature,
                        self.MAX_SPEED,
                    ),
                    delta_time,
                )

    def _record_food_discoveries(
        self,
        creature: Creature,
        visible_food_ids: list[int],
    ) -> None:
        fitness = self.fitness.get(creature.creature_id)
        if fitness is None:
            return

        fitness.record_food_discoveries(visible_food_ids)

    def _apply_carry_intent(self, creature: Creature, action: Action) -> None:
        if action.want_release > 0.5:
            self._release_food_for(creature)
            return

        if action.want_grab <= 0.5:
            return

        if creature.creature_id in self._held_food_by_creature_id:
            return

        food = self._nearest_grabbable_food_for(creature)
        if food is None:
            return

        self._held_food_by_creature_id[creature.creature_id] = food.id
        self._carrier_by_food_id[food.id] = creature.creature_id
        self._sync_carried_food(creature, food)

    def _nearest_grabbable_food_for(self, creature: Creature) -> Food | None:
        mouth_x, mouth_y = self.metabolism.mouth_position(creature)
        candidates = [
            food
            for food in self._eatable_foods_for(creature)
            if food.id not in self._carrier_by_food_id
            and self.metabolism.food_overlaps_mouth(creature, food)
        ]
        if not candidates:
            return None

        return min(
            candidates,
            key=lambda food: (
                (food.position[0] - mouth_x) ** 2
                + (food.position[1] - mouth_y) ** 2
            ),
        )

    def _sync_carried_foods(self) -> None:
        held_foods = getattr(self, "_held_food_by_creature_id", None)
        carriers = getattr(self, "_carrier_by_food_id", None)
        if held_foods is None or carriers is None:
            return

        creatures_by_id = {
            creature.creature_id: creature
            for creature in self.creatures
        }

        for creature_id, food_id in list(held_foods.items()):
            creature = creatures_by_id.get(creature_id)
            food = self._food_by_id(food_id)
            if creature is None or food is None:
                held_foods.pop(creature_id, None)
                carriers.pop(food_id, None)
                continue

            self._sync_carried_food(creature, food)

    def _sync_carried_food(self, creature: Creature, food: Food) -> None:
        offset = creature.radius + food.radius * 0.5
        food.body.position = (
            creature.position[0] + cos(creature.heading) * offset,
            creature.position[1] + sin(creature.heading) * offset,
        )
        food.body.velocity = creature.body.velocity
        food.body.angular_velocity = creature.body.angular_velocity
        self._reindex_food(food)
        reindex_shape = getattr(self.space, "reindex_shape", None)
        if reindex_shape is not None:
            reindex_shape(food.shape)

    def _release_food_for(self, creature: Creature) -> None:
        held_foods = getattr(self, "_held_food_by_creature_id", None)
        carriers = getattr(self, "_carrier_by_food_id", None)
        if held_foods is None or carriers is None:
            return

        food_id = held_foods.pop(creature.creature_id, None)
        if food_id is None:
            return
        carriers.pop(food_id, None)

    def _clear_food_carry(self, food: Food) -> None:
        held_foods = getattr(self, "_held_food_by_creature_id", None)
        carriers = getattr(self, "_carrier_by_food_id", None)
        if held_foods is None or carriers is None:
            return

        carrier_id = carriers.pop(food.id, None)
        if carrier_id is not None:
            held_foods.pop(carrier_id, None)

    def _ignored_food_ids_for(self, creature: Creature) -> set[int]:
        held_foods = getattr(self, "_held_food_by_creature_id", None)
        if held_foods is None:
            return set()
        food_id = held_foods.get(creature.creature_id)
        return set() if food_id is None else {food_id}

    def _food_by_id(self, food_id: int) -> Food | None:
        for food in self.foods:
            if food.id == food_id:
                return food
        return None

    def _update_metabolism(self, delta_time: float) -> None:
        report = self.metabolism.update(
            self.creatures,
            self.foods,
            delta_time,
            self.MAX_SPEED,
            self._eatable_foods_for,
            self._creature_want_to_eat,
        )

        for consumption in report.food_consumptions:
            fitness = self.fitness.get(consumption.creature_id)
            if fitness is not None:
                fitness.record_food(
                    consumption.energy_gained,
                    depleted=consumption.depleted,
                )

        for food in report.touched_foods:
            reindex_shape = getattr(self.space, "reindex_shape", None)
            if reindex_shape is not None and food in self.foods:
                reindex_shape(food.shape)

        for food in report.depleted_foods:
            if food in self.foods:
                self._clear_food_carry(food)
                self.foods.remove(food)
                self._unindex_food(food)
                self.space.remove(food.body, food.shape)

        for creature in report.dead_creatures:
            self._remove_creature(creature)

        if not self.creatures:
            self._recover_extinct_population()

        if self.selected_creature_id is not None and self.selected_creature is None:
            self.selected_creature_id = None

    def _eatable_foods_for(self, creature: Creature) -> list[Food]:
        radius = (
            creature.radius
            + self.config.food.max_food_radius
            + self.config.metabolism.eating_distance
        )
        return self._nearby_foods_for(creature, radius)

    def _nearby_foods_for(self, creature: Creature, radius: float) -> list[Food]:
        creature_x, creature_y = creature.position
        left = creature_x - radius
        right = creature_x + radius
        bottom = creature_y - radius
        top = creature_y + radius
        return self._foods_in_world_bounds(left, bottom, right, top)

    def _foods_in_world_bounds(
        self,
        left: float,
        bottom: float,
        right: float,
        top: float,
    ) -> list[Food]:
        self._ensure_food_grid()
        min_cell_x, min_cell_y = self._food_grid_cell(left, bottom)
        max_cell_x, max_cell_y = self._food_grid_cell(right, top)

        foods: list[Food] = []
        for cell_x in range(min_cell_x, max_cell_x + 1):
            for cell_y in range(min_cell_y, max_cell_y + 1):
                foods.extend(self._food_grid.get((cell_x, cell_y), []))

        return foods

    def _circle_intersects_world_bounds(
        self,
        x: float,
        y: float,
        radius: float,
        left: float,
        bottom: float,
        right: float,
        top: float,
    ) -> bool:
        return (
            x + radius >= left
            and x - radius <= right
            and y + radius >= bottom
            and y - radius <= top
        )

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def _ensure_food_grid(self) -> None:
        if not self._food_grid_dirty:
            return

        self._food_grid.clear()
        self._food_grid_cells_by_id = {}
        for food in self.foods:
            self._index_food(food)
        self._food_grid_dirty = False

    def _index_food(self, food: Food) -> None:
        self._ensure_food_grid_storage()
        cell = self._food_grid_cell(*food.position)
        self._food_grid.setdefault(cell, []).append(food)
        self._food_grid_cells_by_id[self._food_grid_key(food)] = cell

    def _reindex_food(self, food: Food) -> None:
        self._ensure_food_grid_storage()
        if self._food_grid_dirty:
            return

        updated_cell = self._food_grid_cell(*food.position)
        food_key = self._food_grid_key(food)
        current_cell = self._food_grid_cells_by_id.get(food_key)
        if current_cell == updated_cell:
            return

        self._unindex_food(food)
        self._food_grid.setdefault(updated_cell, []).append(food)
        self._food_grid_cells_by_id[food_key] = updated_cell

    def _unindex_food(self, food: Food) -> None:
        self._ensure_food_grid_storage()
        cell = self._food_grid_cells_by_id.pop(self._food_grid_key(food), None)
        if cell is None:
            return

        foods = self._food_grid.get(cell)
        if foods is None:
            return
        try:
            foods.remove(food)
        except ValueError:
            return
        if not foods:
            self._food_grid.pop(cell, None)

    def _ensure_food_grid_storage(self) -> None:
        if not hasattr(self, "_food_grid"):
            self._food_grid = {}
        if not hasattr(self, "_food_grid_cells_by_id"):
            self._food_grid_cells_by_id = {}
        if not hasattr(self, "_food_grid_dirty"):
            self._food_grid_dirty = True
        if not hasattr(self, "_food_grid_cell_size"):
            self._food_grid_cell_size = 100.0

    def _food_grid_key(self, food: Food) -> int:
        return getattr(food, "id", id(food))

    def _food_grid_cell(self, x: float, y: float) -> tuple[int, int]:
        return (
            floor(x / self._food_grid_cell_size),
            floor(y / self._food_grid_cell_size),
        )

    def _remove_creature(self, creature: Creature) -> None:
        self._archive_creature_traits(creature)
        self._release_food_for(creature)
        fitness = self.fitness.get(creature.creature_id)
        if fitness is not None:
            self.neat_controller.archive_brain(
                creature.creature_id, fitness.score(self.config.fitness)
            )

        if creature in self.creatures:
            self.creatures.remove(creature)
            self.space.remove(creature.body, creature.shape)
            self.neat_controller.remove_brain(creature.creature_id)
            self._last_actions.pop(creature.creature_id, None)

        fitness = self.fitness.pop(creature.creature_id, None)
        self.rt_neat.record_death(fitness)
        if fitness is not None:
            self.fitness_archive[creature.creature_id] = fitness

        if self.selected_creature_id == creature.creature_id:
            self.selected_creature_id = None

        self._chronometers.pop(creature.creature_id, None)

    def _archive_creature_traits(self, creature: Creature) -> None:
        genome_id_for = getattr(self.neat_controller, "genome_id_for", None)
        if genome_id_for is None:
            return

        genome_id = genome_id_for(creature.creature_id)
        if genome_id is None:
            return

        if not hasattr(self, "_trait_archive_by_genome_id"):
            self._trait_archive_by_genome_id = {}

        self._trait_archive_by_genome_id[genome_id] = ArchivedCreatureTraits(
            creature_id=creature.creature_id,
            vision=VisionTraits(
                range=creature.vision.range,
                angle=creature.vision.angle,
            ),
            physical_traits=PhysicalTraits(
                radius=creature.physical_traits.radius,
                movement_cost_multiplier=(
                    creature.physical_traits.movement_cost_multiplier
                ),
            ),
            color=creature.color,
            lineage=LineageInfo(
                parent_id=creature.lineage.parent_id,
                generation=creature.lineage.generation,
                mutation_delta=TraitMutationDelta(
                    vision_range=creature.lineage.mutation_delta.vision_range,
                    vision_angle=creature.lineage.mutation_delta.vision_angle,
                    radius=creature.lineage.mutation_delta.radius,
                    movement_cost_multiplier=(
                        creature.lineage.mutation_delta.movement_cost_multiplier
                    ),
                ),
            ),
        )

    def _initial_total_biomass_energy(self) -> float:
        configured_total = self.config.food.total_biomass_energy
        if configured_total is not None:
            return configured_total
        return self._creature_energy() + self._plant_energy()

    def _creature_energy(self) -> float:
        return sum(creature.energy for creature in self.creatures)

    def _plant_energy(self) -> float:
        return sum(food.energy_value for food in self.foods)

    def _available_biomass(self) -> float:
        used_biomass = self._creature_energy() + self._plant_energy()
        return max(0.0, self.total_biomass_energy - used_biomass)

    def _plant_spawn_pressure(self) -> float:
        return self.food_spawner.creature_pressure_factor(len(self.creatures))

    def _biome_area_shares(self) -> dict[str, float]:
        biome_map = getattr(self, "biome_map", None)
        if biome_map is None:
            return {biome.label: 0.0 for biome in Biome}
        return {
            biome.label: biome_map.area_shares.get(biome, 0.0)
            for biome in Biome
        }

    def _biome_food_counts(self) -> dict[str, int]:
        counts = {biome.label: 0 for biome in Biome}
        biome_map = getattr(self, "biome_map", None)
        if biome_map is None:
            return counts
        for food in self.foods:
            counts[biome_map.biome_at(*food.position).label] += 1
        return counts

    def _recover_extinct_population(self) -> None:
        parent_pool_size = max(
            1,
            self.config.population.extinction_recovery_parent_pool,
        )
        parent_genomes = self.neat_controller.best_genomes(parent_pool_size)
        if not parent_genomes:
            return

        available_creature_slots = max(
            0,
            self.config.population.max_creatures - len(self.creatures),
        )
        recovery_count = min(
            self.config.population.extinction_recovery_creatures,
            available_creature_slots,
        )

        recovered_count = 0
        for index in range(recovery_count):
            parent_genome = parent_genomes[index % len(parent_genomes)]
            archived_traits = self._archived_traits_for_genome(parent_genome)
            child_traits = (
                self._mutated_recovery_traits(archived_traits)
                if archived_traits is not None
                else None
            )
            child_id = self._next_creature_id()
            child = self._spawn_creature(
                child_id,
                energy=self.config.metabolism.max_energy,
                color=(
                    child_traits.color
                    if child_traits is not None
                    else self._initial_creature_color(recovered_count)
                ),
                vision=child_traits.vision if child_traits is not None else None,
                physical_traits=(
                    child_traits.physical_traits
                    if child_traits is not None
                    else None
                ),
                lineage=child_traits.lineage if child_traits is not None else None,
            )
            if not self.neat_controller.create_mutated_brain_from_genome(
                parent_genome,
                child_id,
            ):
                self.space.remove(child.body, child.shape)
                continue

            self.creatures.append(child)
            self.fitness[child_id] = CreatureFitness()
            self._chronometers[child_id] = 0.0
            recovered_count += 1

        self.rt_neat.record_extinction_replacements(recovered_count)

    def _archived_traits_for_genome(
        self,
        genome: object,
    ) -> ArchivedCreatureTraits | None:
        genome_id = getattr(genome, "key", None)
        if genome_id is None:
            return None
        return getattr(self, "_trait_archive_by_genome_id", {}).get(genome_id)

    def _mutated_recovery_traits(
        self,
        archived_traits: ArchivedCreatureTraits,
    ) -> ChildCreatureTraits:
        return self._mutated_child_traits_from_parent_values(
            parent_id=archived_traits.creature_id,
            parent_generation=archived_traits.lineage.generation,
            parent_vision=archived_traits.vision,
            parent_physical_traits=archived_traits.physical_traits,
            parent_color=archived_traits.color,
        )

    def _update_reproduction(self, delta_time: float) -> None:
        self._reproduction_accumulator += delta_time
        if self._reproduction_accumulator < self.REPRODUCTION_INTERVAL:
            return

        self._reproduction_accumulator %= self.REPRODUCTION_INTERVAL
        self._refresh_stats()
        self._try_reproduce()

    def _spend_reproduction_energy(self, parent: Creature) -> None:
        parent.energy = max(
            0.0,
            parent.energy - self.config.population.reproduction_energy_cost,
        )

    def _try_reproduce(self) -> bool:
        if len(self.creatures) >= self.config.population.max_creatures:
            return False

        if not self._has_reproduction_resources():
            return False

        if not self.rt_neat.eligible_parent_ids:
            return False

        parent = self._reproduction_parent()
        if parent is None:
            return False

        parent_fitness = self.fitness[parent.creature_id]
        child_id = self._next_creature_id()
        child_traits = self._mutated_child_traits(parent)
        child_position = self._child_spawn_position(
            parent,
            child_traits.physical_traits.radius,
        )

        child = self._spawn_creature(
            child_id,
            position=child_position,
            heading=parent.heading,
            energy=self.config.population.reproduction_energy_cost,
            color=child_traits.color,
            vision=child_traits.vision,
            physical_traits=child_traits.physical_traits,
            lineage=child_traits.lineage,
        )

        if not self.neat_controller.create_child_brain(parent.creature_id, child_id):
            self.space.remove(child.body, child.shape)
            return False

        self.creatures.append(child)
        self.fitness[child_id] = CreatureFitness()
        self._chronometers[child_id] = 0.0

        self._spend_reproduction_energy(parent)
        parent_fitness.record_reproduction()
        self.rt_neat.record_normal_replacement()
        return True

    def _has_reproduction_resources(self) -> bool:
        child_energy = self.config.population.reproduction_energy_cost
        available_biomass = self._available_biomass()
        if available_biomass < child_energy:
            return False

        food_capacity = self.food_spawner.food_capacity(len(self.creatures))
        food_ratio = len(self.foods) / max(1, food_capacity)
        if food_ratio >= self.config.population.reproduction_min_food_ratio:
            return True

        total_biomass = max(self.total_biomass_energy, child_energy)
        available_biomass_ratio = available_biomass / total_biomass
        return (
            self._plant_spawn_pressure()
            >= self.config.population.reproduction_recovery_pressure_threshold
            and available_biomass_ratio
            >= self.config.population.reproduction_min_available_biomass_ratio
        )

    def _reproduction_parent(self) -> Creature | None:
        live_creatures = {
            creature.creature_id: creature
            for creature in self.creatures
        }
        for parent_id in self.rt_neat.eligible_parent_ids:
            parent = live_creatures.get(parent_id)
            if parent is None or parent.creature_id not in self.fitness:
                continue

            parent_action = self._last_actions.get(parent.creature_id)
            if parent_action is not None and parent_action.want_reproduce >= 0.5:
                return parent

        return None

    def _creature_want_to_eat(self, creature: Creature) -> bool:
        action = self._last_actions.get(creature.creature_id)
        if action is None:
            return False
        return action.want_eat >= 0.5

    def _settle_food_motion(self) -> None:
        for food in self.foods:
            food.body.velocity *= 0.84
            food.body.angular_velocity *= 0.62

            if food.body.velocity.length < 0.75:
                food.body.velocity = (0.0, 0.0)

            if abs(food.body.angular_velocity) < 0.2:
                food.body.angular_velocity = 0.0

            self._reindex_food(food)
