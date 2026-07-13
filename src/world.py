from __future__ import annotations

from colorsys import hsv_to_rgb, rgb_to_hsv
from dataclasses import dataclass, field
from math import atan2, cos, floor, hypot, pi, sin
from random import Random, choice

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
from src.vision import BiomeSensorSnapshot, SensorSnapshot, VisionSystem
from src.controller import BaselineFoodController
from src.neat_controller import NeatBrainController, SpeciationResult
from src.persistence import PersistenceManager, SavePriority, SimulationPaths
from src.rt_neat import RtNeatManager
from src.speciation import (
    NeatChangeSummary,
    SpeciesDistanceBreakdown,
    SpeciesRecord,
    SpeciesTraitSnapshot,
)
from src.telemetry import TelemetryDatabase
from src.collision import BOUNDARY_CATEGORY, CREATURE_CATEGORY, FOOD_CATEGORY
from src.communication import AcousticSignal, AcousticSystem, PheromoneSystem

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


@dataclass(slots=True)
class MotionCommand:
    effective_rotate: float
    max_speed: float
    max_angular_speed: float


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

    def __init__(
        self,
        config: SimConfig,
        *,
        bootstrap: bool = True,
        simulation_paths: SimulationPaths | None = None,
    ) -> None:
        self.config = config
        self.rng = Random(7)
        self.elapsed_time = 0.0
        self.fps = 0.0
        self.is_paused = False
        self.simulation_speed = 1.0
        self.show_biome_background = False
        self._physics_accumulator = 0.0
        self._reproduction_accumulator = 0.0
        self._speciation_adjustment_accumulator = 0.0
        self.physics_step_count = 0
        self._last_actions: dict[int, Action] = {}
        self._last_sensor_snapshots: dict[int, SensorSnapshot] = {}
        self._motion_commands: dict[int, MotionCommand] = {}
        self.debug_vision_enabled = config.debug.show_debug_vision_by_default
        self.layout = build_screen_layout(
            config.display.width, config.display.height, config.layout
        )
        self.environment_zoom = config.zoom.default
        self.environment_pan_x = 0.0
        self.environment_pan_y = 0.0
        self.vision = VisionSystem(
            config.vision,
            config.metabolism.eating_distance,
            config.metabolism.stomach_capacity_per_radius,
        )
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = 0.90
        self.space.iterations = 12
        use_spatial_hash = getattr(self.space, "use_spatial_hash", None)
        if use_spatial_hash is not None:
            try:
                use_spatial_hash(cell_size=50.0, static_amount=1000)
            except TypeError:
                use_spatial_hash(50.0, 1000)
        self._creature_by_shape_id: dict[int, Creature] = {}
        self._boundary_shapes: list[pymunk.Shape] = []
        self._rebuild_boundaries()
        self.creatures = self._spawn_creatures() if bootstrap else []
        self._next_creature_id_value = (
            max(
                (creature.creature_id for creature in self.creatures),
                default=0,
            )
            + 1
        )
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
        self.acoustics = AcousticSystem(config.communication)
        self.pheromones = PheromoneSystem(
            config.communication,
            self.biome_map.grid_width,
            self.biome_map.grid_height,
            self.environment_world_bounds,
        )
        for creature in self.creatures:
            self._initialize_creature_fertility_baseline(creature)
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
        if bootstrap:
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

        self.baseline_controller = BaselineFoodController(self.config.action)
        self.neat_controller = NeatBrainController(
            "configs/neat_herbivore.ini",
            compatibility_threshold=config.speciation.compatibility_threshold,
            phenotypic_weight=config.speciation.phenotypic_weight,
            trait_config=config.trait,
            vision_config=config.vision,
        )
        if bootstrap:
            self.neat_controller.assign_initial_brains(self.creatures)
        self.metabolism = Metabolism(
            config.metabolism,
            self.vision,
            config.trait,
            genome_for_creature_id=self._genome_for_creature_id,
            communication_config=config.communication,
        )
        self.rt_neat = RtNeatManager(self.neat_controller)
        self._trait_archive_by_genome_id: dict[int, ArchivedCreatureTraits] = {}
        self.species_history: dict[int, SpeciesRecord] = {}
        if bootstrap:
            self._initialize_luca_record()
        self.use_neat_brains = config.controller.use_neat_brains
        self.show_brain_view = False
        self.time_since_last_quick_save = 0.0
        self.time_since_last_archive_save = 0.0
        self.simulation_paths = simulation_paths or SimulationPaths.create_new(
            config.persistence
        )
        self.telemetry = (
            TelemetryDatabase(self.simulation_paths.telemetry_database)
            if config.persistence.enable_telemetry
            else None
        )
        self.persistence_manager = PersistenceManager()
        self._closed = False
        if bootstrap:
            self._log_initial_telemetry()

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

        self._update_speciation_threshold(delta_time)
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
            self._commit_communication_intents(self.FIXED_TIMESTEP)
            self.space.step(self.FIXED_TIMESTEP)
            self.physics_step_count += 1
            self._settle_food_motion()
            self._apply_top_down_motion()
            self._limit_creature_motion()
            self._sync_carried_foods()
            self._update_fitness_survival(self.FIXED_TIMESTEP)
            self._update_chronometers(self.FIXED_TIMESTEP)
            self._update_metabolism(self.FIXED_TIMESTEP)
            self.pheromones.accumulate(self.FIXED_TIMESTEP)
            self._physics_accumulator -= self.FIXED_TIMESTEP
            steps += 1

        self._spawn_foods(scaled_delta_time)
        self._update_reproduction(scaled_delta_time)
        self._refresh_stats()
        self._follow_selected_creature()
        self._update_persistence_timer(delta_time)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.persistence_manager.close()
        finally:
            if self.telemetry is not None:
                self.telemetry.close()

    @property
    def save_in_progress(self) -> bool:
        return self.persistence_manager.is_busy

    def save_now(self) -> None:
        previous_quick_timer = self.time_since_last_quick_save
        self.time_since_last_quick_save = 0.0
        try:
            self.persistence_manager.save_simulation(
                self,
                self.neat_controller,
                (self.simulation_paths.quick_target(),),
                priority=SavePriority.MANUAL,
            )
        except BaseException:
            self.time_since_last_quick_save = previous_quick_timer
            raise
        self._log_persistence_metrics()

    def _update_persistence_timer(self, delta_time: float) -> None:
        quick_interval = self.config.persistence.quick_save_interval_seconds
        archive_interval = self.config.persistence.archive_save_interval_seconds
        if quick_interval <= 0.0 and archive_interval <= 0.0:
            return

        elapsed = max(0.0, delta_time)
        if quick_interval > 0.0:
            self.time_since_last_quick_save += elapsed
        if archive_interval > 0.0:
            self.time_since_last_archive_save += elapsed
        quick_due = (
            quick_interval > 0.0 and self.time_since_last_quick_save >= quick_interval
        )
        archive_due = (
            archive_interval > 0.0
            and self.time_since_last_archive_save >= archive_interval
        )
        if not quick_due and not archive_due:
            return

        targets = [self.simulation_paths.quick_target()]
        if quick_due:
            self.time_since_last_quick_save %= quick_interval
        if archive_due:
            self.time_since_last_archive_save %= archive_interval
            targets.append(self.simulation_paths.hourly_target())

        self._log_persistence_metrics()
        self.persistence_manager.save_simulation(
            self,
            self.neat_controller,
            tuple(targets),
            priority=SavePriority.AUTO,
        )

    def _log_persistence_metrics(self) -> None:
        telemetry = self.telemetry
        if telemetry is None:
            return
        telemetry.log_metrics(
            self.elapsed_time,
            len(self.creatures),
            len(self.foods),
            self.rt_neat.stats.best_fitness,
        )

    def _log_initial_telemetry(self) -> None:
        telemetry = self.telemetry
        if telemetry is None:
            return
        luca_record = self.species_history.get(1)
        if luca_record is not None:
            telemetry.log_species_record(luca_record)
        for creature in self.creatures:
            self._log_creature_birth(creature)

    def _log_creature_birth(self, creature: Creature) -> None:
        telemetry = getattr(self, "telemetry", None)
        if telemetry is None:
            return
        telemetry.log_creature_birth(
            creature.creature_id,
            creature.lineage.species_id,
            self.elapsed_time,
            creature.vision.range,
            creature.radius,
        )

    def _initialize_luca_record(self) -> None:
        if not self.creatures:
            return
        founder = self.creatures[0]
        food_ratio, population_ratio = self._species_emergence_ratios()
        zero_traits = SpeciesTraitSnapshot(0.0, 0.0, 0.0, 0.0)
        self.species_history[1] = SpeciesRecord(
            species_id=1,
            parent_species_id=None,
            founder_creature_id=founder.creature_id,
            founder_genome_id=self.neat_controller.genome_id_for(founder.creature_id),
            emerged_at=self.elapsed_time,
            founder_color=tuple(founder.color[:3]),
            data_quality="exact",
            founder_traits=SpeciesTraitSnapshot.from_traits(
                founder.physical_traits,
                founder.vision,
            ),
            trait_deltas=zero_traits,
            distances=SpeciesDistanceBreakdown(
                neat_distance=0.0,
                phenotypic_distance=0.0,
                weighted_phenotypic_distance=0.0,
                composite_distance=0.0,
                compatibility_threshold=(
                    self.neat_controller.species_manager.compatibility_threshold
                ),
                phenotypic_weight=(
                    self.neat_controller.species_manager.phenotypic_weight
                ),
                radius_component=0.0,
                vision_range_component=0.0,
                vision_angle_component=0.0,
                movement_cost_component=0.0,
            ),
            neat_changes=NeatChangeSummary.empty(),
            emergence_food_ratio=food_ratio,
            emergence_pop_ratio=population_ratio,
            neural_shifts=(),
        )

    def _record_new_species(
        self,
        founder: Creature,
        result: SpeciationResult,
    ) -> None:
        food_ratio, population_ratio = self._species_emergence_ratios()
        record = SpeciesRecord(
            species_id=result.species_id,
            parent_species_id=result.parent_species_id,
            founder_creature_id=founder.creature_id,
            founder_genome_id=self.neat_controller.genome_id_for(founder.creature_id),
            emerged_at=self.elapsed_time,
            founder_color=tuple(founder.color[:3]),
            data_quality="exact",
            founder_traits=result.founder_traits,
            trait_deltas=result.trait_deltas,
            distances=result.distances,
            neat_changes=result.neat_changes,
            emergence_food_ratio=food_ratio,
            emergence_pop_ratio=population_ratio,
            neural_shifts=result.neural_shifts,
        )
        self.species_history[record.species_id] = record
        telemetry = getattr(self, "telemetry", None)
        if telemetry is not None:
            telemetry.log_species_record(record)

    def _species_emergence_ratios(self) -> tuple[float | None, float | None]:
        creatures = getattr(self, "creatures", None)
        foods = getattr(self, "foods", None)
        food_spawner = getattr(self, "food_spawner", None)
        config = getattr(self, "config", None)
        if creatures is None or config is None:
            return None, None

        max_creatures = max(1, int(config.population.max_creatures))
        population_ratio = len(creatures) / max_creatures
        if foods is None or food_spawner is None:
            return None, population_ratio
        food_capacity = food_spawner.food_capacity(len(creatures))
        return len(foods) / max(1, food_capacity), population_ratio

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
        queried = self._creatures_in_world_bounds(left, bottom, right, top)
        if queried is not None:
            return [
                creature
                for creature in queried
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

    def _nearby_creatures_for(
        self,
        creature: Creature,
        radius: float,
    ) -> list[Creature]:
        center_x, center_y = creature.position
        queried = self._creatures_in_world_bounds(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        )
        candidates = self.creatures if queried is None else queried
        max_distance = max(0.0, radius)
        return [
            other
            for other in candidates
            if hypot(other.position[0] - center_x, other.position[1] - center_y)
            <= max_distance + other.radius
        ]

    def _creatures_in_world_bounds(
        self,
        left: float,
        bottom: float,
        right: float,
        top: float,
    ) -> list[Creature] | None:
        space = getattr(self, "space", None)
        bb_query = getattr(space, "bb_query", None)
        bb_factory = getattr(pymunk, "BB", None)
        filter_factory = getattr(pymunk, "ShapeFilter", None)
        index = getattr(self, "_creature_by_shape_id", None)
        if (
            bb_query is None
            or bb_factory is None
            or filter_factory is None
            or index is None
        ):
            return None

        try:
            viewport_bb = bb_factory(left, bottom, right, top)
            shape_filter = filter_factory(mask=CREATURE_CATEGORY)
            shapes = bb_query(viewport_bb, shape_filter)
        except Exception:
            return None

        creatures: list[Creature] = []
        seen_ids: set[int] = set()
        for shape in shapes:
            creature = index.get(id(shape))
            if creature is None:
                continue
            if creature.creature_id in seen_ids:
                continue
            seen_ids.add(creature.creature_id)
            creatures.append(creature)
        return creatures

    @property
    def selected_creature(self) -> Creature | None:
        for creature in self.creatures:
            if creature.creature_id == self.selected_creature_id:
                return creature
        return None

    def biome_sensor_positions_for(
        self,
        creature: Creature,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        center_x, center_y = creature.position
        heading_x = cos(creature.heading)
        heading_y = sin(creature.heading)
        left_x = -heading_y
        left_y = heading_x
        right_x = heading_y
        right_y = -heading_x
        forward_distance = max(0.0, self.config.biome_sensor.forward_distance)
        side_offset = max(0.0, self.config.biome_sensor.side_offset)

        here = (center_x, center_y)
        forward_x = heading_x * forward_distance
        forward_y = heading_y * forward_distance
        forward_left = (
            center_x + forward_x + left_x * side_offset,
            center_y + forward_y + left_y * side_offset,
        )
        forward_right = (
            center_x + forward_x + right_x * side_offset,
            center_y + forward_y + right_y * side_offset,
        )
        return here, forward_left, forward_right

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
        nearby_creatures = self._nearby_creatures_for(
            creature,
            creature.vision.range + self.config.trait.max_radius,
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
                nearby_creatures,
                self.environment_world_bounds,
                self.MAX_SPEED,
                maturity=maturity,
                clock_tik_tok=clock_tik_tok,
                clock_chronometer=clock_chronometer,
                clock_time_alive=clock_time_alive,
                is_grabbing=is_grabbing,
                ignored_food_ids=ignored_food_ids,
                own_infants=self._own_infant_children_for(creature),
            )
            self._record_food_discoveries(creature, result.visible_food_ids)
            snapshot = result.snapshot
        else:
            snapshot = self.vision.sense(
                creature,
                nearby_foods,
                nearby_creatures,
                self.environment_world_bounds,
                self.MAX_SPEED,
                maturity=maturity,
                clock_tik_tok=clock_tik_tok,
                clock_chronometer=clock_chronometer,
                clock_time_alive=clock_time_alive,
                is_grabbing=is_grabbing,
                ignored_food_ids=ignored_food_ids,
                own_infants=self._own_infant_children_for(creature),
            )

        snapshot.biome = self._biome_sensor_snapshot_for(creature)
        acoustics = getattr(self, "acoustics", None)
        if acoustics is not None:
            snapshot.acoustic = acoustics.sense(
                creature.creature_id,
                creature.position,
                creature.heading,
            )
        pheromones = getattr(self, "pheromones", None)
        if pheromones is not None:
            snapshot.pheromones = pheromones.sense(
                self.pheromone_sensor_positions_for(creature)
            )
        return snapshot

    def pheromone_sensor_positions_for(
        self,
        creature: Creature,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        return self.biome_sensor_positions_for(creature)

    def _biome_sensor_snapshot_for(self, creature: Creature) -> BiomeSensorSnapshot:
        here, forward_left, forward_right = self.biome_sensor_positions_for(creature)
        biome_here = self._biome_fertility_at(*here)
        baseline = getattr(creature, "fertility_baseline", biome_here)
        biome_delta = self._clamp(biome_here - baseline, -1.0, 1.0)

        return BiomeSensorSnapshot(
            here=biome_here,
            forward_left=self._biome_fertility_at(*forward_left),
            forward_right=self._biome_fertility_at(*forward_right),
            delta=biome_delta,
        )

    def _initialize_creature_fertility_baseline(self, creature: Creature) -> None:
        creature.fertility_baseline = self._biome_fertility_at(*creature.position)

    def _adapt_creature_fertility_baseline(
        self,
        creature: Creature,
        snapshot: SensorSnapshot,
    ) -> None:
        baseline = getattr(creature, "fertility_baseline", snapshot.biome.here)
        creature.fertility_baseline = (
            baseline * 0.90 + snapshot.biome.here * 0.10
        )

    def _biome_fertility_at(self, x: float, y: float) -> float:
        biome_map = getattr(self, "biome_map", None)
        if biome_map is None:
            return 0.0
        return self._clamp(float(biome_map.fertility_at(x, y)), 0.0, 1.0)

    def visible_foods_for(self, creature: Creature) -> list[Food]:
        nearby_foods = self._nearby_foods_for(
            creature,
            creature.vision.range + self.config.food.max_food_radius,
        )
        nearby_creatures = self._nearby_creatures_for(
            creature,
            creature.vision.range + self.config.trait.max_radius,
        )
        return self.vision.visible_foods(
            creature,
            nearby_foods,
            nearby_creatures,
            ignored_food_ids=self._ignored_food_ids_for(creature),
        )

    def visible_creatures_for(self, creature: Creature) -> list[Creature]:
        nearby_foods = self._nearby_foods_for(
            creature,
            creature.vision.range + self.config.food.max_food_radius,
        )
        nearby_creatures = self._nearby_creatures_for(
            creature,
            creature.vision.range + self.config.trait.max_radius,
        )
        return self.vision.visible_creatures(
            creature,
            nearby_creatures,
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

        self._remove_creature(selected, death_reason="manual")
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
                color=self._initial_creature_color(0),
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

        creature = Creature(
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
        self._index_creature_shape(shape, creature)
        return creature

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
            parent_species_id=parent.lineage.species_id,
            parent_vision=parent.vision,
            parent_physical_traits=parent.physical_traits,
            parent_color=parent.color,
        )

    def _mutated_child_traits_from_parent_values(
        self,
        parent_id: int | None,
        parent_generation: int,
        parent_species_id: int,
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
                species_id=parent_species_id,
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

    def _new_species_color(self, parent_color: Color) -> Color:
        red, green, blue = parent_color[:3]
        parent_hue, _, _ = rgb_to_hsv(
            red / 255.0,
            green / 255.0,
            blue / 255.0,
        )

        for _ in range(32):
            hue = (parent_hue + self.rng.uniform(0.18, 0.82)) % 1.0
            saturation = self.rng.uniform(0.7, 1.0)
            value = self.rng.uniform(0.8, 1.0)
            candidate = hsv_to_rgb(hue, saturation, value)
            color = tuple(int(channel * 255) for channel in candidate)
            normalized_color = tuple(channel / 255.0 for channel in color)
            if not self._is_food_like_color(normalized_color):
                return color

        for hue_shift in (0.5, 1.0 / 3.0, 2.0 / 3.0):
            candidate = hsv_to_rgb((parent_hue + hue_shift) % 1.0, 0.85, 0.9)
            color = tuple(int(channel * 255) for channel in candidate)
            normalized_color = tuple(channel / 255.0 for channel in color)
            if not self._is_food_like_color(normalized_color):
                return color

        candidate = hsv_to_rgb((parent_hue + 0.5) % 1.0, 1.0, 1.0)
        return tuple(int(channel * 255) for channel in candidate)

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
        """
        Calculate a suitable spawn position for a child creature based on the parent's
        position, heading, and the child's radius. The spawn position is determined by
        ensuring that the child is placed at a safe distance from the parent and within
        the bounds of the environment. The method takes into account the configured
        child spawn distance, the parent's radius, and the child's radius to calculate
        the appropriate spawn position. The resulting position is clamped to ensure that
        the child does not spawn outside the environment's world bounds.

        Args:
            parent (Creature): The parent creature from which the child will spawn.
            child_radius (float): The radius of the child creature.

        Returns:
            tuple[float, float]: The calculated spawn position (x, y) for the child creature.
        """

        # Calculate the distance at which the child should spawn from the parent, ensuring it is at least the configured child spawn distance and accounting for the parent's and child's radii.
        distance = max(
            self.config.population.child_spawn_distance,
            parent.radius + child_radius + 2.0,
        )

        # Calculate the raw spawn position based on the parent's position and heading, moving backward along the parent's heading by the calculated distance.
        parent_x, parent_y = parent.position
        angle = parent.heading + choice((-pi / 4, pi / 4))
        raw_x = (
            parent_x + cos(angle) * distance
        )  # Calculate the x-coordinate of the spawn position based on the parent's heading and distance.
        raw_y = (
            parent_y + sin(angle) * distance
        )  # Calculate the y-coordinate of the spawn position based on the parent's heading and distance.

        # Get the bounds of the environment to ensure the child spawns within these limits.
        left, bottom, right, top = (
            self.environment_world_bounds
        )  # Get the bounds of the environment to ensure the child spawns within these limits.
        radius = (
            child_radius + 2.0
        )  # Calculate the effective radius to ensure the child does not spawn too close to the environment boundaries.
        spawn_x = max(
            left + radius, min(right - radius, raw_x)
        )  # Clamp the x-coordinate of the spawn position to ensure it is within the environment bounds, accounting for the effective radius.
        spawn_y = max(
            bottom + radius, min(top - radius, raw_y)
        )  # Clamp the y-coordinate of the spawn position to ensure it is within the environment bounds, accounting for the effective radius.
        return spawn_x, spawn_y

    def _next_creature_id(self) -> int:
        if not hasattr(self, "_next_creature_id_value"):
            self._next_creature_id_value = (
                max(
                    [
                        0,
                        *(creature.creature_id for creature in self.creatures),
                        *self.fitness,
                        *self.fitness_archive,
                    ]
                )
                + 1
            )
        creature_id = self._next_creature_id_value
        self._next_creature_id_value += 1
        return creature_id

    def _spawn_foods(self, delta_time: float) -> None:
        spawned_foods = self.food_spawner.update(
            delta_time,
            self.environment_world_bounds,
            len(self.foods),
            self._active_species_count(),
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
        """
        Apply the intents of all creatures in the simulation, based on their
        sensor snapshots and the decisions made by their respective controllers.
        """
        if not hasattr(self, "_last_sensor_snapshots"):
            self._last_sensor_snapshots = {}
        if not hasattr(self, "_last_actions"):
            self._last_actions = {}

        for creature in self.creatures:
            creature_id = creature.creature_id
            action = self._last_actions.get(creature_id)
            snapshot = self._last_sensor_snapshots.get(creature_id)
            should_think = (
                action is None
                or snapshot is None
                or (getattr(self, "physics_step_count", 0) + creature_id) % 2 == 0
            )

            if should_think:
                snapshot = self._sensor_snapshot_for(
                    creature,
                    record_food_discoveries=True,
                )
                if self.use_neat_brains:
                    action = self.neat_controller.decide(creature_id, snapshot)
                else:
                    action = self.baseline_controller.decide(snapshot, creature_id)

                self._last_actions[creature.creature_id] = action
                self._last_sensor_snapshots[creature.creature_id] = snapshot
                try:
                    creature.last_action = action
                except AttributeError:
                    pass

                if action.reset_chronometer >= 0.5:
                    self._chronometers[creature.creature_id] = 0.0

                self._apply_carry_intent(creature, action)
                self._adapt_creature_fertility_baseline(creature, snapshot)

            if action is None:
                continue

            stabilize_velocity = (
                False
                if self.use_neat_brains or snapshot is None
                else snapshot.food.visible > 0.0
            )
            self._apply_action(
                creature,
                action,
                snapshot,
                stabilize_velocity=stabilize_velocity,
                apply_stabilizers=not self.use_neat_brains,
            )

    def _commit_communication_intents(self, delta_time: float) -> None:
        acoustics = getattr(self, "acoustics", None)
        pheromones = getattr(self, "pheromones", None)
        if acoustics is None or pheromones is None:
            return

        signals: list[AcousticSignal] = []
        deposit_rate = max(0.0, self.config.communication.pheromone_deposit_rate)
        elapsed = max(0.0, delta_time)
        for creature in self.creatures:
            action = self._last_actions.get(creature.creature_id)
            if action is None:
                continue
            sound_strength = max(0.0, min(1.0, action.emit_sound))
            if sound_strength >= self.config.communication.acoustic_min_emission:
                signals.append(
                    AcousticSignal(
                        emitter_id=creature.creature_id,
                        position=creature.position,
                        strength=sound_strength,
                        tone=max(-1.0, min(1.0, action.sound_tone)),
                    )
                )
            pheromones.deposit(
                creature.position,
                trail_amount=(
                    deposit_rate
                    * max(0.0, min(1.0, action.emit_trail_pheromone))
                    * elapsed
                ),
                alarm_amount=(
                    deposit_rate
                    * max(0.0, min(1.0, action.emit_alarm_pheromone))
                    * elapsed
                ),
            )
        acoustics.replace_signals(signals)

    def _apply_action(
        self,
        creature: Creature,
        action: Action,
        snapshot: SensorSnapshot | None = None,
        stabilize_velocity: bool = False,
        apply_stabilizers: bool = True,
    ) -> None:
        """
        Apply the specified action to the given creature, considering its current state,
        sensor snapshot, and the simulation's configuration. This method calculates the
        necessary forces and torques to apply to the creature's body based on the action's
        parameters, including acceleration, rotation, and panic intensity. It also handles
        flocking behavior and stabilizing the creature's movement when appropriate.

        Args:
            creature (Creature): The creature to which the action will be applied.
            action (Action): The action to apply, containing acceleration, rotation, and other parameters.
            snapshot (SensorSnapshot | None): The sensor snapshot of the creature, used for flocking calculations. If None, flocking forces will not be applied.
            stabilize_velocity (bool): Whether to stabilize the creature's velocity when moving forward. This is typically used when the creature is actively pursuing food.
            apply_stabilizers (bool): Whether to apply stabilizing forces and torques to the creature's body to reduce unwanted angular velocity and maintain smoother movement.
        """

        # Calculate the target thrust and panic intensity based on the action's parameters.
        target_thrust = action.accelerate
        panic = self._clamp(
            getattr(action, "flee_panic_intensity", 0.0),
            0.0,
            1.0,
        )
        # Calculate the sprint multiplier based on the panic intensity and the maximum allowed sprint multiplier from the configuration.
        sprint_multiplier = 1.0 + (
            panic * max(0.0, self.config.action.max_sprint_multiplier)
        )
        # Calculate the current maximum forward and backward forces, as well as the maximum speed and angular speed, based on the sprint multiplier.

        # These values will be used to limit the forces applied to the creature's body and ensure that it does not exceed its physical capabilities.
        current_max_forward_force = (
            self.config.action.max_forward_force * sprint_multiplier
        )

        # The current maximum backward force is calculated similarly, allowing the creature to move backward with a force that is also scaled by the sprint multiplier.
        current_max_backward_force = (
            self.config.action.max_backward_force * sprint_multiplier
        )

        # The current maximum speed and angular speed are also scaled by the sprint multiplier, allowing the creature to move faster and turn more quickly when in a state of panic or urgency.
        current_max_speed = self.MAX_SPEED * sprint_multiplier
        turn_control_gain = self._clamp(
            self.config.action.turn_control_gain,
            0.0,
            1.0,
        )
        active_angular_velocity_retention = self._clamp(
            self.config.action.active_angular_velocity_retention,
            0.0,
            1.0,
        )
        current_max_angular_speed = (
            self.MAX_ANGULAR_SPEED * sprint_multiplier * turn_control_gain
        )

        alpha = self._clamp(
            self.config.action.action_smoothing_alpha,
            0.0,
            1.0,
        )
        previous_acceleration = getattr(creature, "smoothed_acceleration", 0.0)
        smoothed_acceleration = (
            previous_acceleration * (1.0 - alpha) + target_thrust * alpha
        )
        try:
            creature.smoothed_acceleration = smoothed_acceleration
        except AttributeError:
            pass
        thrust = smoothed_acceleration

        if apply_stabilizers and stabilize_velocity and thrust > 0.0:
            self._stabilize_food_tracking_velocity(creature)

        # Calculate the voluntary force to apply to the creature based on its desired acceleration and heading, as well as the current maximum forward and backward forces. This force represents the creature's intentional movement in the direction it is facing.
        voluntary_force = acceleration_force_vector(
            thrust,
            creature.heading,
            current_max_forward_force,
            current_max_backward_force,
        )

        # Calculate the flocking force to apply to the creature based on its sensor snapshot and the flocking behavior of nearby creatures. This force represents the influence of other creatures in the vicinity, encouraging alignment, cohesion, and separation as appropriate.
        flock_force = self._flock_steering_force(
            creature,
            action,
            snapshot,
            current_max_speed,
            current_max_forward_force,
        )

        # Combine the voluntary force and flocking force to determine the total force to apply to the creature's body. The total force is limited to ensure that it does not exceed the current maximum forward force, preventing unrealistic acceleration.
        total_force = self._limit_vector(
            (
                voluntary_force[0] + flock_force[0],
                voluntary_force[1] + flock_force[1],
            ),
            current_max_forward_force,
        )
        # Calculate the flock turn bias based on the flocking force and the creature's current heading. This bias influences the creature's turning behavior, encouraging it to align with the flock's movement while still allowing for individual decision-making.
        flock_turn_bias = self._flock_turn_bias(
            creature,
            flock_force,
            current_max_forward_force,
        )

        # Determine the target turn based on desired rotation and flock turn bias, then smooth it before applying angular control.
        target_turn = self._clamp(action.rotate + flock_turn_bias, -1.0, 1.0)
        previous_rotation = getattr(creature, "smoothed_rotation", 0.0)
        turn = previous_rotation * (1.0 - alpha) + target_turn * alpha
        try:
            creature.smoothed_rotation = turn
        except AttributeError:
            pass

        if not hasattr(self, "_motion_commands"):
            self._motion_commands = {}
        self._motion_commands[creature.creature_id] = MotionCommand(
            effective_rotate=turn,
            max_speed=current_max_speed,
            max_angular_speed=current_max_angular_speed,
        )

        # Apply the calculated total force to the creature's body at its current position, influencing its movement in the simulation. The force is applied in world coordinates, ensuring that it affects the creature's velocity and trajectory appropriately.
        creature.body.apply_force_at_world_point(
            total_force,
            creature.body.position,
        )

        # Apply stabilizing forces and torques to the creature's body if appropriate, reducing unwanted angular velocity and maintaining smoother movement. This is particularly important when the creature is actively pursuing food or moving in a state of panic, as it helps to prevent erratic behavior and maintain control over its movement.
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

        self._apply_turn_control(
            creature,
            turn,
            max_angular_speed=current_max_angular_speed,
        )
        creature.body.angular_velocity *= active_angular_velocity_retention

        if apply_stabilizers and thrust < 0.0:
            creature.body.angular_velocity *= (
                self.config.action.boundary_angular_velocity_retention
            )

        if apply_stabilizers and not stabilize_velocity and thrust > 0.0:
            creature.body.angular_velocity *= (
                self.config.action.search_angular_velocity_retention
            )

    def _flock_steering_force(
        self,
        creature: Creature,
        action: Action,
        snapshot: SensorSnapshot | None,
        max_speed: float,
        max_force: float,
    ) -> tuple[float, float]:
        if snapshot is None:
            return 0.0, 0.0

        flock = snapshot.flock
        separation = self._steering_toward_relative_angle(
            creature,
            flock.separation_relative_heading,
            max_speed,
            max_force,
            flock.separation_strength,
        )

        if flock.flockmate_count <= 0:
            alignment = (0.0, 0.0)
            cohesion = (0.0, 0.0)
        else:
            alignment = self._steering_toward_relative_angle(
                creature,
                flock.average_relative_heading * pi,
                max_speed,
                max_force,
                flock.average_flockmate_proximity,
            )
            cohesion = self._steering_toward_relative_angle(
                creature,
                flock.center_angle * (creature.vision.angle / 2.0),
                max_speed,
                max_force,
                1.0 - flock.center_proximity,
            )

        return (
            separation[0] * getattr(action, "weight_separation", 0.0)
            + alignment[0] * getattr(action, "weight_alignment", 0.0)
            + cohesion[0] * getattr(action, "weight_cohesion", 0.0),
            separation[1] * getattr(action, "weight_separation", 0.0)
            + alignment[1] * getattr(action, "weight_alignment", 0.0)
            + cohesion[1] * getattr(action, "weight_cohesion", 0.0),
        )

    def _steering_toward_relative_angle(
        self,
        creature: Creature,
        relative_angle: float,
        max_speed: float,
        max_force: float,
        strength: float,
    ) -> tuple[float, float]:
        strength = self._clamp(strength, 0.0, 1.0)
        if strength <= 0.0:
            return 0.0, 0.0

        desired_heading = creature.heading + relative_angle
        desired_velocity = (
            cos(desired_heading) * max_speed,
            sin(desired_heading) * max_speed,
        )
        steering = (
            desired_velocity[0] - creature.body.velocity.x,
            desired_velocity[1] - creature.body.velocity.y,
        )
        limited = self._limit_vector(steering, max_force)
        return limited[0] * strength, limited[1] * strength

    def _flock_turn_bias(
        self,
        creature: Creature,
        flock_force: tuple[float, float],
        max_force: float,
    ) -> float:
        magnitude = hypot(*flock_force)
        if magnitude <= 1e-12 or max_force <= 0.0:
            return 0.0

        relative_angle = self._signed_angle(
            atan2(flock_force[1], flock_force[0]) - creature.heading
        )
        magnitude_ratio = self._clamp(magnitude / max_force, 0.0, 1.0)
        return self._clamp(
            (relative_angle / pi)
            * magnitude_ratio
            * self.config.action.max_flock_turn_bias,
            -self.config.action.max_flock_turn_bias,
            self.config.action.max_flock_turn_bias,
        )

    @staticmethod
    def _limit_vector(
        vector: tuple[float, float],
        maximum: float,
    ) -> tuple[float, float]:
        magnitude = hypot(*vector)
        if maximum <= 0.0 or magnitude <= 1e-12:
            return 0.0, 0.0
        if magnitude <= maximum:
            return vector
        scale = maximum / magnitude
        return vector[0] * scale, vector[1] * scale

    @staticmethod
    def _signed_angle(angle: float) -> float:
        while angle > pi:
            angle -= 2.0 * pi
        while angle < -pi:
            angle += 2.0 * pi
        return angle

    def _apply_top_down_motion(self) -> None:
        motion_commands = getattr(self, "_motion_commands", {})
        for creature in self.creatures:
            self._apply_planar_drag(creature)
            command = motion_commands.get(creature.creature_id)
            if command is not None:
                self._apply_turn_control(
                    creature,
                    command.effective_rotate,
                    max_angular_speed=command.max_angular_speed,
                )

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

    def _apply_turn_control(
        self,
        creature: Creature,
        rotate: float,
        *,
        max_angular_speed: float | None = None,
    ) -> None:
        if abs(rotate) < self.config.action.turn_deadzone:
            rotate = 0.0

        angular_speed_limit = (
            self.MAX_ANGULAR_SPEED
            if max_angular_speed is None
            else max(0.0, max_angular_speed)
        )
        target_angular_velocity = rotate * angular_speed_limit
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
        motion_commands = getattr(self, "_motion_commands", {})
        for creature in self.creatures:
            command = motion_commands.get(creature.creature_id)
            max_speed = self.MAX_SPEED if command is None else command.max_speed
            max_angular_speed = (
                self.MAX_ANGULAR_SPEED if command is None else command.max_angular_speed
            )
            velocity = creature.body.velocity
            if velocity.length > max_speed:
                creature.body.velocity = velocity.normalized() * max_speed
            creature.body.angular_velocity = max(
                -max_angular_speed,
                min(max_angular_speed, creature.body.angular_velocity),
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
                previous_age = fitness.age_seconds
                fitness.record_tick(delta_time, creature.speed, self.MAX_SPEED)
                fitness.record_trait_cost(
                    self.metabolism.trait_energy_cost_per_second(
                        creature,
                        self.MAX_SPEED,
                        self._communication_intensities_for(creature.creature_id),
                    ),
                    delta_time,
                )
                self._record_maturity_if_crossed(creature, previous_age, fitness)

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
                (food.position[0] - mouth_x) ** 2 + (food.position[1] - mouth_y) ** 2
            ),
        )

    def _sync_carried_foods(self) -> None:
        held_foods = getattr(self, "_held_food_by_creature_id", None)
        carriers = getattr(self, "_carrier_by_food_id", None)
        if held_foods is None or carriers is None:
            return

        creatures_by_id = {
            creature.creature_id: creature for creature in self.creatures
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
        self._apply_nursing(delta_time)
        with_infant_penalties = self._apply_infant_movement_penalties()
        try:
            report = self.metabolism.update(
                self.creatures,
                self.foods,
                delta_time,
                self.MAX_SPEED,
                self._eatable_foods_for,
                self._creature_want_to_eat,
                {
                    creature.creature_id: getattr(
                        action,
                        "flee_panic_intensity",
                        0.0,
                    )
                    for creature in self.creatures
                    if (action := self._last_actions.get(creature.creature_id))
                    is not None
                },
                energy_cost_multipliers={
                    creature.creature_id: self._senescence_factor_for(creature)
                    for creature in self.creatures
                },
                creature_age_seconds={
                    creature.creature_id: self._creature_age_seconds(creature)
                    for creature in self.creatures
                },
                communication_intensities={
                    creature.creature_id: self._communication_intensities_for(
                        creature.creature_id
                    )
                    for creature in self.creatures
                },
            )
        finally:
            self._restore_movement_multipliers(with_infant_penalties)

        for consumption in report.food_consumptions:
            fitness = self.fitness.get(consumption.creature_id)
            if fitness is not None:
                fitness.record_food(
                    0.0,
                    depleted=consumption.depleted,
                )

        for creature_id, energy_gained in getattr(
            report,
            "digested_energy_gained",
            {},
        ).items():
            fitness = self.fitness.get(creature_id)
            if fitness is not None:
                fitness.record_food(energy_gained, depleted=False)

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
            death_reason = "old_age" if self._is_senescent(creature) else "starvation"
            self._remove_creature(creature, death_reason=death_reason)

        if not self.creatures:
            self._recover_extinct_population()

        if self.selected_creature_id is not None and self.selected_creature is None:
            self.selected_creature_id = None

    def _communication_intensities_for(
        self,
        creature_id: int,
    ) -> tuple[float, float, float]:
        action = self._last_actions.get(creature_id)
        if action is None:
            return (0.0, 0.0, 0.0)
        return (
            max(0.0, min(1.0, action.emit_sound)),
            max(0.0, min(1.0, action.emit_trail_pheromone)),
            max(0.0, min(1.0, action.emit_alarm_pheromone)),
        )

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

    def _creature_age_seconds(self, creature: Creature) -> float:
        fitness = self.fitness.get(creature.creature_id)
        return 0.0 if fitness is None else fitness.age_seconds

    def _senescence_factor_for(self, creature: Creature) -> float:
        population_config = getattr(
            getattr(self, "config", None),
            "population",
            None,
        )
        if population_config is None:
            return 1.0
        over_age = (
            self._creature_age_seconds(creature)
            - population_config.senescence_age_seconds
        )
        if over_age <= 0.0:
            return 1.0
        return 1.0 + over_age * population_config.senescence_cost_multiplier

    def _is_senescent(self, creature: Creature) -> bool:
        population_config = getattr(
            getattr(self, "config", None),
            "population",
            None,
        )
        return (
            population_config is not None
            and self._creature_age_seconds(creature)
            > population_config.senescence_age_seconds
        )

    def _is_infant(self, creature: Creature) -> bool:
        population_config = getattr(getattr(self, "config", None), "population", None)
        if population_config is None:
            return False

        return (
            self._creature_age_seconds(creature) < population_config.infant_maturity_age
        )

    def _own_infant_children_for(self, parent: Creature) -> list[Creature]:
        parent_id = getattr(parent, "creature_id", None)
        return [
            creature
            for creature in self.creatures
            if getattr(getattr(creature, "lineage", None), "parent_id", None)
            == parent_id
            and self._is_infant(creature)
        ]

    def _record_maturity_if_crossed(
        self,
        creature: Creature,
        previous_age: float,
        fitness: CreatureFitness,
    ) -> None:
        maturity_age = self.config.population.infant_maturity_age
        if previous_age >= maturity_age or fitness.age_seconds < maturity_age:
            return

        parent_id = creature.lineage.parent_id
        if parent_id is None:
            return

        parent_fitness = self.fitness.get(parent_id)
        if parent_fitness is None:
            parent_fitness = self.fitness_archive.get(parent_id)
        if parent_fitness is None:
            return

        if creature.creature_id not in parent_fitness.matured_offspring_ids:
            parent_fitness.matured_offspring_ids.append(creature.creature_id)

    def _apply_nursing(self, delta_time: float) -> None:
        if delta_time <= 0.0:
            return

        population_config = getattr(getattr(self, "config", None), "population", None)
        if population_config is None:
            return

        transfer = population_config.nursing_energy_transfer_rate * delta_time
        if transfer <= 0.0:
            return

        for parent in list(self.creatures):
            action = self._last_actions.get(parent.creature_id)
            if action is None or action.want_nurse < 0.5:
                continue

            infant = self._nearest_nursable_infant_for(parent)
            if infant is None or parent.energy <= 0.30 or parent.energy <= transfer:
                continue

            parent.energy -= transfer
            infant.energy = min(
                self.config.metabolism.max_energy,
                infant.energy + transfer,
            )

    def _nearest_nursable_infant_for(self, parent: Creature) -> Creature | None:
        max_distance = parent.radius * 2.5
        max_distance_squared = max_distance * max_distance
        candidates: list[tuple[float, Creature]] = []
        parent_x, parent_y = parent.position

        for infant in self._own_infant_children_for(parent):
            dx = infant.position[0] - parent_x
            dy = infant.position[1] - parent_y
            distance_squared = dx * dx + dy * dy
            if distance_squared <= max_distance_squared:
                candidates.append((distance_squared, infant))

        if not candidates:
            return None

        return min(candidates, key=lambda item: item[0])[1]

    def _apply_infant_movement_penalties(self) -> dict[int, float]:
        if getattr(getattr(self, "config", None), "population", None) is None:
            return {}

        original_multipliers: dict[int, float] = {}
        for creature in self.creatures:
            if not self._is_infant(creature):
                continue

            original_multipliers[creature.creature_id] = (
                creature.physical_traits.movement_cost_multiplier
            )
            creature.physical_traits.movement_cost_multiplier *= 3.0

        return original_multipliers

    def _restore_movement_multipliers(
        self,
        original_multipliers: dict[int, float],
    ) -> None:
        if not original_multipliers:
            return

        for creature in self.creatures:
            original = original_multipliers.get(creature.creature_id)
            if original is not None:
                creature.physical_traits.movement_cost_multiplier = original

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

    def _ensure_creature_shape_index(self) -> None:
        if not hasattr(self, "_creature_by_shape_id"):
            self._creature_by_shape_id = {}

    def _index_creature_shape(
        self,
        shape: pymunk.Shape,
        creature: Creature,
    ) -> None:
        self._ensure_creature_shape_index()
        self._creature_by_shape_id[id(shape)] = creature

    def _unindex_creature_shape(self, creature: Creature) -> None:
        self._ensure_creature_shape_index()
        self._creature_by_shape_id.pop(id(creature.shape), None)

    def _remove_creature(
        self,
        creature: Creature,
        death_reason: str = "unknown",
    ) -> None:
        telemetry = getattr(self, "telemetry", None)
        if telemetry is not None:
            telemetry.log_creature_death(
                creature.creature_id,
                self.elapsed_time,
                death_reason,
            )
        self._archive_creature_traits(creature)
        self._release_food_for(creature)
        fitness = self.fitness.get(creature.creature_id)
        if fitness is not None:
            self.neat_controller.archive_brain(
                creature.creature_id, fitness.score(self.config.fitness)
            )

        if creature in self.creatures:
            self.creatures.remove(creature)
            self._unindex_creature_shape(creature)
            self.space.remove(creature.body, creature.shape)
            self.neat_controller.remove_brain(creature.creature_id)
            self._last_actions.pop(creature.creature_id, None)
            last_snapshots = getattr(self, "_last_sensor_snapshots", None)
            if last_snapshots is not None:
                last_snapshots.pop(creature.creature_id, None)
            motion_commands = getattr(self, "_motion_commands", None)
            if motion_commands is not None:
                motion_commands.pop(creature.creature_id, None)
            acoustics = getattr(self, "acoustics", None)
            if acoustics is not None:
                acoustics.remove_emitter(creature.creature_id)

        fitness = self.fitness.pop(creature.creature_id, None)
        self.rt_neat.record_death(fitness)
        if fitness is not None:
            self.fitness_archive[creature.creature_id] = fitness

        if self.selected_creature_id == creature.creature_id:
            self.selected_creature_id = None

        self._chronometers.pop(creature.creature_id, None)
        self._prune_historical_archives()

    def _prune_historical_archives(self) -> None:
        population_config = self.config.population
        trait_archive = getattr(self, "_trait_archive_by_genome_id", {})
        prune_population = getattr(
            self.neat_controller,
            "prune_population_archive",
            None,
        )
        retained_genome_ids = (
            prune_population(population_config.elite_archive_size)
            if prune_population is not None
            else set(trait_archive)
        )

        retained_species_ids = {
            creature.lineage.species_id for creature in self.creatures
        }
        retained_species_ids.update(
            archived.lineage.species_id
            for genome_id, archived in trait_archive.items()
            if genome_id in retained_genome_ids
        )
        self._trait_archive_by_genome_id = {
            genome_id: archived
            for genome_id, archived in trait_archive.items()
            if genome_id in retained_genome_ids
        }
        prune_representatives = getattr(
            self.neat_controller,
            "prune_species_representatives",
            None,
        )
        if prune_representatives is not None:
            prune_representatives(retained_species_ids)

        protected_parent_ids = {
            creature.lineage.parent_id
            for creature in self.creatures
            if creature.lineage.parent_id is not None and self._is_infant(creature)
        }
        archive_limit = max(0, population_config.fitness_archive_size)
        recent_ids = (
            set(list(self.fitness_archive)[-archive_limit:])
            if archive_limit > 0
            else set()
        )
        retained_fitness_ids = recent_ids | protected_parent_ids
        self.fitness_archive = {
            creature_id: fitness
            for creature_id, fitness in self.fitness_archive.items()
            if creature_id in retained_fitness_ids
        }

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
                species_id=creature.lineage.species_id,
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
        return sum(
            creature.energy
            + max(0.0, getattr(creature, "stomach_energy", 0.0))
            for creature in self.creatures
        )

    def _plant_energy(self) -> float:
        return sum(food.energy_value for food in self.foods)

    def _available_biomass(self) -> float:
        used_biomass = self._creature_energy() + self._plant_energy()
        return max(0.0, self.total_biomass_energy - used_biomass)

    def _active_species_count(self) -> int:
        return len({creature.lineage.species_id for creature in self.creatures})

    def _plant_spawn_pressure(self) -> float:
        food_capacity = self.food_spawner.food_capacity(self._active_species_count())
        return self.food_spawner.food_regrowth_pressure(
            len(self.foods),
            food_capacity,
        )

    def _biome_area_shares(self) -> dict[str, float]:
        biome_map = getattr(self, "biome_map", None)
        if biome_map is None:
            return {biome.label: 0.0 for biome in Biome}
        return {biome.label: biome_map.area_shares.get(biome, 0.0) for biome in Biome}

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
            parent_species_id = (
                archived_traits.lineage.species_id if archived_traits is not None else 1
            )
            parent_color = (
                archived_traits.color
                if archived_traits is not None
                else self._initial_creature_color(0)
            )
            child = self._spawn_creature(
                child_id,
                energy=self.config.metabolism.max_energy,
                color=(
                    child_traits.color
                    if child_traits is not None
                    else self._initial_creature_color(0)
                ),
                vision=child_traits.vision if child_traits is not None else None,
                physical_traits=(
                    child_traits.physical_traits if child_traits is not None else None
                ),
                lineage=child_traits.lineage if child_traits is not None else None,
            )
            child_brain, speciation_result = (
                self.neat_controller.create_mutated_brain_from_genome(
                    parent_genome,
                    child_id,
                    parent_species_id,
                    child.physical_traits,
                    child.vision,
                )
            )
            if child_brain is None:
                self._unindex_creature_shape(child)
                self.space.remove(child.body, child.shape)
                continue
            child.lineage.species_id = speciation_result.species_id
            if speciation_result.is_new_species:
                child.color = self._new_species_color(parent_color)
                self._record_new_species(child, speciation_result)

            self.creatures.append(child)
            self._initialize_creature_fertility_baseline(child)
            self.fitness[child_id] = CreatureFitness()
            self._chronometers[child_id] = 0.0
            self._log_creature_birth(child)
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
            parent_species_id=archived_traits.lineage.species_id,
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

    def _update_speciation_threshold(self, delta_time: float) -> None:
        speciation_config = self.config.speciation
        interval = speciation_config.adjustment_interval_seconds
        if interval <= 0.0:
            return

        self._speciation_adjustment_accumulator += max(0.0, delta_time)
        if self._speciation_adjustment_accumulator < interval:
            return

        intervals_elapsed = int(self._speciation_adjustment_accumulator / interval)
        self._speciation_adjustment_accumulator %= interval
        active_species_count = len(
            {creature.lineage.species_id for creature in self.creatures}
        )
        threshold = self.neat_controller.species_manager.compatibility_threshold
        if active_species_count < speciation_config.target_species_count:
            threshold -= speciation_config.threshold_adjust_rate * intervals_elapsed
        elif active_species_count > speciation_config.target_species_count:
            threshold += speciation_config.threshold_adjust_rate * intervals_elapsed

        self.neat_controller.species_manager.compatibility_threshold = max(
            speciation_config.min_threshold,
            min(speciation_config.max_threshold, threshold),
        )

    def _genome_for_creature_id(self, creature_id: int) -> object | None:
        brain_for = getattr(self.neat_controller, "brain_for", None)
        if brain_for is None:
            return None
        brain = brain_for(creature_id)
        return None if brain is None else brain.genome

    def _reproduction_cost_for(self, parent: Creature) -> float:
        population_config = self.config.population
        genome = self._genome_for_creature_id(parent.creature_id)
        nodes = getattr(genome, "nodes", {}) or {}
        connections = getattr(genome, "connections", {}) or {}
        calculated_cost = (
            population_config.reproduction_energy_cost_base
            + len(nodes) * population_config.reproduction_cost_per_node
            + len(connections) * population_config.reproduction_cost_per_connection
        )
        return min(
            calculated_cost,
            population_config.max_dynamic_reproduction_cost,
        )

    def _spend_reproduction_energy(
        self,
        parent: Creature,
        reproduction_cost: float,
    ) -> None:
        parent.energy = max(
            0.0,
            parent.energy - reproduction_cost,
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

        reproduction_cost = self._reproduction_cost_for(parent)
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
            energy=self.config.population.infant_energy_spawn,
            color=child_traits.color,
            vision=child_traits.vision,
            physical_traits=child_traits.physical_traits,
            lineage=child_traits.lineage,
        )

        child_brain, speciation_result = self.neat_controller.create_child_brain(
            parent.creature_id,
            child_id,
            parent.lineage.species_id,
            child.physical_traits,
            child.vision,
        )
        if child_brain is None:
            self._unindex_creature_shape(child)
            self.space.remove(child.body, child.shape)
            return False
        assert speciation_result is not None
        child.lineage.species_id = speciation_result.species_id
        if speciation_result.is_new_species:
            child.color = self._new_species_color(parent.color)
            self._record_new_species(child, speciation_result)

        self.creatures.append(child)
        self._initialize_creature_fertility_baseline(child)
        self.fitness[child_id] = CreatureFitness()
        self._chronometers[child_id] = 0.0
        self._log_creature_birth(child)

        self._spend_reproduction_energy(parent, reproduction_cost)
        parent_fitness.record_reproduction()
        self.rt_neat.record_normal_replacement()
        return True

    def _has_reproduction_resources(self) -> bool:
        child_energy = self.config.population.reproduction_energy_cost_base
        available_biomass = self._available_biomass()
        if available_biomass < child_energy:
            return False

        food_capacity = self.food_spawner.food_capacity(self._active_species_count())
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
        live_creatures = {creature.creature_id: creature for creature in self.creatures}
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
