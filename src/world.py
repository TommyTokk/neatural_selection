from __future__ import annotations

from dataclasses import dataclass, field, replace
import copy
import hashlib
import inspect
import pickle
from math import atan2, ceil, cos, exp, floor, hypot, isfinite, pi, sin, sqrt
from random import Random
from threading import RLock
from time import monotonic
from typing import Literal

import pymunk
import numpy as np

from configs.sim_config import (
    LiveFoodConfig,
    SimConfig,
)
import src.utils as ut
from src.creature.action import (
    Action,
    acceleration_force_vector,
    is_active_intent,
    neutral_action,
)
from src.biome import Biome, BiomeGenerationHandler
from src.creature import (
    Color,
    Creature,
    FlockingTraits,
    LineageInfo,
    PhysicalTraits,
    TraitMutationDelta,
    VisionTraits,
)
from src.creature.factory import CreatureFactory
from src.creature.genotype import CreatureGenotype, GenotypeManager
from src.creature.evolution import (
    CreatureEvolutionCoordinator,
    EvolutionTransaction,
)
from src.creature.runtime import (
    CreatureActionService,
    CreatureLifecycleService,
    CreaturePerceptionService,
    CreatureResourceService,
    CreatureSocialService,
)
from src.creature.fitness import CreatureFitness, flocking_benchmark_quality
from src.creature.flocking import (
    FlockingRuntimeSnapshot,
    SocialRuntime,
    SocialCompatibilityResolver,
    SocialIntent,
    SocialObservation,
    accepted_counterfactual_contribution,
    blend_desired_velocity,
    calculate_flocking_weights,
    calculate_social_intent,
    configured_social_influence,
)
from src.food import Food
from src.food_spawner import FoodSpawner
from src.creature.metabolism import (
    ActivityResult,
    ENERGY_EPSILON,
    FoodConsumption,
    Metabolism,
    MetabolismReport,
    ResourceCandidate,
    calculate_weighted_activity,
    is_energy_depleted,
)
from src.creature.vision import (
    BiomeSensorSnapshot,
    SENSOR_CONTRACT,
    SensorSnapshot,
    VisionSystem,
)
from src.creature.neat.controller import NeatBrainController
from src.persistence import PersistenceManager, SavePriority, SimulationPaths
from src.creature.neat.rt_neat import RtNeatManager
from src.creature.speciation import (
    ContinuousSpeciesManager,
    NeatChangeSummary,
    SpeciesDistanceBreakdown,
    SpeciesRecord,
    SpeciationResult,
    SpeciesTraitSnapshot,
)
from src.telemetry import TelemetryDatabase
from src.flocking_telemetry import (
    FlockingTelemetryAggregator,
    PersistentGroupTracker,
)
from src.collision import BOUNDARY_CATEGORY, CREATURE_CATEGORY, FOOD_CATEGORY
from src.creature.spatial import (
    BroadPhaseGeometry,
    CandidateBuffer,
    CreatureSpatialIndex,
)
from src.creature.communication import (
    AcousticDebugInfo,
    AcousticSignal,
    AcousticSystem,
    PheromoneSnapshot,
    PheromoneSystem,
)
from src.behavior_observer import (
    BehaviorKind,
    BehaviorObservation,
    BehaviorObserverDiagnostics,
    BehaviorObserverService,
    BehaviorSnapshot,
    ObservationMode,
)
from src.behavior_history import (
    BehaviorHistoryDiagnostics,
    BehaviorTermination,
    CreatureBehaviorHistoryStore,
    CreatureBehaviorReport,
    CreatureHistoryIndexEntry,
    SpeciesBehaviorIndexEntry,
    SpeciesBehaviorReport,
)
from src.counterfactual_neat import (
    CounterfactualDiagnostics,
    CounterfactualProbeInput,
    FocalBrainUpdate,
    PureNeatEvaluator,
    WhySnapshot,
    mapped_probe_behaviors,
)

from src.ui.layouts.screen import build_screen_layout


_EMPTY_LONG_RANGE_SOCIAL_OBSERVATION = SocialObservation().long_range


EnvironmentMapMode = Literal["none", "biome", "pheromones"]


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
    flocking_traits: FlockingTraits = field(default_factory=FlockingTraits)


@dataclass(slots=True)
class ArchivedCreatureTraits:
    creature_id: int
    vision: VisionTraits
    physical_traits: PhysicalTraits
    color: Color
    lineage: LineageInfo
    flocking_traits: FlockingTraits = field(default_factory=FlockingTraits)

    @property
    def genotype(self) -> CreatureGenotype:
        """Return the archived flat fields as an aggregate genotype.

        Parameters
        ----------
        None
            This property receives no external parameters.

        Returns
        -------
        CreatureGenotype
            Independent aggregate genotype suitable for recovery mutation.
        """
        # Preserve the legacy checkpoint fields while exposing the new model.
        return CreatureGenotype(
            copy.deepcopy(self.vision),
            copy.deepcopy(self.physical_traits),
            copy.deepcopy(self.flocking_traits),
            tuple(self.color),
        )

    @classmethod
    def from_creature(cls, creature: Creature) -> ArchivedCreatureTraits:
        """Capture genotype and lineage from a live creature.

        Parameters
        ----------
        creature
            Live creature being archived.

        Returns
        -------
        ArchivedCreatureTraits
            Legacy-compatible detached archive record.
        """
        # Snapshot helpers replace repeated field-by-field trait copying.
        genotype = creature.genotype.snapshot()
        return cls(
            creature_id=creature.creature_id,
            vision=genotype.vision,
            physical_traits=genotype.physical_traits,
            color=genotype.color,
            lineage=creature.lineage.snapshot(),
            flocking_traits=genotype.flocking_traits,
        )


@dataclass(slots=True)
class MotionCommand:
    effective_rotate: float
    max_speed: float
    max_angular_speed: float


@dataclass(slots=True)
class SimulationLagMetrics:
    session_requested_seconds: float = 0.0
    session_completed_seconds: float = 0.0
    pending_seconds: float = 0.0
    session_dropped_seconds: float = 0.0
    effective_speed_multiplier: float = 0.0
    clamped_this_update: bool = False


@dataclass(slots=True)
class _MouthExposureBuffer:
    """Reusable primitive storage for unresolved fixed-step mouth contacts."""

    steps: list[int] = field(default_factory=list)
    creature_ids: list[int] = field(default_factory=list)
    food_ids: list[int] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)
    order: list[int] = field(default_factory=list)
    count: int = 0

    def append(
        self,
        step: int,
        creature_id: int,
        food_id: int,
        duration: float,
    ) -> None:
        index = self.count
        if index == len(self.steps):
            capacity = max(16, len(self.steps) * 2)
            growth = capacity - len(self.steps)
            self.steps.extend([0] * growth)
            self.creature_ids.extend([0] * growth)
            self.food_ids.extend([0] * growth)
            self.durations.extend([0.0] * growth)
            self.order.extend([0] * growth)
        self.steps[index] = int(step)
        self.creature_ids[index] = int(creature_id)
        self.food_ids[index] = int(food_id)
        self.durations[index] = max(0.0, float(duration))
        self.order[index] = index
        self.count += 1

    def clear(self) -> None:
        self.count = 0

    def state(self) -> tuple[tuple[int, int, int, float], ...]:
        return tuple(
            (
                self.steps[index],
                self.creature_ids[index],
                self.food_ids[index],
                self.durations[index],
            )
            for index in range(self.count)
        )

    def sort_order(self) -> None:
        """Sort the reusable index workspace without allocating records."""
        for index in range(self.count):
            self.order[index] = index
        for index in range(1, self.count):
            candidate = self.order[index]
            candidate_key = (
                self.steps[candidate],
                self.food_ids[candidate],
                self.creature_ids[candidate],
            )
            insertion = index
            while insertion > 0:
                previous = self.order[insertion - 1]
                previous_key = (
                    self.steps[previous],
                    self.food_ids[previous],
                    self.creature_ids[previous],
                )
                if previous_key <= candidate_key:
                    break
                self.order[insertion] = previous
                insertion -= 1
            self.order[insertion] = candidate

    def restore(self, records: object) -> None:
        self.clear()
        if not isinstance(records, (tuple, list)):
            return
        for record in records:
            if not isinstance(record, (tuple, list)) or len(record) != 4:
                continue
            step, creature_id, food_id, duration = record
            if (
                type(step) is not int
                or type(creature_id) is not int
                or type(food_id) is not int
            ):
                continue
            try:
                elapsed = float(duration)
            except (TypeError, ValueError):
                continue
            if not isfinite(elapsed) or elapsed <= 0.0:
                continue
            self.append(step, creature_id, food_id, elapsed)


@dataclass(slots=True)
class _MouthExposureRollbackState:
    """State changed while validating authoritative mouth exposures."""

    creature_states: list[tuple[object, float, float]]
    food_states: list[tuple[object, int, float]]
    held_foods: dict[int, Food]
    food_carriers: dict[int, int]


@dataclass(slots=True)
class _FlockSteeringDebug:
    accepted_counterfactual_delta: tuple[float, float]
    max_force: float


@dataclass(frozen=True, slots=True)
class ReproductionRequest:
    parent: Creature
    eligibility_rank: int
    reserved_energy_cost: float
    selection_pool_size: int = 0
    node_count: int = 0
    enabled_connection_count: int = 0
    network_complexity: float = 0.0


@dataclass(frozen=True, slots=True)
class NursingRequest:
    donor: Creature
    target: Creature
    requested_transfer: float


@dataclass(frozen=True, slots=True)
class AcceptedNursingTransfer:
    request: NursingRequest
    allocated_transfer: float


@dataclass(frozen=True, slots=True)
class StagedOffspring:
    request: ReproductionRequest
    child_id: int
    traits: ChildCreatureTraits
    position: tuple[float, float]
    speciation_result: SpeciationResult


@dataclass(slots=True)
class TransactionResolution:
    candidates: dict[int, ResourceCandidate]
    activities: dict[int, ActivityResult]
    reproductions: list[ReproductionRequest]
    nursing_transfers: list[AcceptedNursingTransfer]
    reproduction_attempts: list[tuple[ReproductionRequest, str]] = field(
        default_factory=list
    )


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
        brain_initialization_seed: int | None = None,
    ) -> None:
        """Initialize world-owned environment state and composed services.

        Parameters
        ----------
        config
            Complete simulation configuration.
        bootstrap
            Whether to create the configured initial population and food.
        simulation_paths
            Optional filesystem paths for checkpoints and telemetry output.
        brain_initialization_seed
            Optional neural initialization seed independent of simulation RNG.

        Returns
        -------
        None
            The world and its service graph are initialized in place.

        Notes
        -----
        Service construction preserves the historical random-consumption order;
        in particular, creature spawning still consumes the simulation RNG.
        """
        # Validate configuration before any service allocates persistent state.
        config.flocking.validate()
        config.action.validate()
        config.scheduler.validate()
        self.config = config
        self.fixed_timestep = 1.0 / config.scheduler.physics_hz
        self.rng = Random(config.random_seed)
        self.brain_initialization_seed = (
            config.random_seed
            if brain_initialization_seed is None
            else brain_initialization_seed
        )
        self.elapsed_time = 0.0
        self.fps = 0.0
        self.is_paused = False
        self.simulation_speed = 1.0
        self.environment_map_mode: EnvironmentMapMode = "none"
        self._physics_accumulator = 0.0
        self._simulation_step = 0
        # State capture copies authoritative data while this lock is held.
        # Pickle serialization and disk I/O happen later on the writer thread.
        self._checkpoint_state_lock = RLock()
        self.simulation_lag_metrics = SimulationLagMetrics()
        self._mouth_exposures = _MouthExposureBuffer()
        self._immediate_dead_buffer: list[Creature] = []
        self._reproduction_accumulator = 0.0
        self._speciation_adjustment_accumulator = 0.0
        self._flocking_telemetry_accumulator = 0.0
        self._flocking_capture_origin = 0.0
        self._flocking_capture_ordinal = 1
        self._flocking_capture_due_this_step = False
        self._last_actions: dict[int, Action] = {}
        self._effective_actions: dict[int, Action] = {}
        self._last_sensor_snapshots: dict[int, SensorSnapshot] = {}
        self._last_acoustic_debug: dict[int, AcousticDebugInfo] = {}
        self._last_flock_steering_debug: dict[int, _FlockSteeringDebug] = {}
        self._last_flocking_runtime: dict[int, FlockingRuntimeSnapshot] = {}
        self._cached_social_intentions: dict[int, SocialRuntime] = {}
        self._flocking_benchmark_quality_by_creature_id: dict[int, float] = {}
        self.behavior_observer = BehaviorObserverService(
            config.behavior,
            getattr(config, "counterfactual_why", None),
            getattr(config, "behavior_history", None),
        )
        history_config = config.behavior_history
        self.behavior_history = CreatureBehaviorHistoryStore(
            max_completed_bouts_per_creature=(
                history_config.max_completed_bouts_per_creature
            ),
            max_remembered_creatures=(
                history_config.max_remembered_creatures
            ),
            minimum_stable_bouts=history_config.minimum_stable_bouts,
        )
        self._behavior_history_worker_skipped_seen = 0
        self._behavior_selection_generation = 0
        self._behavior_next_sample_time = 0.0
        self._why_next_probe_time = 0.0
        self._behavior_food_consumption_count = 0
        self._behavior_food_consumed_energy_total = 0.0
        self._behavior_automatic_cohort: dict[int, tuple[int, ...]] = {}
        self._behavior_cohort_dirty = True
        self._behavior_cohort_config_signature = (
            config.behavior.enabled,
            config.behavior.background_representatives_per_species,
        )
        self._behavior_active_subjects: dict[int, int] = {}
        self._behavior_subject_generation_counter = 0
        self._behavior_consumption_totals: dict[int, tuple[int, float]] = {}
        self.brain_contract_reset_occurred = False
        self._flocking_group_tracker = PersistentGroupTracker(
            config.flocking.telemetry.persistence_overlap_threshold
        )
        self._motion_commands: dict[int, MotionCommand] = {}
        self._communication_positions = np.empty((0, 2), dtype=np.float64)
        self._communication_trail_amounts = np.empty(0, dtype=np.float64)
        self._communication_alarm_amounts = np.empty(0, dtype=np.float64)
        self._pheromone_sensor_positions = np.empty((0, 3, 2), dtype=np.float64)
        self._pheromone_sensor_values = np.empty((0, 6), dtype=np.float32)
        self.debug_vision_enabled = config.debug.show_debug_vision_by_default
        self.layout = build_screen_layout(
            config.display.width, config.display.height, config.layout
        )
        self.environment_zoom = config.zoom.default
        self.environment_pan_x = 0.0
        self.environment_pan_y = 0.0
        self._camera_follows_selected_creature = True
        self.vision = VisionSystem(
            config.vision,
            config.metabolism.eating_distance,
            config.metabolism.stomach_capacity_per_radius,
            max_life=config.metabolism.max_life,
            flocking_config=config.flocking,
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
        self._living_creatures: dict[int, Creature] = {}
        self._issued_creature_ids: set[int] = set()
        self.lifecycle = CreatureLifecycleService()
        self.lifecycle.living = self._living_creatures
        self.lifecycle.issued_ids = self._issued_creature_ids
        self._creature_query_filter = pymunk.ShapeFilter(mask=CREATURE_CATEGORY)
        # Kept as a compatibility sentinel for older diagnostics. Positions
        # are owned exclusively by the generation-stamped spatial index.
        self._creature_spatial_state = None
        self._boundary_shapes: list[pymunk.Shape] = []
        self._rebuild_boundaries()
        # Creature construction and genotype logic now have explicit ownership.
        self.genotype_manager = GenotypeManager(
            config,
            self.CREATURE_COLOR_PALETTE,
        )
        self.creature_factory = CreatureFactory(
            config,
            self.space,
            self.genotype_manager,
        )
        self.creatures = self._spawn_creatures() if bootstrap else []
        self._next_creature_id_value = (
            max(
                (creature.creature_id for creature in self.creatures),
                default=0,
            )
            + 1
        )
        self.lifecycle.synchronize_allocator(self._next_creature_id_value)
        for creature in self.creatures:
            self._register_living_creature(creature)
            self._initialize_creature_runtime_state(creature)
        self._creature_spatial_index = CreatureSpatialIndex(
            cell_size=128.0,
            living_registry=self._living_creatures,
        )
        self._candidate_buffer = CandidateBuffer(self._creature_spatial_index)
        self._candidate_buffer_leased = False
        self._spatial_scheduled_queries = 0
        self._spatial_collision_only_queries = 0
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
        self._live_food_config = LiveFoodConfig.from_configs(
            config.biome,
            config.food,
        )
        # Runtime editing must not mutate the configuration shared with the
        # start menu and subsequently created simulations.
        self.food_spawner = FoodSpawner(
            replace(config.food),
            self.rng,
            self.biome_map,
        )
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

        sensor_contract = SENSOR_CONTRACT
        controller_kwargs = {
            "compatibility_threshold": (
                config.speciation.compatibility_threshold
            ),
            "phenotypic_weight": config.speciation.phenotypic_weight,
            "trait_config": config.trait,
            "vision_config": config.vision,
            "flocking_trait_distance_coefficient": (
                config.speciation.flocking_trait_distance_coefficient
            ),
        }
        # Small test/extension controllers written against the historical
        # constructor remain usable. The production controller advertises
        # this argument and selects the current input topology before any
        # genomes are created.
        controller_parameters = inspect.signature(
            NeatBrainController
        ).parameters
        if "sensor_contract" in controller_parameters:
            controller_kwargs["sensor_contract"] = sensor_contract
        if "random_seed" in controller_parameters:
            controller_kwargs["random_seed"] = self.brain_initialization_seed
        if "herding_decay_rate" in controller_parameters:
            controller_kwargs["herding_decay_rate"] = (
                config.flocking.herding_decay_rate
            )
        self.neat_controller = NeatBrainController(
            sensor_contract.neat_config_path,
            **controller_kwargs,
        )
        # Keep the legacy alias while exposing explicit neural/species services.
        self.brain_controller = self.neat_controller
        self.species_manager = getattr(
            self.brain_controller,
            "species_manager",
            None,
        )
        if self.species_manager is None:
            self.species_manager = ContinuousSpeciesManager(
                config.speciation.compatibility_threshold,
                config.speciation.phenotypic_weight,
                config.trait,
                config.vision,
                config.speciation.flocking_trait_distance_coefficient,
            )
            self.brain_controller.species_manager = self.species_manager
        self.evolution = CreatureEvolutionCoordinator(
            self.genotype_manager,
            self.brain_controller,
        )
        self.social_compatibility = SocialCompatibilityResolver(
            config.flocking.compatibility,
            self.evolution.flocking_compatibility,
        )
        self.vision.flock_compatibility_resolver = (
            self.social_compatibility.compatibility
        )
        if bootstrap:
            self.evolution.assign_initial_brains(self.creatures)
        self.metabolism = Metabolism(
            config.metabolism,
            self.vision,
            config.trait,
            genome_for_creature_id=self._genome_for_creature_id,
            communication_config=config.communication,
            food_config=config.food,
        )
        self.rt_neat = RtNeatManager(self.neat_controller, self.rng)
        # Composed services become the debuggable owners of creature-domain state.
        self.perception = CreaturePerceptionService(self.vision)
        self.perception.spatial_index = self._creature_spatial_index
        self.perception.candidate_buffer = self._candidate_buffer
        self.perception.last_snapshots = self._last_sensor_snapshots
        self.perception.last_acoustic_debug = self._last_acoustic_debug
        self.actions = CreatureActionService()
        self.actions.raw = self._last_actions
        self.actions.effective = self._effective_actions
        self.actions.motion_commands = self._motion_commands
        self.social = CreatureSocialService(self.social_compatibility)
        self.social.intentions = self._cached_social_intentions
        self.social.last_runtime = self._last_flocking_runtime
        self.social.last_debug = self._last_flock_steering_debug
        self.social.communication_positions = self._communication_positions
        self.social.communication_trail_amounts = self._communication_trail_amounts
        self.social.communication_alarm_amounts = self._communication_alarm_amounts
        self.resources = CreatureResourceService(self.metabolism)
        self.resources.fitness = self.fitness
        self.resources.held_food_by_creature_id = self._held_food_by_creature_id
        self.resources.carrier_by_food_id = self._carrier_by_food_id
        self.resources.mouth_exposures = self._mouth_exposures
        self.resources.chronometers = self._chronometers
        if not hasattr(self, "_last_digestion_processing_costs_per_second"):
            self._last_digestion_processing_costs_per_second = {}
        resources = getattr(self, "resources", None)
        if resources is not None:
            resources.last_digestion_processing_costs_per_second = (
                self._last_digestion_processing_costs_per_second
            )
        self.lifecycle.add_discard_callback(self.perception.discard)
        self.lifecycle.add_discard_callback(self.actions.discard)
        self.lifecycle.add_discard_callback(self.social.discard)
        self._trait_archive_by_genome_id: dict[int, ArchivedCreatureTraits] = {}
        self.species_history: dict[int, SpeciesRecord] = {}
        if bootstrap:
            self._initialize_luca_record()
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
        lock = getattr(self, "_checkpoint_state_lock", None)
        if lock is None:
            self._update_unlocked(delta_time)
            return
        with lock:
            self._update_unlocked(delta_time)

    def _update_unlocked(self, delta_time: float) -> None:
        behavior_observer = getattr(self, "behavior_observer", None)
        if behavior_observer is not None:
            behavior_observer.poll()
            self._record_behavior_observer_progress()
            self._drain_completed_behavior_bouts()
        if delta_time > 0.0:
            instant_fps = 1.0 / delta_time
            self.fps = (
                instant_fps if self.fps == 0.0 else self.fps * 0.9 + instant_fps * 0.1
            )

        if self.is_paused:
            if not hasattr(self, "simulation_lag_metrics"):
                self.simulation_lag_metrics = SimulationLagMetrics()
            self.simulation_lag_metrics.effective_speed_multiplier = 0.0
            self.simulation_lag_metrics.clamped_this_update = False
            self.simulation_lag_metrics.pending_seconds = (
                getattr(self, "_physics_accumulator", 0.0)
            )
            self._refresh_stats()
            return

        real_delta_time = max(0.0, delta_time)
        scaled_delta_time = real_delta_time * self.simulation_speed
        lag = self.simulation_lag_metrics
        lag.session_requested_seconds += scaled_delta_time
        requested_backlog = self._physics_accumulator + scaled_delta_time
        maximum_backlog = (
            self.fixed_timestep * self.config.scheduler.max_backlog_steps
        )
        admitted_backlog = min(requested_backlog, maximum_backlog)
        dropped = max(0.0, requested_backlog - admitted_backlog)
        self._physics_accumulator = admitted_backlog
        lag.session_dropped_seconds += dropped
        lag.clamped_this_update = dropped > 1e-12

        steps = 0
        completed_this_update = 0.0
        while (
            self._physics_accumulator + 1e-12 >= self.fixed_timestep
            and steps < self.config.scheduler.max_steps_per_frame
        ):
            self._run_fixed_step()
            self._physics_accumulator = max(
                0.0,
                self._physics_accumulator - self.fixed_timestep,
            )
            self._simulation_step += 1
            completed_this_update += self.fixed_timestep
            steps += 1

        lag.session_completed_seconds += completed_this_update
        lag.pending_seconds = self._physics_accumulator
        lag.effective_speed_multiplier = (
            completed_this_update / real_delta_time
            if real_delta_time > 0.0
            else 0.0
        )
        self._follow_selected_creature()
        self._update_persistence_timer(real_delta_time)

    def _run_fixed_step(self) -> None:
        """Execute the current zero-based step without committing its count.

        The speciation threshold retains its historical pre-intention hook.
        Creature decisions and cached motion precede physics; contact exposure
        and direct-death removal follow physics. On completion boundaries, the
        preserved survival/age, flocking fitness, chronometer, reproduction,
        and resource/biology sequence runs before ancillary simulated-time
        systems. ``update`` increments ``_simulation_step`` only after this
        method returns successfully.
        """
        delta_time = self.fixed_timestep
        self.elapsed_time += delta_time
        self._update_speciation_threshold(delta_time)
        self._apply_creature_intents()
        self._commit_communication_intents(delta_time)
        self.space.step(delta_time)
        self._settle_food_motion()
        self._apply_top_down_motion(delta_time)
        self._limit_creature_motion()
        self._sync_carried_foods()
        self._accumulate_mouth_exposures(delta_time)
        self._apply_immediate_direct_damage()
        biology_period = self.config.scheduler.biology_period_steps
        if (self._simulation_step + 1) % biology_period == 0:
            biology_dt = delta_time * biology_period
            # Preserve the historical relative order: survival/age and
            # cooldown state advance before the resource transaction pass.
            self._update_fitness_survival(biology_dt)
            self._update_flocking_benchmark(biology_dt)
            self._update_chronometers(biology_dt)
            self._update_reproduction(biology_dt)
            self._update_metabolism(biology_dt)
        self.pheromones.accumulate(delta_time)
        self._spawn_foods(delta_time)
        self._update_flocking_telemetry(delta_time)
        self._sample_selected_behavior()
        self._sample_selected_why()
        if (
            (self._simulation_step + 1)
            % self.config.scheduler.statistics_period_steps
            == 0
        ):
            self._refresh_stats()

    def _scheduler_validation_failure_point(self, point: str) -> None:
        """Invoke an optional narrow failure injector used by soak tests."""
        injector = getattr(self, "_scheduler_validation_failure_injector", None)
        if callable(injector):
            injector(point)

    def _accumulate_mouth_exposures(self, delta_time: float) -> None:
        """Capture fixed-step eat contacts without mutating food resources."""
        if not hasattr(self.metabolism, "evaluate_candidate"):
            return
        for creature in self.creatures:
            if not self._creature_want_to_eat(creature):
                continue
            # Record every overlapping candidate. Competition cannot be
            # decided here because stomach and food state are intentionally
            # unchanged until the biology pass. The chronological resolver
            # chooses at most one successful food claim per creature and one
            # successful creature claim per food for each physics step.
            for food in self._eatable_foods_for(creature):
                if not self.metabolism.food_overlaps_mouth(creature, food):
                    continue
                self._mouth_exposures.append(
                    self._simulation_step,
                    creature.creature_id,
                    food.id,
                    delta_time,
                )

    def _apply_immediate_direct_damage(self) -> None:
        """Commit queued direct damage before any biology actions can run."""
        dead = self._immediate_dead_buffer
        dead.clear()
        selected_id = getattr(self, "selected_creature_id", None)
        for creature in self.creatures:
            damage = max(
                0.0,
                float(getattr(creature, "pending_direct_life_damage", 0.0)),
            )
            if damage <= 0.0:
                continue
            creature.pending_direct_life_damage = 0.0
            creature.life = max(0.0, creature.life - damage)
            if creature.creature_id == selected_id:
                diagnostics = creature.ledger_diagnostics
                diagnostics.direct_life_damage = damage
                diagnostics.final_life = creature.life
                diagnostics.transaction_status = "direct_damage_committed"
            if creature.life <= 0.0:
                dead.append(creature)
        self._remove_dead_creatures(dead, default_reason="direct_damage")

    def _remove_dead_creatures(
        self,
        candidates: list[Creature],
        *,
        default_reason: str,
    ) -> None:
        """Remove each still-live-list dead creature at most once."""
        for creature in candidates:
            if creature not in self.creatures or creature.life > 0.0:
                continue
            reason = (
                "old_age"
                if default_reason == "metabolic" and self._is_senescent(creature)
                else default_reason
            )
            self._remove_creature(creature, death_reason=reason)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            behavior_observer = getattr(self, "behavior_observer", None)
            if behavior_observer is not None:
                behavior_observer.close()
        finally:
            try:
                self.persistence_manager.close()
            finally:
                if self.telemetry is not None:
                    self.telemetry.close()

    def rebind_creature_services(self) -> None:
        """Reconnect compatibility state after persistence replaces containers.

        Parameters
        ----------
        None
            This method receives no external parameters.

        Returns
        -------
        None
            Every composed service points at authoritative restored state.
        """
        # Persistence intentionally replaces dictionaries atomically after loading.
        self.lifecycle.living = self._living_creatures
        self.lifecycle.issued_ids = self._issued_creature_ids
        self.lifecycle.synchronize_allocator(self._next_creature_id_value)
        self.perception.last_snapshots = self._last_sensor_snapshots
        self.perception.last_acoustic_debug = self._last_acoustic_debug
        self.perception.spatial_index = self._creature_spatial_index
        self.perception.candidate_buffer = self._candidate_buffer
        self.actions.raw = self._last_actions
        self.actions.effective = self._effective_actions
        self.actions.motion_commands = self._motion_commands
        self.social.intentions = self._cached_social_intentions
        self.social.last_runtime = self._last_flocking_runtime
        self.social.last_debug = self._last_flock_steering_debug
        self.social.communication_positions = self._communication_positions
        self.social.communication_trail_amounts = self._communication_trail_amounts
        self.social.communication_alarm_amounts = self._communication_alarm_amounts
        self.resources.fitness = self.fitness
        self.resources.held_food_by_creature_id = self._held_food_by_creature_id
        self.resources.carrier_by_food_id = self._carrier_by_food_id
        self.resources.mouth_exposures = self._mouth_exposures
        self.resources.chronometers = self._chronometers
        self.resources.last_digestion_processing_costs_per_second = (
            self._last_digestion_processing_costs_per_second
        )
        self.species_manager = self.brain_controller.species_manager
        self.evolution.species_manager = self.species_manager

    @property
    def save_in_progress(self) -> bool:
        return self.persistence_manager.is_busy

    @property
    def physics_step_count(self) -> int:
        """Compatibility view of the authoritative completed-step counter."""
        return self._simulation_step

    def save_now(self) -> None:
        lock = getattr(self, "_checkpoint_state_lock", None)
        if lock is None:
            self._save_now_unlocked()
            return
        with lock:
            self._save_now_unlocked()

    def _save_now_unlocked(self) -> None:
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

    def _flocking_capture_deadline(self) -> float:
        interval = self.config.flocking.telemetry.interval_seconds
        return self._flocking_capture_origin + (
            self._flocking_capture_ordinal * interval
        )

    def _flocking_telemetry_is_due(self) -> bool:
        if getattr(self, "telemetry", None) is None:
            return False
        return self.elapsed_time + 1e-12 >= self._flocking_capture_deadline()

    def _advance_flocking_capture_schedule(self) -> None:
        interval = self.config.flocking.telemetry.interval_seconds
        while (
            self.elapsed_time + 1e-12
            >= self._flocking_capture_origin
            + self._flocking_capture_ordinal * interval
        ):
            self._flocking_capture_ordinal += 1
        previous_deadline = self._flocking_capture_origin + (
            self._flocking_capture_ordinal - 1
        ) * interval
        self._flocking_telemetry_accumulator = max(
            0.0,
            self.elapsed_time - previous_deadline,
        )

    def _reset_flocking_capture_schedule(self) -> None:
        self._flocking_capture_origin = getattr(self, "elapsed_time", 0.0)
        self._flocking_capture_ordinal = 1
        self._flocking_capture_due_this_step = False
        self._flocking_telemetry_accumulator = 0.0

    def _update_flocking_benchmark(self, delta_time: float) -> None:
        config = self.config.flocking.benchmark
        if not config.enabled:
            return
        qualities = getattr(
            self,
            "_flocking_benchmark_quality_by_creature_id",
            {},
        )
        for creature_id, quality in qualities.items():
            fitness = self.fitness.get(creature_id)
            if fitness is not None:
                fitness.record_flocking_benchmark_quality(
                    quality,
                    delta_time,
                    config,
                )

    def _update_flocking_telemetry(self, delta_time: float) -> None:
        telemetry = getattr(self, "telemetry", None)
        if telemetry is None:
            return
        del delta_time
        if not getattr(self, "_flocking_capture_due_this_step", False):
            return
        group_config = self.config.flocking.telemetry
        groups = self._flocking_group_tracker.sample(
            self.creatures,
            sim_time=self.elapsed_time,
            group_range=group_config.group_detection_range,
            minimum_compatibility=(
                group_config.minimum_group_compatibility
            ),
            compatibility=self.social_compatibility.compatibility,
            nearby=None,
        )
        for creature_id, (group_id, group_size) in (
            groups.group_by_creature.items()
        ):
            snapshot = self._last_flocking_runtime.get(creature_id)
            if snapshot is not None:
                self._last_flocking_runtime[creature_id] = replace(
                    snapshot,
                    local_group_id=group_id,
                    local_group_size=group_size,
                )
        metrics = FlockingTelemetryAggregator.aggregate(
            sim_time=self.elapsed_time,
            population_size=len(self.creatures),
            runtime=self._last_flocking_runtime,
            groups=groups,
        )
        metrics["benchmark_reward_contribution"] = sum(
            fitness.flocking_benchmark_reward
            for fitness in self.fitness.values()
        )
        telemetry.log_flocking_metrics(metrics)
        self._advance_flocking_capture_schedule()
        self._flocking_capture_due_this_step = False

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
                getattr(founder, "flocking_traits", FlockingTraits()),
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
                flocking_trait_distance=0.0,
                weighted_flocking_trait_distance=0.0,
                flocking_trait_distance_coefficient=(
                    getattr(
                        self.neat_controller.species_manager,
                        "flocking_trait_distance_coefficient",
                        1.0,
                    )
                ),
                separation_gene_component=0.0,
                alignment_gene_component=0.0,
                cohesion_gene_component=0.0,
            ),
            neat_changes=NeatChangeSummary.empty(),
            emergence_food_ratio=food_ratio,
            emergence_pop_ratio=population_ratio,
            neural_shifts=(),
        )

    def start_new_sensing_epoch(self, root_species_id: int) -> None:
        """Reset behavioral evolution while preserving the living world."""
        self.evolution.reset_for_new_sensing_epoch(
            self.creatures,
            root_species_id,
        )

        for creature in self.creatures:
            previous_fitness = self.fitness.get(creature.creature_id)
            biological_age = (
                previous_fitness.age_seconds
                if previous_fitness is not None
                else 0.0
            )
            self.fitness[creature.creature_id] = CreatureFitness(
                age_seconds=biological_age,
                evaluation_start_age_seconds=biological_age,
                last_reproduction_age=biological_age,
            )
            creature.lineage.species_id = root_species_id
            # Neural compatibility starts a new species epoch, but generation,
            # parentage, and inherited mutation history remain biological state.
            creature.last_action = None
            creature.smoothed_rotation = 0.0
            creature.smoothed_acceleration = 0.0

        self.fitness_archive = {}
        self._trait_archive_by_genome_id = {}
        self._last_actions = {}
        self._effective_actions = {}
        self._last_sensor_snapshots = {}
        self._last_acoustic_debug = {}
        self._last_flock_steering_debug = {}
        self._last_flocking_runtime = {}
        self._cached_social_intentions = {}
        self._flocking_benchmark_quality_by_creature_id = {}
        self._motion_commands = {}
        for creature in self.creatures:
            self._initialize_creature_runtime_state(creature)
        self._reset_flocking_capture_schedule()
        self._mark_behavior_cohort_dirty()
        self.rt_neat.stats = type(self.rt_neat.stats)()
        self.rt_neat.eligible_parent_ids = []
        self.rt_neat._lifespan_at_death_total = 0.0
        self.rt_neat._lifespan_at_death_count = 0

        self._reset_behavior_focus(
            getattr(self, "selected_creature_id", None)
        )

        if not self.creatures:
            return

        founder = self.creatures[0]
        food_ratio, population_ratio = self._species_emergence_ratios()
        zero_traits = SpeciesTraitSnapshot(0.0, 0.0, 0.0, 0.0)
        self.species_history[root_species_id] = SpeciesRecord(
            species_id=root_species_id,
            parent_species_id=None,
            founder_creature_id=founder.creature_id,
            founder_genome_id=self.neat_controller.genome_id_for(
                founder.creature_id
            ),
            emerged_at=self.elapsed_time,
            founder_color=tuple(founder.color[:3]),
            data_quality="exact",
            founder_traits=SpeciesTraitSnapshot.from_traits(
                founder.physical_traits,
                founder.vision,
                getattr(founder, "flocking_traits", FlockingTraits()),
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
                flocking_trait_distance=0.0,
                weighted_flocking_trait_distance=0.0,
                flocking_trait_distance_coefficient=(
                    getattr(
                        self.neat_controller.species_manager,
                        "flocking_trait_distance_coefficient",
                        1.0,
                    )
                ),
                separation_gene_component=0.0,
                alignment_gene_component=0.0,
                cohesion_gene_component=0.0,
            ),
            neat_changes=NeatChangeSummary.empty(),
            emergence_food_ratio=food_ratio,
            emergence_pop_ratio=population_ratio,
            neural_shifts=(),
        )

        telemetry = getattr(self, "telemetry", None)
        if telemetry is not None:
            telemetry.log_species_record(self.species_history[root_species_id])

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
        if (
            self.selected_creature is None
            or not self._camera_follow_enabled()
        ):
            self._clamp_environment_pan()
        else:
            self._follow_selected_creature()

    def pan_environment(self, delta_x: float, delta_y: float) -> None:
        self._camera_follows_selected_creature = False
        self.environment_pan_x += delta_x
        self.environment_pan_y += delta_y
        self._clamp_environment_pan()

    def reset_environment_view(self) -> None:
        self._camera_follows_selected_creature = False
        self.environment_pan_x = 0.0
        self.environment_pan_y = 0.0
        self.environment_zoom = self.config.zoom.default
        self._clamp_environment_pan()

    def toggle_pause(self) -> None:
        self.is_paused = not self.is_paused

    def toggle_brain_view(self) -> None:
        self.show_brain_view = not self.show_brain_view

    def toggle_biome_background(self) -> None:
        self.select_environment_map("biome")

    def select_environment_map(self, mode: EnvironmentMapMode) -> None:
        if mode not in {"none", "biome", "pheromones"}:
            raise ValueError(f"Unsupported environment map mode: {mode!r}")
        self.environment_map_mode = (
            "none" if self.environment_map_mode == mode else mode
        )

    @property
    def show_biome_background(self) -> bool:
        return getattr(self, "environment_map_mode", "none") == "biome"

    @show_biome_background.setter
    def show_biome_background(self, visible: bool) -> None:
        current = getattr(self, "environment_map_mode", "none")
        if visible:
            self.environment_map_mode = "biome"
        elif current == "biome":
            self.environment_map_mode = "none"

    def set_simulation_speed(self, speed: float) -> None:
        """Set the target multiplier; completed fixed steps remain authoritative."""
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
        max_distance = max(0.0, radius)
        center_x, center_y, _ = self._creature_spatial_values(creature)
        # Public/debug use outside a fixed-step lease observes current bodies.
        # The production decision path passes its retained candidate buffer
        # directly and never enters this compatibility materialization.
        if not getattr(self, "_candidate_buffer_leased", False):
            source = sorted(self.creatures, key=lambda other: other.creature_id)
            center_x, center_y, _ = self._creature_spatial_values(creature)
            return [
                other
                for other in source
                if other is not creature
                and (
                    (other.position[0] - center_x) ** 2
                    + (other.position[1] - center_y) ** 2
                    <= (max_distance + other.radius) ** 2
                )
            ]
        candidates = self._query_nearby_creatures(creature, max_distance)
        nearby: list[Creature] = []
        for other in candidates:
            other_x, other_y, other_radius = self._creature_spatial_values(other)
            delta_x = other_x - center_x
            delta_y = other_y - center_y
            limit = max_distance + other_radius
            if delta_x * delta_x + delta_y * delta_y <= limit * limit:
                nearby.append(other)
        return nearby

    def _cache_creature_spatial_state(self) -> None:
        """Rebuild from authoritative pre-physics zero-offset circle centres."""
        self._ensure_spatial_runtime()
        if hasattr(self, "_scheduler_validation_failure_injector"):
            self._scheduler_validation_failure_point("grid.rebuild")
        self._creature_spatial_index.rebuild(self.creatures)

    def _ensure_spatial_runtime(self) -> None:
        """Install transient Milestone-3 state for legacy tests/loaders."""
        if not hasattr(self, "_living_creatures"):
            self._living_creatures = {
                creature.creature_id: creature for creature in self.creatures
            }
        if not hasattr(self, "_creature_spatial_index"):
            self._creature_spatial_index = CreatureSpatialIndex(
                cell_size=128.0,
                living_registry=self._living_creatures,
            )
        if not hasattr(self, "_candidate_buffer"):
            self._candidate_buffer = CandidateBuffer(
                self._creature_spatial_index
            )
        if not hasattr(self, "_candidate_buffer_leased"):
            self._candidate_buffer_leased = False

    def _creature_spatial_values(
        self,
        creature: Creature,
    ) -> tuple[float, float, float]:
        index = getattr(self, "_creature_spatial_index", None)
        if index is not None:
            cached = index.values_for(creature)
            if cached is not None:
                return cached
        body = getattr(creature, "body", None)
        position = (
            body.position
            if body is not None
            else getattr(creature, "position", (0.0, 0.0))
        )
        if hasattr(position, "x") and hasattr(position, "y"):
            position_x, position_y = position.x, position.y
        else:
            position_x, position_y = position[0], position[1]
        shape = getattr(creature, "shape", None)
        radius = (
            getattr(shape, "radius")
            if shape is not None and hasattr(shape, "radius")
            else getattr(creature, "radius", 0.0)
        )
        return (
            float(position_x),
            float(position_y),
            float(radius),
        )

    def _query_nearby_creatures(
        self,
        creature: Creature,
        query_distance: float,
    ) -> list[Creature]:
        """Compatibility API that materializes retained-grid candidates."""
        center_x, center_y, _ = self._creature_spatial_values(creature)
        index = getattr(self, "_creature_spatial_index", None)
        if index is None:
            space = getattr(self, "space", None)
            point_query = getattr(space, "point_query", None)
            shape_index = getattr(self, "_creature_by_shape_id", None)
            if point_query is None or shape_index is None:
                return []
            hits = point_query(
                (center_x, center_y),
                max(0.0, query_distance),
                getattr(
                    self,
                    "_creature_query_filter",
                    pymunk.ShapeFilter(mask=CREATURE_CATEGORY),
                ),
            )
            nearby = []
            seen_ids = set()
            for hit in hits:
                other = shape_index.get(id(hit.shape))
                if (
                    other is not None
                    and other is not creature
                    and other.creature_id not in seen_ids
                ):
                    seen_ids.add(other.creature_id)
                    nearby.append(other)
            return nearby
        if not index.valid:
            index.rebuild(self.creatures)
        output = CandidateBuffer(index)
        maximum_radius = max(
            (other.radius for other in self.creatures),
            default=0.0,
        )
        index.query_into(
            center_x,
            center_y,
            max(0.0, query_distance) + maximum_radius,
            output,
        )
        output.sort_by_stable_id()
        return [
            other
            for other in output
            if other is not creature
            and (
                (other.position[0] - center_x) ** 2
                + (other.position[1] - center_y) ** 2
                <= (max(0.0, query_distance) + other.radius) ** 2
            )
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

    @property
    def selected_behavior_snapshot(self) -> BehaviorSnapshot | None:
        observer = getattr(self, "behavior_observer", None)
        if observer is None:
            return None
        snapshot = observer.latest_snapshot
        if snapshot is None:
            return None
        if (
            snapshot.creature_id != self.selected_creature_id
            or snapshot.selection_generation
            != getattr(self, "_behavior_selection_generation", 0)
        ):
            return None
        return snapshot

    @property
    def behavior_observer_diagnostics(self) -> BehaviorObserverDiagnostics:
        observer = getattr(self, "behavior_observer", None)
        if observer is None:
            return BehaviorObserverDiagnostics(worker_health="unavailable")
        return observer.diagnostics

    @property
    def behavior_history_index(
        self,
    ) -> tuple[CreatureHistoryIndexEntry, ...]:
        history = getattr(self, "behavior_history", None)
        return () if history is None else history.index

    def behavior_report_for(
        self,
        creature_id: int,
    ) -> CreatureBehaviorReport | None:
        history = getattr(self, "behavior_history", None)
        return None if history is None else history.report_for(creature_id)

    @property
    def species_behavior_index(self) -> tuple[SpeciesBehaviorIndexEntry, ...]:
        history = getattr(self, "behavior_history", None)
        if history is None:
            return ()
        living_counts: dict[int, int] = {}
        for creature in self.creatures:
            species_id = creature.lineage.species_id
            living_counts[species_id] = living_counts.get(species_id, 0) + 1
        monitored_counts: dict[int, int] = {}
        creature_by_id = {
            creature.creature_id: creature for creature in self.creatures
        }
        for creature_id in self._behavior_active_subjects:
            creature = creature_by_id.get(creature_id)
            if creature is None:
                continue
            species_id = creature.lineage.species_id
            monitored_counts[species_id] = (
                monitored_counts.get(species_id, 0) + 1
            )
        retained_species = {entry.species_id for entry in history.index}
        species_ids = {*living_counts, *retained_species}
        entries: list[SpeciesBehaviorIndexEntry] = []
        for species_id in species_ids:
            report = history.species_report(species_id)
            alive = living_counts.get(species_id, 0)
            entries.append(
                SpeciesBehaviorIndexEntry(
                    species_id=species_id,
                    alive_population=alive,
                    monitored_count=monitored_counts.get(species_id, 0),
                    observed_creature_count=report.observed_creature_count,
                    total_observation_seconds=(
                        report.total_observation_seconds
                    ),
                    completed_bout_count=report.completed_bout_count,
                    active=alive > 0,
                )
            )
        return tuple(
            sorted(
                entries,
                key=lambda entry: (
                    not entry.active,
                    entry.species_id is None,
                    -1 if entry.species_id is None else entry.species_id,
                ),
            )
        )

    @property
    def automatic_behavior_cohort_ids(self) -> frozenset[int]:
        return frozenset(
            creature_id
            for cohort in self._behavior_automatic_cohort.values()
            for creature_id in cohort
        )

    def species_behavior_report_for(
        self,
        species_id: int | None,
    ) -> SpeciesBehaviorReport:
        report = self.behavior_history.species_report(species_id)
        entry = next(
            (
                item
                for item in self.species_behavior_index
                if item.species_id == species_id
            ),
            None,
        )
        return (
            report
            if entry is None
            else replace(
                report,
                alive_population=entry.alive_population,
                monitored_count=entry.monitored_count,
            )
        )

    @property
    def behavior_history_diagnostics(self) -> BehaviorHistoryDiagnostics:
        history = getattr(self, "behavior_history", None)
        return (
            BehaviorHistoryDiagnostics()
            if history is None
            else history.diagnostics
        )

    def _drain_completed_behavior_bouts(self) -> None:
        observer = getattr(self, "behavior_observer", None)
        history = getattr(self, "behavior_history", None)
        if observer is None or history is None:
            return
        drain = getattr(observer, "drain_completed_bouts", None)
        if callable(drain):
            for draft in drain():
                history.append_draft(draft)
        diagnostics = getattr(observer, "diagnostics", None)
        if diagnostics is not None:
            skipped = max(
                0,
                int(
                    getattr(
                        diagnostics,
                        "history_completions_not_recorded",
                        0,
                    )
                ),
            )
            previously_seen = getattr(
                self,
                "_behavior_history_worker_skipped_seen",
                0,
            )
            if skipped > previously_seen:
                history.record_skipped_completions(
                    skipped - previously_seen
                )
            self._behavior_history_worker_skipped_seen = max(
                previously_seen,
                skipped,
            )

    def _record_behavior_observer_progress(self) -> None:
        observer = getattr(self, "behavior_observer", None)
        history = getattr(self, "behavior_history", None)
        if observer is None or history is None:
            return
        drain = getattr(observer, "drain_progress_snapshots", None)
        if callable(drain):
            snapshots = drain()
        else:
            snapshots = tuple(
                getattr(observer, "latest_snapshots", {}).values()
            )
        for snapshot in snapshots:
            history.record_observation_progress(
                snapshot.creature_id,
                snapshot.selection_generation,
                snapshot.simulation_time,
                snapshot.observations_processed,
            )

    @property
    def selected_why_snapshots(self) -> tuple[WhySnapshot, ...]:
        observer = getattr(self, "behavior_observer", None)
        if observer is None:
            return ()
        selected = self.selected_creature
        behavior_snapshot = self.selected_behavior_snapshot
        if selected is None or behavior_snapshot is None:
            return ()
        brain = self.neat_controller.brain_for(selected.creature_id)
        if brain is None:
            return ()
        current_bouts = {
            (
                state.behavior,
                int(getattr(state, "bout_id", 0)),
                getattr(state, "target_id", None),
            )
            for state in behavior_snapshot.behaviors
        }
        sensor = getattr(self, "_last_sensor_snapshots", {}).get(
            selected.creature_id
        )
        food = None if sensor is None else getattr(sensor, "food", None)
        current_food_target_id = (
            getattr(food, "nearest_id", None)
            if food is not None
            and bool(getattr(food, "visible", 0.0) > 0.0)
            else None
        )
        generation = getattr(self, "_behavior_selection_generation", 0)
        return tuple(
            snapshot
            for snapshot in observer.latest_why_snapshots
            if (
                snapshot.creature_id == selected.creature_id
                and snapshot.selection_generation == generation
                and snapshot.brain_revision == brain.brain_revision
                and (
                    snapshot.behavior,
                    snapshot.bout_id,
                    snapshot.target_id,
                )
                in current_bouts
                and (
                    snapshot.target_id is None
                    or snapshot.target_id == current_food_target_id
                )
            )
        )

    @property
    def counterfactual_diagnostics(self) -> CounterfactualDiagnostics:
        observer = getattr(self, "behavior_observer", None)
        if observer is None:
            return CounterfactualDiagnostics(worker_health="unavailable")
        return observer.counterfactual_diagnostics

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
        return self._sensor_snapshot_for(creature)

    def _sensor_snapshot_for(
        self,
        creature: Creature,
        *,
        pheromone_values: np.ndarray | None = None,
        nearby_creatures=None,
        own_infants=None,
    ) -> SensorSnapshot:
        nearby_foods = self._nearby_foods_for(
            creature,
            creature.vision.range + self.config.food.max_food_radius,
        )
        if nearby_creatures is None:
            flocking_config = getattr(self.config, "flocking", None)
            long_range_config = getattr(flocking_config, "long_range", None)
            nearby_creatures = self._nearby_creatures_for(
                creature,
                max(
                    creature.vision.range,
                    getattr(flocking_config, "perception_radius", 0.0),
                    (
                        long_range_config.range
                        if long_range_config is not None
                        and long_range_config.enabled
                        else 0.0
                    ),
                )
                + self.config.trait.max_radius,
            )
        if own_infants is None:
            own_infants = self._own_infant_children_for(creature)

        fitness = self.fitness.get(creature.creature_id)
        age_seconds = 0.0 if fitness is None else fitness.age_seconds
        chronometer = self._chronometers.get(creature.creature_id, 0.0)

        reproductive_readiness = min(
            age_seconds / max(self.config.population.min_reproduction_age, 0.0001),
            1.0,
        )

        clock_tik_tok = 1.0 if int(age_seconds) % 2 == 0 else 0.0
        clock_chronometer = min(chronometer / 20.0, 1.0)
        clock_time_alive = min(age_seconds / 120.0, 1.0)
        ignored_food_ids = self._ignored_food_ids_for(creature)
        is_grabbing = creature.creature_id in self._held_food_by_creature_id

        reuse_scratch = nearby_creatures is getattr(
            self,
            "_active_candidate_buffer",
            None,
        )
        if reuse_scratch:
            snapshot = self.vision.sense_with_visible_food_ids(
                creature,
                nearby_foods,
                nearby_creatures,
                self.environment_world_bounds,
                self.MAX_SPEED,
                reproductive_readiness=reproductive_readiness,
                clock_tik_tok=clock_tik_tok,
                clock_chronometer=clock_chronometer,
                clock_time_alive=clock_time_alive,
                is_grabbing=is_grabbing,
                ignored_food_ids=ignored_food_ids,
                own_infants=own_infants,
                reuse_scratch=True,
            ).snapshot
        else:
            snapshot = self.vision.sense(
                creature,
                nearby_foods,
                nearby_creatures,
                self.environment_world_bounds,
                self.MAX_SPEED,
                reproductive_readiness=reproductive_readiness,
                clock_tik_tok=clock_tik_tok,
                clock_chronometer=clock_chronometer,
                clock_time_alive=clock_time_alive,
                is_grabbing=is_grabbing,
                ignored_food_ids=ignored_food_ids,
                own_infants=own_infants,
            )

        snapshot.biome = self._biome_sensor_snapshot_for(creature)
        acoustics = getattr(self, "acoustics", None)
        if acoustics is not None:
            if (
                self.debug_vision_enabled
                and self.selected_creature_id == creature.creature_id
            ):
                acoustic_result = acoustics.sense_with_debug(
                    creature.creature_id,
                    creature.position,
                    creature.heading,
                )
                snapshot.acoustic = acoustic_result.observation
                self._last_acoustic_debug[creature.creature_id] = (
                    acoustic_result.debug
                )
            else:
                snapshot.acoustic = acoustics.sense(
                    creature.creature_id,
                    creature.position,
                    creature.heading,
                )
                self._last_acoustic_debug.pop(creature.creature_id, None)
        pheromones = getattr(self, "pheromones", None)
        if pheromones is not None:
            if pheromone_values is None:
                snapshot.pheromones = pheromones.sense(
                    self.pheromone_sensor_positions_for(creature)
                )
            else:
                snapshot.pheromones = PheromoneSnapshot(
                    *(float(value) for value in pheromone_values)
                )
        return snapshot

    def pheromone_sensor_positions_for(
        self,
        creature: Creature,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        return self.biome_sensor_positions_for(creature)

    def _biome_sensor_snapshot_for(self, creature: Creature) -> BiomeSensorSnapshot:
        here, forward_left, forward_right = self.biome_sensor_positions_for(creature)
        return BiomeSensorSnapshot.from_probe_samples(
            self._biome_richness_at(*here),
            self._biome_richness_at(*forward_left),
            self._biome_richness_at(*forward_right),
        )

    def _biome_richness_at(self, x: float, y: float) -> float:
        biome_map = getattr(self, "biome_map", None)
        if biome_map is None:
            return 0.0
        return self._clamp(
            float(biome_map.expected_food_density_at(x, y)),
            0.0,
            1.0,
        )

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

    def creature_id_at(self, x: float, y: float) -> int | None:
        """Return the creature rendered under a screen-space pointer."""
        environment = self.layout.environment
        if not ut.contains(environment, x, y):
            return None
        world_x, world_y = self.screen_to_environment(x, y)
        for creature in reversed(self.creatures):
            if creature.contains_point(world_x, world_y) and self._creature_is_visible(
                creature
            ):
                return creature.creature_id
        return None

    def select_creature_by_id(self, creature_id: int | None) -> None:
        """Select a UI-captured target without repeating mutable hit testing."""
        previous_id = getattr(self, "selected_creature_id", None)
        chosen = next(
            (
                creature
                for creature in self.creatures
                if creature.creature_id == creature_id
            ),
            None,
        )
        selected_id = None if chosen is None else chosen.creature_id
        selection_changed = selected_id != getattr(
            self,
            "selected_creature_id",
            None,
        )
        if selection_changed and previous_id is not None:
            self._finalize_behavior_focus(
                BehaviorTermination.MODE_SWITCHED
                if selected_id is None
                else BehaviorTermination.FOCUS_CHANGED
            )
        self.selected_creature_id = selected_id
        if selection_changed:
            self._reset_behavior_focus(selected_id)
            history = getattr(self, "behavior_history", None)
            if history is not None and chosen is not None:
                history.register_creature(
                    chosen.creature_id,
                    chosen.name,
                    self.elapsed_time,
                    species_id=chosen.lineage.species_id,
                    observation_mode=ObservationMode.FOCAL,
                    observation_generation=(
                        self._behavior_selection_generation
                    ),
                    active=True,
                )
                history.set_active_creatures({chosen.creature_id})
        self._camera_follows_selected_creature = chosen is not None
        self._focus_selected_creature()

    def _finalize_behavior_focus(
        self,
        termination: BehaviorTermination,
    ) -> None:
        observer = getattr(self, "behavior_observer", None)
        finalize = getattr(observer, "finalize_focus", None)
        if callable(finalize):
            finalize(termination)

    def _reset_behavior_focus(self, creature_id: int | None) -> None:
        self._behavior_cohort_dirty = True
        self._behavior_selection_generation = (
            getattr(self, "_behavior_selection_generation", 0) + 1
        )
        self._behavior_food_consumption_count = 0
        self._behavior_food_consumed_energy_total = 0.0
        self._behavior_consumption_totals = {}
        now = getattr(self, "elapsed_time", 0.0)
        behavior_config = getattr(
            getattr(self, "config", None),
            "behavior",
            None,
        )
        if behavior_config is None:
            self._behavior_next_sample_time = now
        else:
            self._behavior_next_sample_time = (
                ceil(now * behavior_config.sample_hz - 1e-12)
                / behavior_config.sample_hz
            )
        why_config = getattr(
            getattr(self, "config", None),
            "counterfactual_why",
            None,
        )
        if why_config is None:
            self._why_next_probe_time = now
        else:
            self._why_next_probe_time = (
                ceil(now * why_config.probe_hz - 1e-12)
                / why_config.probe_hz
            )
        observer = getattr(self, "behavior_observer", None)
        if observer is not None:
            observer.set_focus(
                creature_id,
                self._behavior_selection_generation,
            )
            self._behavior_active_subjects = (
                {}
                if creature_id is None
                else {creature_id: self._behavior_selection_generation}
            )
            if creature_id is not None:
                self._behavior_consumption_totals[creature_id] = (0, 0.0)
            history = getattr(self, "behavior_history", None)
            if history is not None:
                history.set_active_creatures(
                    set(self._behavior_active_subjects)
                )
            set_focal_brain = getattr(observer, "set_focal_brain", None)
            if callable(set_focal_brain):
                brain = (
                    None
                    if creature_id is None
                    or not hasattr(self, "neat_controller")
                    else self.neat_controller.brain_for(creature_id)
                )
                try:
                    payload = (
                        None
                        if brain is None
                        else pickle.dumps(
                            PureNeatEvaluator.from_brain(brain),
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    )
                except (AttributeError, pickle.PickleError, TypeError):
                    brain = None
                    payload = None
                set_focal_brain(
                    FocalBrainUpdate(
                        creature_id=creature_id if brain is not None else None,
                        selection_generation=(
                            self._behavior_selection_generation
                        ),
                        brain_revision=(
                            None if brain is None else brain.brain_revision
                        ),
                        evaluator_payload=payload,
                    )
                )

    def _next_behavior_subject_generation(self) -> int:
        # Automatic identities live in a negative namespace so switching a
        # creature to focal mode can never reuse the same worker key.
        self._behavior_subject_generation_counter -= 1
        return self._behavior_subject_generation_counter

    def _mark_behavior_cohort_dirty(self) -> None:
        self._behavior_cohort_dirty = True

    def _behavior_representative_rank(
        self,
        species_id: int,
        creature_id: int,
    ) -> bytes:
        payload = (
            f"{self.config.random_seed}:{species_id}:{creature_id}"
        ).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=16).digest()

    def _sync_automatic_behavior_cohort(self) -> None:
        """Maintain stable background representatives for every live species."""
        config = getattr(getattr(self, "config", None), "behavior", None)
        observer = getattr(self, "behavior_observer", None)
        if (
            observer is None
            or config is None
            or not config.enabled
            or self.selected_creature is not None
        ):
            return
        target = config.background_representatives_per_species
        by_species: dict[int, list[Creature]] = {}
        for creature in self.creatures:
            by_species.setdefault(
                creature.lineage.species_id,
                [],
            ).append(creature)

        next_cohort: dict[int, tuple[int, ...]] = {}
        creature_by_id = {
            creature.creature_id: creature for creature in self.creatures
        }
        for species_id, members in sorted(by_species.items()):
            member_ids = {member.creature_id for member in members}
            retained = [
                creature_id
                for creature_id in self._behavior_automatic_cohort.get(
                    species_id,
                    (),
                )
                if creature_id in member_ids
            ]
            vacancies = max(0, min(target, len(members)) - len(retained))
            candidates = sorted(
                (
                    member
                    for member in members
                    if member.creature_id not in retained
                ),
                key=lambda member: self._behavior_representative_rank(
                    species_id,
                    member.creature_id,
                ),
            )
            retained.extend(
                member.creature_id for member in candidates[:vacancies]
            )
            next_cohort[species_id] = tuple(retained)
        self._behavior_automatic_cohort = next_cohort

        desired_ids = {
            creature_id
            for cohort in next_cohort.values()
            for creature_id in cohort
        }
        active = {
            creature_id: generation
            for creature_id, generation in self._behavior_active_subjects.items()
            if creature_id in desired_ids
        }
        for creature_id in sorted(desired_ids - set(active)):
            active[creature_id] = self._next_behavior_subject_generation()
            self._behavior_consumption_totals[creature_id] = (0, 0.0)
            creature = creature_by_id[creature_id]
            self.behavior_history.register_creature(
                creature_id,
                creature.name,
                self.elapsed_time,
                species_id=creature.lineage.species_id,
                observation_mode=ObservationMode.AUTOMATIC,
                observation_generation=active[creature_id],
                active=True,
            )
        removed_ids = set(self._behavior_active_subjects) - set(active)
        for creature_id in removed_ids:
            self._behavior_consumption_totals.pop(creature_id, None)
        self._behavior_active_subjects = active
        self.behavior_history.set_active_creatures(set(active))
        observer.set_subjects(tuple(sorted(active.items())))
        self._behavior_cohort_dirty = False

    def _behavior_observation_for(
        self,
        creature: Creature,
        generation: int,
    ) -> BehaviorObservation | None:
        sensor = self._last_sensor_snapshots.get(creature.creature_id)
        if sensor is None:
            return None
        food = sensor.food
        flock = sensor.flock
        pheromones = sensor.pheromones
        compatible_group_visible = (
            float(getattr(flock, "flockmate_count", 0.0)) > 1e-12
        )
        group_direction = (
            self._signed_angle(
                float(
                    getattr(
                        flock,
                        "cohesion_absolute_angle",
                        creature.heading,
                    )
                )
                - creature.heading
            )
            if compatible_group_visible
            else None
        )
        group_velocity = tuple(
            getattr(
                flock,
                "actual_average_flockmate_velocity",
                (0.0, 0.0),
            )
        )
        consumption_count, consumed_energy = getattr(
            self,
            "_behavior_consumption_totals",
            {},
        ).get(
            creature.creature_id,
            (
                getattr(self, "_behavior_food_consumption_count", 0)
                if creature.creature_id == self.selected_creature_id
                else 0,
                getattr(self, "_behavior_food_consumed_energy_total", 0.0)
                if creature.creature_id == self.selected_creature_id
                else 0.0,
            ),
        )
        return BehaviorObservation(
            creature_id=creature.creature_id,
            selection_generation=generation,
            simulation_time=self.elapsed_time,
            x=creature.position[0],
            y=creature.position[1],
            heading=creature.heading,
            angular_velocity=float(creature.body.angular_velocity),
            velocity_x=float(creature.body.velocity.x),
            velocity_y=float(creature.body.velocity.y),
            speed=creature.speed,
            nearest_food_id=getattr(food, "nearest_id", None),
            food_visible=bool(getattr(food, "visible", 0.0) > 0.0),
            food_distance=getattr(food, "surface_distance", None),
            food_relative_angle=getattr(food, "relative_angle", None),
            compatible_group_visible=compatible_group_visible,
            compatible_group_count=float(
                getattr(flock, "flockmate_count", 0.0)
            ),
            compatible_group_distance=(
                float(getattr(flock, "center_distance", 0.0))
                if compatible_group_visible
                else None
            ),
            compatible_group_direction=group_direction,
            group_velocity_x=float(group_velocity[0]),
            group_velocity_y=float(group_velocity[1]),
            personal_space_occupied=bool(
                int(getattr(flock, "visible_personal_space_count", 0)) > 0
            ),
            alarm_here=float(getattr(pheromones, "alarm_here", 0.0)),
            alarm_forward_left=float(
                getattr(pheromones, "alarm_forward_left", 0.0)
            ),
            alarm_forward_right=float(
                getattr(pheromones, "alarm_forward_right", 0.0)
            ),
            carrying_food=(
                creature.creature_id
                in getattr(self, "_held_food_by_creature_id", {})
            ),
            food_consumption_count=consumption_count,
            food_consumed_energy_total=consumed_energy,
        )

    def _sample_selected_behavior(self) -> None:
        observer = getattr(self, "behavior_observer", None)
        config = getattr(getattr(self, "config", None), "behavior", None)
        if config is not None:
            signature = (
                config.enabled,
                config.background_representatives_per_species,
            )
            if signature != getattr(
                self,
                "_behavior_cohort_config_signature",
                None,
            ):
                self._behavior_cohort_config_signature = signature
                self._mark_behavior_cohort_dirty()
        if (
            observer is None
            or config is None
            or not config.enabled
        ):
            return
        selected = self.selected_creature
        now = self.elapsed_time
        next_sample = getattr(self, "_behavior_next_sample_time", now)
        if now + 1e-12 < next_sample:
            return
        interval = 1.0 / config.sample_hz
        skipped_intervals = max(
            1,
            int((now - next_sample) // interval) + 1,
        )
        self._behavior_next_sample_time = (
            next_sample + skipped_intervals * interval
        )
        if selected is None and getattr(
            self,
            "_behavior_cohort_dirty",
            True,
        ):
            self._sync_automatic_behavior_cohort()
        active_subjects = getattr(self, "_behavior_active_subjects", None)
        if active_subjects is None:
            active_subjects = (
                {}
                if selected is None
                else {
                    selected.creature_id: getattr(
                        self,
                        "_behavior_selection_generation",
                        0,
                    )
                }
            )
            self._behavior_active_subjects = active_subjects
        creature_by_id = {
            creature.creature_id: creature for creature in self.creatures
        }
        observations = tuple(
            observation
            for creature_id, generation in sorted(
                active_subjects.items()
            )
            if (creature := creature_by_id.get(creature_id)) is not None
            if (
                observation := self._behavior_observation_for(
                    creature,
                    generation,
                )
            )
            is not None
        )
        submit_batch = getattr(observer, "submit_batch", None)
        if callable(submit_batch):
            submit_batch(observations)
        else:
            for observation in observations:
                observer.submit(observation)

    def _sample_selected_why(self) -> None:
        observer = getattr(self, "behavior_observer", None)
        config = getattr(
            getattr(self, "config", None),
            "counterfactual_why",
            None,
        )
        behavior_config = getattr(
            getattr(self, "config", None),
            "behavior",
            None,
        )
        if (
            observer is None
            or config is None
            or not config.enabled
            or behavior_config is None
            or not behavior_config.enabled
        ):
            return
        selected = self.selected_creature
        snapshot = self.selected_behavior_snapshot
        if selected is None or snapshot is None:
            return
        mapped = mapped_probe_behaviors(snapshot.behaviors)
        sensor = getattr(self, "_last_sensor_snapshots", {}).get(
            selected.creature_id
        )
        food = None if sensor is None else getattr(sensor, "food", None)
        target_visible = bool(
            food is not None and getattr(food, "visible", 0.0) > 0.0
        )
        food_target_id = (
            getattr(food, "nearest_id", None)
            if target_visible
            else None
        )
        raw_food_relative_angle = (
            getattr(food, "relative_angle", None)
            if target_visible
            else None
        )
        food_relative_angle = None
        if raw_food_relative_angle is not None:
            try:
                candidate_angle = float(raw_food_relative_angle)
            except (TypeError, ValueError):
                candidate_angle = float("nan")
            if isfinite(candidate_angle) and abs(candidate_angle) <= pi:
                food_relative_angle = candidate_angle
        target_context_valid = (
            target_visible
            and type(food_target_id) is int
            and food_target_id >= 0
            and food_relative_angle is not None
        )
        flock = None if sensor is None else getattr(sensor, "flock", None)
        group_visible = bool(
            flock is not None
            and float(getattr(flock, "flockmate_count", 0.0)) > 1e-12
        )
        group_relative_angle = None
        if group_visible:
            try:
                candidate_group_angle = self._signed_angle(
                    float(getattr(flock, "cohesion_absolute_angle"))
                    - float(selected.heading)
                )
            except (AttributeError, TypeError, ValueError):
                candidate_group_angle = float("nan")
            if isfinite(candidate_group_angle):
                group_relative_angle = candidate_group_angle
        group_context_valid = (
            group_visible and group_relative_angle is not None
        )
        mapped = tuple(
            observed
            for observed in mapped
            if (
                (
                    observed.behavior
                    not in {
                        BehaviorKind.FOOD_ORIENTATION,
                        BehaviorKind.FOOD_APPROACH,
                    }
                    or (
                        target_context_valid
                        and observed.target_id == food_target_id
                    )
                )
                and (
                    observed.behavior is not BehaviorKind.COHESION
                    or group_context_valid
                )
            )
        )
        if not mapped:
            return
        now = self.elapsed_time
        next_probe = getattr(self, "_why_next_probe_time", now)
        if now + 1e-12 < next_probe:
            return
        interval = 1.0 / config.probe_hz
        skipped_intervals = max(
            1,
            int((now - next_probe) // interval) + 1,
        )
        self._why_next_probe_time = (
            next_probe + skipped_intervals * interval
        )
        brain = self.neat_controller.brain_for(selected.creature_id)
        if (
            brain is None
            or len(brain.last_inputs) != SENSOR_CONTRACT.input_count
            or not brain.last_outputs
        ):
            return
        submit_why = getattr(observer, "submit_why", None)
        if not callable(submit_why):
            return
        submit_why(
            CounterfactualProbeInput(
                creature_id=selected.creature_id,
                selection_generation=getattr(
                    self,
                    "_behavior_selection_generation",
                    0,
                ),
                brain_revision=brain.brain_revision,
                simulation_time=now,
                sensor_schema_version=SENSOR_CONTRACT.schema_version,
                behaviors=mapped,
                actual_inputs=tuple(float(value) for value in brain.last_inputs),
                actual_outputs=tuple(
                    float(value) for value in brain.last_outputs
                ),
                submitted_monotonic=monotonic(),
                target_visible=target_visible,
                food_target_id=(
                    int(food_target_id)
                    if target_context_valid
                    else None
                ),
                food_relative_angle=(
                    food_relative_angle
                    if target_context_valid
                    else None
                ),
                group_visible=group_visible,
                group_relative_angle=(
                    group_relative_angle
                    if group_context_valid
                    else None
                ),
            )
        )

    def select_creature_at(self, x: float, y: float) -> None:
        self.select_creature_by_id(self.creature_id_at(x, y))

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
        cohort = self.config.flocking.cohort_spawn
        if cohort.enabled:
            positions = self._cohort_spawn_positions(
                self.config.population.initial_creatures,
            )
            return [
                self._spawn_creature(
                    index + 1,
                    position=positions[index],
                    color=self.genotype_manager.initial_color(0),
                )
                for index in range(self.config.population.initial_creatures)
            ]
        return [
            self._spawn_creature(
                index + 1,
                color=self.genotype_manager.initial_color(0),
            )
            for index in range(self.config.population.initial_creatures)
        ]

    def _cohort_spawn_positions(
        self,
        count: int,
    ) -> list[tuple[float, float]]:
        cohort = self.config.flocking.cohort_spawn
        left, bottom, right, top = self.environment_world_bounds
        margin = self.config.trait.max_radius + 10.0
        positions: list[tuple[float, float]] = []
        while len(positions) < count:
            center_x = self.rng.uniform(left + margin, right - margin)
            center_y = self.rng.uniform(bottom + margin, top - margin)
            cohort_count = min(cohort.size, count - len(positions))
            for _ in range(cohort_count):
                angle = self.rng.uniform(0.0, 2.0 * pi)
                distance = sqrt(self.rng.random()) * cohort.radius
                positions.append(
                    (
                        self._clamp(
                            center_x + cos(angle) * distance,
                            left + margin,
                            right - margin,
                        ),
                        self._clamp(
                            center_y + sin(angle) * distance,
                            bottom + margin,
                            top - margin,
                        ),
                    )
                )
        return positions

    def _spawn_creature(
        self,
        creature_id: int,
        position: tuple[float, float] | None = None,
        heading: float | None = None,
        energy: float | None = None,
        life: float | None = None,
        color: Color | None = None,
        vision: VisionTraits | None = None,
        physical_traits: PhysicalTraits | None = None,
        flocking_traits: FlockingTraits | None = None,
        lineage: LineageInfo | None = None,
    ) -> Creature:
        """Materialize one creature through the domain factory.

        Parameters
        ----------
        creature_id
            Stable identity reserved for the creature.
        position
            Optional world-space position; a random position is used otherwise.
        heading
            Optional body heading in radians.
        energy
            Optional initial usable energy.
        life
            Optional initial life reserve.
        color
            Optional inherited RGB or RGBA colour.
        vision
            Optional inherited visual traits.
        physical_traits
            Optional inherited body and metabolic traits.
        flocking_traits
            Optional inherited social traits.
        lineage
            Optional ancestry and species metadata.

        Returns
        -------
        Creature
            Fully initialized physical creature, not yet registered as live.

        Raises
        ------
        ValueError
            If ``creature_id`` is live or was already issued this session.
        """
        # Resolve bounds once so the factory receives only creature-domain data.
        left, bottom, right, top = self.environment_world_bounds
        existing = getattr(self, "_living_creatures", {}).get(creature_id)
        if existing is not None:
            raise ValueError(f"Duplicate live creature ID {creature_id}.")
        if creature_id in getattr(self, "_issued_creature_ids", ()):
            raise ValueError(
                f"Creature ID {creature_id} was already issued in this session."
            )

        # Delegate physics and genotype construction while World guards identity.
        creature = self.creature_factory.create(
            creature_id,
            (left, bottom, right, top),
            self.rng,
            position=position,
            heading=heading,
            energy=energy,
            life=life,
            color=color,
            vision=vision,
            physical_traits=physical_traits,
            flocking_traits=flocking_traits,
            lineage=lineage,
        )
        self._index_creature_shape(creature.shape, creature)
        return creature

    def _mutated_child_traits(self, parent: Creature) -> ChildCreatureTraits:
        """Build inherited non-neural traits and lineage for one child.

        Parameters
        ----------
        parent
            Live parent supplying genotype and lineage.

        Returns
        -------
        ChildCreatureTraits
            Mutated traits and ancestry ready for neural evolution.
        """
        # Delegate all non-neural mutation to the genotype subsystem.
        return self._mutated_child_traits_from_parent_values(
            parent_id=parent.creature_id,
            parent_generation=parent.lineage.generation,
            parent_species_id=parent.lineage.species_id,
            parent_vision=parent.vision,
            parent_physical_traits=parent.physical_traits,
            parent_flocking_traits=parent.flocking_traits,
            parent_color=parent.color,
        )

    def _mutated_child_traits_from_parent_values(
        self,
        *,
        parent_id: int,
        parent_generation: int,
        parent_species_id: int,
        parent_vision: VisionTraits,
        parent_physical_traits: PhysicalTraits,
        parent_flocking_traits: FlockingTraits,
        parent_color: Color,
    ) -> ChildCreatureTraits:
        """Build child traits from live or archived parent values.

        Parameters
        ----------
        parent_id
            Stable identity of the parent.
        parent_generation
            Parent generation number.
        parent_species_id
            Parent species identity before child speciation.
        parent_vision
            Parent visual traits.
        parent_physical_traits
            Parent physical traits.
        parent_flocking_traits
            Parent social traits.
        parent_color
            Parent RGB or RGBA colour.

        Returns
        -------
        ChildCreatureTraits
            Mutated traits and lineage for the child.
        """
        # One aggregate mutation replaces four previously separate World paths.
        result = self.genotype_manager.mutate(
            CreatureGenotype(
                parent_vision,
                parent_physical_traits,
                parent_flocking_traits,
                parent_color,
            ),
            self.rng,
        )
        genotype = result.genotype
        return ChildCreatureTraits(
            vision=genotype.vision,
            physical_traits=genotype.physical_traits,
            flocking_traits=genotype.flocking_traits,
            color=genotype.color,
            lineage=LineageInfo(
                parent_id=parent_id,
                generation=parent_generation + 1,
                species_id=parent_species_id,
                mutation_delta=result.mutation_delta,
            ),
        )

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
        angle = parent.heading + self.rng.choice((-pi / 4, pi / 4))
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
        """Reserve the next stable creature identity through lifecycle state.

        Parameters
        ----------
        None
            This method receives no external parameters.

        Returns
        -------
        int
            Identity that has not previously been issued in this session.
        """
        # Reconstruct compatibility fixtures that instantiate World without init.
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
        if not hasattr(self, "lifecycle"):
            self.lifecycle = CreatureLifecycleService()
            self.lifecycle.issued_ids = getattr(self, "_issued_creature_ids", set())
        self.lifecycle.synchronize_allocator(self._next_creature_id_value)
        creature_id = self.lifecycle.allocate_id()
        self._next_creature_id_value = self.lifecycle.next_id_value
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
        self._flocking_capture_due_this_step = (
            self._flocking_telemetry_is_due()
        )
        vision = getattr(self, "vision", None)
        if vision is not None:
            vision.failure_injector = (
                self._scheduler_validation_failure_point
                if hasattr(self, "_scheduler_validation_failure_injector")
                else None
            )
        self._cache_creature_spatial_state()
        legacy_spatial_observer = (
            "_apply_creature_intents_with_spatial_cache" in self.__dict__
        )
        if legacy_spatial_observer:
            self._creature_spatial_state = {
                creature.creature_id: self._creature_spatial_values(creature)
                for creature in self.creatures
            }
        try:
            self._apply_creature_intents_with_spatial_cache()
        finally:
            # Space.step() moves bodies immediately after this behavior pass;
            # the completed generation intentionally remains available only
            # as pre-physics runtime state and is rebuilt next fixed step.
            if getattr(self, "_candidate_buffer_leased", False):
                self._release_candidate_buffer()
            self._creature_spatial_state = None

    def _register_living_creature(self, creature: Creature) -> None:
        """Register a successful spawn and advance the stable ID allocator.

        Parameters
        ----------
        creature
            Creature that has completed physics creation.

        Returns
        -------
        None
            Lifecycle and allocator state are updated.

        Raises
        ------
        ValueError
            If the identity is already live or was previously issued.
        """
        # Lifecycle owns identity validation while World maintains the allocator.
        if not hasattr(self, "lifecycle"):
            self.lifecycle = CreatureLifecycleService()
            self.lifecycle.living = getattr(self, "_living_creatures", {})
            self.lifecycle.issued_ids = getattr(self, "_issued_creature_ids", set())
            self._living_creatures = self.lifecycle.living
            self._issued_creature_ids = self.lifecycle.issued_ids
        self.lifecycle.register(creature)
        creature_id = creature.creature_id
        if hasattr(self, "_next_creature_id_value"):
            self._next_creature_id_value = max(
                self._next_creature_id_value,
                creature_id + 1,
            )
            self.lifecycle.synchronize_allocator(self._next_creature_id_value)

    def _unregister_living_creature(self, creature: Creature) -> None:
        """Unregister a creature and notify transient cache owners.

        Parameters
        ----------
        creature
            Creature leaving the live simulation.

        Returns
        -------
        None
            The live registry and registered service caches are cleared.
        """
        # Lifecycle also notifies every transient cache owner in stable order.
        if not hasattr(self, "lifecycle"):
            self.lifecycle = CreatureLifecycleService()
            self.lifecycle.living = getattr(self, "_living_creatures", {})
            self.lifecycle.issued_ids = getattr(self, "_issued_creature_ids", set())
        self.lifecycle.unregister(creature)

    def _acquire_candidate_buffer(self) -> CandidateBuffer:
        if getattr(self, "_candidate_buffer_leased", False):
            raise AssertionError("The world candidate buffer is non-reentrant.")
        self._candidate_buffer_leased = True
        buffer = self._candidate_buffer
        buffer.reset(self._creature_spatial_index, self._creature_spatial_index.generation)
        return buffer

    def _release_candidate_buffer(self) -> None:
        buffer = getattr(self, "_candidate_buffer", None)
        if buffer is not None:
            buffer.invalidate()
        vision = getattr(self, "vision", None)
        if vision is not None:
            vision.clear_scratch()
        self._active_candidate_buffer = None
        self._candidate_buffer_leased = False

    def _broad_phase_geometry_for(
        self,
        creature: Creature,
    ) -> BroadPhaseGeometry:
        self._ensure_spatial_runtime()
        index = self._creature_spatial_index
        if not index.valid:
            index.rebuild(self.creatures)
        maximum_radius = index.maximum_radius
        long_range = self.config.flocking.long_range
        return BroadPhaseGeometry.calculate(
            observer_radius=creature.radius,
            maximum_target_radius=maximum_radius,
            collision_margin=self.config.action.collision_avoidance_margin,
            vision_range=creature.vision.range,
            flock_range=self.config.flocking.perception_radius,
            long_range=long_range.range,
            long_range_enabled=long_range.enabled,
        )

    def _query_envelope_for(
        self,
        creature: Creature,
        *,
        scheduled: bool,
    ) -> float:
        index = self._creature_spatial_index
        observer_radius = creature.radius
        target_radius = index.maximum_radius
        collision = (
            observer_radius
            + self.config.action.collision_avoidance_margin
            + target_radius
        )
        if not scheduled:
            return collision
        vision = (
            0.35 * observer_radius
            + creature.vision.range
            + target_radius
        )
        flocking = self.config.flocking
        long_range = (
            flocking.long_range.range
            if flocking.long_range.enabled
            else 0.0
        )
        return max(
            collision,
            vision,
            flocking.perception_radius,
            long_range,
        )

    def _query_candidates_for(
        self,
        creature: Creature,
        centre_radius: float,
        output: CandidateBuffer,
        *,
        scheduled: bool,
    ) -> None:
        center_x, center_y, _ = self._creature_spatial_values(creature)
        if hasattr(self, "_scheduler_validation_failure_injector"):
            self._scheduler_validation_failure_point("grid.query")
        self._creature_spatial_index.query_into(
            center_x,
            center_y,
            centre_radius,
            output,
        )
        counter_name = (
            "_spatial_scheduled_queries"
            if scheduled
            else "_spatial_collision_only_queries"
        )
        setattr(self, counter_name, getattr(self, counter_name, 0) + 1)
        output.sort_by_stable_id()

    @staticmethod
    def _neutral_action() -> Action:
        """Return a fresh compatibility neutral action.

        Parameters
        ----------
        None
            This callable receives no external parameters.

        Returns
        -------
        Action
            Independent action with every intent disabled.
        """
        # Delegate to the action domain so all neutral construction is canonical.
        return neutral_action()

    def _initialize_creature_runtime_state(self, creature: Creature) -> None:
        """Install deterministic neutral transient state for a live creature."""
        if not hasattr(self, "_last_actions"):
            self._last_actions = {}
        if not hasattr(self, "_effective_actions"):
            self._effective_actions = {}
        if not hasattr(self, "_cached_social_intentions"):
            self._cached_social_intentions = {}
        if not hasattr(self, "_motion_commands"):
            self._motion_commands = {}
        creature_id = creature.creature_id
        action = self._neutral_action()
        self._last_actions[creature_id] = action
        self._effective_actions[creature_id] = action
        try:
            creature.last_action = action
        except AttributeError:
            pass
        self._cached_social_intentions[creature_id] = SocialRuntime()
        self._motion_commands[creature_id] = MotionCommand(
            effective_rotate=0.0,
            max_speed=self.MAX_SPEED,
            max_angular_speed=self.MAX_ANGULAR_SPEED,
        )

    def _decision_phase(self, creature_id: int) -> int:
        if type(creature_id) is not int:
            raise TypeError("creature_id must be a stable integer.")
        return creature_id % self.config.scheduler.decision_period_steps

    def _apply_creature_intents_with_spatial_cache(self) -> None:
        """
        Apply the intents of all creatures in the simulation, based on their
        sensor snapshots and the decisions made by their respective controllers.
        """
        if not hasattr(self, "_last_sensor_snapshots"):
            self._last_sensor_snapshots = {}
        if not hasattr(self, "_last_actions"):
            self._last_actions = {}
        if not hasattr(self, "_effective_actions"):
            self._effective_actions = {}
        if not hasattr(self, "_cached_social_intentions"):
            self._cached_social_intentions = {}
        if not hasattr(self, "_motion_commands"):
            self._motion_commands = {}

        thinking_rows: dict[int, int] = {}
        thinking_creatures: list[Creature] = []
        simulation_step = getattr(self, "_simulation_step", 0)
        decision_period = self.config.scheduler.decision_period_steps
        selected_creature_id = getattr(self, "selected_creature_id", None)
        capture_global_diagnostics = getattr(
            self,
            "_flocking_capture_due_this_step",
            False,
        )
        for creature in self.creatures:
            creature_id = creature.creature_id
            if self._last_actions.get(creature_id) is None:
                self._initialize_creature_runtime_state(creature)
            if (
                simulation_step % decision_period
                == self._decision_phase(creature_id)
            ):
                thinking_rows[creature_id] = len(thinking_creatures)
                thinking_creatures.append(creature)

        pheromones = getattr(self, "pheromones", None)
        if pheromones is not None and thinking_creatures:
            self._ensure_pheromone_sensor_buffer_capacity(len(thinking_creatures))
            for row, creature in enumerate(thinking_creatures):
                self._pheromone_sensor_positions[row] = (
                    self.pheromone_sensor_positions_for(creature)
                )
            pheromones.sense_many(
                self._pheromone_sensor_positions[: len(thinking_creatures)],
                out=self._pheromone_sensor_values[: len(thinking_creatures)],
            )

        for creature in self.creatures:
            creature_id = creature.creature_id
            action = self._last_actions.get(creature_id)
            snapshot = self._last_sensor_snapshots.get(creature_id)
            thinking_row = thinking_rows.get(creature_id)
            should_think = thinking_row is not None
            use_spatial_candidates = (
                hasattr(creature, "radius")
                and hasattr(creature, "vision")
            )
            candidates = None
            if use_spatial_candidates:
                candidates = self._acquire_candidate_buffer()
                self._query_candidates_for(
                    creature,
                    self._query_envelope_for(
                        creature,
                        scheduled=should_think,
                    ),
                    candidates,
                    scheduled=should_think,
                )
                self._active_candidate_buffer = candidates
            capture_inputs = (
                creature_id == selected_creature_id
                or capture_global_diagnostics
            )

            if should_think:
                sensing_kwargs = {
                    "pheromone_values": (
                        None
                        if pheromones is None
                        else self._pheromone_sensor_values[thinking_row]
                    ),
                }
                if use_spatial_candidates:
                    sensing_kwargs["nearby_creatures"] = candidates
                    sensing_kwargs["own_infants"] = self._own_infant_view_for(
                        creature
                    )
                snapshot = self._sensor_snapshot_for(creature, **sensing_kwargs)
                decide_with_capture = getattr(
                    self.neat_controller,
                    "decide_with_input_capture",
                    None,
                )
                decider = (
                    decide_with_capture
                    if capture_inputs and callable(decide_with_capture)
                    else self.neat_controller.decide
                )
                action = self._decide_with_duration(
                    decider,
                    creature_id,
                    snapshot,
                )

                self._last_actions[creature.creature_id] = action
                self._last_sensor_snapshots[creature.creature_id] = snapshot
                try:
                    creature.last_action = action
                except AttributeError:
                    pass

                if is_active_intent(action.reset_chronometer):
                    self._chronometers[creature.creature_id] = 0.0

            if action is None:
                if use_spatial_candidates:
                    self._release_candidate_buffer()
                continue

            if capture_inputs and not should_think and snapshot is not None:
                capture_snapshot = getattr(
                    self.neat_controller,
                    "capture_input_snapshot",
                    None,
                )
                if callable(capture_snapshot):
                    capture_snapshot(creature_id)

            cached_effective_action = self._effective_actions.get(creature_id)
            energy_depleted = is_energy_depleted(creature.energy)
            if should_think:
                effective_action = self._effective_action_for(creature, action)
            elif not energy_depleted:
                # Recovery immediately exposes the authoritative cached raw
                # action again without allocating a replacement.
                effective_action = action
            elif (
                cached_effective_action is None
                or cached_effective_action is action
            ):
                # Allocate the gated view only on the depletion transition.
                effective_action = self._effective_action_for(creature, action)
            else:
                effective_action = cached_effective_action
            self._effective_actions[creature_id] = effective_action
            if should_think:
                self._apply_carry_intent(creature, effective_action)
            self._apply_action(
                creature,
                effective_action,
                snapshot,
                capture_runtime=(
                    creature_id == selected_creature_id
                    or capture_global_diagnostics
                ),
                capture_steering_debug=(
                    creature_id == selected_creature_id
                    and bool(getattr(self, "debug_vision_enabled", False))
                ),
                refresh_intention=should_think,
            )
            if use_spatial_candidates:
                self._release_candidate_buffer()

    def _decide_with_duration(
        self,
        decider,
        creature_id: int,
        snapshot: SensorSnapshot,
    ) -> Action:
        decision_dt = (
            self.fixed_timestep
            * self.config.scheduler.decision_period_steps
        )
        try:
            return decider(
                creature_id,
                snapshot,
                decision_dt=decision_dt,
            )
        except TypeError as error:
            if "decision_dt" not in str(error):
                raise
            return decider(creature_id, snapshot)

    @staticmethod
    def _effective_action_for(creature: Creature, raw_action: Action) -> Action:
        """Gate non-locomotion outputs without mutating neural diagnostics."""
        if not is_energy_depleted(creature.energy):
            return raw_action
        return replace(
            raw_action,
            want_reproduce=0.0,
            want_grab=0.0,
            want_nurse=0.0,
            emit_sound=0.0,
            emit_trail_pheromone=0.0,
            emit_alarm_pheromone=0.0,
        )

    @staticmethod
    def _smoothed_rest_value(
        previous_rest: float,
        rest_intent: float,
        delta_time: float,
        activation_rate: float,
        decay_rate: float,
    ) -> float:
        """Apply the asymmetric, elapsed-time-aware rest response filter."""
        previous = min(1.0, max(0.0, float(previous_rest)))
        intent = min(1.0, max(0.0, float(rest_intent)))
        rate = activation_rate if intent >= previous else decay_rate
        alpha = 1.0 - exp(-max(0.0, float(rate)) * max(0.0, delta_time))
        return min(1.0, max(0.0, previous + (intent - previous) * alpha))

    def _action_for_execution(self, creature_id: int) -> Action | None:
        """Return effective runtime output, with legacy-test compatibility."""
        effective_actions = getattr(self, "_effective_actions", None)
        if effective_actions is None:
            return getattr(self, "_last_actions", {}).get(creature_id)
        return effective_actions.get(creature_id)

    def _commit_communication_intents(self, delta_time: float) -> None:
        acoustics = getattr(self, "acoustics", None)
        pheromones = getattr(self, "pheromones", None)
        if acoustics is None or pheromones is None:
            return

        signals: list[AcousticSignal] = []
        deposit_rate = max(0.0, self.config.communication.pheromone_deposit_rate)
        elapsed = max(0.0, delta_time)
        deposit_count = 0
        for creature in self.creatures:
            action = self._action_for_execution(creature.creature_id)
            if action is None:
                continue
            sound_strength = max(0.0, min(1.0, action.emit_sound))
            if (
                sound_strength
                >= self.config.communication.acoustic_min_emission_strength
            ):
                signals.append(
                    AcousticSignal(
                        emitter_id=creature.creature_id,
                        position=creature.position,
                        strength=sound_strength,
                        tone=max(-1.0, min(1.0, action.sound_tone)),
                    )
                )
            trail_amount = (
                deposit_rate
                * max(0.0, min(1.0, action.emit_trail_pheromone))
                * elapsed
            )
            alarm_amount = (
                deposit_rate
                * max(0.0, min(1.0, action.emit_alarm_pheromone))
                * elapsed
            )
            if trail_amount <= 0.0 and alarm_amount <= 0.0:
                continue
            if deposit_count == 0:
                self._ensure_communication_buffer_capacity(len(self.creatures))
            self._communication_positions[deposit_count] = creature.position
            self._communication_trail_amounts[deposit_count] = trail_amount
            self._communication_alarm_amounts[deposit_count] = alarm_amount
            deposit_count += 1
        if deposit_count:
            pheromones.deposit_many(
                self._communication_positions[:deposit_count],
                self._communication_trail_amounts[:deposit_count],
                self._communication_alarm_amounts[:deposit_count],
            )
        acoustics.replace_signals(signals)

    def _ensure_communication_buffer_capacity(self, required: int) -> None:
        """Grow reusable communication arrays to hold ``required`` emissions.

        Parameters
        ----------
        required
            Minimum number of emission rows needed by the current fixed step.

        Returns
        -------
        None
            World aliases and social-service buffers reference the grown arrays.
        """
        # Reuse existing storage whenever capacity already covers the workload.
        positions = getattr(self, "_communication_positions", None)
        current = 0 if positions is None else positions.shape[0]
        if current >= required:
            return
        capacity = max(16, required, current * 2)
        self._communication_positions = np.empty((capacity, 2), dtype=np.float64)
        self._communication_trail_amounts = np.empty(capacity, dtype=np.float64)
        self._communication_alarm_amounts = np.empty(capacity, dtype=np.float64)
        # Keep social-service reusable buffers aligned after geometric growth.
        social = getattr(self, "social", None)
        if social is not None:
            social.communication_positions = self._communication_positions
            social.communication_trail_amounts = self._communication_trail_amounts
            social.communication_alarm_amounts = self._communication_alarm_amounts

    def _ensure_pheromone_sensor_buffer_capacity(self, required: int) -> None:
        positions = getattr(self, "_pheromone_sensor_positions", None)
        current = 0 if positions is None else positions.shape[0]
        if current >= required:
            return
        capacity = max(16, required, current * 2)
        self._pheromone_sensor_positions = np.empty(
            (capacity, 3, 2),
            dtype=np.float64,
        )
        self._pheromone_sensor_values = np.empty(
            (capacity, 6),
            dtype=np.float32,
        )

    def _apply_action(
        self,
        creature: Creature,
        action: Action,
        snapshot: SensorSnapshot | None = None,
        *,
        capture_runtime: bool = True,
        capture_steering_debug: bool = True,
        refresh_intention: bool = True,
    ) -> None:
        """
        Apply the specified action to the given creature, considering its current state,
        sensor snapshot, and the simulation's configuration. This method calculates the
        necessary forces and torques to apply to the creature's body based on the action's
        parameters, including acceleration, rotation, and panic intensity. It also handles
        flocking behavior when appropriate.

        Args:
            creature (Creature): The creature to which the action will be applied.
            action (Action): The action to apply, containing acceleration, rotation, and other parameters.
            snapshot (SensorSnapshot | None): The sensor snapshot of the creature, used for flocking calculations. If None, flocking forces will not be applied.
        """
        if capture_runtime and not hasattr(self, "_last_flocking_runtime"):
            self._last_flocking_runtime = {}

        rest_intent = self._clamp(getattr(action, "rest", 0.0), 0.0, 1.0)
        fixed_dt = getattr(self, "fixed_timestep", self.FIXED_TIMESTEP)
        creature.rest_intent = rest_intent
        previous_rest = getattr(creature, "smoothed_rest", 0.0)
        smoothed_rest = self._smoothed_rest_value(
            previous_rest,
            rest_intent,
            fixed_dt,
            self.config.action.rest_response_rate,
            self.config.action.rest_decay_rate,
        )
        creature.smoothed_rest = smoothed_rest
        movement_scale = 1.0 - smoothed_rest ** max(
            1e-12,
            self.config.action.rest_movement_exponent,
        )

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
        ) ** (fixed_dt / self.FIXED_TIMESTEP)
        current_max_angular_speed = (
            self.MAX_ANGULAR_SPEED * sprint_multiplier * turn_control_gain
        )

        reference_alpha = self._clamp(
            self.config.action.action_smoothing_alpha,
            0.0,
            1.0,
        )
        alpha = self._physics_rate_alpha(reference_alpha, fixed_dt)
        previous_acceleration = getattr(creature, "smoothed_acceleration", 0.0)
        smoothed_acceleration = (
            previous_acceleration * (1.0 - alpha) + target_thrust * alpha
        )
        try:
            creature.smoothed_acceleration = smoothed_acceleration
        except AttributeError:
            pass
        thrust = smoothed_acceleration

        # Allocate the finite force budget by priority. Lower-priority forces
        # cannot spend budget already consumed by collision avoidance, and
        # their components opposing avoidance are removed entirely.
        remaining_force_budget = max(0.0, current_max_forward_force)
        requested_avoidance_force = self._collision_avoidance_force(
            creature,
            current_max_forward_force,
        )
        mandatory_avoidance_force, remaining_force_budget = (
            self._allocate_force_budget(
                requested_avoidance_force,
                remaining_force_budget,
            )
        )

        current_velocity = (
            creature.body.velocity.x,
            creature.body.velocity.y,
        )
        neural_request = acceleration_force_vector(
            thrust,
            creature.heading,
            current_max_forward_force,
            current_max_backward_force,
        )
        neural_desired_velocity = (
            current_velocity[0] + neural_request[0],
            current_velocity[1] + neural_request[1],
        )
        cached_social = getattr(
            self,
            "_cached_social_intentions",
            {},
        ).get(creature.creature_id)
        if cached_social is not None and not isinstance(
            cached_social, SocialRuntime
        ):
            cached_social = SocialRuntime.from_legacy(cached_social)
            self._cached_social_intentions[creature.creature_id] = cached_social
        if refresh_intention or cached_social is None:
            if not hasattr(self, "_cached_social_intentions"):
                self._cached_social_intentions = {}
            if cached_social is None:
                cached_social = SocialRuntime()
                self._cached_social_intentions[creature.creature_id] = (
                    cached_social
                )
            self._refresh_social_runtime(
                cached_social,
                creature,
                action,
                snapshot,
                current_max_speed,
                current_max_forward_force,
                current_velocity,
                neural_desired_velocity,
                neural_request,
            )
        social_influence = cached_social.influence
        requested_social_contribution = (
            cached_social.requested_contribution_x,
            cached_social.requested_contribution_y,
        )
        blended_request = (
            neural_request[0] + cached_social.requested_contribution_x,
            neural_request[1] + cached_social.requested_contribution_y,
        )
        blended_desired_velocity = (
            current_velocity[0] + blended_request[0],
            current_velocity[1] + blended_request[1],
        )
        (
            voluntary_force,
            _counterfactual_neural_force,
            accepted_counterfactual_delta,
        ) = accepted_counterfactual_contribution(
            blended_request=blended_request,
            neural_request=neural_request,
            mandatory_avoidance=mandatory_avoidance_force,
            remaining_budget=remaining_force_budget,
        )
        voluntary_force = (
            voluntary_force[0] * movement_scale,
            voluntary_force[1] * movement_scale,
        )
        creature.effective_voluntary_motor_effort = self._clamp(
            hypot(*voluntary_force) / max(1e-12, current_max_forward_force),
            0.0,
            1.0,
        )
        turn_steering_force = (
            mandatory_avoidance_force[0] + accepted_counterfactual_delta[0],
            mandatory_avoidance_force[1] + accepted_counterfactual_delta[1],
        )
        effective_herding = self._clamp(
            getattr(action, "herding", 0.0),
            0.0,
            1.0,
        )
        benchmark_config = self.config.flocking.benchmark
        if benchmark_config.enabled and refresh_intention:
            qualities = getattr(
                self,
                "_flocking_benchmark_quality_by_creature_id",
                None,
            )
            if qualities is None:
                qualities = {}
                self._flocking_benchmark_quality_by_creature_id = qualities
            if cached_social.effective_count <= 0.0:
                quality = 0.0
            else:
                group_presence = self._clamp(
                    cached_social.effective_count
                    / max(1, benchmark_config.target_group_size - 1),
                    0.0,
                    1.0,
                )
                alignment_quality = self._clamp(
                    1.0 - cached_social.mean_heading_error / pi,
                    0.0,
                    1.0,
                )
                spacing_quality = exp(
                    -(
                        (
                            cached_social.mean_neighbor_distance
                            - benchmark_config.target_spacing
                        )
                        / benchmark_config.spacing_tolerance
                    )
                    ** 2
                )
                movement_quality = min(
                    hypot(
                        cached_social.mean_group_velocity_x,
                        cached_social.mean_group_velocity_y,
                    )
                    / benchmark_config.reference_speed,
                    1.0,
                )
                quality = (
                    group_presence
                    * alignment_quality
                    * spacing_quality
                    * movement_quality
                )
            qualities[creature.creature_id] = quality

        if capture_runtime:
            raw_neural_herding = effective_herding
            brain_for = getattr(
                getattr(self, "neat_controller", None),
                "brain_for",
                None,
            )
            if callable(brain_for):
                brain = brain_for(creature.creature_id)
                raw_neural_herding = self._clamp(
                    getattr(brain, "last_raw_herding", effective_herding),
                    0.0,
                    1.0,
                )
            self._last_flocking_runtime[creature.creature_id] = FlockingRuntimeSnapshot(
                observation=cached_social.observation,
                intent=cached_social.intent,
                neural_desired_velocity=neural_desired_velocity,
                blended_desired_velocity=blended_desired_velocity,
                mandatory_avoidance=mandatory_avoidance_force,
                requested_social_contribution=requested_social_contribution,
                accepted_counterfactual_delta=accepted_counterfactual_delta,
                social_influence=social_influence,
                raw_neural_herding=raw_neural_herding,
                effective_herding=effective_herding,
                panic=panic,
            )

        if capture_steering_debug:
            if not hasattr(self, "_last_flock_steering_debug"):
                self._last_flock_steering_debug = {}
            self._last_flock_steering_debug[creature.creature_id] = _FlockSteeringDebug(
                accepted_counterfactual_delta,
                current_max_forward_force,
            )
        # The blended voluntary request and mandatory avoidance share the
        # finite force budget.
        total_force = (
            voluntary_force[0]
            + mandatory_avoidance_force[0],
            voluntary_force[1]
            + mandatory_avoidance_force[1],
        )
        # Both social steering and mandatory avoidance may turn the creature away
        # from its direct neural heading.
        flock_turn_bias = self._flock_turn_bias(
            creature,
            turn_steering_force,
            current_max_forward_force,
        )

        # Determine the target turn based on desired rotation and flock turn bias, then smooth it before applying angular control.
        neural_turn_scale = 1.0 - (
            self.config.action.rest_rotation_inhibition
            * smoothed_rest
        )
        target_turn = self._clamp(
            action.rotate * neural_turn_scale + flock_turn_bias,
            -1.0,
            1.0,
        )
        previous_rotation = getattr(creature, "smoothed_rotation", 0.0)
        turn = previous_rotation * (1.0 - alpha) + target_turn * alpha
        try:
            creature.smoothed_rotation = turn
        except AttributeError:
            pass

        if not hasattr(self, "_motion_commands"):
            self._motion_commands = {}
        command = self._motion_commands.get(creature.creature_id)
        if command is None:
            command = MotionCommand(
                effective_rotate=turn,
                max_speed=current_max_speed,
                max_angular_speed=current_max_angular_speed,
            )
            self._motion_commands[creature.creature_id] = command
        else:
            command.effective_rotate = turn
            command.max_speed = current_max_speed
            command.max_angular_speed = current_max_angular_speed

        # Apply the calculated total force to the creature's body at its current position, influencing its movement in the simulation. The force is applied in world coordinates, ensuring that it affects the creature's velocity and trajectory appropriately.
        creature.body.apply_force_at_world_point(
            total_force,
            creature.body.position,
        )

        self._apply_turn_control(
            creature,
            turn,
            max_angular_speed=current_max_angular_speed,
        )
        creature.body.angular_velocity *= active_angular_velocity_retention

    def _flock_component_forces(
        self,
        creature: Creature,
        snapshot: SensorSnapshot,
        max_speed: float,
        max_force: float,
    ) -> tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]:
        flock = snapshot.flock
        separation = self._steering_toward_relative_angle(
            creature,
            self._signed_angle(
                flock.crowd_separation_absolute_angle - creature.heading
            ),
            max_speed,
            max_force,
            flock.crowd_separation_strength,
        )

        if flock.flockmate_count <= 0:
            alignment = (0.0, 0.0)
            cohesion = (0.0, 0.0)
        else:
            alignment = self._steering_toward_velocity(
                creature,
                flock.average_flockmate_velocity,
                max_force,
                flock.average_flockmate_proximity,
            )
            cohesion = self._steering_toward_relative_angle(
                creature,
                self._signed_angle(
                    flock.cohesion_absolute_angle - creature.heading
                ),
                max_speed,
                max_force,
                1.0 - flock.center_proximity,
            )
        return separation, alignment, cohesion

    @staticmethod
    def _social_observation(flock: object) -> SocialObservation:
        effective_count = max(
            0.0,
            float(getattr(flock, "flockmate_count", 0.0)),
        )
        personal_space_count = max(
            0,
            int(getattr(flock, "visible_personal_space_count", 0)),
        )
        separation_strength = max(
            0.0,
            min(
                1.0,
                float(getattr(flock, "crowd_separation_strength", 0.0)),
            ),
        )
        return SocialObservation(
            present=effective_count > 1e-12,
            visible_creature_count=max(
                0,
                int(getattr(flock, "visible_creature_count", 0)),
            ),
            compatible_visible_count=max(
                0,
                int(getattr(flock, "compatible_visible_count", 0)),
            ),
            personal_space_presence=(
                1.0
                if personal_space_count > 0
                else (1.0 if separation_strength > 1e-12 else 0.0)
            ),
            social_presence=max(0.0, min(1.0, effective_count)),
            effective_count=effective_count,
            center_forward=float(getattr(flock, "center_forward", 0.0)),
            center_right=float(getattr(flock, "center_right", 0.0)),
            relative_velocity_forward=float(
                getattr(flock, "relative_velocity_forward", 0.0)
            ),
            relative_velocity_right=float(
                getattr(flock, "relative_velocity_right", 0.0)
            ),
            mean_proximity=max(
                0.0,
                min(
                    1.0,
                    float(
                        getattr(
                            flock,
                            "average_flockmate_proximity",
                            0.0,
                        )
                    ),
                ),
            ),
            center_distance=max(
                0.0,
                float(getattr(flock, "center_distance", 0.0)),
            ),
            mean_neighbor_distance=max(
                0.0,
                float(getattr(flock, "mean_neighbor_distance", 0.0)),
            ),
            mean_heading_error=max(
                0.0,
                float(getattr(flock, "mean_heading_error", 0.0)),
            ),
            mean_group_velocity=tuple(
                getattr(
                    flock,
                    "actual_average_flockmate_velocity",
                    getattr(
                        flock,
                        "average_flockmate_velocity",
                        (0.0, 0.0),
                    ),
                )
            ),
            long_range=getattr(
                flock,
                "long_range",
                _EMPTY_LONG_RANGE_SOCIAL_OBSERVATION,
            ),
        )

    def _refresh_social_runtime(
        self,
        runtime: SocialRuntime,
        creature: Creature,
        action: Action,
        snapshot: SensorSnapshot | None,
        max_speed: float,
        max_force: float,
        current_velocity: tuple[float, float],
        neural_desired_velocity: tuple[float, float],
        neural_request: tuple[float, float],
    ) -> None:
        """Refresh retained scalar social state without immutable hot-path objects."""
        if snapshot is None:
            separation_x = separation_y = 0.0
            alignment_x = alignment_y = 0.0
            cohesion_x = cohesion_y = 0.0
            runtime.observation_present = False
            runtime.visible_creature_count = 0
            runtime.compatible_visible_count = 0
            runtime.personal_space_presence = 0.0
            runtime.social_presence = 0.0
            runtime.effective_count = 0.0
            runtime.center_forward = 0.0
            runtime.center_right = 0.0
            runtime.relative_velocity_forward = 0.0
            runtime.relative_velocity_right = 0.0
            runtime.separation_forward = 0.0
            runtime.separation_right = 0.0
            runtime.mean_proximity = 0.0
            runtime.center_distance = 0.0
            runtime.mean_neighbor_distance = 0.0
            runtime.mean_heading_error = 0.0
            runtime.mean_group_velocity_x = 0.0
            runtime.mean_group_velocity_y = 0.0
            runtime.long_range_intensity = 0.0
            runtime.long_range_direction_forward = 0.0
            runtime.long_range_direction_right = 0.0
            separation_weight = alignment_weight = cohesion_weight = 0.0
            engagement = 0.0
            panic_attenuation = 1.0
        else:
            separation, alignment, cohesion = self._flock_component_forces(
                creature,
                snapshot,
                max_speed,
                max_force,
            )
            separation_x, separation_y = separation
            alignment_x, alignment_y = alignment
            cohesion_x, cohesion_y = cohesion
            flock = snapshot.flock
            effective_count = max(
                0.0,
                float(getattr(flock, "flockmate_count", 0.0)),
            )
            personal_space_count = max(
                0,
                int(getattr(flock, "visible_personal_space_count", 0)),
            )
            separation_strength = self._clamp(
                float(getattr(flock, "crowd_separation_strength", 0.0)),
                0.0,
                1.0,
            )
            runtime.observation_present = effective_count > 1e-12
            runtime.visible_creature_count = max(
                0,
                int(getattr(flock, "visible_creature_count", 0)),
            )
            runtime.compatible_visible_count = max(
                0,
                int(getattr(flock, "compatible_visible_count", 0)),
            )
            runtime.personal_space_presence = (
                1.0
                if personal_space_count > 0 or separation_strength > 1e-12
                else 0.0
            )
            runtime.social_presence = self._clamp(effective_count, 0.0, 1.0)
            runtime.effective_count = effective_count
            runtime.center_forward = float(
                getattr(flock, "center_forward", 0.0)
            )
            runtime.center_right = float(getattr(flock, "center_right", 0.0))
            runtime.relative_velocity_forward = float(
                getattr(flock, "relative_velocity_forward", 0.0)
            )
            runtime.relative_velocity_right = float(
                getattr(flock, "relative_velocity_right", 0.0)
            )
            runtime.separation_forward = 0.0
            runtime.separation_right = 0.0
            runtime.mean_proximity = self._clamp(
                float(
                    getattr(flock, "average_flockmate_proximity", 0.0)
                ),
                0.0,
                1.0,
            )
            runtime.center_distance = max(
                0.0,
                float(getattr(flock, "center_distance", 0.0)),
            )
            runtime.mean_neighbor_distance = max(
                0.0,
                float(getattr(flock, "mean_neighbor_distance", 0.0)),
            )
            runtime.mean_heading_error = max(
                0.0,
                float(getattr(flock, "mean_heading_error", 0.0)),
            )
            group_velocity = getattr(
                flock,
                "actual_average_flockmate_velocity",
                getattr(flock, "average_flockmate_velocity", (0.0, 0.0)),
            )
            runtime.mean_group_velocity_x = float(group_velocity[0])
            runtime.mean_group_velocity_y = float(group_velocity[1])
            long_range = getattr(
                flock,
                "long_range",
                _EMPTY_LONG_RANGE_SOCIAL_OBSERVATION,
            )
            runtime.long_range_intensity = float(long_range.intensity)
            runtime.long_range_direction_forward = float(
                long_range.direction_forward
            )
            runtime.long_range_direction_right = float(
                long_range.direction_right
            )

            traits = creature.flocking_traits
            herding = self._clamp(getattr(action, "herding", 0.0), 0.0, 1.0)
            panic_value = self._clamp(
                getattr(action, "flee_panic_intensity", 0.0),
                0.0,
                1.0,
            )
            separation_gene = self._clamp(traits.separation_gene, 0.0, 1.0)
            alignment_gene = self._clamp(traits.alignment_gene, 0.0, 1.0)
            cohesion_gene = self._clamp(traits.cohesion_gene, 0.0, 1.0)
            minimum_engagement = self._clamp(
                self.config.flocking.minimum_social_engagement,
                0.0,
                1.0,
            )
            suppression = self._clamp(
                self.config.flocking.panic_suppression_strength,
                0.0,
                1.0,
            )
            engagement = runtime.social_presence * (
                minimum_engagement
                + (1.0 - minimum_engagement) * herding
            )
            panic_attenuation = 1.0 - suppression * panic_value
            separation_weight = (
                runtime.personal_space_presence * separation_gene
            )
            alignment_weight = (
                engagement * alignment_gene * panic_attenuation
            )
            cohesion_weight = engagement * cohesion_gene * panic_attenuation

        separation_velocity_x = current_velocity[0] + separation_x
        separation_velocity_y = current_velocity[1] + separation_y
        alignment_velocity_x = current_velocity[0] + alignment_x
        alignment_velocity_y = current_velocity[1] + alignment_y
        cohesion_velocity_x = current_velocity[0] + cohesion_x
        cohesion_velocity_y = current_velocity[1] + cohesion_y
        weighted_x = (
            separation_velocity_x * separation_weight
            + alignment_velocity_x * alignment_weight
            + cohesion_velocity_x * cohesion_weight
        )
        weighted_y = (
            separation_velocity_y * separation_weight
            + alignment_velocity_y * alignment_weight
            + cohesion_velocity_y * cohesion_weight
        )
        weight_sum = separation_weight + alignment_weight + cohesion_weight
        if weight_sum <= 1e-12:
            desired_x, desired_y = current_velocity
        else:
            desired_x = weighted_x / weight_sum
            desired_y = weighted_y / weight_sum
        desired_magnitude = hypot(desired_x, desired_y)
        maximum = max(0.0, float(max_speed))
        if desired_magnitude > maximum and desired_magnitude > 1e-12:
            desired_scale = maximum / desired_magnitude
            desired_x *= desired_scale
            desired_y *= desired_scale
        group_scale = self._clamp(
            max(0.0, runtime.effective_count)
            / max(1, int(self.config.flocking.target_group_size)),
            0.0,
            1.0,
        )
        confidence = self._clamp(
            max(separation_weight, alignment_weight, cohesion_weight)
            * (0.5 + 0.5 * group_scale),
            0.0,
            1.0,
        )
        max_influence = self.config.flocking.max_social_influence
        influence = self._clamp(
            max_influence * confidence,
            0.0,
            max_influence,
        )
        if influence <= 0.0:
            blended_x, blended_y = neural_desired_velocity
        else:
            blended_x = (
                neural_desired_velocity[0] * (1.0 - influence)
                + desired_x * influence
            )
            blended_y = (
                neural_desired_velocity[1] * (1.0 - influence)
                + desired_y * influence
            )

        runtime.desired_velocity_x = desired_x
        runtime.desired_velocity_y = desired_y
        runtime.requested_force_x = desired_x - current_velocity[0]
        runtime.requested_force_y = desired_y - current_velocity[1]
        runtime.confidence = confidence
        runtime.weight_separation = separation_weight
        runtime.weight_alignment = alignment_weight
        runtime.weight_cohesion = cohesion_weight
        runtime.weight_engagement = engagement
        runtime.weight_panic_attenuation = panic_attenuation
        runtime.separation_velocity_x = separation_velocity_x
        runtime.separation_velocity_y = separation_velocity_y
        runtime.alignment_velocity_x = alignment_velocity_x
        runtime.alignment_velocity_y = alignment_velocity_y
        runtime.cohesion_velocity_x = cohesion_velocity_x
        runtime.cohesion_velocity_y = cohesion_velocity_y
        runtime.influence = influence
        runtime.requested_contribution_x = (
            blended_x - current_velocity[0] - neural_request[0]
        )
        runtime.requested_contribution_y = (
            blended_y - current_velocity[1] - neural_request[1]
        )

    def _social_intent(
        self,
        creature: Creature,
        action: Action,
        snapshot: SensorSnapshot | None,
        max_speed: float,
        max_force: float,
    ):
        if snapshot is None:
            observation = SocialObservation()
            return (
                calculate_social_intent(
                    current_velocity=(
                        creature.body.velocity.x,
                        creature.body.velocity.y,
                    ),
                    separation_velocity=(
                        creature.body.velocity.x,
                        creature.body.velocity.y,
                    ),
                    alignment_velocity=(
                        creature.body.velocity.x,
                        creature.body.velocity.y,
                    ),
                    cohesion_velocity=(
                        creature.body.velocity.x,
                        creature.body.velocity.y,
                    ),
                    weights=calculate_flocking_weights(
                        herding=0.0,
                        panic=0.0,
                        separation_gene=0.0,
                        alignment_gene=0.0,
                        cohesion_gene=0.0,
                        personal_space_presence=0.0,
                        social_presence=0.0,
                    ),
                    effective_count=0.0,
                    target_group_size=self.config.flocking.target_group_size,
                    max_speed=max_speed,
                ),
                observation,
            )

        separation, alignment, cohesion = self._flock_component_forces(
            creature,
            snapshot,
            max_speed,
            max_force,
        )
        current_velocity = (
            creature.body.velocity.x,
            creature.body.velocity.y,
        )
        observation = self._social_observation(snapshot.flock)
        traits = creature.flocking_traits
        weights = calculate_flocking_weights(
            herding=getattr(action, "herding", 0.0),
            panic=getattr(action, "flee_panic_intensity", 0.0),
            separation_gene=traits.separation_gene,
            alignment_gene=traits.alignment_gene,
            cohesion_gene=traits.cohesion_gene,
            personal_space_presence=observation.personal_space_presence,
            social_presence=observation.social_presence,
            minimum_social_engagement=(
                self.config.flocking.minimum_social_engagement
            ),
            panic_suppression_strength=(
                self.config.flocking.panic_suppression_strength
            ),
        )

        def desired_from_force(force: tuple[float, float]) -> tuple[float, float]:
            return (
                current_velocity[0] + force[0],
                current_velocity[1] + force[1],
            )

        return (
            calculate_social_intent(
                current_velocity=current_velocity,
                separation_velocity=desired_from_force(separation),
                alignment_velocity=desired_from_force(alignment),
                cohesion_velocity=desired_from_force(cohesion),
                weights=weights,
                effective_count=observation.effective_count,
                target_group_size=self.config.flocking.target_group_size,
                max_speed=max_speed,
            ),
            observation,
        )

    def _steering_toward_velocity(
        self,
        creature: Creature,
        desired_velocity: tuple[float, float],
        max_force: float,
        strength: float,
    ) -> tuple[float, float]:
        """Match flock velocity without inventing a maximum-speed target."""
        strength = self._clamp(strength, 0.0, 1.0)
        if strength <= 0.0:
            return 0.0, 0.0
        steering = (
            desired_velocity[0] - creature.body.velocity.x,
            desired_velocity[1] - creature.body.velocity.y,
        )
        limited = self._limit_vector(steering, max_force)
        return limited[0] * strength, limited[1] * strength

    def _collision_avoidance_force(
        self,
        creature: Creature,
        max_force: float,
    ) -> tuple[float, float]:
        """Return mandatory short-range avoidance for every nearby creature."""
        margin = max(0.0, self.config.action.collision_avoidance_margin)
        scale = max(0.0, self.config.action.collision_avoidance_force_scale)
        if max_force <= 0.0 or scale <= 0.0:
            return 0.0, 0.0

        center_x, center_y, radius = self._creature_spatial_values(creature)
        neighbors = getattr(self, "_active_candidate_buffer", None)
        if neighbors is None:
            # Deterministic reference/debug path outside the fixed-step lease.
            if hasattr(self, "_living_creatures"):
                neighbors = sorted(
                    self._living_creatures.values(),
                    key=lambda other: other.creature_id,
                )
            else:
                neighbors = self._query_nearby_creatures(
                    creature,
                    radius + margin,
                )
        avoidance_x = 0.0
        avoidance_y = 0.0
        spatial_buffer = (
            neighbors if isinstance(neighbors, CandidateBuffer) else None
        )
        if spatial_buffer is not None:
            index = self._creature_spatial_index
            for position in range(spatial_buffer.count):
                slot = spatial_buffer.slots[position]
                neighbor = index.creatures[slot]
                stable_id = index.stable_ids[slot]
                if (
                    index.slot_generations[slot] != index.generation
                    or neighbor is None
                    or neighbor.creature_id != stable_id
                    or index.living_registry.get(stable_id) is not neighbor
                ):
                    index.counters.invalid_slots_skipped += 1
                    continue
                if neighbor is creature:
                    continue
                if hasattr(self, "_scheduler_validation_failure_injector"):
                    self._scheduler_validation_failure_point(
                        "collision.evaluation"
                    )
                away_x = center_x - index.centres_x[slot]
                away_y = center_y - index.centres_y[slot]
                distance_squared = away_x * away_x + away_y * away_y
                safe_distance = max(
                    1e-9,
                    radius + index.radii[slot] + margin,
                )
                if distance_squared >= safe_distance * safe_distance:
                    continue
                if distance_squared <= 1e-24:
                    direction = (
                        1.0
                        if creature.creature_id < neighbor.creature_id
                        else -1.0
                    )
                    unit_x, unit_y = direction, 0.0
                    distance = 0.0
                else:
                    distance = sqrt(distance_squared)
                    unit_x, unit_y = away_x / distance, away_y / distance
                strength = self._clamp(
                    (safe_distance - distance) / safe_distance,
                    0.0,
                    1.0,
                )
                avoidance_x += unit_x * strength
                avoidance_y += unit_y * strength
        else:
            for neighbor in neighbors:
                if neighbor is creature:
                    continue
                if hasattr(self, "_scheduler_validation_failure_injector"):
                    self._scheduler_validation_failure_point(
                        "collision.evaluation"
                    )
                neighbor_x, neighbor_y, neighbor_radius = (
                    self._creature_spatial_values(neighbor)
                )
                away_x = center_x - neighbor_x
                away_y = center_y - neighbor_y
                distance_squared = away_x * away_x + away_y * away_y
                safe_distance = max(
                    1e-9,
                    radius + neighbor_radius + margin,
                )
                if distance_squared >= safe_distance * safe_distance:
                    continue
                if distance_squared <= 1e-24:
                    direction = (
                        1.0
                        if creature.creature_id < neighbor.creature_id
                        else -1.0
                    )
                    unit_x, unit_y = direction, 0.0
                    distance = 0.0
                else:
                    distance = sqrt(distance_squared)
                    unit_x, unit_y = away_x / distance, away_y / distance
                strength = self._clamp(
                    (safe_distance - distance) / safe_distance,
                    0.0,
                    1.0,
                )
                avoidance_x += unit_x * strength
                avoidance_y += unit_y * strength

        magnitude = hypot(avoidance_x, avoidance_y)
        if magnitude <= 1e-12:
            return 0.0, 0.0
        force_magnitude = min(max_force, max_force * scale * min(1.0, magnitude))
        return (
            avoidance_x / magnitude * force_magnitude,
            avoidance_y / magnitude * force_magnitude,
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
        steering_force: tuple[float, float],
        max_force: float,
    ) -> float:
        if max_force <= 0.0:
            return 0.0

        left_x = -sin(creature.heading)
        left_y = cos(creature.heading)
        lateral_force = steering_force[0] * left_x + steering_force[1] * left_y
        lateral_ratio = self._clamp(
            lateral_force / max_force,
            -1.0,
            1.0,
        )
        return lateral_ratio * self.config.action.max_flock_turn_bias

    def _allocate_force_budget(
        self,
        candidate: tuple[float, float],
        remaining_budget: float,
    ) -> tuple[tuple[float, float], float]:
        budget = max(0.0, remaining_budget)
        allocated = self._limit_vector(candidate, budget)
        return allocated, max(0.0, budget - hypot(*allocated))

    @staticmethod
    def _remove_opposing_component(
        candidate: tuple[float, float],
        protected_force: tuple[float, float],
    ) -> tuple[float, float]:
        """Prevent a lower-priority force from reversing collision avoidance."""
        protected_magnitude = hypot(*protected_force)
        if protected_magnitude <= 1e-12:
            return candidate
        unit_x = protected_force[0] / protected_magnitude
        unit_y = protected_force[1] / protected_magnitude
        projection = candidate[0] * unit_x + candidate[1] * unit_y
        if projection >= 0.0:
            return candidate
        return (
            candidate[0] - unit_x * projection,
            candidate[1] - unit_y * projection,
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
        wrapped = (angle + pi) % (2.0 * pi) - pi
        return pi if wrapped == -pi and angle > 0.0 else wrapped

    def _apply_top_down_motion(self, delta_time: float | None = None) -> None:
        for creature in self.creatures:
            self._apply_planar_drag(creature, delta_time)

    def _apply_planar_drag(
        self,
        creature: Creature,
        delta_time: float | None = None,
    ) -> None:
        velocity = creature.body.velocity
        heading = creature.heading
        forward_x = cos(heading)
        forward_y = sin(heading)
        lateral_x = -sin(heading)
        lateral_y = cos(heading)

        forward_speed = velocity.x * forward_x + velocity.y * forward_y
        lateral_speed = velocity.x * lateral_x + velocity.y * lateral_y

        elapsed = (
            getattr(self, "fixed_timestep", self.FIXED_TIMESTEP)
            if delta_time is None
            else max(0.0, delta_time)
        )
        time_scale = elapsed / self.FIXED_TIMESTEP
        forward_speed *= self.config.action.forward_velocity_retention ** time_scale
        lateral_speed *= self.config.action.lateral_velocity_retention ** time_scale
        rest_braking = exp(
            -max(0.0, self.config.action.rest_braking_strength)
            * self._clamp(getattr(creature, "smoothed_rest", 0.0), 0.0, 1.0)
            * elapsed
        )
        forward_speed *= rest_braking
        lateral_speed *= rest_braking

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
        response = self._physics_rate_alpha(
            max(0.0, min(1.0, response)),
            getattr(self, "fixed_timestep", self.FIXED_TIMESTEP),
        )
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
            # Pymunk deliberately leaves angles unbounded.  Long simulations
            # otherwise accumulate thousands of full turns, making downstream
            # angle reduction needlessly expensive and bloating checkpoints.
            creature.body.angle = self._signed_angle(creature.body.angle)

    def _keep_creatures_inside_bounds(self) -> None:
        left, bottom, right, top = self.environment_world_bounds
        reindex_shape = getattr(self.space, "reindex_shape", None)
        for creature in self.creatures:
            x, y = creature.position
            radius = creature.radius + 2.0
            clamped_x = max(left + radius, min(right - radius, x))
            clamped_y = max(bottom + radius, min(top - radius, y))
            if clamped_x != x or clamped_y != y:
                creature.body.position = (clamped_x, clamped_y)
                if reindex_shape is not None:
                    reindex_shape(creature.shape)
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

        self._camera_follows_selected_creature = True
        self.environment_zoom = max(
            self.config.zoom.minimum,
            min(self.config.zoom.maximum, self.SELECTED_CREATURE_ZOOM),
        )
        self._follow_selected_creature()

    def _follow_selected_creature(self) -> None:
        if not self._camera_follow_enabled():
            return
        selected = self.selected_creature
        if selected is None:
            return

        selected_x, selected_y = selected.position
        self.environment_pan_x = -selected_x * self.environment_zoom
        self.environment_pan_y = -selected_y * self.environment_zoom
        self._clamp_environment_pan()

    def _camera_follow_enabled(self) -> bool:
        return bool(
            getattr(self, "_camera_follows_selected_creature", True)
        )

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
            self.elapsed_time,
        )

    def _update_fitness_survival(self, delta_time: float) -> None:
        for creature in self.creatures:
            fitness = self.fitness.get(creature.creature_id)
            if fitness is not None:
                previous_age = fitness.age_seconds
                fitness.record_tick(delta_time, creature.speed)
                self._record_maturity_if_crossed(creature, previous_age, fitness)

    def _apply_carry_intent(self, creature: Creature, action: Action) -> None:
        if is_active_intent(action.want_release):
            self._release_food_for(creature)
            return

        if not is_active_intent(action.want_grab):
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

    def _prepare_reproduction_requests(self) -> list[ReproductionRequest]:
        due = bool(getattr(self, "_reproduction_due_this_step", False))
        self._reproduction_due_this_step = False
        if not due or not self._has_reproduction_resources():
            return []
        eligible_pool = self._eligible_reproduction_parents()
        population_config = getattr(
            getattr(self, "config", None),
            "population",
            None,
        )
        tournament_k1 = getattr(population_config, "tournament_k1", 3)
        tournament_k2 = getattr(population_config, "tournament_k2", 2)
        remaining = list(eligible_pool)
        requests: list[ReproductionRequest] = []
        while remaining:
            selection_pool_size = len(remaining)
            selector = getattr(self.rt_neat, "select_parent", None)
            parent = (
                selector(
                    remaining,
                    tournament_k1,
                    tournament_k2,
                )
                if callable(selector)
                else remaining[0]
            )
            if parent is None:
                break
            remaining.remove(parent)
            rank = len(requests)
            requests.append(
                self._reproduction_request_for(
                    parent,
                    rank,
                    selection_pool_size,
                )
            )
        return sorted(
            requests,
            key=lambda request: (
                request.eligibility_rank,
                request.parent.creature_id,
            ),
        )

    def _reproduction_request_for(
        self,
        parent: Creature,
        rank: int,
        selection_pool_size: int,
    ) -> ReproductionRequest:
        network_size = getattr(self.rt_neat, "network_size", None)
        node_count, enabled_connection_count = (
            network_size(parent)
            if callable(network_size)
            else (0, 0)
        )
        complexity_for = getattr(
            self.rt_neat,
            "network_complexity",
            None,
        )
        network_complexity = (
            complexity_for(parent)
            if callable(complexity_for)
            else float(node_count + enabled_connection_count)
        )
        return ReproductionRequest(
            parent=parent,
            eligibility_rank=rank,
            reserved_energy_cost=self._reproduction_cost_for(parent),
            selection_pool_size=selection_pool_size,
            node_count=node_count,
            enabled_connection_count=enabled_connection_count,
            network_complexity=network_complexity,
        )

    def _eligible_reproduction_parents(self) -> list[Creature]:
        """Return live adults that are eligible and currently want offspring."""
        legacy_ids = set(getattr(self.rt_neat, "eligible_parent_ids", ()))
        checker = getattr(self.rt_neat, "is_reproduction_eligible", None)
        candidates: list[Creature] = []
        for parent in self.creatures:
            fitness = self.fitness.get(parent.creature_id)
            if fitness is None:
                continue
            if callable(checker):
                if not checker(parent, fitness, self.config.population):
                    continue
            elif parent.creature_id not in legacy_ids:
                continue
            action = self._action_for_execution(parent.creature_id)
            if action is None or not is_active_intent(action.want_reproduce):
                continue
            candidates.append(parent)
        return candidates

    def _parent_is_reproduction_eligible(self, parent: Creature) -> bool:
        """Revalidate a queued parent against authoritative current state."""
        if not hasattr(self, "fitness") or not hasattr(self, "rt_neat"):
            return True
        fitness = self.fitness.get(parent.creature_id)
        if fitness is None:
            return False
        checker = getattr(self.rt_neat, "is_reproduction_eligible", None)
        if callable(checker):
            return bool(checker(parent, fitness, self.config.population))
        return parent.creature_id in set(
            getattr(self.rt_neat, "eligible_parent_ids", ())
        )

    def _prepare_nursing_requests(
        self,
        delta_time: float,
    ) -> list[NursingRequest]:
        transfer = (
            max(0.0, self.config.population.nursing_energy_transfer_rate)
            * max(0.0, delta_time)
        )
        if transfer <= 0.0:
            return []
        requests: list[NursingRequest] = []
        for donor in self.creatures:
            action = self._action_for_execution(donor.creature_id)
            if action is None or not is_active_intent(action.want_nurse):
                continue
            target = self._nearest_nursable_infant_for(donor)
            if target is None:
                continue
            requests.append(NursingRequest(donor, target, transfer))
        return requests

    def _energy_demands_for(
        self,
        delta_time: float,
    ) -> tuple[dict[int, float], dict[int, float]]:
        """Calculate biology demand without changing inherited trait values.

        Parameters
        ----------
        delta_time
            Simulated biology interval in seconds.

        Returns
        -------
        tuple[dict[int, float], dict[int, float]]
            Total and powered-movement demand by creature identity.
        """
        demands: dict[int, float] = {}
        powered_movement_demands: dict[int, float] = {}
        # Infant penalties are local calculation inputs, never genotype mutations.
        for creature in self.creatures:
            action = self._action_for_execution(creature.creature_id)
            sprint = (
                0.0
                if action is None
                else getattr(action, "flee_panic_intensity", 0.0)
            )
            physical_traits = getattr(creature, "physical_traits", None)
            inherited_movement_cost = getattr(
                physical_traits,
                "movement_cost_multiplier",
                None,
            )
            movement_cost_multiplier = (
                None
                if inherited_movement_cost is None
                else inherited_movement_cost
                * (3.0 if self._is_infant(creature) else 1.0)
            )
            breakdown = self.metabolism.energy_cost_breakdown_per_second(
                creature,
                self.MAX_SPEED,
                sprint_intensity=sprint,
                age_seconds=self._creature_age_seconds(creature),
                communication_intensities=(
                    self._communication_intensities_for(creature.creature_id)
                ),
                movement_cost_multiplier=movement_cost_multiplier,
            )
            multiplier = max(0.0, delta_time) * self._senescence_factor_for(
                creature
            )
            powered_movement_rate = breakdown.sprint
            if (
                getattr(creature, "effective_voluntary_motor_effort", 0.0)
                > ENERGY_EPSILON
            ):
                powered_movement_rate += breakdown.movement
            demands[creature.creature_id] = breakdown.total * multiplier
            powered_movement_demands[creature.creature_id] = (
                powered_movement_rate * multiplier
            )
        return demands, powered_movement_demands

    def _upkeep_demands_for(self, delta_time: float) -> dict[int, float]:
        """Compatibility view of complete same-step energy demand."""
        return self._energy_demands_for(delta_time)[0]

    def _resolve_resource_transactions(
        self,
        delta_time: float,
    ) -> TransactionResolution:
        self._scheduler_validation_failure_point(
            "biology.resource_candidate_preparation"
        )
        self._scheduler_validation_failure_point(
            "biology.reproduction_preparation"
        )
        reproduction_requests = self._prepare_reproduction_requests()
        self._scheduler_validation_failure_point(
            "biology.nursing_preparation"
        )
        nursing_requests = self._prepare_nursing_requests(delta_time)
        upkeep_demands, powered_movement_demands = self._energy_demands_for(
            delta_time
        )
        baseline_candidates: dict[int, ResourceCandidate] = {}
        baseline_activities: dict[int, ActivityResult] = {}
        for creature in self.creatures:
            self._scheduler_validation_failure_point(
                "biology.digestion_evaluation"
            )
            self._scheduler_validation_failure_point(
                "biology.metabolism_evaluation"
            )
            activity = self._activity_for(creature)
            baseline_activities[creature.creature_id] = activity
            baseline_candidates[creature.creature_id] = (
                self.metabolism.evaluate_candidate(
                    creature,
                    delta_time,
                    total_energy_demand=upkeep_demands[creature.creature_id],
                    powered_movement_energy_demand=(
                        powered_movement_demands[creature.creature_id]
                    ),
                    effective_rest=(
                        creature.smoothed_rest * (1.0 - activity.total)
                    ),
                )
            )

        reproduction_capacity = min(
            1,
            max(
                0,
                self.config.population.max_creatures - len(self.creatures),
            ),
        )
        banned: set[int] = set()
        attempted: dict[int, tuple[ReproductionRequest, str]] = {}
        selected = reproduction_requests[:reproduction_capacity]
        if not selected and not nursing_requests:
            return TransactionResolution(
                candidates=baseline_candidates,
                activities=baseline_activities,
                reproductions=[],
                nursing_transfers=[],
            )
        while True:
            for request in selected:
                attempted.setdefault(
                    request.parent.creature_id,
                    (request, "committed"),
                )
            resolution, failed = self._resolve_transaction_pass(
                delta_time,
                selected,
                nursing_requests,
                upkeep_demands,
                baseline_candidates,
                baseline_activities,
                powered_movement_demands=powered_movement_demands,
            )
            banned.update(failed)
            for creature_id, outcome in failed.items():
                request = attempted[creature_id][0]
                attempted[creature_id] = (request, outcome)
            survivors = [
                request
                for request in selected
                if request.parent.creature_id not in banned
            ]
            promoted = [
                request
                for request in reproduction_requests
                if request.parent.creature_id not in banned
                and request not in survivors
            ]
            next_selected = [
                *survivors,
                *promoted[: max(0, reproduction_capacity - len(survivors))],
            ]
            next_selected.sort(
                key=lambda request: (
                    request.eligibility_rank,
                    request.parent.creature_id,
                )
            )
            if [r.parent.creature_id for r in next_selected] == [
                r.parent.creature_id for r in selected
            ]:
                resolution.reproduction_attempts = list(attempted.values())
                return resolution
            selected = next_selected

    def _resolve_transaction_pass(
        self,
        delta_time: float,
        selected_reproductions: list[ReproductionRequest],
        nursing_requests: list[NursingRequest],
        upkeep_demands: dict[int, float],
        baseline_candidates: dict[int, ResourceCandidate],
        baseline_activities: dict[int, ActivityResult],
        *,
        powered_movement_demands: dict[int, float] | None = None,
    ) -> tuple[TransactionResolution, dict[int, str]]:
        powered_demands = powered_movement_demands or {
            creature_id: 0.0 for creature_id in upkeep_demands
        }
        selected_by_id = {
            request.parent.creature_id: request
            for request in selected_reproductions
        }
        failed_reproductions: dict[int, str] = {}
        chosen_candidates: dict[int, ResourceCandidate] = {}
        chosen_activities: dict[int, ActivityResult] = {}
        accepted: list[AcceptedNursingTransfer] = []

        def choose(
            creature: Creature,
            nursing_transfer: float = 0.0,
        ) -> tuple[ResourceCandidate, bool]:
            creature_id = creature.creature_id
            reproduction = selected_by_id.get(creature_id)
            if (
                reproduction is not None
                and creature_id not in failed_reproductions
                and not self._parent_is_reproduction_eligible(creature)
            ):
                failed_reproductions[creature_id] = "eligibility_rejected"
            reproduction_cost = (
                0.0
                if reproduction is None
                or creature_id in failed_reproductions
                else reproduction.reserved_energy_cost
            )
            has_action = reproduction_cost > 0.0 or nursing_transfer > 0.0
            if not has_action:
                candidate = baseline_candidates[creature_id]
                activity = baseline_activities[creature_id]
                chosen_candidates[creature_id] = candidate
                chosen_activities[creature_id] = activity
                return candidate, candidate.survives
            activity = self._activity_for(
                creature,
                reproduction_selected=reproduction_cost > 0.0,
                nursing_transfer=nursing_transfer,
            )
            candidate = self.metabolism.evaluate_candidate(
                creature,
                delta_time,
                total_energy_demand=(
                    upkeep_demands[creature_id]
                    + reproduction_cost
                    + nursing_transfer
                ),
                powered_movement_energy_demand=powered_demands.get(
                    creature_id,
                    0.0,
                ),
                effective_rest=(
                    creature.smoothed_rest * (1.0 - activity.total)
                ),
            )
            if not candidate.survives:
                if (
                    reproduction is not None
                    and creature_id not in failed_reproductions
                ):
                    failed_reproductions[creature_id] = "resource_rejected"
                candidate = baseline_candidates[creature_id]
                activity = baseline_activities[creature_id]
                chosen_candidates[creature_id] = candidate
                chosen_activities[creature_id] = activity
                return candidate, False
            chosen_candidates[creature_id] = candidate
            chosen_activities[creature_id] = activity
            return candidate, True

        grouped: dict[int, list[NursingRequest]] = {}
        targets: dict[int, Creature] = {}
        for request in nursing_requests:
            target_id = request.target.creature_id
            grouped.setdefault(target_id, []).append(request)
            targets[target_id] = request.target
        target_order = sorted(
            targets.values(),
            key=lambda target: (
                -int(getattr(target.lineage, "generation", 0)),
                target.creature_id,
            ),
        )
        for target in target_order:
            target_candidate = chosen_candidates.get(target.creature_id)
            if target_candidate is None:
                target_candidate, _ = choose(target)
            if not target_candidate.survives:
                continue
            remaining_headroom = max(
                0.0,
                self.config.metabolism.max_energy
                - target_candidate.final_energy,
            )
            for request in sorted(
                grouped[target.creature_id],
                key=lambda item: item.donor.creature_id,
            ):
                allocation = min(
                    max(0.0, request.requested_transfer),
                    remaining_headroom,
                )
                if allocation <= 1e-12:
                    continue
                donor_candidate, action_survives = choose(
                    request.donor,
                    allocation,
                )
                if not action_survives or not donor_candidate.survives:
                    continue
                accepted.append(AcceptedNursingTransfer(request, allocation))
                remaining_headroom = max(0.0, remaining_headroom - allocation)

        for creature in self.creatures:
            if creature.creature_id not in chosen_candidates:
                choose(creature)

        surviving_reproductions = [
            request
            for request in selected_reproductions
            if request.parent.creature_id not in failed_reproductions
        ]
        return (
            TransactionResolution(
                candidates=chosen_candidates,
                activities=chosen_activities,
                reproductions=surviving_reproductions,
                nursing_transfers=accepted,
            ),
            failed_reproductions,
        )

    def _stage_final_reproductions(
        self,
        requests: list[ReproductionRequest],
    ) -> tuple[list[StagedOffspring], object | None, object | None]:
        """Stage accepted offspring against isolated evolution and RNG state.

        Parameters
        ----------
        requests
            Deterministically ordered reproduction requests to stage.

        Returns
        -------
        tuple[list[StagedOffspring], object | None, object | None]
            Offspring records, shadow evolution state, and staged RNG state.

        Raises
        ------
        RuntimeError
            If a selected parent unexpectedly lacks a neural brain while staging.

        Notes
        -----
        Live RNGs, allocators, representatives, and revisions remain unchanged
        until :meth:`_commit_staged_reproductions` accepts the whole batch.
        """
        # Empty batches bypass all cloning and preserve allocator positions.
        if not requests:
            return [], None, None
        # Full worlds stage the composed evolution and simulation RNG together.
        evolution = getattr(self, "evolution", None)
        if evolution is not None:
            shadow_state: object = evolution.begin_transaction(self.rng)
            shadow_controller = shadow_state.coordinator.brain_controller
            shadow_rng = shadow_state.simulation_rng
        else:
            # Compatibility path supports focused tests with lightweight fakes.
            shadow_factory = getattr(
                self.neat_controller,
                "transaction_shadow",
                None,
            )
            shadow_controller = (
                shadow_factory()
                if callable(shadow_factory)
                else copy.deepcopy(self.neat_controller)
            )
            shadow_rng = Random()
            shadow_rng.setstate(self.rng.getstate())
            shadow_state = shadow_controller
        live_rng = self.rng
        staged: list[StagedOffspring] = []
        first_child_id = self._next_creature_id_value
        try:
            self.rng = shadow_rng
            for offset, request in enumerate(requests):
                parent = request.parent
                child_id = first_child_id + offset
                traits = self._mutated_child_traits(parent)
                position = self._child_spawn_position(
                    parent,
                    traits.physical_traits.radius,
                )
                if isinstance(shadow_state, EvolutionTransaction):
                    plan = shadow_state.coordinator.finalize_child(
                        parent,
                        child_id,
                        CreatureGenotype(
                            traits.vision,
                            traits.physical_traits,
                            traits.flocking_traits,
                            traits.color,
                        ),
                        traits.lineage,
                        shadow_rng,
                    )
                    if plan is None:
                        raise RuntimeError(
                            "Final reproduction staging lost a parent brain."
                        )
                    speciation = plan.speciation_result
                    traits.color = plan.genotype.color
                    traits.lineage = plan.lineage
                else:
                    child_brain, speciation = shadow_controller.create_child_brain(
                        parent.creature_id,
                        child_id,
                        parent.lineage.species_id,
                        traits.physical_traits,
                        traits.vision,
                        traits.flocking_traits,
                    )
                    if child_brain is None or speciation is None:
                        raise RuntimeError(
                            "Final reproduction staging lost a parent brain."
                        )
                    traits.lineage.species_id = speciation.species_id
                    if speciation.is_new_species:
                        traits.color = self.genotype_manager.new_species_color(
                            parent.color,
                            self.rng,
                        )
                staged.append(
                    StagedOffspring(
                        request=request,
                        child_id=child_id,
                        traits=traits,
                        position=position,
                        speciation_result=speciation,
                    )
                )
        finally:
            self.rng = live_rng
        return staged, shadow_state, shadow_rng.getstate()

    def _commit_staged_reproductions(
        self,
        staged: list[StagedOffspring],
        shadow_controller: object | None,
        staged_rng_state: object | None,
    ) -> None:
        """Commit an accepted reproduction batch and materialize its creatures.

        Parameters
        ----------
        staged
            Successfully planned offspring in deterministic insertion order.
        shadow_controller
            Composed transaction or legacy controller shadow used for staging.
        staged_rng_state
            Legacy staged simulation RNG state retained for test compatibility.

        Returns
        -------
        None
            Evolution, RNG, physics, lifecycle, and telemetry state are committed.
        """
        # A partial or missing transaction is never observable by the live world.
        if not staged or shadow_controller is None:
            return
        # Commit the composed shadow atomically when the full service is present.
        if isinstance(shadow_controller, EvolutionTransaction):
            self.evolution.commit_transaction(shadow_controller, self.rng)
            self.species_manager = self.brain_controller.species_manager
        else:
            controller = self.neat_controller
            for name in (
                "config",
                "population",
                "brains",
                "species_manager",
                "_next_genome_id_value",
                "_next_brain_revision_value",
                "_evolution_rng",
                "_pairwise_compatibility_distance_cache",
            ):
                if hasattr(shadow_controller, name):
                    setattr(controller, name, getattr(shadow_controller, name))
            if staged_rng_state is not None:
                self.rng.setstate(staged_rng_state)

        for offspring in staged:
            parent = offspring.request.parent
            traits = offspring.traits
            child = self._spawn_creature(
                offspring.child_id,
                position=offspring.position,
                heading=parent.heading,
                energy=self.config.population.infant_energy_spawn,
                color=traits.color,
                vision=traits.vision,
                physical_traits=traits.physical_traits,
                flocking_traits=traits.flocking_traits,
                lineage=traits.lineage,
            )
            self.creatures.append(child)
            self._register_living_creature(child)
            self._initialize_creature_runtime_state(child)
            self._mark_behavior_cohort_dirty()
            self.fitness[child.creature_id] = CreatureFitness()
            self._chronometers[child.creature_id] = 0.0
            self._log_creature_birth(child)
            if offspring.speciation_result.is_new_species:
                self._record_new_species(
                    child,
                    offspring.speciation_result,
                )
            parent_fitness = self.fitness.get(parent.creature_id)
            if parent_fitness is not None:
                parent_fitness.record_reproduction()
            self.rt_neat.record_normal_replacement()
        self._next_creature_id_value = staged[-1].child_id + 1
        self.lifecycle.synchronize_allocator(self._next_creature_id_value)

    def _capture_mouth_exposure_rollback_state(
        self,
    ) -> _MouthExposureRollbackState:
        buffer = getattr(self, "_mouth_exposures", None)
        if buffer is None or buffer.count <= 0:
            return _MouthExposureRollbackState([], [], {}, {})
        creature_ids = {
            buffer.creature_ids[index] for index in range(buffer.count)
        }
        food_ids = {buffer.food_ids[index] for index in range(buffer.count)}
        return _MouthExposureRollbackState(
            creature_states=[
                (
                    creature,
                    float(getattr(creature, "stomach_energy", 0.0)),
                    float(
                        getattr(creature, "stomach_difficulty_load", 0.0)
                    ),
                )
                for creature in self.creatures
                if creature.creature_id in creature_ids
            ],
            food_states=[
                (food, index, float(food.energy_value))
                for index, food in enumerate(self.foods)
                if food.id in food_ids
            ],
            held_foods=dict(
                getattr(self, "_held_food_by_creature_id", {})
            ),
            food_carriers=dict(getattr(self, "_carrier_by_food_id", {})),
        )

    def _restore_mouth_exposure_rollback_state(
        self,
        state: _MouthExposureRollbackState,
    ) -> None:
        for creature, stomach_energy, difficulty_load in state.creature_states:
            creature.stomach_energy = stomach_energy
            try:
                creature.stomach_difficulty_load = difficulty_load
            except AttributeError:
                pass

        missing_foods: list[tuple[int, object]] = []
        for food, original_index, energy_value in state.food_states:
            food.energy_value = energy_value
            resize = getattr(food, "_resize_for_remaining_energy", None)
            if callable(resize):
                resize()
            if food not in self.foods:
                missing_foods.append((original_index, food))
        for original_index, food in sorted(missing_foods):
            self.foods.insert(min(original_index, len(self.foods)), food)
            body = getattr(food, "body", None)
            shape = getattr(food, "shape", None)
            space = getattr(self, "space", None)
            if (
                space is not None
                and body is not None
                and shape is not None
                and getattr(body, "space", None) is None
            ):
                space.add(body, shape)
            index_food = getattr(self, "_index_food", None)
            if callable(index_food):
                index_food(food)

        reindex_shape = getattr(getattr(self, "space", None), "reindex_shape", None)
        if callable(reindex_shape):
            for food, _original_index, _energy_value in state.food_states:
                if food in self.foods and hasattr(food, "shape"):
                    reindex_shape(food.shape)
        if hasattr(self, "_held_food_by_creature_id"):
            self._held_food_by_creature_id = dict(state.held_foods)
        if hasattr(self, "_carrier_by_food_id"):
            self._carrier_by_food_id = dict(state.food_carriers)

    def _resolve_accumulated_mouth_exposures(
        self,
        *,
        clear_on_success: bool = True,
        rollback_state: _MouthExposureRollbackState | None = None,
    ) -> MetabolismReport:
        """Resolve chronological fixed-step contacts before digestion."""
        buffer = getattr(self, "_mouth_exposures", None)
        if buffer is None:
            return MetabolismReport()
        if buffer.count <= 0:
            return MetabolismReport()

        rollback = rollback_state or self._capture_mouth_exposure_rollback_state()

        creatures_by_id = {
            creature.creature_id: creature for creature in self.creatures
        }
        foods_by_id = {food.id: food for food in self.foods}
        touched_foods: list[Food] = []
        touched_ids: set[int] = set()
        depleted_foods: list[Food] = []
        depleted_ids: set[int] = set()
        claimed_food_step: set[tuple[int, int]] = set()
        claimed_creature_step: set[tuple[int, int]] = set()
        consumptions: list[FoodConsumption] = []
        try:
            self._scheduler_validation_failure_point(
                "exposure.before_validation"
            )
            buffer.sort_order()
            for order_index in range(buffer.count):
                record_index = buffer.order[order_index]
                physics_step = buffer.steps[record_index]
                creature_id = buffer.creature_ids[record_index]
                creature = creatures_by_id.get(creature_id)
                food_id = buffer.food_ids[record_index]
                food = foods_by_id.get(food_id)
                food_step = (physics_step, food_id)
                creature_step = (physics_step, creature_id)
                if (
                    creature is None
                    or creature.life <= 0.0
                    or food is None
                    or food_id in depleted_ids
                    or food_step in claimed_food_step
                    or creature_step in claimed_creature_step
                ):
                    continue
                self._scheduler_validation_failure_point(
                    "exposure.before_mutation"
                )
                consumption = self.metabolism.eat(
                    creature,
                    food,
                    buffer.durations[record_index],
                )
                self._scheduler_validation_failure_point(
                    "exposure.after_stomach_mutation"
                )
                self._scheduler_validation_failure_point(
                    "exposure.after_food_mutation"
                )
                if (
                    consumption.energy_swallowed <= 0.0
                    and not consumption.depleted
                ):
                    continue
                claimed_food_step.add(food_step)
                claimed_creature_step.add(creature_step)
                if food_id not in touched_ids:
                    touched_ids.add(food_id)
                    touched_foods.append(food)
                if consumption.depleted:
                    depleted_ids.add(food_id)
                    depleted_foods.append(food)
                    clear_carry = getattr(self, "_clear_food_carry", None)
                    if callable(clear_carry):
                        clear_carry(food)
                    self._scheduler_validation_failure_point(
                        "exposure.after_carried_food_mutation"
                    )
                consumptions.append(
                    FoodConsumption(
                        creature_id=creature.creature_id,
                        food=food,
                        energy_swallowed=consumption.energy_swallowed,
                        depleted=consumption.depleted,
                    )
                )
                self._scheduler_validation_failure_point(
                    "exposure.after_valid_claim"
                )
            if clear_on_success:
                self._scheduler_validation_failure_point(
                    "exposure.before_buffer_clear"
                )
                buffer.clear()
        except BaseException:
            self._restore_mouth_exposure_rollback_state(rollback)
            raise

        return MetabolismReport(
            depleted_foods=depleted_foods,
            touched_foods=touched_foods,
            food_consumptions=consumptions,
        )

    def _update_metabolism(self, delta_time: float) -> None:
        if not hasattr(self.metabolism, "evaluate_candidate"):
            self._update_metabolism_legacy_adapter(delta_time)
            return
        # Contacts represent bites accumulated since the previous biology
        # boundary. Fill stomachs before evaluating the batched digestion and
        # resource ledger so prior fixed-step bites are not stale.
        rollback = self._capture_mouth_exposure_rollback_state()
        try:
            eating_report = self._resolve_accumulated_mouth_exposures(
                clear_on_success=False,
                rollback_state=rollback,
            )
            self._complete_metabolism_update(delta_time, eating_report)
        except BaseException:
            self._restore_mouth_exposure_rollback_state(rollback)
            raise
        else:
            exposure_buffer = getattr(self, "_mouth_exposures", None)
            if exposure_buffer is not None:
                exposure_buffer.clear()

    def _complete_metabolism_update(
        self,
        delta_time: float,
        eating_report: MetabolismReport,
    ) -> None:
        """Commit resource transactions, digestion, deaths, and staged births.

        Parameters
        ----------
        delta_time
            Biological update duration in seconds.
        eating_report
            Resolved mouth-contact consumption preceding resource evaluation.

        Returns
        -------
        None
            Creature ledgers, resources, lifecycle, and diagnostics are committed.
        """
        # Resolve all candidates before staging evolution to retain atomicity.
        resolution = self._resolve_resource_transactions(delta_time)
        staged_offspring, shadow_controller, staged_rng_state = (
            self._stage_final_reproductions(resolution.reproductions)
        )

        reproduction_ids = {
            request.parent.creature_id
            for request in resolution.reproductions
        }
        nursing_donor_ids = {
            transfer.request.donor.creature_id
            for transfer in resolution.nursing_transfers
            if transfer.allocated_transfer > 0.0
        }
        action_creature_ids = reproduction_ids | nursing_donor_ids
        selected_creature_id = getattr(self, "selected_creature_id", None)
        processing_costs: dict[int, float] = {}
        dead_creatures: list[Creature] = []

        self._scheduler_validation_failure_point("biology.commit_boundary")
        for creature in list(self.creatures):
            creature_id = creature.creature_id
            candidate = resolution.candidates[creature.creature_id]
            status = (
                "action_committed"
                if creature_id in action_creature_ids
                else "baseline_committed"
            )
            record_diagnostics = creature_id == selected_creature_id
            self.metabolism.commit_candidate(
                creature,
                candidate,
                transaction_status=status,
                record_diagnostics=record_diagnostics,
            )
            self._commit_activity_diagnostics(
                creature,
                resolution.activities[creature_id],
                record_diagnostics=record_diagnostics,
            )
            self._scheduler_validation_failure_point(
                "biology.after_creature_commit"
            )
            if candidate.digestion.processing_cost > 0.0:
                processing_costs[creature_id] = (
                    candidate.digestion.processing_cost
                )
            if creature.life <= 0.0:
                dead_creatures.append(creature)

        for transfer in resolution.nursing_transfers:
            target = transfer.request.target
            if target.life <= 0.0:
                continue
            target.energy = min(
                self.config.metabolism.max_energy,
                target.energy + transfer.allocated_transfer,
            )

        if staged_offspring:
            self._commit_staged_reproductions(
                staged_offspring,
                shadow_controller,
                staged_rng_state,
            )

        self._scheduler_validation_failure_point(
            "biology.post_death_processing"
        )
        self._remove_dead_creatures(
            dead_creatures,
            default_reason="metabolic",
        )

        self._scheduler_validation_failure_point(
            "biology.dependent_fitness_bookkeeping"
        )
        for consumption in eating_report.food_consumptions:
            self._record_behavior_food_consumption(consumption)
            fitness = self.fitness.get(consumption.creature_id)
            if fitness is not None:
                fitness.record_food(depleted=consumption.depleted)

        self._last_digestion_processing_costs_per_second = {
            creature_id: (
                cost / delta_time if delta_time > 0.0 else 0.0
            )
            for creature_id, cost in processing_costs.items()
        }
        self.resources.last_digestion_processing_costs_per_second = (
            self._last_digestion_processing_costs_per_second
        )
        for food in eating_report.touched_foods:
            reindex_shape = getattr(self.space, "reindex_shape", None)
            if reindex_shape is not None and food in self.foods:
                reindex_shape(food.shape)

        for food in eating_report.depleted_foods:
            if food in self.foods:
                self._clear_food_carry(food)
                self.foods.remove(food)
                self._unindex_food(food)
                self.space.remove(food.body, food.shape)

        if not self.creatures:
            self._recover_extinct_population()

        if self.selected_creature_id is not None and self.selected_creature is None:
            self.selected_creature_id = None
            self._reset_behavior_focus(None)

        self._log_parent_selection_attempts(resolution.reproduction_attempts)

    def _update_metabolism_legacy_adapter(self, delta_time: float) -> None:
        """Run the historical monolithic metabolism extension interface.

        Parameters
        ----------
        delta_time
            Fixed-step biological duration in seconds.

        Returns
        -------
        None
            Resource, food, death, and diagnostic state are updated in place.

        Notes
        -----
        Infant movement penalties are runtime inputs and never mutate genotype.
        """
        # Nursing remains the first resource transition in this compatibility path.
        self._apply_nursing(delta_time)
        # Runtime overrides preserve immutable hereditary movement traits.
        movement_cost_multipliers: dict[int, float] = {}
        for creature in self.creatures:
            physical_traits = getattr(creature, "physical_traits", None)
            inherited_cost = getattr(
                physical_traits,
                "movement_cost_multiplier",
                None,
            )
            if inherited_cost is not None:
                movement_cost_multipliers[creature.creature_id] = (
                    inherited_cost
                    * (3.0 if self._is_infant(creature) else 1.0)
                )
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
                if (
                    action := getattr(self, "_effective_actions", {}).get(
                        creature.creature_id,
                        self._last_actions.get(creature.creature_id),
                    )
                ) is not None
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
            movement_cost_multipliers=movement_cost_multipliers,
        )

        for consumption in report.food_consumptions:
            self._record_behavior_food_consumption(consumption)
            fitness = self.fitness.get(consumption.creature_id)
            if fitness is not None:
                fitness.record_food(depleted=consumption.depleted)
        processing_costs = getattr(report, "digestion_processing_costs", {})
        self._last_digestion_processing_costs_per_second = {
            creature_id: cost / delta_time if delta_time > 0.0 else 0.0
            for creature_id, cost in processing_costs.items()
        }
        resources = getattr(self, "resources", None)
        if resources is not None:
            resources.last_digestion_processing_costs_per_second = (
                self._last_digestion_processing_costs_per_second
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
            death_reason = (
                "old_age" if self._is_senescent(creature) else "starvation"
            )
            self._remove_creature(creature, death_reason=death_reason)
        if not self.creatures:
            self._recover_extinct_population()
        if self.selected_creature_id is not None and self.selected_creature is None:
            self.selected_creature_id = None
            self._reset_behavior_focus(None)

    def _record_behavior_food_consumption(self, consumption: object) -> None:
        creature_id = getattr(consumption, "creature_id", None)
        active_subjects = getattr(self, "_behavior_active_subjects", None)
        if (
            active_subjects is not None
            and creature_id not in active_subjects
        ) or (
            active_subjects is None
            and creature_id != self.selected_creature_id
        ):
            return
        totals = getattr(self, "_behavior_consumption_totals", {})
        count, energy = totals.get(
            creature_id,
            (
                getattr(self, "_behavior_food_consumption_count", 0)
                if creature_id == self.selected_creature_id
                else 0,
                getattr(self, "_behavior_food_consumed_energy_total", 0.0)
                if creature_id == self.selected_creature_id
                else 0.0,
            ),
        )
        swallowed = max(
            0.0,
            float(getattr(consumption, "energy_swallowed", 0.0)),
        )
        totals[creature_id] = (
            count + 1,
            energy + swallowed,
        )
        self._behavior_consumption_totals = totals
        if creature_id == self.selected_creature_id:
            self._behavior_food_consumption_count = count + 1
            self._behavior_food_consumed_energy_total = energy + swallowed

    def _communication_intensities_for(
        self,
        creature_id: int,
    ) -> tuple[float, float, float]:
        action = self._action_for_execution(creature_id)
        if action is None:
            return (0.0, 0.0, 0.0)
        return (
            max(0.0, min(1.0, action.emit_sound)),
            max(0.0, min(1.0, action.emit_trail_pheromone)),
            max(0.0, min(1.0, action.emit_alarm_pheromone)),
        )

    def _activity_for(
        self,
        creature: Creature,
        *,
        reproduction_selected: bool = False,
        nursing_transfer: float = 0.0,
    ) -> ActivityResult:
        """Calculate weighted activity without mutating creature state."""
        sound, trail, alarm = self._communication_intensities_for(
            creature.creature_id
        )
        communication_config = self.config.communication
        acoustic_rate = max(
            0.0,
            communication_config.acoustic_energy_cost_per_second,
        )
        pheromone_rate = max(
            0.0,
            communication_config.pheromone_energy_cost_per_second,
        )
        maximum_communication_cost = acoustic_rate + 2.0 * pheromone_rate
        communication_cost = (
            0.0
            if maximum_communication_cost <= 0.0
            else (
                sound * acoustic_rate
                + (trail + alarm) * pheromone_rate
            )
            / maximum_communication_cost
        )
        command = getattr(
            getattr(self, "_motion_commands", {}).get(creature.creature_id),
            "effective_rotate",
            0.0,
        )
        return calculate_weighted_activity(
            voluntary_motor_effort=getattr(
                creature,
                "effective_voluntary_motor_effort",
                0.0,
            ),
            normalized_speed=(
                0.0
                if self.MAX_SPEED <= 0.0
                else creature.speed / self.MAX_SPEED
            ),
            turn_command=command,
            normalized_angular_speed=(
                creature.body.angular_velocity / self.MAX_ANGULAR_SPEED
                if self.MAX_ANGULAR_SPEED > 0.0
                else 0.0
            ),
            communication_cost=communication_cost,
            reproduction_selected=reproduction_selected,
            nursing_transfer=nursing_transfer,
        )

    @staticmethod
    def _commit_activity_diagnostics(
        creature: Creature,
        activity: ActivityResult,
        *,
        record_diagnostics: bool = True,
    ) -> None:
        """Update the creature's reusable activity diagnostics in place."""
        creature.activity = activity.total
        creature.effective_rest = creature.smoothed_rest * (1.0 - activity.total)
        if not record_diagnostics:
            return
        diagnostics = creature.ledger_diagnostics.activity
        diagnostics.voluntary_motor_effort = activity.voluntary_motor_effort
        diagnostics.normalized_speed = activity.normalized_speed
        diagnostics.turn = activity.turn
        diagnostics.communication = activity.communication
        diagnostics.reproduction = activity.reproduction
        diagnostics.nursing = activity.nursing
        diagnostics.weighted_total = activity.total

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
        return list(self._own_infant_view_for(parent))

    def _own_infant_view_for(self, parent: Creature):
        index = getattr(self, "_creature_spatial_index", None)
        if index is not None and index.valid:
            return index.family_view(parent.creature_id, self._is_infant)
        # Current-position debug/load fallback; normal fixed steps always have
        # a complete family generation before sensing or nursing.
        living = getattr(self, "_living_creatures", None)
        source = self.creatures if living is None else living.values()
        return tuple(
            creature
            for creature in sorted(
                source,
                key=lambda other: other.creature_id,
            )
            if getattr(getattr(creature, "lineage", None), "parent_id", None)
            == parent.creature_id
            and self._is_infant(creature)
        )

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
            action = getattr(self, "_effective_actions", {}).get(
                parent.creature_id,
                self._last_actions.get(parent.creature_id),
            )
            if action is None or not is_active_intent(action.want_nurse):
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
        parent_x, parent_y = parent.position
        nearest: Creature | None = None
        nearest_key: tuple[float, int] | None = None
        for infant in self._own_infant_view_for(parent):
            dx = infant.position[0] - parent_x
            dy = infant.position[1] - parent_y
            distance_squared = dx * dx + dy * dy
            if distance_squared <= max_distance_squared:
                key = (distance_squared, infant.creature_id)
                if nearest_key is None or key < nearest_key:
                    nearest = infant
                    nearest_key = key
        return nearest

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

    def _physics_rate_alpha(self, reference_alpha: float, delta_time: float) -> float:
        """Scale a historical 60 Hz interpolation coefficient by elapsed time."""
        alpha = self._clamp(reference_alpha, 0.0, 1.0)
        elapsed = max(0.0, delta_time)
        return 1.0 - (1.0 - alpha) ** (elapsed / self.FIXED_TIMESTEP)

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
        was_selected = self.selected_creature_id == creature.creature_id
        if was_selected:
            self._finalize_behavior_focus(BehaviorTermination.CREATURE_DIED)
        elif creature.creature_id in getattr(
            self,
            "_behavior_active_subjects",
            {},
        ):
            observer = getattr(self, "behavior_observer", None)
            finalize_subject = getattr(observer, "finalize_subject", None)
            if callable(finalize_subject):
                finalize_subject(
                    (
                        creature.creature_id,
                        self._behavior_active_subjects[creature.creature_id],
                    ),
                    BehaviorTermination.CREATURE_DIED,
                )
        history = getattr(self, "behavior_history", None)
        if history is not None:
            history.mark_deceased(creature.creature_id, self.elapsed_time)
        getattr(self, "_behavior_active_subjects", {}).pop(
            creature.creature_id,
            None,
        )
        getattr(self, "_behavior_consumption_totals", {}).pop(
            creature.creature_id,
            None,
        )
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
                creature.creature_id, fitness.score(creature)
            )

        if creature in self.creatures:
            # Slots become invalid through the registry before any remaining
            # removal side effects can expose the entity to a consumer.
            self._unregister_living_creature(creature)
            self.creatures.remove(creature)
            self._mark_behavior_cohort_dirty()
            self._unindex_creature_shape(creature)
            self.space.remove(creature.body, creature.shape)
            self.neat_controller.remove_brain(creature.creature_id)
            social_compatibility = getattr(self, "social_compatibility", None)
            if social_compatibility is not None:
                social_compatibility.discard_creature(creature.creature_id)
            self._last_actions.pop(creature.creature_id, None)
            effective_actions = getattr(self, "_effective_actions", None)
            if effective_actions is not None:
                effective_actions.pop(creature.creature_id, None)
            last_snapshots = getattr(self, "_last_sensor_snapshots", None)
            if last_snapshots is not None:
                last_snapshots.pop(creature.creature_id, None)
            acoustic_debug = getattr(self, "_last_acoustic_debug", None)
            if acoustic_debug is not None:
                acoustic_debug.pop(creature.creature_id, None)
            flock_debug = getattr(self, "_last_flock_steering_debug", None)
            if flock_debug is not None:
                flock_debug.pop(creature.creature_id, None)
            flock_runtime = getattr(self, "_last_flocking_runtime", None)
            if flock_runtime is not None:
                flock_runtime.pop(creature.creature_id, None)
            cached_social = getattr(self, "_cached_social_intentions", None)
            if cached_social is not None:
                cached_social.pop(creature.creature_id, None)
            benchmark_qualities = getattr(
                self,
                "_flocking_benchmark_quality_by_creature_id",
                None,
            )
            if benchmark_qualities is not None:
                benchmark_qualities.pop(creature.creature_id, None)
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

        if was_selected:
            self.selected_creature_id = None
            self._reset_behavior_focus(None)

        self._chronometers.pop(creature.creature_id, None)
        self._prune_historical_archives()

    def _prune_historical_archives(self) -> None:
        """Prune aligned neural, genotype, species, and fitness archives.

        Parameters
        ----------
        None
            This method receives no external parameters.

        Returns
        -------
        None
            Historical archives are reduced to configured retention limits.
        """
        # Genotype and neural archives share genome IDs and must be pruned together.
        population_config = self.config.population
        trait_archive = getattr(self, "_trait_archive_by_genome_id", {})
        active_species_ids = {
            creature.lineage.species_id for creature in self.creatures
        }
        evolution = getattr(self, "evolution", None)
        if evolution is not None:
            self._trait_archive_by_genome_id = evolution.prune_archives(
                trait_archive,
                active_species_ids,
                population_config.elite_archive_size,
            )
        else:
            # Focused legacy fixtures may provide only controller-level methods.
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
            retained_species_ids = set(active_species_ids)
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
        """Archive one creature genotype under its neural genome identity.

        Parameters
        ----------
        creature
            Live creature leaving the population.

        Returns
        -------
        None
            A detached archive entry is stored when a genome exists.
        """
        # Neural identity is the integration key used by extinction recovery.
        genome_id_for = getattr(self.neat_controller, "genome_id_for", None)
        if genome_id_for is None:
            return
        genome_id = genome_id_for(creature.creature_id)
        if genome_id is None:
            return
        if not hasattr(self, "_trait_archive_by_genome_id"):
            self._trait_archive_by_genome_id = {}
        self._trait_archive_by_genome_id[genome_id] = (
            ArchivedCreatureTraits.from_creature(creature)
        )

    def _initial_total_biomass_energy(self) -> float:
        configured_total = self.config.food.total_biomass_energy
        if configured_total is not None:
            return configured_total
        return self._creature_energy() + self._plant_energy()

    @property
    def live_food_config(self) -> LiveFoodConfig:
        """Return the immutable snapshot currently driving food spawning."""
        return self._live_food_config

    def set_live_food_config_value(
        self,
        name: str,
        value: int | float,
    ) -> LiveFoodConfig:
        """Update one live food value and return the normalized snapshot."""
        if name not in self._live_food_config.to_primitive():
            raise KeyError(f"Unknown live food configuration field {name!r}.")

        current = self._live_food_config
        if name in ("max_food_items", "low_food_burst_items"):
            normalized: int | float = max(0, int(round(float(value))))
        else:
            normalized = float(value)

        if name == "critical_food_ratio":
            normalized = min(
                max(0.0, normalized),
                current.low_food_pressure_threshold,
            )
        elif name == "low_food_pressure_threshold":
            normalized = max(
                current.critical_food_ratio,
                min(1.0, normalized),
            )

        updated = replace(current, **{name: normalized})
        self.apply_live_food_config(updated)
        return self._live_food_config

    def apply_live_food_config(self, settings: LiveFoodConfig) -> None:
        """Atomically apply a complete live food configuration snapshot."""
        lock = getattr(self, "_checkpoint_state_lock", None)
        if lock is None:
            self._apply_live_food_config_unlocked(settings)
            return
        with lock:
            self._apply_live_food_config_unlocked(settings)

    def _apply_live_food_config_unlocked(
        self,
        settings: LiveFoodConfig,
    ) -> None:
        current = self._live_food_config
        burst_changed = any(
            getattr(current, name) != getattr(settings, name)
            for name in (
                "low_food_pressure_threshold",
                "critical_food_ratio",
                "low_food_burst_items",
                "low_food_burst_interval",
            )
        )
        richness_changed = any(
            getattr(current, name) != getattr(settings, name)
            for name in (
                "forest_spawn_weight",
                "bushes_spawn_weight",
                "prairie_spawn_weight",
            )
        )

        food_config = self.food_spawner.config
        food_config.max_food_items = settings.max_food_items
        food_config.low_food_pressure_threshold = (
            settings.low_food_pressure_threshold
        )
        food_config.critical_food_ratio = settings.critical_food_ratio
        food_config.low_food_burst_items = settings.low_food_burst_items
        food_config.low_food_burst_interval = settings.low_food_burst_interval

        if richness_changed:
            self.biome_map = replace(
                self.biome_map,
                spawn_weights={
                    Biome.FOREST: settings.forest_spawn_weight,
                    Biome.BUSHES: settings.bushes_spawn_weight,
                    Biome.PRAIRIE: settings.prairie_spawn_weight,
                },
            )
            self.food_spawner.biome_map = self.biome_map
        if burst_changed:
            self.food_spawner.reset_low_food_burst_state()
        self._live_food_config = settings

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
        """Repopulate an empty world from retained neural and genotype archives.

        Parameters
        ----------
        None
            This method receives no external parameters.

        Returns
        -------
        None
            Recovery offspring are materialized up to configured capacity.
        """
        # Select archived neural parents before consuming any simulation RNG.
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
                else self.genotype_manager.initial_color(0)
            )
            child = self._spawn_creature(
                child_id,
                energy=self.config.metabolism.max_energy,
                color=(
                    child_traits.color
                    if child_traits is not None
                    else self.genotype_manager.initial_color(0)
                ),
                vision=child_traits.vision if child_traits is not None else None,
                physical_traits=(
                    child_traits.physical_traits if child_traits is not None else None
                ),
                flocking_traits=(
                    child_traits.flocking_traits if child_traits is not None else None
                ),
                lineage=child_traits.lineage if child_traits is not None else None,
            )
            evolution = getattr(self, "evolution", None)
            if evolution is not None:
                recovery_plan = evolution.finalize_from_genome(
                    parent_genome,
                    parent_color,
                    child_id,
                    child.genotype,
                    child.lineage,
                    self.rng,
                )
                child_brain = recovery_plan.brain
                speciation_result = recovery_plan.speciation_result
            else:
                child_brain, speciation_result = (
                    self.neat_controller.create_mutated_brain_from_genome(
                        parent_genome,
                        child_id,
                        parent_species_id,
                        child.physical_traits,
                        child.vision,
                        child.flocking_traits,
                    )
                )
            if child_brain is None:
                self._unindex_creature_shape(child)
                self.space.remove(child.body, child.shape)
                continue
            child.lineage.species_id = speciation_result.species_id
            if speciation_result.is_new_species and evolution is None:
                child.color = self.genotype_manager.new_species_color(parent_color, self.rng)
            if speciation_result.is_new_species:
                self._record_new_species(child, speciation_result)

            self.creatures.append(child)
            self._register_living_creature(child)
            self._initialize_creature_runtime_state(child)
            self._mark_behavior_cohort_dirty()
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
            parent_flocking_traits=archived_traits.flocking_traits,
            parent_color=archived_traits.color,
        )

    def _update_reproduction(self, delta_time: float) -> None:
        """Mark whether this fixed step may stage reproduction requests."""
        self._reproduction_accumulator += delta_time
        if self._reproduction_accumulator < self.REPRODUCTION_INTERVAL:
            self._reproduction_due_this_step = False
            return

        self._reproduction_accumulator %= self.REPRODUCTION_INTERVAL
        self._reproduction_due_this_step = True

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
        """Attempt one immediate reproduction outside the batched scheduler path.

        Parameters
        ----------
        None
            This method receives no external parameters.

        Returns
        -------
        bool
            Whether one offspring was fully evolved and registered.
        """
        # Capacity and resource guards run before allocating a stable identity.
        if len(self.creatures) >= self.config.population.max_creatures:
            return False

        if not self._has_reproduction_resources():
            return False

        eligible_pool = self._eligible_reproduction_parents()
        parent = self._select_reproduction_parent(eligible_pool)
        if parent is None:
            return False

        request = self._reproduction_request_for(
            parent,
            0,
            len(eligible_pool),
        )
        if not self._parent_is_reproduction_eligible(parent):
            self._log_parent_selection_attempts(
                [(request, "eligibility_rejected")]
            )
            return False

        reproduction_cost = request.reserved_energy_cost
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
            flocking_traits=child_traits.flocking_traits,
            lineage=child_traits.lineage,
        )

        evolution = getattr(self, "evolution", None)
        if evolution is not None:
            offspring_plan = evolution.finalize_child(
                parent,
                child_id,
                child.genotype,
                child.lineage,
                self.rng,
            )
            child_brain = None if offspring_plan is None else offspring_plan.brain
            speciation_result = (
                None
                if offspring_plan is None
                else offspring_plan.speciation_result
            )
        else:
            child_brain, speciation_result = self.neat_controller.create_child_brain(
                parent.creature_id,
                child_id,
                parent.lineage.species_id,
                child.physical_traits,
                child.vision,
                child.flocking_traits,
            )
        if child_brain is None or speciation_result is None:
            self._unindex_creature_shape(child)
            self.space.remove(child.body, child.shape)
            self._log_parent_selection_attempts(
                [(request, "resource_rejected")]
            )
            return False
        child.lineage.species_id = speciation_result.species_id
        if speciation_result.is_new_species and evolution is None:
            child.color = self.genotype_manager.new_species_color(parent.color, self.rng)
        if speciation_result.is_new_species:
            self._record_new_species(child, speciation_result)

        self.creatures.append(child)
        self._register_living_creature(child)
        self._initialize_creature_runtime_state(child)
        self._mark_behavior_cohort_dirty()
        self.fitness[child_id] = CreatureFitness()
        self._chronometers[child_id] = 0.0
        self._log_creature_birth(child)

        self._spend_reproduction_energy(parent, reproduction_cost)
        parent_fitness.record_reproduction()
        self.rt_neat.record_normal_replacement()
        self._log_parent_selection_attempts([(request, "committed")])
        return True

    def _log_parent_selection_attempts(
        self,
        attempts: list[tuple[ReproductionRequest, str]],
    ) -> None:
        telemetry = getattr(self, "telemetry", None)
        log_events = getattr(telemetry, "log_parent_selection_events", None)
        if not callable(log_events) or not attempts:
            return
        population_config = self.config.population
        rows: list[dict[str, object]] = []
        for request, outcome in attempts:
            parent = request.parent
            try:
                gathered = float(parent.total_energy_gathered)
            except (AttributeError, TypeError, ValueError, OverflowError):
                gathered = 0.0
            complexity = request.network_complexity
            rows.append(
                {
                    "sim_time": float(getattr(self, "elapsed_time", 0.0)),
                    "parent_creature_id": parent.creature_id,
                    "species_id": parent.lineage.species_id,
                    "total_energy_gathered": (
                        gathered if isfinite(gathered) else 0.0
                    ),
                    "node_count": request.node_count,
                    "enabled_connection_count": (
                        request.enabled_connection_count
                    ),
                    "network_complexity": (
                        complexity if isfinite(complexity) else None
                    ),
                    "eligible_pool_size": request.selection_pool_size,
                    "tournament_k1": population_config.tournament_k1,
                    "tournament_k2": population_config.tournament_k2,
                    "outcome": outcome,
                }
            )
        log_events(rows)

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
        eligible_pool = self._eligible_reproduction_parents()
        return self._select_reproduction_parent(eligible_pool)

    def _select_reproduction_parent(
        self,
        eligible_pool: list[Creature],
    ) -> Creature | None:
        if not eligible_pool:
            return None
        selector = getattr(self.rt_neat, "select_parent", None)
        population_config = getattr(
            getattr(self, "config", None),
            "population",
            None,
        )
        if not callable(selector) or population_config is None:
            return eligible_pool[0]
        return selector(
            eligible_pool,
            population_config.tournament_k1,
            population_config.tournament_k2,
        )

    def _creature_want_to_eat(self, creature: Creature) -> bool:
        action = self._last_actions.get(creature.creature_id)
        if action is None:
            return False
        physical_traits = getattr(creature, "physical_traits", None)
        inherited_capacity = getattr(
            physical_traits,
            "stomach_capacity",
            None,
        )
        stomach_capacity = (
            max(0.0, float(inherited_capacity))
            if inherited_capacity is not None
            else max(
                0.0,
                creature.radius
                * self.config.metabolism.stomach_capacity_per_radius,
            )
        )
        stomach_space = max(
            0.0,
            stomach_capacity - max(0.0, creature.stomach_energy),
        )
        return stomach_space > 0.0 and is_active_intent(action.want_eat)

    def _settle_food_motion(self) -> None:
        for food in self.foods:
            food.body.velocity *= 0.84
            food.body.angular_velocity *= 0.62

            if food.body.velocity.length < 0.75:
                food.body.velocity = (0.0, 0.0)

            if abs(food.body.angular_velocity) < 0.2:
                food.body.angular_velocity = 0.0

            self._reindex_food(food)
