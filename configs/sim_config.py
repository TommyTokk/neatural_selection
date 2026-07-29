from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, pi

Color = tuple[int, int, int] | tuple[int, int, int, int]


class PheromoneBoundaryMode(str, Enum):
    """Boundary behavior for diffusion and world-position mapping."""

    REFLECT = "reflect"
    WRAP = "wrap"
    ABSORB = "absorb"


class SocialCompatibilityMode(str, Enum):
    LEGACY = "legacy"
    SPECIES = "species"
    SOCIAL_TAG = "social_tag"


@dataclass(slots=True)
class LongRangeSocialConfig:
    enabled: bool = False
    range: float = 400.0
    strength: float = 1.0


@dataclass(slots=True)
class CohortSpawnConfig:
    enabled: bool = False
    size: int = 6
    radius: float = 150.0


@dataclass(slots=True)
class SocialCompatibilityConfig:
    #mode: SocialCompatibilityMode = SocialCompatibilityMode.LEGACY
    mode: SocialCompatibilityMode = SocialCompatibilityMode.SOCIAL_TAG
    social_tag_sigma: float = 0.35


@dataclass(slots=True)
class FlockingTelemetryConfig:
    interval_seconds: float = 1.0
    group_detection_range: float = 150.0
    minimum_group_compatibility: float = 0.5
    persistence_overlap_threshold: float = 0.5


@dataclass(slots=True)
class FlockingBenchmarkConfig:
    enabled: bool = False
    reward_rate: float = 0.01
    target_group_size: int = 4
    target_spacing: float = 60.0
    spacing_tolerance: float = 30.0
    reference_speed: float = 50.0
    max_per_evaluation: float = 1.0


@dataclass(slots=True)
class FlockingConfig:
    minimum_social_engagement: float = 0.25
    panic_suppression_strength: float = 0.5
    max_social_influence: float = 0.35
    target_group_size: int = 4
    preferred_personal_space: float = 60.0
    long_range: LongRangeSocialConfig = field(
        default_factory=LongRangeSocialConfig
    )
    cohort_spawn: CohortSpawnConfig = field(default_factory=CohortSpawnConfig)
    compatibility: SocialCompatibilityConfig = field(
        default_factory=SocialCompatibilityConfig
    )
    telemetry: FlockingTelemetryConfig = field(
        default_factory=FlockingTelemetryConfig
    )
    benchmark: FlockingBenchmarkConfig = field(
        default_factory=FlockingBenchmarkConfig
    )

    def __post_init__(self) -> None:
        try:
            self.compatibility.mode = SocialCompatibilityMode(
                self.compatibility.mode
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid social compatibility configuration: {error}"
            ) from error

        fractions = {
            "minimum_social_engagement": self.minimum_social_engagement,
            "panic_suppression_strength": self.panic_suppression_strength,
            "max_social_influence": self.max_social_influence,
            "long_range.strength": self.long_range.strength,
            "telemetry.minimum_group_compatibility": (
                self.telemetry.minimum_group_compatibility
            ),
            "telemetry.persistence_overlap_threshold": (
                self.telemetry.persistence_overlap_threshold
            ),
        }
        for name, value in fractions.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and within [0, 1].")

        positive = {
            "preferred_personal_space": self.preferred_personal_space,
            "long_range.range": self.long_range.range,
            "cohort_spawn.radius": self.cohort_spawn.radius,
            "compatibility.social_tag_sigma": (
                self.compatibility.social_tag_sigma
            ),
            "telemetry.interval_seconds": self.telemetry.interval_seconds,
            "telemetry.group_detection_range": (
                self.telemetry.group_detection_range
            ),
            "benchmark.spacing_tolerance": (
                self.benchmark.spacing_tolerance
            ),
            "benchmark.reference_speed": self.benchmark.reference_speed,
        }
        for name, value in positive.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive.")
        if self.telemetry.interval_seconds < 0.25:
            raise ValueError("flocking telemetry interval must be at least 0.25s.")
        for name, value in {
            "target_group_size": self.target_group_size,
            "cohort_spawn.size": self.cohort_spawn.size,
            "benchmark.target_group_size": self.benchmark.target_group_size,
        }.items():
            if type(value) is not int or value < 2:
                raise ValueError(f"{name} must be an integer of at least 2.")
        nonnegative = {
            "benchmark.reward_rate": self.benchmark.reward_rate,
            "benchmark.target_spacing": self.benchmark.target_spacing,
            "benchmark.max_per_evaluation": (
                self.benchmark.max_per_evaluation
            ),
        }
        for name, value in nonnegative.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and nonnegative.")

    def validate(self) -> None:
        """Revalidate values after command-line or UI mutation."""
        self.__post_init__()


@dataclass(slots=True)
class DisplayConfig:
    width: int = 1440
    height: int = 900
    title: str = "neat_game_of_life"
    resizable: bool = True
    gl_version: tuple[int, int] = (3, 3)
    antialiasing: bool = True
    vsync: bool = False


@dataclass(slots=True)
class LayoutConfig:
    outer_padding: int = 20
    panel_gap: int = 16
    top_bar_height: int = 62
    left_panel_width: int = 92
    min_environment_width: int = 420
    min_sidebar_width: int = 84
    panel_radius: int = 16
    card_radius: int = 8
    environment_radius: int = 0


@dataclass(slots=True)
class ThemeConfig:
    window_background: Color = (9, 13, 19)
    panel_background: Color = (251, 248, 255)
    panel_background_alt: Color = (222, 224, 238)
    panel_border: Color = (194, 199, 206)
    environment_background: Color = (9, 13, 19)
    environment_grid: Color = (54, 63, 76, 72)
    environment_border: Color = (9, 13, 19)
    environment_text: Color = (240, 239, 255)
    environment_text_muted: Color = (132, 139, 152)
    accent: Color = (66, 97, 125)
    accent_soft: Color = (167, 199, 231)
    text_primary: Color = (22, 26, 50)
    text_muted: Color = (66, 71, 77)
    herbivore_fill: Color = (142, 203, 161)
    herbivore_outline: Color = (44, 88, 67)
    selected_outline: Color = (186, 26, 26)
    food_fill: Color = (142, 219, 114)
    vision_fill: Color = (142, 203, 161, 42)
    card_background: Color = (244, 242, 255)


@dataclass(slots=True)
class DebugConfig:
    vision_toggle_label: str = "V"
    show_debug_vision_by_default: bool = False


@dataclass(slots=True)
class PersistenceConfig:
    simulation_root_directory: str = "saves"
    quick_save_interval_seconds: float = 120.0
    archive_save_interval_seconds: float = 3600.0
    enable_telemetry: bool = True


@dataclass(slots=True)
class EnvironmentConfig:
    world_width: float = 3200.0
    world_height: float = 2200.0


@dataclass(slots=True)
class BiomeConfig:
    seed: int = 42
    grid_width: int = 64
    grid_height: int = 44
    noise_scale: float = 800.0
    octaves: int = 3
    persistence: float = 0.5
    lacunarity: float = 2.0

    forest_target_share: float = 0.25
    bushes_target_share: float = 0.40
    prairie_target_share: float = 0.35

    forest_spawn_weight: float = 2.75
    bushes_spawn_weight: float = 1.25
    prairie_spawn_weight: float = 0.25
    uniform_spawn_chance: float = 0.10
    max_spawn_attempts: int = 32

    forest_color: Color = (20, 58, 43, 255)
    bushes_color: Color = (43, 74, 49, 255)
    prairie_color: Color = (75, 80, 47, 255)


@dataclass(slots=True)
class BiomeSensorConfig:
    forward_distance: float = 96.0
    side_offset: float = 48.0
    trend_time_constant_seconds: float = 3.0


@dataclass(slots=True)
class CommunicationConfig:
    acoustic_range: float = 480.0
    acoustic_min_emission_strength: float = 0.05
    acoustic_hearing_threshold: float = 0.05
    acoustic_energy_cost_per_second: float = 0.006

    pheromone_update_interval: float = 0.25
    # World distance squared per simulated second. This preserves the previous
    # visual rate on the default 64 x 44 grid while remaining resolution-aware.
    pheromone_diffusion_coefficient: float = 390.0
    pheromone_evaporation_rate: float = 0.08
    pheromone_max_concentration: float = 1.0
    pheromone_deposit_rate: float = 0.75
    pheromone_energy_cost_per_second: float = 0.002
    pheromone_max_updates_per_tick: int = 4
    pheromone_boundary_mode: PheromoneBoundaryMode = PheromoneBoundaryMode.REFLECT

    def __post_init__(self) -> None:
        finite_nonnegative = (
            "acoustic_range",
            "acoustic_energy_cost_per_second",
            "pheromone_diffusion_coefficient",
            "pheromone_evaporation_rate",
            "pheromone_max_concentration",
            "pheromone_deposit_rate",
            "pheromone_energy_cost_per_second",
        )
        for name in finite_nonnegative:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number, got {value!r}.")
            if not isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and nonnegative, got {value!r}."
                )

        for name in (
            "acoustic_min_emission_strength",
            "acoustic_hearing_threshold",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(
                    f"{name} must be finite and within [0, 1], got {value!r}."
                )

        interval = self.pheromone_update_interval
        if (
            isinstance(interval, bool)
            or not isinstance(interval, (int, float))
            or not isfinite(interval)
            or interval <= 0.0
        ):
            raise ValueError(
                "pheromone_update_interval must be finite and positive, "
                f"got {interval!r}."
            )

        update_cap = self.pheromone_max_updates_per_tick
        if type(update_cap) is not int or update_cap < 1:
            raise ValueError(
                "pheromone_max_updates_per_tick must be a positive integer, "
                f"got {update_cap!r}."
            )

        try:
            self.pheromone_boundary_mode = PheromoneBoundaryMode(
                self.pheromone_boundary_mode
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "pheromone_boundary_mode must be one of "
                f"{[mode.value for mode in PheromoneBoundaryMode]}, got "
                f"{self.pheromone_boundary_mode!r}."
            ) from error


@dataclass(slots=True)
class ZoomConfig:
    default: float = 1.0
    minimum: float = 0.3
    maximum: float = 3.0
    step: float = 0.12


@dataclass(slots=True)
class MetabolismConfig:
    max_energy: float = 1
    basic_metabolism_rate: float = 0.01
    brain_upkeep_per_node: float = 0.0003
    brain_upkeep_per_connection: float = 0.0001
    movement_energy_cost_factor: float = 0.02
    sprint_energy_cost_per_second: float = 0.04
    eating_distance: float = 8
    micro_food_remainder_ratio: float = 0.10
    stomach_capacity_per_radius: float = 0.1
    digestion_rate_per_second: float = 0.2
    digestion_efficiency: float = 0.9
    max_bite_size_per_second: float = 0.5
    starvation_energy_threshold: float = 0.3


@dataclass(slots=True)
class PopulationConfig:
    initial_creatures: int = 40
    max_creatures: int = 55
    elite_archive_size: int = 256
    fitness_archive_size: int = 256
    extinction_recovery_creatures: int = 35
    extinction_recovery_parent_pool: int = 5
    senescence_age_seconds: float = 200.0
    senescence_cost_multiplier: float = 0.05
    min_reproduction_age: float = 20.0
    reproduction_cooldown: float = 12.0
    reproduction_energy_threshold: float = 0.8
    reproduction_energy_cost_base: float = 0.35
    reproduction_cost_per_node: float = 0.008
    reproduction_cost_per_connection: float = 0.002
    max_dynamic_reproduction_cost: float = 0.75
    infant_energy_spawn: float = 0.15
    infant_maturity_age: float = 12.0
    nursing_energy_transfer_rate: float = 0.05
    child_spawn_distance: float = 34.0
    reproduction_min_food_ratio: float = 0.2
    reproduction_min_available_biomass_ratio: float = 0.02
    reproduction_recovery_pressure_threshold: float = 0.25


@dataclass(slots=True)
class SpeciationConfig:
    compatibility_threshold: float = 3.5
    phenotypic_weight: float = 2.0
    flocking_trait_distance_coefficient: float = 1.0
    target_species_count: int = 5
    min_threshold: float = 2.0
    max_threshold: float = 7.0
    threshold_adjust_rate: float = 0.05
    adjustment_interval_seconds: float = 5.0


@dataclass(slots=True)
class FitnessConfig:
    age_weight: float = 0.1  # Increased to reward baseline survival
    food_discovery_weight: float = 0.5  # Increased to encourage exploration
    food_discovery_cap: int = 25
    # food_eaten_weight: float = 8               REMOVED (Prevents double-dipping with energy)
    energy_gained_weight: float = 30.0  # Lowered to balance with reproduction
    energy_efficiency_weight: float = (
        10.0  # SLASHED from 120 to fix the "Zen Monk" immobile exploit
    )
    efficiency_min_age_seconds: float = 20.0
    movement_effort_penalty: float = (
        0.02  # Lowered so active foragers aren't punished heavily
    )
    offspring_weight: float = 20.0  # MASSIVELY INCREASED (Base reward for giving birth)
    matured_offspring_weight: float = (
        40.0  # INCREASED (Bonus reward if the child survives)
    )
    trait_energy_cost_penalty_weight: float = (
        2.0  # Lowered so big creatures can still evolve
    )


@dataclass(slots=True)
class TraitConfig:
    default_radius: float = 16.0
    min_radius: float = 12.0
    max_radius: float = 22.0
    initial_radius_jitter: float = 2.0
    radius_mutation_stddev: float = 1.0

    default_movement_cost_multiplier: float = 1.0
    min_movement_cost_multiplier: float = 0.75
    max_movement_cost_multiplier: float = 1.35
    initial_movement_cost_jitter: float = 0.08
    movement_cost_mutation_stddev: float = 0.04

    default_separation_gene: float = 0.5
    default_alignment_gene: float = 0.5
    default_cohesion_gene: float = 0.5
    initial_flocking_gene_stdev: float = 0.08
    flocking_gene_mutation_rate: float = 0.05
    flocking_gene_mutation_power: float = 0.05
    flocking_gene_replace_rate: float = 0.005
    default_social_tag_x: float = 0.5
    default_social_tag_y: float = 0.5
    initial_social_tag_stdev: float = 0.15
    social_tag_mutation_rate: float = 0.05
    social_tag_mutation_power: float = 0.05
    social_tag_replace_rate: float = 0.005

    body_metabolism_cost_factor: float = 0.006


@dataclass(slots=True)
class VisionConfig:
    default_range: float = 98.0
    default_angle: float = 0.95
    fovea_ratio: float = 0.33

    min_range: float = 100.0
    max_range: float = 200.0
    min_angle: float = 0.35
    max_angle: float = pi

    base_energy_cost: float = 0.002
    area_energy_cost_factor: float = 0.018
    boundary_warning_distance: float = 90.0


@dataclass(slots=True)
class FoodConfig:
    initial_food_items: int = 300
    max_food_items: int = 300
    low_food_pressure_threshold: float = (
        0.5  # Food ratio below which low-food recovery can accumulate.
    )
    low_food_burst_items: int = (
        215  # Number of food items to spawn in a burst when food pressure is low.
    )
    low_food_burst_interval: float = 0.75  # Minimum interval in seconds between food bursts when food pressure is low.
    total_biomass_energy: float | None = None
    max_biomass_spawns_per_second: float = 10.0
    critical_food_ratio: float = 0.15  # Burst below 15% capacity

    min_food_radius: float = 6.0
    max_food_radius: float = 10.0
    energy_density: float = 0.002


@dataclass(slots=True)
class ActionConfig:
    max_forward_force: float = 125.0
    max_backward_force: float = 70.0
    max_turn_torque: float = 260.0
    max_sprint_multiplier: float = 0.5
    max_flock_turn_bias: float = 0.65
    collision_avoidance_margin: float = 8.0
    collision_avoidance_force_scale: float = 1.0
    action_smoothing_alpha: float = 0.8
    active_angular_velocity_retention: float = 0.80
    turn_control_gain: float = 0.65
    turn_response: float = 0.72
    turn_damping: float = 0.88
    turn_deadzone: float = 0.03
    angular_stop_threshold: float = 0.05
    forward_velocity_retention: float = 0.992
    lateral_velocity_retention: float = 0.72
    linear_stop_threshold: float = 0.05


@dataclass(slots=True)
class SimConfig:
    random_seed: int = 7
    display: DisplayConfig = field(default_factory=DisplayConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    biome: BiomeConfig = field(default_factory=BiomeConfig)
    zoom: ZoomConfig = field(default_factory=ZoomConfig)

    # Metabolism config
    metabolism: MetabolismConfig = field(default_factory=MetabolismConfig)

    # Fitness config
    fitness: FitnessConfig = field(default_factory=FitnessConfig)

    # Population config
    population: PopulationConfig = field(default_factory=PopulationConfig)

    # Speciation config
    speciation: SpeciationConfig = field(default_factory=SpeciationConfig)

    # Action config
    action: ActionConfig = field(default_factory=ActionConfig)

    # Trait config
    trait: TraitConfig = field(default_factory=TraitConfig)

    # Vision config
    vision: VisionConfig = field(default_factory=VisionConfig)

    # Food config
    food: FoodConfig = field(default_factory=FoodConfig)

    # Biome smell sensor config
    biome_sensor: BiomeSensorConfig = field(default_factory=BiomeSensorConfig)

    # Acoustic and pheromone communication
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)

    # Evolutionary flocking architecture and experiments.
    flocking: FlockingConfig = field(default_factory=FlockingConfig)


def build_sim_config() -> SimConfig:
    return SimConfig()
