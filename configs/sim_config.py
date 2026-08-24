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
    enabled: bool = True
    reward_rate: float = 0.01
    target_group_size: int = 4
    target_spacing: float = 60.0
    spacing_tolerance: float = 30.0
    reference_speed: float = 50.0
    max_per_evaluation: float = 1.0


@dataclass(slots=True)
class FlockingConfig:
    default_separation_gene: float = 0.5
    default_alignment_gene: float = 0.5
    default_cohesion_gene: float = 0.5
    initial_flocking_gene_stdev: float = 0.08
    flocking_gene_mutation_rate: float = 0.05
    flocking_gene_mutation_sigma_u: float = 0.20
    flocking_gene_replace_rate: float = 0.005
    default_social_tag_x: float = 0.5
    default_social_tag_y: float = 0.5
    initial_social_tag_stdev: float = 0.15
    social_tag_mutation_rate: float = 0.05
    social_tag_mutation_sigma_u: float = 0.20
    social_tag_replace_rate: float = 0.005

    minimum_social_engagement: float = 0.25
    panic_suppression_strength: float = 0.5
    max_social_influence: float = 0.35
    herding_decay_rate: float = 0.15
    target_group_size: int = 4
    perception_radius: float = 150.0
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

        inherited_fractions = {
            "default_separation_gene": self.default_separation_gene,
            "default_alignment_gene": self.default_alignment_gene,
            "default_cohesion_gene": self.default_cohesion_gene,
            "default_social_tag_x": self.default_social_tag_x,
            "default_social_tag_y": self.default_social_tag_y,
            "flocking_gene_mutation_rate": self.flocking_gene_mutation_rate,
            "flocking_gene_replace_rate": self.flocking_gene_replace_rate,
            "social_tag_mutation_rate": self.social_tag_mutation_rate,
            "social_tag_replace_rate": self.social_tag_replace_rate,
        }
        for name, value in inherited_fractions.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"flocking.{name} must be finite and within [0, 1].")
        for prefix in ("flocking_gene", "social_tag"):
            mutation_rate = getattr(self, f"{prefix}_mutation_rate")
            replace_rate = getattr(self, f"{prefix}_replace_rate")
            if mutation_rate + replace_rate > 1.0:
                raise ValueError(
                    f"flocking.{prefix}_mutation_rate + {prefix}_replace_rate "
                    "must not exceed 1.0."
                )
        for name in (
            "flocking_gene_mutation_sigma_u",
            "social_tag_mutation_sigma_u",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"flocking.{name} must be finite and positive.")
        for name in ("initial_flocking_gene_stdev", "initial_social_tag_stdev"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"flocking.{name} must be finite and nonnegative.")

        if (
            isinstance(self.herding_decay_rate, bool)
            or not isinstance(self.herding_decay_rate, (int, float))
            or not isfinite(self.herding_decay_rate)
            or not 0.0 < self.herding_decay_rate <= 1.0
        ):
            raise ValueError(
                "herding_decay_rate must be finite and within (0, 1]."
            )

        positive = {
            "perception_radius": self.perception_radius,
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
    flock_perception_outline: Color = (255, 165, 0, 210)
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
    world_width: float = 3520.0
    world_height: float = 2420.0


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
    max_life: float = 1.0
    initial_life_fraction: float = 1.0
    life_damage_per_energy_deficit: float = 0.25
    movement_life_penalty_max_multiplier: float = 4.0
    rest_digestion_efficiency_bonus: float = 0.10
    # Rest may reduce activity and improve digestion, but it does not create
    # energy in the thermodynamic model.
    rest_energy_recovery_per_second: float = 0.0
    rest_healing_rate_per_second: float = 0.01
    rest_healing_energy_cost_per_life: float = 1.0
    basic_metabolism_rate: float = 0.005
    brain_upkeep_per_enabled_connection: float = 0.00008
    movement_energy_cost_factor: float = 0.02
    sprint_energy_cost_per_second: float = 0.04
    eating_distance: float = 8
    micro_food_remainder_ratio: float = 0.10
    stomach_capacity_per_radius: float = 0.1
    digestion_rate_per_second: float = 0.2
    digestion_efficiency: float = 0.9
    max_bite_size_per_second: float = 0.5
    starvation_energy_threshold: float = 0.3
    digestive_upkeep_at_default_per_second: float = 0.004
    max_digestive_upkeep_per_second: float = 0.012
    digestive_capacity_upkeep_weight: float = 0.40
    digestive_rate_upkeep_weight: float = 0.35
    digestive_efficiency_upkeep_weight: float = 0.25
    digestion_processing_base_fraction: float = 0.08
    digestion_rate_cost_factor: float = 1.5
    min_food_difficulty_multiplier: float = 0.75
    max_food_difficulty_multiplier: float = 1.25
    max_digestion_processing_fraction: float = 0.5


@dataclass(slots=True)
class PopulationConfig:
    initial_creatures: int = 40
    max_creatures: int = 55
    genome_archive_size: int = 256
    fitness_archive_size: int = 256
    extinction_recovery_creatures: int = 35
    extinction_recovery_parent_pool: int = 5
    senescence_age_seconds: float = 200.0
    senescence_cost_multiplier: float = 0.05
    reproduction_energy_fraction: float = 0.75
    child_energy_investment_fraction: float = 0.45
    birth_conversion_efficiency: float = 0.90
    maturity_age_seconds: float = 10.0
    birth_cooldown_seconds: float = 5.0
    # The action layer uses symmetric sigmoid centering, so 0.2 maps back to
    # a raw sigmoid probability of 0.6.
    reproduction_intent_threshold: float = 0.2
    child_spawn_max_attempts: int = 16
    nursing_energy_transfer_rate: float = 0.05
    child_spawn_distance: float = 34.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate thermodynamic birth settings at construction and startup."""
        for name in (
            "reproduction_energy_fraction",
            "child_energy_investment_fraction",
            "birth_conversion_efficiency",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"population.{name} must be within (0, 1].")
        if (
            not isfinite(float(self.reproduction_intent_threshold))
            or not 0.0 <= self.reproduction_intent_threshold <= 1.0
        ):
            raise ValueError(
                "population.reproduction_intent_threshold must be within [0, 1]."
            )
        for name in ("maturity_age_seconds", "birth_cooldown_seconds"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"population.{name} cannot be negative.")
        if (
            type(self.child_spawn_max_attempts) is not int
            or self.child_spawn_max_attempts <= 0
        ):
            raise ValueError(
                "population.child_spawn_max_attempts must be a positive integer."
            )

    # Checkpoint/test compatibility for pre-v25 timing names. The simulation
    # reads only the physiological fields above.
    @property
    def min_reproduction_age(self) -> float:
        return self.maturity_age_seconds

    @min_reproduction_age.setter
    def min_reproduction_age(self, value: float) -> None:
        self.maturity_age_seconds = float(value)

    @property
    def infant_maturity_age(self) -> float:
        return self.maturity_age_seconds

    @infant_maturity_age.setter
    def infant_maturity_age(self, value: float) -> None:
        self.maturity_age_seconds = float(value)

    @property
    def reproduction_cooldown(self) -> float:
        return self.birth_cooldown_seconds

    @reproduction_cooldown.setter
    def reproduction_cooldown(self, value: float) -> None:
        self.birth_cooldown_seconds = float(value)

    @property
    def reproduction_energy_threshold(self) -> float:
        return self.reproduction_energy_fraction

    @reproduction_energy_threshold.setter
    def reproduction_energy_threshold(self, value: float) -> None:
        self.reproduction_energy_fraction = float(value)

    @property
    def elite_archive_size(self) -> int:
        """Compatibility alias for the unranked genome archive limit."""
        return self.genome_archive_size

    @elite_archive_size.setter
    def elite_archive_size(self, value: int) -> None:
        self.genome_archive_size = int(value)


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
class TraitConfig:
    default_radius: float = 16.0
    min_radius: float = 12.0
    max_radius: float = 22.0
    initial_radius_jitter: float = 2.0
    radius_mutation_sigma_u: float = 0.42

    default_movement_cost_multiplier: float = 1.0
    min_movement_cost_multiplier: float = 0.75
    max_movement_cost_multiplier: float = 1.35
    initial_movement_cost_jitter: float = 0.08
    movement_cost_mutation_sigma_u: float = 0.27

    default_stomach_capacity: float = 1.6
    min_stomach_capacity: float = 0.8
    max_stomach_capacity: float = 2.6
    initial_stomach_capacity_jitter: float = 0.15
    stomach_capacity_mutation_sigma_u: float = 0.27

    default_digestion_rate: float = 0.20
    min_digestion_rate: float = 0.05
    max_digestion_rate: float = 0.40
    initial_digestion_rate_jitter: float = 0.025
    digestion_rate_mutation_sigma_u: float = 0.23

    default_digestion_efficiency: float = 0.90
    min_digestion_efficiency: float = 0.55
    max_digestion_efficiency: float = 0.98
    initial_digestion_efficiency_jitter: float = 0.02
    digestion_efficiency_mutation_sigma_u: float = 0.23
    digestive_trait_mutation_rate: float = 0.15

    body_metabolism_cost_factor: float = 0.004

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate inherited physical and digestive trait configuration."""
        ranges = (
            ("radius", self.min_radius, self.default_radius, self.max_radius),
            (
                "movement_cost_multiplier",
                self.min_movement_cost_multiplier,
                self.default_movement_cost_multiplier,
                self.max_movement_cost_multiplier,
            ),
            (
                "stomach_capacity",
                self.min_stomach_capacity,
                self.default_stomach_capacity,
                self.max_stomach_capacity,
            ),
            (
                "digestion_rate",
                self.min_digestion_rate,
                self.default_digestion_rate,
                self.max_digestion_rate,
            ),
            (
                "digestion_efficiency",
                self.min_digestion_efficiency,
                self.default_digestion_efficiency,
                self.max_digestion_efficiency,
            ),
        )
        for name, minimum, default, maximum in ranges:
            values = (minimum, default, maximum)
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                for value in values
            ):
                raise ValueError(f"trait.{name} bounds and default must be finite.")
            if maximum <= minimum:
                raise ValueError(f"trait.{name} range must have positive width.")
            if not minimum <= default <= maximum:
                raise ValueError(f"trait.default_{name} must be within its bounds.")

        for name in (
            "radius_mutation_sigma_u",
            "movement_cost_mutation_sigma_u",
            "stomach_capacity_mutation_sigma_u",
            "digestion_rate_mutation_sigma_u",
            "digestion_efficiency_mutation_sigma_u",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"trait.{name} must be finite and positive.")
        for name in (
            "initial_radius_jitter",
            "initial_movement_cost_jitter",
            "initial_stomach_capacity_jitter",
            "initial_digestion_rate_jitter",
            "initial_digestion_efficiency_jitter",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"trait.{name} must be finite and nonnegative.")
        mutation_rate = self.digestive_trait_mutation_rate
        if (
            isinstance(mutation_rate, bool)
            or not isinstance(mutation_rate, (int, float))
            or not isfinite(mutation_rate)
            or not 0.0 <= mutation_rate <= 1.0
        ):
            raise ValueError(
                "trait.digestive_trait_mutation_rate must be within [0, 1]."
            )


@dataclass(slots=True)
class VisionConfig:
    default_range: float = 150.0
    default_angle: float = 0.95
    range_mutation_sigma_u: float = 0.32
    angle_mutation_sigma_u: float = 0.17
    fovea_ratio: float = 0.33

    min_range: float = 100.0
    max_range: float = 200.0
    min_angle: float = 0.35
    max_angle: float = pi

    base_energy_cost: float = 0.001
    area_energy_cost_factor: float = 0.005
    boundary_warning_distance: float = 90.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate visual trait bounds, defaults, and latent mutation powers."""
        for name, minimum, default, maximum in (
            ("range", self.min_range, self.default_range, self.max_range),
            ("angle", self.min_angle, self.default_angle, self.max_angle),
        ):
            values = (minimum, default, maximum)
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                for value in values
            ):
                raise ValueError(f"vision.{name} bounds and default must be finite.")
            if maximum <= minimum:
                raise ValueError(f"vision.{name} range must have positive width.")
            if not minimum <= default <= maximum:
                raise ValueError(f"vision.default_{name} must be within its bounds.")
        for name in ("range_mutation_sigma_u", "angle_mutation_sigma_u"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"vision.{name} must be finite and positive.")


@dataclass(slots=True)
class FoodConfig:
    initial_food_items: int = 363
    max_food_items: int = 363
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


@dataclass(frozen=True, slots=True)
class BiomeFoodClusterProfile:
    """Spawn characteristics for one biome's food patches.

    ``cluster_count_weight`` is a relative patch-frequency weight.  It is
    intentionally independent from biome area and ordinary pellet fertility.
    A zero weight disables newly-created patches for that biome.
    """

    cluster_count_weight: float
    pellets_per_cluster: tuple[int, int]
    spread_radius: tuple[float, float]
    energy_multiplier: tuple[float, float]
    bite_capacity: tuple[int, int]

    def __post_init__(self) -> None:
        if (
            isinstance(self.cluster_count_weight, bool)
            or not isfinite(self.cluster_count_weight)
            or self.cluster_count_weight < 0.0
        ):
            raise ValueError("cluster_count_weight must be finite and nonnegative.")
        self._validate_range("pellets_per_cluster", self.pellets_per_cluster, 1)
        self._validate_range("spread_radius", self.spread_radius, 0.0)
        self._validate_range("energy_multiplier", self.energy_multiplier, 0.000001)
        self._validate_range("bite_capacity", self.bite_capacity, 1)

    @staticmethod
    def _validate_range(name: str, values: tuple[object, object], minimum: float) -> None:
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two values.")
        low, high = values
        if name in ("pellets_per_cluster", "bite_capacity"):
            if type(low) is not int or type(high) is not int:
                raise ValueError(f"{name} values must be integers.")
        elif any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            for value in values
        ):
            raise ValueError(f"{name} values must be finite numbers.")
        if low < minimum or high < low:
            raise ValueError(
                f"{name} must be ordered and no smaller than {minimum}."
            )


@dataclass(slots=True)
class FoodClusterConfig:
    """Configuration for the clustered portion of mixed food spawning."""

    cluster_spawn_share: float = 0.30
    prairie_max: float | None = None
    bush_max: float | None = None
    depletion_ratio: float = 0.10
    cooldown_ticks: tuple[int, int] = (600, 1200)
    minimum_relocation_distance: float = 1.0
    max_sampling_attempts: int = 96
    radius_shrink_factor: float = 0.80
    minimum_radius_ratio: float = 0.25
    center_sampling_attempts: int = 32
    prairie: BiomeFoodClusterProfile = field(
        default_factory=lambda: BiomeFoodClusterProfile(
            0.0, (1, 1), (0.0, 0.0), (1.0, 1.0), (1, 1)
        )
    )
    bushes: BiomeFoodClusterProfile = field(
        default_factory=lambda: BiomeFoodClusterProfile(
            3.0, (3, 6), (20.0, 40.0), (1.0, 1.0), (1, 2)
        )
    )
    forest: BiomeFoodClusterProfile = field(
        default_factory=lambda: BiomeFoodClusterProfile(
            1.0, (12, 25), (50.0, 100.0), (2.0, 5.0), (2, 6)
        )
    )

    def __post_init__(self) -> None:
        for name in ("cluster_spawn_share", "depletion_ratio", "minimum_radius_ratio"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and within [0, 1].")
        for name in ("prairie_max", "bush_max"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be None or finite within [0, 1].")
        if (
            self.prairie_max is not None
            and self.bush_max is not None
            and self.prairie_max >= self.bush_max
        ):
            raise ValueError("prairie_max must be smaller than bush_max.")
        low_cooldown, high_cooldown = self.cooldown_ticks
        if (
            type(low_cooldown) is not int
            or type(high_cooldown) is not int
            or low_cooldown < 0
            or high_cooldown < low_cooldown
        ):
            raise ValueError("cooldown_ticks must be ordered nonnegative integers.")
        if type(self.max_sampling_attempts) is not int or self.max_sampling_attempts < 1:
            raise ValueError("max_sampling_attempts must be a positive integer.")
        if type(self.center_sampling_attempts) is not int or self.center_sampling_attempts < 1:
            raise ValueError("center_sampling_attempts must be a positive integer.")
        if not 0.0 < self.radius_shrink_factor < 1.0:
            raise ValueError("radius_shrink_factor must be within (0, 1).")
        if self.minimum_radius_ratio <= 0.0:
            raise ValueError("minimum_radius_ratio must be positive.")
        if self.minimum_relocation_distance < 0.0:
            raise ValueError("minimum_relocation_distance must be nonnegative.")

    @property
    def independent_spawn_share(self) -> float:
        return 1.0 - self.cluster_spawn_share


@dataclass(frozen=True, slots=True)
class LiveFoodConfig:
    """Food and biome-richness values that may change during a run."""

    forest_spawn_weight: float
    bushes_spawn_weight: float
    prairie_spawn_weight: float
    max_food_items: int
    low_food_pressure_threshold: float
    critical_food_ratio: float
    low_food_burst_items: int
    low_food_burst_interval: float
    cluster_spawn_share: float = 0.30

    def __post_init__(self) -> None:
        for name in (
            "forest_spawn_weight",
            "bushes_spawn_weight",
            "prairie_spawn_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and nonnegative.")

        for name in ("max_food_items", "low_food_burst_items"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")

        for name in (
            "low_food_pressure_threshold",
            "critical_food_ratio",
            "cluster_spawn_share",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and within [0, 1].")
        if self.critical_food_ratio > self.low_food_pressure_threshold:
            raise ValueError(
                "critical_food_ratio must not exceed "
                "low_food_pressure_threshold."
            )

        interval = self.low_food_burst_interval
        if (
            isinstance(interval, bool)
            or not isinstance(interval, (int, float))
            or not isfinite(interval)
            or interval <= 0.0
        ):
            raise ValueError(
                "low_food_burst_interval must be finite and positive."
            )

    @classmethod
    def from_configs(
        cls,
        biome: BiomeConfig,
        food: FoodConfig,
        clusters: FoodClusterConfig | None = None,
    ) -> LiveFoodConfig:
        return cls(
            forest_spawn_weight=float(biome.forest_spawn_weight),
            bushes_spawn_weight=float(biome.bushes_spawn_weight),
            prairie_spawn_weight=float(biome.prairie_spawn_weight),
            max_food_items=int(food.max_food_items),
            low_food_pressure_threshold=float(
                food.low_food_pressure_threshold
            ),
            critical_food_ratio=float(food.critical_food_ratio),
            low_food_burst_items=int(food.low_food_burst_items),
            low_food_burst_interval=float(food.low_food_burst_interval),
            cluster_spawn_share=float(
                (clusters or FoodClusterConfig()).cluster_spawn_share
            ),
        )

    def to_primitive(self) -> dict[str, int | float]:
        return {
            "forest_spawn_weight": self.forest_spawn_weight,
            "bushes_spawn_weight": self.bushes_spawn_weight,
            "prairie_spawn_weight": self.prairie_spawn_weight,
            "max_food_items": self.max_food_items,
            "low_food_pressure_threshold": self.low_food_pressure_threshold,
            "critical_food_ratio": self.critical_food_ratio,
            "low_food_burst_items": self.low_food_burst_items,
            "low_food_burst_interval": self.low_food_burst_interval,
            "cluster_spawn_share": self.cluster_spawn_share,
        }

    @classmethod
    def from_primitive(
        cls,
        value: object,
        *,
        fallback: LiveFoodConfig,
    ) -> LiveFoodConfig:
        if not isinstance(value, dict):
            raise TypeError("live food configuration must be a dictionary.")
        defaults = fallback.to_primitive()
        defaults.update(value)
        return cls(
            forest_spawn_weight=float(defaults["forest_spawn_weight"]),
            bushes_spawn_weight=float(defaults["bushes_spawn_weight"]),
            prairie_spawn_weight=float(defaults["prairie_spawn_weight"]),
            max_food_items=int(defaults["max_food_items"]),
            low_food_pressure_threshold=float(
                defaults["low_food_pressure_threshold"]
            ),
            critical_food_ratio=float(defaults["critical_food_ratio"]),
            low_food_burst_items=int(defaults["low_food_burst_items"]),
            low_food_burst_interval=float(
                defaults["low_food_burst_interval"]
            ),
            cluster_spawn_share=float(defaults["cluster_spawn_share"]),
        )


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
    rest_response_rate: float = 3.0
    rest_decay_rate: float = 1.5
    rest_movement_exponent: float = 2.0
    rest_rotation_inhibition: float = 0.5
    rest_braking_strength: float = 2.5

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate rest parameters after construction or runtime mutation."""
        finite_nonnegative = (
            "rest_response_rate",
            "rest_decay_rate",
            "rest_movement_exponent",
            "rest_rotation_inhibition",
            "rest_braking_strength",
        )
        for name in finite_nonnegative:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"action.{name} must be finite and nonnegative.")
        if self.rest_movement_exponent <= 0.0:
            raise ValueError("action.rest_movement_exponent must be positive.")
        if self.rest_rotation_inhibition > 1.0:
            raise ValueError(
                "action.rest_rotation_inhibition must be within [0, 1]."
            )


@dataclass(slots=True)
class BehaviorObserverConfig:
    """Configuration for automatic and focal temporal observation."""

    enabled: bool = True
    background_representatives_per_species: int = 3
    sample_hz: float = 10.0
    window_seconds: float = 2.5
    bout_start_seconds: float = 0.5
    bout_end_grace_seconds: float = 0.3
    feeding_display_seconds: float = 0.75
    input_queue_capacity: int = 8
    result_queue_capacity: int = 4
    rest_speed_threshold: float = 2.0
    food_visibility_ratio: float = 0.60
    trend_consistency_ratio: float = 0.67
    orientation_min_error_reduction: float = 0.15
    orientation_min_turn_rate: float = 0.10
    approach_min_closing_speed: float = 8.0
    movement_alignment_threshold: float = 0.35
    cohesion_min_closing_speed: float = 5.0
    cohesion_min_velocity_alignment: float = 0.75
    alarm_retreat_min_speed: float = 10.0
    alarm_min_level: float = 0.10
    alarm_min_spatial_gradient: float = 0.02
    alarm_min_temporal_drop: float = 0.03

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("behavior.enabled must be a boolean.")
        if (
            type(self.background_representatives_per_species) is not int
            or self.background_representatives_per_species < 0
        ):
            raise ValueError(
                "behavior.background_representatives_per_species must be "
                "a nonnegative integer."
            )
        positive = {
            "sample_hz": self.sample_hz,
            "window_seconds": self.window_seconds,
            "bout_start_seconds": self.bout_start_seconds,
            "feeding_display_seconds": self.feeding_display_seconds,
            "rest_speed_threshold": self.rest_speed_threshold,
            "orientation_min_error_reduction": (
                self.orientation_min_error_reduction
            ),
            "orientation_min_turn_rate": self.orientation_min_turn_rate,
            "approach_min_closing_speed": self.approach_min_closing_speed,
            "cohesion_min_closing_speed": self.cohesion_min_closing_speed,
            "alarm_retreat_min_speed": self.alarm_retreat_min_speed,
            "alarm_min_level": self.alarm_min_level,
            "alarm_min_spatial_gradient": self.alarm_min_spatial_gradient,
            "alarm_min_temporal_drop": self.alarm_min_temporal_drop,
        }
        for name, value in positive.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(
                    f"behavior.{name} must be finite and positive."
                )
        grace = self.bout_end_grace_seconds
        if (
            isinstance(grace, bool)
            or not isinstance(grace, (int, float))
            or not isfinite(grace)
            or grace < 0.0
        ):
            raise ValueError(
                "behavior.bout_end_grace_seconds must be finite and "
                "nonnegative."
            )
        if self.window_seconds < self.bout_start_seconds:
            raise ValueError(
                "behavior.window_seconds must be at least "
                "behavior.bout_start_seconds."
            )
        for name in ("input_queue_capacity", "result_queue_capacity"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(
                    f"behavior.{name} must be a positive integer."
                )
        for name in (
            "food_visibility_ratio",
            "trend_consistency_ratio",
            "movement_alignment_threshold",
            "cohesion_min_velocity_alignment",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(
                    f"behavior.{name} must be finite and within [0, 1]."
                )


@dataclass(slots=True)
class CounterfactualWhyConfig:
    """Configuration for focal counterfactual NEAT explanations."""

    enabled: bool = True
    probe_hz: float = 5.0
    history_capacity: int = 64
    control_queue_capacity: int = 1
    probe_queue_capacity: int = 2
    result_queue_capacity: int = 4
    target_center_dead_zone_radians: float = 0.05

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("counterfactual_why.enabled must be a boolean.")
        if (
            isinstance(self.probe_hz, bool)
            or not isinstance(self.probe_hz, (int, float))
            or not isfinite(self.probe_hz)
            or self.probe_hz <= 0.0
        ):
            raise ValueError(
                "counterfactual_why.probe_hz must be finite and positive."
            )
        if (
            isinstance(self.target_center_dead_zone_radians, bool)
            or not isinstance(
                self.target_center_dead_zone_radians,
                (int, float),
            )
            or not isfinite(self.target_center_dead_zone_radians)
            or not 0.0 <= self.target_center_dead_zone_radians < pi
        ):
            raise ValueError(
                "counterfactual_why.target_center_dead_zone_radians "
                "must be finite and within [0, pi)."
            )
        for name in (
            "history_capacity",
            "control_queue_capacity",
            "probe_queue_capacity",
            "result_queue_capacity",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(
                    f"counterfactual_why.{name} must be a positive integer."
                )


@dataclass(slots=True)
class BehaviorHistoryConfig:
    """Configuration for compact completed focal-behaviour history."""

    max_completed_bouts_per_creature: int = 256
    max_remembered_creatures: int = 16
    minimum_stable_bouts: int = 3
    active_metric_sample_capacity: int = 512
    completion_queue_capacity: int = 64
    completion_outbox_soft_capacity: int = 256
    completion_outbox_hard_capacity: int = 1024
    completion_outbox_recovery_capacity: int = 128

    def __post_init__(self) -> None:
        for name in (
            "max_completed_bouts_per_creature",
            "max_remembered_creatures",
            "minimum_stable_bouts",
            "active_metric_sample_capacity",
            "completion_queue_capacity",
            "completion_outbox_soft_capacity",
            "completion_outbox_hard_capacity",
            "completion_outbox_recovery_capacity",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(
                    f"behavior_history.{name} must be a positive integer."
                )
        if self.active_metric_sample_capacity < 4:
            raise ValueError(
                "behavior_history.active_metric_sample_capacity must be "
                "at least four."
            )
        recovery = self.completion_outbox_recovery_capacity
        soft = self.completion_outbox_soft_capacity
        hard = self.completion_outbox_hard_capacity
        if not recovery < soft < hard:
            raise ValueError(
                "behavior_history completion outbox capacities must satisfy "
                "recovery < soft < hard."
            )


@dataclass(slots=True)
class SchedulerConfig:
    """Deterministic fixed-step scheduling configuration."""

    physics_hz: int = 60
    decision_period_steps: int = 3
    biology_period_steps: int = 3
    statistics_period_steps: int = 12
    max_steps_per_frame: int = 5
    max_backlog_steps: int = 60

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in (
            "physics_hz",
            "decision_period_steps",
            "biology_period_steps",
            "statistics_period_steps",
            "max_steps_per_frame",
            "max_backlog_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"scheduler.{name} must be a positive integer.")

        for name in (
            "decision_period_steps",
            "biology_period_steps",
            "statistics_period_steps",
        ):
            period = getattr(self, name)
            if self.physics_hz % period != 0:
                raise ValueError(
                    f"scheduler.physics_hz must be exactly divisible by "
                    f"scheduler.{name}."
                )
        if self.max_backlog_steps < self.max_steps_per_frame:
            raise ValueError(
                "scheduler.max_backlog_steps must be at least "
                "scheduler.max_steps_per_frame."
            )

    @property
    def decision_hz(self) -> int:
        return self.physics_hz // self.decision_period_steps

    @property
    def biology_hz(self) -> int:
        return self.physics_hz // self.biology_period_steps

    @property
    def statistics_hz(self) -> int:
        return self.physics_hz // self.statistics_period_steps


@dataclass(slots=True)
class SimConfig:
    random_seed: int = 7
    display: DisplayConfig = field(default_factory=DisplayConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    biome: BiomeConfig = field(default_factory=BiomeConfig)
    zoom: ZoomConfig = field(default_factory=ZoomConfig)

    # Metabolism config
    metabolism: MetabolismConfig = field(default_factory=MetabolismConfig)

    # Population config
    population: PopulationConfig = field(default_factory=PopulationConfig)

    # Speciation config
    speciation: SpeciationConfig = field(default_factory=SpeciationConfig)

    # Action config
    action: ActionConfig = field(default_factory=ActionConfig)

    # Automatic and focal temporal behaviour observer
    behavior: BehaviorObserverConfig = field(
        default_factory=BehaviorObserverConfig
    )
    counterfactual_why: CounterfactualWhyConfig = field(
        default_factory=CounterfactualWhyConfig
    )
    behavior_history: BehaviorHistoryConfig = field(
        default_factory=BehaviorHistoryConfig
    )

    # Trait config
    trait: TraitConfig = field(default_factory=TraitConfig)

    # Vision config
    vision: VisionConfig = field(default_factory=VisionConfig)

    # Food config
    food: FoodConfig = field(default_factory=FoodConfig)
    food_clusters: FoodClusterConfig = field(default_factory=FoodClusterConfig)

    # Biome smell sensor config
    biome_sensor: BiomeSensorConfig = field(default_factory=BiomeSensorConfig)

    # Acoustic and pheromone communication
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)

    # Evolutionary flocking architecture and experiments.
    flocking: FlockingConfig = field(default_factory=FlockingConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate mutable startup configuration before service creation."""
        self.trait.validate()
        self.vision.validate()
        self.flocking.validate()
        self.action.validate()
        self.scheduler.validate()
        self.population.validate()


def build_sim_config() -> SimConfig:
    return SimConfig()
