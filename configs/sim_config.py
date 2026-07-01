from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

Color = tuple[int, int, int] | tuple[int, int, int, int]


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
class ControllerConfig:
    use_neat_brains: bool = True


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
    forward_distance: float = 48.0
    side_offset: float = 24.0
    delta_scale: float = 10.0


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
    movement_energy_cost_factor: float = 0.02
    sprint_energy_cost_per_second: float = 0.04
    eating_distance: float = 8
    micro_food_remainder_ratio: float = 0.10
    starvation_energy_threshold: float = 0.3


@dataclass(slots=True)
class PopulationConfig:
    initial_creatures: int = 50
    max_creatures: int = 100
    extinction_recovery_creatures: int = 35
    extinction_recovery_parent_pool: int = 5
    min_reproduction_age: float = 20.0
    reproduction_cooldown: float = 12.0
    reproduction_energy_threshold: float = 0.8
    reproduction_energy_cost: float = 0.5
    infant_energy_spawn: float = 0.15
    infant_maturity_age: float = 5.0
    nursing_energy_transfer_rate: float = 0.05
    child_spawn_distance: float = 34.0
    reproduction_min_food_ratio: float = 0.2
    reproduction_min_available_biomass_ratio: float = 0.02
    reproduction_recovery_pressure_threshold: float = 0.25


@dataclass(slots=True)
class SpeciationConfig:
    compatibility_threshold: float = 3.0
    phenotypic_weight: float = 2.0


@dataclass(slots=True)
class FitnessConfig:
    age_weight: float = 0.03
    food_discovery_weight: float = 0.2
    food_discovery_cap: int = 25
    food_eaten_weight: float = 8
    energy_gained_weight: float = 80
    energy_efficiency_weight: float = 120
    efficiency_min_age_seconds: float = 20
    movement_effort_penalty: float = 0.08
    offspring_weight: float = 2.0
    matured_offspring_weight: float = 30.0
    trait_energy_cost_penalty_weight: float = 5.0


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

    body_metabolism_cost_factor: float = 0.006


@dataclass(slots=True)
class VisionConfig:
    default_range: float = 98.0
    default_angle: float = 0.95
    fovea_ratio: float = 0.33

    min_range: float = 90.0
    max_range: float = 160.0
    min_angle: float = 0.35
    max_angle: float = pi

    base_energy_cost: float = 0.002
    area_energy_cost_factor: float = 0.018
    boundary_warning_distance: float = 90.0


@dataclass(slots=True)
class FoodConfig:
    initial_food_items: int = 540
    max_food_items: int = 650
    low_creature_food_bonus_items: int = (
        430  # Number of bonus food items to spawn when creature count is low.
    )
    low_creature_food_bonus_threshold: int = (
        10  # Creature count threshold below which bonus food items will spawn.
    )
    low_creature_burst_items: int = (
        170  # Number of food items to spawn in a burst when creature count is low.
    )
    low_creature_burst_interval: float = 0.75  # Minimum interval in seconds between food bursts when creature count is low.
    low_food_pressure_threshold: float = 0.5  # Food spawn pressure threshold below which low food spawn rate multiplier is applied.
    low_food_spawn_rate_multiplier: float = 3.0  # Multiplier for food spawn rate when food pressure is below the low_food_pressure_threshold.
    low_food_burst_items: int = (
        215  # Number of food items to spawn in a burst when food pressure is low.
    )
    low_food_burst_interval: float = 0.75  # Minimum interval in seconds between food bursts when food pressure is low.
    total_biomass_energy: float | None = None
    max_biomass_spawns_per_second: float = 20.0
    biomass_spawn_pressure_exponent: float = 1.6
    creature_pressure_midpoint: float = 18.0
    creature_pressure_steepness: float = 3.0
    creature_pressure_spawn_cutoff: float = 0.05

    food_regrowth_midpoint_ratio: float = 0.65  # Higher spawn rate below 65% capacity
    food_regrowth_steepness: float = 0.12  # Sharpness of the logistic curve
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
    turn_response: float = 0.72
    turn_damping: float = 0.88
    turn_deadzone: float = 0.03
    angular_stop_threshold: float = 0.05
    forward_velocity_retention: float = 0.992
    lateral_velocity_retention: float = 0.72
    linear_stop_threshold: float = 0.05
    search_turn: float = 0.018
    search_acceleration: float = 0.82
    search_turn_interval_min: int = 35
    search_turn_interval_max: int = 95
    search_straight_probability: float = 0.3
    search_turn_jitter: float = 0.0
    search_angular_velocity_retention: float = 0.4  # Higher values make creatures more likely to continue turning in the same direction during search, while lower values make them more likely to change direction.
    boundary_avoidance_turn: float = 0.72
    boundary_avoidance_min_turn: float = 0.18
    boundary_avoidance_acceleration: float = 0.75
    boundary_escape_pressure: float = 0.72
    boundary_escape_turn_threshold: float = 0.65
    boundary_escape_acceleration: float = -0.35
    boundary_angular_velocity_retention: float = 0.72
    food_turn_factor: float = 0.25
    min_food_acceleration: float = 0.7
    centered_food_angular_damping: float = 0.65
    centered_food_angular_velocity_retention: float = 0.15
    food_tracking_lateral_velocity_retention: float = 0.35
    food_tracking_backward_velocity_retention: float = 0.65
    low_energy_threshold: float = 0.25
    low_energy_acceleration_factor: float = 0.5


@dataclass(slots=True)
class SimConfig:
    display: DisplayConfig = field(default_factory=DisplayConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
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


def build_sim_config() -> SimConfig:
    return SimConfig()
