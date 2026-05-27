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
    top_bar_height: int = 76
    left_panel_width: int = 320
    min_environment_width: int = 420
    min_sidebar_width: int = 240
    panel_radius: int = 18
    card_radius: int = 14
    environment_radius: int = 20


@dataclass(slots=True)
class ThemeConfig:
    window_background: Color = (18, 22, 30)
    panel_background: Color = (224, 238, 247)
    panel_background_alt: Color = (238, 230, 248)
    panel_border: Color = (137, 153, 180)
    environment_background: Color = (13, 21, 27)
    environment_grid: Color = (38, 53, 61, 95)
    environment_border: Color = (93, 126, 122)
    environment_text: Color = (210, 232, 226)
    environment_text_muted: Color = (132, 163, 164)
    accent: Color = (76, 146, 132)
    accent_soft: Color = (174, 223, 209)
    text_primary: Color = (29, 36, 48)
    text_muted: Color = (87, 101, 122)
    herbivore_fill: Color = (142, 203, 161)
    herbivore_outline: Color = (44, 88, 67)
    selected_outline: Color = (232, 63, 63)
    food_fill: Color = (192, 226, 130)
    vision_fill: Color = (142, 203, 161, 42)
    card_background: Color = (248, 244, 232)


@dataclass(slots=True)
class DebugConfig:
    vision_toggle_label: str = "V"
    show_debug_vision_by_default: bool = False


@dataclass(slots=True)
class ControllerConfig:
    use_neat_brains: bool = True


@dataclass(slots=True)
class ZoomConfig:
    default: float = 1.0
    minimum: float = 0.5
    maximum: float = 3.0
    step: float = 0.12


@dataclass(slots=True)
class MetabolismConfig:
    max_energy: float = 1
    basic_metabolism_rate: float = 0.01
    movement_energy_cost_factor: float = 0.02
    eating_distance: float = 8
    starvation_energy_threshold: float = 0.3


@dataclass(slots=True)
class PopulationConfig:
    initial_creatures: int = 20
    max_creatures: int = 30
    extinction_recovery_creatures: int = 20
    extinction_recovery_parent_pool: int = 5
    min_reproduction_age: float = 20.0
    reproduction_cooldown: float = 12.0
    reproduction_energy_threshold: float = 0.8
    reproduction_energy_cost: float = 0.5
    child_spawn_distance: float = 34.0


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
    offspring_weight: float = 12


@dataclass(slots=True)
class VisionConfig:
    default_range: float = 98.0
    default_angle: float = 0.95

    min_range: float = 90.0
    max_range: float = 160.0
    min_angle: float = 0.35
    max_angle: float = pi

    base_energy_cost: float = 0.002
    area_energy_cost_factor: float = 0.018
    boundary_warning_distance: float = 90.0


@dataclass(slots=True)
class FoodConfig:
    initial_food_items: int = 250
    max_food_items: int = 300
    low_creature_food_bonus_items: int = (
        200  # Number of bonus food items to spawn when creature count is low.
    )
    low_creature_food_bonus_threshold: int = (
        10  # Creature count threshold below which bonus food items will spawn.
    )
    low_creature_burst_items: int = (
        80  # Number of food items to spawn in a burst when creature count is low.
    )
    low_creature_burst_interval: float = 0.75  # Minimum interval in seconds between food bursts when creature count is low.
    low_food_pressure_threshold: float = 0.5  # Food spawn pressure threshold below which low food spawn rate multiplier is applied.
    low_food_spawn_rate_multiplier: float = 3.0  # Multiplier for food spawn rate when food pressure is below the low_food_pressure_threshold.
    low_food_burst_items: int = (
        100  # Number of food items to spawn in a burst when food pressure is low.
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
    centered_food_angle_threshold: float = 0.12
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
    zoom: ZoomConfig = field(default_factory=ZoomConfig)

    # Metabolism config
    metabolism: MetabolismConfig = field(default_factory=MetabolismConfig)

    # Fitness config
    fitness: FitnessConfig = field(default_factory=FitnessConfig)

    # Population config
    population: PopulationConfig = field(default_factory=PopulationConfig)

    # Action config
    action: ActionConfig = field(default_factory=ActionConfig)

    # Vision config
    vision: VisionConfig = field(default_factory=VisionConfig)

    # Food config
    food: FoodConfig = field(default_factory=FoodConfig)


def build_sim_config() -> SimConfig:
    return SimConfig()
