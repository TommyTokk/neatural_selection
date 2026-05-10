from __future__ import annotations

from dataclasses import dataclass, field

Color = tuple[int, int, int] | tuple[int, int, int, int]


@dataclass(slots=True)
class DisplayConfig:
    width: int = 1440
    height: int = 900
    title: str = "neat_game_of_life"
    resizable: bool = True


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
    reproduction_energy_threshold: float = 0.8
    reproduction_energy_cost: float = 0.5
    eating_distance: float = 8
    starvation_energy_threshold: float = 0.3

@dataclass(slots=True)
class VisionConfig:
    default_range: float = 98.0
    default_angle: float = 0.95

    min_range:float = 45.0
    max_range:float = 160.0
    min_angle: float = 0.35
    max_angle: float = 1.40

    base_energy_cost: float = 0.002
    area_energy_cost_factor: float = 0.018

@dataclass(slots=True)
class FoodConfig:
    initial_food_items: int = 12
    spawn_interval: float = 2.0
    max_food_items: int = 60
    min_food_radius: float = 6.0
    max_food_radius: float = 10.0
    energy_density: float = 0.002


@dataclass(slots=True)
class SimConfig:
    display: DisplayConfig = field(default_factory=DisplayConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    zoom: ZoomConfig = field(default_factory=ZoomConfig)

    # Metabolism config
    metabolism: MetabolismConfig = field(default_factory=MetabolismConfig)

    # Vision config
    vision: VisionConfig = field(default_factory=VisionConfig)

    # Food config
    food: FoodConfig = field(default_factory=FoodConfig)






def build_sim_config() -> SimConfig:
    return SimConfig()
