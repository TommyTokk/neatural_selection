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
    outer_padding: int = 24
    panel_gap: int = 18
    top_bar_height: int = 84
    right_panel_width: int = 320
    bottom_bar_height: int = 140
    min_environment_width: int = 420
    min_sidebar_width: int = 240


@dataclass(slots=True)
class ThemeConfig:
    window_background: Color = (245, 244, 238)
    panel_background: Color = (250, 249, 245)
    panel_border: Color = (58, 67, 78)
    environment_background: Color = (230, 237, 229)
    environment_grid: Color = (195, 207, 194)
    environment_border: Color = (43, 56, 47)
    accent: Color = (69, 114, 92)
    accent_soft: Color = (179, 214, 190)
    text_primary: Color = (31, 38, 46)
    text_muted: Color = (97, 108, 121)
    herbivore_fill: Color = (108, 167, 108)
    herbivore_outline: Color = (31, 74, 40)
    selected_outline: Color = (224, 152, 54)
    food_fill: Color = (118, 183, 84)
    vision_fill: Color = (108, 167, 108, 50)
    card_background: Color = (240, 237, 228)


@dataclass(slots=True)
class DebugConfig:
    vision_toggle_label: str = "V"
    show_debug_vision_by_default: bool = True


@dataclass(slots=True)
class SimConfig:
    display: DisplayConfig = field(default_factory=DisplayConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


def build_sim_config() -> SimConfig:
    return SimConfig()
