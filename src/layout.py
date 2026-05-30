from __future__ import annotations

from dataclasses import dataclass

from configs.sim_config import LayoutConfig

import arcade


@dataclass(slots=True)
class ScreenLayout:
    window: arcade.Rect
    top_bar: arcade.Rect
    environment: arcade.Rect
    left_sidebar: arcade.Rect


def build_screen_layout(
    window_width: int, window_height: int, config: LayoutConfig
) -> ScreenLayout:
    pad = float(config.outer_padding)
    icon_button_size = 58.0
    icon_button_gap = 20.0
    rail_vertical_padding = 32.0
    rail_width = max(float(config.min_sidebar_width), float(config.left_panel_width))
    rail_height = icon_button_size * 3 + icon_button_gap * 2 + rail_vertical_padding
    title_width = min(292.0, max(220.0, window_width - pad * 2.0))
    title_height = float(config.top_bar_height)

    top_bar = arcade.LBWH(
        pad,
        max(pad, window_height - pad - title_height),
        title_width,
        title_height,
    )
    left_sidebar = arcade.LBWH(
        pad,
        max(
            pad,
            min(
                window_height - pad - rail_height,
                window_height / 2.0 - rail_height / 2.0,
            ),
        ),
        rail_width,
        rail_height,
    )
    environment = arcade.LBWH(0, 0, window_width, window_height)
    window = arcade.LBWH(0, 0, window_width, window_height)

    return ScreenLayout(
        window=window,
        top_bar=top_bar,
        environment=environment,
        left_sidebar=left_sidebar,
    )
