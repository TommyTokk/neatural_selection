from __future__ import annotations

from dataclasses import dataclass

from configs.sim_config import LayoutConfig

import arcade


@dataclass(slots=True)
class ScreenLayout:
    window: arcade.Rect
    top_bar: arcade.Rect
    environment: arcade.Rect
    right_sidebar: arcade.Rect
    bottom_bar: arcade.Rect


def build_screen_layout(
    window_width: int, window_height: int, config: LayoutConfig
) -> ScreenLayout:
    pad = float(config.outer_padding)
    gap = float(config.panel_gap)

    usable_left = pad
    usable_right = max(pad, window_width - pad)
    usable_bottom = pad
    usable_top = max(pad, window_height - pad)

    usable_width = usable_right - usable_left
    usable_height = usable_top - usable_bottom

    top_height = min(float(config.top_bar_height), max(64.0, usable_height * 0.16))
    bottom_height = min(
        float(config.bottom_bar_height), max(100.0, usable_height * 0.2)
    )
    top_bar = arcade.LBWH(
        usable_left, usable_top - top_height, usable_width, top_height
    )
    bottom_bar = arcade.LBWH(usable_left, usable_bottom, usable_width, bottom_height)
    content_bottom = bottom_bar.top + gap
    content_top = top_bar.bottom - gap
    content_height = max(100.0, content_top - content_bottom)

    sidebar_width = min(
        float(config.right_panel_width),
        max(float(config.min_sidebar_width), usable_width * 0.28),
    )
    environment_width = usable_width - sidebar_width - gap

    if environment_width < config.min_environment_width:
        sidebar_width = max(
            float(config.min_sidebar_width),
            usable_width - gap - float(config.min_environment_width),
        )
        environment_width = max(280.0, usable_width - sidebar_width - gap)

    environment = arcade.LBWH(
        usable_left, content_bottom, environment_width, content_height
    )
    right_sidebar = arcade.LBWH(
        environment.right + gap, content_bottom, sidebar_width, content_height
    )
    window = arcade.LBWH(0, 0, window_width, window_height)

    return ScreenLayout(
        window=window,
        top_bar=top_bar,
        environment=environment,
        right_sidebar=right_sidebar,
        bottom_bar=bottom_bar,
    )
