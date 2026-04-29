from __future__ import annotations

import arcade

from configs.sim_config import SimConfig, build_sim_config
from src.rendering import EnvironmentRenderer
from src.world import World


class NeatGameView(arcade.View):
    def __init__(self, config: SimConfig | None = None) -> None:
        super().__init__()
        self.config = config or build_sim_config()
        self.world = World(self.config)
        self.environment_renderer = EnvironmentRenderer(self.config)
        self.background_color = self.config.theme.window_background

    def on_show_view(self) -> None:
        if self.window is not None:
            self.window.background_color = self.config.theme.window_background

    def on_resize(self, width: int, height: int) -> bool | None:
        self.world.resize(width, height)
        return super().on_resize(width, height)

    def on_update(self, delta_time: float) -> None:
        self.world.update(delta_time)

    def on_draw(self) -> None:
        self.clear()
        self.environment_renderer.draw(self.world)
        self.ui_renderer.draw(self.world)

    def on_mouse_press(
        self, x: int, y: int, button: int, modifiers: int
    ) -> bool | None:
        self.world.select_creature_at(x, y)
        return super().on_mouse_press(x, y, button, modifiers)

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.V:
            self.world.toggle_debug_vision()
        return super().on_key_press(symbol, modifiers)


def create_and_run(config: SimConfig | None = None) -> None:
    active_config = config or build_sim_config()
    window = arcade.Window(
        active_config.display.width,
        active_config.display.height,
        active_config.display.title,
        resizable=active_config.display.resizable,
    )
    window.show_view(NeatGameView(active_config))
    arcade.run()
