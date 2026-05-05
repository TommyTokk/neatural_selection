from __future__ import annotations

import arcade

from configs.sim_config import SimConfig, build_sim_config
from src.rendering import EnvironmentRenderer
from src.ui import UiRenderer
from src.world import World


class NeatGameView(arcade.View):
    def __init__(self, config: SimConfig | None = None) -> None:
        super().__init__()
        self.config = config or build_sim_config()
        self.world = World(self.config)
        self.environment_renderer = EnvironmentRenderer(self.config)
        self.background_color = self.config.theme.window_background
        self.ui_renderer = UiRenderer(self.config)
        self._is_dragging_environment = False
        self._drag_distance = 0.0

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
        env = self.world.layout.environment
        if button == arcade.MOUSE_BUTTON_LEFT and env.left <= x <= env.right and env.bottom <= y <= env.top:
            self._is_dragging_environment = True
            self._drag_distance = 0.0
        return super().on_mouse_press(x, y, button, modifiers)

    def on_mouse_release(
        self, x: int, y: int, button: int, modifiers: int
    ) -> bool | None:
        if button == arcade.MOUSE_BUTTON_LEFT and self._is_dragging_environment:
            if self._drag_distance < 5.0:
                self.world.select_creature_at(x, y)
            self._is_dragging_environment = False
        return super().on_mouse_release(x, y, button, modifiers)

    def on_mouse_drag(
        self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int
    ) -> bool | None:
        if self._is_dragging_environment and (buttons & arcade.MOUSE_BUTTON_LEFT):
            self.world.pan_environment(dx, dy)
            self._drag_distance += abs(dx) + abs(dy)
        return super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)

    def on_mouse_scroll(
        self, x: int, y: int, scroll_x: int, scroll_y: int
    ) -> bool | None:
        if self.world.layout.environment.left <= x <= self.world.layout.environment.right and self.world.layout.environment.bottom <= y <= self.world.layout.environment.top:
            self.world.adjust_environment_zoom(scroll_y)
        return super().on_mouse_scroll(x, y, scroll_x, scroll_y)

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.D:
            self.world.toggle_debug_vision()
        if symbol == arcade.key.EQUAL:
            self.world.adjust_environment_zoom(1)
        if symbol == arcade.key.MINUS:
            self.world.adjust_environment_zoom(-1)
        if symbol == arcade.key.R:
            self.world.environment_pan_x = 0.0
            self.world.environment_pan_y = 0.0
            self.world.environment_zoom = self.config.zoom.default
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
