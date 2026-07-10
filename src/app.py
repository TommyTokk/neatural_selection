from __future__ import annotations

from src.graphics import configure_graphics, log_graphics_context

configure_graphics()

import arcade

from configs.sim_config import SimConfig, build_sim_config
from src.menu import StartMenuView
from src.persistence import PersistenceManager
from src.rendering import EnvironmentRenderer
from src.ui import UiRenderer
from src.world import World


class NeatGameView(arcade.View):
    def __init__(
        self,
        config: SimConfig | None = None,
        *,
        world: World | None = None,
    ) -> None:
        super().__init__()
        if world is None:
            self.config = config or build_sim_config()
            self.world = World(self.config)
        else:
            self.world = world
            self.config = world.config
        self.environment_renderer = EnvironmentRenderer(self.config)
        self.background_color = self.config.theme.window_background
        self.ui_renderer = UiRenderer(self.config)
        self._is_dragging_environment = False
        self._is_dragging_ui_control = False
        self._drag_distance = 0.0
        self._command_keys_down: set[int] = set()

    def on_show_view(self) -> None:
        if self.window is not None:
            self.window.background_color = self.config.theme.window_background

    def on_hide_view(self) -> None:
        self.ui_renderer.close()
        self.world.close()

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
        if button == arcade.MOUSE_BUTTON_LEFT and self.ui_renderer.handle_mouse_press(
            self.world, x, y
        ):
            self._is_dragging_ui_control = True
            self._is_dragging_environment = False
            self._drag_distance = 0.0
            return super().on_mouse_press(x, y, button, modifiers)

        env = self.world.layout.environment
        if button == arcade.MOUSE_BUTTON_LEFT and env.left <= x <= env.right and env.bottom <= y <= env.top:
            self._is_dragging_environment = True
            self._drag_distance = 0.0
        return super().on_mouse_press(x, y, button, modifiers)

    def on_mouse_release(
        self, x: int, y: int, button: int, modifiers: int
    ) -> bool | None:
        if button == arcade.MOUSE_BUTTON_LEFT and self._is_dragging_ui_control:
            self.ui_renderer.handle_mouse_release()
            self._is_dragging_ui_control = False
            return super().on_mouse_release(x, y, button, modifiers)

        if button == arcade.MOUSE_BUTTON_LEFT and self._is_dragging_environment:
            if self._drag_distance < 5.0:
                self.world.select_creature_at(x, y)
            self._is_dragging_environment = False
        return super().on_mouse_release(x, y, button, modifiers)

    def on_mouse_drag(
        self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int
    ) -> bool | None:
        if self._is_dragging_ui_control and (buttons & arcade.MOUSE_BUTTON_LEFT):
            self.ui_renderer.handle_mouse_drag(self.world, x, y)
            return super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)

        if self._is_dragging_environment and (buttons & arcade.MOUSE_BUTTON_LEFT):
            self.world.pan_environment(dx, dy)
            self._drag_distance += abs(dx) + abs(dy)
        return super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)

    def on_mouse_scroll(
        self, x: int, y: int, scroll_x: int, scroll_y: int
    ) -> bool | None:
        if self.ui_renderer.handle_mouse_scroll(
            x,
            y,
            scroll_y,
            scroll_x,
            bool(self._command_keys_down),
        ):
            return super().on_mouse_scroll(x, y, scroll_x, scroll_y)

        if self.world.layout.environment.left <= x <= self.world.layout.environment.right and self.world.layout.environment.bottom <= y <= self.world.layout.environment.top:
            self.world.adjust_environment_zoom(scroll_y)
        return super().on_mouse_scroll(x, y, scroll_x, scroll_y)

    def on_mouse_motion(
        self, x: int, y: int, dx: int, dy: int
    ) -> bool | None:
        self.ui_renderer.handle_mouse_motion(self.world, x, y, dx, dy)
        return super().on_mouse_motion(x, y, dx, dy)

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol in self._command_key_symbols():
            self._command_keys_down.add(symbol)
        if self.ui_renderer.handle_key_press(self.world, symbol, modifiers):
            return super().on_key_press(symbol, modifiers)
        if symbol == arcade.key.SPACE:
            self.world.toggle_pause()
        if symbol == arcade.key.V:
            self.world.toggle_debug_vision()
        if symbol == arcade.key.D or symbol == arcade.key.RIGHT:
            self.world.increase_simulation_speed()
        if symbol == arcade.key.A or symbol == arcade.key.LEFT:
            self.world.decrease_simulation_speed()
        if symbol == getattr(arcade.key, "_0", -1) or symbol == getattr(
            arcade.key, "NUM_0", -2
        ) or symbol == ord("0"):
            self.world.reset_simulation_speed()
        if symbol == arcade.key.EQUAL:
            self.world.adjust_environment_zoom(1)
        if symbol == arcade.key.MINUS:
            self.world.adjust_environment_zoom(-1)
        if symbol == arcade.key.R:
            self.world.reset_environment_view()
        return super().on_key_press(symbol, modifiers)

    def on_key_release(self, symbol: int, modifiers: int) -> bool | None:
        self._command_keys_down.discard(symbol)
        return super().on_key_release(symbol, modifiers)

    @staticmethod
    def _command_key_symbols() -> tuple[int, ...]:
        return tuple(
            key
            for key in (
                getattr(arcade.key, "LCOMMAND", None),
                getattr(arcade.key, "RCOMMAND", None),
            )
            if key is not None
        )


def create_and_run(config: SimConfig | None = None) -> None:
    active_config = config or build_sim_config()
    window = arcade.Window(
        active_config.display.width,
        active_config.display.height,
        active_config.display.title,
        resizable=active_config.display.resizable,
        gl_version=active_config.display.gl_version,
        antialiasing=active_config.display.antialiasing,
        vsync=active_config.display.vsync,
    )
    log_graphics_context()
    window.show_view(
        StartMenuView(
            active_config,
            lambda: NeatGameView(active_config),
            lambda checkpoint: NeatGameView(
                world=PersistenceManager.load_checkpoint(
                    active_config,
                    checkpoint,
                )
            ),
        )
    )
    arcade.run()
