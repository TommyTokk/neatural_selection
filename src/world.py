from __future__ import annotations

from dataclasses import dataclass
from random import Random

from configs.sim_config import SimConfig
import src.utils as ut
from src.creature import Creature
from src.food import Food

from src.layout import ScreenLayout, build_screen_layout


@dataclass(slots=True)
class WorldStats:
    environment_name: str = "Herbivore Basin"
    generation_label: str = "Prototype Layout"
    herbivore_count: int = 0
    food_count: int = 0


class World:
    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.rng = Random(7)
        self.elapsed_time = 0.0
        self.debug_vision_enabled = config.debug.show_debug_vision_by_default
        self.layout = build_screen_layout(
            config.display.width, config.display.height, config.layout
        )
        self.environment_zoom = config.zoom.default
        self.environment_pan_x = 0.0
        self.environment_pan_y = 0.0
        # Temporary fixture so renderer paths are visible while NEAT logic is in progress.
        self.creatures = [
            Creature(
                creature_id=1,
                name="Demo Herbivore",
                anchor_x=0.55,
                anchor_y=0.52,
                drift_radius=0.01,
                drift_speed=0.2,
                heading_speed=0.1,
                energy=0.78,
                color=self.config.theme.herbivore_fill,
            )
        ]
        self.foods = [
            Food(
                x_ratio=0.59,
                y_ratio=0.54,
                radius=6.0,
            )
        ]
        self.selected_creature_id: int | None = None
        self.stats = WorldStats(
            herbivore_count=len(self.creatures),
            food_count=len(self.foods),
        )

    def resize(self, width: int, height: int) -> None:
        self.layout = build_screen_layout(width, height, self.config.layout)

    def update(self, delta_time: float) -> None:
        self.elapsed_time += delta_time

    def adjust_environment_zoom(self, scroll_y: float) -> None:
        if scroll_y == 0:
            return
        direction = 1 if scroll_y > 0 else -1
        factor = 1.0 + direction * self.config.zoom.step
        updated_zoom = self.environment_zoom * factor
        self.environment_zoom = max(
            self.config.zoom.minimum,
            min(self.config.zoom.maximum, updated_zoom),
        )

    def pan_environment(self, delta_x: float, delta_y: float) -> None:
        self.environment_pan_x += delta_x
        self.environment_pan_y += delta_y

    def screen_to_environment(self, x: float, y: float) -> tuple[float, float]:
        bounds = self.layout.environment
        center_x = bounds.center_x
        center_y = bounds.center_y
        model_x = center_x + (x - center_x - self.environment_pan_x) / self.environment_zoom
        model_y = center_y + (y - center_y - self.environment_pan_y) / self.environment_zoom
        return model_x, model_y

    @property
    def selected_creature(self) -> Creature | None:
        for creature in self.creatures:
            if creature.creature_id == self.selected_creature_id:
                return creature
        return None

    def toggle_debug_vision(self) -> None:
        self.debug_vision_enabled = not self.debug_vision_enabled

    def select_creature_at(self, x: float, y: float) -> None:
        environment = self.layout.environment
        if not ut.contains(environment, x, y):
            self.selected_creature_id = None
            return
        world_x, world_y = self.screen_to_environment(x, y)
        chosen: Creature | None = None
        for creature in reversed(self.creatures):
            if creature.contains_screen_point(
                world_x, world_y, environment, self.elapsed_time
            ):
                chosen = creature
                break
        self.selected_creature_id = None if chosen is None else chosen.creature_id
