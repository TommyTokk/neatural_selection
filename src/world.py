from __future__ import annotations

from dataclasses import dataclass
from random import Random

from configs.sim_config import SimConfig

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
        self.stats = WorldStats(
            herbivore_count=len(self.creatures),
            food_count=len(self.foods),
        )

    def resize(self, width: int, height: int) -> None:
        self.layout = build_screen_layout(width, height, self.config.layout)
    
    def update(self, delta_time: float) -> None:
        self.elapsed_time += delta_time

    def toggle_debug_vision(self) -> None:
        self.debug_vision_enabled = not self.debug_vision_enabled
