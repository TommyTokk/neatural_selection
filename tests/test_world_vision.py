from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
import sys
import unittest

if "arcade" not in sys.modules:
    arcade = ModuleType("arcade")

    @dataclass(slots=True)
    class FakeRect:
        left: float
        bottom: float
        width: float
        height: float

        @property
        def right(self) -> float:
            return self.left + self.width

        @property
        def top(self) -> float:
            return self.bottom + self.height

    def fake_lbwh(left: float, bottom: float, width: float, height: float) -> FakeRect:
        return FakeRect(left, bottom, width, height)

    arcade.Rect = FakeRect
    arcade.LBWH = fake_lbwh
    sys.modules["arcade"] = arcade

for optional_module in ("neat", "pymunk"):
    try:
        __import__(optional_module)
    except ModuleNotFoundError:
        sys.modules[optional_module] = ModuleType(optional_module)

from configs.sim_config import build_sim_config
from src.creature import VisionTraits
from src.world import World


class FakeRng:
    def __init__(self, gaussian_values: list[float]) -> None:
        self.gaussian_values = gaussian_values

    def gauss(self, mean: float, deviation: float) -> float:
        del mean, deviation
        return self.gaussian_values.pop(0)


class WorldVisionMutationTest(unittest.TestCase):
    def make_world_with_mutations(self, mutations: list[float]) -> World:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.rng = FakeRng(mutations)
        return world

    def test_mutated_vision_inherits_parent_values_with_small_variation(self) -> None:
        world = self.make_world_with_mutations([6.0, -0.05])

        child_vision = world._mutated_vision(
            VisionTraits(
                range=120.0,
                angle=0.80,
            )
        )

        self.assertAlmostEqual(child_vision.range, 126.0)
        self.assertAlmostEqual(child_vision.angle, 0.75)

    def test_mutated_vision_clamps_to_config_bounds(self) -> None:
        world = self.make_world_with_mutations([-100.0, 100.0])

        child_vision = world._mutated_vision(
            VisionTraits(
                range=world.config.vision.min_range,
                angle=world.config.vision.max_angle,
            )
        )

        self.assertEqual(child_vision.range, world.config.vision.min_range)
        self.assertEqual(child_vision.angle, world.config.vision.max_angle)


if __name__ == "__main__":
    unittest.main()
