from __future__ import annotations

import sys
from types import ModuleType
import unittest

for optional_module in ("arcade", "neat", "pymunk"):
    if optional_module not in sys.modules:
        sys.modules[optional_module] = ModuleType(optional_module)

from src.world import World


class WorldSimulationSpeedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world = object.__new__(World)
        self.world.simulation_speed = 1.0

    def test_set_simulation_speed_clamps_to_five_x(self) -> None:
        self.world.set_simulation_speed(999.0)

        self.assertEqual(self.world.simulation_speed, 5.0)

    def test_repeated_speed_up_stops_at_five_x(self) -> None:
        for _ in range(100):
            self.world.increase_simulation_speed()

        self.assertEqual(self.world.simulation_speed, 5.0)

    def test_reset_returns_to_one_x(self) -> None:
        self.world.set_simulation_speed(5.0)

        self.world.reset_simulation_speed()

        self.assertEqual(self.world.simulation_speed, 1.0)

    def test_speed_rounds_to_quarter_step(self) -> None:
        self.world.set_simulation_speed(1.37)

        self.assertEqual(self.world.simulation_speed, 1.25)


if __name__ == "__main__":
    unittest.main()
