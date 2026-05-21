from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
import sys
import unittest

if "pymunk" not in sys.modules:
    sys.modules["pymunk"] = ModuleType("pymunk")

from configs.sim_config import MetabolismConfig
from src.metabolism import Metabolism


@dataclass(slots=True)
class FakeCreature:
    position: tuple[float, float]
    radius: float
    heading: float
    energy: float = 0.5


@dataclass(slots=True)
class FakeFood:
    id: int
    position: tuple[float, float]
    radius: float
    energy_value: float = 0.1


class MetabolismMouthEatingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metabolism = Metabolism(
            MetabolismConfig(eating_distance=4.0),
            vision=None,
        )

    def test_food_in_front_mouth_is_eatable(self) -> None:
        creature = FakeCreature(position=(0.0, 0.0), radius=10.0, heading=0.0)
        food = FakeFood(id=1, position=(13.0, 0.0), radius=3.0)

        self.assertIs(
            self.metabolism.find_eatable_food(creature, [food], []),
            food,
        )

    def test_side_body_overlap_is_not_eatable(self) -> None:
        creature = FakeCreature(position=(0.0, 0.0), radius=10.0, heading=0.0)
        food = FakeFood(id=1, position=(0.0, 10.0), radius=3.0)

        self.assertIsNone(
            self.metabolism.find_eatable_food(creature, [food], []),
        )

    def test_back_body_overlap_is_not_eatable(self) -> None:
        creature = FakeCreature(position=(0.0, 0.0), radius=10.0, heading=0.0)
        food = FakeFood(id=1, position=(-10.0, 0.0), radius=3.0)

        self.assertIsNone(
            self.metabolism.find_eatable_food(creature, [food], []),
        )

    def test_food_in_front_without_contact_is_not_eatable(self) -> None:
        creature = FakeCreature(position=(0.0, 0.0), radius=10.0, heading=0.0)
        food = FakeFood(id=1, position=(17.0, 0.0), radius=3.0)

        self.assertIsNone(
            self.metabolism.find_eatable_food(creature, [food], []),
        )

    def test_front_side_body_contact_is_not_eatable(self) -> None:
        creature = FakeCreature(position=(0.0, 0.0), radius=10.0, heading=0.0)
        food = FakeFood(id=1, position=(10.0, 8.0), radius=3.0)

        self.assertIsNone(
            self.metabolism.find_eatable_food(creature, [food], []),
        )

    def test_ignored_food_is_not_eatable_even_when_in_mouth(self) -> None:
        creature = FakeCreature(position=(0.0, 0.0), radius=10.0, heading=0.0)
        food = FakeFood(id=1, position=(13.0, 0.0), radius=3.0)

        self.assertIsNone(
            self.metabolism.find_eatable_food(creature, [food], [food]),
        )


if __name__ == "__main__":
    unittest.main()
