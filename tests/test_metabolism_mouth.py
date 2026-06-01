from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
import sys
import unittest

try:
    import pymunk  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pymunk"] = ModuleType("pymunk")

from configs.sim_config import MetabolismConfig
from src.metabolism import Metabolism
from src.food import FoodConsumptionResult


@dataclass(slots=True)
class FakeCreature:
    position: tuple[float, float]
    radius: float
    heading: float
    energy: float = 0.5
    creature_id: int = 1
    speed: float = 0.0
    physical_traits: object = field(
        default_factory=lambda: SimpleNamespace(movement_cost_multiplier=1.0)
    )


@dataclass(slots=True)
class FakeFood:
    id: int
    position: tuple[float, float]
    radius: float
    energy_value: float = 0.1
    original_energy_value: float = 0.1

    def consume_energy(
        self,
        requested_energy: float,
        min_remainder_ratio: float,
    ) -> FoodConsumptionResult:
        if requested_energy <= 0.0:
            return FoodConsumptionResult(energy_removed=0.0, depleted=False)

        remaining_energy = self.energy_value - requested_energy
        minimum_remainder = self.original_energy_value * min_remainder_ratio
        previous_energy = self.energy_value
        if remaining_energy <= minimum_remainder + 1e-12:
            self.energy_value = 0.0
            return FoodConsumptionResult(
                energy_removed=previous_energy,
                depleted=True,
            )

        self.energy_value = remaining_energy
        return FoodConsumptionResult(
            energy_removed=requested_energy,
            depleted=False,
        )


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

    def test_eat_tops_creature_to_max_and_leaves_remaining_food(self) -> None:
        metabolism = Metabolism(
            MetabolismConfig(max_energy=1.0, micro_food_remainder_ratio=0.10),
            vision=None,
        )
        creature = FakeCreature(
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
            energy=0.8,
        )
        food = FakeFood(
            id=1,
            position=(13.0, 0.0),
            radius=3.0,
            energy_value=0.5,
            original_energy_value=0.5,
        )

        consumption = metabolism.eat(creature, food)

        self.assertAlmostEqual(creature.energy, 1.0)
        self.assertAlmostEqual(consumption.energy_gained, 0.2)
        self.assertFalse(consumption.depleted)
        self.assertAlmostEqual(food.energy_value, 0.3)

    def test_eat_depletes_micro_remainder_without_overfilling_creature(self) -> None:
        metabolism = Metabolism(
            MetabolismConfig(max_energy=1.0, micro_food_remainder_ratio=0.10),
            vision=None,
        )
        creature = FakeCreature(
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
            energy=0.55,
        )
        food = FakeFood(
            id=1,
            position=(13.0, 0.0),
            radius=3.0,
            energy_value=0.5,
            original_energy_value=0.5,
        )

        consumption = metabolism.eat(creature, food)

        self.assertAlmostEqual(creature.energy, 1.0)
        self.assertAlmostEqual(consumption.energy_gained, 0.45)
        self.assertTrue(consumption.depleted)
        self.assertEqual(food.energy_value, 0.0)

    def test_eat_does_nothing_when_creature_is_full(self) -> None:
        metabolism = Metabolism(
            MetabolismConfig(max_energy=1.0, micro_food_remainder_ratio=0.10),
            vision=None,
        )
        creature = FakeCreature(
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
            energy=1.0,
        )
        food = FakeFood(id=1, position=(13.0, 0.0), radius=3.0, energy_value=0.5)

        consumption = metabolism.eat(creature, food)

        self.assertEqual(creature.energy, 1.0)
        self.assertEqual(consumption.energy_gained, 0.0)
        self.assertFalse(consumption.depleted)
        self.assertEqual(food.energy_value, 0.5)

    def test_update_allows_only_one_creature_to_bite_same_food_per_tick(self) -> None:
        metabolism = Metabolism(
            MetabolismConfig(
                max_energy=1.0,
                basic_metabolism_rate=0.0,
                movement_energy_cost_factor=0.0,
                micro_food_remainder_ratio=0.10,
            ),
            vision=SimpleNamespace(energy_cost_per_second=lambda creature: 0.0),
        )
        first_creature = FakeCreature(
            creature_id=1,
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
            energy=0.5,
        )
        second_creature = FakeCreature(
            creature_id=2,
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
            energy=0.5,
        )
        food = FakeFood(
            id=1,
            position=(13.0, 0.0),
            radius=3.0,
            energy_value=0.8,
            original_energy_value=0.8,
        )

        report = metabolism.update(
            [first_creature, second_creature],
            [food],
            delta_time=0.0,
            max_speed=1.0,
        )

        self.assertAlmostEqual(first_creature.energy, 1.0)
        self.assertAlmostEqual(second_creature.energy, 0.5)
        self.assertEqual(report.touched_foods, [food])
        self.assertEqual(report.depleted_foods, [])
        self.assertEqual(len(report.food_consumptions), 1)


if __name__ == "__main__":
    unittest.main()
