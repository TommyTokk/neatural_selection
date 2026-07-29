from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, sqrt
from types import ModuleType, SimpleNamespace
import sys
import unittest

try:
    import pymunk  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pymunk"] = ModuleType("pymunk")

from configs.sim_config import MetabolismConfig, TraitConfig
from src.metabolism import Metabolism
from src.food import Food, FoodConsumptionResult


@dataclass(slots=True)
class FakeCreature:
    position: tuple[float, float]
    radius: float
    heading: float
    energy: float = 0.5
    stomach_energy: float = 0.0
    stomach_difficulty_load: float = 0.0
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
    consume_energy_calls: int = 0

    def consume_energy(
        self,
        requested_energy: float,
        min_remainder_ratio: float,
    ) -> FoodConsumptionResult:
        self.consume_energy_calls += 1
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

    def test_eat_fills_stomach_and_leaves_active_energy_unchanged(self) -> None:
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

        consumption = metabolism.eat(creature, food, delta_time=0.4)

        self.assertAlmostEqual(creature.energy, 0.8)
        self.assertAlmostEqual(creature.stomach_energy, 0.2)
        self.assertAlmostEqual(consumption.energy_swallowed, 0.2)
        self.assertFalse(consumption.depleted)
        self.assertAlmostEqual(food.energy_value, 0.3)

    def test_eat_depletes_only_when_bite_reaches_remaining_food(self) -> None:
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

        consumption = metabolism.eat(creature, food, delta_time=1.0)

        self.assertAlmostEqual(creature.energy, 0.55)
        self.assertAlmostEqual(creature.stomach_energy, 0.5)
        self.assertAlmostEqual(consumption.energy_swallowed, 0.5)
        self.assertTrue(consumption.depleted)
        self.assertEqual(food.energy_value, 0.0)

    def test_eat_does_nothing_when_stomach_is_full(self) -> None:
        metabolism = Metabolism(
            MetabolismConfig(max_energy=1.0, micro_food_remainder_ratio=0.10),
            vision=None,
        )
        creature = FakeCreature(
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
            energy=1.0,
            stomach_energy=1.0,
        )
        food = FakeFood(id=1, position=(13.0, 0.0), radius=3.0, energy_value=0.5)

        consumption = metabolism.eat(creature, food, delta_time=1.0)

        self.assertEqual(creature.energy, 1.0)
        self.assertEqual(creature.stomach_energy, 1.0)
        self.assertEqual(consumption.energy_swallowed, 0.0)
        self.assertFalse(consumption.depleted)
        self.assertEqual(food.energy_value, 0.5)
        self.assertEqual(food.consume_energy_calls, 0)

    def test_bite_is_limited_by_remaining_stomach_capacity(self) -> None:
        metabolism = Metabolism(
            MetabolismConfig(
                stomach_capacity_per_radius=0.1,
                max_bite_size_per_second=0.5,
            ),
            vision=None,
        )
        creature = FakeCreature(
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
            stomach_energy=0.95,
        )
        food = FakeFood(id=1, position=(13.0, 0.0), radius=3.0, energy_value=0.5)

        consumption = metabolism.eat(creature, food, delta_time=1.0)

        self.assertAlmostEqual(consumption.energy_swallowed, 0.05)
        self.assertAlmostEqual(creature.stomach_energy, 1.0)
        self.assertAlmostEqual(food.energy_value, 0.45)

    def test_inherited_stomach_capacity_is_independent_of_radius(self) -> None:
        metabolism = Metabolism(
            MetabolismConfig(max_bite_size_per_second=1.0),
            vision=None,
        )
        creature = FakeCreature(
            position=(0.0, 0.0),
            radius=22.0,
            heading=0.0,
            stomach_energy=0.75,
            physical_traits=SimpleNamespace(
                movement_cost_multiplier=1.0,
                stomach_capacity=0.8,
            ),
        )
        food = FakeFood(
            id=1,
            position=(25.0, 0.0),
            radius=3.0,
            energy_value=0.5,
        )

        consumption = metabolism.eat(creature, food, delta_time=1.0)

        self.assertAlmostEqual(consumption.energy_swallowed, 0.05)
        self.assertAlmostEqual(creature.stomach_energy, 0.8)

    def test_small_and_large_pellets_add_weighted_difficulty(self) -> None:
        metabolism = Metabolism(
            MetabolismConfig(max_bite_size_per_second=1.0),
            vision=None,
        )
        creature = FakeCreature(
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
            physical_traits=SimpleNamespace(
                movement_cost_multiplier=1.0,
                stomach_capacity=2.0,
            ),
        )
        small = FakeFood(
            id=1,
            position=(13.0, 0.0),
            radius=6.0,
            energy_value=0.2,
            original_energy_value=0.2,
        )
        large = FakeFood(
            id=2,
            position=(13.0, 0.0),
            radius=10.0,
            energy_value=0.2,
            original_energy_value=0.2,
        )

        metabolism.eat(creature, small, delta_time=1.0)
        metabolism.eat(creature, large, delta_time=1.0)

        self.assertAlmostEqual(creature.stomach_energy, 0.4)
        self.assertAlmostEqual(
            creature.stomach_difficulty_load,
            0.2 * 0.75 + 0.2 * 1.25,
        )

    def test_final_bite_tolerance_is_bounded_and_conserves_energy(self) -> None:
        metabolism = Metabolism(
            MetabolismConfig(
                max_bite_size_per_second=0.5,
                micro_food_remainder_ratio=0.10,
            ),
            vision=None,
        )
        creature = FakeCreature(
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
        )
        within_tolerance = FakeFood(
            id=1,
            position=(13.0, 0.0),
            radius=3.0,
            energy_value=0.54,
            original_energy_value=0.54,
        )

        consumption = metabolism.eat(
            creature,
            within_tolerance,
            delta_time=1.0,
        )

        self.assertAlmostEqual(consumption.energy_swallowed, 0.54)
        self.assertEqual(within_tolerance.energy_value, 0.0)
        self.assertTrue(consumption.depleted)

    def test_microscopic_food_remainder_is_swallowed_within_tolerance(self) -> None:
        metabolism = Metabolism(
            MetabolismConfig(max_bite_size_per_second=0.1),
            vision=None,
        )
        creature = FakeCreature(
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
        )
        food = FakeFood(
            id=1,
            position=(13.0, 0.0),
            radius=3.0,
            energy_value=0.105,
            original_energy_value=0.105,
        )

        consumption = metabolism.eat(creature, food, delta_time=1.0)

        self.assertAlmostEqual(consumption.energy_swallowed, 0.105)
        self.assertAlmostEqual(creature.stomach_energy, 0.105)
        self.assertTrue(consumption.depleted)
        self.assertEqual(food.energy_value, 0.0)

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
            delta_time=1.0,
            max_speed=1.0,
        )

        self.assertAlmostEqual(first_creature.energy, second_creature.energy)
        self.assertAlmostEqual(first_creature.stomach_energy, 0.5)
        self.assertEqual(second_creature.stomach_energy, 0.0)
        self.assertEqual(report.touched_foods, [food])
        self.assertEqual(report.depleted_foods, [])
        self.assertEqual(len(report.food_consumptions), 1)

    def test_update_skips_food_lookup_when_eating_is_gated(self) -> None:
        metabolism = Metabolism(
            MetabolismConfig(
                max_energy=1.0,
                basic_metabolism_rate=0.0,
                movement_energy_cost_factor=0.0,
            ),
            vision=SimpleNamespace(energy_cost_per_second=lambda creature: 0.0),
            trait_config=TraitConfig(body_metabolism_cost_factor=0.0),
        )
        creature = FakeCreature(
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
            stomach_energy=1.0,
        )
        food = FakeFood(id=1, position=(13.0, 0.0), radius=3.0)
        lookup_calls: list[int] = []

        report = metabolism.update(
            [creature],
            [food],
            delta_time=1.0,
            max_speed=1.0,
            nearby_foods_for=lambda candidate: (
                lookup_calls.append(candidate.creature_id) or [food]
            ),
            can_eat=lambda candidate: False,
        )

        self.assertEqual(lookup_calls, [])
        self.assertEqual(report.food_consumptions, [])
        self.assertEqual(food.consume_energy_calls, 0)

    def test_digest_applies_efficiency_and_processing_cost(self) -> None:
        metabolism = Metabolism(
            MetabolismConfig(
                max_energy=1.0,
                digestion_rate_per_second=0.2,
                digestion_efficiency=0.9,
            ),
            vision=None,
        )
        creature = FakeCreature(
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
            energy=0.0,
            stomach_energy=1.0,
        )

        gained = metabolism.digest(creature, delta_time=1.0)

        self.assertAlmostEqual(creature.stomach_energy, 0.8)
        self.assertAlmostEqual(creature.energy, 0.1602)
        self.assertAlmostEqual(gained, 0.1602)

    def test_digest_reports_only_energy_admitted_below_cap(self) -> None:
        metabolism = Metabolism(
            MetabolismConfig(
                max_energy=1.0,
                digestion_rate_per_second=0.2,
                digestion_efficiency=0.9,
            ),
            vision=None,
        )
        creature = FakeCreature(
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
            energy=0.95,
            stomach_energy=1.0,
        )

        gained = metabolism.digest(creature, delta_time=1.0)

        self.assertAlmostEqual(creature.stomach_energy, 0.9375780274656679)
        self.assertAlmostEqual(creature.energy, 1.0)
        self.assertAlmostEqual(gained, 0.05)

    def test_partial_bite_conserves_swallowed_energy_and_shrinks_food(self) -> None:
        metabolism = Metabolism(
            MetabolismConfig(
                stomach_capacity_per_radius=0.1,
                max_bite_size_per_second=0.1,
            ),
            vision=None,
        )
        creature = FakeCreature(
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
        )
        density = 0.4 / pi
        food = Food(id=1, x=13.0, y=0.0, radius=1.0, energy_density=density)

        consumption = metabolism.eat(creature, food, delta_time=1.0)

        self.assertAlmostEqual(consumption.energy_swallowed, 0.1)
        self.assertAlmostEqual(creature.stomach_energy, 0.1)
        self.assertAlmostEqual(food.energy_value, 0.3)
        self.assertAlmostEqual(food.radius, sqrt(0.3 / (pi * density)))
        self.assertAlmostEqual(food.shape.radius, food.radius)

    def test_update_uses_simultaneous_digestion_and_upkeep_ledger(self) -> None:
        metabolism = Metabolism(
            MetabolismConfig(
                max_energy=1.0,
                basic_metabolism_rate=0.1,
                movement_energy_cost_factor=0.0,
                sprint_energy_cost_per_second=0.0,
                digestion_rate_per_second=0.2,
                digestion_efficiency=0.9,
            ),
            vision=SimpleNamespace(energy_cost_per_second=lambda creature: 0.0),
            trait_config=TraitConfig(body_metabolism_cost_factor=0.0),
        )
        creature = FakeCreature(
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
            energy=0.0,
            stomach_energy=0.2,
        )

        report = metabolism.update(
            [creature],
            [],
            delta_time=1.0,
            max_speed=1.0,
        )

        self.assertAlmostEqual(creature.energy, 0.0562)
        self.assertAlmostEqual(creature.stomach_energy, 0.0)
        self.assertAlmostEqual(report.digested_energy_gained[1], 0.1602)
        self.assertAlmostEqual(report.digestion_processing_costs[1], 0.0198)

    def test_full_energy_digests_nothing_without_upkeep(self) -> None:
        metabolism = Metabolism(
            MetabolismConfig(),
            vision=None,
        )
        creature = FakeCreature(
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
            energy=1.0,
            stomach_energy=1.0,
            stomach_difficulty_load=1.0,
        )

        gained = metabolism.digest(creature, delta_time=1.0)

        self.assertEqual(gained, 0.0)
        self.assertEqual(creature.energy, 1.0)
        self.assertEqual(creature.stomach_energy, 1.0)
        self.assertEqual(creature.stomach_difficulty_load, 1.0)

    def test_small_food_costs_less_to_digest_than_large_food(self) -> None:
        metabolism = Metabolism(MetabolismConfig(), vision=None)
        small = FakeCreature(
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
            energy=0.0,
            stomach_energy=0.2,
            stomach_difficulty_load=0.2 * 0.75,
        )
        large = FakeCreature(
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
            energy=0.0,
            stomach_energy=0.2,
            stomach_difficulty_load=0.2 * 1.25,
        )

        small_gain = metabolism.digest(small, 1.0)
        large_gain = metabolism.digest(large, 1.0)

        self.assertGreater(small_gain, large_gain)

    def test_faster_digestion_has_higher_cost_and_faster_availability(
        self,
    ) -> None:
        metabolism = Metabolism(MetabolismConfig(), vision=None)
        slow = FakeCreature(
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
            energy=0.0,
            stomach_energy=1.0,
            stomach_difficulty_load=1.0,
            physical_traits=SimpleNamespace(
                movement_cost_multiplier=1.0,
                stomach_capacity=1.6,
                digestion_rate=0.1,
                digestion_efficiency=0.9,
            ),
        )
        fast = FakeCreature(
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
            energy=0.0,
            stomach_energy=1.0,
            stomach_difficulty_load=1.0,
            physical_traits=SimpleNamespace(
                movement_cost_multiplier=1.0,
                stomach_capacity=1.6,
                digestion_rate=0.4,
                digestion_efficiency=0.9,
            ),
        )

        slow_gain = metabolism.digest(slow, delta_time=1.0)
        fast_gain = metabolism.digest(fast, delta_time=1.0)

        self.assertGreater(fast_gain, slow_gain)
        self.assertGreater(
            metabolism.digestion_processing_fraction(0.4, 1.0),
            metabolism.digestion_processing_fraction(0.1, 1.0),
        )

    def test_digestion_is_timestep_equivalent_without_energy_cap(self) -> None:
        metabolism = Metabolism(MetabolismConfig(), vision=None)
        one_step = FakeCreature(
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
            energy=0.0,
            stomach_energy=1.0,
            stomach_difficulty_load=1.0,
        )
        sixty_steps = FakeCreature(
            position=(0.0, 0.0),
            radius=10.0,
            heading=0.0,
            energy=0.0,
            stomach_energy=1.0,
            stomach_difficulty_load=1.0,
        )

        metabolism.digest(one_step, 1.0)
        for _ in range(60):
            metabolism.digest(sixty_steps, 1.0 / 60.0)

        self.assertAlmostEqual(one_step.energy, sixty_steps.energy)
        self.assertAlmostEqual(
            one_step.stomach_energy,
            sixty_steps.stomach_energy,
        )
        self.assertAlmostEqual(
            one_step.stomach_difficulty_load,
            sixty_steps.stomach_difficulty_load,
        )


if __name__ == "__main__":
    unittest.main()
