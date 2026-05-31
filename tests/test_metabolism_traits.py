from __future__ import annotations

from dataclasses import dataclass
import unittest

from configs.sim_config import MetabolismConfig, TraitConfig
from src.creature import PhysicalTraits
from src.metabolism import Metabolism


@dataclass(slots=True)
class FakeCreature:
    radius: float
    speed: float
    energy: float
    physical_traits: PhysicalTraits


class FakeVision:
    def __init__(self, cost: float) -> None:
        self.cost = cost

    def energy_cost_per_second(self, creature: FakeCreature) -> float:
        del creature
        return self.cost


class MetabolismTraitCostTest(unittest.TestCase):
    def test_large_body_and_movement_multiplier_increase_energy_cost(self) -> None:
        metabolism = Metabolism(
            MetabolismConfig(
                basic_metabolism_rate=0.0,
                movement_energy_cost_factor=0.02,
            ),
            FakeVision(cost=0.01),
            TraitConfig(
                max_radius=20.0,
                body_metabolism_cost_factor=0.008,
            ),
        )
        efficient_creature = FakeCreature(
            radius=10.0,
            speed=50.0,
            energy=1.0,
            physical_traits=PhysicalTraits(
                radius=10.0,
                movement_cost_multiplier=1.0,
            ),
        )
        costly_creature = FakeCreature(
            radius=20.0,
            speed=50.0,
            energy=1.0,
            physical_traits=PhysicalTraits(
                radius=20.0,
                movement_cost_multiplier=1.5,
            ),
        )

        efficient_cost = metabolism.energy_cost_breakdown_per_second(
            efficient_creature,
            max_speed=100.0,
        )
        costly_cost = metabolism.energy_cost_breakdown_per_second(
            costly_creature,
            max_speed=100.0,
        )

        self.assertGreater(costly_cost.total, efficient_cost.total)
        self.assertGreater(costly_cost.trait, efficient_cost.trait)


if __name__ == "__main__":
    unittest.main()
