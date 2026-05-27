from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
import sys
import unittest

if "arcade" not in sys.modules:
    arcade = ModuleType("arcade")
    arcade.Rect = object
    sys.modules["arcade"] = arcade


class _Body:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.position = SimpleNamespace(x=0.0, y=0.0)


class _Circle:
    def __init__(self, body: _Body, radius: float) -> None:
        self.body = body
        self.radius = radius


if "pymunk" not in sys.modules:
    sys.modules["pymunk"] = SimpleNamespace(
        Body=_Body,
        Circle=_Circle,
        ShapeFilter=lambda **kwargs: SimpleNamespace(**kwargs),
        moment_for_circle=lambda *args: 1.0,
    )

for optional_module in ("neat",):
    if optional_module not in sys.modules:
        sys.modules[optional_module] = ModuleType(optional_module)

from configs.sim_config import build_sim_config
from src.action import Action
from src.creature import VisionTraits
from src.fitness import CreatureFitness
from src.world import World


@dataclass(slots=True)
class FakeCreature:
    creature_id: int
    energy: float = 1.0
    heading: float = 0.0
    position: tuple[float, float] = (0.0, 0.0)
    color: tuple[int, int, int] = (86, 156, 214)
    vision: VisionTraits = field(
        default_factory=lambda: VisionTraits(range=100.0, angle=1.0)
    )


class FakeBrainController:
    def __init__(self) -> None:
        self.created_children: list[tuple[int, int]] = []

    def create_child_brain(
        self,
        parent_creature_id: int,
        child_creature_id: int,
    ) -> bool:
        self.created_children.append((parent_creature_id, child_creature_id))
        return True


class WorldReproductionTest(unittest.TestCase):
    def test_reproduction_skips_unwilling_top_eligible_parent(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.config.population.max_creatures = 10
        world.config.population.reproduction_energy_cost = 0.5
        world.creatures = [
            FakeCreature(creature_id=1),
            FakeCreature(creature_id=2),
        ]
        world.fitness = {
            1: CreatureFitness(age_seconds=30.0),
            2: CreatureFitness(age_seconds=30.0),
        }
        world.fitness_archive = {}
        world._last_actions = {
            1: Action(
                accelerate=0.0,
                rotate=0.0,
                want_reproduce=1.0,
                want_eat=0.0,
                reset_chronometer=0.0,
            ),
            2: Action(
                accelerate=0.0,
                rotate=0.0,
                want_reproduce=0.0,
                want_eat=0.0,
                reset_chronometer=0.0,
            ),
        }
        world.rt_neat = SimpleNamespace(
            eligible_parent_ids=[2, 1],
            stats=SimpleNamespace(births=0),
        )
        world.neat_controller = FakeBrainController()
        world._chronometers = {}
        world._child_spawn_position = lambda parent: (1.0, 1.0)
        world._mutated_creature_color = lambda color: color
        world._mutated_vision = lambda vision: vision
        world._spawn_creature = lambda creature_id, **kwargs: FakeCreature(
            creature_id=creature_id,
            energy=kwargs["energy"],
            heading=kwargs["heading"],
            position=kwargs["position"],
            color=kwargs["color"],
            vision=kwargs["vision"],
        )

        self.assertTrue(world._try_reproduce())

        self.assertEqual(world.neat_controller.created_children, [(1, 3)])
        self.assertEqual(world.rt_neat.stats.births, 1)
        self.assertEqual(len(world.creatures), 3)
        self.assertEqual(world.creatures[-1].creature_id, 3)
        self.assertEqual(world.fitness[1].offspring_count, 1)
        self.assertEqual(world.creatures[0].energy, 0.5)


if __name__ == "__main__":
    unittest.main()
