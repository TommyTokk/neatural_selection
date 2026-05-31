from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
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


try:
    import pymunk  # noqa: F401
except ModuleNotFoundError:
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
from src.creature import LineageInfo, PhysicalTraits, VisionTraits
from src.fitness import CreatureFitness
from src.rt_neat import RtNeatManager
from src.world import ArchivedCreatureTraits, World


@dataclass(slots=True)
class FakeCreature:
    creature_id: int
    energy: float = 1.0
    heading: float = 0.0
    position: tuple[float, float] = (0.0, 0.0)
    color: tuple[int, int, int] = (86, 156, 214)
    body: object = field(default_factory=object)
    shape: object = field(default_factory=object)
    vision: VisionTraits = field(
        default_factory=lambda: VisionTraits(range=100.0, angle=1.0)
    )
    physical_traits: PhysicalTraits = field(
        default_factory=lambda: PhysicalTraits(
            radius=16.0,
            movement_cost_multiplier=1.0,
        )
    )
    lineage: LineageInfo = field(default_factory=LineageInfo)


class FakeBrainController:
    def __init__(self) -> None:
        self.created_children: list[tuple[int, int]] = []
        self.archived: list[tuple[int, float]] = []
        self.removed: list[int] = []

    def create_child_brain(
        self,
        parent_creature_id: int,
        child_creature_id: int,
    ) -> bool:
        self.created_children.append((parent_creature_id, child_creature_id))
        return True

    def archive_brain(self, creature_id: int, fitness_score: float) -> bool:
        self.archived.append((creature_id, fitness_score))
        return True

    def remove_brain(self, creature_id: int) -> None:
        self.removed.append(creature_id)


class FakeFoodSpawner:
    def __init__(self, food_capacity: int = 10, pressure: float = 1.0) -> None:
        self._food_capacity = food_capacity
        self._pressure = pressure

    def food_capacity(self, creature_count: int) -> int:
        return self._food_capacity

    def creature_pressure_factor(self, creature_count: int) -> float:
        return self._pressure


class FakeGenome:
    pass


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
        world.foods = [SimpleNamespace(energy_value=0.01) for _ in range(5)]
        world.food_spawner = FakeFoodSpawner(food_capacity=10)
        world.total_biomass_energy = 10.0
        world.rng = Random(7)
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
        world.rt_neat = RtNeatManager(brain_controller=None)
        world.rt_neat.eligible_parent_ids = [2, 1]
        world.neat_controller = FakeBrainController()
        world._chronometers = {}
        world._child_spawn_position = lambda parent, child_radius: (1.0, 1.0)
        world._spawn_creature = lambda creature_id, **kwargs: FakeCreature(
            creature_id=creature_id,
            energy=kwargs["energy"],
            heading=kwargs["heading"],
            position=kwargs["position"],
            color=kwargs["color"],
            vision=kwargs["vision"],
            physical_traits=kwargs["physical_traits"],
            lineage=kwargs["lineage"],
        )

        self.assertTrue(world._try_reproduce())

        self.assertEqual(world.neat_controller.created_children, [(1, 3)])
        self.assertEqual(world.rt_neat.stats.births, 1)
        self.assertEqual(world.rt_neat.stats.normal_replacements, 1)
        self.assertEqual(len(world.creatures), 3)
        self.assertEqual(world.creatures[-1].creature_id, 3)
        self.assertEqual(world.creatures[-1].lineage.parent_id, 1)
        self.assertEqual(world.creatures[-1].lineage.generation, 1)
        self.assertEqual(world.fitness[1].offspring_count, 1)
        self.assertEqual(world.creatures[0].energy, 0.5)

    def test_kill_selected_creature_removes_live_state(self) -> None:
        world = object.__new__(World)
        creature = FakeCreature(creature_id=1)
        world.config = build_sim_config()
        world.creatures = [creature]
        world.fitness = {1: CreatureFitness(age_seconds=10.0)}
        world.fitness_archive = {}
        world.selected_creature_id = 1
        world.neat_controller = FakeBrainController()
        world.rt_neat = RtNeatManager(brain_controller=None)
        world._last_actions = {1: Action(0.0, 0.0, 0.0, 0.0, 0.0)}
        world._chronometers = {1: 4.0}
        world.space = SimpleNamespace(remove=lambda *args: None)
        world._recover_extinct_population = lambda: None
        world._refresh_stats = lambda: None

        self.assertTrue(world.kill_selected_creature())

        self.assertEqual(world.creatures, [])
        self.assertIsNone(world.selected_creature_id)
        self.assertNotIn(1, world.fitness)
        self.assertIn(1, world.fitness_archive)
        self.assertNotIn(1, world._last_actions)
        self.assertNotIn(1, world._chronometers)
        self.assertEqual(world.neat_controller.removed, [1])
        self.assertEqual(world.rt_neat.stats.deaths, 1)
        self.assertAlmostEqual(world.rt_neat.stats.average_lifespan_at_death, 10.0)

    def test_kill_selected_creature_without_selection_is_noop(self) -> None:
        world = object.__new__(World)
        world.creatures = [FakeCreature(creature_id=1)]
        world.selected_creature_id = None

        self.assertFalse(world.kill_selected_creature())
        self.assertEqual(len(world.creatures), 1)

    def test_reproduction_blocks_at_population_cap(self) -> None:
        world = self._world_ready_to_reproduce()
        world.config.population.max_creatures = 2

        self.assertFalse(world._try_reproduce())
        self.assertEqual(world.rt_neat.stats.births, 0)

    def test_reproduction_blocks_without_child_biomass(self) -> None:
        world = self._world_ready_to_reproduce()
        world.total_biomass_energy = sum(creature.energy for creature in world.creatures)

        self.assertFalse(world._try_reproduce())
        self.assertEqual(world.rt_neat.stats.births, 0)

    def test_reproduction_blocks_under_severe_resource_pressure(self) -> None:
        world = self._world_ready_to_reproduce()
        world.foods = []
        world.food_spawner = FakeFoodSpawner(food_capacity=10, pressure=0.1)

        self.assertFalse(world._try_reproduce())
        self.assertEqual(world.rt_neat.stats.births, 0)

    def test_extinction_recovery_counts_replacements_and_respects_cap(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.config.population.max_creatures = 3
        world.config.population.extinction_recovery_creatures = 10
        world.config.population.extinction_recovery_parent_pool = 2
        world.config.metabolism.max_energy = 1.0
        world.creatures = []
        world.fitness = {}
        world.fitness_archive = {}
        world._chronometers = {}
        world.rt_neat = RtNeatManager(brain_controller=None)
        world.neat_controller = SimpleNamespace(
            best_genomes=lambda count: [FakeGenome(), FakeGenome()],
            create_mutated_brain_from_genome=lambda parent_genome, child_id: True,
        )
        world._spawn_creature = lambda creature_id, **kwargs: FakeCreature(
            creature_id=creature_id,
            energy=kwargs["energy"],
            color=kwargs["color"],
        )
        world._initial_creature_color = lambda index: (86, 156, 214)

        world._recover_extinct_population()

        self.assertEqual(len(world.creatures), 3)
        self.assertEqual(world.rt_neat.stats.births, 3)
        self.assertEqual(world.rt_neat.stats.extinction_replacements, 3)

    def test_extinction_recovery_mutates_archived_parent_traits(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.config.population.max_creatures = 1
        world.config.population.extinction_recovery_creatures = 1
        world.config.population.extinction_recovery_parent_pool = 1
        world.config.metabolism.max_energy = 1.0
        world.creatures = []
        world.fitness = {}
        world.fitness_archive = {}
        world._chronometers = {}
        world.rng = Random(7)
        world.rt_neat = RtNeatManager(brain_controller=None)
        parent_genome = SimpleNamespace(key=5)
        world.neat_controller = SimpleNamespace(
            best_genomes=lambda count: [parent_genome],
            create_mutated_brain_from_genome=lambda parent_genome, child_id: True,
        )
        world._trait_archive_by_genome_id = {
            5: ArchivedCreatureTraits(
                creature_id=7,
                vision=VisionTraits(range=120.0, angle=1.2),
                physical_traits=PhysicalTraits(
                    radius=18.0,
                    movement_cost_multiplier=1.1,
                ),
                color=(86, 156, 214),
                lineage=LineageInfo(parent_id=3, generation=2),
            )
        }
        world._spawn_creature = lambda creature_id, **kwargs: FakeCreature(
            creature_id=creature_id,
            energy=kwargs["energy"],
            color=kwargs["color"],
            vision=kwargs["vision"],
            physical_traits=kwargs["physical_traits"],
            lineage=kwargs["lineage"],
        )

        world._recover_extinct_population()

        recovered = world.creatures[0]
        self.assertEqual(recovered.lineage.parent_id, 7)
        self.assertEqual(recovered.lineage.generation, 3)
        self.assertNotEqual(recovered.vision.range, 120.0)
        self.assertNotEqual(recovered.physical_traits.radius, 18.0)

    def _world_ready_to_reproduce(self) -> World:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.config.population.max_creatures = 10
        world.config.population.reproduction_energy_cost = 0.5
        world.creatures = [FakeCreature(creature_id=1), FakeCreature(creature_id=2)]
        world.fitness = {
            1: CreatureFitness(age_seconds=30.0),
            2: CreatureFitness(age_seconds=30.0),
        }
        world.fitness_archive = {}
        world.foods = [SimpleNamespace(energy_value=0.01) for _ in range(5)]
        world.food_spawner = FakeFoodSpawner(food_capacity=10)
        world.total_biomass_energy = 10.0
        world.rng = Random(7)
        world._last_actions = {
            1: Action(0.0, 0.0, 1.0, 0.0, 0.0),
            2: Action(0.0, 0.0, 0.0, 0.0, 0.0),
        }
        world.rt_neat = RtNeatManager(brain_controller=None)
        world.rt_neat.eligible_parent_ids = [1, 2]
        world.neat_controller = FakeBrainController()
        world._chronometers = {}
        world._child_spawn_position = lambda parent, child_radius: (1.0, 1.0)
        world._spawn_creature = lambda creature_id, **kwargs: FakeCreature(
            creature_id=creature_id,
            energy=kwargs["energy"],
            heading=kwargs["heading"],
            position=kwargs["position"],
            color=kwargs["color"],
            vision=kwargs["vision"],
            physical_traits=kwargs["physical_traits"],
            lineage=kwargs["lineage"],
        )
        return world


if __name__ == "__main__":
    unittest.main()
