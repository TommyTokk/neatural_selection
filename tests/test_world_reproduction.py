from __future__ import annotations

from colorsys import rgb_to_hsv
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
from src.creature import FlockingTraits, LineageInfo, PhysicalTraits, VisionTraits
from src.fitness import CreatureFitness
from src.neat_controller import NeatBrainController, SpeciationResult
from src.rt_neat import RtNeatManager
from src.speciation import SpeciesDistanceBreakdown, SpeciesTraitSnapshot
from src.world import ArchivedCreatureTraits, World


@dataclass(slots=True)
class FakeCreature:
    creature_id: int
    energy: float = 1.0
    total_energy_gathered: float = 0.0
    stomach_energy: float = 0.0
    heading: float = 0.0
    position: tuple[float, float] = (0.0, 0.0)
    speed: float = 0.0
    biome_fertility_ema: float = 0.0
    biome_fertility_ema_updated_at: float = 0.0
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
    flocking_traits: FlockingTraits = field(default_factory=FlockingTraits)
    lineage: LineageInfo = field(default_factory=LineageInfo)

    @property
    def radius(self) -> float:
        return self.physical_traits.radius


def speciation_result(
    species_id: int,
    parent_species_id: int,
    is_new_species: bool,
) -> SpeciationResult:
    zero = SpeciesTraitSnapshot(0.0, 0.0, 0.0, 0.0)
    return SpeciationResult(
        species_id=species_id,
        parent_species_id=parent_species_id,
        is_new_species=is_new_species,
        founder_traits=SpeciesTraitSnapshot(16.0, 100.0, 1.0, 1.0),
        trait_deltas=zero,
        distances=SpeciesDistanceBreakdown(
            neat_distance=3.1 if is_new_species else 0.0,
            phenotypic_distance=0.0,
            weighted_phenotypic_distance=0.0,
            composite_distance=3.1 if is_new_species else 0.0,
            compatibility_threshold=3.0,
            phenotypic_weight=2.0,
            radius_component=0.0,
            vision_range_component=0.0,
            vision_angle_component=0.0,
            movement_cost_component=0.0,
        ),
    )


class FakeMetabolism:
    def __init__(self) -> None:
        self.movement_multipliers_seen: dict[int, float] = {}
        self.energy_cost_multipliers_seen: dict[int, float] = {}

    def update(self, creatures, *args, **kwargs):
        del args
        self.movement_multipliers_seen = {
            creature.creature_id: creature.physical_traits.movement_cost_multiplier
            for creature in creatures
        }
        self.energy_cost_multipliers_seen = dict(
            kwargs.get("energy_cost_multipliers") or {}
        )
        return SimpleNamespace(
            food_consumptions=[],
            touched_foods=[],
            depleted_foods=[],
            dead_creatures=[],
        )


class FakeBrainController:
    def __init__(self) -> None:
        self.created_children: list[tuple[int, int, int]] = []
        self.created_child_traits: list[tuple[PhysicalTraits, VisionTraits]] = []
        self.archived: list[tuple[int, float]] = []
        self.removed: list[int] = []

    def create_child_brain(
        self,
        parent_creature_id: int,
        child_creature_id: int,
        parent_species_id: int,
        child_physical_traits: PhysicalTraits,
        child_vision: VisionTraits,
        child_flocking_traits: FlockingTraits,
    ) -> tuple[object, SpeciationResult]:
        del child_flocking_traits
        self.created_children.append(
            (parent_creature_id, child_creature_id, parent_species_id)
        )
        self.created_child_traits.append((child_physical_traits, child_vision))
        return object(), speciation_result(
            parent_species_id,
            parent_species_id,
            False,
        )

    def archive_brain(self, creature_id: int, fitness_score: float) -> bool:
        self.archived.append((creature_id, fitness_score))
        return True

    def remove_brain(self, creature_id: int) -> None:
        self.removed.append(creature_id)


class FakeFoodSpawner:
    def __init__(self, food_capacity: int = 10, pressure: float = 1.0) -> None:
        self._food_capacity = food_capacity
        self._pressure = pressure

    def food_capacity(self, active_species_count: int | None = None) -> int:
        del active_species_count
        return self._food_capacity

    def food_regrowth_pressure(
        self,
        current_food_count: int,
        food_capacity: int,
    ) -> float:
        del current_food_count, food_capacity
        return self._pressure


class FakeGenome:
    def __init__(self, fitness: float | None = None) -> None:
        self.fitness = fitness


class WorldReproductionTest(unittest.TestCase):
    def test_creature_biomass_includes_stomach_contents(self) -> None:
        world = object.__new__(World)
        world.creatures = [
            FakeCreature(creature_id=1, energy=0.4, stomach_energy=0.3),
            FakeCreature(creature_id=2, energy=0.2, stomach_energy=0.1),
        ]

        self.assertAlmostEqual(world._creature_energy(), 1.0)

    def test_initial_creatures_share_species_one_color(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.config.population.initial_creatures = 3
        seen_colors: list[tuple[int, int, int]] = []

        def spawn(creature_id: int, **kwargs: object) -> FakeCreature:
            color = kwargs["color"]
            assert isinstance(color, tuple)
            seen_colors.append(color)
            return FakeCreature(creature_id=creature_id, color=color)

        world._spawn_creature = spawn

        creatures = world._spawn_creatures()

        self.assertEqual(len(set(seen_colors)), 1)
        self.assertTrue(all(creature.lineage.species_id == 1 for creature in creatures))

    def test_historical_archives_are_bounded_and_keep_required_records(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.config.population.elite_archive_size = 2
        world.config.population.fitness_archive_size = 2
        infant = FakeCreature(
            creature_id=20,
            lineage=LineageInfo(parent_id=10, species_id=2),
        )
        world.creatures = [infant]
        world.fitness = {20: CreatureFitness(age_seconds=1.0)}
        world.fitness_archive = {
            creature_id: CreatureFitness(age_seconds=10.0)
            for creature_id in (10, 11, 12, 13)
        }

        controller = NeatBrainController.__new__(NeatBrainController)
        controller.population = SimpleNamespace(
            population={
                genome_id: SimpleNamespace(
                    key=genome_id,
                    fitness=float(genome_id),
                )
                for genome_id in (1, 2, 3, 4, 100)
            }
        )
        controller.brains = {20: SimpleNamespace(genome_id=100)}
        controller.species_manager = SimpleNamespace(
            representatives={
                species_id: (SimpleNamespace(key=species_id), None, None)
                for species_id in range(1, 7)
            }
        )
        world.neat_controller = controller
        world._trait_archive_by_genome_id = {
            genome_id: ArchivedCreatureTraits(
                creature_id=genome_id,
                vision=VisionTraits(range=100.0, angle=1.0),
                physical_traits=PhysicalTraits(radius=16.0),
                color=(1, 2, 3),
                lineage=LineageInfo(species_id=genome_id + 1),
            )
            for genome_id in (1, 2, 3, 4)
        }

        world._prune_historical_archives()

        self.assertEqual(set(controller.population.population), {3, 4, 100})
        self.assertEqual(set(world._trait_archive_by_genome_id), {3, 4})
        self.assertEqual(
            set(controller.species_manager.representatives),
            {1, 2, 4, 5},
        )
        self.assertEqual(set(world.fitness_archive), {10, 12, 13})

    def test_monotonic_creature_ids_survive_archive_pruning(self) -> None:
        world = object.__new__(World)
        world.creatures = []
        world.fitness = {}
        world.fitness_archive = {}
        world._next_creature_id_value = 500

        self.assertEqual(world._next_creature_id(), 500)
        self.assertEqual(world._next_creature_id(), 501)

    def test_dynamic_speciation_threshold_tracks_living_species(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.config.speciation.adjustment_interval_seconds = 5.0
        world.config.speciation.threshold_adjust_rate = 0.05
        world._speciation_adjustment_accumulator = 0.0
        species_manager = SimpleNamespace(compatibility_threshold=3.5)
        world.neat_controller = SimpleNamespace(species_manager=species_manager)

        world.creatures = [
            FakeCreature(creature_id=1, lineage=LineageInfo(species_id=1)),
            FakeCreature(creature_id=2, lineage=LineageInfo(species_id=2)),
        ]
        world._update_speciation_threshold(4.0)
        self.assertEqual(species_manager.compatibility_threshold, 3.5)
        world._update_speciation_threshold(1.0)
        self.assertAlmostEqual(species_manager.compatibility_threshold, 3.45)

        world.creatures = [
            FakeCreature(
                creature_id=species_id,
                lineage=LineageInfo(species_id=species_id),
            )
            for species_id in range(1, 11)
        ]
        world._update_speciation_threshold(5.0)
        self.assertAlmostEqual(species_manager.compatibility_threshold, 3.5)

        world.creatures = [
            FakeCreature(
                creature_id=species_id,
                lineage=LineageInfo(species_id=species_id),
            )
            for species_id in range(1, 6)
        ]
        world._update_speciation_threshold(5.0)
        self.assertAlmostEqual(species_manager.compatibility_threshold, 3.5)

    def test_dynamic_speciation_threshold_clamps_and_retains_overshoot(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.config.speciation.adjustment_interval_seconds = 5.0
        world._speciation_adjustment_accumulator = 0.0
        species_manager = SimpleNamespace(compatibility_threshold=3.5)
        world.neat_controller = SimpleNamespace(species_manager=species_manager)
        world.creatures = [
            FakeCreature(creature_id=1, lineage=LineageInfo(species_id=1))
        ]

        world._update_speciation_threshold(10.25)
        self.assertAlmostEqual(species_manager.compatibility_threshold, 3.4)
        self.assertAlmostEqual(world._speciation_adjustment_accumulator, 0.25)

        species_manager.compatibility_threshold = (
            world.config.speciation.min_threshold
        )
        world._update_speciation_threshold(4.75)
        self.assertEqual(
            species_manager.compatibility_threshold,
            world.config.speciation.min_threshold,
        )

        species_manager.compatibility_threshold = (
            world.config.speciation.max_threshold
        )
        world.creatures = [
            FakeCreature(
                creature_id=species_id,
                lineage=LineageInfo(species_id=species_id),
            )
            for species_id in range(1, 11)
        ]
        world._update_speciation_threshold(5.0)
        self.assertEqual(
            species_manager.compatibility_threshold,
            world.config.speciation.max_threshold,
        )

    def test_reproduction_skips_unwilling_top_eligible_parent(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.config.population.max_creatures = 10
        world.config.population.reproduction_energy_cost_base = 0.4
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
                want_grab=0.0,
                want_release=0.0,
            ),
            2: Action(
                accelerate=0.0,
                rotate=0.0,
                want_reproduce=0.0,
                want_eat=0.0,
                reset_chronometer=0.0,
                want_grab=0.0,
                want_release=0.0,
            ),
        }
        world.rt_neat = RtNeatManager(brain_controller=None)
        world.rt_neat.eligible_parent_ids = [2, 1]
        world.neat_controller = FakeBrainController()
        world.species_history = {}
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

        self.assertEqual(world.neat_controller.created_children, [(1, 3, 1)])
        self.assertEqual(
            world.neat_controller.created_child_traits,
            [
                (
                    world.creatures[-1].physical_traits,
                    world.creatures[-1].vision,
                )
            ],
        )
        self.assertEqual(world.rt_neat.stats.births, 1)
        self.assertEqual(world.rt_neat.stats.normal_replacements, 1)
        self.assertEqual(len(world.creatures), 3)
        self.assertEqual(world.creatures[-1].creature_id, 3)
        self.assertEqual(
            world.creatures[-1].energy,
            world.config.population.infant_energy_spawn,
        )
        self.assertEqual(world.creatures[-1].total_energy_gathered, 0.0)
        self.assertEqual(world.creatures[-1].lineage.parent_id, 1)
        self.assertEqual(world.creatures[-1].lineage.generation, 1)
        self.assertEqual(world.fitness[1].offspring_count, 1)
        self.assertEqual(world.creatures[0].energy, 0.6)
        self.assertEqual(world.species_history, {})

    def test_reproduction_intent_uses_strict_centered_threshold(self) -> None:
        world = object.__new__(World)
        parent = FakeCreature(creature_id=1)
        world.creatures = [parent]
        world.fitness = {1: CreatureFitness(age_seconds=30.0)}
        world.rt_neat = SimpleNamespace(eligible_parent_ids=[1])

        for inactive_value in (0.0, 0.1):
            world._last_actions = {
                1: Action(0.0, 0.0, inactive_value, 0.0, 0.0, 0.0, 0.0)
            }
            self.assertIsNone(world._reproduction_parent())

        world._last_actions = {
            1: Action(0.0, 0.0, 0.100001, 0.0, 0.0, 0.0, 0.0)
        }
        self.assertIs(world._reproduction_parent(), parent)

    def test_parent_pool_requires_current_eligibility_and_intent(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        ineligible = FakeCreature(
            creature_id=1,
            energy=0.79,
            total_energy_gathered=100.0,
        )
        unwilling = FakeCreature(
            creature_id=2,
            energy=1.0,
            total_energy_gathered=50.0,
        )
        eligible = FakeCreature(
            creature_id=3,
            energy=1.0,
            total_energy_gathered=1.0,
        )
        world.creatures = [ineligible, unwilling, eligible]
        world.fitness = {
            creature.creature_id: CreatureFitness(age_seconds=30.0)
            for creature in world.creatures
        }
        world.rt_neat = RtNeatManager(brain_controller=None, rng=Random(7))
        world._last_actions = {
            1: Action(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            2: Action(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            3: Action(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        }

        self.assertIs(world._reproduction_parent(), eligible)

    def test_eating_intent_uses_strict_centered_threshold(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        creature = FakeCreature(creature_id=1)

        for inactive_value in (0.0, 0.1):
            world._last_actions = {
                1: Action(0.0, 0.0, 0.0, inactive_value, 0.0, 0.0, 0.0)
            }
            self.assertFalse(world._creature_want_to_eat(creature))

        world._last_actions = {
            1: Action(0.0, 0.0, 0.0, 0.100001, 0.0, 0.0, 0.0)
        }
        self.assertTrue(world._creature_want_to_eat(creature))

    def test_eating_intent_is_gated_when_stomach_is_full(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        creature = FakeCreature(creature_id=1)
        world._last_actions = {
            1: Action(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        }
        creature.stomach_energy = (
            creature.radius
            * world.config.metabolism.stomach_capacity_per_radius
        )

        self.assertFalse(world._creature_want_to_eat(creature))

    def test_reproduction_cost_scales_with_brain_complexity_and_caps(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        parent = FakeCreature(creature_id=1)

        small_genome = SimpleNamespace(
            nodes={node_id: object() for node_id in range(5)},
            connections={connection_id: object() for connection_id in range(10)},
        )
        large_genome = SimpleNamespace(
            nodes={node_id: object() for node_id in range(20)},
            connections={connection_id: object() for connection_id in range(30)},
        )
        active_genome = small_genome
        world.neat_controller = SimpleNamespace(
            brain_for=lambda creature_id: SimpleNamespace(genome=active_genome)
        )

        self.assertAlmostEqual(world._reproduction_cost_for(parent), 0.41)
        active_genome = large_genome
        self.assertAlmostEqual(world._reproduction_cost_for(parent), 0.57)
        world.config.population.max_dynamic_reproduction_cost = 0.5
        self.assertAlmostEqual(world._reproduction_cost_for(parent), 0.5)

    def test_missing_brain_uses_base_reproduction_cost(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.neat_controller = SimpleNamespace(brain_for=lambda creature_id: None)

        self.assertAlmostEqual(
            world._reproduction_cost_for(FakeCreature(creature_id=1)),
            world.config.population.reproduction_energy_cost_base,
        )

    def test_reproduction_resource_gate_uses_base_cost(self) -> None:
        world = self._world_ready_to_reproduce()
        used_biomass = sum(creature.energy for creature in world.creatures) + sum(
            food.energy_value for food in world.foods
        )
        world.total_biomass_energy = used_biomass + 0.45

        self.assertTrue(world._has_reproduction_resources())

    def test_failed_child_creation_does_not_spend_reproduction_energy(self) -> None:
        world = self._world_ready_to_reproduce()
        parent = world.creatures[0]
        parent.energy = 1.0
        genome = SimpleNamespace(
            nodes={node_id: object() for node_id in range(20)},
            connections={
                connection_id: object() for connection_id in range(30)
            },
        )
        world.neat_controller = SimpleNamespace(
            brain_for=lambda creature_id: SimpleNamespace(genome=genome),
            create_child_brain=lambda *args: (None, None),
        )
        world.space = SimpleNamespace(remove=lambda *args: None)

        self.assertFalse(world._try_reproduce())
        self.assertEqual(parent.energy, 1.0)

    def test_reproduction_assigns_new_species_and_overwrites_color(self) -> None:
        world = self._world_ready_to_reproduce()
        parent = world.creatures[0]
        parent.lineage.species_id = 7
        world.neat_controller = SimpleNamespace(
            create_child_brain=lambda parent_id, child_id, parent_species_id, physical, vision, flocking: (
                object(),
                speciation_result(8, parent_species_id, True),
            ),
            genome_id_for=lambda creature_id: 80,
        )
        founder_color = (240, 48, 180)
        world._new_species_color = lambda parent_color: founder_color
        species_records: list[object] = []
        world.telemetry = SimpleNamespace(
            log_species_record=species_records.append,
            log_creature_birth=lambda *args: None,
        )

        self.assertTrue(world._try_reproduce())

        child = world.creatures[-1]
        self.assertEqual(child.lineage.species_id, 8)
        self.assertEqual(child.color, founder_color)
        record = world.species_history[8]
        self.assertEqual(record.parent_species_id, 7)
        self.assertEqual(record.founder_creature_id, child.creature_id)
        self.assertEqual(record.founder_genome_id, 80)
        self.assertEqual(record.founder_color, founder_color)
        self.assertEqual(record.data_quality, "exact")
        self.assertEqual(record.emergence_food_ratio, 0.5)
        self.assertEqual(record.emergence_pop_ratio, 0.2)
        self.assertEqual(record.neural_shifts, ())
        self.assertEqual(species_records, [record])

    def test_same_species_reproduction_keeps_inherited_color_jitter(self) -> None:
        world = self._world_ready_to_reproduce()
        parent = world.creatures[0]
        parent.lineage.species_id = 4

        self.assertTrue(world._try_reproduce())

        child = world.creatures[-1]
        parent_hue = rgb_to_hsv(*(channel / 255.0 for channel in parent.color))[0]
        child_hue = rgb_to_hsv(*(channel / 255.0 for channel in child.color))[0]
        hue_distance = min(
            abs(parent_hue - child_hue),
            1.0 - abs(parent_hue - child_hue),
        )
        self.assertEqual(child.lineage.species_id, 4)
        self.assertLessEqual(hue_distance, 0.04)

    def test_new_species_color_is_saturated_bright_and_distinct(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.rng = Random(7)
        parent_color = (86, 156, 214)

        color = world._new_species_color(parent_color)

        parent_hue = rgb_to_hsv(*(channel / 255.0 for channel in parent_color))[0]
        hue, saturation, value = rgb_to_hsv(
            *(channel / 255.0 for channel in color)
        )
        hue_distance = min(abs(parent_hue - hue), 1.0 - abs(parent_hue - hue))
        self.assertGreaterEqual(hue_distance, 0.175)
        self.assertGreaterEqual(saturation, 0.69)
        self.assertGreaterEqual(value, 0.79)
        self.assertFalse(
            world._is_food_like_color(tuple(channel / 255.0 for channel in color))
        )

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
        world._last_actions = {1: Action(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}
        world._last_flock_steering_debug = {1: object()}
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
        self.assertNotIn(1, world._last_flock_steering_debug)
        self.assertNotIn(1, world._chronometers)
        self.assertTrue(world._behavior_cohort_dirty)
        self.assertEqual(world.neat_controller.removed, [1])
        self.assertEqual(world.rt_neat.stats.deaths, 1)
        self.assertAlmostEqual(world.rt_neat.stats.average_lifespan_at_death, 10.0)

    def test_own_infant_children_filters_to_direct_infant_children(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.config.population.infant_maturity_age = 5.0
        parent = FakeCreature(creature_id=1)
        own_infant = FakeCreature(
            creature_id=2,
            lineage=LineageInfo(parent_id=1),
        )
        unrelated_infant = FakeCreature(
            creature_id=3,
            lineage=LineageInfo(parent_id=99),
        )
        mature_child = FakeCreature(
            creature_id=4,
            lineage=LineageInfo(parent_id=1),
        )
        world.creatures = [parent, own_infant, unrelated_infant, mature_child]
        world.fitness = {
            1: CreatureFitness(age_seconds=30.0),
            2: CreatureFitness(age_seconds=1.0),
            3: CreatureFitness(age_seconds=1.0),
            4: CreatureFitness(age_seconds=6.0),
        }

        self.assertEqual(world._own_infant_children_for(parent), [own_infant])

    def test_maturity_reward_is_recorded_once_for_parent(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.config.population.infant_maturity_age = 5.0
        parent = FakeCreature(creature_id=1)
        child = FakeCreature(creature_id=2, lineage=LineageInfo(parent_id=1))
        world.creatures = [parent, child]
        world.fitness = {
            1: CreatureFitness(age_seconds=30.0),
            2: CreatureFitness(age_seconds=4.9),
        }
        child_fitness = world.fitness[2]

        world._record_maturity_if_crossed(child, 4.9, child_fitness)
        self.assertEqual(world.fitness[1].matured_offspring_ids, [])

        child_fitness.age_seconds = 5.0
        world._record_maturity_if_crossed(child, 4.9, child_fitness)
        world._record_maturity_if_crossed(child, 4.9, child_fitness)

        self.assertEqual(world.fitness[1].matured_offspring_ids, [2])

    def test_nursing_transfers_energy_to_nearest_own_infant(self) -> None:
        world = self._world_ready_for_parenting()
        parent, own_infant, farther_infant, unrelated_infant = world.creatures

        world._apply_nursing(2.0)

        self.assertAlmostEqual(parent.energy, 0.9)
        self.assertAlmostEqual(own_infant.energy, 0.3)
        self.assertAlmostEqual(farther_infant.energy, 0.2)
        self.assertAlmostEqual(unrelated_infant.energy, 0.2)

    def test_nursing_intent_uses_strict_centered_threshold(self) -> None:
        for inactive_value in (0.0, 0.1):
            world = self._world_ready_for_parenting()
            parent, own_infant = world.creatures[0], world.creatures[1]
            world._last_actions[1].want_nurse = inactive_value

            world._apply_nursing(2.0)

            self.assertAlmostEqual(parent.energy, 1.0)
            self.assertAlmostEqual(own_infant.energy, 0.2)

        world = self._world_ready_for_parenting()
        parent, own_infant = world.creatures[0], world.creatures[1]
        world._last_actions[1].want_nurse = 0.100001

        world._apply_nursing(2.0)

        self.assertAlmostEqual(parent.energy, 0.9)
        self.assertAlmostEqual(own_infant.energy, 0.3)

    def test_nursing_respects_parent_energy_and_infant_capacity(self) -> None:
        world = self._world_ready_for_parenting()
        parent, own_infant = world.creatures[0], world.creatures[1]
        parent.energy = 0.28
        own_infant.energy = 0.05

        world._apply_nursing(1.0)

        self.assertAlmostEqual(parent.energy, 0.28)
        self.assertAlmostEqual(own_infant.energy, 0.05)

        parent.energy = 0.30
        own_infant.energy = 0.05
        world._apply_nursing(1.0)

        self.assertAlmostEqual(parent.energy, 0.30)
        self.assertAlmostEqual(own_infant.energy, 0.05)

        parent.energy = 1.0
        own_infant.energy = 0.98
        world._apply_nursing(2.0)

        self.assertAlmostEqual(parent.energy, 0.9)
        self.assertAlmostEqual(own_infant.energy, 1.0)

    def test_infant_movement_penalty_is_temporary_during_metabolism(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.config.population.infant_maturity_age = 5.0
        infant = FakeCreature(
            creature_id=1,
            physical_traits=PhysicalTraits(
                radius=16.0,
                movement_cost_multiplier=1.25,
            ),
        )
        adult = FakeCreature(
            creature_id=2,
            physical_traits=PhysicalTraits(
                radius=16.0,
                movement_cost_multiplier=1.1,
            ),
        )
        world.creatures = [infant, adult]
        world.fitness = {
            1: CreatureFitness(age_seconds=1.0),
            2: CreatureFitness(age_seconds=6.0),
        }
        world._last_actions = {}
        world.foods = []
        world.metabolism = FakeMetabolism()
        world.MAX_SPEED = 170.0
        world.selected_creature_id = None

        world._update_metabolism(1.0)

        self.assertAlmostEqual(world.metabolism.movement_multipliers_seen[1], 3.75)
        self.assertAlmostEqual(world.metabolism.movement_multipliers_seen[2], 1.1)
        self.assertAlmostEqual(infant.physical_traits.movement_cost_multiplier, 1.25)

    def test_senescence_factor_boundaries_and_trait_preservation(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.config.population.senescence_age_seconds = 120.0
        world.config.population.senescence_cost_multiplier = 0.05
        creatures = [
            FakeCreature(
                creature_id=1,
                physical_traits=PhysicalTraits(
                    radius=16.0,
                    movement_cost_multiplier=1.2,
                ),
            ),
            FakeCreature(
                creature_id=2,
                physical_traits=PhysicalTraits(
                    radius=16.0,
                    movement_cost_multiplier=1.1,
                ),
            ),
            FakeCreature(
                creature_id=3,
                physical_traits=PhysicalTraits(
                    radius=16.0,
                    movement_cost_multiplier=0.9,
                ),
            ),
        ]
        world.creatures = creatures
        world.fitness = {
            1: CreatureFitness(age_seconds=119.9),
            2: CreatureFitness(age_seconds=120.0),
            3: CreatureFitness(age_seconds=130.0),
        }
        world._last_actions = {}
        world.foods = []
        world.metabolism = FakeMetabolism()
        world.MAX_SPEED = 170.0
        world.selected_creature_id = None

        world._update_metabolism(1.0)

        self.assertEqual(
            world.metabolism.energy_cost_multipliers_seen,
            {1: 1.0, 2: 1.0, 3: 1.5},
        )
        self.assertEqual(
            [
                creature.physical_traits.movement_cost_multiplier
                for creature in creatures
            ],
            [1.2, 1.1, 0.9],
        )

    def test_senescence_death_reason_is_strictly_above_threshold(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.config.population.senescence_age_seconds = 120.0
        at_threshold = FakeCreature(creature_id=1)
        above_threshold = FakeCreature(creature_id=2)
        world.creatures = [at_threshold, above_threshold]
        world.fitness = {
            1: CreatureFitness(age_seconds=120.0),
            2: CreatureFitness(age_seconds=120.1),
        }
        world._last_actions = {}
        world.foods = []
        world.metabolism = FakeMetabolism()
        world.metabolism.update = lambda *args, **kwargs: SimpleNamespace(
            food_consumptions=[],
            touched_foods=[],
            depleted_foods=[],
            dead_creatures=[at_threshold, above_threshold],
        )
        world.MAX_SPEED = 170.0
        world.selected_creature_id = None
        death_reasons: list[tuple[int, str]] = []
        world._remove_creature = lambda creature, death_reason: death_reasons.append(
            (creature.creature_id, death_reason)
        )

        world._update_metabolism(1.0)

        self.assertEqual(
            death_reasons,
            [(1, "starvation"), (2, "old_age")],
        )

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
            create_mutated_brain_from_genome=(
                lambda parent_genome, child_id, parent_species_id, physical, vision, flocking: (
                    object(),
                    speciation_result(
                        parent_species_id,
                        parent_species_id,
                        False,
                    ),
                )
            ),
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
        forwarded_traits: list[tuple[PhysicalTraits, VisionTraits]] = []

        def create_recovered_brain(
            parent_genome: object,
            child_id: int,
            parent_species_id: int,
            child_physical_traits: PhysicalTraits,
            child_vision: VisionTraits,
            child_flocking_traits: FlockingTraits,
        ) -> tuple[object, SpeciationResult]:
            del parent_genome, child_id, child_flocking_traits
            forwarded_traits.append((child_physical_traits, child_vision))
            return object(), speciation_result(
                parent_species_id,
                parent_species_id,
                False,
            )

        world.neat_controller = SimpleNamespace(
            best_genomes=lambda count: [parent_genome],
            create_mutated_brain_from_genome=create_recovered_brain,
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
                lineage=LineageInfo(parent_id=3, generation=2, species_id=6),
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
        self.assertEqual(recovered.lineage.species_id, 6)
        self.assertNotEqual(recovered.vision.range, 120.0)
        self.assertNotEqual(recovered.physical_traits.radius, 18.0)
        self.assertEqual(
            forwarded_traits,
            [(recovered.physical_traits, recovered.vision)],
        )

    def test_extinction_recovery_can_found_new_species(self) -> None:
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
            create_mutated_brain_from_genome=lambda parent, child_id, parent_species_id, physical, vision, flocking: (
                object(),
                speciation_result(10, parent_species_id, True),
            ),
            genome_id_for=lambda creature_id: 100,
        )
        world._trait_archive_by_genome_id = {
            5: ArchivedCreatureTraits(
                creature_id=7,
                vision=VisionTraits(range=120.0, angle=1.2),
                physical_traits=PhysicalTraits(radius=18.0),
                color=(86, 156, 214),
                lineage=LineageInfo(species_id=6),
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
        world.species_history = {}
        world.elapsed_time = 0.0
        world.telemetry = None
        founder_color = (220, 40, 190)
        world._new_species_color = lambda parent_color: founder_color

        world._recover_extinct_population()

        self.assertEqual(world.creatures[0].lineage.species_id, 10)
        self.assertEqual(world.creatures[0].color, founder_color)
        self.assertEqual(world.species_history[10].parent_species_id, 6)

    def test_extinction_recovery_refresh_does_not_score_newborn_genomes(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.config.population.max_creatures = 1
        world.config.population.extinction_recovery_creatures = 1
        world.config.population.extinction_recovery_parent_pool = 1
        world.config.metabolism.max_energy = 1.0
        world.creatures = []
        world.fitness = {}
        world.fitness_archive = {}
        world.foods = []
        world.total_biomass_energy = 1.0
        world.elapsed_time = 0.0
        world.stats = SimpleNamespace()
        world._chronometers = {}
        world.food_spawner = FakeFoodSpawner(food_capacity=10)
        world.rt_neat = RtNeatManager(brain_controller=None)
        world._spawn_creature = lambda creature_id, **kwargs: FakeCreature(
            creature_id=creature_id,
            energy=kwargs["energy"],
            color=kwargs["color"],
        )
        world._initial_creature_color = lambda index: (86, 156, 214)

        parent_genome = FakeGenome(fitness=12.0)
        newborn_genomes: dict[int, FakeGenome] = {}

        class RecoveryBrainController:
            def best_genomes(self, count: int) -> list[FakeGenome]:
                del count
                return [parent_genome]

            def create_mutated_brain_from_genome(
                self,
                parent: FakeGenome,
                child_id: int,
                parent_species_id: int,
                child_physical_traits: PhysicalTraits,
                child_vision: VisionTraits,
                child_flocking_traits: FlockingTraits,
            ) -> tuple[object, SpeciationResult]:
                del parent, child_physical_traits, child_vision, child_flocking_traits
                newborn_genomes[child_id] = FakeGenome(fitness=None)
                return object(), speciation_result(
                    parent_species_id,
                    parent_species_id,
                    False,
                )

            def update_genome_fitness(
                self,
                creature_id: int,
                fitness_score: float,
            ) -> bool:
                newborn_genomes[creature_id].fitness = fitness_score
                return True

        world.neat_controller = RecoveryBrainController()

        world._recover_extinct_population()
        world._refresh_stats()

        recovered_id = world.creatures[0].creature_id
        self.assertIsNone(newborn_genomes[recovered_id].fitness)

    def _world_ready_to_reproduce(self) -> World:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.config.population.max_creatures = 10
        world.config.population.reproduction_energy_cost_base = 0.4
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
            1: Action(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            2: Action(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        }
        world.rt_neat = RtNeatManager(brain_controller=None)
        world.rt_neat.eligible_parent_ids = [1, 2]
        world.neat_controller = FakeBrainController()
        world.species_history = {}
        world.elapsed_time = 0.0
        world.telemetry = None
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

    def _world_ready_for_parenting(self) -> World:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.config.population.nursing_energy_transfer_rate = 0.05
        world.config.population.infant_maturity_age = 5.0
        parent = FakeCreature(creature_id=1, energy=1.0)
        own_infant = FakeCreature(
            creature_id=2,
            energy=0.2,
            position=(10.0, 0.0),
            lineage=LineageInfo(parent_id=1),
        )
        farther_infant = FakeCreature(
            creature_id=3,
            energy=0.2,
            position=(20.0, 0.0),
            lineage=LineageInfo(parent_id=1),
        )
        unrelated_infant = FakeCreature(
            creature_id=4,
            energy=0.2,
            position=(5.0, 0.0),
            lineage=LineageInfo(parent_id=99),
        )
        world.creatures = [parent, own_infant, farther_infant, unrelated_infant]
        world.fitness = {
            creature.creature_id: CreatureFitness(age_seconds=1.0)
            for creature in world.creatures
        }
        world._last_actions = {
            1: Action(
                accelerate=0.0,
                rotate=0.0,
                want_reproduce=0.0,
                want_eat=0.0,
                reset_chronometer=0.0,
                want_grab=0.0,
                want_release=0.0,
                want_nurse=1.0,
            )
        }
        return world


if __name__ == "__main__":
    unittest.main()
