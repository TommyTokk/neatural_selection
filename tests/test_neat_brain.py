from __future__ import annotations

from itertools import count
from types import ModuleType, SimpleNamespace
import sys
import unittest


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

try:
    import neat
except ModuleNotFoundError:
    sys.modules["neat"] = ModuleType("neat")
    import neat

from src.neat_brain import NeatBrain
from src.neat_controller import NeatBrainController
from src.vision import BoundarySnapshot, SensorSnapshot, VisionTargetSnapshot


class FakeNetwork:
    def __init__(self, outputs: list[float]) -> None:
        self.outputs = outputs
        self.activate_count = 0

    def activate(self, inputs: list[float]) -> list[float]:
        self.activate_count += 1
        return self.outputs


def empty_target() -> VisionTargetSnapshot:
    return VisionTargetSnapshot(
        visible=0.0,
        proximity=0.0,
        angle=0.0,
        density=0.0,
        count=0,
    )


def sensor_snapshot() -> SensorSnapshot:
    return SensorSnapshot(
        food=empty_target(),
        creatures=empty_target(),
        walls=empty_target(),
        boundary=BoundarySnapshot(pressure=0.0, turn=0.0),
        energy=1.0,
        speed=0.0,
        vision_range=0.0,
        vision_angle=0.0,
        vision_energy_cost=0.0,
        maturity=0.0,
        visible_food_count=0.0,
        visible_creature_count=0.0,
        clock_tik_tok=0.0,
        clock_chronometer=0.0,
        clock_time_alive=0.0,
        is_grabbing=0.0,
    )


class NeatBrainActionMappingTest(unittest.TestCase):
    def decide_with_outputs(self, outputs: list[float]):
        brain = NeatBrain(
            genome_id=1,
            genome=SimpleNamespace(),
            network=FakeNetwork(outputs),
        )
        return brain.decide(sensor_snapshot())

    def test_neutral_movement_outputs_map_to_stillness(self) -> None:
        action = self.decide_with_outputs([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.assertAlmostEqual(action.accelerate, 0.0)
        self.assertAlmostEqual(action.rotate, 0.0)

    def test_acceleration_output_is_signed(self) -> None:
        action = self.decide_with_outputs([0.25, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.assertAlmostEqual(action.accelerate, -0.5)
        self.assertAlmostEqual(action.rotate, 0.0)

    def test_rotation_output_is_signed(self) -> None:
        left_action = self.decide_with_outputs(
            [0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        right_action = self.decide_with_outputs(
            [0.5, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0]
        )

        self.assertAlmostEqual(left_action.rotate, -0.5)
        self.assertAlmostEqual(right_action.rotate, 0.5)

    def test_intent_outputs_remain_normalized(self) -> None:
        action = self.decide_with_outputs(
            [0.5, 0.5, 1.2, -0.2, 0.75, 0.25, 1.2, 0.8]
        )

        self.assertEqual(action.want_reproduce, 1.0)
        self.assertEqual(action.want_eat, 0.0)
        self.assertEqual(action.reset_chronometer, 0.75)
        self.assertEqual(action.want_grab, 0.25)
        self.assertEqual(action.want_release, 1.0)
        self.assertEqual(action.want_nurse, 0.8)

    def test_missing_carry_outputs_default_to_neutral(self) -> None:
        action = self.decide_with_outputs([0.5, 0.5, 0.0, 0.0, 0.0])

        self.assertEqual(action.want_grab, 0.5)
        self.assertEqual(action.want_release, 0.5)
        self.assertEqual(action.want_nurse, 0.5)
        self.assertEqual(action.flee_panic_intensity, 0.0)
        self.assertEqual(action.weight_separation, 0.0)
        self.assertEqual(action.weight_alignment, 0.0)
        self.assertEqual(action.weight_cohesion, 0.0)

    def test_flocking_outputs_are_appended_and_clamped(self) -> None:
        action = self.decide_with_outputs(
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2, 0.25, -0.2, 0.75]
        )

        self.assertEqual(action.flee_panic_intensity, 1.0)
        self.assertEqual(action.weight_separation, 0.25)
        self.assertEqual(action.weight_alignment, 0.0)
        self.assertEqual(action.weight_cohesion, 0.75)

    def test_communication_outputs_are_appended_and_neutral_at_half(self) -> None:
        neutral = self.decide_with_outputs([0.5] * 16)
        active = self.decide_with_outputs([0.5] * 12 + [0.75, 0.25, 1.0, 0.6])

        self.assertEqual(neutral.emit_sound, 0.0)
        self.assertEqual(neutral.emit_trail_pheromone, 0.0)
        self.assertEqual(neutral.emit_alarm_pheromone, 0.0)
        self.assertAlmostEqual(active.emit_sound, 0.5)
        self.assertAlmostEqual(active.sound_tone, -0.5)
        self.assertAlmostEqual(active.emit_trail_pheromone, 1.0)
        self.assertAlmostEqual(active.emit_alarm_pheromone, 0.2)


class NeatBrainNetworkCachingTest(unittest.TestCase):
    def test_from_genome_compiles_network_once_and_reuses_it(self) -> None:
        created_networks: list[FakeNetwork] = []
        fake_network = FakeNetwork([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

        class FakeFeedForwardNetwork:
            @staticmethod
            def create(genome: object, config: object) -> FakeNetwork:
                created_networks.append(fake_network)
                return fake_network

        original_nn = getattr(neat, "nn", None)
        neat.nn = SimpleNamespace(FeedForwardNetwork=FakeFeedForwardNetwork)

        try:
            config = SimpleNamespace(
                genome_config=SimpleNamespace(output_keys=[0, 1, 2, 3, 4, 5, 6])
            )
            genome = SimpleNamespace(nodes={})

            brain = NeatBrain.from_genome(1, genome, config)
            first_network = brain.network

            brain.decide(sensor_snapshot())
            brain.decide(sensor_snapshot())
            brain.decide(sensor_snapshot())
        finally:
            if original_nn is None:
                del neat.nn
            else:
                neat.nn = original_nn

        self.assertEqual(created_networks, [fake_network])
        self.assertIs(brain.network, first_network)
        self.assertEqual(fake_network.activate_count, 3)


class LegacyBrainContractMigrationTest(unittest.TestCase):
    def test_missing_output_nodes_are_added_inert_and_disconnected(self) -> None:
        class FakeNode:
            def __init__(self, key: int) -> None:
                self.key = key
                self.bias = 0.0

            def init_attributes(self, config: object) -> None:
                del config

        genome = SimpleNamespace(
            nodes={key: FakeNode(key) for key in range(8)},
            connections={},
        )
        genome_config = SimpleNamespace(
            output_keys=list(range(16)),
            node_gene_type=FakeNode,
            bias_min_value=-5.0,
        )
        controller = NeatBrainController.__new__(NeatBrainController)
        controller.config = SimpleNamespace(genome_config=genome_config)
        controller.population = SimpleNamespace(population={1: genome})
        controller.species_manager = SimpleNamespace(representatives={1: genome})

        controller.migrate_legacy_brain_contract()

        self.assertEqual(set(genome.nodes), set(range(16)))
        for key in range(8, 16):
            self.assertEqual(genome.nodes[key].bias, -5.0)
        self.assertEqual(genome.connections, {})


class NeatArchivePruningTest(unittest.TestCase):
    def test_population_archive_keeps_live_and_highest_fitness_genomes(self) -> None:
        controller = NeatBrainController.__new__(NeatBrainController)
        genomes = {
            genome_id: SimpleNamespace(key=genome_id, fitness=float(genome_id))
            for genome_id in range(1, 5001)
        }
        controller.population = SimpleNamespace(population=genomes)
        controller.brains = {
            101: SimpleNamespace(genome_id=1),
            102: SimpleNamespace(genome_id=2),
        }

        retained = controller.prune_population_archive(3)

        self.assertEqual(retained, {1, 2, 4998, 4999, 5000})
        self.assertEqual(set(controller.population.population), retained)

    def test_monotonic_genome_ids_do_not_depend_on_retained_population(self) -> None:
        controller = NeatBrainController.__new__(NeatBrainController)
        controller.population = SimpleNamespace(population={2: object()})
        controller.brains = {}
        controller.species_manager = SimpleNamespace(representatives={})
        controller._next_genome_id_value = 1000

        self.assertEqual(controller._next_genome_id(), 1000)
        controller.population.population.clear()
        self.assertEqual(controller._next_genome_id(), 1001)


class EvolutionAllocatorPersistenceTest(unittest.TestCase):
    def test_restore_reconstructs_allocators_above_every_loaded_gene(self) -> None:
        low_genome = SimpleNamespace(
            nodes={12: object()},
            connections={
                (-1, 0): SimpleNamespace(innovation=20),
            },
        )
        evolved_genome = SimpleNamespace(
            nodes={30: object()},
            connections={
                (12, 0): SimpleNamespace(innovation=50),
            },
        )
        innovation_tracker = SimpleNamespace(
            global_counter=10,
            generation_innovations={"stale": 10},
        )
        genome_config = SimpleNamespace(
            num_outputs=12,
            output_keys=list(range(12)),
            node_indexer=count(13),
            innovation_tracker=innovation_tracker,
        )
        controller = NeatBrainController.__new__(NeatBrainController)
        controller.config = SimpleNamespace(genome_config=genome_config)
        controller.population = SimpleNamespace(
            population={1: low_genome, 2: evolved_genome},
        )
        controller.species_manager = SimpleNamespace(representatives={})

        controller.restore_evolution_allocators()

        self.assertEqual(next(genome_config.node_indexer), 31)
        self.assertEqual(innovation_tracker.global_counter, 50)
        self.assertEqual(innovation_tracker.generation_innovations, {})


if __name__ == "__main__":
    unittest.main()
