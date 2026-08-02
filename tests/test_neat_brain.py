from __future__ import annotations

from dataclasses import astuple
from itertools import count
from math import tanh
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch


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

from src.action import ACTION_OUTPUT_COUNT, ACTION_OUTPUT_NAMES, BrainOutputIndex
from src.neat_brain import NeatBrain
from src.neat_controller import NeatBrainController
from src.vision import BoundarySnapshot, SensorSnapshot, VisionTargetSnapshot


class FakeNetwork:
    def __init__(self, outputs: object) -> None:
        self.outputs = outputs
        self.activate_count = 0

    def activate(self, inputs: list[float]) -> object:
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
        reproductive_readiness=0.0,
        visible_food_count=0.0,
        visible_creature_count=0.0,
        clock_tik_tok=0.0,
        clock_chronometer=0.0,
        clock_time_alive=0.0,
        is_grabbing=0.0,
    )


class NeatBrainActionMappingTest(unittest.TestCase):
    def test_output_schema_is_contiguous_and_named(self) -> None:
        self.assertEqual(
            [int(output) for output in BrainOutputIndex],
            list(range(15)),
        )
        self.assertEqual(ACTION_OUTPUT_COUNT, 15)
        self.assertEqual(ACTION_OUTPUT_NAMES[9], "herding")
        self.assertEqual(ACTION_OUTPUT_NAMES[10:], (
            "emit_sound",
            "sound_tone",
            "emit_trail_pheromone",
            "emit_alarm_pheromone",
            "rest",
        ))

    def test_rest_uses_positive_centered_evidence_only(self) -> None:
        outputs = [0.0] * ACTION_OUTPUT_COUNT
        outputs[BrainOutputIndex.REST] = 0.75

        action = self.decide_with_outputs(outputs)

        self.assertEqual(action.rest, 0.75)

    def test_transaction_shadow_does_not_advance_live_genome_allocator(self) -> None:
        controller = NeatBrainController(Path("configs/neat_herbivore.ini"))
        live_next = controller._next_genome_id_value

        shadow = controller.transaction_shadow()
        allocated = shadow._next_genome_id()

        self.assertEqual(allocated, live_next)
        self.assertEqual(controller._next_genome_id_value, live_next)

    def test_transaction_shadow_reuses_unmodified_live_objects(self) -> None:
        controller = NeatBrainController(Path("configs/neat_herbivore.ini"))

        shadow = controller.transaction_shadow()

        self.assertIsNot(shadow.population.population, controller.population.population)
        self.assertIsNot(shadow.brains, controller.brains)
        self.assertIsNot(
            shadow.species_manager.representatives,
            controller.species_manager.representatives,
        )
        for genome_id, genome in controller.population.population.items():
            self.assertIs(shadow.population.population[genome_id], genome)
        for creature_id, brain in controller.brains.items():
            self.assertIs(shadow.brains[creature_id], brain)
        for species_id, representative in (
            controller.species_manager.representatives.items()
        ):
            self.assertIs(
                shadow.species_manager.representatives[species_id],
                representative,
            )

        shadow.population.population[-1] = object()
        shadow.brains[-1] = object()
        shadow.species_manager.representatives[-1] = object()
        self.assertNotIn(-1, controller.population.population)
        self.assertNotIn(-1, controller.brains)
        self.assertNotIn(-1, controller.species_manager.representatives)

        live_tracker = getattr(
            controller.config.genome_config,
            "innovation_tracker",
            None,
        )
        shadow_tracker = getattr(
            shadow.config.genome_config,
            "innovation_tracker",
            None,
        )
        if live_tracker is not None and shadow_tracker is not None:
            live_counter = live_tracker.global_counter
            shadow_tracker.global_counter += 1
            self.assertEqual(live_tracker.global_counter, live_counter)

    def test_new_brain_has_legacy_rate_and_zero_transient_herding_state(
        self,
    ) -> None:
        brain = self.make_brain([0.0] * ACTION_OUTPUT_COUNT)

        self.assertEqual(brain.herding_decay_rate, 1.0)
        self.assertEqual(brain.herding_state, 0.0)
        self.assertEqual(brain.last_raw_herding, 0.0)

    def test_invalid_brain_decay_rates_fail(self) -> None:
        for rate in (0.0, -0.1, float("nan"), float("inf"), 1.01):
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError):
                    NeatBrain(
                        genome_id=1,
                        genome=SimpleNamespace(),
                        network=FakeNetwork([]),
                        herding_decay_rate=rate,
                    )

    def make_brain(
        self,
        outputs: object,
        activations: list[str] | None = None,
        herding_decay_rate: float = 1.0,
    ) -> NeatBrain:
        return NeatBrain(
            genome_id=1,
            genome=SimpleNamespace(),
            network=FakeNetwork(outputs),
            herding_decay_rate=herding_decay_rate,
            output_activations=(
                ["clamped"] * ACTION_OUTPUT_COUNT
                if activations is None
                else activations
            ),
        )

    def decide_with_outputs(
        self,
        outputs: object,
        activations: list[str] | None = None,
    ):
        brain = self.make_brain(outputs, activations)
        return brain.decide(sensor_snapshot())

    def test_neutral_centered_outputs_produce_neutral_action(self) -> None:
        brain = self.make_brain([0.0] * ACTION_OUTPUT_COUNT)

        action = brain.decide(sensor_snapshot())

        self.assertEqual(brain.last_outputs, [0.0] * ACTION_OUTPUT_COUNT)
        self.assertTrue(all(value == 0.0 for value in astuple(action)))

    def test_signed_controls_use_centered_values_directly(self) -> None:
        outputs = [0.0] * ACTION_OUTPUT_COUNT
        outputs[0] = -1.0
        outputs[1] = 0.4
        outputs[BrainOutputIndex.ACOUSTIC_TONE] = 1.0

        action = self.decide_with_outputs(outputs)

        self.assertEqual(action.accelerate, -1.0)
        self.assertEqual(action.rotate, 0.4)
        self.assertEqual(action.sound_tone, 1.0)

    def test_positive_outputs_discard_negative_evidence(self) -> None:
        outputs = [0.0] * ACTION_OUTPUT_COUNT
        outputs[2:6] = [-1.0, 0.0, 0.4, 1.0]

        action = self.decide_with_outputs(outputs)

        self.assertEqual(action.want_reproduce, 0.0)
        self.assertEqual(action.want_eat, 0.0)
        self.assertEqual(action.reset_chronometer, 0.4)
        self.assertEqual(action.want_grab, 1.0)

    def test_herding_uses_positive_centered_evidence_only(self) -> None:
        for centered, expected in (
            (-1.0, 0.0),
            (0.0, 0.0),
            (0.4, 0.4),
            (1.0, 1.0),
        ):
            with self.subTest(centered=centered):
                outputs = [0.0] * ACTION_OUTPUT_COUNT
                outputs[BrainOutputIndex.HERDING] = centered
                self.assertEqual(
                    self.decide_with_outputs(outputs).herding,
                    expected,
                )

    def test_herding_pulse_is_integrated_instead_of_applied_instantly(
        self,
    ) -> None:
        outputs = [0.0] * ACTION_OUTPUT_COUNT
        outputs[BrainOutputIndex.HERDING] = 1.0
        brain = self.make_brain(outputs, herding_decay_rate=0.15)

        action = brain.decide(sensor_snapshot())

        self.assertEqual(brain.last_outputs[BrainOutputIndex.HERDING], 1.0)
        self.assertEqual(brain.last_raw_herding, 1.0)
        self.assertAlmostEqual(brain.herding_state, 0.15)
        self.assertAlmostEqual(action.herding, 0.15)

    def test_sustained_herding_rises_monotonically_toward_one(self) -> None:
        outputs = [0.0] * ACTION_OUTPUT_COUNT
        outputs[BrainOutputIndex.HERDING] = 1.0
        brain = self.make_brain(outputs, herding_decay_rate=0.15)

        values = [
            brain.decide(sensor_snapshot()).herding
            for _ in range(50)
        ]

        self.assertTrue(
            all(first < second for first, second in zip(values, values[1:]))
        )
        self.assertAlmostEqual(values[-1], 1.0 - 0.85**50)
        self.assertGreater(values[-1], 0.999)

    def test_herding_decays_geometrically_after_raw_input_stops(self) -> None:
        outputs = [0.0] * ACTION_OUTPUT_COUNT
        outputs[BrainOutputIndex.HERDING] = 1.0
        brain = self.make_brain(outputs, herding_decay_rate=0.15)
        brain.decide(sensor_snapshot())
        initial_state = brain.herding_state
        brain.network.outputs[BrainOutputIndex.HERDING] = 0.0

        values = [
            brain.decide(sensor_snapshot()).herding
            for _ in range(4)
        ]

        self.assertAlmostEqual(values[0], initial_state * 0.85)
        self.assertTrue(
            all(first > second for first, second in zip(values, values[1:]))
        )
        self.assertAlmostEqual(values[-1], initial_state * 0.85**4)

    def test_legacy_rate_follows_raw_herding_exactly(self) -> None:
        outputs = [0.0] * ACTION_OUTPUT_COUNT
        brain = self.make_brain(outputs, herding_decay_rate=1.0)

        for raw in (1.0, 0.2, 0.0, 0.8):
            brain.network.outputs[BrainOutputIndex.HERDING] = raw
            action = brain.decide(sensor_snapshot())
            self.assertEqual(brain.last_raw_herding, raw)
            self.assertEqual(brain.herding_state, raw)
            self.assertEqual(action.herding, raw)

    def test_invalid_raw_herding_evidence_remains_bounded(self) -> None:
        outputs = [0.0] * ACTION_OUTPUT_COUNT
        brain = self.make_brain(outputs, herding_decay_rate=0.15)

        for raw in (-10.0, float("nan"), float("inf"), float("-inf")):
            brain.network.outputs[BrainOutputIndex.HERDING] = raw
            action = brain.decide(sensor_snapshot())
            self.assertGreaterEqual(brain.last_raw_herding, 0.0)
            self.assertLessEqual(brain.last_raw_herding, 1.0)
            self.assertGreaterEqual(action.herding, 0.0)
            self.assertLessEqual(action.herding, 1.0)

    def test_all_action_fields_remain_within_their_documented_ranges(self) -> None:
        outputs = [-2.0, 2.0] * 8

        action = self.decide_with_outputs(outputs)

        self.assertGreaterEqual(action.accelerate, -1.0)
        self.assertLessEqual(action.accelerate, 1.0)
        self.assertGreaterEqual(action.rotate, -1.0)
        self.assertLessEqual(action.rotate, 1.0)
        self.assertGreaterEqual(action.sound_tone, -1.0)
        self.assertLessEqual(action.sound_tone, 1.0)
        positive_fields = (
            "want_reproduce",
            "want_eat",
            "reset_chronometer",
            "want_grab",
            "want_release",
            "want_nurse",
            "flee_panic_intensity",
            "herding",
            "emit_sound",
            "emit_trail_pheromone",
            "emit_alarm_pheromone",
        )
        for field_name in positive_fields:
            value = getattr(action, field_name)
            self.assertGreaterEqual(value, 0.0, field_name)
            self.assertLessEqual(value, 1.0, field_name)

    def test_missing_outputs_are_filled_with_centered_neutral_values(self) -> None:
        brain = self.make_brain([0.4, -0.4])

        action = brain.decide(sensor_snapshot())

        self.assertEqual(
            brain.last_outputs,
            [0.4, -0.4] + [0.0] * (ACTION_OUTPUT_COUNT - 2),
        )
        self.assertEqual(action.accelerate, 0.4)
        self.assertEqual(action.rotate, -0.4)
        self.assertEqual(action.want_reproduce, 0.0)
        self.assertEqual(action.emit_alarm_pheromone, 0.0)

    def test_exactly_fourteen_outputs_are_preserved(self) -> None:
        outputs = [index / 20.0 for index in range(ACTION_OUTPUT_COUNT)]
        brain = self.make_brain(outputs)

        brain.decide(sensor_snapshot())

        self.assertEqual(brain.last_outputs, outputs)

    def test_excess_outputs_are_ignored(self) -> None:
        brain = self.make_brain(
            [0.0] * ACTION_OUTPUT_COUNT + [1.0, -1.0]
        )

        brain.decide(sensor_snapshot())

        self.assertEqual(brain.last_outputs, [0.0] * ACTION_OUTPUT_COUNT)

    def test_non_iterable_network_result_produces_neutral_action(self) -> None:
        brain = self.make_brain(1.0)

        action = brain.decide(sensor_snapshot())

        self.assertEqual(brain.last_outputs, [0.0] * ACTION_OUTPUT_COUNT)
        self.assertTrue(all(value == 0.0 for value in astuple(action)))

    def test_communication_outputs_use_centered_action_semantics(self) -> None:
        neutral = self.decide_with_outputs([0.0] * ACTION_OUTPUT_COUNT)
        active = self.decide_with_outputs(
            [0.0] * 10 + [0.5, -0.5, 1.0, 0.2]
        )

        self.assertEqual(neutral.emit_sound, 0.0)
        self.assertEqual(neutral.emit_trail_pheromone, 0.0)
        self.assertEqual(neutral.emit_alarm_pheromone, 0.0)
        self.assertEqual(active.emit_sound, 0.5)
        self.assertEqual(active.sound_tone, -0.5)
        self.assertEqual(active.emit_trail_pheromone, 1.0)
        self.assertEqual(active.emit_alarm_pheromone, 0.2)

    def test_relu_signed_outputs_are_intentionally_one_sided(self) -> None:
        outputs = [0.0] * ACTION_OUTPUT_COUNT
        outputs[1] = 0.4
        outputs[BrainOutputIndex.ACOUSTIC_TONE] = 1.2
        brain = self.make_brain(outputs, ["relu"] * ACTION_OUTPUT_COUNT)

        action = brain.decide(sensor_snapshot())

        self.assertEqual(action.accelerate, 0.0)
        self.assertEqual(action.rotate, 0.4)
        self.assertEqual(action.sound_tone, 1.0)


class NeatBrainOutputNormalizationTest(unittest.TestCase):
    def setUp(self) -> None:
        brain = NeatBrain(
            genome_id=1,
            genome=SimpleNamespace(),
            network=FakeNetwork([]),
        )
        self.brain = brain

    def test_sigmoid_is_remapped_to_centered_range(self) -> None:
        self.assertEqual(self.brain._center_output(0.0, "sigmoid"), -1.0)
        self.assertEqual(self.brain._center_output(0.5, "sigmoid"), 0.0)
        self.assertEqual(self.brain._center_output(1.0, "sigmoid"), 1.0)

    def test_tanh_is_clamped_as_centered(self) -> None:
        self.assertEqual(self.brain._center_output(-1.0, "tanh"), -1.0)
        self.assertEqual(self.brain._center_output(0.0, "tanh"), 0.0)
        self.assertEqual(self.brain._center_output(1.0, "tanh"), 1.0)

    def test_clamped_zero_remains_centered_neutral(self) -> None:
        self.assertEqual(self.brain._center_output(-1.0, "clamped"), -1.0)
        self.assertEqual(self.brain._center_output(0.0, "clamped"), 0.0)
        self.assertEqual(self.brain._center_output(1.0, "clamped"), 1.0)

    def test_relu_is_bounded_without_a_half_range_shift(self) -> None:
        self.assertEqual(self.brain._center_output(0.0, "relu"), 0.0)
        self.assertEqual(self.brain._center_output(0.2, "relu"), 0.2)
        self.assertEqual(self.brain._center_output(2.0, "relu"), 1.0)

    def test_lelu_uses_a_symmetric_squash(self) -> None:
        self.assertAlmostEqual(self.brain._center_output(-0.2, "lelu"), tanh(-0.2))
        self.assertEqual(self.brain._center_output(0.0, "lelu"), 0.0)
        self.assertAlmostEqual(self.brain._center_output(0.2, "lelu"), tanh(0.2))

    def test_invalid_outputs_are_centered_neutral(self) -> None:
        invalid_values = [float("nan"), float("inf"), float("-inf"), object()]
        for value in invalid_values:
            with self.subTest(value=value):
                self.assertEqual(self.brain._center_output(value, "sigmoid"), 0.0)

    def test_unsupported_activation_uses_finite_tanh_fallback(self) -> None:
        self.assertAlmostEqual(
            self.brain._center_output(0.4, "custom_activation"),
            tanh(0.4),
        )


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

    def test_from_genome_propagates_rate_and_resets_transient_state(self) -> None:
        class FakeFeedForwardNetwork:
            @staticmethod
            def create(genome: object, config: object) -> FakeNetwork:
                del genome, config
                return FakeNetwork([])

        original_nn = getattr(neat, "nn", None)
        neat.nn = SimpleNamespace(FeedForwardNetwork=FakeFeedForwardNetwork)

        try:
            config = SimpleNamespace(
                genome_config=SimpleNamespace(output_keys=[])
            )
            brain = NeatBrain.from_genome(
                1,
                SimpleNamespace(nodes={}),
                config,
                herding_decay_rate=0.15,
            )
        finally:
            if original_nn is None:
                del neat.nn
            else:
                neat.nn = original_nn

        self.assertEqual(brain.herding_decay_rate, 0.15)
        self.assertEqual(brain.herding_state, 0.0)
        self.assertEqual(brain.last_raw_herding, 0.0)

    def test_controller_factory_propagates_rate_to_rebuilt_brains(self) -> None:
        controller = NeatBrainController.__new__(NeatBrainController)
        controller.config = object()
        controller.sensor_contract = SimpleNamespace(input_names=("constant",))
        controller.herding_decay_rate = 0.15
        rebuilt_brain = SimpleNamespace(
            last_input_names=(),
            herding_decay_rate=1.0,
        )
        genome = SimpleNamespace()

        with patch(
            "src.neat_controller.NeatBrain.from_genome",
            return_value=rebuilt_brain,
        ) as from_genome:
            result = controller._brain_from_genome(7, genome)

        self.assertIs(result, rebuilt_brain)
        self.assertEqual(result.last_input_names, ("constant",))
        self.assertEqual(result.herding_decay_rate, 0.15)
        from_genome.assert_called_once_with(
            7,
            genome,
            controller.config,
        )

    def test_rebuild_reads_activations_in_configured_output_key_order(self) -> None:
        class FakeFeedForwardNetwork:
            @staticmethod
            def create(genome: object, config: object) -> FakeNetwork:
                del genome, config
                return FakeNetwork([])

        original_nn = getattr(neat, "nn", None)
        neat.nn = SimpleNamespace(FeedForwardNetwork=FakeFeedForwardNetwork)

        try:
            config = SimpleNamespace(
                genome_config=SimpleNamespace(output_keys=[12, 3, 99])
            )
            genome = SimpleNamespace(
                nodes={
                    3: SimpleNamespace(activation="TANH"),
                    12: SimpleNamespace(activation="sigmoid"),
                    99: SimpleNamespace(activation="clamped"),
                }
            )

            first_brain = NeatBrain.from_genome(1, genome, config)
            genome.nodes[3].activation = "lelu"
            rebuilt_brain = NeatBrain.from_genome(2, genome, config)
        finally:
            if original_nn is None:
                del neat.nn
            else:
                neat.nn = original_nn

        self.assertEqual(
            first_brain.output_activations,
            ["sigmoid", "tanh", "clamped"],
        )
        self.assertEqual(
            rebuilt_brain.output_activations,
            ["sigmoid", "lelu", "clamped"],
        )


class NeatConfigurationTest(unittest.TestCase):
    def test_herbivore_config_enables_supported_activation_and_aggregation_modes(
        self,
    ) -> None:
        if not hasattr(neat, "Config"):
            self.skipTest("neat-python is not installed")
        config_path = (
            Path(__file__).resolve().parents[1] / "configs" / "neat_herbivore.ini"
        )

        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(config_path),
        )

        genome_config = config.genome_config
        self.assertEqual(len(genome_config.output_keys), ACTION_OUTPUT_COUNT)
        self.assertEqual(genome_config.activation_default, "sigmoid")
        self.assertEqual(genome_config.activation_mutate_rate, 0.01)
        self.assertEqual(
            genome_config.activation_options,
            ["sigmoid", "tanh", "clamped", "relu", "lelu"],
        )
        self.assertEqual(genome_config.aggregation_default, "sum")
        self.assertEqual(genome_config.aggregation_mutate_rate, 0.005)
        self.assertEqual(
            genome_config.aggregation_options,
            ["sum", "mean", "maxabs"],
        )


class SensorUsageTest(unittest.TestCase):
    def test_direct_hidden_disabled_and_disconnected_paths(self) -> None:
        def gene(source: int, target: int, enabled: bool = True) -> SimpleNamespace:
            return SimpleNamespace(key=(source, target), enabled=enabled)

        genome = SimpleNamespace(
            connections={
                (-1, 0): gene(-1, 0),
                (-2, 10): gene(-2, 10),
                (10, 1): gene(10, 1),
                (-3, 0): gene(-3, 0, enabled=False),
                (-4, 11): gene(-4, 11),
            }
        )
        brain = NeatBrain(
            genome_id=1,
            genome=genome,
            network=FakeNetwork([]),
            last_inputs=[0.1, 0.2, 0.3, 0.4],
        )

        usage = brain.sensor_usage([-1, -2, -3, -4], [0, 1])

        self.assertEqual(usage[0].input_name, "constant")
        self.assertEqual(usage[0].current_value, 0.1)
        self.assertTrue(usage[0].has_enabled_path)
        self.assertEqual(usage[0].reachable_action_outputs, ("accelerate",))
        self.assertTrue(usage[1].has_enabled_path)
        self.assertEqual(usage[1].reachable_action_outputs, ("rotate",))
        self.assertFalse(usage[2].has_enabled_path)
        self.assertEqual(usage[2].reachable_action_outputs, ())
        self.assertFalse(usage[3].has_enabled_path)

    def test_one_sensor_reaches_multiple_outputs_in_action_order(self) -> None:
        def gene(source: int, target: int) -> SimpleNamespace:
            return SimpleNamespace(key=(source, target), enabled=True)

        genome = SimpleNamespace(
            connections={
                (-1, 10): gene(-1, 10),
                (10, 2): gene(10, 2),
                (-1, 0): gene(-1, 0),
            }
        )
        brain = NeatBrain(
            genome_id=1,
            genome=genome,
            network=FakeNetwork([]),
            last_inputs=[0.1],
        )

        usage = brain.sensor_usage([-1], [0, 1, 2])

        self.assertTrue(usage[0].has_enabled_path)
        self.assertEqual(
            usage[0].reachable_action_outputs,
            ("accelerate", "want_reproduce"),
        )

    def test_disabled_connections_are_not_passed_to_neat_graph_helper(self) -> None:
        def gene(source: int, target: int, enabled: bool) -> SimpleNamespace:
            return SimpleNamespace(key=(source, target), enabled=enabled)

        genome = SimpleNamespace(
            connections={
                (-1, 0): gene(-1, 0, True),
                (-2, 1): gene(-2, 1, False),
            }
        )
        brain = NeatBrain(
            genome_id=1,
            genome=genome,
            network=FakeNetwork([]),
            last_inputs=[0.1, 0.2],
        )

        with patch(
            "src.neat_brain.required_for_output",
            wraps=neat.graphs.required_for_output,
        ) as graph_helper:
            usage = brain.sensor_usage([-1, -2], [0, 1])

        self.assertTrue(usage[0].has_enabled_path)
        self.assertFalse(usage[1].has_enabled_path)
        self.assertEqual(graph_helper.call_count, 2)
        for helper_call in graph_helper.call_args_list:
            self.assertEqual(helper_call.args[2], [(-1, 0)])


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
