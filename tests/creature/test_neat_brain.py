from __future__ import annotations

from dataclasses import astuple
from itertools import count
from math import tanh
from pathlib import Path
from random import Random
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch


class _Body:
    def __init__(self, *args: object, **kwargs: object) -> None:
        """Exercise init behavior.
        
        Parameters
        ----------
        args
            Value supplied to ``args`` by the test scenario.
        kwargs
            Value supplied to ``kwargs`` by the test scenario.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the init test intent explicit.
        self.position = SimpleNamespace(x=0.0, y=0.0)


class _Circle:
    def __init__(self, body: _Body, radius: float) -> None:
        """Exercise init behavior.
        
        Parameters
        ----------
        body
            Value supplied to ``body`` by the test scenario.
        radius
            Value supplied to ``radius`` by the test scenario.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the init test intent explicit.
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

from src.action import (
    ACTION_OUTPUT_COUNT,
    ACTION_OUTPUT_NAMES,
    Action,
    BrainOutputIndex,
)
from src.neat_brain import NeatBrain
from src.neat_controller import NeatBrainController
from src.creature.genotype import FlockingTraits, PhysicalTraits, VisionTraits
from src.vision import (
    BoundarySnapshot,
    SENSOR_CONTRACT,
    SensorSnapshot,
    VisionTargetSnapshot,
)


class FakeNetwork:
    def __init__(self, outputs: object) -> None:
        """Exercise init behavior.
        
        Parameters
        ----------
        outputs
            Value supplied to ``outputs`` by the test scenario.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the init test intent explicit.
        self.outputs = outputs
        self.activate_count = 0
        self.input_references: list[list[float]] = []
        self.input_values: list[list[float]] = []

    def activate(self, inputs: list[float]) -> object:
        """Exercise activate behavior.
        
        Parameters
        ----------
        inputs
            Value supplied to ``inputs`` by the test scenario.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the activate test intent explicit.
        self.activate_count += 1
        self.input_references.append(inputs)
        self.input_values.append(list(inputs))
        return self.outputs


def empty_target() -> VisionTargetSnapshot:
    """Exercise empty target behavior.
    
    Parameters
    ----------
    None
        This callable receives no external parameters.
    
    Returns
    -------
    None
        The test completes through assertions.
    """
    # Keep the empty target test intent explicit.
    return VisionTargetSnapshot(
        visible=0.0,
        proximity=0.0,
        angle=0.0,
        density=0.0,
        count=0,
    )


def sensor_snapshot() -> SensorSnapshot:
    """Exercise sensor snapshot behavior.
    
    Parameters
    ----------
    None
        This callable receives no external parameters.
    
    Returns
    -------
    None
        The test completes through assertions.
    """
    # Keep the sensor snapshot test intent explicit.
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
    def test_shipped_config_matches_runtime_sensor_contract(self) -> None:
        """Exercise test shipped config matches runtime sensor contract behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test shipped config matches runtime sensor contract test intent explicit.
        controller = NeatBrainController(Path("configs/neat_herbivore.ini"))

        self.assertEqual(
            len(controller.config.genome_config.input_keys),
            SENSOR_CONTRACT.input_count,
        )

    def test_runtime_input_count_mismatch_fails_before_activation(self) -> None:
        """Exercise test runtime input count mismatch fails before activation behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test runtime input count mismatch fails before activation test intent explicit.
        network = FakeNetwork([0.0] * ACTION_OUTPUT_COUNT)
        network.input_nodes = list(range(SENSOR_CONTRACT.input_count - 1))
        brain = NeatBrain(
            genome_id=1,
            genome=SimpleNamespace(),
            network=network,
        )

        with self.assertRaisesRegex(RuntimeError, "input count mismatch"):
            brain.decide(sensor_snapshot())

        self.assertEqual(network.activate_count, 0)

    def test_startup_contract_rejects_mismatched_config_input_count(self) -> None:
        """Exercise test startup contract rejects mismatched config input count behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test startup contract rejects mismatched config input count test intent explicit.
        controller = NeatBrainController.__new__(NeatBrainController)
        controller.config = SimpleNamespace(
            genome_config=SimpleNamespace(
                input_keys=list(range(SENSOR_CONTRACT.input_count - 1)),
                output_keys=list(range(ACTION_OUTPUT_COUNT)),
            )
        )
        controller.sensor_contract = SENSOR_CONTRACT

        with self.assertRaisesRegex(ValueError, "input count mismatch"):
            controller._validate_network_contract()

    def test_output_schema_is_contiguous_and_named(self) -> None:
        """Exercise test output schema is contiguous and named behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test output schema is contiguous and named test intent explicit.
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
        """Exercise test rest uses positive centered evidence only behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test rest uses positive centered evidence only test intent explicit.
        outputs = [0.0] * ACTION_OUTPUT_COUNT
        outputs[BrainOutputIndex.REST] = 0.75

        action = self.decide_with_outputs(outputs)

        self.assertEqual(action.rest, 0.75)

    def test_transaction_shadow_does_not_advance_live_genome_allocator(self) -> None:
        """Exercise test transaction shadow does not advance live genome allocator behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test transaction shadow does not advance live genome allocator test intent explicit.
        controller = NeatBrainController(Path("configs/neat_herbivore.ini"))
        live_next = controller._next_genome_id_value

        shadow = controller.transaction_shadow()
        allocated = shadow._next_genome_id()

        self.assertEqual(allocated, live_next)
        self.assertEqual(controller._next_genome_id_value, live_next)

    def test_transaction_shadow_reuses_unmodified_live_objects(self) -> None:
        """Exercise test transaction shadow reuses unmodified live objects behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test transaction shadow reuses unmodified live objects test intent explicit.
        controller = NeatBrainController(Path("configs/neat_herbivore.ini"))
        representative_genome = next(
            iter(controller.population.population.values())
        )
        controller.species_manager.register_initial_representative(
            representative_genome,
            PhysicalTraits(radius=16.0),
            VisionTraits(range=150.0, angle=0.95),
            flocking_traits=FlockingTraits(),
        )

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
            shadow_representative = shadow.species_manager.representatives[
                species_id
            ]
            self.assertIsNot(shadow_representative, representative)
            self.assertIs(shadow_representative[0], representative[0])
            for shadow_traits, live_traits in zip(
                shadow_representative[1:],
                representative[1:],
            ):
                self.assertIsNot(shadow_traits, live_traits)
                self.assertEqual(shadow_traits, live_traits)

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
        """Exercise test new brain has legacy rate and zero transient herding state behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test new brain has legacy rate and zero transient herding state test intent explicit.
        brain = self.make_brain([0.0] * ACTION_OUTPUT_COUNT)

        self.assertEqual(brain.herding_decay_rate, 1.0)
        self.assertEqual(brain.herding_state, 0.0)
        self.assertEqual(brain.last_raw_herding, 0.0)

    def test_activation_buffer_is_reused_without_publishing_inspector_state(
        self,
    ) -> None:
        """Exercise test activation buffer is reused without publishing inspector state behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test activation buffer is reused without publishing inspector state test intent explicit.
        brain = self.make_brain([0.0] * ACTION_OUTPUT_COUNT)
        snapshot = sensor_snapshot()
        buffer_id = id(brain._input_buffer)

        brain.decide(snapshot)
        first_values = snapshot.as_inputs()
        snapshot.energy = 0.25
        brain.decide(snapshot)

        self.assertEqual(id(brain._input_buffer), buffer_id)
        self.assertIs(brain.network.input_references[0], brain._input_buffer)
        self.assertIs(brain.network.input_references[1], brain._input_buffer)
        self.assertEqual(brain.network.input_values[0], first_values)
        self.assertEqual(brain.network.input_values[1], snapshot.as_inputs())
        self.assertEqual(brain.last_inputs, [])

    def test_captured_inputs_remain_stable_until_intentional_recapture(
        self,
    ) -> None:
        """Exercise test captured inputs remain stable until intentional recapture behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test captured inputs remain stable until intentional recapture test intent explicit.
        brain = self.make_brain([0.0] * ACTION_OUTPUT_COUNT)
        snapshot = sensor_snapshot()

        brain.decide(snapshot, capture_inputs=True)
        captured = brain.last_inputs
        captured_values = list(captured)
        snapshot.energy = 0.25
        brain.decide(snapshot)

        self.assertIs(brain.last_inputs, captured)
        self.assertEqual(captured, captured_values)
        brain.capture_input_snapshot()
        self.assertIsNot(brain.last_inputs, captured)
        self.assertEqual(brain.last_inputs, snapshot.as_inputs())
        self.assertEqual(captured, captured_values)

    def test_live_decision_returns_an_already_clamped_action(self) -> None:
        """Exercise test live decision returns an already clamped action behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test live decision returns an already clamped action test intent explicit.
        outputs = [-2.0, 2.0] * 8
        brain = self.make_brain(outputs)

        with patch.object(
            Action,
            "clamped",
            side_effect=AssertionError("live path called Action.clamped"),
        ):
            action = brain.decide(sensor_snapshot())

        centered = brain.last_outputs
        positive = lambda index: max(0.0, min(1.0, centered[index]))
        legacy_action = Action(
            accelerate=centered[BrainOutputIndex.ACCELERATE],
            rotate=centered[BrainOutputIndex.ROTATE],
            want_reproduce=positive(BrainOutputIndex.REPRODUCE),
            want_eat=positive(BrainOutputIndex.EAT),
            reset_chronometer=positive(BrainOutputIndex.RESET_CHRONOMETER),
            want_grab=positive(BrainOutputIndex.GRAB_FOOD),
            want_release=positive(BrainOutputIndex.RELEASE_FOOD),
            want_nurse=positive(BrainOutputIndex.NURSE),
            flee_panic_intensity=positive(BrainOutputIndex.PANIC),
            herding=brain.herding_state,
            emit_sound=positive(BrainOutputIndex.ACOUSTIC_EMISSION),
            sound_tone=centered[BrainOutputIndex.ACOUSTIC_TONE],
            emit_trail_pheromone=positive(BrainOutputIndex.TRAIL_PHEROMONE),
            emit_alarm_pheromone=positive(BrainOutputIndex.ALARM_PHEROMONE),
            rest=positive(BrainOutputIndex.REST),
        ).clamped()

        self.assertIs(brain.last_action, action)
        self.assertEqual(action, legacy_action)
        self.assertEqual(action, action.clamped())

    def test_invalid_brain_decay_rates_fail(self) -> None:
        """Exercise test invalid brain decay rates fail behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test invalid brain decay rates fail test intent explicit.
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
        """Exercise make brain behavior.
        
        Parameters
        ----------
        outputs
            Value supplied to ``outputs`` by the test scenario.
        activations
            Value supplied to ``activations`` by the test scenario.
        herding_decay_rate
            Value supplied to ``herding_decay_rate`` by the test scenario.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the make brain test intent explicit.
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
        """Exercise decide with outputs behavior.
        
        Parameters
        ----------
        outputs
            Value supplied to ``outputs`` by the test scenario.
        activations
            Value supplied to ``activations`` by the test scenario.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the decide with outputs test intent explicit.
        brain = self.make_brain(outputs, activations)
        return brain.decide(sensor_snapshot())

    def test_neutral_centered_outputs_produce_neutral_action(self) -> None:
        """Exercise test neutral centered outputs produce neutral action behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test neutral centered outputs produce neutral action test intent explicit.
        brain = self.make_brain([0.0] * ACTION_OUTPUT_COUNT)

        action = brain.decide(sensor_snapshot())

        self.assertEqual(brain.last_outputs, [0.0] * ACTION_OUTPUT_COUNT)
        self.assertTrue(all(value == 0.0 for value in astuple(action)))

    def test_signed_controls_use_centered_values_directly(self) -> None:
        """Exercise test signed controls use centered values directly behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test signed controls use centered values directly test intent explicit.
        outputs = [0.0] * ACTION_OUTPUT_COUNT
        outputs[0] = -1.0
        outputs[1] = 0.4
        outputs[BrainOutputIndex.ACOUSTIC_TONE] = 1.0

        action = self.decide_with_outputs(outputs)

        self.assertEqual(action.accelerate, -1.0)
        self.assertEqual(action.rotate, 0.4)
        self.assertEqual(action.sound_tone, 1.0)

    def test_positive_outputs_discard_negative_evidence(self) -> None:
        """Exercise test positive outputs discard negative evidence behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test positive outputs discard negative evidence test intent explicit.
        outputs = [0.0] * ACTION_OUTPUT_COUNT
        outputs[2:6] = [-1.0, 0.0, 0.4, 1.0]

        action = self.decide_with_outputs(outputs)

        self.assertEqual(action.want_reproduce, 0.0)
        self.assertEqual(action.want_eat, 0.0)
        self.assertEqual(action.reset_chronometer, 0.4)
        self.assertEqual(action.want_grab, 1.0)

    def test_herding_uses_positive_centered_evidence_only(self) -> None:
        """Exercise test herding uses positive centered evidence only behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test herding uses positive centered evidence only test intent explicit.
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
        """Exercise test herding pulse is integrated instead of applied instantly behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test herding pulse is integrated instead of applied instantly test intent explicit.
        outputs = [0.0] * ACTION_OUTPUT_COUNT
        outputs[BrainOutputIndex.HERDING] = 1.0
        brain = self.make_brain(outputs, herding_decay_rate=0.15)

        action = brain.decide(sensor_snapshot())

        self.assertEqual(brain.last_outputs[BrainOutputIndex.HERDING], 1.0)
        self.assertEqual(brain.last_raw_herding, 1.0)
        self.assertAlmostEqual(brain.herding_state, 0.15)
        self.assertAlmostEqual(action.herding, 0.15)

    def test_sustained_herding_rises_monotonically_toward_one(self) -> None:
        """Exercise test sustained herding rises monotonically toward one behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test sustained herding rises monotonically toward one test intent explicit.
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
        """Exercise test herding decays geometrically after raw input stops behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test herding decays geometrically after raw input stops test intent explicit.
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
        """Exercise test legacy rate follows raw herding exactly behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test legacy rate follows raw herding exactly test intent explicit.
        outputs = [0.0] * ACTION_OUTPUT_COUNT
        brain = self.make_brain(outputs, herding_decay_rate=1.0)

        for raw in (1.0, 0.2, 0.0, 0.8):
            brain.network.outputs[BrainOutputIndex.HERDING] = raw
            action = brain.decide(sensor_snapshot())
            self.assertEqual(brain.last_raw_herding, raw)
            self.assertEqual(brain.herding_state, raw)
            self.assertEqual(action.herding, raw)

    def test_invalid_raw_herding_evidence_remains_bounded(self) -> None:
        """Exercise test invalid raw herding evidence remains bounded behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test invalid raw herding evidence remains bounded test intent explicit.
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
        """Exercise test all action fields remain within their documented ranges behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test all action fields remain within their documented ranges test intent explicit.
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
        """Exercise test missing outputs are filled with centered neutral values behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test missing outputs are filled with centered neutral values test intent explicit.
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
        """Exercise test exactly fourteen outputs are preserved behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test exactly fourteen outputs are preserved test intent explicit.
        outputs = [index / 20.0 for index in range(ACTION_OUTPUT_COUNT)]
        brain = self.make_brain(outputs)

        brain.decide(sensor_snapshot())

        self.assertEqual(brain.last_outputs, outputs)

    def test_excess_outputs_are_ignored(self) -> None:
        """Exercise test excess outputs are ignored behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test excess outputs are ignored test intent explicit.
        brain = self.make_brain(
            [0.0] * ACTION_OUTPUT_COUNT + [1.0, -1.0]
        )

        brain.decide(sensor_snapshot())

        self.assertEqual(brain.last_outputs, [0.0] * ACTION_OUTPUT_COUNT)

    def test_non_iterable_network_result_produces_neutral_action(self) -> None:
        """Exercise test non iterable network result produces neutral action behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test non iterable network result produces neutral action test intent explicit.
        brain = self.make_brain(1.0)

        action = brain.decide(sensor_snapshot())

        self.assertEqual(brain.last_outputs, [0.0] * ACTION_OUTPUT_COUNT)
        self.assertTrue(all(value == 0.0 for value in astuple(action)))

    def test_communication_outputs_use_centered_action_semantics(self) -> None:
        """Exercise test communication outputs use centered action semantics behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test communication outputs use centered action semantics test intent explicit.
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
        """Exercise test relu signed outputs are intentionally one sided behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test relu signed outputs are intentionally one sided test intent explicit.
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
        """Exercise setUp behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the setUp test intent explicit.
        brain = NeatBrain(
            genome_id=1,
            genome=SimpleNamespace(),
            network=FakeNetwork([]),
        )
        self.brain = brain

    def test_sigmoid_is_remapped_to_centered_range(self) -> None:
        """Exercise test sigmoid is remapped to centered range behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test sigmoid is remapped to centered range test intent explicit.
        self.assertEqual(self.brain._center_output(0.0, "sigmoid"), -1.0)
        self.assertEqual(self.brain._center_output(0.5, "sigmoid"), 0.0)
        self.assertEqual(self.brain._center_output(1.0, "sigmoid"), 1.0)

    def test_tanh_is_clamped_as_centered(self) -> None:
        """Exercise test tanh is clamped as centered behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test tanh is clamped as centered test intent explicit.
        self.assertEqual(self.brain._center_output(-1.0, "tanh"), -1.0)
        self.assertEqual(self.brain._center_output(0.0, "tanh"), 0.0)
        self.assertEqual(self.brain._center_output(1.0, "tanh"), 1.0)

    def test_clamped_zero_remains_centered_neutral(self) -> None:
        """Exercise test clamped zero remains centered neutral behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test clamped zero remains centered neutral test intent explicit.
        self.assertEqual(self.brain._center_output(-1.0, "clamped"), -1.0)
        self.assertEqual(self.brain._center_output(0.0, "clamped"), 0.0)
        self.assertEqual(self.brain._center_output(1.0, "clamped"), 1.0)

    def test_relu_is_bounded_without_a_half_range_shift(self) -> None:
        """Exercise test relu is bounded without a half range shift behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test relu is bounded without a half range shift test intent explicit.
        self.assertEqual(self.brain._center_output(0.0, "relu"), 0.0)
        self.assertEqual(self.brain._center_output(0.2, "relu"), 0.2)
        self.assertEqual(self.brain._center_output(2.0, "relu"), 1.0)

    def test_lelu_uses_a_symmetric_squash(self) -> None:
        """Exercise test lelu uses a symmetric squash behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test lelu uses a symmetric squash test intent explicit.
        self.assertAlmostEqual(self.brain._center_output(-0.2, "lelu"), tanh(-0.2))
        self.assertEqual(self.brain._center_output(0.0, "lelu"), 0.0)
        self.assertAlmostEqual(self.brain._center_output(0.2, "lelu"), tanh(0.2))

    def test_invalid_outputs_are_centered_neutral(self) -> None:
        """Exercise test invalid outputs are centered neutral behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test invalid outputs are centered neutral test intent explicit.
        invalid_values = [float("nan"), float("inf"), float("-inf"), object()]
        for value in invalid_values:
            with self.subTest(value=value):
                self.assertEqual(self.brain._center_output(value, "sigmoid"), 0.0)

    def test_unsupported_activation_uses_finite_tanh_fallback(self) -> None:
        """Exercise test unsupported activation uses finite tanh fallback behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test unsupported activation uses finite tanh fallback test intent explicit.
        self.assertAlmostEqual(
            self.brain._center_output(0.4, "custom_activation"),
            tanh(0.4),
        )


class NeatBrainNetworkCachingTest(unittest.TestCase):
    def test_from_genome_compiles_network_once_and_reuses_it(self) -> None:
        """Exercise test from genome compiles network once and reuses it behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test from genome compiles network once and reuses it test intent explicit.
        created_networks: list[FakeNetwork] = []
        fake_network = FakeNetwork([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

        class FakeRecurrentNetwork:
            @staticmethod
            def create(genome: object, config: object) -> FakeNetwork:
                """Exercise create behavior.
                
                Parameters
                ----------
                genome
                    Value supplied to ``genome`` by the test scenario.
                config
                    Value supplied to ``config`` by the test scenario.
                
                Returns
                -------
                None
                    The test completes through assertions.
                """
                # Keep the create test intent explicit.
                created_networks.append(fake_network)
                return fake_network

        original_nn = getattr(neat, "nn", None)
        neat.nn = SimpleNamespace(RecurrentNetwork=FakeRecurrentNetwork)

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
        """Exercise test from genome propagates rate and resets transient state behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test from genome propagates rate and resets transient state test intent explicit.
        class FakeRecurrentNetwork:
            @staticmethod
            def create(genome: object, config: object) -> FakeNetwork:
                """Exercise create behavior.
                
                Parameters
                ----------
                genome
                    Value supplied to ``genome`` by the test scenario.
                config
                    Value supplied to ``config`` by the test scenario.
                
                Returns
                -------
                None
                    The test completes through assertions.
                """
                # Keep the create test intent explicit.
                del genome, config
                return FakeNetwork([])

        original_nn = getattr(neat, "nn", None)
        neat.nn = SimpleNamespace(RecurrentNetwork=FakeRecurrentNetwork)

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
        """Exercise test controller factory propagates rate to rebuilt brains behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test controller factory propagates rate to rebuilt brains test intent explicit.
        controller = NeatBrainController.__new__(NeatBrainController)
        controller.config = SimpleNamespace(
            genome_config=SimpleNamespace(output_keys=[])
        )
        controller.sensor_contract = SimpleNamespace(input_names=("constant",))
        controller.herding_decay_rate = 0.15
        rebuilt_brain = SimpleNamespace(
            last_input_names=(),
            herding_decay_rate=1.0,
        )
        genome = SimpleNamespace()

        with patch(
            "src.creature.neat.controller.NeatBrain.from_genome",
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
        """Exercise test rebuild reads activations in configured output key order behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test rebuild reads activations in configured output key order test intent explicit.
        class FakeRecurrentNetwork:
            @staticmethod
            def create(genome: object, config: object) -> FakeNetwork:
                """Exercise create behavior.
                
                Parameters
                ----------
                genome
                    Value supplied to ``genome`` by the test scenario.
                config
                    Value supplied to ``config`` by the test scenario.
                
                Returns
                -------
                None
                    The test completes through assertions.
                """
                # Keep the create test intent explicit.
                del genome, config
                return FakeNetwork([])

        original_nn = getattr(neat, "nn", None)
        neat.nn = SimpleNamespace(RecurrentNetwork=FakeRecurrentNetwork)

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
        """Exercise test herbivore config enables supported activation and aggregation modes behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test herbivore config enables supported activation and aggregation modes test intent explicit.
        if not hasattr(neat, "Config"):
            self.skipTest("neat-python is not installed")
        config_path = (
            Path(__file__).resolve().parents[2] / "configs" / "neat_herbivore.ini"
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
            ["sigmoid", "tanh", "clamped"],
        )
        self.assertFalse(genome_config.feed_forward)
        self.assertEqual(genome_config.aggregation_default, "sum")
        self.assertEqual(genome_config.aggregation_mutate_rate, 0.005)
        self.assertEqual(
            genome_config.aggregation_options,
            ["sum", "mean", "maxabs"],
        )

    def test_configured_genomes_create_distinct_43_by_15_recurrent_networks(
        self,
    ) -> None:
        """Compile configured genomes into isolated recurrent networks.

        Parameters
        ----------
        None
            This callable receives no external parameters.
        Returns
        -------
        None
            Assertions verify network type, dimensions, and clean state.
        """
        # Exercise the real neat-python phenotype factory used by every agent.
        if not hasattr(neat, "Config"):
            self.skipTest("neat-python is not installed")
        config_path = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "neat_herbivore.ini"
        )
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(config_path),
        )
        population = neat.Population(config)
        genome_id, genome = next(iter(population.population.items()))

        first = NeatBrain.from_genome(genome_id, genome, config)
        second = NeatBrain.from_genome(genome_id, genome, config)

        self.assertIsInstance(first.network, neat.nn.RecurrentNetwork)
        self.assertIsNot(first.network, second.network)
        self.assertEqual(len(first.network.input_nodes), SENSOR_CONTRACT.input_count)
        self.assertEqual(
            len(first.network.activate([0.0] * SENSOR_CONTRACT.input_count)),
            ACTION_OUTPUT_COUNT,
        )
        self.assertEqual(second.network.active, 0)
        self.assertTrue(
            all(
                value == 0.0
                for buffer in second.network.values
                for value in buffer.values()
            )
        )


@unittest.skipUnless(
    hasattr(getattr(neat, "nn", None), "RecurrentNetwork"),
    "neat-python is not installed",
)
class RecurrentNetworkStateTest(unittest.TestCase):
    @staticmethod
    def identity(value: float) -> float:
        """Return an input unchanged for deterministic recurrent tests.

        Parameters
        ----------
        value
            Scalar node input.
        Returns
        -------
        float
            Unchanged node value.
        """
        # Avoid nonlinear activation effects in propagation assertions.
        return value

    @classmethod
    def make_network(cls):
        """Build a three-hop discrete recurrent test network.

        Parameters
        ----------
        None
            This callable receives no external parameters.
        Returns
        -------
        neat.nn.RecurrentNetwork
            Deterministic input-hidden-hidden-output chain.
        """
        # Construct node evaluations directly to control every hop.
        return neat.nn.RecurrentNetwork(
            [-1],
            [0],
            [
                (1, cls.identity, sum, 0.0, 1.0, [(-1, 1.0)]),
                (2, cls.identity, sum, 0.0, 1.0, [(1, 1.0)]),
                (0, cls.identity, sum, 0.0, 1.0, [(2, 1.0)]),
            ],
        )

    @classmethod
    def make_brain(cls) -> NeatBrain:
        """Wrap a deterministic recurrent network in a brain.

        Parameters
        ----------
        None
            This callable receives no external parameters.
        Returns
        -------
        NeatBrain
            Brain suitable for state snapshot and cloning tests.
        """
        # Keep the fixture independent from genome compilation randomness.
        return NeatBrain(
            genome_id=1,
            genome=SimpleNamespace(),
            network=cls.make_network(),
            output_activations=["clamped"],
        )

    def test_multi_hop_signal_advances_one_connection_per_tick(self) -> None:
        """Verify synchronous recurrent propagation latency.

        Parameters
        ----------
        None
            This callable receives no external parameters.
        Returns
        -------
        None
            Assertions confirm one connection is crossed per activation.
        """
        # Three connections require three consecutive activation ticks.
        network = self.make_network()

        self.assertEqual(network.activate([1.0]), [0.0])
        self.assertEqual(network.activate([1.0]), [0.0])
        self.assertEqual(network.activate([1.0]), [1.0])

    def test_state_export_is_shallow_isolated_and_restorable(self) -> None:
        """Verify shallow state copies are isolated and restorable.

        Parameters
        ----------
        None
            This callable receives no external parameters.
        Returns
        -------
        None
            Assertions validate copied dictionary identities and values.
        """
        # Advance first so both the active index and node buffers are meaningful.
        brain = self.make_brain()
        brain.network.activate([1.0])
        exported = brain.export_network_state()

        self.assertIsNotNone(exported)
        self.assertEqual(exported["active"], brain.network.active)
        self.assertEqual(exported["values"], brain.network.values)
        self.assertIsNot(exported["values"], brain.network.values)
        self.assertIsNot(exported["values"][0], brain.network.values[0])
        self.assertIsNot(exported["values"][1], brain.network.values[1])

        brain.network.activate([0.0])
        brain.restore_network_state(exported)
        self.assertEqual(brain.export_network_state(), exported)

    def test_restore_rejects_missing_and_extra_node_ids(self) -> None:
        """Reject recurrent state whose node IDs differ from topology.

        Parameters
        ----------
        None
            This callable receives no external parameters.
        Returns
        -------
        None
            Both missing and unexpected node IDs raise validation errors.
        """
        # Mutate copies so failed restores cannot affect the live buffers.
        brain = self.make_brain()
        exported = brain.export_network_state()
        missing = {
            "active": exported["active"],
            "values": [dict(exported["values"][0]), dict(exported["values"][1])],
        }
        missing["values"][0].pop(next(iter(missing["values"][0])))
        with self.assertRaisesRegex(ValueError, "node IDs do not match"):
            brain.restore_network_state(missing)

        extra = {
            "active": exported["active"],
            "values": [dict(exported["values"][0]), dict(exported["values"][1])],
        }
        extra["values"][1][999] = 0.0
        with self.assertRaisesRegex(ValueError, "node IDs do not match"):
            brain.restore_network_state(extra)

    def test_clone_and_agents_have_isolated_recurrent_buffers(self) -> None:
        """Verify clones and creature brains do not share state buffers.

        Parameters
        ----------
        None
            This callable receives no external parameters.
        Returns
        -------
        None
            State changes remain local to one cloned or creature network.
        """
        # Create independent brains from identical test topology.
        first = self.make_brain()
        second = self.make_brain()
        first.network.activate([1.0])
        clone = first.clone_network()

        self.assertIsNot(clone, first.network)
        self.assertEqual(clone.values, first.network.values)
        self.assertIsNot(clone.values[0], first.network.values[0])
        clone.activate([0.0])
        self.assertNotEqual(clone.active, first.network.active)
        self.assertEqual(second.network.activate([0.0]), [0.0])
        self.assertEqual(second.network.values[0][1], 0.0)


class SensorUsageTest(unittest.TestCase):
    def test_direct_hidden_disabled_and_disconnected_paths(self) -> None:
        """Exercise test direct hidden disabled and disconnected paths behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test direct hidden disabled and disconnected paths test intent explicit.
        def gene(source: int, target: int, enabled: bool = True) -> SimpleNamespace:
            """Exercise gene behavior.
            
            Parameters
            ----------
            source
                Value supplied to ``source`` by the test scenario.
            target
                Value supplied to ``target`` by the test scenario.
            enabled
                Value supplied to ``enabled`` by the test scenario.
            
            Returns
            -------
            None
                The test completes through assertions.
            """
            # Keep the gene test intent explicit.
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
        """Exercise test one sensor reaches multiple outputs in action order behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test one sensor reaches multiple outputs in action order test intent explicit.
        def gene(source: int, target: int) -> SimpleNamespace:
            """Exercise gene behavior.
            
            Parameters
            ----------
            source
                Value supplied to ``source`` by the test scenario.
            target
                Value supplied to ``target`` by the test scenario.
            
            Returns
            -------
            None
                The test completes through assertions.
            """
            # Keep the gene test intent explicit.
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
        """Exercise test disabled connections are not passed to neat graph helper behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test disabled connections are not passed to neat graph helper test intent explicit.
        def gene(source: int, target: int, enabled: bool) -> SimpleNamespace:
            """Exercise gene behavior.
            
            Parameters
            ----------
            source
                Value supplied to ``source`` by the test scenario.
            target
                Value supplied to ``target`` by the test scenario.
            enabled
                Value supplied to ``enabled`` by the test scenario.
            
            Returns
            -------
            None
                The test completes through assertions.
            """
            # Keep the gene test intent explicit.
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
            "src.creature.neat.brain.required_for_output",
            wraps=neat.graphs.required_for_output,
        ) as graph_helper:
            usage = brain.sensor_usage([-1, -2], [0, 1])

        self.assertTrue(usage[0].has_enabled_path)
        self.assertFalse(usage[1].has_enabled_path)
        self.assertEqual(graph_helper.call_count, 2)
        for helper_call in graph_helper.call_args_list:
            self.assertEqual(helper_call.args[2], [(-1, 0)])


class NeatArchivePruningTest(unittest.TestCase):
    def test_population_archive_keeps_live_and_unranked_sample(self) -> None:
        """Exercise test population archive keeps live and highest fitness genomes behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test population archive keeps live and highest fitness genomes test intent explicit.
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
        controller._evolution_rng = Random(7)

        retained = controller.prune_population_archive(3)

        self.assertEqual({1, 2}, retained & {1, 2})
        self.assertEqual(len(retained), 5)
        self.assertEqual(set(controller.population.population), retained)

    def test_monotonic_genome_ids_do_not_depend_on_retained_population(self) -> None:
        """Exercise test monotonic genome ids do not depend on retained population behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test monotonic genome ids do not depend on retained population test intent explicit.
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
        """Exercise test restore reconstructs allocators above every loaded gene behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test restore reconstructs allocators above every loaded gene test intent explicit.
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
