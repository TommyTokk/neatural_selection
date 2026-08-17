from __future__ import annotations

from dataclasses import fields
from math import cos, exp, pi, sin, sqrt
import unittest

import numpy as np
from numpy.testing import assert_allclose

from configs.sim_config import (
    CommunicationConfig,
    PheromoneBoundaryMode,
)
from src.communication import (
    AcousticObservation,
    AcousticSignal,
    AcousticSystem,
    PheromoneChannel,
    PheromoneSystem,
)


class AcousticSystemTest(unittest.TestCase):
    def make_system(self, **overrides: object) -> AcousticSystem:
        """Exercise make system behavior.
        
        Parameters
        ----------
        overrides
            Value supplied to ``overrides`` by the test scenario.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the make system test intent explicit.
        values: dict[str, object] = {
            "acoustic_range": 100.0,
            "acoustic_min_emission_strength": 0.01,
            "acoustic_hearing_threshold": 0.01,
        }
        values.update(overrides)
        return AcousticSystem(CommunicationConfig(**values))

    def test_observation_contains_no_privileged_metadata(self) -> None:
        """Exercise test observation contains no privileged metadata behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test observation contains no privileged metadata test intent explicit.
        self.assertEqual(
            [field.name for field in fields(AcousticObservation)],
            ["strength", "direction_sin", "direction_cos", "tone"],
        )

    def test_communication_defaults_are_explicit(self) -> None:
        """Exercise test communication defaults are explicit behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test communication defaults are explicit test intent explicit.
        config = CommunicationConfig()
        self.assertEqual(config.acoustic_min_emission_strength, 0.05)
        self.assertEqual(config.acoustic_hearing_threshold, 0.05)
        self.assertEqual(config.pheromone_diffusion_coefficient, 390.0)
        self.assertEqual(config.pheromone_max_updates_per_tick, 4)
        self.assertIs(
            config.pheromone_boundary_mode,
            PheromoneBoundaryMode.REFLECT,
        )

    def test_empty_and_zero_range_systems_return_zeros(self) -> None:
        """Exercise test empty and zero range systems return zeros behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test empty and zero range systems return zeros test intent explicit.
        empty = self.make_system().sense(1, (0.0, 0.0), 0.0)
        zero_range = self.make_system(acoustic_range=0.0)
        zero_range.replace_signals([AcousticSignal(2, (0.0, 0.0), 1.0, 0.0)])
        self.assertEqual(empty, AcousticObservation())
        self.assertEqual(
            zero_range.sense(1, (0.0, 0.0), 0.0),
            AcousticObservation(),
        )

    def test_emission_and_hearing_thresholds_are_independent(self) -> None:
        """Exercise test emission and hearing thresholds are independent behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test emission and hearing thresholds are independent test intent explicit.
        system = self.make_system(
            acoustic_min_emission_strength=0.05,
            acoustic_hearing_threshold=0.01,
        )
        system.replace_signals([AcousticSignal(2, (10.0, 0.0), 0.05, 0.2)])

        observation = system.sense(1, (0.0, 0.0), 0.0)

        self.assertAlmostEqual(observation.strength, 0.05 * 0.9**2)
        self.assertGreater(observation.strength, 0.01)

    def test_attenuation_range_and_self_exclusion(self) -> None:
        """Exercise test attenuation range and self exclusion behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test attenuation range and self exclusion test intent explicit.
        system = self.make_system(acoustic_hearing_threshold=0.0)
        system.replace_signals(
            [
                AcousticSignal(1, (0.0, 0.0), 1.0, -1.0),
                AcousticSignal(2, (0.0, 0.0), 1.0, 0.0),
            ]
        )
        self.assertAlmostEqual(system.sense(1, (0.0, 0.0), 0.0).strength, 1.0)
        self.assertAlmostEqual(system.sense(1, (-50.0, 0.0), 0.0).strength, 0.25)
        self.assertAlmostEqual(system.sense(1, (-100.0, 0.0), 0.0).strength, 0.0)
        self.assertEqual(system.sense(1, (-100.0001, 0.0), 0.0), AcousticObservation())

    def test_cardinal_directions_and_nonzero_heading(self) -> None:
        """Exercise test cardinal directions and nonzero heading behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test cardinal directions and nonzero heading test intent explicit.
        system = self.make_system()
        expected = {
            (10.0, 0.0): (0.0, 1.0),
            (0.0, 10.0): (1.0, 0.0),
            (-10.0, 0.0): (0.0, -1.0),
            (0.0, -10.0): (-1.0, 0.0),
        }
        for position, (expected_sin, expected_cos) in expected.items():
            with self.subTest(position=position):
                system.replace_signals([AcousticSignal(2, position, 1.0, 0.0)])
                observation = system.sense(1, (0.0, 0.0), 0.0)
                self.assertAlmostEqual(observation.direction_sin, expected_sin)
                self.assertAlmostEqual(observation.direction_cos, expected_cos)

        system.replace_signals([AcousticSignal(2, (10.0, 0.0), 1.0, 0.0)])
        turned = system.sense(1, (0.0, 0.0), pi / 2.0)
        self.assertAlmostEqual(turned.direction_sin, -1.0)
        self.assertAlmostEqual(turned.direction_cos, 0.0, places=7)

    def test_coincident_direction_is_neutral_and_debug_is_separate(self) -> None:
        """Exercise test coincident direction is neutral and debug is separate behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test coincident direction is neutral and debug is separate test intent explicit.
        system = self.make_system()
        signal = AcousticSignal(2, (-10.0, -10.0), 0.8, -0.25)
        system.replace_signals([signal])

        result = system.sense_with_debug(1, signal.position, 1.2)

        self.assertEqual(result.observation.direction_sin, 0.0)
        self.assertEqual(result.observation.direction_cos, 0.0)
        self.assertEqual(result.debug.source_id, 2)
        self.assertEqual(result.debug.source_position, signal.position)

    def test_equal_strength_tie_prefers_lower_emitter_id(self) -> None:
        """Exercise test equal strength tie prefers lower emitter id behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test equal strength tie prefers lower emitter id test intent explicit.
        system = self.make_system()
        system.replace_signals(
            [
                AcousticSignal(3, (10.0, 0.0), 1.0, 0.3),
                AcousticSignal(2, (10.0, 0.0), 1.0, 0.2),
            ]
        )
        result = system.sense_with_debug(1, (0.0, 0.0), 0.0)
        self.assertEqual(result.debug.source_id, 2)
        self.assertAlmostEqual(result.observation.tone, 0.2)

    def test_signal_validation_rejects_corrupt_values(self) -> None:
        """Exercise test signal validation rejects corrupt values behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test signal validation rejects corrupt values test intent explicit.
        system = self.make_system()
        invalid = (
            AcousticSignal(True, (0.0, 0.0), 1.0, 0.0),
            AcousticSignal(2, (float("nan"), 0.0), 1.0, 0.0),
            AcousticSignal(2, (0.0, 0.0), float("inf"), 0.0),
            AcousticSignal(2, (0.0, 0.0), -0.1, 0.0),
            AcousticSignal(2, (0.0, 0.0), 1.0, 1.1),
        )
        for signal in invalid:
            with self.subTest(signal=signal), self.assertRaises(ValueError):
                system.replace_signals([signal])

    def test_spatial_index_matches_brute_force_reference(self) -> None:
        """Exercise test spatial index matches brute force reference behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test spatial index matches brute force reference test intent explicit.
        rng = np.random.default_rng(20260714)
        system = self.make_system(acoustic_range=80.0)
        signals = [
            AcousticSignal(
                emitter_id=index,
                position=(float(position[0]), float(position[1])),
                strength=float(strength),
                tone=float(tone),
            )
            for index, position, strength, tone in zip(
                range(200),
                rng.uniform(-500.0, 500.0, size=(200, 2)),
                rng.uniform(0.01, 1.0, size=200),
                rng.uniform(-1.0, 1.0, size=200),
            )
        ]
        system.replace_signals(signals)

        for receiver_id in range(1000, 1030):
            position_array = rng.uniform(-500.0, 500.0, size=2)
            position = (float(position_array[0]), float(position_array[1]))
            heading = float(rng.uniform(-pi, pi))
            expected = self._brute_force(signals, receiver_id, position, heading)
            actual = system.sense_with_debug(receiver_id, position, heading)
            self.assertEqual(actual.debug.source_id, expected[1])
            assert_allclose(
                (
                    actual.observation.strength,
                    actual.observation.direction_sin,
                    actual.observation.direction_cos,
                    actual.observation.tone,
                ),
                expected[0],
                rtol=1e-12,
                atol=1e-12,
            )

    def test_negative_coordinates_cells_and_duplicate_emitters(self) -> None:
        """Exercise test negative coordinates cells and duplicate emitters behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test negative coordinates cells and duplicate emitters test intent explicit.
        system = self.make_system(acoustic_range=10.0)
        system.replace_signals(
            [
                AcousticSignal(2, (-20.0, -20.0), 1.0, -0.5),
                AcousticSignal(2, (-9.9, -10.0), 1.0, 0.5),
                AcousticSignal(3, (1000.0, 1000.0), 1.0, 0.0),
            ]
        )
        result = system.sense_with_debug(1, (-10.0, -10.0), 0.0)
        self.assertEqual(result.debug.source_id, 2)
        self.assertEqual(result.observation.tone, 0.5)
        self.assertLess(system.last_candidate_checks, len(system.signals))

    @staticmethod
    def _brute_force(
        signals: list[AcousticSignal],
        receiver_id: int,
        position: tuple[float, float],
        heading: float,
    ) -> tuple[tuple[float, float, float, float], int | None]:
        """Exercise brute force behavior.
        
        Parameters
        ----------
        signals
            Value supplied to ``signals`` by the test scenario.
        receiver_id
            Value supplied to ``receiver_id`` by the test scenario.
        position
            Value supplied to ``position`` by the test scenario.
        heading
            Value supplied to ``heading`` by the test scenario.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the brute force test intent explicit.
        best: tuple[float, int, AcousticSignal, float, float, float] | None = None
        for signal in signals:
            if signal.emitter_id == receiver_id or signal.strength < 0.01:
                continue
            dx = signal.position[0] - position[0]
            dy = signal.position[1] - position[1]
            distance_squared = dx * dx + dy * dy
            if distance_squared > 80.0**2:
                continue
            distance = sqrt(distance_squared)
            heard = signal.strength * (1.0 - distance / 80.0) ** 2
            if heard < 0.01:
                continue
            candidate = (heard, -signal.emitter_id, signal, dx, dy, distance)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None:
            return (0.0, 0.0, 0.0, 0.0), None
        heard, _tie, signal, dx, dy, distance = best
        if distance <= 1e-12:
            direction_sin = direction_cos = 0.0
        else:
            unit_x = dx / distance
            unit_y = dy / distance
            direction_cos = unit_x * cos(heading) + unit_y * sin(heading)
            direction_sin = unit_y * cos(heading) - unit_x * sin(heading)
        return (heard, direction_sin, direction_cos, signal.tone), signal.emitter_id


class PheromoneSystemTest(unittest.TestCase):
    def make_system(
        self,
        *,
        grid_width: int = 8,
        grid_height: int = 6,
        bounds: tuple[float, float, float, float] = (0.0, 0.0, 80.0, 60.0),
        **overrides: object,
    ) -> PheromoneSystem:
        """Exercise make system behavior.
        
        Parameters
        ----------
        grid_width
            Value supplied to ``grid_width`` by the test scenario.
        grid_height
            Value supplied to ``grid_height`` by the test scenario.
        bounds
            Value supplied to ``bounds`` by the test scenario.
        overrides
            Value supplied to ``overrides`` by the test scenario.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the make system test intent explicit.
        values: dict[str, object] = {
            "pheromone_update_interval": 0.25,
            "pheromone_diffusion_coefficient": 30.0,
            "pheromone_evaporation_rate": 0.08,
            "pheromone_max_concentration": 10.0,
        }
        values.update(overrides)
        return PheromoneSystem(
            CommunicationConfig(**values),
            grid_width,
            grid_height,
            bounds,
        )

    def test_scalar_deposit_and_sampling_are_bilinear(self) -> None:
        """Exercise test scalar deposit and sampling are bilinear behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test scalar deposit and sampling are bilinear test intent explicit.
        system = self.make_system(grid_width=3, grid_height=3, bounds=(0, 0, 2, 2))
        system.deposit((0.5, 0.5), trail_amount=1.0, alarm_amount=0.5)
        assert_allclose(system.trail[:2, :2], np.full((2, 2), 0.25))
        assert_allclose(system.alarm[:2, :2], np.full((2, 2), 0.125))
        self.assertAlmostEqual(system.sample_trail((0.5, 0.5)), 0.25)
        self.assertAlmostEqual(system.sample_alarm((0.5, 0.5)), 0.125)

    def test_edge_and_corner_deposits_preserve_total_weight(self) -> None:
        """Exercise test edge and corner deposits preserve total weight behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test edge and corner deposits preserve total weight test intent explicit.
        system = self.make_system()
        for position in ((0.0, 0.0), (80.0, 60.0), (80.0, 30.0)):
            system.trail.fill(0.0)
            system.deposit(position, trail_amount=0.75)
            self.assertAlmostEqual(float(system.trail.sum()), 0.75, places=6)

    def test_batch_deposit_matches_scalar_with_duplicate_indices(self) -> None:
        """Exercise test batch deposit matches scalar with duplicate indices behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test batch deposit matches scalar with duplicate indices test intent explicit.
        positions = np.array(
            [[0.0, 0.0], [35.0, 27.0], [35.0, 27.0], [80.0, 60.0]],
            dtype=np.float64,
        )
        trail = np.array([0.2, 0.3, 0.1, 0.4])
        alarm = np.array([0.1, 0.2, 0.4, 0.3])
        scalar = self.make_system()
        batch = self.make_system()
        for position, trail_amount, alarm_amount in zip(positions, trail, alarm):
            scalar.deposit(
                tuple(position),
                trail_amount=float(trail_amount),
                alarm_amount=float(alarm_amount),
            )
        batch.deposit_many(positions, trail, alarm)
        assert_allclose(batch.trail, scalar.trail, rtol=1e-6, atol=1e-7)
        assert_allclose(batch.alarm, scalar.alarm, rtol=1e-6, atol=1e-7)

    def test_scalar_and_batch_deposits_clip_to_float32_maximum(self) -> None:
        """Exercise test scalar and batch deposits clip to float32 maximum behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test scalar and batch deposits clip to float32 maximum test intent explicit.
        scalar = self.make_system(pheromone_max_concentration=0.5)
        batch = self.make_system(pheromone_max_concentration=0.5)
        scalar.deposit((20.0, 20.0), trail_amount=10.0, alarm_amount=10.0)
        batch.deposit_many(
            np.array([[20.0, 20.0], [20.0, 20.0]]),
            np.array([5.0, 5.0]),
            np.array([5.0, 5.0]),
        )
        self.assertEqual(scalar.trail.dtype, np.float32)
        self.assertEqual(batch.alarm.dtype, np.float32)
        self.assertLessEqual(float(scalar.trail.max()), 0.5)
        self.assertLessEqual(float(batch.alarm.max()), 0.5)

    def test_channel_validation_and_independence(self) -> None:
        """Exercise test channel validation and independence behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test channel validation and independence test intent explicit.
        system = self.make_system()
        system.deposit((10.0, 10.0), trail_amount=0.5)
        self.assertGreater(system.sample((10.0, 10.0), PheromoneChannel.TRAIL), 0.0)
        self.assertEqual(system.sample_alarm((10.0, 10.0)), 0.0)
        with self.assertRaisesRegex(ValueError, "channel"):
            system.sample((10.0, 10.0), "unknown")

    def test_vectorized_sense_matches_scalar_and_is_float32(self) -> None:
        """Exercise test vectorized sense matches scalar and is float32 behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test vectorized sense matches scalar and is float32 test intent explicit.
        system = self.make_system()
        system.deposit((20.0, 20.0), trail_amount=0.8, alarm_amount=0.4)
        positions = np.array(
            [
                [[20.0, 20.0], [21.0, 20.0], [20.0, 21.0]],
                [[0.0, 0.0], [80.0, 60.0], [40.0, 30.0]],
            ]
        )
        actual = system.sense_many(positions)
        expected_rows = []
        for triple in positions:
            snapshot = system.sense(tuple(map(tuple, triple)))
            expected_rows.append(
                (
                    snapshot.trail_here,
                    snapshot.trail_forward_left,
                    snapshot.trail_forward_right,
                    snapshot.alarm_here,
                    snapshot.alarm_forward_left,
                    snapshot.alarm_forward_right,
                )
            )
        expected = np.asarray(expected_rows, dtype=np.float32)
        self.assertEqual(actual.dtype, np.float32)
        assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_boundary_position_semantics(self) -> None:
        """Exercise test boundary position semantics behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test boundary position semantics test intent explicit.
        reflect = self.make_system(pheromone_boundary_mode="reflect")
        reflect.trail[0, 0] = 0.75
        self.assertEqual(reflect.sample_trail((-100.0, -100.0)), 0.75)

        wrap = self.make_system(pheromone_boundary_mode="wrap")
        wrap.trail[0, 0] = 0.5
        self.assertEqual(wrap.sample_trail((80.0, 60.0)), 0.5)

        absorb = self.make_system(pheromone_boundary_mode="absorb")
        absorb.deposit((-1.0, 0.0), trail_amount=1.0)
        self.assertEqual(float(absorb.trail.sum()), 0.0)
        self.assertEqual(absorb.sample_trail((-1.0, 0.0)), 0.0)

    def test_reflect_and_wrap_conserve_mass_while_absorb_loses_it(self) -> None:
        """Exercise test reflect and wrap conserve mass while absorb loses it behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test reflect and wrap conserve mass while absorb loses it test intent explicit.
        totals: dict[PheromoneBoundaryMode, float] = {}
        for mode in PheromoneBoundaryMode:
            system = self.make_system(
                pheromone_boundary_mode=mode,
                pheromone_evaporation_rate=0.0,
            )
            system.trail[0, 0] = 1.0
            system.advance(0.25)
            totals[mode] = float(system.trail.sum())
            self.assertGreaterEqual(float(system.trail.min()), -1e-7)
        self.assertAlmostEqual(totals[PheromoneBoundaryMode.REFLECT], 1.0, places=6)
        self.assertAlmostEqual(totals[PheromoneBoundaryMode.WRAP], 1.0, places=6)
        self.assertLess(totals[PheromoneBoundaryMode.ABSORB], 1.0)

    def test_large_public_timestep_uses_stable_substeps_without_creating_mass(self) -> None:
        """Exercise test large public timestep uses stable substeps without creating mass behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test large public timestep uses stable substeps without creating mass test intent explicit.
        system = self.make_system(
            grid_width=16,
            grid_height=16,
            bounds=(0.0, 0.0, 15.0, 15.0),
            pheromone_diffusion_coefficient=1.0,
            pheromone_evaporation_rate=0.0,
        )
        system.trail[8, 8] = 1.0
        system.advance(2.0)
        self.assertGreater(system.diffusion_substep_count, 1)
        self.assertEqual(system.update_count, 1)
        self.assertGreaterEqual(float(system.trail.min()), -1e-7)
        self.assertAlmostEqual(float(system.trail.sum()), 1.0, places=5)

    def test_zero_diffusion_and_exponential_evaporation(self) -> None:
        """Exercise test zero diffusion and exponential evaporation behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test zero diffusion and exponential evaporation test intent explicit.
        system = self.make_system(pheromone_diffusion_coefficient=0.0)
        system.trail[2, 2] = 1.0
        previous = 1.0
        for _ in range(4):
            system.advance(0.25)
            current = float(system.trail.sum())
            self.assertLess(current, previous)
            previous = current
        self.assertAlmostEqual(previous, exp(-0.08), places=6)

    def test_accumulator_equivalence_and_catch_up_cap(self) -> None:
        """Exercise test accumulator equivalence and catch up cap behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test accumulator equivalence and catch up cap test intent explicit.
        accumulated = self.make_system()
        manual = self.make_system()
        accumulated.trail[2, 2] = manual.trail[2, 2] = 1.0
        for _ in range(60):
            accumulated.accumulate(1.0 / 60.0)
        for _ in range(4):
            manual.advance(0.25)
        assert_allclose(accumulated.trail, manual.trail, rtol=1e-6, atol=1e-7)

        processed = accumulated.accumulate(2.1)
        self.assertEqual(processed, 4)
        self.assertEqual(accumulated.last_dropped_updates, 4)
        self.assertAlmostEqual(accumulated.last_dropped_time, 1.0)
        self.assertAlmostEqual(accumulated.accumulator, 0.1)

    def test_world_space_diffusion_is_resolution_consistent(self) -> None:
        """Exercise test world space diffusion is resolution consistent behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test world space diffusion is resolution consistent test intent explicit.
        moments = []
        for size in (21, 41):
            system = self.make_system(
                grid_width=size,
                grid_height=size,
                bounds=(0.0, 0.0, 100.0, 100.0),
                pheromone_diffusion_coefficient=10.0,
                pheromone_evaporation_rate=0.0,
            )
            system.deposit((50.0, 50.0), trail_amount=1.0)
            system.advance(1.0)
            coordinates = np.linspace(0.0, 100.0, size)
            x, y = np.meshgrid(coordinates, coordinates)
            radius_squared = (x - 50.0) ** 2 + (y - 50.0) ** 2
            moments.append(float((system.trail * radius_squared).sum() / system.trail.sum()))
        assert_allclose(moments[0], moments[1], rtol=0.05, atol=0.05)

    def test_repeated_batch_and_diffusion_sequence_is_deterministic(self) -> None:
        """Exercise test repeated batch and diffusion sequence is deterministic behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test repeated batch and diffusion sequence is deterministic test intent explicit.
        positions = np.array([[12.5, 8.0], [35.0, 27.0], [12.5, 8.0]])
        trail = np.array([0.2, 0.3, 0.4])
        alarm = np.array([0.4, 0.1, 0.2])
        first = self.make_system(pheromone_evaporation_rate=0.0)
        second = self.make_system(pheromone_evaporation_rate=0.0)
        for system in (first, second):
            system.deposit_many(positions, trail, alarm)
            system.advance(0.75)
        self.assertTrue(np.array_equal(first.trail, second.trail))
        self.assertTrue(np.array_equal(first.alarm, second.alarm))

    def test_restore_round_trip_and_corruption_rejection(self) -> None:
        """Exercise test restore round trip and corruption rejection behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test restore round trip and corruption rejection test intent explicit.
        system = self.make_system()
        system.deposit((20.0, 20.0), trail_amount=0.5, alarm_amount=0.25)
        target = self.make_system()
        target.restore(system.trail, system.alarm, 0.1, system.state_metadata())
        assert_allclose(target.trail, system.trail)
        assert_allclose(target.alarm, system.alarm)
        self.assertEqual(target.accumulator, 0.1)

        invalid_grids = (
            np.zeros((1, 1), dtype=np.float32),
            np.full(system.trail.shape, np.nan, dtype=np.float32),
            np.full(system.trail.shape, np.inf, dtype=np.float32),
            np.full(system.trail.shape, -np.inf, dtype=np.float32),
            np.full(system.trail.shape, -1.0, dtype=np.float32),
            np.full(system.trail.shape, 11.0, dtype=np.float32),
        )
        for values in invalid_grids:
            with self.subTest(values=values), self.assertRaises(ValueError):
                target.restore(values, system.alarm, 0.0)
        for accumulator in (
            -1.0,
            float("nan"),
            float("inf"),
            float("-inf"),
            0.25,
        ):
            with self.subTest(accumulator=accumulator), self.assertRaises(ValueError):
                target.restore(system.trail, system.alarm, accumulator)
        bad_metadata = system.state_metadata()
        bad_metadata["boundary_mode"] = "wrap"
        with self.assertRaisesRegex(ValueError, "boundary mode"):
            target.restore(system.trail, system.alarm, 0.0, bad_metadata)

    def test_configuration_and_public_timestep_validation(self) -> None:
        """Exercise test configuration and public timestep validation behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test configuration and public timestep validation test intent explicit.
        invalid_configs = (
            {"acoustic_range": float("nan")},
            {"acoustic_min_emission_strength": -0.1},
            {"acoustic_hearing_threshold": 1.1},
            {"pheromone_update_interval": 0.0},
            {"pheromone_diffusion_coefficient": -1.0},
            {"pheromone_max_updates_per_tick": 1.5},
            {"pheromone_boundary_mode": "unknown"},
        )
        for values in invalid_configs:
            with self.subTest(values=values), self.assertRaises(ValueError):
                CommunicationConfig(**values)
        system = self.make_system()
        for timestep in (-1.0, float("nan"), float("inf")):
            with self.subTest(timestep=timestep), self.assertRaises(ValueError):
                system.advance(timestep)
            with self.assertRaises(ValueError):
                system.accumulate(timestep)

    def test_grid_geometry_validation(self) -> None:
        """Exercise test grid geometry validation behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test grid geometry validation test intent explicit.
        config = CommunicationConfig()
        invalid = (
            (1, 2, (0.0, 0.0, 1.0, 1.0)),
            (2.5, 2, (0.0, 0.0, 1.0, 1.0)),
            (2, 2, (0.0, 0.0, 0.0, 1.0)),
            (2, 2, (0.0, 0.0, 1.0, float("inf"))),
        )
        for width, height, bounds in invalid:
            with self.subTest(width=width, height=height, bounds=bounds):
                with self.assertRaises(ValueError):
                    PheromoneSystem(config, width, height, bounds)


if __name__ == "__main__":
    unittest.main()
