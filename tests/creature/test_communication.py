from __future__ import annotations

from dataclasses import fields
from math import cos, exp, pi, sin, sqrt
import unittest

import numpy as np
from numpy.testing import assert_allclose

from configs.sim_config import (
    CommunicationConfig,
    PheromoneBoundaryMode,
    PheromoneConfig,
)
from src.communication import (
    AcousticObservation,
    AcousticSignal,
    AcousticSystem,
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
        self.assertEqual(config.pheromone.diffusion_coefficient, 390.0)
        self.assertEqual(config.pheromone.decay_rate, 0.08)
        self.assertIs(
            config.pheromone.boundary_mode,
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
    def make_system(self, **overrides: object) -> PheromoneSystem:
        """Create a compact pheromone system for focused tests.

        Parameters
        ----------
        overrides
            Pheromone configuration values replacing test defaults.

        Returns
        -------
        PheromoneSystem
            Empty width-major test field.
        """
        # Keep geometry asymmetric so swapped X/Y axes are observable.
        values = {
            "diffusion_coefficient": 0.0,
            "decay_rate": 0.0,
            "max_concentration": 1.0,
            "boundary_mode": PheromoneBoundaryMode.REFLECT,
        }
        values.update(overrides)
        return PheromoneSystem(
            PheromoneConfig(**values),
            5,
            3,
            (0.0, 0.0, 4.0, 2.0),
        )

    def test_field_is_strictly_width_major_rgb(self) -> None:
        """Verify the public tensor and scalar sampling axis contract.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Assertions verify width-major RGB storage.
        """
        # Use unequal width and height to expose transposition.
        system = self.make_system()
        self.assertEqual(system.field.shape, (5, 3, 3))
        system.field[3, 1] = (0.2, 0.4, 0.6)
        assert_allclose(system.sample(3.0, 1.0), (0.2, 0.4, 0.6))

    def test_bilinear_splat_conserves_channels_at_edges(self) -> None:
        """Verify edge splats conserve every RGB channel.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Assertions verify safe edge accumulation.
        """
        # Deposit on the maximum corner where upper indices become duplicates.
        system = self.make_system()
        system.deposit(4.0, 2.0, (0.2, 0.4, 0.6))
        assert_allclose(system.field.sum(axis=(0, 1)), (0.2, 0.4, 0.6))
        assert_allclose(system.field[4, 2], (0.2, 0.4, 0.6))

    def test_batch_and_scalar_deposition_match(self) -> None:
        """Verify scalar and batched bilinear deposits are equivalent.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Assertions compare complete RGB tensors.
        """
        # Exercise duplicate-index accumulation through both APIs.
        positions = np.asarray(((0.25, 0.75), (3.8, 1.9)))
        colors = np.asarray(((0.2, 0.3, 0.1), (0.1, 0.2, 0.3)))
        scalar = self.make_system()
        batch = self.make_system()
        for position, color in zip(positions, colors):
            scalar.deposit(float(position[0]), float(position[1]), color)
        batch.deposit_many(positions, colors)
        assert_allclose(batch.field, scalar.field)

    def test_sense_many_returns_probe_and_rgb_axes(self) -> None:
        """Verify batched sensing returns creature, probe, and RGB axes.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Assertions verify output shape and values.
        """
        # Seed a channel-distinct cell to make the final axis unambiguous.
        system = self.make_system()
        system.field[1, 1] = (0.25, 0.5, 0.75)
        probes = np.asarray(((((1.0, 1.0), (0.0, 0.0), (4.0, 2.0))),))
        sensed = system.sense_many(probes)
        self.assertEqual(sensed.shape, (1, 3, 3))
        assert_allclose(sensed[0, 0], (0.25, 0.5, 0.75))

    def test_diffusion_uses_x_and_y_axes_without_channel_leakage(self) -> None:
        """Verify diffusion respects spatial axes and channel independence.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Assertions verify neighbor propagation and empty channels.
        """
        # A Red-only impulse must remain Red while reaching X and Y neighbors.
        system = self.make_system(diffusion_coefficient=0.1)
        system.field[2, 1, 0] = 1.0
        system.advance(0.1)
        self.assertGreater(system.field[1, 1, 0], 0.0)
        self.assertGreater(system.field[3, 1, 0], 0.0)
        self.assertGreater(system.field[2, 0, 0], 0.0)
        self.assertGreater(system.field[2, 2, 0], 0.0)
        self.assertEqual(float(system.field[..., 1:].sum()), 0.0)

    def test_absorb_and_wrap_boundaries_are_safe(self) -> None:
        """Verify absorb and wrap deposition policies avoid invalid indices.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Assertions verify rejection and periodic mapping.
        """
        # Probe both an outside absorb point and an outside periodic point.
        absorb = self.make_system(boundary_mode="absorb")
        absorb.deposit(-1.0, 1.0, (1.0, 1.0, 1.0))
        self.assertEqual(float(absorb.field.sum()), 0.0)
        wrap = self.make_system(boundary_mode="wrap")
        wrap.deposit(4.1, 1.0, (0.2, 0.3, 0.4))
        assert_allclose(wrap.field.sum(axis=(0, 1)), (0.2, 0.3, 0.4))

    def test_restore_validates_shape_and_axis_metadata(self) -> None:
        """Verify persistence restoration enforces the RGB tensor contract.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Assertions verify valid restoration and invalid rejection.
        """
        # Restore one valid field before exercising incompatible metadata.
        system = self.make_system()
        values = np.full(system.field.shape, 0.25, dtype=np.float32)
        system.restore(values, system.state_metadata())
        assert_allclose(system.field, values)
        with self.assertRaises(ValueError):
            system.restore(np.zeros((3, 5, 3), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
