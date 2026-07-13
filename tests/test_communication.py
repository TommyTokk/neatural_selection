from __future__ import annotations

from math import exp
import unittest

import numpy as np

from configs.sim_config import CommunicationConfig
from src.communication import AcousticSignal, AcousticSystem, PheromoneSystem


class AcousticSystemTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CommunicationConfig(
            acoustic_range=100.0,
            acoustic_min_emission=0.01,
        )
        self.system = AcousticSystem(self.config)

    def test_relative_direction_is_continuous_behind_receiver(self) -> None:
        self.system.replace_signals([AcousticSignal(2, (-1.0, 0.001), 1.0, 0.2)])
        above = self.system.sense(1, (0.0, 0.0), 0.0)
        self.system.replace_signals([AcousticSignal(2, (-1.0, -0.001), 1.0, 0.2)])
        below = self.system.sense(1, (0.0, 0.0), 0.0)

        self.assertAlmostEqual(above.direction_cos, -1.0, places=5)
        self.assertAlmostEqual(below.direction_cos, -1.0, places=5)
        self.assertLess(abs(above.direction_sin - below.direction_sin), 0.003)

    def test_cardinal_directions_are_body_relative(self) -> None:
        expected = {
            (10.0, 0.0): (0.0, 1.0),
            (0.0, 10.0): (1.0, 0.0),
            (-10.0, 0.0): (0.0, -1.0),
            (0.0, -10.0): (-1.0, 0.0),
        }
        for position, (expected_sin, expected_cos) in expected.items():
            with self.subTest(position=position):
                self.system.replace_signals([AcousticSignal(2, position, 1.0, 0.0)])
                snapshot = self.system.sense(1, (0.0, 0.0), 0.0)
                self.assertAlmostEqual(snapshot.direction_sin, expected_sin)
                self.assertAlmostEqual(snapshot.direction_cos, expected_cos)

    def test_ignores_self_range_and_chooses_strongest_with_stable_tie(self) -> None:
        self.system.replace_signals(
            [
                AcousticSignal(1, (0.0, 0.0), 1.0, -1.0),
                AcousticSignal(3, (10.0, 0.0), 1.0, 0.3),
                AcousticSignal(2, (10.0, 0.0), 1.0, 0.2),
                AcousticSignal(4, (101.0, 0.0), 1.0, 0.4),
            ]
        )

        snapshot = self.system.sense(1, (0.0, 0.0), 0.0)

        self.assertEqual(snapshot.source_id, 2)
        self.assertAlmostEqual(snapshot.strength, 0.81)
        self.assertAlmostEqual(snapshot.tone, 0.2)

    def test_absent_and_subthreshold_signals_return_zeros(self) -> None:
        self.system.replace_signals([AcousticSignal(2, (0.0, 0.0), 0.001, 1.0)])
        snapshot = self.system.sense(1, (0.0, 0.0), 0.0)
        self.assertEqual(
            (
                snapshot.strength,
                snapshot.direction_sin,
                snapshot.direction_cos,
                snapshot.tone,
            ),
            (0.0, 0.0, 0.0, 0.0),
        )


class PheromoneSystemTest(unittest.TestCase):
    def make_system(self, **overrides: float) -> PheromoneSystem:
        values = {
            "pheromone_update_interval": 0.25,
            "pheromone_diffusion_coefficient": 0.15,
            "pheromone_evaporation_rate": 0.08,
            "pheromone_max_concentration": 10.0,
        }
        values.update(overrides)
        return PheromoneSystem(
            CommunicationConfig(**values),
            8,
            6,
            (0.0, 0.0, 80.0, 60.0),
        )

    def test_bilinear_deposit_conserves_amount_and_channels_are_independent(self) -> None:
        system = self.make_system()
        system.deposit((35.0, 27.0), trail_amount=0.6, alarm_amount=0.25)

        self.assertAlmostEqual(float(system.trail.sum()), 0.6, places=6)
        self.assertAlmostEqual(float(system.alarm.sum()), 0.25, places=6)
        self.assertFalse(np.array_equal(system.trail, system.alarm))

    def test_sampling_clamps_world_positions_to_grid_bounds(self) -> None:
        system = self.make_system()
        system.trail[0, 0] = 0.75

        self.assertAlmostEqual(system.sample((-100.0, -100.0), "trail"), 0.75)
        self.assertEqual(system.sample((-100.0, -100.0), "alarm"), 0.0)

    def test_diffusion_is_symmetric_and_no_flux_preserves_mass(self) -> None:
        system = self.make_system(pheromone_evaporation_rate=0.0)
        system.trail[3, 4] = 1.0

        system.advance(0.25)

        self.assertAlmostEqual(float(system.trail.sum()), 1.0, places=6)
        self.assertAlmostEqual(system.trail[2, 4], system.trail[4, 4])
        self.assertAlmostEqual(system.trail[3, 3], system.trail[3, 5])

    def test_evaporation_uses_exponential_decay(self) -> None:
        system = self.make_system(pheromone_diffusion_coefficient=0.0)
        system.trail[2, 2] = 1.0

        system.advance(0.25)

        self.assertAlmostEqual(system.trail[2, 2], exp(-0.08 * 0.25), places=6)

    def test_fixed_accumulator_runs_four_updates_and_keeps_remainder(self) -> None:
        system = self.make_system()

        for _ in range(60):
            system.accumulate(1.0 / 60.0)
        system.accumulate(0.1)

        self.assertEqual(system.update_count, 4)
        self.assertAlmostEqual(system.accumulator, 0.1)

    def test_rejects_unstable_diffusion_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unstable pheromone diffusion"):
            self.make_system(
                pheromone_update_interval=0.5,
                pheromone_diffusion_coefficient=0.6,
            )


if __name__ == "__main__":
    unittest.main()
