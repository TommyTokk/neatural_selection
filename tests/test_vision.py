from __future__ import annotations

import unittest
from dataclasses import dataclass
from math import pi

from configs.sim_config import VisionConfig
from src.creature import VisionTraits
from src.vision import SENSOR_INPUT_COUNT, VisionSystem


@dataclass(slots=True)
class FakeCreature:
    position: tuple[float, float]
    radius: float
    heading: float
    energy: float
    vision: VisionTraits
    speed: float = 0.0
    creature_id: int = 1


def creature_at(
    position: tuple[float, float],
    *,
    radius: float = 0.0,
    heading: float = 0.0,
    energy: float = 0.75,
    vision_range: float = 100.0,
    vision_angle: float = pi / 2,
) -> FakeCreature:
    return FakeCreature(
        position=position,
        radius=radius,
        heading=heading,
        energy=energy,
        vision=VisionTraits(range=vision_range, angle=vision_angle),
    )


class VisionWallSensorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vision = VisionSystem(VisionConfig())

    def sense_inputs(
        self,
        creature: FakeCreature,
        bounds: tuple[float, float, float, float],
    ) -> list[float]:
        return self.vision.sense(
            creature,
            foods=[],
            creatures=[],
            world_bounds=bounds,
            max_speed=100.0,
        ).as_inputs()

    def test_wall_directly_ahead_within_range_is_visible(self) -> None:
        inputs = self.sense_inputs(
            creature_at((55.0, 50.0), radius=5.0, vision_range=50.0),
            (0.0, 0.0, 100.0, 100.0),
        )

        self.assertGreater(inputs[4], 0.0)
        self.assertAlmostEqual(inputs[4], 0.2)
        self.assertAlmostEqual(inputs[5], 0.0)

    def test_wall_outside_vision_cone_is_not_visible(self) -> None:
        inputs = self.sense_inputs(
            creature_at(
                (90.0, 50.0),
                heading=pi / 2,
                vision_range=30.0,
                vision_angle=pi / 6,
            ),
            (0.0, 0.0, 100.0, 200.0),
        )

        self.assertEqual(inputs[4], 0.0)
        self.assertEqual(inputs[5], 0.0)

    def test_wall_at_left_edge_reports_negative_angle(self) -> None:
        inputs = self.sense_inputs(
            creature_at((50.0, 50.0), vision_range=100.0),
            (0.0, 0.0, 150.0, 300.0),
        )

        self.assertGreater(inputs[4], 0.0)
        self.assertAlmostEqual(inputs[5], -1.0)

    def test_wall_at_right_edge_reports_positive_angle(self) -> None:
        inputs = self.sense_inputs(
            creature_at((50.0, 50.0), vision_range=100.0),
            (0.0, -200.0, 150.0, 100.0),
        )

        self.assertGreater(inputs[4], 0.0)
        self.assertAlmostEqual(inputs[5], 1.0)

    def test_wall_farther_than_vision_range_is_not_visible(self) -> None:
        inputs = self.sense_inputs(
            creature_at((50.0, 50.0), vision_range=40.0),
            (0.0, 0.0, 200.0, 200.0),
        )

        self.assertEqual(inputs[4], 0.0)
        self.assertEqual(inputs[5], 0.0)

    def test_sensor_input_contract_includes_wall_inputs(self) -> None:
        inputs = self.sense_inputs(
            creature_at((55.0, 50.0), radius=5.0, vision_range=50.0),
            (0.0, 0.0, 100.0, 100.0),
        )

        self.assertEqual(SENSOR_INPUT_COUNT, 7)
        self.assertEqual(len(inputs), SENSOR_INPUT_COUNT)
        self.assertAlmostEqual(inputs[6], 0.75)


if __name__ == "__main__":
    unittest.main()
