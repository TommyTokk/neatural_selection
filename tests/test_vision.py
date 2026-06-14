from __future__ import annotations

import unittest
from dataclasses import dataclass
from math import cos, pi, sin
import sys
import types


class _Body:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.position = types.SimpleNamespace(x=0.0, y=0.0)


class _Circle:
    def __init__(self, body: _Body, radius: float) -> None:
        self.body = body
        self.radius = radius


sys.modules.setdefault(
    "pymunk",
    types.SimpleNamespace(
        Body=_Body,
        Circle=_Circle,
        ShapeFilter=lambda **kwargs: types.SimpleNamespace(**kwargs),
        moment_for_circle=lambda *args: 1.0,
    ),
)

from configs.sim_config import ActionConfig, MetabolismConfig, VisionConfig
from src.controller import BaselineFoodController
from src.creature import VisionTraits
from src.metabolism import Metabolism
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


@dataclass(slots=True)
class FakeFood:
    id: int
    position: tuple[float, float]
    radius: float


def creature_at(
    position: tuple[float, float],
    *,
    radius: float = 0.0,
    heading: float = 0.0,
    energy: float = 0.75,
    vision_range: float = 100.0,
    vision_angle: float = pi / 2,
    creature_id: int = 1,
) -> FakeCreature:
    return FakeCreature(
        position=position,
        radius=radius,
        heading=heading,
        energy=energy,
        vision=VisionTraits(range=vision_range, angle=vision_angle),
        creature_id=creature_id,
    )


class VisionOcclusionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vision = VisionSystem(VisionConfig())

    def sense_snapshot(
        self,
        creature: FakeCreature,
        *,
        foods: list[FakeFood] | None = None,
        creatures: list[FakeCreature] | None = None,
    ):
        return self.vision.sense(
            creature,
            foods=[] if foods is None else foods,
            creatures=[] if creatures is None else creatures,
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
        )

    def test_creature_directly_behind_creature_is_occluded(self) -> None:
        observer = creature_at((0.0, 0.0), radius=5.0)
        front = creature_at((30.0, 0.0), radius=10.0, creature_id=2)
        behind = creature_at((60.0, 0.0), radius=10.0, creature_id=3)

        snapshot = self.sense_snapshot(
            observer,
            creatures=[observer, front, behind],
        )

        self.assertEqual(snapshot.creatures.count, 1)
        self.assertGreater(snapshot.creatures.proximity_center, 0.0)
        self.assertGreater(snapshot.creatures.proximity_left, 0.0)
        self.assertGreater(snapshot.creatures.proximity_right, 0.0)
        self.assertEqual(
            self.vision.visible_creatures(observer, [observer, front, behind]),
            [front],
        )

    def test_partly_exposed_creature_remains_visible(self) -> None:
        observer = creature_at((0.0, 0.0), radius=5.0)
        front = creature_at((30.0, 0.0), radius=10.0, creature_id=2)
        offset = creature_at((60.0, 25.0), radius=10.0, creature_id=3)

        snapshot = self.sense_snapshot(
            observer,
            creatures=[observer, front, offset],
        )

        self.assertEqual(snapshot.creatures.count, 2)
        self.assertEqual(
            self.vision.visible_creatures(observer, [observer, front, offset]),
            [front, offset],
        )

    def test_food_behind_creature_is_occluded(self) -> None:
        observer = creature_at((0.0, 0.0), radius=5.0)
        blocker = creature_at((30.0, 0.0), radius=10.0, creature_id=2)
        hidden_food = FakeFood(id=1, position=(60.0, 0.0), radius=5.0)

        snapshot = self.sense_snapshot(
            observer,
            foods=[hidden_food],
            creatures=[observer, blocker],
        )

        self.assertEqual(snapshot.food.count, 0)
        self.assertEqual(
            self.vision.visible_foods(observer, [hidden_food], [observer, blocker]),
            [],
        )

    def test_creature_behind_food_is_occluded(self) -> None:
        observer = creature_at((0.0, 0.0), radius=5.0)
        blocker_food = FakeFood(id=1, position=(30.0, 0.0), radius=10.0)
        hidden_creature = creature_at((60.0, 0.0), radius=5.0, creature_id=2)

        snapshot = self.sense_snapshot(
            observer,
            foods=[blocker_food],
            creatures=[observer, hidden_creature],
        )

        self.assertEqual(snapshot.creatures.count, 0)
        self.assertEqual(
            self.vision.visible_creatures(
                observer,
                [observer, hidden_creature],
                [blocker_food],
            ),
            [],
        )

    def test_food_behind_food_is_occluded(self) -> None:
        observer = creature_at((0.0, 0.0), radius=5.0)
        blocker_food = FakeFood(id=1, position=(30.0, 0.0), radius=10.0)
        hidden_food = FakeFood(id=2, position=(60.0, 0.0), radius=5.0)

        snapshot = self.sense_snapshot(observer, foods=[blocker_food, hidden_food])

        self.assertEqual(snapshot.food.count, 1)
        self.assertEqual(
            self.vision.visible_foods(observer, [blocker_food, hidden_food]),
            [blocker_food],
        )

    def test_ignored_food_does_not_occlude_food_behind_it(self) -> None:
        observer = creature_at((0.0, 0.0), radius=5.0)
        carried_food = FakeFood(id=1, position=(30.0, 0.0), radius=10.0)
        visible_food = FakeFood(id=2, position=(60.0, 0.0), radius=5.0)

        snapshot = self.vision.sense(
            observer,
            foods=[carried_food, visible_food],
            creatures=[],
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
            ignored_food_ids={carried_food.id},
        )

        self.assertEqual(snapshot.food.count, 1)
        self.assertEqual(
            self.vision.visible_foods(
                observer,
                [carried_food, visible_food],
                ignored_food_ids={carried_food.id},
            ),
            [visible_food],
        )


class VisionEyeOriginBlindZoneTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metabolism_config = MetabolismConfig()
        self.vision = VisionSystem(
            VisionConfig(),
            self.metabolism_config.eating_distance,
        )
        self.metabolism = Metabolism(self.metabolism_config, self.vision, None)

    def sense_snapshot(
        self,
        creature: FakeCreature,
        foods: list[FakeFood],
    ):
        return self.vision.sense(
            creature,
            foods=foods,
            creatures=[],
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
        )

    def test_side_body_food_inside_blind_zone_is_not_visible(self) -> None:
        creature = creature_at((0.0, 0.0), radius=10.0, heading=0.0)
        side_food = FakeFood(id=1, position=(10.0, 8.0), radius=3.0)

        snapshot = self.sense_snapshot(creature, [side_food])

        self.assertEqual(snapshot.food.count, 0)
        self.assertEqual(self.vision.visible_foods(creature, [side_food]), [])

    def test_food_directly_in_mouth_path_remains_visible(self) -> None:
        creature = creature_at((0.0, 0.0), radius=10.0, heading=0.0)
        mouth_food = FakeFood(id=1, position=(13.0, 0.0), radius=3.0)

        snapshot = self.vision.sense(
            creature,
            foods=[mouth_food],
            creatures=[],
            world_bounds=(-1000.0, -1000.0, 1000.0, 1000.0),
            max_speed=100.0,
        )

        self.assertEqual(snapshot.food.count, 1)
        self.assertGreater(snapshot.food.proximity_center, 0.0)
        self.assertGreater(snapshot.food.proximity_left, 0.0)
        self.assertGreater(snapshot.food.proximity_right, 0.0)
        self.assertEqual(self.vision.visible_foods(creature, [mouth_food]), [mouth_food])

    def test_food_at_mouth_threshold_bypasses_blind_zone(self) -> None:
        creature = creature_at((0.0, 0.0), radius=10.0, heading=0.0)
        mouth_food = FakeFood(id=1, position=(10.0, 0.0), radius=3.0)

        snapshot = self.vision.sense(
            creature,
            foods=[mouth_food],
            creatures=[],
            world_bounds=(-1000.0, -1000.0, 1000.0, 1000.0),
            max_speed=100.0,
        )

        self.assertEqual(snapshot.food.count, 1)
        self.assertGreater(snapshot.food.proximity_left, 0.0)
        self.assertGreater(snapshot.food.proximity_center, 0.0)
        self.assertGreater(snapshot.food.proximity_right, 0.0)
        self.assertFalse(
            self.vision._food_in_mouth_blind_zone(
                creature,
                mouth_food.position,
                mouth_food.radius,
            )
        )

    def test_touch_exemption_matches_metabolism_mouth_overlap(self) -> None:
        creature = creature_at((0.0, 0.0), radius=10.0, heading=0.0)
        positions = [
            (10.0, 0.0),
            (10.0, 4.0),
            (10.0, 8.0),
            (7.9, 0.0),
            (13.0, 0.0),
        ]

        for index, position in enumerate(positions):
            with self.subTest(position=position):
                food = FakeFood(id=index, position=position, radius=3.0)

                self.assertEqual(
                    self.vision._food_touches_mouth(
                        creature,
                        food.position,
                        food.radius,
                    ),
                    self.metabolism.food_overlaps_mouth(creature, food),
                )

    def test_lateral_mouth_contact_activates_all_food_sectors(self) -> None:
        creature = creature_at(
            (0.0, 0.0),
            radius=16.0,
            heading=0.0,
            vision_angle=VisionConfig().default_angle,
        )
        mouth_food = FakeFood(id=1, position=(16.0, 6.4), radius=3.0)

        snapshot = self.sense_snapshot(creature, [mouth_food])

        self.assertEqual(snapshot.food.count, 1)
        self.assertEqual(self.vision.visible_foods(creature, [mouth_food]), [mouth_food])
        self.assertAlmostEqual(snapshot.food.proximity_left, 1.0)
        self.assertAlmostEqual(snapshot.food.proximity_center, 1.0)
        self.assertAlmostEqual(snapshot.food.proximity_right, 1.0)

    def test_narrow_fov_mouth_contact_activates_all_food_sectors(self) -> None:
        creature = creature_at(
            (0.0, 0.0),
            radius=22.0,
            heading=0.0,
            vision_angle=VisionConfig().min_angle,
        )
        mouth_food = FakeFood(id=1, position=(22.0, 8.8), radius=3.0)

        snapshot = self.sense_snapshot(creature, [mouth_food])

        self.assertEqual(snapshot.food.count, 1)
        self.assertEqual(self.vision.visible_foods(creature, [mouth_food]), [mouth_food])
        self.assertAlmostEqual(snapshot.food.proximity_left, 1.0)
        self.assertAlmostEqual(snapshot.food.proximity_center, 1.0)
        self.assertAlmostEqual(snapshot.food.proximity_right, 1.0)

    def test_baseline_controller_drives_straight_for_mouth_contact_food(self) -> None:
        creature = creature_at(
            (0.0, 0.0),
            radius=16.0,
            heading=0.0,
            vision_angle=VisionConfig().default_angle,
        )
        mouth_food = FakeFood(id=1, position=(16.0, 6.4), radius=3.0)

        snapshot = self.vision.sense(
            creature,
            foods=[mouth_food],
            creatures=[],
            world_bounds=(-1000.0, -1000.0, 1000.0, 1000.0),
            max_speed=100.0,
        )
        action = BaselineFoodController(ActionConfig()).decide(
            snapshot,
            creature.creature_id,
        )

        self.assertAlmostEqual(action.rotate, 0.0)
        self.assertAlmostEqual(action.accelerate, 1.0)

    def test_forward_food_near_mouth_remains_visible(self) -> None:
        creature = creature_at((0.0, 0.0), radius=10.0, heading=0.0)
        forward_food = FakeFood(id=1, position=(17.0, 0.0), radius=3.0)

        snapshot = self.sense_snapshot(creature, [forward_food])

        self.assertEqual(snapshot.food.count, 1)
        self.assertGreater(snapshot.food.proximity_center, 0.0)
        self.assertEqual(snapshot.food.proximity_left, 0.0)
        self.assertEqual(snapshot.food.proximity_right, 0.0)
        self.assertEqual(
            self.vision.visible_foods(creature, [forward_food]),
            [forward_food],
        )

    def test_food_farther_in_eye_cone_remains_visible(self) -> None:
        creature = creature_at((0.0, 0.0), radius=10.0, heading=0.0)
        front_food = FakeFood(id=1, position=(24.0, 0.0), radius=3.0)

        snapshot = self.sense_snapshot(creature, [front_food])

        self.assertEqual(snapshot.food.count, 1)
        self.assertGreater(snapshot.food.proximity_center, 0.0)
        self.assertEqual(snapshot.food.proximity_left, 0.0)
        self.assertEqual(snapshot.food.proximity_right, 0.0)
        self.assertEqual(self.vision.visible_foods(creature, [front_food]), [front_food])

    def test_sector_proximities_use_max_closeness_per_sector(self) -> None:
        creature = creature_at((0.0, 0.0), vision_range=100.0, vision_angle=pi / 2)
        left_food = FakeFood(
            id=1,
            position=(40.0 * cos(-pi / 6), 40.0 * sin(-pi / 6)),
            radius=0.0,
        )
        center_food = FakeFood(id=2, position=(70.0, 0.0), radius=0.0)
        right_food = FakeFood(
            id=3,
            position=(20.0 * cos(pi / 6), 20.0 * sin(pi / 6)),
            radius=0.0,
        )

        snapshot = self.sense_snapshot(creature, [left_food, center_food, right_food])

        self.assertEqual(snapshot.food.count, 3)
        self.assertAlmostEqual(snapshot.food.proximity_left, 0.6)
        self.assertAlmostEqual(snapshot.food.proximity_center, 0.3)
        self.assertAlmostEqual(snapshot.food.proximity_right, 0.8)

    def test_close_food_interval_activates_all_sectors(self) -> None:
        creature = creature_at((0.0, 0.0), vision_range=100.0, vision_angle=pi / 2)
        close_food = FakeFood(id=1, position=(10.0, 0.0), radius=10.0)

        snapshot = self.sense_snapshot(creature, [close_food])

        self.assertEqual(snapshot.food.count, 1)
        self.assertAlmostEqual(snapshot.food.proximity_left, 1.0)
        self.assertAlmostEqual(snapshot.food.proximity_center, 1.0)
        self.assertAlmostEqual(snapshot.food.proximity_right, 1.0)

    def test_food_interval_overlap_can_activate_two_adjacent_sectors(self) -> None:
        creature = creature_at((0.0, 0.0), vision_range=100.0, vision_angle=pi / 2)
        food = FakeFood(
            id=1,
            position=(50.0 * cos(-0.35), 50.0 * sin(-0.35)),
            radius=6.0,
        )

        snapshot = self.sense_snapshot(creature, [food])

        self.assertEqual(snapshot.food.count, 1)
        self.assertAlmostEqual(snapshot.food.proximity_left, 0.56)
        self.assertAlmostEqual(snapshot.food.proximity_center, 0.56)
        self.assertAlmostEqual(snapshot.food.proximity_right, 0.0)


class VisionWallSensorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vision = VisionSystem(VisionConfig())

    def sense_inputs(
        self,
        creature: FakeCreature,
        bounds: tuple[float, float, float, float],
    ) -> list[float]:
        return self.sense_snapshot(creature, bounds).as_inputs()

    def sense_snapshot(
        self,
        creature: FakeCreature,
        bounds: tuple[float, float, float, float],
    ):
        return self.vision.sense(
            creature,
            foods=[],
            creatures=[],
            world_bounds=bounds,
            max_speed=100.0,
        )

    def test_wall_directly_ahead_within_range_is_visible(self) -> None:
        snapshot = self.sense_snapshot(
            creature_at((55.0, 50.0), radius=5.0, vision_range=50.0),
            (0.0, 0.0, 100.0, 100.0),
        )

        self.assertGreater(snapshot.walls.proximity_center, 0.0)
        self.assertAlmostEqual(snapshot.walls.proximity_center, 0.2)
        self.assertEqual(snapshot.walls.proximity_left, 0.0)
        self.assertEqual(snapshot.walls.proximity_right, 0.0)

    def test_wall_outside_vision_cone_is_not_visible(self) -> None:
        snapshot = self.sense_snapshot(
            creature_at(
                (90.0, 50.0),
                heading=pi / 2,
                vision_range=30.0,
                vision_angle=pi / 6,
            ),
            (0.0, 0.0, 100.0, 200.0),
        )

        self.assertEqual(snapshot.walls.proximity_left, 0.0)
        self.assertEqual(snapshot.walls.proximity_center, 0.0)
        self.assertEqual(snapshot.walls.proximity_right, 0.0)

    def test_wall_at_left_edge_reports_left_sector(self) -> None:
        snapshot = self.sense_snapshot(
            creature_at((50.0, 50.0), vision_range=100.0),
            (0.0, 0.0, 150.0, 300.0),
        )

        self.assertGreater(snapshot.walls.proximity_left, 0.0)
        self.assertEqual(snapshot.walls.proximity_right, 0.0)

    def test_wall_at_right_edge_reports_right_sector(self) -> None:
        snapshot = self.sense_snapshot(
            creature_at((50.0, 50.0), vision_range=100.0),
            (0.0, -200.0, 150.0, 100.0),
        )

        self.assertEqual(snapshot.walls.proximity_left, 0.0)
        self.assertGreater(snapshot.walls.proximity_right, 0.0)

    def test_wall_fuzzy_interval_activates_adjacent_sector_boundary(self) -> None:
        snapshot = self.sense_snapshot(
            creature_at((0.0, 0.0), vision_range=100.0, vision_angle=pi / 2),
            (50.0, 13.0, 51.0, 14.0),
        )

        self.assertEqual(snapshot.walls.proximity_left, 0.0)
        self.assertGreater(snapshot.walls.proximity_center, 0.0)
        self.assertGreater(snapshot.walls.proximity_right, 0.0)

    def test_wall_farther_than_vision_range_is_not_visible(self) -> None:
        snapshot = self.sense_snapshot(
            creature_at((50.0, 50.0), vision_range=40.0),
            (0.0, 0.0, 200.0, 200.0),
        )

        self.assertEqual(snapshot.walls.proximity_left, 0.0)
        self.assertEqual(snapshot.walls.proximity_center, 0.0)
        self.assertEqual(snapshot.walls.proximity_right, 0.0)

    def test_sensor_input_contract_includes_wall_and_grabbing_inputs(self) -> None:
        inputs = self.sense_inputs(
            creature_at((55.0, 50.0), radius=5.0, vision_range=50.0),
            (0.0, 0.0, 100.0, 100.0),
        )

        self.assertEqual(SENSOR_INPUT_COUNT, 20)
        self.assertEqual(len(inputs), SENSOR_INPUT_COUNT)
        self.assertAlmostEqual(inputs[0], 1.0)
        self.assertAlmostEqual(inputs[1], 0.25)
        self.assertAlmostEqual(inputs[3], 0.75)
        self.assertAlmostEqual(inputs[16], 0.0)
        self.assertAlmostEqual(inputs[17], 0.2)
        self.assertAlmostEqual(inputs[18], 0.0)
        self.assertAlmostEqual(inputs[19], 0.0)

    def test_grabbing_input_is_binary_and_appended_to_sensor_contract(self) -> None:
        snapshot = self.vision.sense(
            creature_at((55.0, 50.0), radius=5.0, vision_range=50.0),
            foods=[],
            creatures=[],
            world_bounds=(0.0, 0.0, 100.0, 100.0),
            max_speed=100.0,
            is_grabbing=True,
        )

        self.assertEqual(snapshot.is_grabbing, 1.0)
        self.assertEqual(snapshot.as_inputs()[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
