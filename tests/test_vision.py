from __future__ import annotations

import unittest
from dataclasses import dataclass
from math import atan2, cos, pi, sin
import sys
import types
from unittest.mock import patch


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

from configs.sim_config import (
    FlockingConfig,
    MetabolismConfig,
    SocialCompatibilityConfig,
    SocialCompatibilityMode,
    VisionConfig,
)
from src.creature import VisionTraits
from src.metabolism import Metabolism
from src.vision import SENSOR_INPUT_COUNT, SENSOR_INPUT_NAMES, VisionSystem


@dataclass(slots=True)
class FakeCreature:
    position: tuple[float, float]
    radius: float
    heading: float
    energy: float
    vision: VisionTraits
    velocity: tuple[float, float] = (0.0, 0.0)
    speed: float = 0.0
    creature_id: int = 1
    species_id: int = 1
    stomach_energy: float = 0.0

    @property
    def body(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            velocity=types.SimpleNamespace(
                x=self.velocity[0],
                y=self.velocity[1],
            )
        )


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
    velocity: tuple[float, float] = (0.0, 0.0),
    creature_id: int = 1,
    species_id: int = 1,
) -> FakeCreature:
    return FakeCreature(
        position=position,
        radius=radius,
        heading=heading,
        energy=energy,
        vision=VisionTraits(range=vision_range, angle=vision_angle),
        velocity=velocity,
        creature_id=creature_id,
        species_id=species_id,
    )


class VisionVisibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vision = VisionSystem(VisionConfig())

    def test_signed_angle_wraps_large_accumulated_headings(self) -> None:
        full_turns = 10_000 * 2.0 * pi

        self.assertAlmostEqual(
            self.vision._signed_angle(full_turns + pi / 3.0),
            pi / 3.0,
            places=10,
        )
        self.assertAlmostEqual(
            self.vision._signed_angle(-full_turns - pi / 3.0),
            -pi / 3.0,
            places=10,
        )

    def sense_snapshot(
        self,
        creature: FakeCreature,
        *,
        foods: list[FakeFood] | None = None,
        creatures: list[FakeCreature] | None = None,
        own_infants: list[FakeCreature] | None = None,
    ):
        return self.vision.sense(
            creature,
            foods=[] if foods is None else foods,
            creatures=[] if creatures is None else creatures,
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
            own_infants=own_infants,
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
        self.assertEqual(snapshot.flock.flockmate_count, 2)
        self.assertGreater(snapshot.creatures.proximity, 0.0)
        self.assertAlmostEqual(snapshot.creatures.angle, 0.0)
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

    def test_partly_exposed_food_behind_creature_remains_visible(self) -> None:
        observer = creature_at((0.0, 0.0), radius=5.0)
        blocker = creature_at((30.0, 0.0), radius=10.0, creature_id=2)
        exposed_food = FakeFood(id=1, position=(60.0, 20.0), radius=5.0)

        snapshot = self.sense_snapshot(
            observer,
            foods=[exposed_food],
            creatures=[observer, blocker],
        )

        self.assertEqual(snapshot.food.count, 1)
        self.assertEqual(
            self.vision.visible_foods(observer, [exposed_food], [observer, blocker]),
            [exposed_food],
        )

    def test_occluded_food_is_excluded_from_visible_food_ids(self) -> None:
        observer = creature_at((0.0, 0.0), radius=5.0)
        blocker = creature_at((30.0, 0.0), radius=10.0, creature_id=2)
        hidden_food = FakeFood(id=1, position=(60.0, 0.0), radius=5.0)
        exposed_food = FakeFood(id=2, position=(40.0, 25.0), radius=5.0)

        with patch("src.vision.Food", FakeFood):
            result = self.vision.sense_with_visible_food_ids(
                observer,
                foods=[hidden_food, exposed_food],
                creatures=[observer, blocker],
                world_bounds=(-100.0, -100.0, 100.0, 100.0),
                max_speed=100.0,
            )

        self.assertEqual(result.snapshot.food.count, 1)
        self.assertEqual(result.visible_food_ids, [exposed_food.id])
        exposed_candidate = self.vision._vision_candidate(
            observer,
            "food",
            exposed_food,
            exposed_food.position,
            exposed_food.radius,
        )
        self.assertIsNotNone(exposed_candidate)
        self.assertAlmostEqual(
            result.snapshot.food.proximity,
            exposed_candidate.closeness,
        )
        self.assertAlmostEqual(
            result.snapshot.food.angle,
            exposed_candidate.signed_angle / (pi / 4.0),
        )

    def test_own_infant_behind_creature_is_occluded(self) -> None:
        observer = creature_at((0.0, 0.0), radius=5.0)
        blocker = creature_at((30.0, 0.0), radius=10.0, creature_id=2)
        hidden_infant = creature_at((60.0, 0.0), radius=5.0, creature_id=3)

        snapshot = self.sense_snapshot(
            observer,
            creatures=[observer, blocker, hidden_infant],
            own_infants=[hidden_infant],
        )

        self.assertEqual(snapshot.creatures.count, 1)
        self.assertEqual(snapshot.own_infants.count, 0)
        self.assertEqual(snapshot.flock.flockmate_count, 2)

    def test_visible_own_infant_is_not_occluded_by_its_creature_entry(self) -> None:
        observer = creature_at((0.0, 0.0), radius=5.0)
        infant = creature_at((40.0, 0.0), radius=5.0, creature_id=2)

        snapshot = self.sense_snapshot(
            observer,
            creatures=[observer, infant],
            own_infants=[infant],
        )

        self.assertEqual(snapshot.creatures.count, 1)
        self.assertEqual(snapshot.own_infants.count, 1)
        self.assertEqual(snapshot.flock.flockmate_count, 1)

    def test_own_infant_snapshot_uses_target_proximity_and_angle(self) -> None:
        observer = creature_at((0.0, 0.0), radius=5.0)
        infant = creature_at((46.75, 0.0), radius=5.0, creature_id=2)

        snapshot = self.sense_snapshot(observer, own_infants=[infant])

        self.assertEqual(snapshot.own_infants.count, 1)
        self.assertAlmostEqual(snapshot.own_infants.proximity, 0.6)
        self.assertAlmostEqual(snapshot.own_infants.angle, 0.0)

    def test_own_infant_snapshot_ignores_unlisted_infants(self) -> None:
        observer = creature_at((0.0, 0.0), radius=5.0)
        unrelated_infant = creature_at((45.0, 0.0), radius=5.0, creature_id=2)

        snapshot = self.sense_snapshot(observer, creatures=[observer, unrelated_infant])

        self.assertEqual(snapshot.creatures.count, 1)
        self.assertEqual(snapshot.own_infants.count, 0)
        self.assertAlmostEqual(snapshot.own_infants.proximity, 0.0)
        self.assertAlmostEqual(snapshot.own_infants.angle, 0.0)

    def test_creature_behind_food_remains_visible(self) -> None:
        observer = creature_at((0.0, 0.0), radius=5.0)
        blocker_food = FakeFood(id=1, position=(30.0, 0.0), radius=10.0)
        hidden_creature = creature_at((60.0, 0.0), radius=5.0, creature_id=2)

        snapshot = self.sense_snapshot(
            observer,
            foods=[blocker_food],
            creatures=[observer, hidden_creature],
        )

        self.assertEqual(snapshot.creatures.count, 1)
        self.assertEqual(
            self.vision.visible_creatures(
                observer,
                [observer, hidden_creature],
                [blocker_food],
            ),
            [hidden_creature],
        )

    def test_food_behind_food_remains_visible(self) -> None:
        observer = creature_at((0.0, 0.0), radius=5.0)
        blocker_food = FakeFood(id=1, position=(30.0, 0.0), radius=10.0)
        hidden_food = FakeFood(id=2, position=(60.0, 0.0), radius=5.0)

        snapshot = self.sense_snapshot(observer, foods=[blocker_food, hidden_food])

        self.assertEqual(snapshot.food.count, 2)
        self.assertEqual(
            self.vision.visible_foods(observer, [blocker_food, hidden_food]),
            [blocker_food, hidden_food],
        )

    def test_ignored_food_is_not_reported(self) -> None:
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


class VisionEyeOriginTest(unittest.TestCase):
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

    def test_side_body_food_inside_cone_is_visible(self) -> None:
        creature = creature_at((0.0, 0.0), radius=10.0, heading=0.0)
        side_food = FakeFood(id=1, position=(10.0, 8.0), radius=3.0)

        snapshot = self.sense_snapshot(creature, [side_food])

        self.assertEqual(snapshot.food.count, 1)
        self.assertEqual(self.vision.visible_foods(creature, [side_food]), [side_food])

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
        self.assertAlmostEqual(snapshot.food.proximity, 1.0)
        self.assertAlmostEqual(snapshot.food.angle, 0.0)
        self.assertEqual(self.vision.visible_foods(creature, [mouth_food]), [mouth_food])

    def test_food_at_mouth_threshold_remains_visible(self) -> None:
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
        self.assertAlmostEqual(snapshot.food.proximity, 1.0)
        self.assertAlmostEqual(snapshot.food.angle, 0.0)

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

    def test_lateral_mouth_contact_drives_straight(self) -> None:
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
        self.assertAlmostEqual(snapshot.food.proximity, 1.0)
        self.assertAlmostEqual(snapshot.food.angle, 0.0)

    def test_narrow_fov_mouth_contact_drives_straight(self) -> None:
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
        self.assertAlmostEqual(snapshot.food.proximity, 1.0)
        self.assertAlmostEqual(snapshot.food.angle, 0.0)

    def test_mouth_contact_food_does_not_hide_farther_food(self) -> None:
        creature = creature_at((0.0, 0.0), radius=10.0, heading=0.0)
        mouth_food = FakeFood(id=1, position=(13.0, 0.0), radius=3.0)
        farther_food = FakeFood(id=2, position=(40.0, 0.0), radius=4.0)

        snapshot = self.sense_snapshot(creature, [mouth_food, farther_food])

        self.assertEqual(snapshot.food.count, 2)
        self.assertAlmostEqual(snapshot.food.proximity, 1.0)
        self.assertAlmostEqual(snapshot.food.angle, 0.0)
        self.assertEqual(
            self.vision.visible_foods(creature, [mouth_food, farther_food]),
            [mouth_food, farther_food],
        )

    def test_mouth_contact_food_does_not_hide_farther_creature(self) -> None:
        creature = creature_at((0.0, 0.0), radius=10.0, heading=0.0)
        mouth_food = FakeFood(id=1, position=(13.0, 0.0), radius=3.0)
        farther_creature = creature_at((40.0, 0.0), radius=5.0, creature_id=2)

        snapshot = self.vision.sense(
            creature,
            foods=[mouth_food],
            creatures=[creature, farther_creature],
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
        )

        self.assertEqual(snapshot.food.count, 1)
        self.assertEqual(snapshot.creatures.count, 1)
        self.assertEqual(
            self.vision.visible_creatures(
                creature,
                [creature, farther_creature],
                [mouth_food],
            ),
            [farther_creature],
        )

    def test_zero_vision_range_returns_empty_targets(self) -> None:
        creature = creature_at((0.0, 0.0), radius=10.0, vision_range=0.0)
        food = FakeFood(id=1, position=(13.0, 0.0), radius=3.0)
        other = creature_at((10.0, 0.0), radius=3.0, creature_id=2)

        snapshot = self.vision.sense(
            creature,
            foods=[food],
            creatures=[creature, other],
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
        )

        self.assertEqual(snapshot.food.count, 0)
        self.assertEqual(snapshot.creatures.count, 0)

    def test_zero_vision_angle_returns_empty_targets(self) -> None:
        creature = creature_at((0.0, 0.0), radius=10.0, vision_angle=0.0)
        food = FakeFood(id=1, position=(13.0, 0.0), radius=3.0)
        other = creature_at((10.0, 0.0), radius=3.0, creature_id=2)

        snapshot = self.vision.sense(
            creature,
            foods=[food],
            creatures=[creature, other],
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
        )

        self.assertEqual(snapshot.food.count, 0)
        self.assertEqual(snapshot.creatures.count, 0)

    def test_forward_food_near_mouth_remains_visible(self) -> None:
        creature = creature_at((0.0, 0.0), radius=10.0, heading=0.0)
        forward_food = FakeFood(id=1, position=(17.0, 0.0), radius=3.0)

        snapshot = self.sense_snapshot(creature, [forward_food])

        self.assertEqual(snapshot.food.count, 1)
        self.assertGreater(snapshot.food.proximity, 0.0)
        self.assertAlmostEqual(snapshot.food.angle, 0.0)
        self.assertEqual(
            self.vision.visible_foods(creature, [forward_food]),
            [forward_food],
        )

    def test_food_farther_in_eye_cone_remains_visible(self) -> None:
        creature = creature_at((0.0, 0.0), radius=10.0, heading=0.0)
        front_food = FakeFood(id=1, position=(24.0, 0.0), radius=3.0)

        snapshot = self.sense_snapshot(creature, [front_food])

        self.assertEqual(snapshot.food.count, 1)
        self.assertGreater(snapshot.food.proximity, 0.0)
        self.assertAlmostEqual(snapshot.food.angle, 0.0)
        self.assertEqual(self.vision.visible_foods(creature, [front_food]), [front_food])

    def test_centered_food_reports_proximity_and_zero_angle(self) -> None:
        creature = creature_at((0.0, 0.0), vision_range=100.0, vision_angle=pi / 2)
        center_food = FakeFood(
            id=1,
            position=(50.0, 0.0),
            radius=0.0,
        )

        snapshot = self.sense_snapshot(creature, [center_food])

        self.assertEqual(snapshot.food.count, 1)
        self.assertAlmostEqual(snapshot.food.proximity, 0.5)
        self.assertAlmostEqual(snapshot.food.angle, 0.0)

    def test_left_food_reports_positive_counterclockwise_angle(self) -> None:
        creature = creature_at((0.0, 0.0), vision_range=100.0, vision_angle=pi / 2)
        left_food = FakeFood(
            id=1,
            position=(50.0 * cos(0.35), 50.0 * sin(0.35)),
            radius=1.0,
        )

        snapshot = self.sense_snapshot(creature, [left_food])

        self.assertEqual(snapshot.food.count, 1)
        self.assertAlmostEqual(snapshot.food.proximity, 0.51)
        self.assertAlmostEqual(snapshot.food.angle, 0.35 / (pi / 4))

    def test_right_food_reports_negative_clockwise_angle(self) -> None:
        creature = creature_at((0.0, 0.0), vision_range=100.0, vision_angle=pi / 2)
        right_food = FakeFood(
            id=1,
            position=(50.0 * cos(-0.35), 50.0 * sin(-0.35)),
            radius=1.0,
        )

        snapshot = self.sense_snapshot(creature, [right_food])

        self.assertEqual(snapshot.food.count, 1)
        self.assertAlmostEqual(snapshot.food.proximity, 0.51)
        self.assertAlmostEqual(snapshot.food.angle, -0.35 / (pi / 4))

    def test_nearest_target_supplies_both_proximity_and_angle(self) -> None:
        creature = creature_at((0.0, 0.0), vision_range=100.0, vision_angle=pi / 2)
        left_food = FakeFood(
            id=1,
            position=(50.0 * cos(-0.4), 50.0 * sin(-0.4)),
            radius=5.0,
        )
        center_food = FakeFood(id=2, position=(50.0, 0.0), radius=5.0)
        nearest_left_food = FakeFood(
            id=3,
            position=(30.0 * cos(0.4), 30.0 * sin(0.4)),
            radius=5.0,
        )

        snapshot = self.sense_snapshot(
            creature,
            [left_food, center_food, nearest_left_food],
        )

        self.assertEqual(snapshot.food.count, 3)
        self.assertAlmostEqual(snapshot.food.proximity, 0.75)
        self.assertAlmostEqual(snapshot.food.angle, 0.4 / (pi / 4))
        self.assertEqual(snapshot.food.nearest_id, 3)
        self.assertAlmostEqual(snapshot.food.surface_distance, 25.0)
        self.assertAlmostEqual(snapshot.food.relative_angle, 0.4)

    def test_close_food_reports_max_proximity_and_center_angle(self) -> None:
        creature = creature_at((0.0, 0.0), vision_range=100.0, vision_angle=pi / 2)
        close_food = FakeFood(id=1, position=(10.0, 0.0), radius=10.0)

        snapshot = self.sense_snapshot(creature, [close_food])

        self.assertEqual(snapshot.food.count, 1)
        self.assertAlmostEqual(snapshot.food.proximity, 1.0)
        self.assertAlmostEqual(snapshot.food.angle, 0.0)

    def test_food_angle_is_continuous_near_old_sector_boundary(self) -> None:
        creature = creature_at((0.0, 0.0), vision_range=100.0, vision_angle=pi / 2)
        food = FakeFood(
            id=1,
            position=(50.0 * cos(-0.3), 50.0 * sin(-0.3)),
            radius=6.0,
        )

        snapshot = self.sense_snapshot(creature, [food])

        self.assertEqual(snapshot.food.count, 1)
        self.assertAlmostEqual(snapshot.food.proximity, 0.56)
        self.assertAlmostEqual(snapshot.food.angle, -0.3 / (pi / 4))


class VisionWallSensorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vision = VisionSystem(
            VisionConfig(),
            flocking_config=FlockingConfig(
                compatibility=SocialCompatibilityConfig(
                    mode=SocialCompatibilityMode.SPECIES
                )
            ),
        )

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

        self.assertGreater(snapshot.walls.proximity, 0.0)
        self.assertAlmostEqual(snapshot.walls.proximity, 0.2)
        self.assertAlmostEqual(snapshot.walls.angle, 0.0)

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

        self.assertEqual(snapshot.walls.proximity, 0.0)
        self.assertEqual(snapshot.walls.angle, 0.0)

    def test_wall_below_heading_reports_negative_angle(self) -> None:
        snapshot = self.sense_snapshot(
            creature_at((50.0, 50.0), vision_range=100.0),
            (0.0, 0.0, 150.0, 300.0),
        )

        self.assertGreater(snapshot.walls.proximity, 0.0)
        self.assertLess(snapshot.walls.angle, 0.0)

    def test_wall_above_heading_reports_positive_angle(self) -> None:
        snapshot = self.sense_snapshot(
            creature_at((50.0, 50.0), vision_range=100.0),
            (0.0, -200.0, 150.0, 100.0),
        )

        self.assertGreater(snapshot.walls.proximity, 0.0)
        self.assertGreater(snapshot.walls.angle, 0.0)

    def test_wall_angle_is_continuous_near_old_sector_boundary(self) -> None:
        snapshot = self.sense_snapshot(
            creature_at((0.0, 0.0), vision_range=100.0, vision_angle=pi / 2),
            (50.0, 13.0, 51.0, 14.0),
        )

        self.assertGreater(snapshot.walls.proximity, 0.0)
        self.assertGreater(snapshot.walls.angle, 0.0)

    def test_wall_farther_than_vision_range_is_not_visible(self) -> None:
        snapshot = self.sense_snapshot(
            creature_at((50.0, 50.0), vision_range=40.0),
            (0.0, 0.0, 200.0, 200.0),
        )

        self.assertEqual(snapshot.walls.proximity, 0.0)
        self.assertEqual(snapshot.walls.angle, 0.0)

    def test_sensor_input_contract_includes_wall_grabbing_and_biome_inputs(self) -> None:
        inputs = self.sense_inputs(
            creature_at((55.0, 50.0), radius=5.0, vision_range=50.0),
            (0.0, 0.0, 100.0, 100.0),
        )
        first_seventeen_inputs = inputs[:17]

        self.assertEqual(SENSOR_INPUT_COUNT, 43)
        self.assertEqual(len(inputs), SENSOR_INPUT_COUNT)
        self.assertEqual(SENSOR_INPUT_NAMES[1], "feeding_drive")
        self.assertEqual(SENSOR_INPUT_NAMES[2], "reproductive_readiness")
        self.assertEqual(
            SENSOR_INPUT_NAMES[17:21],
            (
                "biome_fertility_here",
                "biome_fertility_left_gradient",
                "biome_fertility_right_gradient",
                "biome_fertility_trend",
            ),
        )
        self.assertAlmostEqual(first_seventeen_inputs[0], 1.0)
        self.assertAlmostEqual(first_seventeen_inputs[1], 0.25)
        self.assertAlmostEqual(first_seventeen_inputs[3], 0.75)
        self.assertAlmostEqual(first_seventeen_inputs[14], 0.2)
        self.assertAlmostEqual(first_seventeen_inputs[15], 0.0)
        self.assertAlmostEqual(first_seventeen_inputs[16], 0.0)
        self.assertGreaterEqual(inputs[17], 0.0)
        self.assertLessEqual(inputs[17], 1.0)
        for biome_gradient_or_trend in inputs[18:21]:
            self.assertGreaterEqual(biome_gradient_or_trend, -1.0)
            self.assertLessEqual(biome_gradient_or_trend, 1.0)
        self.assertAlmostEqual(inputs[21], 0.0)
        self.assertAlmostEqual(inputs[22], 0.0)
        self.assertEqual(inputs[23:32], [0.0] * 9)
        self.assertEqual(inputs[32:43], [0.0] * 11)

    def test_stomach_fullness_is_the_33rd_input_and_clamps(self) -> None:
        creature = creature_at((50.0, 50.0), radius=10.0)
        creature.stomach_energy = 0.5
        inputs = self.sense_inputs(creature, (0.0, 0.0, 100.0, 100.0))
        self.assertAlmostEqual(inputs[32], 0.5)

        creature.stomach_energy = 2.0
        inputs = self.sense_inputs(creature, (0.0, 0.0, 100.0, 100.0))
        self.assertEqual(inputs[32], 1.0)

    def test_feeding_drive_requires_low_energy_and_stomach_capacity(self) -> None:
        creature = creature_at((50.0, 50.0), radius=10.0, energy=0.1)
        creature.stomach_energy = 0.25

        inputs = self.sense_inputs(creature, (0.0, 0.0, 100.0, 100.0))
        self.assertAlmostEqual(inputs[1], 0.9 * 0.75)

        creature.stomach_energy = 1.0
        inputs = self.sense_inputs(creature, (0.0, 0.0, 100.0, 100.0))
        self.assertEqual(inputs[1], 0.0)

        creature.stomach_energy = 0.0
        inputs = self.sense_inputs(creature, (0.0, 0.0, 100.0, 100.0))
        self.assertAlmostEqual(inputs[1], 0.9)

    def test_cohesion_and_alignment_use_only_same_species_creatures(
        self,
    ) -> None:
        observer = creature_at((0.0, 0.0), radius=10.0, vision_range=100.0)
        flockmate = creature_at(
            (40.0, 10.0),
            heading=pi / 2,
            velocity=(0.0, 10.0),
            creature_id=2,
            species_id=1,
        )
        other_species = creature_at(
            (20.0, -2.0),
            heading=-pi / 2,
            creature_id=3,
            species_id=2,
        )

        snapshot = self.vision.sense(
            observer,
            foods=[],
            creatures=[observer, flockmate, other_species],
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
        )

        self.assertEqual(snapshot.flock.flockmate_count, 1)
        self.assertEqual(snapshot.visible_creature_count, 2)
        self.assertAlmostEqual(
            snapshot.flock.center_proximity,
            1.0 - ((40.0**2 + 10.0**2) ** 0.5 / 150.0),
        )
        self.assertGreater(snapshot.flock.center_angle, 0.0)
        self.assertAlmostEqual(snapshot.flock.average_relative_heading, 0.5)
        self.assertAlmostEqual(
            snapshot.flock.average_flockmate_proximity,
            1.0 - ((40.0**2 + 10.0**2) ** 0.5 / 150.0),
        )
        self.assertGreater(snapshot.flock.crowd_separation_strength, 0.0)
        self.assertLess(snapshot.flock.crowd_separation_absolute_angle, 0.0)
        self.assertEqual(snapshot.flock.average_flockmate_velocity, (0.0, 10.0))

    def test_symmetric_neighbors_cancel_separation_field(self) -> None:
        observer = creature_at(
            (0.0, 0.0),
            radius=5.0,
            vision_range=100.0,
            vision_angle=2.0 * pi,
        )
        left = creature_at((-10.0, 0.0), creature_id=2)
        right = creature_at((10.0, 0.0), creature_id=3)

        snapshot = self.vision.sense(
            observer,
            foods=[],
            creatures=[observer, left, right],
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
        )

        self.assertAlmostEqual(snapshot.flock.crowd_separation_strength, 0.0)
        self.assertAlmostEqual(snapshot.flock.crowd_separation_absolute_angle, 0.0)

    def test_crowded_left_side_produces_rightward_separation(self) -> None:
        observer = creature_at(
            (0.0, 0.0),
            radius=5.0,
            vision_range=100.0,
            vision_angle=2.0 * pi,
        )
        neighbors = [
            creature_at((-10.0, -5.0), creature_id=2),
            creature_at((-10.0, 0.0), creature_id=3),
            creature_at((-10.0, 5.0), creature_id=4),
            creature_at((10.0, 0.0), creature_id=5),
        ]

        snapshot = self.vision.sense(
            observer,
            foods=[],
            creatures=[observer, *neighbors],
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
        )

        self.assertGreater(snapshot.flock.crowd_separation_strength, 0.0)
        self.assertAlmostEqual(
            snapshot.flock.crowd_separation_absolute_angle,
            0.0,
            places=10,
        )

    def test_separation_ignores_visible_neighbors_outside_personal_space(
        self,
    ) -> None:
        observer = creature_at(
            (0.0, 0.0),
            radius=10.0,
            vision_range=200.0,
            vision_angle=2.0 * pi,
        )
        close_neighbor = creature_at((2.0, 0.0), creature_id=2)
        distant_neighbors = [
            creature_at(
                (
                    cos((index + 1) * 0.5) * 150.0,
                    sin((index + 1) * 0.5) * 150.0,
                ),
                creature_id=index + 3,
            )
            for index in range(10)
        ]

        snapshot = self.vision.sense(
            observer,
            foods=[],
            creatures=[observer, close_neighbor, *distant_neighbors],
            world_bounds=(-250.0, -250.0, 250.0, 250.0),
            max_speed=100.0,
        )

        self.assertAlmostEqual(
            snapshot.flock.crowd_separation_strength,
            1.0 - 2.0 / 60.0,
        )
        self.assertAlmostEqual(
            snapshot.flock.crowd_separation_absolute_angle,
            pi,
        )

    def test_personal_space_boundary_is_strictly_excluded(self) -> None:
        observer = creature_at(
            (0.0, 0.0),
            radius=10.0,
            vision_range=100.0,
        )
        boundary_neighbor = creature_at((60.0, 0.0), creature_id=2)

        snapshot = self.vision.sense(
            observer,
            foods=[],
            creatures=[observer, boundary_neighbor],
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
        )

        self.assertEqual(snapshot.flock.crowd_separation_strength, 0.0)

    def test_crowd_separation_falls_continuously_to_zero_at_four_radii(
        self,
    ) -> None:
        observer = creature_at(
            (0.0, 0.0),
            radius=10.0,
            vision_range=100.0,
        )

        just_inside = self.vision.sense(
            observer,
            foods=[],
            creatures=[
                observer,
                creature_at((59.999, 0.0), creature_id=2),
            ],
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
        )
        at_boundary = self.vision.sense(
            observer,
            foods=[],
            creatures=[observer, creature_at((60.0, 0.0), creature_id=3)],
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
        )

        self.assertAlmostEqual(
            just_inside.flock.crowd_separation_strength,
            1.0 - 59.999 / 60.0,
        )
        self.assertEqual(at_boundary.flock.crowd_separation_strength, 0.0)

    def test_multiple_neighbors_accumulate_crowd_separation_before_clamping(
        self,
    ) -> None:
        observer = creature_at(
            (0.0, 0.0),
            radius=10.0,
            vision_range=100.0,
            vision_angle=2.0 * pi,
        )
        first = creature_at((10.0, -4.0), creature_id=2)
        second = creature_at((10.0, 4.0), creature_id=3)

        one_neighbor = self.vision.sense(
            observer,
            foods=[],
            creatures=[observer, first],
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
        )
        two_neighbors = self.vision.sense(
            observer,
            foods=[],
            creatures=[observer, first, second],
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
        )

        self.assertGreater(
            two_neighbors.flock.crowd_separation_strength,
            one_neighbor.flock.crowd_separation_strength,
        )
        self.assertEqual(two_neighbors.flock.crowd_separation_strength, 1.0)

    def test_compatibility_weights_all_flock_means_and_effective_count(
        self,
    ) -> None:
        self.vision.flock_compatibility_resolver = (
            lambda _observer, neighbor: {2: 0.25, 3: 0.75}[neighbor.creature_id]
        )
        observer = creature_at(
            (0.0, 0.0),
            vision_range=100.0,
            vision_angle=2.0 * pi,
        )
        first = creature_at(
            (20.0, -10.0),
            velocity=(4.0, 0.0),
            creature_id=2,
        )
        second = creature_at(
            (20.0, 10.0),
            velocity=(0.0, 8.0),
            creature_id=3,
        )

        snapshot = self.vision.sense(
            observer,
            foods=[],
            creatures=[observer, first, second],
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
        )

        self.assertEqual(snapshot.flock.flockmate_count, 1.0)
        self.assertAlmostEqual(snapshot.as_inputs()[24], 0.25)
        self.assertEqual(snapshot.flock.average_flockmate_velocity, (1.0, 6.0))
        self.assertAlmostEqual(
            snapshot.flock.average_flockmate_proximity,
            1.0 - ((20.0**2 + 10.0**2) ** 0.5 / 150.0),
        )
        self.assertAlmostEqual(snapshot.flock.cohesion_absolute_angle, atan2(5.0, 20.0))

    def test_zero_average_velocity_falls_back_to_observer_velocity(self) -> None:
        observer = creature_at(
            (0.0, 0.0),
            vision_range=100.0,
            vision_angle=2.0 * pi,
            velocity=(3.0, 4.0),
        )
        flockmates = [
            creature_at((20.0, -10.0), velocity=(10.0, 0.0), creature_id=2),
            creature_at((20.0, 10.0), velocity=(-10.0, 0.0), creature_id=3),
        ]

        snapshot = self.vision.sense(
            observer,
            foods=[],
            creatures=[observer, *flockmates],
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
        )

        self.assertEqual(snapshot.flock.average_flockmate_velocity, (3.0, 4.0))
        self.assertAlmostEqual(
            snapshot.flock.average_relative_heading,
            atan2(4.0, 3.0) / pi,
        )

    def test_close_other_species_creature_does_not_contribute_soft_force(
        self,
    ) -> None:
        observer = creature_at(
            (0.0, 0.0),
            radius=10.0,
            vision_range=100.0,
        )
        other_species = creature_at(
            (20.0, 0.0),
            creature_id=2,
            species_id=2,
            velocity=(0.0, 10.0),
        )

        snapshot = self.vision.sense(
            observer,
            foods=[],
            creatures=[observer, other_species],
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
        )

        self.assertEqual(snapshot.flock.flockmate_count, 0)
        self.assertEqual(snapshot.flock.crowd_separation_strength, 0.0)
        self.assertEqual(snapshot.flock.visible_personal_space_count, 0)
        self.assertEqual(snapshot.flock.center_proximity, 0.0)
        self.assertEqual(snapshot.flock.average_flockmate_velocity, (0.0, 0.0))

    def test_alignment_uses_flockmate_velocity_instead_of_heading(self) -> None:
        observer = creature_at((0.0, 0.0), vision_range=100.0)
        flockmates = [
            creature_at(
                (20.0, -5.0),
                heading=pi / 2,
                velocity=(10.0, 0.0),
                creature_id=2,
            ),
            creature_at(
                (20.0, 5.0),
                heading=pi / 2,
                velocity=(20.0, 0.0),
                creature_id=3,
            ),
        ]

        snapshot = self.vision.sense(
            observer,
            foods=[],
            creatures=[observer, *flockmates],
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
        )

        self.assertEqual(snapshot.flock.average_flockmate_velocity, (15.0, 0.0))
        self.assertAlmostEqual(snapshot.flock.average_relative_heading, 0.0)

    def test_average_flockmate_proximity_uses_distance_falloff(self) -> None:
        observer = creature_at((0.0, 0.0), vision_range=100.0)

        far_snapshot = self.vision.sense(
            observer,
            foods=[],
            creatures=[observer, creature_at((99.0, 0.0), creature_id=2)],
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
        )
        near_snapshot = self.vision.sense(
            observer,
            foods=[],
            creatures=[observer, creature_at((10.0, 0.0), creature_id=3)],
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
        )

        self.assertAlmostEqual(
            far_snapshot.flock.average_flockmate_proximity,
            1.0 - 99.0 / 150.0,
        )
        self.assertAlmostEqual(
            near_snapshot.flock.average_flockmate_proximity,
            1.0 - 10.0 / 150.0,
        )

    def test_flock_inputs_are_zero_without_same_species_flockmates(self) -> None:
        observer = creature_at((0.0, 0.0), vision_range=100.0)
        stranger = creature_at(
            (20.0, 0.0),
            creature_id=2,
            species_id=2,
        )

        snapshot = self.vision.sense(
            observer,
            foods=[],
            creatures=[observer, stranger],
            world_bounds=(-100.0, -100.0, 100.0, 100.0),
            max_speed=100.0,
        )

        self.assertEqual(snapshot.as_inputs()[23:32], [0.0] * 9)
        self.assertEqual(snapshot.flock.flockmate_count, 0)
        self.assertEqual(snapshot.flock.crowd_separation_strength, 0.0)
        self.assertEqual(snapshot.flock.average_flockmate_proximity, 0.0)

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
        self.assertEqual(snapshot.as_inputs()[16], 1.0)


if __name__ == "__main__":
    unittest.main()
