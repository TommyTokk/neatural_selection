from __future__ import annotations

from math import hypot, pi
from types import SimpleNamespace
import unittest

from configs.sim_config import SimConfig
from src.action import Action
from src.vision import FlockSensorSnapshot
from src.world import World


class FakeBody:
    def __init__(self) -> None:
        self.position = (0.0, 0.0)
        self.velocity = SimpleNamespace(x=0.0, y=0.0)
        self.angular_velocity = 0.0
        self.torque = 0.0
        self.applied_force = (0.0, 0.0)

    def apply_force_at_world_point(self, force, point) -> None:
        del point
        self.applied_force = tuple(force)


def action(**overrides: float) -> Action:
    values = {
        "accelerate": 0.0,
        "rotate": 0.0,
        "want_reproduce": 0.0,
        "want_eat": 0.0,
        "reset_chronometer": 0.0,
        "want_grab": 0.0,
        "want_release": 0.0,
    }
    values.update(overrides)
    return Action(**values)


class WorldFlockingMotionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World.__new__(World)
        self.world.config = SimConfig()
        self.world.config.action.action_smoothing_alpha = 1.0
        self.world.config.action.active_angular_velocity_retention = 1.0
        self.world.config.action.turn_control_gain = 1.0
        self.world._motion_commands = {}
        self.creature = SimpleNamespace(
            creature_id=1,
            heading=0.0,
            vision=SimpleNamespace(angle=pi / 2),
            body=FakeBody(),
            smoothed_rotation=0.0,
            smoothed_acceleration=0.0,
        )

    def test_action_smoothing_uses_default_alpha_on_first_tick(self) -> None:
        self.world.config.action.action_smoothing_alpha = (
            SimConfig().action.action_smoothing_alpha
        )

        self.world._apply_action(
            self.creature,
            action(accelerate=1.0, rotate=1.0),
            snapshot=SimpleNamespace(flock=FlockSensorSnapshot()),
            apply_stabilizers=False,
        )

        self.assertAlmostEqual(self.creature.smoothed_acceleration, 0.8)
        self.assertAlmostEqual(self.creature.smoothed_rotation, 0.8)
        self.assertAlmostEqual(self.creature.body.applied_force[0], 100.0)
        self.assertAlmostEqual(self.creature.body.applied_force[1], 0.0)
        self.assertAlmostEqual(self.world._motion_commands[1].effective_rotate, 0.8)

    def test_action_smoothing_converges_across_cached_action_ticks(self) -> None:
        self.world.config.action.action_smoothing_alpha = 0.3
        cached_action = action(accelerate=1.0, rotate=1.0)
        snapshot = SimpleNamespace(flock=FlockSensorSnapshot())

        for _ in range(3):
            self.world._apply_action(
                self.creature,
                cached_action,
                snapshot=snapshot,
                apply_stabilizers=False,
            )

        self.assertAlmostEqual(self.creature.smoothed_acceleration, 0.657)
        self.assertAlmostEqual(self.creature.smoothed_rotation, 0.657)
        self.assertAlmostEqual(self.creature.body.applied_force[0], 82.125)
        self.assertLess(self.world._motion_commands[1].effective_rotate, 1.0)

    def test_active_angular_velocity_damping_applies_after_turn_control(self) -> None:
        self.world.config.action.turn_response = 1.0
        self.world.config.action.turn_control_gain = 1.0
        self.world.config.action.active_angular_velocity_retention = 0.80

        self.world._apply_action(
            self.creature,
            action(rotate=1.0),
            snapshot=SimpleNamespace(flock=FlockSensorSnapshot()),
            apply_stabilizers=False,
        )

        self.assertAlmostEqual(
            self.creature.body.angular_velocity,
            self.world.MAX_ANGULAR_SPEED * 0.80,
        )

    def test_turn_control_gain_reduces_angular_speed_limit(self) -> None:
        self.world.config.action.turn_control_gain = 0.65

        self.world._apply_action(
            self.creature,
            action(rotate=1.0),
            snapshot=SimpleNamespace(flock=FlockSensorSnapshot()),
            apply_stabilizers=False,
        )

        self.assertAlmostEqual(
            self.world._motion_commands[1].max_angular_speed,
            self.world.MAX_ANGULAR_SPEED * 0.65,
        )

    def test_zero_new_outputs_preserve_legacy_force_and_limits(self) -> None:
        self.world._apply_action(
            self.creature,
            action(accelerate=0.5, rotate=0.25),
            snapshot=SimpleNamespace(flock=FlockSensorSnapshot()),
            apply_stabilizers=False,
        )

        self.assertAlmostEqual(self.creature.body.applied_force[0], 62.5)
        self.assertAlmostEqual(self.creature.body.applied_force[1], 0.0)
        command = self.world._motion_commands[1]
        self.assertAlmostEqual(command.max_speed, self.world.MAX_SPEED)
        self.assertAlmostEqual(
            command.max_angular_speed,
            self.world.MAX_ANGULAR_SPEED,
        )
        self.assertAlmostEqual(command.effective_rotate, 0.25)

    def test_full_panic_scales_force_speed_and_turn_limits_by_one_and_a_half(
        self,
    ) -> None:
        self.world._apply_action(
            self.creature,
            action(accelerate=1.0, flee_panic_intensity=1.0),
            snapshot=SimpleNamespace(flock=FlockSensorSnapshot()),
            apply_stabilizers=False,
        )

        self.assertAlmostEqual(self.creature.body.applied_force[0], 187.5)
        command = self.world._motion_commands[1]
        self.assertAlmostEqual(command.max_speed, self.world.MAX_SPEED * 1.5)
        self.assertAlmostEqual(
            command.max_angular_speed,
            self.world.MAX_ANGULAR_SPEED * 1.5,
        )

    def test_separation_alignment_and_cohesion_have_independent_weights(self) -> None:
        flock = FlockSensorSnapshot(
            center_proximity=0.25,
            center_angle=0.5,
            average_relative_heading=0.5,
            flockmate_count=1,
            nearest_neighbor_proximity=1.0,
            nearest_neighbor_angle=0.0,
        )
        snapshot = SimpleNamespace(flock=flock)

        separation = self.world._flock_steering_force(
            self.creature,
            action(weight_separation=1.0),
            snapshot,
            self.world.MAX_SPEED,
            self.world.config.action.max_forward_force,
        )
        alignment = self.world._flock_steering_force(
            self.creature,
            action(weight_alignment=1.0),
            snapshot,
            self.world.MAX_SPEED,
            self.world.config.action.max_forward_force,
        )
        cohesion = self.world._flock_steering_force(
            self.creature,
            action(weight_cohesion=1.0),
            snapshot,
            self.world.MAX_SPEED,
            self.world.config.action.max_forward_force,
        )

        self.assertLess(separation[0], 0.0)
        self.assertAlmostEqual(separation[1], 0.0, places=10)
        self.assertGreater(alignment[1], 0.0)
        self.assertGreater(cohesion[0], 0.0)
        self.assertGreater(cohesion[1], 0.0)

    def test_combined_voluntary_and_flock_force_is_bounded(self) -> None:
        flock = FlockSensorSnapshot(
            center_proximity=0.0,
            center_angle=1.0,
            average_relative_heading=0.5,
            flockmate_count=1,
            nearest_neighbor_proximity=1.0,
            nearest_neighbor_angle=-1.0,
        )
        self.world._apply_action(
            self.creature,
            action(
                accelerate=1.0,
                weight_separation=1.0,
                weight_alignment=1.0,
                weight_cohesion=1.0,
            ),
            snapshot=SimpleNamespace(flock=flock),
            apply_stabilizers=False,
        )

        self.assertLessEqual(
            hypot(*self.creature.body.applied_force),
            self.world.config.action.max_forward_force + 1e-9,
        )
        self.assertNotEqual(
            self.world._motion_commands[1].effective_rotate,
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
