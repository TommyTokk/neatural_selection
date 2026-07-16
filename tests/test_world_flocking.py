from __future__ import annotations

from math import hypot, pi
from random import Random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from configs.sim_config import SimConfig
from src.action import Action, calculate_flocking_weights
from src.creature import FlockingTraits
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
            vision=SimpleNamespace(range=1.0, angle=pi / 2),
            lineage=SimpleNamespace(species_id=1),
            body=FakeBody(),
            smoothed_rotation=0.0,
            smoothed_acceleration=0.0,
            flocking_traits=FlockingTraits(),
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
            separation_relative_heading=pi,
            separation_strength=1.0,
            average_flockmate_proximity=0.75,
        )
        snapshot = SimpleNamespace(flock=flock)

        self.creature.flocking_traits = FlockingTraits(1.0, 0.0, 0.0)
        separation = self.world._flock_steering_force(
            self.creature,
            action(herding=1.0),
            snapshot,
            self.world.MAX_SPEED,
            self.world.config.action.max_forward_force,
        )
        self.creature.flocking_traits = FlockingTraits(0.0, 1.0, 0.0)
        alignment = self.world._flock_steering_force(
            self.creature,
            action(herding=1.0),
            snapshot,
            self.world.MAX_SPEED,
            self.world.config.action.max_forward_force,
        )
        self.creature.flocking_traits = FlockingTraits(0.0, 0.0, 1.0)
        cohesion = self.world._flock_steering_force(
            self.creature,
            action(herding=1.0),
            snapshot,
            self.world.MAX_SPEED,
            self.world.config.action.max_forward_force,
        )

        self.assertLess(separation[0], 0.0)
        self.assertAlmostEqual(separation[1], 0.0, places=10)
        self.assertGreater(alignment[1], 0.0)
        self.assertGreater(cohesion[0], 0.0)
        self.assertGreater(cohesion[1], 0.0)

    def test_alignment_force_is_attenuated_by_flockmate_proximity(self) -> None:
        self.creature.flocking_traits = FlockingTraits(0.0, 1.0, 0.0)

        def alignment_force(proximity: float) -> tuple[float, float]:
            flock = FlockSensorSnapshot(
                average_relative_heading=0.5,
                flockmate_count=1,
                average_flockmate_proximity=proximity,
            )
            return self.world._flock_steering_force(
                self.creature,
                action(herding=1.0),
                SimpleNamespace(flock=flock),
                self.world.MAX_SPEED,
                self.world.config.action.max_forward_force,
            )

        far_magnitude = hypot(*alignment_force(0.01))
        near_magnitude = hypot(*alignment_force(0.90))
        unattenuated_magnitude = min(
            self.world.MAX_SPEED,
            self.world.config.action.max_forward_force,
        )

        self.assertAlmostEqual(far_magnitude, unattenuated_magnitude * 0.01)
        self.assertAlmostEqual(near_magnitude, unattenuated_magnitude * 0.90)
        self.assertAlmostEqual(near_magnitude / far_magnitude, 90.0)

    def test_combined_voluntary_and_flock_force_is_bounded(self) -> None:
        flock = FlockSensorSnapshot(
            center_proximity=0.0,
            center_angle=1.0,
            average_relative_heading=0.5,
            flockmate_count=1,
            separation_relative_heading=-pi / 2.0,
            separation_strength=1.0,
            average_flockmate_proximity=1.0,
        )
        self.creature.flocking_traits = FlockingTraits(1.0, 1.0, 1.0)
        self.world._apply_action(
            self.creature,
            action(
                accelerate=1.0,
                herding=1.0,
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

    def test_flocking_weights_follow_genes_herding_and_panic(self) -> None:
        genes = dict(
            separation_gene=0.9,
            alignment_gene=0.4,
            cohesion_gene=0.7,
        )
        self.assertEqual(
            calculate_flocking_weights(herding=0.0, panic=0.0, **genes),
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(
            calculate_flocking_weights(herding=1.0, panic=0.0, **genes),
            (0.9, 0.4, 0.7),
        )
        self.assertEqual(
            calculate_flocking_weights(herding=1.0, panic=1.0, **genes),
            (0.9, 0.0, 0.0),
        )

    def test_collision_avoidance_ignores_species_herding_and_field_of_view(
        self,
    ) -> None:
        self.creature.position = (0.0, 0.0)
        self.creature.radius = 10.0
        for neighbor_species_id in (1, 99):
            with self.subTest(neighbor_species_id=neighbor_species_id):
                neighbor = SimpleNamespace(
                    creature_id=2,
                    position=(15.0, 0.0),
                    radius=10.0,
                    lineage=SimpleNamespace(species_id=neighbor_species_id),
                )
                self.world.creatures = [self.creature, neighbor]

                force = self.world._collision_avoidance_force(
                    self.creature,
                    self.world.config.action.max_forward_force,
                )

                self.assertLess(force[0], 0.0)
                self.assertAlmostEqual(force[1], 0.0)

    def test_exact_overlap_avoidance_is_deterministic_and_opposite(self) -> None:
        first = SimpleNamespace(creature_id=1, position=(0.0, 0.0), radius=10.0)
        second = SimpleNamespace(creature_id=2, position=(0.0, 0.0), radius=10.0)
        self.world.creatures = [first, second]

        first_force = self.world._collision_avoidance_force(first, 100.0)
        second_force = self.world._collision_avoidance_force(second, 100.0)

        self.assertEqual(first_force, (100.0, 0.0))
        self.assertEqual(second_force, (-100.0, 0.0))


class FlockingTraitEvolutionTest(unittest.TestCase):
    def world(self) -> World:
        world = World.__new__(World)
        world.config = SimConfig()
        world.rng = Random(7)
        return world

    def test_traits_clamp_to_biological_bounds(self) -> None:
        self.assertEqual(FlockingTraits(-1.0, 0.4, 2.0), FlockingTraits(0.0, 0.4, 1.0))

    def test_initialization_is_seeded_and_bounded(self) -> None:
        first = self.world()
        second = self.world()

        first_traits = first._initial_flocking_traits()
        second_traits = second._initial_flocking_traits()

        self.assertEqual(first_traits, second_traits)
        for value in (
            first_traits.separation_gene,
            first_traits.alignment_gene,
            first_traits.cohesion_gene,
        ):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_no_mutation_inherits_each_gene_independently(self) -> None:
        world = self.world()
        world.config.trait.flocking_gene_mutation_rate = 0.0
        world.config.trait.flocking_gene_replace_rate = 0.0
        parent = FlockingTraits(0.9, 0.2, 0.6)

        child, delta = world._mutated_flocking_traits(parent)

        self.assertEqual(child, parent)
        self.assertEqual(
            (delta.separation_gene, delta.alignment_gene, delta.cohesion_gene),
            (0.0, 0.0, 0.0),
        )

    def test_mutating_one_gene_does_not_modify_the_other_two(self) -> None:
        world = self.world()
        world.config.trait.flocking_gene_mutation_rate = 0.5
        world.config.trait.flocking_gene_replace_rate = 0.0
        world.rng = Mock()
        world.rng.random.side_effect = [0.1, 0.9, 0.9]
        world.rng.gauss.return_value = 0.1
        parent = FlockingTraits(0.4, 0.5, 0.6)

        child, _ = world._mutated_flocking_traits(parent)

        self.assertAlmostEqual(child.separation_gene, 0.5)
        self.assertEqual(child.alignment_gene, parent.alignment_gene)
        self.assertEqual(child.cohesion_gene, parent.cohesion_gene)

    def test_replacement_is_independent_and_records_the_clamped_delta(self) -> None:
        world = self.world()
        world.config.trait.flocking_gene_mutation_rate = 0.0
        world.config.trait.flocking_gene_replace_rate = 0.005
        world.rng = Mock()
        world.rng.random.side_effect = [0.001, 0.9, 0.9]
        world.rng.uniform.return_value = 0.8
        parent = FlockingTraits(0.4, 0.5, 0.6)

        child, delta = world._mutated_flocking_traits(parent)

        self.assertEqual(child, FlockingTraits(0.8, 0.5, 0.6))
        self.assertAlmostEqual(delta.separation_gene, 0.4)
        self.assertEqual(delta.alignment_gene, 0.0)
        self.assertEqual(delta.cohesion_gene, 0.0)


if __name__ == "__main__":
    unittest.main()
