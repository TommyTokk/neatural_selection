from __future__ import annotations

from math import hypot, pi
from random import Random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import pymunk

from configs.sim_config import SimConfig
from src.action import Action
from src.collision import CREATURE_CATEGORY, FOOD_CATEGORY
from src.creature import (
    Creature,
    FlockingTraits,
    LineageInfo,
    PhysicalTraits,
    VisionTraits,
)
from src.flocking import calculate_flocking_weights
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

    def test_flock_turn_bias_uses_signed_lateral_projection(self) -> None:
        self.creature.heading = pi / 2.0
        max_force = 100.0

        forward = self.world._flock_turn_bias(
            self.creature,
            (0.0, 25.0),
            max_force,
        )
        backward = self.world._flock_turn_bias(
            self.creature,
            (0.0, -25.0),
            max_force,
        )
        left = self.world._flock_turn_bias(
            self.creature,
            (-25.0, 0.0),
            max_force,
        )
        right = self.world._flock_turn_bias(
            self.creature,
            (25.0, 0.0),
            max_force,
        )

        self.assertAlmostEqual(forward, 0.0, places=12)
        self.assertAlmostEqual(backward, 0.0, places=12)
        self.assertGreater(left, 0.0)
        self.assertLess(right, 0.0)

    def test_noisy_backward_force_cannot_create_large_turn_flip(self) -> None:
        self.creature.heading = pi / 2.0
        max_force = 100.0

        biases = (
            self.world._flock_turn_bias(
                self.creature,
                (0.0001, -25.0),
                max_force,
            ),
            self.world._flock_turn_bias(
                self.creature,
                (-0.0001, -25.0),
                max_force,
            ),
        )

        for bias in biases:
            self.assertLess(abs(bias), 1e-6)

    def test_flock_turn_bias_clamps_excessive_lateral_force(self) -> None:
        self.creature.heading = pi / 2.0

        left = self.world._flock_turn_bias(
            self.creature,
            (-1_000.0, 0.0),
            100.0,
        )
        right = self.world._flock_turn_bias(
            self.creature,
            (1_000.0, 0.0),
            100.0,
        )

        self.assertEqual(left, self.world.config.action.max_flock_turn_bias)
        self.assertEqual(right, -self.world.config.action.max_flock_turn_bias)

    def test_mandatory_avoidance_retains_lateral_turn_bias(self) -> None:
        self.creature.heading = pi / 2.0
        self.world._collision_avoidance_force = (
            lambda _creature, _max_force: (-25.0, 0.0)
        )

        self.world._apply_action(
            self.creature,
            action(),
            snapshot=SimpleNamespace(flock=FlockSensorSnapshot()),
        )

        self.assertGreater(
            self.world._motion_commands[self.creature.creature_id].effective_rotate,
            0.0,
        )

    def test_motion_limit_bounds_accumulated_body_angle(self) -> None:
        body = pymunk.Body(1.0, 1.0)
        body.angle = 10_000 * 2.0 * pi + pi / 4.0
        creature = SimpleNamespace(creature_id=99, body=body)
        self.world.creatures = [creature]

        self.world._limit_creature_motion()

        self.assertAlmostEqual(body.angle, pi / 4.0, places=10)

    def test_action_smoothing_uses_default_alpha_on_first_tick(self) -> None:
        self.world.config.action.action_smoothing_alpha = (
            SimConfig().action.action_smoothing_alpha
        )

        self.world._apply_action(
            self.creature,
            action(accelerate=1.0, rotate=1.0),
            snapshot=SimpleNamespace(flock=FlockSensorSnapshot()),
        )

        self.assertAlmostEqual(self.creature.smoothed_acceleration, 0.8)
        self.assertAlmostEqual(self.creature.smoothed_rotation, 0.8)
        self.assertAlmostEqual(self.creature.body.applied_force[0], 100.0)
        self.assertAlmostEqual(self.creature.body.applied_force[1], 0.0)
        self.assertAlmostEqual(self.world._motion_commands[1].effective_rotate, 0.8)

    def test_depleted_effective_action_still_applies_locomotion(self) -> None:
        self.creature.energy = 0.0
        raw = action(
            accelerate=1.0,
            rotate=0.5,
            flee_panic_intensity=1.0,
            emit_sound=1.0,
        )
        effective = self.world._effective_action_for(self.creature, raw)

        self.world._apply_action(
            self.creature,
            effective,
            snapshot=SimpleNamespace(flock=FlockSensorSnapshot()),
        )

        self.assertGreater(self.creature.body.applied_force[0], 0.0)
        self.assertGreater(self.world._motion_commands[1].effective_rotate, 0.0)
        self.assertEqual(effective.emit_sound, 0.0)

    def test_action_smoothing_converges_across_cached_action_ticks(self) -> None:
        self.world.config.action.action_smoothing_alpha = 0.3
        cached_action = action(accelerate=1.0, rotate=1.0)
        snapshot = SimpleNamespace(flock=FlockSensorSnapshot())

        for _ in range(3):
            self.world._apply_action(
                self.creature,
                cached_action,
                snapshot=snapshot,
            )

        self.assertAlmostEqual(self.creature.smoothed_acceleration, 0.657)
        self.assertAlmostEqual(self.creature.smoothed_rotation, 0.657)
        self.assertAlmostEqual(self.creature.body.applied_force[0], 82.125)
        self.assertLess(self.world._motion_commands[1].effective_rotate, 1.0)

    def test_motion_command_storage_is_updated_in_place(self) -> None:
        snapshot = SimpleNamespace(flock=FlockSensorSnapshot())
        self.world._apply_action(
            self.creature,
            action(rotate=0.25),
            snapshot=snapshot,
        )
        command = self.world._motion_commands[1]

        self.world._apply_action(
            self.creature,
            action(rotate=-0.5, flee_panic_intensity=1.0),
            snapshot=snapshot,
        )

        self.assertIs(self.world._motion_commands[1], command)
        self.assertAlmostEqual(command.effective_rotate, -0.5)
        self.assertAlmostEqual(command.max_speed, self.world.MAX_SPEED * 1.5)

    def test_flocking_diagnostic_wrappers_are_optional(self) -> None:
        self.world._last_flocking_runtime = {}
        self.world._last_flock_steering_debug = {}

        self.world._apply_action(
            self.creature,
            action(herding=0.5),
            snapshot=SimpleNamespace(flock=FlockSensorSnapshot()),
            capture_runtime=False,
            capture_steering_debug=False,
        )

        self.assertEqual(self.world._last_flocking_runtime, {})
        self.assertEqual(self.world._last_flock_steering_debug, {})
        self.assertIn(1, self.world._flocking_benchmark_quality_by_creature_id)

    def test_active_angular_velocity_damping_applies_after_turn_control(self) -> None:
        self.world.config.action.turn_response = 1.0
        self.world.config.action.turn_control_gain = 1.0
        self.world.config.action.active_angular_velocity_retention = 0.80

        self.world._apply_action(
            self.creature,
            action(rotate=1.0),
            snapshot=SimpleNamespace(flock=FlockSensorSnapshot()),
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
        )

        self.assertAlmostEqual(
            self.world._motion_commands[1].max_angular_speed,
            self.world.MAX_ANGULAR_SPEED * 0.65,
        )

    def test_zero_social_evidence_preserves_neural_force_and_limits(self) -> None:
        self.world._apply_action(
            self.creature,
            action(accelerate=0.5, rotate=0.25),
            snapshot=SimpleNamespace(flock=FlockSensorSnapshot()),
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
        )

        self.assertAlmostEqual(self.creature.body.applied_force[0], 187.5)
        command = self.world._motion_commands[1]
        self.assertAlmostEqual(command.max_speed, self.world.MAX_SPEED * 1.5)
        self.assertAlmostEqual(
            command.max_angular_speed,
            self.world.MAX_ANGULAR_SPEED * 1.5,
        )

    def test_force_allocator_deducts_each_component_magnitude(self) -> None:
        first, remaining = self.world._allocate_force_budget((60.0, 0.0), 100.0)
        second, remaining = self.world._allocate_force_budget((0.0, 80.0), remaining)

        self.assertEqual(first, (60.0, 0.0))
        self.assertEqual(second, (0.0, 40.0))
        self.assertEqual(remaining, 0.0)
        self.assertLessEqual(hypot(*first) + hypot(*second), 100.0)

    def test_combined_voluntary_and_flock_force_is_bounded(self) -> None:
        flock = FlockSensorSnapshot(
            center_proximity=0.0,
            center_angle=1.0,
            average_relative_heading=0.5,
            flockmate_count=1,
            crowd_separation_absolute_angle=-pi / 2.0,
            cohesion_absolute_angle=pi / 4.0,
            crowd_separation_strength=1.0,
            average_flockmate_proximity=1.0,
            average_flockmate_velocity=(0.0, 100.0),
        )
        self.creature.flocking_traits = FlockingTraits(1.0, 1.0, 1.0)
        self.world._apply_action(
            self.creature,
            action(
                accelerate=1.0,
                herding=1.0,
            ),
            snapshot=SimpleNamespace(flock=flock),
        )

        self.assertLessEqual(
            hypot(*self.creature.body.applied_force),
            self.world.config.action.max_forward_force + 1e-9,
        )
        self.assertNotEqual(
            self.world._motion_commands[1].effective_rotate,
            0.0,
        )

    def test_turn_control_is_applied_once_across_complete_physics_tick(
        self,
    ) -> None:
        self.world.config.action.turn_response = 0.5
        self.world.creatures = [self.creature]

        self.world._apply_action(
            self.creature,
            action(rotate=1.0),
            snapshot=SimpleNamespace(flock=FlockSensorSnapshot()),
        )
        angular_velocity_after_action = self.creature.body.angular_velocity

        self.world._apply_top_down_motion()

        self.assertAlmostEqual(
            angular_velocity_after_action,
            self.world.MAX_ANGULAR_SPEED * 0.5,
        )
        self.assertAlmostEqual(
            self.creature.body.angular_velocity,
            angular_velocity_after_action,
        )

    def test_action_caches_zero_flock_force_without_sensor_data(self) -> None:
        self.world._apply_action(
            self.creature,
            action(herding=1.0),
            snapshot=None,
        )

        self.assertEqual(
            self.world._last_flock_steering_debug[
                1
            ].accepted_counterfactual_delta,
            (0.0, 0.0),
        )

    def test_runtime_distinguishes_raw_and_effective_herding(self) -> None:
        self.world.neat_controller = SimpleNamespace(
            brain_for=lambda _creature_id: SimpleNamespace(
                last_raw_herding=0.9,
            )
        )

        self.world._apply_action(
            self.creature,
            action(herding=0.3),
            snapshot=None,
        )

        runtime = self.world._last_flocking_runtime[1]
        self.assertEqual(runtime.raw_neural_herding, 0.9)
        self.assertEqual(runtime.effective_herding, 0.3)

    def test_minimum_engagement_can_act_when_neural_herding_is_zero(self) -> None:
        snapshot = SimpleNamespace(
            flock=FlockSensorSnapshot(
                center_proximity=0.0,
                center_angle=0.5,
                flockmate_count=1,
            )
        )

        self.world._apply_action(
            self.creature,
            action(herding=0.0),
            snapshot=snapshot,
        )

        self.assertNotEqual(
            self.world._last_flock_steering_debug[
                1
            ].accepted_counterfactual_delta,
            (0.0, 0.0),
        )

    def test_new_sensing_epoch_clears_flock_debug_cache(self) -> None:
        world = World.__new__(World)
        world.creatures = []
        world.neat_controller = SimpleNamespace(
            reset_for_new_sensing_epoch=lambda creatures, root_species_id: None
        )
        world.fitness_archive = {}
        world._trait_archive_by_genome_id = {}
        world._last_actions = {}
        world._last_sensor_snapshots = {}
        world._last_acoustic_debug = {}
        world._last_flock_steering_debug = {1: object()}
        world._motion_commands = {}
        world.rt_neat = SimpleNamespace(
            stats=SimpleNamespace(),
            eligible_parent_ids=[],
            _lifespan_at_death_total=1.0,
            _lifespan_at_death_count=1,
        )

        world.start_new_sensing_epoch(root_species_id=1)

        self.assertEqual(world._last_flock_steering_debug, {})

    def test_flocking_weights_follow_genes_herding_and_panic(self) -> None:
        genes = dict(
            separation_gene=0.9,
            alignment_gene=0.4,
            cohesion_gene=0.7,
        )
        inactive = calculate_flocking_weights(
            herding=0.0,
            panic=0.0,
            personal_space_presence=0.0,
            social_presence=0.0,
            **genes,
        )
        active = calculate_flocking_weights(
            herding=1.0,
            panic=0.0,
            personal_space_presence=1.0,
            social_presence=1.0,
            **genes,
        )
        panicked = calculate_flocking_weights(
            herding=1.0,
            panic=1.0,
            personal_space_presence=1.0,
            social_presence=1.0,
            **genes,
        )
        self.assertEqual(
            (inactive.separation, inactive.alignment, inactive.cohesion),
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(
            (active.separation, active.alignment, active.cohesion),
            (0.9, 0.4, 0.7),
        )
        self.assertEqual(
            (panicked.separation, panicked.alignment, panicked.cohesion),
            (0.9, 0.2, 0.35),
        )

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
        world.rng.random.side_effect = [0.1, 0.9, 0.9, 0.9, 0.9]
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
        world.rng.random.side_effect = [0.001, 0.9, 0.9, 0.9, 0.9]
        world.rng.uniform.return_value = 0.8
        parent = FlockingTraits(0.4, 0.5, 0.6)

        child, delta = world._mutated_flocking_traits(parent)

        self.assertEqual(child, FlockingTraits(0.8, 0.5, 0.6))
        self.assertAlmostEqual(delta.separation_gene, 0.4)
        self.assertEqual(delta.alignment_gene, 0.0)
        self.assertEqual(delta.cohesion_gene, 0.0)


class SpatialCollisionQueryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World.__new__(World)
        self.world.config = SimConfig()
        self.world.space = pymunk.Space()
        self.world.space.use_spatial_hash(50.0, 1000)
        self.world.creatures = []
        self.world._creature_by_shape_id = {}
        self.world._creature_query_filter = pymunk.ShapeFilter(
            mask=CREATURE_CATEGORY
        )
        self.world._creature_spatial_state = None

    def add_creature(
        self,
        creature_id: int,
        position: tuple[float, float],
        radius: float,
        *,
        species_id: int = 1,
    ) -> Creature:
        body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0.0, radius))
        body.position = position
        shape = pymunk.Circle(body, radius)
        shape.filter = pymunk.ShapeFilter(
            categories=CREATURE_CATEGORY,
            mask=CREATURE_CATEGORY | FOOD_CATEGORY,
        )
        self.world.space.add(body, shape)
        creature = Creature(
            creature_id=creature_id,
            name=f"Creature {creature_id}",
            body=body,
            shape=shape,
            energy=1.0,
            vision=VisionTraits(range=1.0, angle=pi / 4.0),
            physical_traits=PhysicalTraits(radius=radius),
            color=(1, 2, 3),
            lineage=LineageInfo(species_id=species_id),
        )
        self.world.creatures.append(creature)
        self.world._index_creature_shape(shape, creature)
        return creature

    def add_food_shape(
        self,
        position: tuple[float, float],
        radius: float,
    ) -> pymunk.Circle:
        body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0.0, radius))
        body.position = position
        shape = pymunk.Circle(body, radius)
        shape.filter = pymunk.ShapeFilter(categories=FOOD_CATEGORY)
        self.world.space.add(body, shape)
        return shape

    def test_point_query_filters_self_food_outside_and_distant_shapes(self) -> None:
        observer = self.add_creature(1, (0.0, 0.0), 10.0)
        near = self.add_creature(2, (20.0, 0.0), 10.0)
        self.add_creature(3, (-28.01, 0.0), 10.0)
        self.add_creature(4, (100.0, 0.0), 10.0)
        self.add_food_shape((5.0, 0.0), 10.0)

        nearby = self.world._query_nearby_creatures(observer, 18.0)

        self.assertEqual({creature.creature_id for creature in nearby}, {2})
        self.assertNotIn(observer, nearby)
        self.assertIn(near, nearby)
        force = self.world._collision_avoidance_force(observer, 100.0)
        self.assertLess(force[0], 0.0)
        self.assertEqual(force[1], 0.0)

    def test_exact_avoidance_boundary_produces_no_force(self) -> None:
        observer = self.add_creature(1, (0.0, 0.0), 10.0)
        self.add_creature(2, (28.0, 0.0), 10.0)

        self.assertEqual(
            self.world._collision_avoidance_force(observer, 100.0),
            (0.0, 0.0),
        )

    def test_unequal_radius_query_uses_candidate_shape_geometry(self) -> None:
        observer = self.add_creature(1, (0.0, 0.0), 5.0)
        large = self.add_creature(2, (42.0, 0.0), 30.0)
        self.add_creature(3, (-43.01, 0.0), 30.0)

        nearby = self.world._query_nearby_creatures(observer, 13.0)

        self.assertEqual(nearby, [large])

    def test_avoidance_is_cross_species_out_of_fov_and_herding_independent(
        self,
    ) -> None:
        observer = self.add_creature(1, (0.0, 0.0), 10.0, species_id=1)
        self.add_creature(2, (-15.0, 0.0), 10.0, species_id=99)

        force = self.world._collision_avoidance_force(observer, 100.0)

        self.assertGreater(force[0], 0.0)
        self.assertAlmostEqual(force[1], 0.0)

    def test_exact_overlap_avoidance_is_deterministic_and_opposite(self) -> None:
        first = self.add_creature(1, (0.0, 0.0), 10.0)
        second = self.add_creature(2, (0.0, 0.0), 10.0)

        first_force = self.world._collision_avoidance_force(first, 100.0)
        second_force = self.world._collision_avoidance_force(second, 100.0)

        self.assertEqual(first_force, (100.0, 0.0))
        self.assertEqual(second_force, (-100.0, 0.0))

    def test_spatial_force_matches_full_scan_reference(self) -> None:
        observer = self.add_creature(1, (0.0, 0.0), 10.0)
        self.add_creature(2, (12.0, 4.0), 8.0)
        self.add_creature(3, (-18.0, 3.0), 14.0, species_id=2)
        self.add_creature(4, (80.0, 0.0), 20.0)

        expected = self.full_scan_reference(observer, 100.0)
        actual = self.world._collision_avoidance_force(observer, 100.0)

        self.assertAlmostEqual(actual[0], expected[0])
        self.assertAlmostEqual(actual[1], expected[1])

    def test_reindexed_manual_move_updates_query_location(self) -> None:
        observer = self.add_creature(1, (0.0, 0.0), 10.0)
        neighbor = self.add_creature(2, (100.0, 0.0), 10.0)
        self.assertEqual(self.world._query_nearby_creatures(observer, 18.0), [])

        neighbor.body.position = (15.0, 0.0)
        self.world.space.reindex_shape(neighbor.shape)

        self.assertEqual(
            self.world._query_nearby_creatures(observer, 18.0),
            [neighbor],
        )

    def test_world_bounds_clamp_reindexes_only_the_moved_creature(self) -> None:
        observer = self.add_creature(1, (0.0, 0.0), 10.0)
        neighbor = self.add_creature(2, (100.0, 0.0), 10.0)
        self.world.config.environment.world_width = 50.0
        self.world.config.environment.world_height = 200.0

        self.world._keep_creatures_inside_bounds()

        self.assertEqual(tuple(neighbor.body.position), (13.0, 0.0))
        self.assertEqual(
            self.world._query_nearby_creatures(observer, 18.0),
            [neighbor],
        )

    def test_shape_lookup_registration_and_removal_are_synchronized(self) -> None:
        creature = self.add_creature(1, (0.0, 0.0), 10.0)
        self.assertIs(
            self.world._creature_by_shape_id[id(creature.shape)],
            creature,
        )

        self.world._unindex_creature_shape(creature)

        self.assertNotIn(id(creature.shape), self.world._creature_by_shape_id)

    def test_behavior_cache_is_invalidated_after_the_intent_pass(self) -> None:
        creature = self.add_creature(1, (3.0, 4.0), 10.0)
        observed: list[tuple[float, float, float]] = []
        self.world._apply_creature_intents_with_spatial_cache = lambda: (
            observed.append(
                self.world._creature_spatial_state[creature.creature_id]
            )
        )

        self.world._apply_creature_intents()

        self.assertEqual(observed, [(3.0, 4.0, 10.0)])
        self.assertIsNone(self.world._creature_spatial_state)

    def full_scan_reference(
        self,
        creature: Creature,
        max_force: float,
    ) -> tuple[float, float]:
        margin = self.world.config.action.collision_avoidance_margin
        center_x, center_y = creature.body.position
        avoidance_x = avoidance_y = 0.0
        for neighbor in self.world.creatures:
            if neighbor is creature:
                continue
            neighbor_x, neighbor_y = neighbor.body.position
            away_x = center_x - neighbor_x
            away_y = center_y - neighbor_y
            distance = hypot(away_x, away_y)
            safe_distance = creature.shape.radius + neighbor.shape.radius + margin
            if distance >= safe_distance:
                continue
            if distance <= 1e-12:
                unit_x = 1.0 if creature.creature_id < neighbor.creature_id else -1.0
                unit_y = 0.0
            else:
                unit_x, unit_y = away_x / distance, away_y / distance
            strength = (safe_distance - distance) / safe_distance
            avoidance_x += unit_x * strength
            avoidance_y += unit_y * strength
        magnitude = hypot(avoidance_x, avoidance_y)
        if magnitude <= 1e-12:
            return 0.0, 0.0
        force_magnitude = min(max_force, max_force * min(1.0, magnitude))
        return (
            avoidance_x / magnitude * force_magnitude,
            avoidance_y / magnitude * force_magnitude,
        )


if __name__ == "__main__":
    unittest.main()
