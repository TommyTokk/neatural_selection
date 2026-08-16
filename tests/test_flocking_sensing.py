from __future__ import annotations

from pathlib import Path
from math import cos, pi, sin
from random import Random
from types import SimpleNamespace
import unittest

from configs.sim_config import (
    SocialCompatibilityMode,
    build_sim_config,
)
from src.creature import (
    FlockingTraits,
    LineageInfo,
    PhysicalTraits,
    TraitMutationDelta,
    VisionTraits,
)
from src.flocking import SocialCompatibilityResolver
from src.neat_controller import calculate_flocking_trait_distance
from src.persistence import (
    CHECKPOINT_VERSION,
    CheckpointContractError,
    PersistenceManager,
    SimulationPaths,
)
from src.vision import (
    SENSOR_CONTRACT,
    SENSOR_INPUT_NAMES,
    VisionSystem,
)
from src.world import World
from tests.test_vision import creature_at


class SchemaSevenSensingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = build_sim_config()
        self.config.flocking.long_range.enabled = True
        self.vision = VisionSystem(
            self.config.vision,
            flocking_config=self.config.flocking,
        )

    def test_contract_has_exact_count(self) -> None:
        self.assertEqual(SENSOR_CONTRACT.input_count, 43)
        self.assertEqual(SENSOR_CONTRACT.schema_version, 7)

    def test_write_inputs_matches_allocating_compatibility_api(self) -> None:
        observer = creature_at((0.0, 0.0), vision_range=100.0)
        neighbor = creature_at((40.0, 0.0), creature_id=2)
        snapshot = self.vision.sense(
            observer,
            [],
            [observer, neighbor],
            (-200.0, -200.0, 200.0, 200.0),
            100.0,
        )
        output = [-99.0] * SENSOR_CONTRACT.input_count
        identity = id(output)

        snapshot.write_inputs(output)

        self.assertEqual(id(output), identity)
        self.assertEqual(output, snapshot.as_inputs())
        with self.assertRaisesRegex(ValueError, "length"):
            snapshot.write_inputs(output[:-1])

    def test_presence_distinguishes_centered_flock_from_absence(self) -> None:
        observer = creature_at(
            (0.0, 0.0),
            radius=10.0,
            vision_range=100.0,
        )
        centered = creature_at(
            (40.0, 0.0),
            creature_id=2,
            species_id=1,
        )
        snapshot = self.vision.sense(
            observer,
            [],
            [observer, centered],
            (-200.0, -200.0, 200.0, 200.0),
            100.0,
        )
        inputs = snapshot.as_inputs()
        self.assertEqual(len(inputs), 43)
        self.assertGreater(inputs[22], 0.0)
        self.assertAlmostEqual(inputs[23], 0.25)
        self.assertAlmostEqual(inputs[25], 0.0)

    def test_center_and_relative_velocity_are_in_the_observer_body_frame(
        self,
    ) -> None:
        observer = creature_at(
            (0.0, 0.0),
            heading=pi / 2.0,
            velocity=(0.0, 10.0),
            radius=10.0,
            vision_range=100.0,
        )
        stationary_ahead = creature_at(
            (0.0, 40.0),
            creature_id=2,
            species_id=1,
            velocity=(0.0, 0.0),
        )
        snapshot = self.vision.sense(
            observer,
            [],
            [observer, stationary_ahead],
            (-200.0, -200.0, 200.0, 200.0),
            100.0,
        )
        inputs = snapshot.as_inputs()
        self.assertGreater(inputs[24], 0.0)
        self.assertAlmostEqual(inputs[25], 0.0, places=12)
        self.assertAlmostEqual(inputs[26], -0.05, places=12)
        self.assertAlmostEqual(inputs[27], 0.0, places=12)

    def test_incompatible_neighbor_does_not_enter_personal_space(self) -> None:
        observer = creature_at(
            (0.0, 0.0),
            radius=10.0,
            vision_range=100.0,
        )
        incompatible = creature_at(
            (50.0, 0.0),
            creature_id=2,
            species_id=2,
        )
        snapshot = self.vision.sense(
            observer,
            [],
            [observer, incompatible],
            (-200.0, -200.0, 200.0, 200.0),
            100.0,
        )
        self.assertEqual(snapshot.flock.social_presence, 0.0)
        self.assertEqual(snapshot.flock.visible_personal_space_count, 0)
        self.assertEqual(snapshot.flock.crowd_separation_strength, 0.0)

    def test_boid_perception_is_omnidirectional(self) -> None:
        observer = creature_at(
            (0.0, 0.0),
            radius=10.0,
            vision_range=100.0,
            vision_angle=0.4,
        )
        behind = creature_at(
            (-100.0, 0.0),
            creature_id=2,
            species_id=1,
        )
        snapshot = self.vision.sense(
            observer,
            [],
            [observer, behind],
            (-200.0, -200.0, 200.0, 200.0),
            100.0,
        )
        self.assertEqual(snapshot.visible_creature_count, 0)
        self.assertEqual(snapshot.flock.flockmate_count, 1.0)
        self.assertLess(snapshot.flock.center_forward, 0.0)

    def test_boid_radius_uses_squared_center_distance(self) -> None:
        self.config.flocking.long_range.enabled = False
        compatibility_calls: list[int] = []

        def compatibility(_observer, neighbor):
            compatibility_calls.append(neighbor.creature_id)
            return 1.0

        self.vision.flock_compatibility_resolver = compatibility
        observer = creature_at((0.0, 0.0), creature_id=1)
        at_boundary = creature_at((150.0, 0.0), creature_id=2)
        outside = creature_at((150.001, 0.0), creature_id=3)

        snapshot = self.vision.sense(
            observer,
            [],
            [observer, at_boundary, outside],
            (-200.0, -200.0, 200.0, 200.0),
            100.0,
        )

        self.assertEqual(compatibility_calls, [2])
        self.assertEqual(snapshot.flock.flockmate_count, 1.0)
        self.assertEqual(snapshot.flock.visible_creature_count, 1)
        self.assertEqual(snapshot.flock.average_flockmate_proximity, 0.0)

    def test_boid_aggregation_is_single_pass_with_strict_compatibility_gate(
        self,
    ) -> None:
        self.config.flocking.long_range.enabled = False
        observer = creature_at((0.0, 0.0), creature_id=1)
        accepted = creature_at((10.0, 0.0), creature_id=2)
        incompatible = creature_at((20.0, 0.0), creature_id=3)
        negative = creature_at((30.0, 0.0), creature_id=4)
        outside = creature_at((151.0, 0.0), creature_id=5)
        compatibility_calls: list[int] = []

        def compatibility(_observer, neighbor):
            compatibility_calls.append(neighbor.creature_id)
            return {2: 0.5, 3: 0.0, 4: -0.5}[neighbor.creature_id]

        self.vision.flock_compatibility_resolver = compatibility
        distance_calls: list[float] = []
        linear_distance = self.vision._linear_distance

        def counted_distance(distance_squared: float) -> float:
            distance_calls.append(distance_squared)
            return linear_distance(distance_squared)

        self.vision._linear_distance = counted_distance

        class SinglePassNeighbors:
            def __init__(self, values) -> None:
                self.values = values
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                if self.iterations > 1:
                    raise AssertionError("Boid candidates were rescanned.")
                return iter(self.values)

        neighbors = SinglePassNeighbors(
            [observer, accepted, incompatible, negative, outside]
        )
        snapshot = self.vision._flock_snapshot(
            observer,
            [],
            neighbors,
            100.0,
        )

        self.assertEqual(neighbors.iterations, 1)
        self.assertEqual(compatibility_calls, [2, 3, 4])
        self.assertEqual(distance_calls, [100.0])
        self.assertEqual(snapshot.flockmate_count, 0.5)
        self.assertEqual(snapshot.visible_creature_count, 1)
        self.assertEqual(snapshot.compatible_visible_count, 1)
        self.assertEqual(snapshot.visible_personal_space_count, 1)
        self.assertAlmostEqual(
            snapshot.crowd_separation_strength,
            0.5 * (1.0 - 10.0 / 60.0),
        )

    def test_any_positive_compatibility_enters_boid_accumulators(self) -> None:
        self.config.flocking.long_range.enabled = False
        self.vision.flock_compatibility_resolver = (
            lambda _observer, _neighbor: 1e-15
        )
        observer = creature_at((0.0, 0.0), creature_id=1)
        neighbor = creature_at((20.0, 0.0), creature_id=2)

        snapshot = self.vision.sense(
            observer,
            [],
            [observer, neighbor],
            (-200.0, -200.0, 200.0, 200.0),
            100.0,
        )

        self.assertEqual(snapshot.flock.flockmate_count, 1e-15)
        self.assertEqual(snapshot.flock.visible_creature_count, 1)
        self.assertGreater(snapshot.flock.crowd_separation_strength, 0.0)

    def test_left_and_right_centres_have_opposite_local_signs(self) -> None:
        observer = creature_at(
            (0.0, 0.0),
            radius=10.0,
            vision_range=100.0,
            vision_angle=2.0 * pi,
        )

        def right_component(y: float) -> float:
            neighbor = creature_at(
                (40.0, y),
                creature_id=2,
                species_id=1,
            )
            return self.vision.sense(
                observer,
                [],
                [observer, neighbor],
                (-200.0, -200.0, 200.0, 200.0),
                100.0,
            ).as_inputs()[26]

        self.assertLess(right_component(20.0), 0.0)
        self.assertGreater(right_component(-20.0), 0.0)

    def test_target_scaled_count_is_bounded(self) -> None:
        observer = creature_at(
            (0.0, 0.0),
            radius=10.0,
            vision_range=100.0,
            vision_angle=2.0 * pi,
        )
        neighbors = [
            creature_at(
                (
                    50.0 * cos(index * 2.0 * pi / 10.0),
                    50.0 * sin(index * 2.0 * pi / 10.0),
                ),
                creature_id=index + 2,
                species_id=1,
            )
            for index in range(10)
        ]
        value = self.vision.sense(
            observer,
            [],
            [observer, *neighbors],
            (-200.0, -200.0, 200.0, 200.0),
            100.0,
        ).as_inputs()[24]
        self.assertEqual(value, 1.0)

    def test_long_range_observation_does_not_require_fov(self) -> None:
        observer = creature_at(
            (0.0, 0.0),
            radius=10.0,
            vision_range=100.0,
            vision_angle=0.4,
        )
        behind = creature_at(
            (-200.0, 0.0),
            creature_id=2,
            species_id=1,
        )
        snapshot = self.vision.sense(
            observer,
            [],
            [observer, behind],
            (-500.0, -500.0, 500.0, 500.0),
            100.0,
        )
        self.assertEqual(snapshot.visible_creature_count, 0)
        self.assertGreater(snapshot.flock.long_range.intensity, 0.0)
        self.assertLess(snapshot.flock.long_range.direction_forward, 0.0)

    def test_world_uses_one_expanded_creature_query(self) -> None:
        self.config.persistence.enable_telemetry = False
        self.config.population.initial_creatures = 2
        self.config.food.initial_food_items = 0
        paths = SimulationPaths(Path(".").resolve())
        world = World(self.config, simulation_paths=paths)
        try:
            brain = world.neat_controller.brain_for(
                world.creatures[0].creature_id
            )
            self.assertEqual(
                brain.last_input_names,
                SENSOR_INPUT_NAMES,
            )
            calls: list[float] = []
            original = world._nearby_creatures_for

            def counted(observer, query_range):
                calls.append(query_range)
                return original(observer, query_range)

            world._nearby_creatures_for = counted
            observer = world.creatures[0]
            world._sensor_snapshot_for(observer)
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                calls[0],
                max(
                    observer.vision.range,
                    self.config.flocking.perception_radius,
                    self.config.flocking.long_range.range,
                )
                + self.config.trait.max_radius,
            )

            calls.clear()
            self.config.flocking.long_range.enabled = False
            observer.vision.range = 100.0
            world._sensor_snapshot_for(observer)
            self.assertEqual(
                calls,
                [
                    self.config.flocking.perception_radius
                    + self.config.trait.max_radius
                ],
            )
        finally:
            world.close()


class SocialTagCompatibilityTest(unittest.TestCase):
    def _creature(
        self,
        creature_id: int,
        tag: tuple[float, float],
        species: int = 1,
    ):
        return SimpleNamespace(
            creature_id=creature_id,
            flocking_traits=FlockingTraits(
                social_tag_x=tag[0],
                social_tag_y=tag[1],
            ),
            lineage=SimpleNamespace(species_id=species),
        )

    def test_social_tag_is_symmetric_and_monotonic(self) -> None:
        config = build_sim_config().flocking.compatibility
        config.mode = SocialCompatibilityMode.SOCIAL_TAG
        resolver = SocialCompatibilityResolver(config, lambda _a, _b: 0.0)
        first = self._creature(1, (0.0, 0.0))
        near = self._creature(2, (0.1, 0.0))
        far = self._creature(3, (0.8, 0.0))
        self.assertAlmostEqual(
            resolver.compatibility(first, near),
            resolver.compatibility(near, first),
        )
        self.assertGreater(
            resolver.compatibility(first, near),
            resolver.compatibility(first, far),
        )

    def test_species_mode_is_independent_of_tags(self) -> None:
        config = build_sim_config().flocking.compatibility
        config.mode = SocialCompatibilityMode.SPECIES
        resolver = SocialCompatibilityResolver(config, lambda _a, _b: 0.0)
        self.assertEqual(
            resolver.compatibility(
                self._creature(1, (0.0, 0.0), 4),
                self._creature(2, (1.0, 1.0), 4),
            ),
            1.0,
        )

    def test_social_tag_cache_is_invalidated_when_creature_dies(self) -> None:
        config = build_sim_config().flocking.compatibility
        config.mode = SocialCompatibilityMode.SOCIAL_TAG
        resolver = SocialCompatibilityResolver(config, lambda _a, _b: 0.0)
        first = self._creature(1, (0.0, 0.0))
        second = self._creature(2, (0.0, 0.0))
        self.assertEqual(resolver.compatibility(first, second), 1.0)
        second.flocking_traits = FlockingTraits(
            social_tag_x=1.0,
            social_tag_y=1.0,
        )
        self.assertEqual(resolver.compatibility(first, second), 1.0)
        resolver.discard_creature(second.creature_id)
        self.assertLess(resolver.compatibility(first, second), 0.01)

    def test_social_tags_ignore_adaptive_threshold_and_controller_genome(self) -> None:
        config = build_sim_config().flocking.compatibility
        config.mode = SocialCompatibilityMode.SOCIAL_TAG
        threshold = {"value": 1.0}
        resolver = SocialCompatibilityResolver(
            config,
            lambda _a, _b: threshold["value"],
        )
        first = self._creature(1, (0.2, 0.3))
        second = self._creature(2, (0.5, 0.7))
        first.genome = object()
        second.genome = object()
        before = resolver.compatibility(first, second)
        threshold["value"] = 0.0
        first.genome = SimpleNamespace(mutated=True)
        resolver.discard_creature(first.creature_id)
        after = resolver.compatibility(first, second)
        self.assertEqual(before, after)

    def test_social_tags_do_not_enter_reproductive_trait_distance(self) -> None:
        first = FlockingTraits(
            separation_gene=0.1,
            alignment_gene=0.2,
            cohesion_gene=0.3,
            social_tag_x=0.0,
            social_tag_y=0.0,
        )
        second = FlockingTraits(
            separation_gene=0.1,
            alignment_gene=0.2,
            cohesion_gene=0.3,
            social_tag_x=1.0,
            social_tag_y=1.0,
        )
        self.assertEqual(
            calculate_flocking_trait_distance(first, second),
            (0.0, 0.0, 0.0, 0.0),
        )

    def test_social_tags_inherit_exactly_when_mutation_is_disabled(self) -> None:
        config = build_sim_config()
        config.flocking.compatibility.mode = (
            SocialCompatibilityMode.SOCIAL_TAG
        )
        config.trait.flocking_gene_replace_rate = 0.0
        config.trait.flocking_gene_mutation_rate = 0.0
        config.trait.social_tag_replace_rate = 0.0
        config.trait.social_tag_mutation_rate = 0.0
        world = object.__new__(World)
        world.config = config
        world.rng = Random(4)
        parent = FlockingTraits(
            separation_gene=0.1,
            alignment_gene=0.2,
            cohesion_gene=0.3,
            social_tag_x=0.4,
            social_tag_y=0.6,
        )
        child, delta = world._mutated_flocking_traits(parent)
        self.assertEqual(child, parent)
        self.assertEqual(delta.social_tag_x, 0.0)
        self.assertEqual(delta.social_tag_y, 0.0)

    def test_trait_archive_preserves_social_tags_and_deltas(self) -> None:
        world = object.__new__(World)
        world.neat_controller = SimpleNamespace(
            genome_id_for=lambda _creature_id: 99
        )
        creature = SimpleNamespace(
            creature_id=7,
            vision=VisionTraits(range=120.0, angle=1.5),
            physical_traits=PhysicalTraits(
                radius=15.0,
                movement_cost_multiplier=1.0,
            ),
            flocking_traits=FlockingTraits(
                social_tag_x=0.2,
                social_tag_y=0.8,
            ),
            color=(10, 20, 30),
            lineage=LineageInfo(
                mutation_delta=TraitMutationDelta(
                    social_tag_x=-0.1,
                    social_tag_y=0.15,
                )
            ),
        )
        world._archive_creature_traits(creature)
        archived = world._trait_archive_by_genome_id[99]
        self.assertEqual(archived.flocking_traits.social_tag_x, 0.2)
        self.assertEqual(archived.flocking_traits.social_tag_y, 0.8)
        self.assertEqual(archived.lineage.mutation_delta.social_tag_x, -0.1)
        self.assertEqual(archived.lineage.mutation_delta.social_tag_y, 0.15)


class CheckpointContractPolicyTest(unittest.TestCase):
    def test_cross_contract_fails_without_explicit_opt_in(self) -> None:
        legacy = build_sim_config()
        legacy.persistence.enable_telemetry = False
        legacy.population.initial_creatures = 2
        legacy.food.initial_food_items = 1
        paths = SimulationPaths(Path(".").resolve())
        world = World(legacy, simulation_paths=paths)
        restored = None
        try:
            state = PersistenceManager._capture_state(
                world,
                world.neat_controller,
            )
            state["brain_contract"]["sensor_schema"] = 6
            state["brain_contract"]["inputs"] = 44
            current = build_sim_config()
            current.persistence.enable_telemetry = False
            current.population.initial_creatures = 2
            current.food.initial_food_items = 1
            with self.assertRaises(CheckpointContractError):
                PersistenceManager._restore_world(state, current, paths)
            restored = PersistenceManager._restore_world(
                state,
                current,
                paths,
                allow_brain_contract_reset=True,
            )
            self.assertEqual(
                len(restored.neat_controller.config.genome_config.input_keys),
                43,
            )
            self.assertTrue(restored.brain_contract_reset_occurred)
            for creature in restored.creatures:
                brain = restored.neat_controller.brain_for(
                    creature.creature_id
                )
                self.assertEqual(brain.herding_state, 0.0)
                self.assertEqual(brain.last_raw_herding, 0.0)
        finally:
            world.close()
            if restored is not None:
                restored.close()

    def test_schema_seven_round_trip_preserves_contract_not_transient_vectors(
        self,
    ) -> None:
        config = build_sim_config()
        config.flocking.compatibility.mode = (
            SocialCompatibilityMode.SOCIAL_TAG
        )
        config.persistence.enable_telemetry = False
        config.population.initial_creatures = 2
        config.food.initial_food_items = 0
        paths = SimulationPaths(Path(".").resolve())
        world = World(config, simulation_paths=paths)
        restored = None
        try:
            world.update(World.FIXED_TIMESTEP)
            state = PersistenceManager._capture_state(
                world,
                world.neat_controller,
            )
            self.assertEqual(state["version"], CHECKPOINT_VERSION)
            self.assertEqual(state["brain_contract"]["sensor_schema"], 7)
            self.assertEqual(state["brain_contract"]["inputs"], 43)
            serialized_keys: set[str] = set()

            def collect_keys(value) -> None:
                if isinstance(value, dict):
                    serialized_keys.update(str(key) for key in value)
                    for child in value.values():
                        collect_keys(child)
                elif isinstance(value, (list, tuple)):
                    for child in value:
                        collect_keys(child)

            collect_keys(state)
            self.assertNotIn("_last_flocking_runtime", serialized_keys)
            self.assertNotIn("neural_desired_velocity", serialized_keys)
            self.assertNotIn(
                "accepted_counterfactual_delta",
                serialized_keys,
            )
            self.assertNotIn("herding_state", serialized_keys)
            self.assertNotIn("last_raw_herding", serialized_keys)

            tags = [
                (
                    creature.flocking_traits.social_tag_x,
                    creature.flocking_traits.social_tag_y,
                )
                for creature in world.creatures
            ]
            restored = PersistenceManager._restore_world(
                state,
                config,
                paths,
            )
            self.assertFalse(restored.brain_contract_reset_occurred)
            self.assertEqual(restored._last_flocking_runtime, {})
            for creature in restored.creatures:
                brain = restored.neat_controller.brain_for(
                    creature.creature_id
                )
                self.assertEqual(brain.herding_state, 0.0)
                self.assertEqual(brain.last_raw_herding, 0.0)
            self.assertEqual(
                [
                    (
                        creature.flocking_traits.social_tag_x,
                        creature.flocking_traits.social_tag_y,
                    )
                    for creature in restored.creatures
                ],
                tags,
            )
        finally:
            world.close()
            if restored is not None:
                restored.close()

    def test_older_schema_three_checkpoint_receives_neutral_social_tags(
        self,
    ) -> None:
        config = build_sim_config()
        config.persistence.enable_telemetry = False
        config.population.initial_creatures = 1
        config.food.initial_food_items = 0
        paths = SimulationPaths(Path(".").resolve())
        world = World(config, simulation_paths=paths)
        restored = None
        try:
            state = PersistenceManager._capture_state(
                world,
                world.neat_controller,
            )
            state["version"] = 12
            traits = state["creatures"][0]["flocking_traits"]
            state["creatures"][0]["flocking_traits"] = SimpleNamespace(
                separation_gene=traits.separation_gene,
                alignment_gene=traits.alignment_gene,
                cohesion_gene=traits.cohesion_gene,
            )
            restored = PersistenceManager._restore_world(
                state,
                config,
                paths,
            )
            restored_traits = restored.creatures[0].flocking_traits
            self.assertEqual(
                restored_traits.social_tag_x,
                config.trait.default_social_tag_x,
            )
            self.assertEqual(
                restored_traits.social_tag_y,
                config.trait.default_social_tag_y,
            )
        finally:
            world.close()
            if restored is not None:
                restored.close()


if __name__ == "__main__":
    unittest.main()
