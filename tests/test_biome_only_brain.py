from __future__ import annotations

from dataclasses import astuple
from pathlib import Path
import unittest

import neat
import numpy as np
import pymunk

from configs.sim_config import SimConfig
from src.action import BrainOutputIndex
from src.biome import Biome, BiomeMap
from src.creature import Creature, PhysicalTraits, VisionTraits
from src.neat_brain import NeatBrain
from src.vision import SENSOR_INPUT_NAMES, VisionSystem
from src.world import World


CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "neat_herbivore.ini"
)
WORLD_BOUNDS = (-200.0, -200.0, 200.0, 200.0)
BIOME_SENSOR_NAMES = SENSOR_INPUT_NAMES[17:20]
SPAWN_WEIGHTS = {
    Biome.PRAIRIE: 0.25,
    Biome.BUSHES: 1.25,
    Biome.FOREST: 2.75,
}


def biome_only_brain(genome_id: int) -> NeatBrain:
    """Build a NEAT brain driven only by expected food-density sensing."""
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(CONFIG_PATH),
    )
    genome = neat.DefaultGenome(genome_id)
    for output_key in config.genome_config.output_keys:
        genome.nodes[output_key] = genome.create_node(
            config.genome_config,
            output_key,
        )
    for node in genome.nodes.values():
        node.bias = 0.0
        node.response = 1.0
        node.activation = "tanh"
        node.aggregation = "sum"

    # Lateral contrast steers toward the richer probe. Local richness and the
    # forward contrast drive acceleration, with a sufficiently poor region
    # ahead producing reverse acceleration.
    wiring = (
        ("local_richness", BrainOutputIndex.ACCELERATE, 1.0),
        ("forward_gradient", BrainOutputIndex.ACCELERATE, 2.0),
        ("lateral_gradient", BrainOutputIndex.ROTATE, 3.0),
    )
    input_keys = config.genome_config.input_keys
    output_keys = config.genome_config.output_keys
    for innovation, (sensor_name, output_index, weight) in enumerate(wiring, 1):
        source = input_keys[SENSOR_INPUT_NAMES.index(sensor_name)]
        target = output_keys[int(output_index)]
        connection = genome.create_connection(
            config.genome_config,
            source,
            target,
            innovation,
        )
        connection.weight = weight
        connection.enabled = True
        genome.connections[connection.key] = connection

    return NeatBrain.from_genome(genome_id, genome, config)


def split_biome_map(
    *,
    axis: str,
    rich_positive: bool,
) -> BiomeMap:
    size = 101
    biome_ids = np.full((size, size), int(Biome.PRAIRIE), dtype=np.uint8)
    midpoint = size // 2
    if axis == "y":
        rich_slice = slice(midpoint, None) if rich_positive else slice(None, midpoint)
        biome_ids[rich_slice, :] = int(Biome.FOREST)
    elif axis == "x":
        rich_slice = slice(midpoint, None) if rich_positive else slice(None, midpoint)
        biome_ids[:, rich_slice] = int(Biome.FOREST)
    else:
        raise ValueError(f"Unsupported split axis: {axis!r}")

    return BiomeMap(
        biome_ids=biome_ids,
        render_rgba=np.zeros((size, size, 4), dtype=np.uint8),
        world_bounds=WORLD_BOUNDS,
        area_shares={
            Biome.PRAIRIE: 0.5,
            Biome.BUSHES: 0.0,
            Biome.FOREST: 0.5,
        },
        spawn_weights=SPAWN_WEIGHTS,
        uniform_spawn_chance=0.0,
        max_spawn_attempts=32,
    )


def probe_creature(
    *,
    position: tuple[float, float],
    heading: float = 0.0,
) -> Creature:
    radius = 12.0
    body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0.0, radius))
    body.position = position
    body.angle = heading
    shape = pymunk.Circle(body, radius)
    return Creature(
        creature_id=1,
        name="Biome probe",
        body=body,
        shape=shape,
        energy=1.0,
        vision=VisionTraits(range=120.0, angle=1.0),
        physical_traits=PhysicalTraits(radius=radius),
        color=(255, 255, 255),
    )


class BiomeOnlyBrainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimConfig()
        self.config.action.max_forward_force = 30.0
        self.config.action.action_smoothing_alpha = 1.0
        self.config.action.active_angular_velocity_retention = 1.0
        self.config.action.turn_control_gain = 0.25
        self.config.action.turn_response = 1.0
        self.config.action.collision_avoidance_force_scale = 0.0
        self.config.action.forward_velocity_retention = 1.0
        self.config.action.lateral_velocity_retention = 1.0
        self.config.action.linear_stop_threshold = 0.0
        self.vision = VisionSystem(self.config.vision)

        self.world = World.__new__(World)
        self.world.config = self.config
        self.world.creatures = []
        self.world._motion_commands = {}

    def snapshot_for(
        self,
        creature: Creature,
        biome_map: BiomeMap,
    ):
        self.world.biome_map = biome_map
        snapshot = self.vision.sense(
            creature,
            foods=[],
            creatures=[creature],
            world_bounds=WORLD_BOUNDS,
            max_speed=World.MAX_SPEED,
        )
        snapshot.biome = self.world._biome_sensor_snapshot_for(creature)
        return snapshot

    def simulate(
        self,
        creature: Creature,
        biome_map: BiomeMap,
        brain: NeatBrain,
        steps: int,
    ) -> None:
        space = pymunk.Space()
        space.gravity = (0.0, 0.0)
        space.damping = 0.9
        space.add(creature.body, creature.shape)
        self.world.creatures = [creature]

        for _ in range(steps):
            snapshot = self.snapshot_for(creature, biome_map)
            action = brain.decide(snapshot)
            self.world._apply_action(
                creature,
                action,
                snapshot,
            )
            space.step(1.0 / 60.0)
            self.world._apply_top_down_motion()
            self.world._limit_creature_motion()

    def test_only_biome_sensors_reach_accelerate_and_rotate(self) -> None:
        brain = biome_only_brain(1)
        genome_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(CONFIG_PATH),
        ).genome_config
        enabled_edges = {
            connection.key
            for connection in brain.genome.connections.values()
            if connection.enabled
        }

        self.assertEqual(
            {source for source, _ in enabled_edges},
            {
                genome_config.input_keys[SENSOR_INPUT_NAMES.index(name)]
                for name in BIOME_SENSOR_NAMES
            },
        )
        self.assertEqual(
            {target for _, target in enabled_edges},
            {
                genome_config.output_keys[BrainOutputIndex.ACCELERATE],
                genome_config.output_keys[BrainOutputIndex.ROTATE],
            },
        )

    def test_brain_turns_toward_the_richer_food_side(self) -> None:
        brain = biome_only_brain(1)
        creature = probe_creature(position=(0.0, -20.0))
        rich_left_snapshot = self.snapshot_for(
            creature,
            split_biome_map(axis="y", rich_positive=True),
        )
        rich_left_action = brain.decide(rich_left_snapshot)
        right_probe = probe_creature(position=(0.0, 20.0))
        rich_right_snapshot = self.snapshot_for(
            right_probe,
            split_biome_map(axis="y", rich_positive=False),
        )
        rich_right_action = brain.decide(rich_right_snapshot)

        self.assertGreater(rich_left_snapshot.biome.lateral_gradient, 0.0)
        self.assertGreater(rich_left_snapshot.biome.forward_gradient, 0.0)
        self.assertGreater(rich_left_action.accelerate, 0.0)
        self.assertGreater(rich_left_action.rotate, 0.0)
        self.assertLess(rich_right_snapshot.biome.lateral_gradient, 0.0)
        self.assertGreater(rich_right_snapshot.biome.forward_gradient, 0.0)
        self.assertGreater(rich_right_action.accelerate, 0.0)
        self.assertLess(rich_right_action.rotate, 0.0)

        for action in (rich_left_action, rich_right_action):
            active_indices = {
                index
                for index, value in enumerate(astuple(action))
                if abs(value) > 1e-12
            }
            self.assertEqual(
                active_indices,
                {BrainOutputIndex.ACCELERATE, BrainOutputIndex.ROTATE},
            )

    def test_creature_moves_into_the_more_food_rich_biome(self) -> None:
        biome_map = split_biome_map(axis="y", rich_positive=True)
        creature = probe_creature(position=(0.0, -20.0))
        initial_richness = biome_map.expected_food_density_at(*creature.position)

        self.simulate(creature, biome_map, biome_only_brain(1), steps=120)

        final_richness = biome_map.expected_food_density_at(*creature.position)
        self.assertGreater(creature.position[1], -20.0)
        self.assertGreater(final_richness, initial_richness)

    def test_creature_reverses_away_from_poorer_terrain_ahead(self) -> None:
        # Rich food is west (negative x); the probe faces east from the rich
        # side while both forward probes reach into depleted prairie.
        biome_map = split_biome_map(axis="x", rich_positive=False)
        creature = probe_creature(
            position=(-20.0, 0.0),
            heading=0.0,
        )
        initial_snapshot = self.snapshot_for(creature, biome_map)
        initial_action = biome_only_brain(1).decide(initial_snapshot)

        self.assertLess(initial_snapshot.biome.forward_gradient, 0.0)
        self.assertLess(initial_action.accelerate, 0.0)

        self.simulate(creature, biome_map, biome_only_brain(2), steps=90)

        self.assertLess(creature.position[0], -20.0)


if __name__ == "__main__":
    unittest.main()
