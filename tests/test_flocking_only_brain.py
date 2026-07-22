from __future__ import annotations

from dataclasses import astuple
from math import hypot, pi
from pathlib import Path
import unittest

import neat
import pymunk

from configs.sim_config import SimConfig
from src.action import BrainOutputIndex
from src.creature import (
    Creature,
    FlockingTraits,
    LineageInfo,
    PhysicalTraits,
    VisionTraits,
)
from src.neat_brain import NeatBrain
from src.vision import SENSOR_INPUT_NAMES, VisionSystem
from src.world import World


CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "neat_herbivore.ini"
)
FLOCK_SENSOR_NAMES = SENSOR_INPUT_NAMES[23:27]


def flocking_only_brain(genome_id: int) -> NeatBrain:
    """Build a real NEAT network whose only evidence is social flock data."""
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

    # Keep every disconnected action exactly neutral. Tanh also preserves the
    # sign of the two directional flock sensors connected to rotate.
    for node in genome.nodes.values():
        node.bias = 0.0
        node.response = 1.0
        node.activation = "tanh"
        node.aggregation = "sum"

    input_keys = config.genome_config.input_keys
    output_keys = config.genome_config.output_keys
    wiring = (
        # Center direction and average heading steer the creature.
        ("flock_center_angle", BrainOutputIndex.ROTATE, 2.0),
        ("flock_average_relative_heading", BrainOutputIndex.ROTATE, 1.0),
        # Proximity and flockmate count switch on the physical herding force.
        ("flock_center_proximity", BrainOutputIndex.HERDING, 0.5),
        ("flockmate_count", BrainOutputIndex.HERDING, 5.0),
    )
    for innovation, (sensor_name, output_index, weight) in enumerate(wiring, 1):
        input_index = SENSOR_INPUT_NAMES.index(sensor_name)
        source = input_keys[input_index]
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


def creature(
    creature_id: int,
    position: tuple[float, float],
    heading: float,
    *,
    separation_gene: float = 0.0,
) -> Creature:
    radius = 12.0
    body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0.0, radius))
    body.position = position
    body.angle = heading
    shape = pymunk.Circle(body, radius)
    return Creature(
        creature_id=creature_id,
        name=f"Flocking probe {creature_id}",
        body=body,
        shape=shape,
        energy=1.0,
        vision=VisionTraits(range=240.0, angle=pi),
        physical_traits=PhysicalTraits(radius=radius),
        color=(255, 255, 255),
        flocking_traits=FlockingTraits(
            separation_gene=separation_gene,
            alignment_gene=1.0,
            cohesion_gene=1.0,
        ),
        lineage=LineageInfo(species_id=1),
    )


class FlockingOnlyBrainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimConfig()
        self.config.action.max_forward_force = 20.0
        self.config.action.action_smoothing_alpha = 1.0
        self.config.action.active_angular_velocity_retention = 1.0
        self.config.action.turn_control_gain = 1.0
        self.config.action.turn_response = 1.0
        self.config.action.collision_avoidance_force_scale = 0.0
        self.config.action.forward_velocity_retention = 1.0
        self.config.action.lateral_velocity_retention = 1.0
        self.config.action.linear_stop_threshold = 0.0
        self.vision = VisionSystem(self.config.vision)

    def snapshot_for(
        self,
        observer: Creature,
        creatures: list[Creature],
    ):
        return self.vision.sense(
            observer,
            foods=[],
            creatures=creatures,
            world_bounds=(-500.0, -500.0, 500.0, 500.0),
            max_speed=World.MAX_SPEED,
        )

    def test_only_flock_sensors_can_reach_rotate_and_herding(self) -> None:
        brain = flocking_only_brain(1)
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
        connected_inputs = {source for source, _ in enabled_edges}
        connected_outputs = {target for _, target in enabled_edges}
        expected_inputs = {
            genome_config.input_keys[SENSOR_INPUT_NAMES.index(name)]
            for name in FLOCK_SENSOR_NAMES
        }
        expected_outputs = {
            genome_config.output_keys[BrainOutputIndex.ROTATE],
            genome_config.output_keys[BrainOutputIndex.HERDING],
        }

        self.assertEqual(connected_inputs, expected_inputs)
        self.assertEqual(connected_outputs, expected_outputs)

    def test_flock_sensors_activate_opposite_turns_and_herding(self) -> None:
        lower = creature(1, (0.0, -70.0), 0.0)
        upper = creature(2, (0.0, 70.0), 0.0)
        creatures = [lower, upper]

        lower_action = flocking_only_brain(1).decide(
            self.snapshot_for(lower, creatures)
        )
        upper_action = flocking_only_brain(2).decide(
            self.snapshot_for(upper, creatures)
        )

        self.assertGreater(lower_action.rotate, 0.0)
        self.assertLess(upper_action.rotate, 0.0)
        self.assertGreater(lower_action.herding, 0.9)
        self.assertGreater(upper_action.herding, 0.9)
        for action in (lower_action, upper_action):
            values = astuple(action)
            active_indices = {
                index for index, value in enumerate(values) if abs(value) > 1e-12
            }
            self.assertEqual(
                active_indices,
                {BrainOutputIndex.ROTATE, BrainOutputIndex.HERDING},
            )

    def test_flocking_only_brains_turn_toward_and_pull_the_pair_together(self) -> None:
        lower = creature(1, (0.0, -70.0), 0.0)
        upper = creature(2, (0.0, 70.0), 0.0)
        creatures = [lower, upper]
        brains = {
            lower.creature_id: flocking_only_brain(1),
            upper.creature_id: flocking_only_brain(2),
        }
        space = pymunk.Space()
        space.gravity = (0.0, 0.0)
        space.damping = 1.0
        for member in creatures:
            space.add(member.body, member.shape)

        world = World.__new__(World)
        world.config = self.config
        world.creatures = creatures
        world._motion_commands = {}
        initial_distance = hypot(
            upper.position[0] - lower.position[0],
            upper.position[1] - lower.position[1],
        )

        for _ in range(30):
            snapshots = {
                member.creature_id: self.snapshot_for(member, creatures)
                for member in creatures
            }
            for member in creatures:
                action = brains[member.creature_id].decide(
                    snapshots[member.creature_id]
                )
                world._apply_action(
                    member,
                    action,
                    snapshots[member.creature_id],
                    apply_stabilizers=False,
                )
            space.step(1.0 / 60.0)
            world._apply_top_down_motion()
            world._limit_creature_motion()

        final_distance = hypot(
            upper.position[0] - lower.position[0],
            upper.position[1] - lower.position[1],
        )

        self.assertGreater(lower.heading, 0.0)
        self.assertLess(upper.heading, 0.0)
        self.assertLess(final_distance, initial_distance - 2.0)

    def test_separation_keeps_flockmates_apart_without_collision_avoidance(
        self,
    ) -> None:
        radius = 12.0
        lower = creature(
            1,
            (0.0, -15.0),
            pi / 2.0,
            separation_gene=1.0,
        )
        upper = creature(
            2,
            (0.0, 15.0),
            -pi / 2.0,
            separation_gene=1.0,
        )
        creatures = [lower, upper]
        brains = {
            lower.creature_id: flocking_only_brain(1),
            upper.creature_id: flocking_only_brain(2),
        }
        space = pymunk.Space()
        space.gravity = (0.0, 0.0)
        space.damping = 0.9
        for member in creatures:
            # If the bodies were solid, Pymunk contact resolution could hide a
            # broken separation rule. Sensor shapes let only flock steering
            # maintain the gap in this controlled scenario.
            member.shape.sensor = True
            space.add(member.body, member.shape)

        world = World.__new__(World)
        world.config = self.config
        world.creatures = creatures
        world._motion_commands = {}
        distances: list[float] = []

        for _ in range(600):
            snapshots = {
                member.creature_id: self.snapshot_for(member, creatures)
                for member in creatures
            }
            for member in creatures:
                action = brains[member.creature_id].decide(
                    snapshots[member.creature_id]
                )
                world._apply_action(
                    member,
                    action,
                    snapshots[member.creature_id],
                    apply_stabilizers=False,
                )
            space.step(1.0 / 60.0)
            world._apply_top_down_motion()
            world._limit_creature_motion()
            distances.append(
                hypot(
                    upper.position[0] - lower.position[0],
                    upper.position[1] - lower.position[1],
                )
            )

        contact_distance = radius * 2.0
        personal_space_distance = radius * 4.0
        settled_distances = distances[-120:]

        self.assertGreater(min(distances), contact_distance + 4.0)
        self.assertLess(max(settled_distances), personal_space_distance + 8.0)


if __name__ == "__main__":
    unittest.main()
