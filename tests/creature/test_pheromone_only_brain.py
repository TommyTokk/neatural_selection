from __future__ import annotations

from dataclasses import astuple
from pathlib import Path
import unittest

import neat
import pymunk

from configs.sim_config import SimConfig
from src.action import BrainOutputIndex
from src.communication import PheromoneSystem
from src.creature import Creature, PhysicalTraits, VisionTraits
from src.neat_brain import NeatBrain
from src.vision import SENSOR_INPUT_NAMES, VisionSystem
from src.world import World


CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "neat_herbivore.ini"
)
WORLD_BOUNDS = (-300.0, -300.0, 300.0, 300.0)
PHEROMONE_SENSOR_NAMES = tuple(
    name for name in SENSOR_INPUT_NAMES if "pheromone" in name
)


def pheromone_only_brain(genome_id: int) -> NeatBrain:
    """Build a NEAT brain driven only by trail and alarm concentrations.
    
    Parameters
    ----------
    genome_id
        Value supplied to ``genome_id`` by the test scenario.
    
    Returns
    -------
    None
        Result produced by this test helper.
    """
    # Keep the pheromone only brain test intent explicit.
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

    # Trail turns toward the stronger side. Alarm uses the opposite sign, so
    # the same spatial observation produces an escape turn. Any sensed
    # pheromone supplies forward drive; alarm_here therefore escapes a patch
    # even when there is no left/right difference.
    wiring = (
        ("trail_pheromone_here", BrainOutputIndex.ACCELERATE, 2.0),
        ("trail_pheromone_forward_left", BrainOutputIndex.ACCELERATE, 2.0),
        ("trail_pheromone_forward_right", BrainOutputIndex.ACCELERATE, 2.0),
        ("alarm_pheromone_here", BrainOutputIndex.ACCELERATE, 2.0),
        ("alarm_pheromone_forward_left", BrainOutputIndex.ACCELERATE, 2.0),
        ("alarm_pheromone_forward_right", BrainOutputIndex.ACCELERATE, 2.0),
        ("trail_pheromone_forward_left", BrainOutputIndex.ROTATE, 3.0),
        ("trail_pheromone_forward_right", BrainOutputIndex.ROTATE, -3.0),
        ("alarm_pheromone_forward_left", BrainOutputIndex.ROTATE, -3.0),
        ("alarm_pheromone_forward_right", BrainOutputIndex.ROTATE, 3.0),
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


def probe_creature() -> Creature:
    """Exercise probe creature behavior.
    
    Parameters
    ----------
    None
        This callable receives no external parameters.
    
    Returns
    -------
    None
        The test completes through assertions.
    """
    # Keep the probe creature test intent explicit.
    radius = 12.0
    body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0.0, radius))
    body.position = (0.0, 0.0)
    shape = pymunk.Circle(body, radius)
    return Creature(
        creature_id=1,
        name="Pheromone probe",
        body=body,
        shape=shape,
        energy=1.0,
        vision=VisionTraits(range=120.0, angle=1.0),
        physical_traits=PhysicalTraits(radius=radius),
        color=(255, 255, 255),
    )


class PheromoneOnlyBrainTest(unittest.TestCase):
    def setUp(self) -> None:
        """Exercise setUp behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the setUp test intent explicit.
        self.config = SimConfig()
        self.config.action.max_forward_force = 20.0
        self.config.action.action_smoothing_alpha = 1.0
        self.config.action.active_angular_velocity_retention = 1.0
        self.config.action.turn_control_gain = 0.25
        self.config.action.turn_response = 1.0
        self.config.action.collision_avoidance_force_scale = 0.0
        self.config.action.forward_velocity_retention = 1.0
        self.config.action.lateral_velocity_retention = 1.0
        self.config.action.linear_stop_threshold = 0.0
        self.config.communication.pheromone_diffusion_coefficient = 0.0
        self.config.communication.pheromone_evaporation_rate = 0.0
        self.vision = VisionSystem(self.config.vision)

        self.world = World.__new__(World)
        self.world.config = self.config
        self.world.creatures = []
        self.world._motion_commands = {}

    def make_field(self, *, trail: bool) -> PheromoneSystem:
        """Exercise make field behavior.
        
        Parameters
        ----------
        trail
            Value supplied to ``trail`` by the test scenario.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the make field test intent explicit.
        field = PheromoneSystem(
            self.config.communication,
            grid_width=61,
            grid_height=61,
            world_bounds=WORLD_BOUNDS,
        )
        # A horizontal signal band crosses the initial forward-left sensor.
        # Depositing through the public API keeps interpolation and sampling in
        # the test path.
        for x in range(-280, 281, 10):
            field.deposit(
                (float(x), 48.0),
                trail_amount=1.0 if trail else 0.0,
                alarm_amount=0.0 if trail else 1.0,
            )
        return field

    def snapshot_for(
        self,
        creature: Creature,
        field: PheromoneSystem,
    ):
        """Exercise snapshot for behavior.
        
        Parameters
        ----------
        creature
            Value supplied to ``creature`` by the test scenario.
        field
            Value supplied to ``field`` by the test scenario.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the snapshot for test intent explicit.
        snapshot = self.vision.sense(
            creature,
            foods=[],
            creatures=[creature],
            world_bounds=WORLD_BOUNDS,
            max_speed=World.MAX_SPEED,
        )
        snapshot.pheromones = field.sense(
            self.world.pheromone_sensor_positions_for(creature)
        )
        return snapshot

    def simulate(
        self,
        creature: Creature,
        field: PheromoneSystem,
        brain: NeatBrain,
    ) -> None:
        """Exercise simulate behavior.
        
        Parameters
        ----------
        creature
            Value supplied to ``creature`` by the test scenario.
        field
            Value supplied to ``field`` by the test scenario.
        brain
            Value supplied to ``brain`` by the test scenario.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the simulate test intent explicit.
        space = pymunk.Space()
        space.gravity = (0.0, 0.0)
        space.damping = 0.9
        space.add(creature.body, creature.shape)
        self.world.creatures = [creature]

        for _ in range(60):
            snapshot = self.snapshot_for(creature, field)
            action = brain.decide(snapshot)
            self.world._apply_action(
                creature,
                action,
                snapshot,
            )
            space.step(1.0 / 60.0)
            self.world._apply_top_down_motion()
            self.world._limit_creature_motion()

    def test_only_pheromone_sensors_reach_accelerate_and_rotate(self) -> None:
        """Exercise test only pheromone sensors reach accelerate and rotate behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test only pheromone sensors reach accelerate and rotate test intent explicit.
        brain = pheromone_only_brain(1)
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
                for name in PHEROMONE_SENSOR_NAMES
            },
        )
        self.assertEqual(
            {target for _, target in enabled_edges},
            {
                genome_config.output_keys[BrainOutputIndex.ACCELERATE],
                genome_config.output_keys[BrainOutputIndex.ROTATE],
            },
        )

    def test_trail_attracts_while_alarm_on_same_side_repels(self) -> None:
        """Exercise test trail attracts while alarm on same side repels behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test trail attracts while alarm on same side repels test intent explicit.
        creature = probe_creature()
        brain = pheromone_only_brain(1)
        trail_snapshot = self.snapshot_for(creature, self.make_field(trail=True))
        trail_action = brain.decide(trail_snapshot)
        alarm_snapshot = self.snapshot_for(creature, self.make_field(trail=False))
        alarm_action = brain.decide(alarm_snapshot)

        self.assertGreater(
            trail_snapshot.pheromones.trail_forward_left,
            trail_snapshot.pheromones.trail_forward_right,
        )
        self.assertGreater(trail_action.accelerate, 0.0)
        self.assertGreater(trail_action.rotate, 0.0)
        self.assertGreater(
            alarm_snapshot.pheromones.alarm_forward_left,
            alarm_snapshot.pheromones.alarm_forward_right,
        )
        self.assertGreater(alarm_action.accelerate, 0.0)
        self.assertLess(alarm_action.rotate, 0.0)

        for action in (trail_action, alarm_action):
            active_indices = {
                index
                for index, value in enumerate(astuple(action))
                if abs(value) > 1e-12
            }
            self.assertEqual(
                active_indices,
                {BrainOutputIndex.ACCELERATE, BrainOutputIndex.ROTATE},
            )

    def test_creature_moves_toward_trail_band(self) -> None:
        """Exercise test creature moves toward trail band behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test creature moves toward trail band test intent explicit.
        creature = probe_creature()
        initial_distance = abs(creature.position[1] - 48.0)

        self.simulate(creature, self.make_field(trail=True), pheromone_only_brain(1))

        self.assertGreater(creature.position[1], 0.0)
        self.assertLess(abs(creature.position[1] - 48.0), initial_distance)

    def test_creature_moves_away_from_alarm_band(self) -> None:
        """Exercise test creature moves away from alarm band behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test creature moves away from alarm band test intent explicit.
        creature = probe_creature()
        initial_distance = abs(creature.position[1] - 48.0)

        self.simulate(creature, self.make_field(trail=False), pheromone_only_brain(1))

        self.assertLess(creature.position[1], 0.0)
        self.assertGreater(abs(creature.position[1] - 48.0), initial_distance)


if __name__ == "__main__":
    unittest.main()
