from __future__ import annotations

from pathlib import Path
import unittest

import neat

from src.action import BrainOutputIndex
from src.communication import PheromoneSnapshot
from src.neat_brain import NeatBrain
from src.vision import BoundarySnapshot, SENSOR_INPUT_NAMES, SensorSnapshot, VisionTargetSnapshot


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "neat_herbivore.ini"


def empty_target() -> VisionTargetSnapshot:
    """Create an empty visual target fixture.

    Parameters
    ----------
    None
        This helper receives no external parameters.

    Returns
    -------
    VisionTargetSnapshot
        Target containing no visible entity.
    """
    # Reuse one canonical zero-valued target layout.
    return VisionTargetSnapshot(0.0, 0.0, 0.0, 0.0, 0)


def rgb_snapshot() -> SensorSnapshot:
    """Create a sensor fixture with distinct RGB gradient values.

    Parameters
    ----------
    None
        This helper receives no external parameters.

    Returns
    -------
    SensorSnapshot
        Complete snapshot containing controlled pheromone probes.
    """
    # Make Red local-only and Blue forward-biased for index-level assertions.
    return SensorSnapshot(
        food=empty_target(), creatures=empty_target(), walls=empty_target(),
        boundary=BoundarySnapshot(0.0, 0.0), energy=1.0, speed=0.0,
        vision_range=100.0, vision_angle=1.0, vision_energy_cost=0.0,
        reproductive_readiness=0.0, visible_food_count=0.0,
        visible_creature_count=0.0, clock_tik_tok=0.0,
        clock_chronometer=0.0, clock_time_alive=0.0, is_grabbing=0.0,
        pheromones=PheromoneSnapshot(
            local=(0.8, 0.0, 0.1),
            forward_left=(0.8, 0.0, 0.6),
            forward_right=(0.8, 0.0, 0.6),
        ),
    )


class RGBPheromoneOnlyBrainTest(unittest.TestCase):
    def test_rgb_gradient_input_order_and_values(self) -> None:
        """Verify schema-v8 RGB values occupy their exact input slots.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Assertions verify the packed sensor vector.
        """
        # Read all inputs through the same API used by live brains.
        inputs = rgb_snapshot().as_inputs()
        self.assertEqual(len(inputs), 46)
        self.assertEqual(inputs[36:39], [0.8, 0.0, 0.1])
        self.assertAlmostEqual(inputs[39], 0.0)
        self.assertAlmostEqual(inputs[42], 0.0)
        self.assertAlmostEqual(inputs[44], 1.0)
        self.assertEqual(inputs[45], 1.0)

    def test_manually_wired_genome_uses_v8_rgb_indices(self) -> None:
        """Verify a manual genome connects by schema-v8 sensor names.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Assertions verify Red and Blue neural dependencies.
        """
        # Construct explicit input-output connections without evolved topology.
        config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation, str(CONFIG_PATH),
        )
        genome = neat.DefaultGenome(1)
        for output_key in config.genome_config.output_keys:
            genome.nodes[output_key] = genome.create_node(config.genome_config, output_key)
        for node in genome.nodes.values():
            node.bias = 0.0
            node.response = 1.0
            node.activation = "tanh"
            node.aggregation = "sum"
        wiring = (
            ("pheromone_local_red", BrainOutputIndex.ACCELERATE, 3.0),
            ("pheromone_forward_blue", BrainOutputIndex.ROTATE, 3.0),
        )
        for innovation, (sensor_name, output_index, weight) in enumerate(wiring, 1):
            source = config.genome_config.input_keys[SENSOR_INPUT_NAMES.index(sensor_name)]
            target = config.genome_config.output_keys[int(output_index)]
            connection = genome.create_connection(
                config.genome_config, source, target, innovation
            )
            connection.weight = weight
            connection.enabled = True
            genome.connections[connection.key] = connection

        action = NeatBrain.from_genome(1, genome, config).decide(rgb_snapshot())

        self.assertGreater(action.accelerate, 0.0)
        self.assertGreater(action.rotate, 0.0)
        self.assertEqual((action.emit_red, action.emit_green, action.emit_blue), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
