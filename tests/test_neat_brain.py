from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys
import unittest


class _Body:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.position = SimpleNamespace(x=0.0, y=0.0)


class _Circle:
    def __init__(self, body: _Body, radius: float) -> None:
        self.body = body
        self.radius = radius


try:
    import pymunk  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pymunk"] = SimpleNamespace(
        Body=_Body,
        Circle=_Circle,
        ShapeFilter=lambda **kwargs: SimpleNamespace(**kwargs),
        moment_for_circle=lambda *args: 1.0,
    )

try:
    import neat
except ModuleNotFoundError:
    sys.modules["neat"] = ModuleType("neat")
    import neat

from src.neat_brain import NeatBrain
from src.vision import BoundarySnapshot, SensorSnapshot, VisionTargetSnapshot


class FakeNetwork:
    def __init__(self, outputs: list[float]) -> None:
        self.outputs = outputs
        self.activate_count = 0

    def activate(self, inputs: list[float]) -> list[float]:
        self.activate_count += 1
        return self.outputs


def empty_target() -> VisionTargetSnapshot:
    return VisionTargetSnapshot(
        visible=0.0,
        nearest_closeness=0.0,
        nearest_angle=0.0,
        density=0.0,
        count=0,
    )


def sensor_snapshot() -> SensorSnapshot:
    return SensorSnapshot(
        food=empty_target(),
        creatures=empty_target(),
        walls=empty_target(),
        boundary=BoundarySnapshot(pressure=0.0, turn=0.0),
        energy=1.0,
        speed=0.0,
        vision_range=0.0,
        vision_angle=0.0,
        vision_energy_cost=0.0,
        maturity=0.0,
        visible_food_count=0.0,
        visible_creature_count=0.0,
        clock_tik_tok=0.0,
        clock_chronometer=0.0,
        clock_time_alive=0.0,
    )


class NeatBrainActionMappingTest(unittest.TestCase):
    def decide_with_outputs(self, outputs: list[float]):
        brain = NeatBrain(
            genome_id=1,
            genome=SimpleNamespace(),
            network=FakeNetwork(outputs),
        )
        return brain.decide(sensor_snapshot())

    def test_neutral_movement_outputs_map_to_stillness(self) -> None:
        action = self.decide_with_outputs([0.5, 0.5, 0.0, 0.0, 0.0])

        self.assertAlmostEqual(action.accelerate, 0.0)
        self.assertAlmostEqual(action.rotate, 0.0)

    def test_acceleration_output_is_signed(self) -> None:
        action = self.decide_with_outputs([0.25, 0.5, 0.0, 0.0, 0.0])

        self.assertAlmostEqual(action.accelerate, -0.5)
        self.assertAlmostEqual(action.rotate, 0.0)

    def test_rotation_output_is_signed(self) -> None:
        left_action = self.decide_with_outputs([0.5, 0.25, 0.0, 0.0, 0.0])
        right_action = self.decide_with_outputs([0.5, 0.75, 0.0, 0.0, 0.0])

        self.assertAlmostEqual(left_action.rotate, -0.5)
        self.assertAlmostEqual(right_action.rotate, 0.5)

    def test_intent_outputs_remain_normalized(self) -> None:
        action = self.decide_with_outputs([0.5, 0.5, 1.2, -0.2, 0.75])

        self.assertEqual(action.want_reproduce, 1.0)
        self.assertEqual(action.want_eat, 0.0)
        self.assertEqual(action.reset_chronometer, 0.75)


class NeatBrainNetworkCachingTest(unittest.TestCase):
    def test_from_genome_compiles_network_once_and_reuses_it(self) -> None:
        created_networks: list[FakeNetwork] = []
        fake_network = FakeNetwork([0.5, 0.5, 0.0, 0.0, 0.0])

        class FakeFeedForwardNetwork:
            @staticmethod
            def create(genome: object, config: object) -> FakeNetwork:
                created_networks.append(fake_network)
                return fake_network

        original_nn = getattr(neat, "nn", None)
        neat.nn = SimpleNamespace(FeedForwardNetwork=FakeFeedForwardNetwork)

        try:
            config = SimpleNamespace(
                genome_config=SimpleNamespace(output_keys=[0, 1, 2, 3, 4])
            )
            genome = SimpleNamespace(nodes={})

            brain = NeatBrain.from_genome(1, genome, config)
            first_network = brain.network

            brain.decide(sensor_snapshot())
            brain.decide(sensor_snapshot())
            brain.decide(sensor_snapshot())
        finally:
            if original_nn is None:
                del neat.nn
            else:
                neat.nn = original_nn

        self.assertEqual(created_networks, [fake_network])
        self.assertIs(brain.network, first_network)
        self.assertEqual(fake_network.activate_count, 3)


if __name__ == "__main__":
    unittest.main()
