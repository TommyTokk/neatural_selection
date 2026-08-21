from __future__ import annotations

from types import SimpleNamespace
import unittest

from configs.sim_config import build_sim_config
from src.action import ACTION_OUTPUT_COUNT
from src.behavior_observer import (
    BehaviorKind,
    BehaviorSnapshot,
    BehaviorStateSnapshot,
    BoutStatus,
)
from src.counterfactual_neat import WhySnapshot
from src.vision import SENSOR_INPUT_COUNT
from src.world import World


class _WhyObserver:
    def __init__(self) -> None:
        self.latest_snapshot = None
        self.latest_why_snapshots = ()
        self.probes = []
        self.focuses = []
        self.brains = []

    def set_focus(self, creature_id, generation) -> None:
        self.focuses.append((creature_id, generation))
        self.latest_snapshot = None
        self.latest_why_snapshots = ()

    def set_focal_brain(self, update) -> bool:
        self.brains.append(update)
        return True

    def submit_why(self, probe) -> bool:
        self.probes.append(probe)
        return True


class _FixedNetwork:
    def activate(self, _inputs):
        return [0.2] * ACTION_OUTPUT_COUNT


def _state(
    behavior: BehaviorKind,
    *,
    bout_id: int = 1,
    target_id: int | None = None,
) -> BehaviorStateSnapshot:
    if (
        target_id is None
        and behavior
        in {
            BehaviorKind.FOOD_ORIENTATION,
            BehaviorKind.FOOD_APPROACH,
        }
    ):
        target_id = 5
    return BehaviorStateSnapshot(
        behavior=behavior,
        status=BoutStatus.ACTIVE,
        evidence_score=0.8,
        duration_seconds=1.0,
        evidence=(),
        bout_id=bout_id,
        target_id=target_id,
    )


def world_shell(behavior: BehaviorKind) -> World:
    world = object.__new__(World)
    world.config = build_sim_config()
    world.elapsed_time = 0.0
    world._why_next_probe_time = 0.0
    world._behavior_selection_generation = 2
    world.selected_creature_id = 4
    creature = SimpleNamespace(creature_id=4, heading=0.1)
    world.creatures = [creature]
    brain = SimpleNamespace(
        brain_revision=7,
        last_inputs=[0.1] * SENSOR_INPUT_COUNT,
        last_outputs=[0.2] * ACTION_OUTPUT_COUNT,
        output_activations=["clamped"] * ACTION_OUTPUT_COUNT,
        has_captured_activation_state=True,
        clone_network=lambda **_kwargs: _FixedNetwork(),
        captured_activation_network_state=lambda: {
            "active": 0,
            "values": [{0: 0.0}, {0: 0.0}],
        },
    )
    world.neat_controller = SimpleNamespace(
        brain_for=lambda creature_id: brain if creature_id == 4 else None
    )
    world._last_sensor_snapshots = {
        4: SimpleNamespace(
            food=SimpleNamespace(
                visible=1.0,
                nearest_id=5,
                relative_angle=-0.25,
            ),
            flock=SimpleNamespace(
                flockmate_count=3.0,
                cohesion_absolute_angle=0.5,
            ),
        )
    }
    world.behavior_observer = _WhyObserver()
    world.behavior_observer.latest_snapshot = BehaviorSnapshot(
        creature_id=4,
        selection_generation=2,
        simulation_time=0.0,
        behaviors=(_state(behavior),),
        observations_processed=1,
        produced_monotonic=0.0,
    )
    return world


class WorldCounterfactualTest(unittest.TestCase):
    def test_mapped_behavior_probes_at_five_hz(self) -> None:
        world = world_shell(BehaviorKind.FOOD_APPROACH)

        world._sample_selected_why()
        world.elapsed_time = 0.1
        world._sample_selected_why()
        world.elapsed_time = 0.2
        world._sample_selected_why()

        self.assertEqual(len(world.behavior_observer.probes), 2)
        probe = world.behavior_observer.probes[-1]
        self.assertEqual(probe.brain_revision, 7)
        self.assertEqual(
            probe.behaviors[0].behavior,
            BehaviorKind.FOOD_APPROACH,
        )
        self.assertTrue(probe.target_visible)
        self.assertEqual(probe.food_target_id, 5)
        self.assertEqual(probe.food_relative_angle, -0.25)
        self.assertEqual(probe.behaviors[0].target_id, 5)
        self.assertEqual(
            probe.network_state,
            {"active": 0, "values": [{0: 0.0}, {0: 0.0}]},
        )

    def test_resting_alone_never_submits_a_neural_probe(self) -> None:
        world = world_shell(BehaviorKind.RESTING)

        world._sample_selected_why()

        self.assertEqual(world.behavior_observer.probes, [])

    def test_missing_target_defers_only_target_oriented_behaviors(self) -> None:
        world = world_shell(BehaviorKind.FOOD_APPROACH)
        world._last_sensor_snapshots[4] = SimpleNamespace(
            food=SimpleNamespace(
                visible=0.0,
                nearest_id=None,
                relative_angle=None,
            )
        )

        world._sample_selected_why()

        self.assertEqual(world.behavior_observer.probes, [])

        world.behavior_observer.latest_snapshot = BehaviorSnapshot(
            creature_id=4,
            selection_generation=2,
            simulation_time=0.0,
            behaviors=(
                _state(BehaviorKind.FOOD_APPROACH),
                _state(BehaviorKind.FEEDING),
            ),
            observations_processed=1,
            produced_monotonic=0.0,
        )
        world._sample_selected_why()

        self.assertEqual(len(world.behavior_observer.probes), 1)
        self.assertEqual(
            tuple(
                behavior.behavior
                for behavior in world.behavior_observer.probes[0].behaviors
            ),
            (BehaviorKind.FEEDING,),
        )

    def test_cohesion_probe_carries_factual_group_heading(self) -> None:
        world = world_shell(BehaviorKind.COHESION)

        world._sample_selected_why()

        self.assertEqual(len(world.behavior_observer.probes), 1)
        probe = world.behavior_observer.probes[0]
        self.assertTrue(probe.group_visible)
        self.assertAlmostEqual(probe.group_relative_angle, 0.4)

    def test_missing_group_context_defers_only_cohesion(self) -> None:
        world = world_shell(BehaviorKind.COHESION)
        world._last_sensor_snapshots[4].flock.flockmate_count = 0.0
        world.behavior_observer.latest_snapshot = BehaviorSnapshot(
            creature_id=4,
            selection_generation=2,
            simulation_time=0.0,
            behaviors=(
                _state(BehaviorKind.COHESION),
                _state(BehaviorKind.FEEDING),
            ),
            observations_processed=1,
            produced_monotonic=0.0,
        )

        world._sample_selected_why()

        self.assertEqual(len(world.behavior_observer.probes), 1)
        self.assertEqual(
            tuple(
                behavior.behavior
                for behavior in world.behavior_observer.probes[0].behaviors
            ),
            (BehaviorKind.FEEDING,),
        )

    def test_mismatched_target_id_defers_target_oriented_probe(self) -> None:
        world = world_shell(BehaviorKind.FOOD_ORIENTATION)
        world._last_sensor_snapshots[4].food.nearest_id = 6

        world._sample_selected_why()

        self.assertEqual(world.behavior_observer.probes, [])

    def test_nonfinite_target_angle_defers_target_oriented_probe(self) -> None:
        for value in (float("nan"), float("inf"), None):
            with self.subTest(value=value):
                world = world_shell(BehaviorKind.FOOD_ORIENTATION)
                world._last_sensor_snapshots[4].food.relative_angle = value

                world._sample_selected_why()

                self.assertEqual(world.behavior_observer.probes, [])

    def test_selected_why_snapshots_reject_stale_bout(self) -> None:
        world = world_shell(BehaviorKind.FOOD_APPROACH)
        current = WhySnapshot(
            creature_id=4,
            selection_generation=2,
            brain_revision=7,
            simulation_time=0.0,
            behavior=BehaviorKind.FOOD_APPROACH,
            status=BoutStatus.ACTIVE,
            bout_id=1,
            behavior_duration=1.0,
            effects=(),
            produced_monotonic=0.0,
            target_id=5,
        )
        stale = SimpleNamespace(**{
            field: getattr(current, field)
            for field in current.__dataclass_fields__
        })
        stale.bout_id = 99
        wrong_target = SimpleNamespace(**{
            field: getattr(current, field)
            for field in current.__dataclass_fields__
        })
        wrong_target.target_id = 6
        world.behavior_observer.latest_why_snapshots = (
            current,
            stale,
            wrong_target,
        )

        self.assertEqual(world.selected_why_snapshots, (current,))

        world._last_sensor_snapshots[4].food.nearest_id = 6

        self.assertEqual(world.selected_why_snapshots, ())


if __name__ == "__main__":
    unittest.main()
