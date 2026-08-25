"""Deterministic before/after harness for Milestone 1 hot-loop cleanup.

Run this unchanged on the baseline and optimized revisions. Timing and counters
are collected separately so instrumentation does not contaminate the timings.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_flocking import fill_to_physical_capacity  # noqa: E402
from configs.sim_config import build_sim_config  # noqa: E402
import src.action as action_module  # noqa: E402
import src.neat_brain as neat_brain_module  # noqa: E402
from src.neat_brain import NeatBrain  # noqa: E402
from src.persistence import SimulationPaths  # noqa: E402
from src.vision import SensorSnapshot  # noqa: E402
import src.world as world_module  # noqa: E402
from src.world import World  # noqa: E402


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def make_world() -> World:
    config = build_sim_config()
    config.random_seed = 11
    config.flocking.long_range.enabled = True
    config.persistence.enable_telemetry = False
    config.persistence.quick_save_interval_seconds = 0.0
    config.persistence.archive_save_interval_seconds = 0.0
    # Keep scheduler baselines comparable with the pre-clustering workload.
    config.food_clusters.cluster_spawn_share = 0.0
    config.population.initial_creatures = min(
        config.population.max_creatures,
        50,
    )
    world = World(config, simulation_paths=SimulationPaths(ROOT))
    fill_to_physical_capacity(world)
    return world


def measure_case(
    fixed_steps_per_frame: int,
    repetitions: int,
    warmup_frames: int,
    measured_frames: int,
) -> dict[str, object]:
    samples_ms: list[float] = []
    run_medians_ms: list[float] = []
    for _ in range(repetitions):
        world = make_world()
        try:
            world.set_simulation_speed(float(fixed_steps_per_frame))
            for _ in range(warmup_frames):
                world.update(World.FIXED_TIMESTEP)
            population = len(world.creatures)
            run_samples: list[float] = []
            for _ in range(measured_frames):
                started = perf_counter()
                world.update(World.FIXED_TIMESTEP)
                run_samples.append((perf_counter() - started) * 1_000.0)
            if len(world.creatures) != population:
                raise RuntimeError("Benchmark population changed during timing.")
            samples_ms.extend(run_samples)
            run_medians_ms.append(statistics.median(run_samples))
        finally:
            world.close()

    median_ms = statistics.median(samples_ms)
    noise = (
        (percentile(run_medians_ms, 0.95) - percentile(run_medians_ms, 0.05))
        / max(2.0 * statistics.median(run_medians_ms), 1e-12)
        * 100.0
    )
    return {
        "fixed_steps_per_frame": fixed_steps_per_frame,
        "samples": len(samples_ms),
        "median_ms": median_ms,
        "p95_ms": percentile(samples_ms, 0.95),
        "baseline_noise_percent": noise,
        "run_medians_ms": run_medians_ms,
    }


def collect_counters(warmup_steps: int, measured_steps: int) -> dict[str, object]:
    world = make_world()
    counters = {
        "runtime_snapshots": 0,
        "motion_commands": 0,
        "actions": 0,
        "cohort_synchronizations": 0,
        "deposit_many_calls": 0,
        "inactive_batches": 0,
        "inactive_deposition_calls_avoided": 0,
        "as_inputs_calls": 0,
        "inspector_input_snapshots": 0,
    }
    sensing_ms: list[float] = []
    intent_ms: list[float] = []

    runtime_type = world_module.FlockingRuntimeSnapshot
    motion_type = world_module.MotionCommand
    action_type = neat_brain_module.Action
    original_as_inputs = SensorSnapshot.as_inputs
    original_capture = getattr(NeatBrain, "capture_input_snapshot", None)
    original_sync = world._sync_automatic_behavior_cohort
    original_deposit = world.pheromones.deposit_many
    original_commit = world._commit_communication_intents
    original_sense = world._sensor_snapshot_for
    original_intents = world._apply_creature_intents

    def counted_runtime(*args, **kwargs):
        counters["runtime_snapshots"] += 1
        return runtime_type(*args, **kwargs)

    def counted_motion(*args, **kwargs):
        counters["motion_commands"] += 1
        return motion_type(*args, **kwargs)

    def counted_action(*args, **kwargs):
        counters["actions"] += 1
        return action_type(*args, **kwargs)

    def counted_as_inputs(snapshot):
        counters["as_inputs_calls"] += 1
        return original_as_inputs(snapshot)

    def counted_capture(brain):
        counters["inspector_input_snapshots"] += 1
        assert original_capture is not None
        return original_capture(brain)

    def counted_sync():
        counters["cohort_synchronizations"] += 1
        return original_sync()

    def counted_deposit(*args, **kwargs):
        counters["deposit_many_calls"] += 1
        return original_deposit(*args, **kwargs)

    def counted_commit(delta_time):
        active = False
        for creature in world.creatures:
            action = world._action_for_execution(creature.creature_id)
            if action is not None and (
                action.emit_red > 0.0
                or action.emit_green > 0.0
                or action.emit_blue > 0.0
            ):
                active = True
                break
        if not active:
            counters["inactive_batches"] += 1
        deposits_before = counters["deposit_many_calls"]
        result = original_commit(delta_time)
        if (
            not active
            and counters["deposit_many_calls"] == deposits_before
        ):
            counters["inactive_deposition_calls_avoided"] += 1
        return result

    def timed_sense(*args, **kwargs):
        started = perf_counter()
        try:
            return original_sense(*args, **kwargs)
        finally:
            sensing_ms.append((perf_counter() - started) * 1_000.0)

    def timed_intents():
        started = perf_counter()
        try:
            return original_intents()
        finally:
            intent_ms.append((perf_counter() - started) * 1_000.0)

    try:
        for _ in range(warmup_steps):
            world.update(World.FIXED_TIMESTEP)
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(world_module, "FlockingRuntimeSnapshot", counted_runtime)
            )
            stack.enter_context(patch.object(world_module, "MotionCommand", counted_motion))
            stack.enter_context(patch.object(neat_brain_module, "Action", counted_action))
            stack.enter_context(patch.object(action_module, "Action", counted_action))
            stack.enter_context(
                patch.object(SensorSnapshot, "as_inputs", counted_as_inputs)
            )
            if original_capture is not None:
                stack.enter_context(
                    patch.object(
                        NeatBrain,
                        "capture_input_snapshot",
                        counted_capture,
                    )
                )
            world._sync_automatic_behavior_cohort = counted_sync
            world.pheromones.deposit_many = counted_deposit
            world._commit_communication_intents = counted_commit
            world._sensor_snapshot_for = timed_sense
            world._apply_creature_intents = timed_intents
            for _ in range(measured_steps):
                world.update(World.FIXED_TIMESTEP)

            # Measure the all-inactive communication case independently of the
            # seeded networks' current emissions.  This keeps the counter
            # deterministic while leaving timing runs entirely uninstrumented.
            for action in world._last_actions.values():
                action.emit_red = 0.0
                action.emit_green = 0.0
                action.emit_blue = 0.0
            for _ in range(measured_steps):
                world._commit_communication_intents(World.FIXED_TIMESTEP)
    finally:
        world.close()

    counters["sensing_median_ms"] = statistics.median(sensing_ms)
    counters["sensing_p95_ms"] = percentile(sensing_ms, 0.95)
    counters["intent_median_ms"] = statistics.median(intent_ms)
    counters["intent_p95_ms"] = percentile(intent_ms, 0.95)
    return counters


def comparison(
    baseline: dict[str, object],
    current: dict[str, object],
) -> dict[str, object]:
    timing: dict[str, object] = {}
    for name in ("one_step", "five_steps"):
        before = baseline[name]
        after = current[name]
        before_median = float(before["median_ms"])
        after_median = float(after["median_ms"])
        noise = float(before["baseline_noise_percent"])
        improvement = (
            (before_median - after_median)
            / max(before_median, 1e-12)
            * 100.0
        )
        threshold = noise if noise >= 3.0 else 3.0
        timing[name] = {
            "median_absolute_change_ms": after_median - before_median,
            "median_improvement_percent": improvement,
            "p95_absolute_change_ms": (
                float(after["p95_ms"]) - float(before["p95_ms"])
            ),
            "baseline_noise_percent": noise,
            "positive_threshold_percent": threshold,
            "performance_positive": improvement > threshold,
        }
    counter_deltas = {
        key: float(current["counters"].get(key, 0.0))
        - float(baseline["counters"].get(key, 0.0))
        for key in sorted(
            {*baseline["counters"], *current["counters"]}
        )
    }
    return {"timing": timing, "counter_deltas": counter_deltas}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--measured-frames", type=int, default=60)
    parser.add_argument("--counter-steps", type=int, default=120)
    parser.add_argument(
        "--compare",
        type=Path,
        help="Baseline JSON produced by this harness.",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    repetitions = 3 if args.quick else max(10, args.repetitions)
    warmups = 5 if args.quick else max(1, args.warmup_frames)
    frames = 8 if args.quick else max(1, args.measured_frames)
    counter_steps = 10 if args.quick else max(1, args.counter_steps)
    result = {
        "seed": 11,
        "population": 55,
        "repetitions": repetitions,
        "one_step": measure_case(1, repetitions, warmups, frames),
        "five_steps": measure_case(5, repetitions, warmups, frames),
        "counters": collect_counters(warmups, counter_steps),
    }
    if args.compare is not None:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        result["comparison"] = comparison(baseline, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
