"""Measure hybrid behaviour-observer producer and worker overhead.

Run from the repository root:

    python benchmarks/benchmark_behavior_observer.py --samples 5000
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.sim_config import BehaviorHistoryConfig, BehaviorObserverConfig
from configs.sim_config import build_sim_config
from src.behavior_history import CreatureBehaviorHistoryStore
from src.behavior_observer import (
    BehaviorObservation,
    BehaviorObserverService,
    TemporalBehaviorAnalyzer,
)
from src.persistence import SimulationPaths
from src.world import World


ROOT = Path(__file__).resolve().parents[1]


def sample(index: int, hz: float) -> BehaviorObservation:
    simulation_time = index / hz
    distance = max(0.0, 150.0 - simulation_time * 12.0)
    angle = max(0.0, 0.8 - simulation_time * 0.18)
    return BehaviorObservation(
        creature_id=1,
        selection_generation=1,
        simulation_time=simulation_time,
        x=simulation_time * 20.0,
        y=0.0,
        heading=0.0,
        angular_velocity=0.5,
        velocity_x=20.0,
        velocity_y=0.0,
        speed=20.0,
        nearest_food_id=5,
        food_visible=True,
        food_distance=distance,
        food_relative_angle=angle,
    )


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def benchmark_seeded_world(
    observer_enabled: bool,
    steps: int,
) -> list[float]:
    """Measure fixed-step frame time in the same seeded world scenario."""
    config = build_sim_config()
    config.random_seed = 19
    config.behavior.enabled = observer_enabled
    config.persistence.enable_telemetry = False
    config.persistence.quick_save_interval_seconds = 0.0
    config.persistence.archive_save_interval_seconds = 0.0
    world = World(config, simulation_paths=SimulationPaths(ROOT))
    try:
        if world.creatures:
            world.select_creature_by_id(world.creatures[0].creature_id)
        for _ in range(min(30, max(5, steps // 5))):
            world.update(World.FIXED_TIMESTEP)
        timings = []
        for _ in range(steps):
            started = time.perf_counter()
            world.update(World.FIXED_TIMESTEP)
            timings.append((time.perf_counter() - started) * 1000.0)
        return timings
    finally:
        world.close()


def benchmark_long_term_history(completed_bouts: int) -> None:
    """Exercise finalized-bout throughput and prove configured retention."""
    observer_config = BehaviorObserverConfig(feeding_display_seconds=0.01)
    history_config = BehaviorHistoryConfig()
    analyzer = TemporalBehaviorAnalyzer(
        observer_config,
        history_config,
    )
    store = CreatureBehaviorHistoryStore(
        max_completed_bouts_per_creature=(
            history_config.max_completed_bouts_per_creature
        ),
        max_remembered_creatures=(
            history_config.max_remembered_creatures
        ),
        minimum_stable_bouts=history_config.minimum_stable_bouts,
    )
    store.register_creature(1, "Benchmark focal", 0.0)
    started = time.perf_counter()
    for index in range(completed_bouts):
        simulation_time = index * 0.1
        base = sample(index, observer_config.sample_hz)
        analyzer.process(
            replace(
                base,
                simulation_time=simulation_time,
                food_consumption_count=index + 1,
                food_consumed_energy_total=(index + 1) * 0.1,
            )
        )
        analyzer.process(
            replace(
                base,
                simulation_time=simulation_time + 0.02,
                food_consumption_count=index + 1,
                food_consumed_energy_total=(index + 1) * 0.1,
            )
        )
        for draft in analyzer.drain_completed_bouts():
            store.append_draft(draft)
    elapsed = time.perf_counter() - started
    diagnostics = store.diagnostics
    expected_retained = min(
        completed_bouts,
        history_config.max_completed_bouts_per_creature,
    )
    if diagnostics.completed_bouts_stored != expected_retained:
        raise RuntimeError(
            "Long-term history exceeded or failed its configured retention "
            "bound."
        )
    if len(store._seen_sources) > store._seen_source_capacity:
        raise RuntimeError("Completion deduplication memory exceeded its bound.")
    print(
        "Long-term completion throughput: "
        f"{completed_bouts / max(elapsed, 1e-12):,.0f} bouts/s"
    )
    print(
        "Long-term retained / dropped detail: "
        f"{diagnostics.completed_bouts_stored} / "
        f"{diagnostics.detailed_bouts_dropped}"
    )


def benchmark_batched_cohort(subject_count: int, batch_count: int) -> None:
    """Exercise a maximum-population cohort without subject starvation."""
    config = BehaviorObserverConfig(input_queue_capacity=32)
    service = BehaviorObserverService(config)
    subjects = tuple(
        (creature_id, creature_id)
        for creature_id in range(1, subject_count + 1)
    )
    started = time.perf_counter()
    try:
        service.set_subjects(subjects)
        for tick in range(batch_count):
            base = sample(tick, config.sample_hz)
            service.submit_batch(
                tuple(
                    replace(
                        base,
                        creature_id=creature_id,
                        selection_generation=creature_id,
                    )
                    for creature_id in range(1, subject_count + 1)
                )
            )
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            service.poll()
            if (
                len(service.latest_snapshots) == subject_count
                and service.diagnostics.observations_processed
                + service.diagnostics.samples_dropped
                >= subject_count * batch_count
            ):
                break
            time.sleep(0.005)
        missing = subject_count - len(service.latest_snapshots)
        if missing:
            raise RuntimeError(
                f"Batched cohort starved {missing} of {subject_count} subjects."
            )
        diagnostics = service.diagnostics
    finally:
        service.close()
    elapsed = time.perf_counter() - started
    print(
        "Batched cohort subjects / ticks / throughput: "
        f"{subject_count} / {batch_count} / "
        f"{subject_count * batch_count / max(elapsed, 1e-12):,.0f} samples/s"
    )
    print(
        "Batched cohort processed / dropped: "
        f"{diagnostics.observations_processed} / {diagnostics.samples_dropped}"
    )


def run(
    sample_count: int,
    world_steps: int,
    history_bouts: int,
    cohort_subjects: int,
    cohort_batches: int,
) -> None:
    config = BehaviorObserverConfig()
    observations = [sample(index, config.sample_hz) for index in range(sample_count)]

    baseline_timings: list[float] = []
    for observation in observations:
        started = time.perf_counter()
        _ = observation.simulation_time
        baseline_timings.append((time.perf_counter() - started) * 1000.0)

    analyzer = TemporalBehaviorAnalyzer(config)
    core_started = time.perf_counter()
    for observation in observations:
        analyzer.process(observation)
    core_elapsed = time.perf_counter() - core_started

    service = BehaviorObserverService(config)
    enqueue_timings: list[float] = []
    try:
        service.set_focus(1, 1)
        for observation in observations:
            started = time.perf_counter()
            service.submit(observation)
            enqueue_timings.append((time.perf_counter() - started) * 1000.0)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            service.poll()
            diagnostics = service.diagnostics
            if (
                diagnostics.observations_processed
                + diagnostics.samples_dropped
                >= sample_count
            ):
                break
            time.sleep(0.005)
        diagnostics = service.diagnostics
    finally:
        service.close()

    baseline_median = statistics.median(baseline_timings)
    enqueue_median = statistics.median(enqueue_timings)
    print("Temporal behaviour observer benchmark")
    print(f"Samples requested: {sample_count}")
    print(
        "Disabled/no-op producer median: "
        f"{baseline_median:.6f} ms"
    )
    print(
        "Enabled enqueue median / p95: "
        f"{enqueue_median:.6f} / "
        f"{percentile(enqueue_timings, 0.95):.6f} ms"
    )
    print(
        "Incremental analyzer throughput: "
        f"{sample_count / max(core_elapsed, 1e-12):,.0f} samples/s"
    )
    print(
        "Worker processed / producer dropped: "
        f"{diagnostics.observations_processed} / "
        f"{diagnostics.samples_dropped}"
    )
    print(
        "Latest result latency: "
        + (
            "unavailable"
            if diagnostics.result_latency_ms is None
            else f"{diagnostics.result_latency_ms:.2f} ms"
        )
    )
    if history_bouts:
        benchmark_long_term_history(history_bouts)
    if cohort_subjects and cohort_batches:
        benchmark_batched_cohort(cohort_subjects, cohort_batches)
    if world_steps:
        disabled_frames = benchmark_seeded_world(False, world_steps)
        enabled_frames = benchmark_seeded_world(True, world_steps)
        disabled_median = statistics.median(disabled_frames)
        enabled_median = statistics.median(enabled_frames)
        degradation = (
            (enabled_median - disabled_median)
            / max(disabled_median, 1e-12)
            * 100.0
        )
        print(
            "Seeded world median frame time, disabled / enabled: "
            f"{disabled_median:.3f} / {enabled_median:.3f} ms"
        )
        print(
            "Seeded world median FPS, disabled / enabled: "
            f"{1000.0 / disabled_median:.1f} / "
            f"{1000.0 / enabled_median:.1f}"
        )
        print(
            "Seeded world median frame-time degradation: "
            f"{degradation:+.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--world-steps", type=int, default=180)
    parser.add_argument("--history-bouts", type=int, default=10000)
    parser.add_argument("--cohort-subjects", type=int, default=55)
    parser.add_argument("--cohort-batches", type=int, default=100)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.world_steps < 0:
        parser.error("--world-steps must be nonnegative")
    if args.history_bouts < 0:
        parser.error("--history-bouts must be nonnegative")
    if args.cohort_subjects < 0 or args.cohort_batches < 0:
        parser.error("cohort benchmark values must be nonnegative")
    run(
        args.samples,
        args.world_steps,
        args.history_bouts,
        args.cohort_subjects,
        args.cohort_batches,
    )


if __name__ == "__main__":
    main()
