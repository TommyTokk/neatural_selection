"""Measure focal behaviour-observer producer and worker overhead.

Run from the repository root:

    python benchmarks/benchmark_behavior_observer.py --samples 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.sim_config import BehaviorObserverConfig
from configs.sim_config import build_sim_config
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


def run(sample_count: int, world_steps: int) -> None:
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
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.world_steps < 0:
        parser.error("--world-steps must be nonnegative")
    run(args.samples, args.world_steps)


if __name__ == "__main__":
    main()
