"""Benchmark counterfactual activation cost and seeded-world overhead."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.sim_config import (
    BehaviorObserverConfig,
    CounterfactualWhyConfig,
    build_sim_config,
)
from src.behavior_observer import (
    BehaviorKind,
    BehaviorObservation,
    BehaviorObserverService,
    BoutStatus,
)
from src.counterfactual_neat import (
    CounterfactualProbeInput,
    FocalBrainUpdate,
    ProbeBehavior,
    PureNeatEvaluator,
)
from src.vision import SENSOR_CONTRACT
from src.persistence import SimulationPaths
from src.world import World


ROOT = Path(__file__).resolve().parents[1]


class DelayedNetwork:
    """Picklable benchmark wrapper for deliberately slow WHY activations."""

    def __init__(self, network: object, delay_seconds: float) -> None:
        self.network = network
        self.delay_seconds = delay_seconds

    def activate(self, inputs):
        time.sleep(self.delay_seconds)
        return self.network.activate(inputs)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def benchmark_world(
    why_enabled: bool,
    steps: int,
) -> tuple[list[float], int]:
    config = build_sim_config()
    config.random_seed = 19
    config.counterfactual_why.enabled = why_enabled
    config.persistence.enable_telemetry = False
    config.persistence.quick_save_interval_seconds = 0.0
    config.persistence.archive_save_interval_seconds = 0.0
    world = World(config, simulation_paths=SimulationPaths(ROOT))
    try:
        if world.creatures:
            world.select_creature_by_id(world.creatures[0].creature_id)
        for _ in range(30):
            world.update(World.FIXED_TIMESTEP)
        timings: list[float] = []
        for _ in range(steps):
            started = time.perf_counter()
            world.update(World.FIXED_TIMESTEP)
            timings.append((time.perf_counter() - started) * 1000.0)
        return timings, world.counterfactual_diagnostics.probe_requests
    finally:
        world.close()


def behavior_sample(index: int) -> BehaviorObservation:
    simulation_time = index / 10.0
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
        food_distance=150.0 - simulation_time * 12.0,
        food_relative_angle=max(0.0, 0.8 - simulation_time * 0.18),
    )


def benchmark_priority(
    evaluator: PureNeatEvaluator,
    inputs: tuple[float, ...],
    outputs: tuple[float, ...],
    slow_ms: float,
    sample_count: int,
) -> None:
    service = BehaviorObserverService(
        BehaviorObserverConfig(input_queue_capacity=max(8, sample_count)),
        CounterfactualWhyConfig(probe_queue_capacity=1),
    )
    delayed = PureNeatEvaluator(
        DelayedNetwork(evaluator.network, max(0.0, slow_ms) / 1000.0),
        evaluator.output_activations,
    )
    started = time.perf_counter()
    try:
        service.set_focus(1, 1)
        service.set_focal_brain(
            FocalBrainUpdate(1, 1, 1, pickle.dumps(delayed))
        )
        time.sleep(0.05)
        service.submit_why(
            CounterfactualProbeInput(
                creature_id=1,
                selection_generation=1,
                brain_revision=1,
                simulation_time=0.0,
                sensor_schema_version=SENSOR_CONTRACT.schema_version,
                behaviors=(
                    ProbeBehavior(
                        BehaviorKind.FOOD_APPROACH,
                        BoutStatus.ACTIVE,
                        1,
                        0.0,
                        target_id=1,
                    ),
                ),
                actual_inputs=inputs,
                actual_outputs=outputs,
                submitted_monotonic=time.monotonic(),
                target_visible=True,
                food_target_id=1,
                food_relative_angle=0.25,
            )
        )
        for index in range(sample_count):
            service.submit(behavior_sample(index))
            service.poll()
            time.sleep(0.001)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            service.poll()
            if (
                service.diagnostics.observations_processed >= sample_count
                and service.latest_why_snapshots
            ):
                break
            time.sleep(0.005)
        elapsed = time.perf_counter() - started
        behavior = service.diagnostics
        why = service.counterfactual_diagnostics
        print(
            f"Slow-WHY priority ({slow_ms:.1f} ms/eval): "
            f"{behavior.observations_processed}/{sample_count} behavior "
            f"samples in {elapsed * 1000.0:.1f} ms; "
            f"behavior drops {behavior.samples_dropped}; "
            f"WHY evals {why.evaluations_performed}, "
            f"superseded {why.probes_superseded}"
        )
    finally:
        service.close()


def run(
    activations: int,
    world_steps: int,
    slow_ms: float,
    priority_samples: int,
) -> None:
    config = build_sim_config()
    config.persistence.enable_telemetry = False
    world = World(config, simulation_paths=SimulationPaths(ROOT))
    try:
        world.update(World.FIXED_TIMESTEP)
        brain = next(iter(world.neat_controller.brains.values()))
        evaluator = PureNeatEvaluator.from_brain(brain)
        inputs = tuple(brain.last_inputs) or tuple(
            0.0 for _ in range(SENSOR_CONTRACT.input_count)
        )
        started = time.perf_counter()
        for _ in range(activations):
            evaluator.evaluate(inputs)
        elapsed = time.perf_counter() - started
        print("Counterfactual NEAT benchmark")
        print(
            f"Pure activation: {elapsed / activations * 1_000_000:.3f} us"
        )
        print(
            f"Representative nodes/connections: "
            f"{len(brain.genome.nodes)}/{len(brain.genome.connections)}"
        )
        factual_inputs = inputs
        factual_outputs = tuple(brain.last_outputs) or evaluator.evaluate(inputs)
    finally:
        world.close()

    disabled, _ = benchmark_world(False, world_steps)
    enabled, requests = benchmark_world(True, world_steps)
    disabled_median = statistics.median(disabled)
    enabled_median = statistics.median(enabled)
    degradation = (
        (enabled_median - disabled_median)
        / max(disabled_median, 1e-12)
        * 100.0
    )
    print(
        "Frame median disabled/enabled: "
        f"{disabled_median:.3f}/{enabled_median:.3f} ms"
    )
    print(
        "Frame p95 disabled/enabled: "
        f"{percentile(disabled, 0.95):.3f}/"
        f"{percentile(enabled, 0.95):.3f} ms"
    )
    print(f"Median degradation: {degradation:+.2f}%")
    print(f"Organic WHY probe requests: {requests}")
    if requests == 0:
        print(
            "No mapped bout emerged in this seeded interval; use the "
            "deterministic worker tests for loaded-probe priority coverage."
        )
    benchmark_priority(
        evaluator,
        factual_inputs,
        factual_outputs,
        slow_ms,
        priority_samples,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=int, default=20_000)
    parser.add_argument("--world-steps", type=int, default=180)
    parser.add_argument("--slow-ms", type=float, default=5.0)
    parser.add_argument("--priority-samples", type=int, default=30)
    args = parser.parse_args()
    if (
        args.activations < 1
        or args.world_steps < 1
        or args.slow_ms < 0.0
        or args.priority_samples < 1
    ):
        parser.error("benchmark counts must be positive and delay nonnegative")
    run(
        args.activations,
        args.world_steps,
        args.slow_ms,
        args.priority_samples,
    )


if __name__ == "__main__":
    main()
