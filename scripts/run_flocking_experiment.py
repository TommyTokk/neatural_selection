from __future__ import annotations

import argparse
from dataclasses import asdict
from enum import Enum
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.sim_config import (  # noqa: E402
    SocialCompatibilityMode,
    build_sim_config,
)
from src.persistence import SimulationPaths  # noqa: E402
from src.world import World  # noqa: E402


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument(
        "--compatibility-mode",
        choices=[item.value for item in SocialCompatibilityMode],
    )
    parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction)
    parser.add_argument("--cohort", action=argparse.BooleanOptionalAction)
    parser.add_argument("--long-range", action=argparse.BooleanOptionalAction)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_directory.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "hourly").mkdir(exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    config = build_sim_config()
    config.random_seed = args.seed
    config.biome.seed = args.seed
    config.persistence.simulation_root_directory = str(output.parent)
    if args.compatibility_mode is not None:
        config.flocking.compatibility.mode = SocialCompatibilityMode(
            args.compatibility_mode
        )
    if args.cohort is not None:
        config.flocking.cohort_spawn.enabled = args.cohort
    if args.long_range is not None:
        config.flocking.long_range.enabled = args.long_range
    if args.benchmark is not None:
        config.flocking.benchmark.enabled = args.benchmark
    paths = SimulationPaths(output)
    _write_json(output / "configuration.json", asdict(config))
    _write_json(output / "seed.json", {"seed": args.seed})

    world = World(config, simulation_paths=paths)
    emergence_streak = 0.0
    longest_emergence_streak = 0.0
    last_sample_index = 0
    last_centroid_displacement = 0.0
    try:
        steps = max(0, round(args.duration / World.FIXED_TIMESTEP))
        for _ in range(steps):
            world.update(World.FIXED_TIMESTEP)
            sample_index = int(
                world.elapsed_time
                / config.flocking.telemetry.interval_seconds
                + 1e-9
            )
            if sample_index <= last_sample_index:
                continue
            last_sample_index = sample_index
            runtime_sample = world._last_flocking_runtime
            grouped_sample = [
                item
                for item in runtime_sample.values()
                if item.local_group_size >= 3
            ]
            grouped_fraction = (
                0.0
                if not world.creatures
                else len(grouped_sample) / len(world.creatures)
            )
            mean_heading_error = (
                0.0
                if not grouped_sample
                else sum(
                    item.observation.mean_heading_error
                    for item in grouped_sample
                )
                / len(grouped_sample)
            )
            qualifying_groups = [
                group
                for group in world._flocking_group_tracker.previous.values()
                if len(group.members) >= 3
            ]
            last_centroid_displacement = (
                0.0
                if not qualifying_groups
                else sum(
                    group.displacement for group in qualifying_groups
                )
                / len(qualifying_groups)
            )
            qualifies = (
                grouped_fraction >= 0.30
                and mean_heading_error <= 0.35
                and last_centroid_displacement >= 1.0
            )
            if qualifies:
                emergence_streak += (
                    config.flocking.telemetry.interval_seconds
                )
                longest_emergence_streak = max(
                    longest_emergence_streak,
                    emergence_streak,
                )
            else:
                emergence_streak = 0.0
        runtime = world._last_flocking_runtime
        grouped = [
            item
            for item in runtime.values()
            if item.local_group_size >= 3
        ]
        summary = {
            "compatibility_mode": (
                config.flocking.compatibility.mode.value
            ),
            "duration": world.elapsed_time,
            "population": len(world.creatures),
            "grouped_fraction": (
                0.0
                if not world.creatures
                else len(grouped) / len(world.creatures)
            ),
            "mean_heading_error": (
                0.0
                if not grouped
                else sum(
                    item.observation.mean_heading_error for item in grouped
                )
                / len(grouped)
            ),
            "benchmark_reward": sum(
                item.flocking_benchmark_reward
                for item in world.fitness.values()
            ),
            "mean_group_centroid_displacement": (
                last_centroid_displacement
            ),
            "longest_emergence_streak_seconds": (
                longest_emergence_streak
            ),
            "emergence_criterion_met": (
                longest_emergence_streak >= 60.0
            ),
            "brain_contract_reset_occurred": (
                world.brain_contract_reset_occurred
            ),
        }
        _write_json(output / "flocking_summary.json", summary)
        world.save_now()
        world.persistence_manager.flush()
    finally:
        world.close()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
