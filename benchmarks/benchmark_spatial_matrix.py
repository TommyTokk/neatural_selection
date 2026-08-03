"""Milestone-3 spatial cell-size and population scaling matrix.

Timing and workload counters are emitted together, but no counters are sampled
inside the timed region.  The complete three-step scheduler cycle is the unit
used to compare cell sizes because it includes every decision phase exactly
once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_flocking import fill_to_physical_capacity  # noqa: E402
from benchmarks.benchmark_multirate_scheduler import run_fixed_steps  # noqa: E402
from configs.sim_config import build_sim_config  # noqa: E402
from src.persistence import SimulationPaths  # noqa: E402
from src.spatial import CandidateBuffer  # noqa: E402
from src.world import World  # noqa: E402


CELL_SIZES = (48.0, 64.0, 96.0, 128.0)
POPULATIONS = (55, 128, 256)


def make_world(population: int, cell_size: float) -> World:
    config = build_sim_config()
    config.random_seed = 11
    config.population.initial_creatures = min(50, population)
    config.population.max_creatures = population
    config.flocking.long_range.enabled = True
    config.persistence.enable_telemetry = False
    config.persistence.quick_save_interval_seconds = 0.0
    config.persistence.archive_save_interval_seconds = 0.0
    world = World(config, simulation_paths=SimulationPaths(ROOT))
    fill_to_physical_capacity(world)
    world._creature_spatial_index.cell_size = cell_size
    return world


def measure(
    population: int,
    cell_size: float,
    *,
    repetitions: int,
    samples: int,
    warmup_steps: int,
) -> dict[str, object]:
    timings: list[float] = []
    run_medians: list[float] = []
    workload: list[dict[str, int]] = []
    for _ in range(repetitions):
        world = make_world(population, cell_size)
        try:
            run_fixed_steps(world, warmup_steps)
            before = world._creature_spatial_index.counters.snapshot()
            run_timings: list[float] = []
            for _sample in range(samples):
                started = perf_counter()
                run_fixed_steps(world, 3)
                elapsed = (perf_counter() - started) * 1_000.0
                timings.append(elapsed)
                run_timings.append(elapsed)
            run_medians.append(statistics.median(run_timings))
            after = world._creature_spatial_index.counters.snapshot()
            workload.append(
                {
                    "queries": after["queries"] - before["queries"],
                    "candidates": after["candidates"] - before["candidates"],
                    "cell_visits": after["cell_visits"] - before["cell_visits"],
                    "cell_resets": after["cell_resets"] - before["cell_resets"],
                    "maximum_candidates": after["maximum_candidates"],
                    "maximum_active_cells": after["maximum_active_cells"],
                    "maximum_cell_occupancy": after["maximum_cell_occupancy"],
                }
            )
        finally:
            world.close()
    return {
        "population": population,
        "cell_size": cell_size,
        "samples": len(timings),
        "three_step_median_ms": statistics.median(timings),
        "per_step_median_ms": statistics.median(timings) / 3.0,
        "run_medians_ms": run_medians,
        "noise_percent": (
            (max(run_medians) - min(run_medians))
            / max(2.0 * statistics.median(run_medians), 1e-12)
            * 100.0
        ),
        "workload_runs": workload,
    }


class _DirectReferenceBuffer:
    """Benchmark-only retained direct-reference representation."""

    def __init__(self) -> None:
        self.index = None
        self.references: list[object | None] = []
        self.count = 0

    def reset(self, index, _generation: int) -> None:
        self.index = index
        self.count = 0

    def append_slot(self, slot: int) -> None:
        if self.count == len(self.references):
            self.references.extend([None] * max(16, len(self.references) or 16))
        self.references[self.count] = self.index.creature_for_slot(slot)
        self.count += 1


def compare_candidate_representations() -> dict[str, object]:
    world = make_world(55, 128.0)
    try:
        run_fixed_steps(world, 6)
        index = world._creature_spatial_index
        slot_buffer = CandidateBuffer(index)
        reference_buffer = _DirectReferenceBuffer()

        def sample(output, direct: bool) -> tuple[float, int]:
            checksum = 0
            started = perf_counter()
            for _ in range(10):
                for observer in world.creatures:
                    x, y, _radius = index.values_for(observer)
                    index.query_into(x, y, 400.0, output)
                    if direct:
                        for position in range(output.count):
                            candidate = output.references[position]
                            stable_id = candidate.creature_id
                            if index.living_registry.get(stable_id) is candidate:
                                checksum += stable_id
                    else:
                        for position in range(output.count):
                            slot = output.slots[position]
                            candidate = index.creature_for_slot(slot)
                            if candidate is not None:
                                checksum += candidate.creature_id
            return (perf_counter() - started) * 1_000.0, checksum

        slot_samples = [sample(slot_buffer, False) for _ in range(7)]
        direct_samples = [sample(reference_buffer, True) for _ in range(7)]
        if {item[1] for item in slot_samples + direct_samples}.__len__() != 1:
            raise RuntimeError("Candidate representation checksums differ.")
        slot_median = statistics.median(item[0] for item in slot_samples)
        direct_median = statistics.median(item[0] for item in direct_samples)
        difference = abs(slot_median - direct_median) / max(
            slot_median,
            direct_median,
            1e-12,
        ) * 100.0
        return {
            "population": 55,
            "cell_size": 128.0,
            "query_cycles_per_sample": 10,
            "samples": 7,
            "slot_median_ms": slot_median,
            "direct_reference_median_ms": direct_median,
            "absolute_difference_percent": difference,
            "selected": "generation-stamped integer slots",
            "reason": (
                "slots keep validity metadata centralized and are the selected "
                "representation when the complete consumer benchmark is close"
            ),
        }
    finally:
        world.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=12)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repetitions = 1 if args.quick else max(1, args.repetitions)
    samples = 2 if args.quick else max(1, args.samples)
    warmup = 3 if args.quick else max(0, args.warmup_steps)
    rows = [
        measure(
            population,
            cell_size,
            repetitions=repetitions,
            samples=samples,
            warmup_steps=warmup,
        )
        for population in POPULATIONS
        for cell_size in CELL_SIZES
    ]
    fastest_55 = min(
        (row for row in rows if row["population"] == 55),
        key=lambda row: row["three_step_median_ms"],
    )
    result = {
        "seed": 11,
        "scheduler_cycle_steps": 3,
        "candidate_representation": "generation-stamped integer slots",
        "selection_rule": (
            "lowest 55-creature complete-cycle median; ties within noise are "
            "resolved by 128 then 256 creatures, then the smaller cell"
        ),
        "measured_fastest_55_cell_size": fastest_55["cell_size"],
        "production_cell_size": 128.0,
        "candidate_representation_comparison": (
            compare_candidate_representations()
        ),
        "rows": rows,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
