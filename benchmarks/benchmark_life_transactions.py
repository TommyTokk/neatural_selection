from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graphics import configure_graphics  # noqa: E402

configure_graphics()

from configs.sim_config import build_sim_config  # noqa: E402
from benchmarks.benchmark_flocking import force_benchmark_birth  # noqa: E402
from src.action import Action  # noqa: E402
from src.persistence import SimulationPaths  # noqa: E402
from src.world import ReproductionRequest, World  # noqa: E402


def percentile_95(samples: list[float]) -> float:
    ordered = sorted(samples)
    return ordered[max(0, ceil(len(ordered) * 0.95) - 1)]


def summarize(
    population: int,
    scenario: str,
    samples: list[float],
) -> None:
    print(
        f"{population},{scenario},{len(samples)},"
        f"{statistics.median(samples) * 1e3:.3f},"
        f"{percentile_95(samples) * 1e3:.3f},"
        f"{max(samples) * 1e3:.3f}"
    )


def fill_population(world: World, target: int) -> None:
    if len(world.creatures) >= target:
        return
    parent = world.creatures[0]
    world.total_biomass_energy += 10_000.0
    world.rt_neat.eligible_parent_ids = [parent.creature_id]
    world._last_actions[parent.creature_id] = Action(
        accelerate=0.0,
        rotate=0.0,
        want_reproduce=1.0,
        want_eat=0.0,
        reset_chronometer=0.0,
        want_grab=0.0,
        want_release=0.0,
    )
    while len(world.creatures) < target:
        parent.energy = world.config.metabolism.max_energy
        if not force_benchmark_birth(world, parent):
            raise RuntimeError("Could not fill transaction benchmark population.")


def make_world(population: int) -> World:
    config = build_sim_config()
    config.random_seed = 17
    config.persistence.enable_telemetry = False
    config.persistence.quick_save_interval_seconds = 0.0
    config.persistence.archive_save_interval_seconds = 0.0
    config.population.initial_creatures = min(population, 50)
    config.population.max_creatures = population
    config.food.initial_food_items = 300
    world = World(
        config,
        brain_initialization_seed=17,
        simulation_paths=SimulationPaths(ROOT),
    )
    fill_population(world, population)
    return world


def benchmark_idle_transactions(
    world: World,
    steps: int,
    warmups: int,
) -> list[float]:
    world._prepare_reproduction_requests = lambda: []
    world._prepare_nursing_requests = lambda _delta: []
    for _ in range(warmups):
        world._resolve_resource_transactions(World.FIXED_TIMESTEP)
    samples: list[float] = []
    for _ in range(steps):
        started = perf_counter()
        world._resolve_resource_transactions(World.FIXED_TIMESTEP)
        samples.append(perf_counter() - started)
    return samples


def benchmark_reproduction_staging(
    world: World,
    repeats: int,
    warmups: int,
) -> list[float]:
    parent = world.creatures[0]
    request = ReproductionRequest(
        parent=parent,
        eligibility_rank=0,
        reserved_energy_cost=world._reproduction_cost_for(parent),
    )
    for _ in range(warmups):
        world._stage_final_reproductions([request])
    samples: list[float] = []
    for _ in range(repeats):
        started = perf_counter()
        world._stage_final_reproductions([request])
        samples.append(perf_counter() - started)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark life-ledger transactions and reproduction staging."
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--staging-repeats", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=10)
    arguments = parser.parse_args()
    if arguments.steps <= 0 or arguments.staging_repeats <= 0:
        parser.error("benchmark repeat counts must be positive")
    if arguments.warmups < 0:
        parser.error("warmups must be nonnegative")

    print("population,scenario,samples,median_ms,p95_ms,max_ms")
    for population in (40, 55):
        world = make_world(population)
        try:
            summarize(
                population,
                "idle_transaction",
                benchmark_idle_transactions(
                    world,
                    arguments.steps,
                    arguments.warmups,
                ),
            )
            summarize(
                population,
                "reproduction_staging",
                benchmark_reproduction_staging(
                    world,
                    arguments.staging_repeats,
                    arguments.warmups,
                ),
            )
        finally:
            world.close()


if __name__ == "__main__":
    main()
