from __future__ import annotations

from pathlib import Path
import random
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.sim_config import build_sim_config  # noqa: E402
from src.action import Action  # noqa: E402
from src.flocking import accepted_counterfactual_contribution  # noqa: E402
from src.flocking_telemetry import (  # noqa: E402
    FlockingTelemetryAggregator,
    PersistentGroupTracker,
)
from src.persistence import SimulationPaths  # noqa: E402
from src.world import World  # noqa: E402


def force_benchmark_birth(world: World, parent: object) -> bool:
    """Stage and commit one thermodynamic child for benchmark setup."""
    request = world._reproduction_request_for(parent)
    staged, shadow, rng_state = world._stage_final_reproductions([request])
    if not staged:
        return False
    parent.energy = max(0.0, parent.energy - request.parent_investment)
    world._commit_staged_reproductions(staged, shadow, rng_state)
    return True


def fill_to_physical_capacity(world: World) -> None:
    """Create normal rtNEAT offspring until the configured physical cap."""
    if not world.creatures:
        return
    world.total_biomass_energy += 10_000.0
    while len(world.creatures) < world.config.population.max_creatures:
        created_child = False
        for parent in tuple(world.creatures):
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
            world._effective_actions[parent.creature_id] = (
                world._last_actions[parent.creature_id]
            )
            parent.energy = world.config.metabolism.max_energy
            if force_benchmark_birth(world, parent):
                created_child = True
                break
        if not created_child:
            raise RuntimeError(
                "Could not populate the benchmark to the physical cap."
            )
    for creature in world.creatures:
        creature.energy = world.config.metabolism.max_energy


def main() -> None:
    random.seed(11)
    config = build_sim_config()
    config.random_seed = 11
    config.flocking.long_range.enabled = True
    config.persistence.enable_telemetry = False
    config.persistence.quick_save_interval_seconds = 0.0
    config.persistence.archive_save_interval_seconds = 0.0
    # Food clustering is benchmarked separately from flocking throughput.
    config.food_clusters.cluster_spawn_share = 0.0
    # The physical cap is 55, while the shipped NEAT population contains 50
    # genomes. Fifty is therefore the largest valid fully controlled starting
    # population (the remaining five slots are reserved for runtime births).
    config.population.initial_creatures = min(
        config.population.max_creatures,
        50,
    )
    world = World(config, simulation_paths=SimulationPaths(ROOT))
    try:
        fill_to_physical_capacity(world)
        benchmark_population = len(world.creatures)
        iterations = 120
        started = perf_counter()
        for _ in range(iterations):
            world.update(World.FIXED_TIMESTEP)
        elapsed = perf_counter() - started
        if len(world.creatures) != benchmark_population:
            raise RuntimeError(
                "Benchmark population changed during the measured window."
            )
        print(
            "expanded sensing + control intent + counterfactual allocation: "
            f"{elapsed:.6f}s total, "
            f"{elapsed / iterations * 1000.0:.3f}ms/step "
            f"({iterations} steps, {benchmark_population} creatures)"
        )

        allocation_iterations = 100_000
        started = perf_counter()
        for _ in range(allocation_iterations):
            accepted_counterfactual_contribution(
                blended_request=(80.0, 35.0),
                neural_request=(65.0, -20.0),
                mandatory_avoidance=(-12.0, 4.0),
                remaining_budget=100.0,
            )
        allocation_elapsed = perf_counter() - started
        print(
            "counterfactual allocator: "
            f"{allocation_elapsed:.6f}s total, "
            f"{allocation_elapsed / allocation_iterations * 1e6:.3f}us/call"
        )

        tracker = PersistentGroupTracker(
            config.flocking.telemetry.persistence_overlap_threshold
        )
        group_iterations = 120
        started = perf_counter()
        sample = None
        for index in range(group_iterations):
            sample = tracker.sample(
                world.creatures,
                sim_time=world.elapsed_time + index,
                group_range=(
                    config.flocking.telemetry.group_detection_range
                ),
                minimum_compatibility=(
                    config.flocking.telemetry.minimum_group_compatibility
                ),
                compatibility=world.social_compatibility.compatibility,
                nearby=world._query_nearby_creatures,
            )
        group_elapsed = perf_counter() - started
        print(
            "connected-component group detection: "
            f"{group_elapsed:.6f}s total, "
            f"{group_elapsed / group_iterations * 1000.0:.3f}ms/sample"
        )

        aggregation_iterations = 10_000
        started = perf_counter()
        for _ in range(aggregation_iterations):
            FlockingTelemetryAggregator.aggregate(
                sim_time=world.elapsed_time,
                population_size=len(world.creatures),
                runtime=world._last_flocking_runtime,
                groups=sample,
            )
        aggregation_elapsed = perf_counter() - started
        print(
            "telemetry aggregation: "
            f"{aggregation_elapsed:.6f}s total, "
            f"{aggregation_elapsed / aggregation_iterations * 1e6:.3f}us/sample"
        )
    finally:
        world.close()


if __name__ == "__main__":
    main()
