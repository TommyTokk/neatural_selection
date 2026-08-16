from __future__ import annotations

import argparse
from math import ceil, hypot
from pathlib import Path
import statistics
import sys
from time import perf_counter
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.sim_config import build_sim_config
from src.fitness import CreatureFitness
from src.persistence import SimulationPaths
from src.world import World


def make_world(population: int) -> World:
    config = build_sim_config()
    config.persistence.enable_telemetry = False
    config.population.initial_creatures = min(population, 50)
    config.population.max_creatures = population
    config.food.initial_food_items = 0
    world = World(
        config,
        simulation_paths=SimulationPaths(
            Path("/tmp/neat_game_of_life_neighbor_benchmark")
        ),
    )
    for creature_id in range(len(world.creatures) + 1, population + 1):
        creature = world._spawn_creature(creature_id)
        world.creatures.append(creature)
        world.fitness[creature_id] = CreatureFitness()
        world._chronometers[creature_id] = 0.0
    for index, creature in enumerate(world.creatures):
        pair_index = index // 2
        creature.body.position = (
            250.0 + (pair_index % 6) * 400.0 + (index % 2) * 36.0,
            250.0 + (pair_index // 6) * 350.0,
        )
        world.space.reindex_shape(creature.shape)
    world.set_simulation_speed(5.0)
    return world


def full_scan_avoidance(
    world: World,
    creature: object,
    max_force: float,
) -> tuple[float, float]:
    margin = max(0.0, world.config.action.collision_avoidance_margin)
    scale = max(0.0, world.config.action.collision_avoidance_force_scale)
    if max_force <= 0.0 or scale <= 0.0:
        return 0.0, 0.0
    position = creature.position
    avoidance_x = avoidance_y = 0.0
    for neighbor in world.creatures:
        if neighbor is creature:
            continue
        away_x = position[0] - neighbor.position[0]
        away_y = position[1] - neighbor.position[1]
        distance = hypot(away_x, away_y)
        safe_distance = max(
            1e-9,
            creature.radius + neighbor.radius + margin,
        )
        if distance >= safe_distance:
            continue
        if distance <= 1e-12:
            unit_x = (
                1.0 if creature.creature_id < neighbor.creature_id else -1.0
            )
            unit_y = 0.0
        else:
            unit_x, unit_y = away_x / distance, away_y / distance
        strength = (safe_distance - distance) / safe_distance
        avoidance_x += unit_x * strength
        avoidance_y += unit_y * strength
    magnitude = hypot(avoidance_x, avoidance_y)
    if magnitude <= 1e-12:
        return 0.0, 0.0
    force_magnitude = min(max_force, max_force * scale * min(1.0, magnitude))
    return (
        avoidance_x / magnitude * force_magnitude,
        avoidance_y / magnitude * force_magnitude,
    )


def measure_updates(
    world: World,
    warmups: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmups):
        world.update(1.0 / 30.0)
    samples: list[float] = []
    for _ in range(repeats):
        started = perf_counter()
        world.update(1.0 / 30.0)
        samples.append(perf_counter() - started)
    return samples


def percentile_95(samples: list[float]) -> float:
    ordered = sorted(samples)
    return ordered[max(0, ceil(len(ordered) * 0.95) - 1)]


def summarize(label: str, samples: list[float]) -> None:
    mean = statistics.fmean(samples)
    print(
        f"{label},{mean * 1e3:.3f},{statistics.median(samples) * 1e3:.3f},"
        f"{percentile_95(samples) * 1e3:.3f},{1.0 / mean:.1f}"
    )


def spatial_candidate_checks(world: World) -> int:
    margin = max(0.0, world.config.action.collision_avoidance_margin)
    world._cache_creature_spatial_state()
    try:
        per_step = sum(
            len(
                world._query_nearby_creatures(
                    creature,
                    creature.shape.radius + margin,
                )
            )
            for creature in world.creatures
        )
    finally:
        world._creature_spatial_state = None
    return per_step * World.MAX_FRAME_STEPS


def run(population: int, warmups: int, repeats: int) -> None:
    old_world = make_world(population)
    old_world._collision_avoidance_force = MethodType(
        full_scan_avoidance,
        old_world,
    )
    try:
        old_samples = measure_updates(old_world, warmups, repeats)
    finally:
        old_world.close()

    new_world = make_world(population)
    try:
        spatial_checks = spatial_candidate_checks(new_world)
        new_samples = measure_updates(new_world, warmups, repeats)
    finally:
        new_world.close()

    print("implementation,mean_ms,median_ms,p95_ms,projected_fps")
    summarize("full_scan", old_samples)
    summarize("point_query", new_samples)
    old_mean = statistics.fmean(old_samples)
    new_mean = statistics.fmean(new_samples)
    print(f"speedup,{old_mean / new_mean:.2f}x")
    print(
        f"initial_candidate_checks_per_rendered_frame,"
        f"full_scan={population * (population - 1) * World.MAX_FRAME_STEPS},"
        f"point_query={spatial_checks}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=60)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        max(1, arguments.population),
        1 if arguments.quick else max(0, arguments.warmups),
        8 if arguments.quick else max(1, arguments.repeats),
    )
