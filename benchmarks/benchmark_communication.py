from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import floor, sqrt
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.sim_config import CommunicationConfig, PheromoneConfig
from src.communication import AcousticSignal, AcousticSystem, PheromoneSystem


@dataclass(frozen=True, slots=True)
class Timing:
    median_seconds: float
    minimum_seconds: float


def measure(function: Callable[[], object], warmups: int, repeats: int) -> Timing:
    for _ in range(warmups):
        function()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        samples.append(perf_counter() - start)
    return Timing(statistics.median(samples), min(samples))


def brute_force_signal(
    signals: list[AcousticSignal],
    receiver_id: int,
    position: tuple[float, float],
    maximum_range: float,
    hearing_threshold: float,
) -> tuple[int | None, float, int]:
    maximum_range_squared = maximum_range * maximum_range
    best_id: int | None = None
    best_strength = -1.0
    checks = 0
    for signal in signals:
        checks += 1
        if signal.emitter_id == receiver_id:
            continue
        dx = signal.position[0] - position[0]
        dy = signal.position[1] - position[1]
        distance_squared = dx * dx + dy * dy
        if distance_squared > maximum_range_squared:
            continue
        heard = signal.strength * (1.0 - sqrt(distance_squared) / maximum_range) ** 2
        if heard < hearing_threshold:
            continue
        if heard > best_strength or (
            heard == best_strength and (best_id is None or signal.emitter_id < best_id)
        ):
            best_id = signal.emitter_id
            best_strength = heard
    return best_id, max(0.0, best_strength), checks


def acoustic_scene(
    population: int,
    *,
    dense: bool,
    seed: int,
) -> tuple[list[AcousticSignal], list[tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    extent = 200.0 if dense else max(2_000.0, sqrt(population) * 500.0)
    emitter_positions = rng.uniform(-extent, extent, size=(population, 2))
    receiver_positions = rng.uniform(-extent, extent, size=(population, 2))
    strengths = rng.uniform(0.05, 1.0, size=population)
    tones = rng.uniform(-1.0, 1.0, size=population)
    signals = [
        AcousticSignal(
            index,
            (float(position[0]), float(position[1])),
            float(strength),
            float(tone),
        )
        for index, position, strength, tone in zip(
            range(population), emitter_positions, strengths, tones
        )
    ]
    receivers = [
        (float(position[0]), float(position[1])) for position in receiver_positions
    ]
    return signals, receivers


def benchmark_acoustics(warmups: int, repeats: int, quick: bool) -> None:
    print("\nAcoustic workloads")
    print(
        "scene,population,spatial_median_ms,spatial_min_ms,"
        "brute_median_ms,brute_min_ms,candidates_spatial,candidates_brute,speedup"
    )
    scenarios = [(100, False), (1_000, False), (10_000, False), (1_000, True)]
    if quick:
        scenarios = [(100, False), (1_000, False), (2_000, True)]
    for population, dense in scenarios:
        signals, receivers = acoustic_scene(
            population,
            dense=dense,
            seed=population + int(dense),
        )
        config = CommunicationConfig(
            acoustic_range=100.0,
            acoustic_min_emission_strength=0.05,
            acoustic_hearing_threshold=0.01,
        )
        system = AcousticSystem(config)
        rebuild = measure(lambda: system.replace_signals(signals), warmups, repeats)
        system.replace_signals(signals)
        brute_receiver_count = population if population <= 1_000 else 100
        brute_receivers = receivers[:brute_receiver_count]

        def spatial_pass() -> int:
            checks = 0
            for offset, receiver in enumerate(receivers):
                system.sense(population + offset, receiver, 0.0)
                checks += system.last_candidate_checks
            return checks

        def brute_pass() -> int:
            checks = 0
            for offset, receiver in enumerate(brute_receivers):
                checks += brute_force_signal(
                    signals,
                    population + offset,
                    receiver,
                    100.0,
                    0.01,
                )[2]
            return checks

        for offset, receiver in enumerate(brute_receivers[:10]):
            spatial = system.sense_with_debug(population + offset, receiver, 0.0)
            brute_id, brute_strength, _checks = brute_force_signal(
                signals,
                population + offset,
                receiver,
                100.0,
                0.01,
            )
            if spatial.debug.source_id != brute_id or not np.isclose(
                spatial.observation.strength,
                brute_strength,
            ):
                raise RuntimeError("Spatial and brute-force acoustic results differ.")

        spatial_timing = measure(spatial_pass, warmups, repeats)
        brute_timing = measure(brute_pass, warmups, repeats)
        spatial_checks = spatial_pass()
        brute_checks = brute_pass()
        normalized_brute = (
            brute_timing.median_seconds * population / brute_receiver_count
        )
        speedup = normalized_brute / spatial_timing.median_seconds
        scene = "dense" if dense else "sparse"
        print(
            f"{scene},{population},{spatial_timing.median_seconds * 1e3:.3f},"
            f"{spatial_timing.minimum_seconds * 1e3:.3f},"
            f"{brute_timing.median_seconds * 1e3:.3f},"
            f"{brute_timing.minimum_seconds * 1e3:.3f},"
            f"{spatial_checks},{brute_checks},{speedup:.2f}x"
        )
        print(
            f"  index rebuild median/min: {rebuild.median_seconds * 1e3:.3f}/"
            f"{rebuild.minimum_seconds * 1e3:.3f} ms; brute receivers: "
            f"{brute_receiver_count}"
        )


def benchmark_pheromones(warmups: int, repeats: int, quick: bool) -> None:
    print("\nPheromone workloads")
    rng = np.random.default_rng(4421)
    deposit_count = 2_000 if quick else 10_000
    positions = rng.uniform(0.0, 1_000.0, size=(deposit_count, 2))
    colors = rng.uniform(0.0, 0.001, size=(deposit_count, 3))
    config = PheromoneConfig(diffusion_coefficient=10.0, max_concentration=100.0)

    def scalar_deposit() -> None:
        system = PheromoneSystem(config, 128, 128, (0.0, 0.0, 1_000.0, 1_000.0))
        for position, color in zip(positions, colors):
            system.deposit(
                float(position[0]), float(position[1]), color,
            )

    def batch_deposit() -> None:
        system = PheromoneSystem(config, 128, 128, (0.0, 0.0, 1_000.0, 1_000.0))
        system.deposit_many(positions, colors)

    grid_x = positions[:, 0] / 1_000.0 * 127.0
    grid_y = positions[:, 1] / 1_000.0 * 127.0
    column0 = np.floor(grid_x).astype(np.intp)
    row0 = np.floor(grid_y).astype(np.intp)
    column1 = np.minimum(column0 + 1, 127)
    row1 = np.minimum(row0 + 1, 127)
    u = grid_x - column0
    v = grid_y - row0
    add_at_indices = np.concatenate(
        (
            row0 * 128 + column0,
            row0 * 128 + column1,
            row1 * 128 + column0,
            row1 * 128 + column1,
        )
    )
    add_at_weights = (
        (1.0 - u) * (1.0 - v),
        u * (1.0 - v),
        (1.0 - u) * v,
        u * v,
    )

    def add_at_deposit() -> None:
        rgb_grid = np.zeros((128 * 128, 3), dtype=np.float32)
        np.add.at(
            rgb_grid,
            add_at_indices,
            np.concatenate(tuple(colors * weight[:, None] for weight in add_at_weights)),
        )

    scalar = measure(scalar_deposit, warmups, repeats)
    batch = measure(batch_deposit, warmups, repeats)
    add_at = measure(add_at_deposit, warmups, repeats)
    print(
        f"deposits={deposit_count}: scalar median/min "
        f"{scalar.median_seconds * 1e3:.3f}/{scalar.minimum_seconds * 1e3:.3f} ms; "
        f"batch {batch.median_seconds * 1e3:.3f}/{batch.minimum_seconds * 1e3:.3f} ms; "
        f"np.add.at {add_at.median_seconds * 1e3:.3f}/"
        f"{add_at.minimum_seconds * 1e3:.3f} ms; "
        f"batch speedup {scalar.median_seconds / batch.median_seconds:.2f}x"
    )

    sensor_count = 2_000 if quick else 10_000
    sensor_positions = rng.uniform(0.0, 1_000.0, size=(sensor_count, 3, 2))
    sensing_system = PheromoneSystem(
        config,
        128,
        128,
        (0.0, 0.0, 1_000.0, 1_000.0),
    )
    sensing_system.field[:] = rng.random(sensing_system.field.shape)
    sensing = measure(
        lambda: sensing_system.sense_many(sensor_positions),
        warmups,
        repeats,
    )
    print(
        f"sense_many N={sensor_count}: median/min "
        f"{sensing.median_seconds * 1e3:.3f}/{sensing.minimum_seconds * 1e3:.3f} ms"
    )

    sizes = [64, 128] if quick else [64, 128, 256, 512]
    for size in sizes:
        system = PheromoneSystem(
            PheromoneConfig(diffusion_coefficient=390.0, decay_rate=0.08),
            size,
            size,
            (-1_600.0, -1_100.0, 1_600.0, 1_100.0),
        )
        system.field[size // 2, size // 2] = 1.0
        timing = measure(lambda: system.advance(1.0 / 60.0), warmups, repeats)
        print(
            f"diffusion {size}x{size}, RGB: median/min "
            f"{timing.median_seconds * 1e3:.3f}/{timing.minimum_seconds * 1e3:.3f} ms"
        )

    subdivision = PheromoneSystem(
        PheromoneConfig(diffusion_coefficient=1.0, decay_rate=0.08),
        128,
        128,
        (0.0, 0.0, 127.0, 127.0),
    )
    subdivision.field[64, 64] = 1.0
    stable_dt = min(1.0 / 60.0, subdivision.maximum_stable_timestep)
    timing = measure(lambda: subdivision.advance(stable_dt), warmups, repeats)
    print(
        f"fixed stable timestep: median/min "
        f"{timing.median_seconds * 1e3:.3f}/{timing.minimum_seconds * 1e3:.3f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run smaller validation workloads while retaining each benchmark type.",
    )
    arguments = parser.parse_args()
    warmups = 1 if arguments.quick else 2
    repeats = 3 if arguments.quick else 7
    benchmark_acoustics(warmups, repeats, arguments.quick)
    benchmark_pheromones(warmups, repeats, arguments.quick)


if __name__ == "__main__":
    main()
