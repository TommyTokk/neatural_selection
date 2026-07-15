from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.sim_config import CommunicationConfig
from src.communication import AcousticSignal, AcousticSystem, PheromoneSystem


def run_workload(population: int = 2_000, ticks: int = 30) -> None:
    rng = np.random.default_rng(982451653)
    config = CommunicationConfig(
        acoustic_range=100.0,
        acoustic_hearing_threshold=0.01,
        pheromone_max_concentration=100.0,
    )
    acoustics = AcousticSystem(config)
    pheromones = PheromoneSystem(
        config,
        128,
        128,
        (-1_600.0, -1_100.0, 1_600.0, 1_100.0),
    )
    positions = rng.uniform(
        (-1_600.0, -1_100.0),
        (1_600.0, 1_100.0),
        size=(population, 2),
    )
    sensor_offsets = np.array(((0.0, 0.0), (48.0, 24.0), (48.0, -24.0)))
    sensor_positions = positions[:, None, :] + sensor_offsets[None, :, :]
    strengths = rng.uniform(0.05, 1.0, size=population)
    tones = rng.uniform(-1.0, 1.0, size=population)
    deposits = rng.uniform(0.0, 0.001, size=population)

    for tick in range(ticks):
        signals = [
            AcousticSignal(
                emitter_id=index,
                position=(float(position[0]), float(position[1])),
                strength=float(strengths[index]),
                tone=float(tones[index]),
            )
            for index, position in enumerate(positions)
        ]
        acoustics.replace_signals(signals)
        for index, position in enumerate(positions):
            acoustics.sense(
                population + index,
                (float(position[0]), float(position[1])),
                0.0,
            )
        pheromones.deposit_many(positions, deposits, deposits)
        pheromones.sense_many(sensor_positions)
        pheromones.accumulate(1.0 / 60.0)
        positions[:, 0] += 0.01 * ((tick % 3) - 1)


if __name__ == "__main__":
    run_workload()
