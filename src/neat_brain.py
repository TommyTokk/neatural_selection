from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import neat

from src.action import Action
from src.vision import SensorSnapshot

@dataclass(slots=True)
class NeatBrain:
    genome_id: int
    genome: Any
    network: neat.nn.FeedForwardNetwork

    @classmethod
    def from_genome(cls, genome_id: int, genome: Any, config: neat.Config) -> NeatBrain:
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        return cls(
            genome_id=genome_id,
            genome=genome,
            network=network,
        )

    def decide(self, snapshot: SensorSnapshot) -> Action:
        outputs = self.network.activate(snapshot.as_inputs())

        accelerate = outputs[0] if len(outputs) > 0 else 0.0
        rotate = outputs[1] if len(outputs) > 1 else 0.0
        herding = outputs[2] if len(outputs) > 2 else 0.0

        return Action(
            accelerate=accelerate,
            rotate=rotate,
            herding=herding,
        ).clamped()