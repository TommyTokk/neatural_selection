from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import neat

from src.action import Action
from src.vision import SensorSnapshot

@dataclass(slots=True)
class NeatBrain:
    genome_id: int
    genome: Any
    network: neat.nn.FeedForwardNetwork
    last_inputs: list[float] = field(default_factory=list)
    last_outputs: list[float] = field(default_factory=list)
    last_action: Action | None = None

    @classmethod
    def from_genome(cls, genome_id: int, genome: Any, config: neat.Config) -> NeatBrain:
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        return cls(
            genome_id=genome_id,
            genome=genome,
            network=network,
        )

    def decide(self, snapshot: SensorSnapshot) -> Action:
        self.last_inputs = snapshot.as_inputs()
        outputs = self.network.activate(self.last_inputs)
        self.last_outputs = list(outputs)

        accelerate = outputs[0] if len(outputs) > 0 else 0.0
        rotate = outputs[1] if len(outputs) > 1 else 0.0
        herding = outputs[2] if len(outputs) > 2 else 0.0

        self.last_action = Action(
            accelerate=accelerate,
            rotate=rotate,
            herding=herding,
        ).clamped()
        return self.last_action
