from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

import neat

from src.action import ACTION_OUTPUT_COUNT, Action
from src.vision import SensorSnapshot

SIGMOID_NEUTRAL_OUTPUT = 0.5
DEFAULT_ACTION_OUTPUTS = [SIGMOID_NEUTRAL_OUTPUT] * ACTION_OUTPUT_COUNT


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
        raw_outputs = self.network.activate(self.last_inputs)
        outputs = self._normalize_outputs(raw_outputs)
        self.last_outputs = outputs
        action_outputs = [
            self._signed_action_output(output)
            for output in outputs
        ]

        self.last_action = Action(
            accelerate=action_outputs[0],
            rotate=action_outputs[1],
        ).clamped()
        return self.last_action

    def _normalize_outputs(self, raw_outputs: Any) -> list[float]:
        try:
            output_values = list(raw_outputs)
        except TypeError:
            return DEFAULT_ACTION_OUTPUTS.copy()

        normalized = [
            self._safe_output(value)
            for value in output_values[:ACTION_OUTPUT_COUNT]
        ]

        missing_outputs = ACTION_OUTPUT_COUNT - len(normalized)
        if missing_outputs > 0:
            normalized.extend(DEFAULT_ACTION_OUTPUTS[:missing_outputs])

        return normalized

    def _safe_output(self, value: Any) -> float:
        try:
            output = float(value)
        except (TypeError, ValueError):
            return SIGMOID_NEUTRAL_OUTPUT

        if not isfinite(output):
            return SIGMOID_NEUTRAL_OUTPUT

        return max(0.0, min(1.0, output))

    def _signed_action_output(self, value: float) -> float:
        return (value - SIGMOID_NEUTRAL_OUTPUT) * 2.0
