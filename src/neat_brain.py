from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

import neat

from src.action import (
    ACTION_OUTPUT_COUNT,
    NEUTRAL_NETWORK_OUTPUT,
    Action,
    signed_output,
)
from src.vision import SensorSnapshot

DEFAULT_ACTION_OUTPUTS = [NEUTRAL_NETWORK_OUTPUT] * ACTION_OUTPUT_COUNT


@dataclass(slots=True)
class NeatBrain:
    genome_id: int
    genome: Any
    network: neat.nn.FeedForwardNetwork
    output_activations: list[str] = field(default_factory=list)
    last_inputs: list[float] = field(default_factory=list)
    last_outputs: list[float] = field(default_factory=list)
    last_action: Action | None = None

    @classmethod
    def from_genome(cls, genome_id: int, genome: Any, config: neat.Config) -> NeatBrain:
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        output_activations = cls._output_activations_for(genome, config)
        return cls(
            genome_id=genome_id,
            genome=genome,
            network=network,
            output_activations=output_activations,
        )

    def decide(self, snapshot: SensorSnapshot) -> Action:
        self.last_inputs = snapshot.as_inputs()
        raw_outputs = self.network.activate(self.last_inputs)
        outputs = self._normalize_outputs(raw_outputs)
        self.last_outputs = outputs

        self.last_action = Action(
            accelerate=self._acceleration_action_output(outputs[0]),
            rotate=self._signed_action_output(outputs[1]),
        ).clamped()
        return self.last_action

    def _normalize_outputs(self, raw_outputs: Any) -> list[float]:
        try:
            output_values = list(raw_outputs)
        except TypeError:
            return DEFAULT_ACTION_OUTPUTS.copy()

        normalized = [
            self._safe_output(value, self._output_activation(index))
            for index, value in enumerate(output_values[:ACTION_OUTPUT_COUNT])
        ]

        missing_outputs = ACTION_OUTPUT_COUNT - len(normalized)
        if missing_outputs > 0:
            normalized.extend(DEFAULT_ACTION_OUTPUTS[:missing_outputs])

        return normalized

    def _safe_output(self, value: Any, activation: str | None) -> float:
        try:
            output = float(value)
        except (TypeError, ValueError):
            return NEUTRAL_NETWORK_OUTPUT

        if not isfinite(output):
            return NEUTRAL_NETWORK_OUTPUT

        if activation in {"tanh", "sin"}:
            return (self._clamp(output, -1.0, 1.0) + 1.0) * 0.5

        if activation in {"relu", "abs"}:
            return NEUTRAL_NETWORK_OUTPUT + self._clamp(output, 0.0, 1.0) * 0.5

        return max(0.0, min(1.0, output))

    def _output_activation(self, index: int) -> str | None:
        if index >= len(self.output_activations):
            return None
        return self.output_activations[index]

    @staticmethod
    def _output_activations_for(genome: Any, config: neat.Config) -> list[str]:
        output_keys = list(config.genome_config.output_keys)
        activations: list[str] = []
        nodes = getattr(genome, "nodes", {})

        for key in output_keys[:ACTION_OUTPUT_COUNT]:
            node = nodes.get(key)
            activation = getattr(node, "activation", None)
            activations.append(
                str(activation).strip().lower() if activation is not None else ""
            )

        return activations

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def _signed_action_output(self, value: float) -> float:
        return signed_output(value)

    def _acceleration_action_output(self, value: float) -> float:
        return max(0.0, self._signed_action_output(value))
