from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

import neat
from neat.graphs import required_for_output

from src.action import (
    ACTION_OUTPUT_COUNT,
    ACTION_OUTPUT_NAMES,
    NEUTRAL_NETWORK_OUTPUT,
    Action,
    signed_output,
)
from src.vision import SENSOR_INPUT_NAMES, SensorSnapshot

DEFAULT_ACTION_OUTPUTS = [
    *([NEUTRAL_NETWORK_OUTPUT] * 8),
    *([0.0] * (ACTION_OUTPUT_COUNT - 8)),
]


@dataclass(frozen=True, slots=True)
class SensorUsage:
    input_name: str
    current_value: float
    has_enabled_path: bool
    reachable_action_outputs: tuple[str, ...]


@dataclass(slots=True)
class NeatBrain:
    """
    Represents a NEAT brain for a creature, encapsulating its genome, neural network,
    and decision-making capabilities based on sensor inputs.
    """
    genome_id: int # Genome ID for the NEAT brain
    genome: Any # Genome object representing the neural network structure
    network: neat.nn.FeedForwardNetwork # Neural network created from the genome
    output_activations: list[str] = field(default_factory=list)# List of activation functions for each output node
    last_inputs: list[float] = field(default_factory=list)# Last sensor inputs received by the brain
    last_outputs: list[float] = field(default_factory=list)# Last outputs produced by the neural network
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
        """
        Decide on an action based on the current sensor snapshot.
        This method processes the sensor inputs through the neural network and
        normalizes the outputs to produce a valid action for the creature.
        
        Args:
            snapshot (SensorSnapshot): The current sensor snapshot of the creature's environment.

        Returns:
            Action: The action decided by the neural network based on the sensor inputs.
        """

        # Store the last inputs from the sensor snapshot
        
        # Convert the sensor snapshot to a list of inputs for the neural network
        self.last_inputs = snapshot.as_inputs()

        # Activate the neural network with the last inputs to get raw outputs
        raw_outputs = self.network.activate(self.last_inputs)

        # Normalize the raw outputs to ensure they are within valid ranges for actions
        outputs = self._normalize_outputs(raw_outputs)
        
        # Store the last outputs for reference
        self.last_outputs = outputs

        # Create an Action object based on the normalized outputs and clamp its values to ensure they are within valid ranges
        # The Action object is then stored as the last action taken by the brain.
        self.last_action = Action(
            accelerate=self._signed_action_output(outputs[0]),#  Normalize the first output for acceleration
            rotate=self._signed_action_output(outputs[1]), # Normalize the second output for rotation
            want_reproduce=outputs[2], # Use the third output directly for reproduction desire
            want_eat=outputs[3], # Use the fourth output directly for eating desire
            reset_chronometer=outputs[4], # Use the fifth output directly for chronometer reset desire
            want_grab=outputs[5], # Use the sixth output directly for grab desire
            want_release=outputs[6], # Use the seventh output directly for release desire
            want_nurse=outputs[7],# Use the eighth output directly for nursing desire
            flee_panic_intensity=outputs[8],# Use the ninth output directly for panic intensity in fleeing
            weight_separation=outputs[9],# Use the tenth output directly for separation weight in flocking behavior
            weight_alignment=outputs[10],# Use the eleventh output directly for alignment weight in flocking behavior
            weight_cohesion=outputs[11],# Use the twelfth output directly for cohesion weight in flocking behavior
            emit_sound=self._positive_action_output(outputs[12]),
            sound_tone=self._signed_action_output(outputs[13]),
            emit_trail_pheromone=self._positive_action_output(outputs[14]),
            emit_alarm_pheromone=self._positive_action_output(outputs[15]),
        ).clamped()
        return self.last_action

    def sensor_usage(
        self,
        input_keys: list[int] | tuple[int, ...],
        output_keys: list[int] | tuple[int, ...],
    ) -> tuple[SensorUsage, ...]:
        """Report which live inputs can reach actions through enabled genes."""
        enabled_connections = [
            connection.key
            for connection in self.genome.connections.values()
            if connection.enabled
        ]
        input_key_set = set(input_keys)
        reachable_by_input: dict[int, set[int]] = {
            input_key: set() for input_key in input_keys
        }
        for output_key in output_keys:
            required_nodes = required_for_output(
                input_keys,
                [output_key],
                enabled_connections,
            )
            for source, target in enabled_connections:
                if source in input_key_set and target in required_nodes:
                    reachable_by_input[source].add(output_key)

        output_names = {
            key: (
                ACTION_OUTPUT_NAMES[index]
                if index < len(ACTION_OUTPUT_NAMES)
                else str(key)
            )
            for index, key in enumerate(output_keys)
        }
        result: list[SensorUsage] = []
        for index, input_key in enumerate(input_keys):
            ordered_outputs = tuple(
                output_names[key]
                for key in output_keys
                if key in reachable_by_input[input_key]
            )
            result.append(
                SensorUsage(
                    input_name=(
                        SENSOR_INPUT_NAMES[index]
                        if index < len(SENSOR_INPUT_NAMES)
                        else str(input_key)
                    ),
                    current_value=(
                        self.last_inputs[index]
                        if index < len(self.last_inputs)
                        else 0.0
                    ),
                    has_enabled_path=bool(ordered_outputs),
                    reachable_action_outputs=ordered_outputs,
                )
            )
        return tuple(result)

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
            normalized.extend(DEFAULT_ACTION_OUTPUTS[len(normalized) :])

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
        return self._clamp(signed_output(value), -1.0, 1.0)

    def _positive_action_output(self, value: float) -> float:
        return self._clamp(signed_output(value), 0.0, 1.0)
