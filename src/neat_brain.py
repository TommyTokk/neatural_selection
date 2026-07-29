from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, tanh
from typing import Any

import neat
from neat.graphs import required_for_output

from src.action import (
    ACTION_OUTPUT_COUNT,
    ACTION_OUTPUT_NAMES,
    Action,
    BrainOutputIndex,
)
from src.vision import SENSOR_INPUT_NAMES, SensorSnapshot

DEFAULT_CENTERED_OUTPUTS = [0.0] * ACTION_OUTPUT_COUNT


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
    genome_id: int  # Genome ID for the NEAT brain
    genome: Any  # Genome object representing the neural network structure
    network: neat.nn.FeedForwardNetwork  # Network created from the genome
    output_activations: list[str] = field(default_factory=list)
    last_inputs: list[float] = field(default_factory=list)
    # Last activation-aware outputs, centered independently in [-1, 1].
    last_outputs: list[float] = field(default_factory=list)
    last_action: Action | None = None
    last_input_names: tuple[str, ...] = SENSOR_INPUT_NAMES

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

        self.last_inputs = snapshot.as_inputs()
        self.last_input_names = snapshot.sensor_contract.input_names
        raw_outputs = self.network.activate(self.last_inputs)
        centered_outputs = self._normalize_outputs(raw_outputs)
        self.last_outputs = centered_outputs

        self.last_action = Action(
            accelerate=centered_outputs[BrainOutputIndex.ACCELERATE],
            rotate=centered_outputs[BrainOutputIndex.ROTATE],
            want_reproduce=self._positive_action_output(
                centered_outputs[BrainOutputIndex.REPRODUCE]
            ),
            want_eat=self._positive_action_output(
                centered_outputs[BrainOutputIndex.EAT]
            ),
            reset_chronometer=self._positive_action_output(
                centered_outputs[BrainOutputIndex.RESET_CHRONOMETER]
            ),
            want_grab=self._positive_action_output(
                centered_outputs[BrainOutputIndex.GRAB_FOOD]
            ),
            want_release=self._positive_action_output(
                centered_outputs[BrainOutputIndex.RELEASE_FOOD]
            ),
            want_nurse=self._positive_action_output(
                centered_outputs[BrainOutputIndex.NURSE]
            ),
            flee_panic_intensity=self._positive_action_output(
                centered_outputs[BrainOutputIndex.PANIC]
            ),
            herding=self._positive_action_output(
                centered_outputs[BrainOutputIndex.HERDING]
            ),
            emit_sound=self._positive_action_output(
                centered_outputs[BrainOutputIndex.ACOUSTIC_EMISSION]
            ),
            sound_tone=centered_outputs[BrainOutputIndex.ACOUSTIC_TONE],
            emit_trail_pheromone=self._positive_action_output(
                centered_outputs[BrainOutputIndex.TRAIL_PHEROMONE]
            ),
            emit_alarm_pheromone=self._positive_action_output(
                centered_outputs[BrainOutputIndex.ALARM_PHEROMONE]
            ),
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
                        self.last_input_names[index]
                        if index < len(self.last_input_names)
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
        """Return 14 independent, finite neural outputs centered in [-1, 1]."""
        try:
            output_values = list(raw_outputs)
        except TypeError:
            return DEFAULT_CENTERED_OUTPUTS.copy()

        centered = [
            self._center_output(value, self._output_activation(index))
            for index, value in enumerate(output_values[:ACTION_OUTPUT_COUNT])
        ]

        missing_outputs = ACTION_OUTPUT_COUNT - len(centered)
        if missing_outputs > 0:
            centered.extend(DEFAULT_CENTERED_OUTPUTS[len(centered) :])

        return centered

    def _center_output(self, value: Any, activation: str | None) -> float:
        """Convert one already-activated NEAT output to centered [-1, 1]."""
        try:
            output = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0

        if not isfinite(output):
            return 0.0

        activation_name = (
            "" if activation is None else str(activation).strip().lower()
        )

        if activation_name == "sigmoid":
            return 2.0 * self._clamp(output, 0.0, 1.0) - 1.0

        if activation_name in {"tanh", "clamped"}:
            return self._clamp(output, -1.0, 1.0)

        if activation_name == "relu":
            return self._clamp(output, 0.0, 1.0)

        if activation_name == "lelu":
            return tanh(output)

        return tanh(output)

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

    def _positive_action_output(self, value: float) -> float:
        return self._clamp(value, 0.0, 1.0)
