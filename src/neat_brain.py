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
from src.vision import SENSOR_INPUT_COUNT, SENSOR_INPUT_NAMES, SensorSnapshot

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
    brain_revision: int = 0
    herding_decay_rate: float = 1.0
    output_activations: list[str] = field(default_factory=list)
    last_inputs: list[float] = field(default_factory=list)
    # Last activation-aware outputs, centered independently in [-1, 1].
    last_outputs: list[float] = field(default_factory=list)
    last_action: Action | None = None
    last_input_names: tuple[str, ...] = SENSOR_INPUT_NAMES
    herding_state: float = field(default=0.0, init=False)
    last_raw_herding: float = field(default=0.0, init=False)
    _input_buffer: list[float] = field(
        default_factory=lambda: [0.0] * SENSOR_INPUT_COUNT,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.herding_decay_rate, bool)
            or not isinstance(self.herding_decay_rate, (int, float))
            or not isfinite(self.herding_decay_rate)
            or not 0.0 < self.herding_decay_rate <= 1.0
        ):
            raise ValueError(
                "herding_decay_rate must be finite and within (0, 1]."
            )

    @classmethod
    def from_genome(
        cls,
        genome_id: int,
        genome: Any,
        config: neat.Config,
        herding_decay_rate: float = 1.0,
    ) -> NeatBrain:
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        output_activations = cls._output_activations_for(genome, config)
        return cls(
            genome_id=genome_id,
            genome=genome,
            network=network,
            herding_decay_rate=herding_decay_rate,
            output_activations=output_activations,
        )

    def decide(
        self,
        snapshot: SensorSnapshot,
        *,
        capture_inputs: bool = False,
    ) -> Action:
        """
        Decide on an action based on the current sensor snapshot.
        This method processes the sensor inputs through the neural network and
        normalizes the outputs to produce a valid action for the creature.
        
        Args:
            snapshot (SensorSnapshot): The current sensor snapshot of the creature's environment.

        Returns:
            Action: The action decided by the neural network based on the sensor inputs.
        """

        if len(self._input_buffer) != snapshot.sensor_contract.input_count:
            self._input_buffer = [0.0] * snapshot.sensor_contract.input_count
        snapshot.write_inputs(self._input_buffer)
        if capture_inputs:
            self.capture_input_snapshot()
        self.last_input_names = snapshot.sensor_contract.input_names
        raw_outputs = self.network.activate(self._input_buffer)
        centered_outputs = self._normalize_outputs(raw_outputs)
        self.last_outputs = centered_outputs
        self.last_raw_herding = self._positive_action_output(
            centered_outputs[BrainOutputIndex.HERDING]
        )
        self.herding_state = self._clamp(
            self.herding_state * (1.0 - self.herding_decay_rate)
            + self.last_raw_herding * self.herding_decay_rate,
            0.0,
            1.0,
        )

        self.last_action = Action(
            accelerate=self._clamp(
                centered_outputs[BrainOutputIndex.ACCELERATE], -1.0, 1.0
            ),
            rotate=self._clamp(
                centered_outputs[BrainOutputIndex.ROTATE], -1.0, 1.0
            ),
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
            herding=self.herding_state,
            emit_sound=self._positive_action_output(
                centered_outputs[BrainOutputIndex.ACOUSTIC_EMISSION]
            ),
            sound_tone=self._clamp(
                centered_outputs[BrainOutputIndex.ACOUSTIC_TONE], -1.0, 1.0
            ),
            emit_trail_pheromone=self._positive_action_output(
                centered_outputs[BrainOutputIndex.TRAIL_PHEROMONE]
            ),
            emit_alarm_pheromone=self._positive_action_output(
                centered_outputs[BrainOutputIndex.ALARM_PHEROMONE]
            ),
            rest=self._positive_action_output(
                centered_outputs[BrainOutputIndex.REST]
            ),
        )
        return self.last_action

    def capture_input_snapshot(self) -> None:
        """Publish a stable copy of the latest activation inputs."""
        self.last_inputs = list(self._input_buffer)

    def evaluate_pure(self, inputs: list[float] | tuple[float, ...]) -> tuple[float, ...]:
        """Evaluate this exact network without mutating live/debug state."""
        raw_outputs = self.network.activate(inputs)
        return self.normalize_outputs_pure(
            raw_outputs,
            tuple(self.output_activations),
        )

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
        return list(
            self.normalize_outputs_pure(
                raw_outputs,
                tuple(self.output_activations),
            )
        )

    @classmethod
    def normalize_outputs_pure(
        cls,
        raw_outputs: Any,
        output_activations: tuple[str, ...],
    ) -> tuple[float, ...]:
        """Normalize network outputs without reading or writing runtime state."""
        try:
            output_values = list(raw_outputs)
        except TypeError:
            return tuple(DEFAULT_CENTERED_OUTPUTS)

        centered = [
            cls._center_output_value(
                value,
                (
                    output_activations[index]
                    if index < len(output_activations)
                    else None
                ),
            )
            for index, value in enumerate(output_values[:ACTION_OUTPUT_COUNT])
        ]

        missing_outputs = ACTION_OUTPUT_COUNT - len(centered)
        if missing_outputs > 0:
            centered.extend(DEFAULT_CENTERED_OUTPUTS[len(centered) :])

        return tuple(centered)

    def _center_output(self, value: Any, activation: str | None) -> float:
        """Convert one already-activated NEAT output to centered [-1, 1]."""
        return self._center_output_value(value, activation)

    @staticmethod
    def _center_output_value(value: Any, activation: str | None) -> float:
        """Pure implementation of activation-aware output centering."""
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
            return 2.0 * max(0.0, min(1.0, output)) - 1.0

        if activation_name in {"tanh", "clamped"}:
            return max(-1.0, min(1.0, output))

        if activation_name == "relu":
            return max(0.0, min(1.0, output))

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
