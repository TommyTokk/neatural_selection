from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isfinite, log1p, tanh
from typing import Any

import neat
from neat.graphs import required_for_output

from src.creature.action import (
    ACTION_OUTPUT_COUNT,
    ACTION_OUTPUT_NAMES,
    Action,
    BrainOutputIndex,
)
from src.creature.vision import SENSOR_INPUT_COUNT, SENSOR_INPUT_NAMES, SensorSnapshot

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
    network: neat.nn.RecurrentNetwork  # Stateful network created from the genome
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
    _last_activation_network_state: dict[str, Any] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Execute post init behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        Returns
        -------
        None
            Result produced by this creature-domain operation.
        
        Raises
        ------
        ValueError
            If an input or restored value violates validation rules.
        """
        # Keep post init behavior explicit in its owning subsystem.
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
        """Execute from genome behavior.

Parameters
----------
genome_id
    Input used by this creature-domain operation.
genome
    Input used by this creature-domain operation.
config
    Input used by this creature-domain operation.
herding_decay_rate
    Input used by this creature-domain operation.
Returns
-------
NeatBrain
    Result produced by this creature-domain operation."""
        # Keep from genome behavior explicit in its owning subsystem.
        network = neat.nn.RecurrentNetwork.create(genome, config)
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
        decision_dt: float | None = None,
    ) -> Action:
        """Decide on an action based on the current sensor snapshot.
                This method processes the sensor inputs through the neural network and
                normalizes the outputs to produce a valid action for the creature.
        
                Args:
                    snapshot (SensorSnapshot): The current sensor snapshot of the creature's environment.
        
                Returns:
                    Action: The action decided by the neural network based on the sensor inputs.
        
        Parameters
        ----------
        snapshot
            Input used by this creature-domain operation.
        capture_inputs
            Input used by this creature-domain operation.
        decision_dt
            Input used by this creature-domain operation.
        Returns
        -------
        Action
            Result produced by this creature-domain operation.
        
        Raises
        ------
        RuntimeError
            If runtime state violates the callable invariant.
        """
        # Keep decide behavior explicit in its owning subsystem.

        if len(self._input_buffer) != snapshot.sensor_contract.input_count:
            self._input_buffer = [0.0] * snapshot.sensor_contract.input_count
        snapshot.write_inputs(self._input_buffer)
        network_input_nodes = getattr(self.network, "input_nodes", None)
        if (
            network_input_nodes is not None
            and len(self._input_buffer) != len(network_input_nodes)
        ):
            raise RuntimeError(
                "Runtime sensor input count mismatch. "
                f"Vision: {len(self._input_buffer)}, "
                f"network: {len(network_input_nodes)}"
            )
        if capture_inputs:
            self.capture_input_snapshot()
            self._last_activation_network_state = self.export_network_state()
        self.last_input_names = snapshot.sensor_contract.input_names
        raw_outputs = self.network.activate(self._input_buffer)
        centered_outputs = self._normalize_outputs(raw_outputs)
        self.last_outputs = centered_outputs
        self.last_raw_herding = self._positive_action_output(
            centered_outputs[BrainOutputIndex.HERDING]
        )
        herding_alpha = self._elapsed_herding_alpha(decision_dt)
        self.herding_state = self._clamp(
            self.herding_state * (1.0 - herding_alpha)
            + self.last_raw_herding * herding_alpha,
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
            emit_red=self._positive_action_output(
                centered_outputs[BrainOutputIndex.EMIT_RED]
            ),
            emit_green=self._positive_action_output(
                centered_outputs[BrainOutputIndex.EMIT_GREEN]
            ),
            emit_blue=self._positive_action_output(
                centered_outputs[BrainOutputIndex.EMIT_BLUE]
            ),
            rest=self._positive_action_output(
                centered_outputs[BrainOutputIndex.REST]
            ),
        )
        return self.last_action

    def _elapsed_herding_alpha(self, decision_dt: float | None) -> float:
        """Preserve the historical 30 Hz filter response across cadences.

Parameters
----------
decision_dt
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep elapsed herding alpha behavior explicit in its owning subsystem.
        alpha = self.herding_decay_rate
        if decision_dt is None or alpha >= 1.0:
            return alpha
        elapsed = max(0.0, float(decision_dt))
        if elapsed <= 0.0 or alpha <= 0.0:
            return 0.0
        reference_dt = 1.0 / 30.0
        response_rate = -log1p(-alpha) / reference_dt
        return 1.0 - exp(-response_rate * elapsed)

    def capture_input_snapshot(self) -> None:
        """Publish a stable copy of the latest activation inputs.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep capture input snapshot behavior explicit in its owning subsystem.
        self.last_inputs = list(self._input_buffer)

    def evaluate_pure(self, inputs: list[float] | tuple[float, ...]) -> tuple[float, ...]:
        """Evaluate this exact network without mutating live/debug state.

Parameters
----------
inputs
    Input used by this creature-domain operation.
Returns
-------
tuple[float, ...]
    Result produced by this creature-domain operation."""
        # Keep evaluate pure behavior explicit in its owning subsystem.
        network = self.clone_network()
        raw_outputs = network.activate(inputs)
        return self.normalize_outputs_pure(
            raw_outputs,
            tuple(self.output_activations),
        )

    def export_network_state(self) -> dict[str, Any] | None:
        """Return an isolated shallow copy of recurrent activation state.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
dict[str, Any] | None
    Active buffer and two copied value dictionaries when state is available.
"""
        # Delegate shape checks and copying to the network-level helper.
        return self._export_network_state(self.network)

    def current_node_activation(self, node_key: int) -> float | None:
        """Return one node's latest finite recurrent activation.

Parameters
----------
node_key
    NEAT input, hidden, or output node identifier.
Returns
-------
float | None
    Value in the active recurrent buffer, or ``None`` when unavailable.
"""
        # Read the active buffer directly so diagnostics never advance the brain.
        values = getattr(self.network, "values", None)
        active = getattr(self.network, "active", None)
        if (
            isinstance(active, bool)
            or active not in (0, 1)
            or not isinstance(values, list)
            or len(values) != 2
            or not isinstance(values[active], dict)
        ):
            return None
        try:
            value = float(values[active][node_key])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        return value if isfinite(value) else None

    def current_output_signal(self, node_key: int) -> float | None:
        """Return the latest centered signal published by one output node.

Parameters
----------
node_key
    NEAT output-node identifier.
Returns
-------
float | None
    Centered output value, or ``None`` before a decision or for non-outputs.
"""
        # Map the compiled output-node order onto the normalized output snapshot.
        output_nodes = getattr(self.network, "output_nodes", None)
        if not isinstance(output_nodes, (list, tuple)):
            return None
        try:
            index = output_nodes.index(node_key)
            value = float(self.last_outputs[index])
        except (ValueError, IndexError, TypeError, OverflowError):
            return None
        return value if isfinite(value) else None

    @staticmethod
    def _export_network_state(network: Any) -> dict[str, Any] | None:
        """Copy recurrent state from one compiled network.

Parameters
----------
network
    Network whose active index and value buffers are copied.
Returns
-------
dict[str, Any] | None
    Primitive recurrent state, or ``None`` for a stateless network.
"""
        # Copy only scalar state and the two flat node-value dictionaries.
        values = getattr(network, "values", None)
        active = getattr(network, "active", None)
        if (
            isinstance(active, bool)
            or active not in (0, 1)
            or not isinstance(values, list)
            or len(values) != 2
            or not all(isinstance(buffer, dict) for buffer in values)
        ):
            return None
        return {
            "active": int(active),
            "values": [dict(values[0]), dict(values[1])],
        }

    def restore_network_state(self, state: object) -> None:
        """Restore recurrent buffers after validating exact node IDs.

Parameters
----------
state
    Serialized active index and two node-value buffers.
Returns
-------
None
    The live network receives isolated state dictionaries.
Raises
------
ValueError
    If the state shape or node IDs do not match the network.
"""
        # Restore only through the shared strict validation path.
        self._restore_network_state(self.network, state)

    @property
    def has_captured_activation_state(self) -> bool:
        """Return whether counterfactual replay has exact pre-tick state.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
bool
    Whether a captured decision has published its pre-activation buffers.
"""
        # A missing snapshot means diagnostic replay must wait for a decision.
        return self._last_activation_network_state is not None

    def captured_activation_network_state(self) -> dict[str, Any] | None:
        """Return an isolated copy of the latest pre-decision RNN state.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
dict[str, Any] | None
    Copied active index and value buffers, or ``None`` before capture.
"""
        # Publish fresh dictionaries so queued probes cannot alias live state.
        state = self._last_activation_network_state
        if state is None:
            return None
        values = state["values"]
        return {
            "active": int(state["active"]),
            "values": [dict(values[0]), dict(values[1])],
        }

    def clone_network(
        self,
        *,
        before_last_activation: bool = False,
    ) -> neat.nn.RecurrentNetwork:
        """Build an independent network with shallow-copied recurrent state.

Parameters
----------
before_last_activation
    Whether to use the latest captured pre-decision buffers.
Returns
-------
neat.nn.RecurrentNetwork
    Independent recurrent evaluator with the same compiled topology.
"""
        # Select either the diagnostic pre-state or the current live state.
        state = (
            self._last_activation_network_state
            if before_last_activation
            and self._last_activation_network_state is not None
            else self.export_network_state()
        )
        return self._clone_network(self.network, state)

    @staticmethod
    def _clone_network(
        network: Any,
        state: object,
    ) -> neat.nn.RecurrentNetwork:
        """Clone compiled topology and buffers without deep copying.

Parameters
----------
network
    Compiled recurrent network supplying immutable topology data.
state
    Recurrent state to install in the clone.
Returns
-------
neat.nn.RecurrentNetwork
    Independent network, or the original stateless test evaluator.
"""
        # Reuse node-evaluation objects while allocating fresh state dictionaries.
        input_nodes = getattr(network, "input_nodes", None)
        output_nodes = getattr(network, "output_nodes", None)
        node_evals = getattr(network, "node_evals", None)
        if input_nodes is None or output_nodes is None or node_evals is None:
            # Lightweight stateless test/debug networks can safely be shared.
            return network
        clone = neat.nn.RecurrentNetwork(
            list(input_nodes),
            list(output_nodes),
            list(node_evals),
        )
        if state is not None:
            NeatBrain._restore_network_state(clone, state)
        return clone

    @staticmethod
    def _restore_network_state(network: Any, state: object) -> None:
        """Validate and install recurrent state on a compiled network.

Parameters
----------
network
    Target recurrent network with initialized node dictionaries.
state
    Candidate active index and pair of node-value dictionaries.
Returns
-------
None
    Validated state is installed using fresh shallow dictionaries.
Raises
------
ValueError
    If buffer count, active index, or expected node IDs do not match.
"""
        # Reject malformed or topology-incompatible checkpoint state.
        if not isinstance(state, dict):
            raise ValueError("Recurrent network state must be a dictionary.")
        active = state.get("active")
        values = state.get("values")
        if isinstance(active, bool) or active not in (0, 1):
            raise ValueError("Recurrent network active buffer must be 0 or 1.")
        if (
            not isinstance(values, (list, tuple))
            or len(values) != 2
            or not all(isinstance(buffer, dict) for buffer in values)
        ):
            raise ValueError(
                "Recurrent network state must contain exactly two dictionaries."
            )
        expected_values = getattr(network, "values", None)
        if (
            not isinstance(expected_values, list)
            or len(expected_values) != 2
            or not all(isinstance(buffer, dict) for buffer in expected_values)
        ):
            raise ValueError("Target network does not expose recurrent buffers.")
        for index, (restored, expected) in enumerate(
            zip(values, expected_values)
        ):
            if set(restored) != set(expected):
                missing = sorted(
                    set(expected) - set(restored),
                    key=repr,
                )
                extra = sorted(
                    set(restored) - set(expected),
                    key=repr,
                )
                raise ValueError(
                    "Recurrent network buffer node IDs do not match "
                    f"for buffer {index}; missing={missing}, extra={extra}."
                )
        network.active = int(active)
        network.values = [dict(values[0]), dict(values[1])]

    def sensor_usage(
        self,
        input_keys: list[int] | tuple[int, ...],
        output_keys: list[int] | tuple[int, ...],
    ) -> tuple[SensorUsage, ...]:
        """Report which live inputs can reach actions through enabled genes.

Parameters
----------
input_keys
    Input used by this creature-domain operation.
output_keys
    Input used by this creature-domain operation.
Returns
-------
tuple[SensorUsage, ...]
    Result produced by this creature-domain operation."""
        # Keep sensor usage behavior explicit in its owning subsystem.
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
        """Return 16 independent, finite neural outputs centered in [-1, 1].

Parameters
----------
raw_outputs
    Input used by this creature-domain operation.
Returns
-------
list[float]
    Result produced by this creature-domain operation."""
        # Keep normalize outputs behavior explicit in its owning subsystem.
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
        """Normalize network outputs without reading or writing runtime state.

Parameters
----------
raw_outputs
    Input used by this creature-domain operation.
output_activations
    Input used by this creature-domain operation.
Returns
-------
tuple[float, ...]
    Result produced by this creature-domain operation."""
        # Keep normalize outputs pure behavior explicit in its owning subsystem.
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
        """Convert one already-activated NEAT output to centered [-1, 1].

Parameters
----------
value
    Input used by this creature-domain operation.
activation
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep center output behavior explicit in its owning subsystem.
        return self._center_output_value(value, activation)

    @staticmethod
    def _center_output_value(value: Any, activation: str | None) -> float:
        """Pure implementation of activation-aware output centering.

Parameters
----------
value
    Input used by this creature-domain operation.
activation
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep center output value behavior explicit in its owning subsystem.
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
        """Execute output activation behavior.

Parameters
----------
index
    Input used by this creature-domain operation.
Returns
-------
str | None
    Result produced by this creature-domain operation."""
        # Keep output activation behavior explicit in its owning subsystem.
        if index >= len(self.output_activations):
            return None
        return self.output_activations[index]

    @staticmethod
    def _output_activations_for(genome: Any, config: neat.Config) -> list[str]:
        """Execute output activations for behavior.

Parameters
----------
genome
    Input used by this creature-domain operation.
config
    Input used by this creature-domain operation.
Returns
-------
list[str]
    Result produced by this creature-domain operation."""
        # Keep output activations for behavior explicit in its owning subsystem.
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
        """Execute clamp behavior.

Parameters
----------
value
    Input used by this creature-domain operation.
minimum
    Input used by this creature-domain operation.
maximum
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep clamp behavior explicit in its owning subsystem.
        return max(minimum, min(maximum, value))

    def _positive_action_output(self, value: float) -> float:
        """Execute positive action output behavior.

Parameters
----------
value
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep positive action output behavior explicit in its owning subsystem.
        return self._clamp(value, 0.0, 1.0)
