from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import ceil, cos, exp, floor, isfinite, sin, sqrt

import numpy as np

from configs.sim_config import CommunicationConfig, PheromoneBoundaryMode, PheromoneConfig


@dataclass(frozen=True, slots=True)
class AcousticSignal:
    emitter_id: int
    position: tuple[float, float]
    strength: float
    tone: float


@dataclass(frozen=True, slots=True)
class AcousticObservation:
    """Acoustic quantities available to an AI-controlled receiver."""

    strength: float = 0.0
    direction_sin: float = 0.0
    direction_cos: float = 0.0
    tone: float = 0.0


@dataclass(frozen=True, slots=True)
class AcousticDebugInfo:
    """Privileged acoustic metadata for visualization and diagnostics only."""

    source_id: int | None = None
    source_position: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class AcousticSenseResult:
    observation: AcousticObservation = AcousticObservation()
    debug: AcousticDebugInfo = AcousticDebugInfo()


@dataclass(frozen=True, slots=True)
class PheromoneSnapshot:
    """Raw RGB samples at the local, forward-left, and forward-right probes."""

    local: tuple[float, float, float] = (0.0, 0.0, 0.0)
    forward_left: tuple[float, float, float] = (0.0, 0.0, 0.0)
    forward_right: tuple[float, float, float] = (0.0, 0.0, 0.0)


_EMPTY_ACOUSTIC_OBSERVATION = AcousticObservation()
_EMPTY_ACOUSTIC_DEBUG = AcousticDebugInfo()


class AcousticSystem:
    """Instantaneous broadcasts indexed by emission-range-sized cells."""

    def __init__(self, config: CommunicationConfig) -> None:
        """Execute init behavior.

Parameters
----------
config
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep init behavior explicit in its owning subsystem.
        self.config = config
        self._range = float(config.acoustic_range)
        self._range_squared = self._range * self._range
        self._inverse_range = 0.0 if self._range == 0.0 else 1.0 / self._range
        self._minimum_emission = float(config.acoustic_min_emission_strength)
        self._hearing_threshold = float(config.acoustic_hearing_threshold)
        self._cell_size = self._range
        self.signals: dict[int, AcousticSignal] = {}
        self._spatial_index: dict[tuple[int, int], list[AcousticSignal]] = {}
        self.last_candidate_checks = 0
        self.total_candidate_checks = 0

    def replace_signals(self, signals: Iterable[AcousticSignal]) -> None:
        """Validate, filter, and index one current signal per emitter.

Parameters
----------
signals
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep replace signals behavior explicit in its owning subsystem.

        retained: dict[int, AcousticSignal] = {}
        for signal in signals:
            validated = self._validated_signal(signal)
            if validated.strength >= self._minimum_emission:
                retained[validated.emitter_id] = validated

        spatial_index: dict[tuple[int, int], list[AcousticSignal]] = {}
        if self._cell_size > 0.0:
            inverse_cell_size = 1.0 / self._cell_size
            for signal in retained.values():
                cell = (
                    floor(signal.position[0] * inverse_cell_size),
                    floor(signal.position[1] * inverse_cell_size),
                )
                spatial_index.setdefault(cell, []).append(signal)

        self.signals = retained
        self._spatial_index = spatial_index

    def remove_emitter(self, emitter_id: int) -> None:
        """Execute remove emitter behavior.

Parameters
----------
emitter_id
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep remove emitter behavior explicit in its owning subsystem.
        signal = self.signals.pop(emitter_id, None)
        if signal is None or self._cell_size <= 0.0:
            return
        cell = (
            floor(signal.position[0] / self._cell_size),
            floor(signal.position[1] / self._cell_size),
        )
        bucket = self._spatial_index.get(cell)
        if bucket is None:
            return
        for index, candidate in enumerate(bucket):
            if candidate.emitter_id == emitter_id:
                bucket.pop(index)
                break
        if not bucket:
            self._spatial_index.pop(cell, None)

    def sense(
        self,
        receiver_id: int,
        position: tuple[float, float],
        heading: float,
    ) -> AcousticObservation:
        """Return only sensory information available to the receiver.

Parameters
----------
receiver_id
    Input used by this creature-domain operation.
position
    Input used by this creature-domain operation.
heading
    Input used by this creature-domain operation.
Returns
-------
AcousticObservation
    Result produced by this creature-domain operation."""
        # Keep sense behavior explicit in its owning subsystem.

        observation, _signal = self._sense(receiver_id, position, heading)
        return observation

    def sense_with_debug(
        self,
        receiver_id: int,
        position: tuple[float, float],
        heading: float,
    ) -> AcousticSenseResult:
        """Sense once and include privileged source metadata explicitly.

Parameters
----------
receiver_id
    Input used by this creature-domain operation.
position
    Input used by this creature-domain operation.
heading
    Input used by this creature-domain operation.
Returns
-------
AcousticSenseResult
    Result produced by this creature-domain operation."""
        # Keep sense with debug behavior explicit in its owning subsystem.

        observation, signal = self._sense(receiver_id, position, heading)
        debug = (
            _EMPTY_ACOUSTIC_DEBUG
            if signal is None
            else AcousticDebugInfo(signal.emitter_id, signal.position)
        )
        return AcousticSenseResult(observation, debug)

    def _sense(
        self,
        receiver_id: int,
        position: tuple[float, float],
        heading: float,
    ) -> tuple[AcousticObservation, AcousticSignal | None]:
        """Execute sense behavior.
        
        Parameters
        ----------
        receiver_id
            Input used by this creature-domain operation.
        position
            Input used by this creature-domain operation.
        heading
            Input used by this creature-domain operation.
        Returns
        -------
        tuple[AcousticObservation, AcousticSignal | None]
            Result produced by this creature-domain operation.
        
        Raises
        ------
        ValueError
            If an input or restored value violates validation rules.
        """
        # Keep sense behavior explicit in its owning subsystem.
        if type(receiver_id) is not int:
            raise ValueError(f"receiver_id must be an integer, got {receiver_id!r}.")
        receiver_x, receiver_y = self._validated_position(position, "position")
        if isinstance(heading, bool) or not isinstance(heading, (int, float)):
            raise ValueError(f"heading must be finite, got {heading!r}.")
        heading = float(heading)
        if not isfinite(heading):
            raise ValueError(f"heading must be finite, got {heading!r}.")

        if self._range <= 0.0:
            self.last_candidate_checks = 0
            return _EMPTY_ACOUSTIC_OBSERVATION, None

        inverse_cell_size = 1.0 / self._cell_size
        minimum_column = floor((receiver_x - self._range) * inverse_cell_size)
        maximum_column = floor((receiver_x + self._range) * inverse_cell_size)
        minimum_row = floor((receiver_y - self._range) * inverse_cell_size)
        maximum_row = floor((receiver_y + self._range) * inverse_cell_size)

        best_signal: AcousticSignal | None = None
        best_strength = -1.0
        best_distance = 0.0
        best_dx = 0.0
        best_dy = 0.0
        candidate_checks = 0

        for row in range(minimum_row, maximum_row + 1):
            for column in range(minimum_column, maximum_column + 1):
                bucket = self._spatial_index.get((column, row))
                if bucket is None:
                    continue
                for signal in bucket:
                    candidate_checks += 1
                    if signal.emitter_id == receiver_id:
                        continue
                    dx = signal.position[0] - receiver_x
                    dy = signal.position[1] - receiver_y
                    distance_squared = dx * dx + dy * dy
                    if distance_squared > self._range_squared:
                        continue
                    if signal.strength < best_strength:
                        continue
                    distance = sqrt(distance_squared)
                    attenuation = 1.0 - distance * self._inverse_range
                    heard_strength = signal.strength * attenuation * attenuation
                    if heard_strength < self._hearing_threshold:
                        continue
                    if (
                        heard_strength > best_strength
                        or (
                            heard_strength == best_strength
                            and (
                                best_signal is None
                                or signal.emitter_id < best_signal.emitter_id
                            )
                        )
                    ):
                        best_signal = signal
                        best_strength = heard_strength
                        best_distance = distance
                        best_dx = dx
                        best_dy = dy

        self.last_candidate_checks = candidate_checks
        self.total_candidate_checks += candidate_checks
        if best_signal is None:
            return _EMPTY_ACOUSTIC_OBSERVATION, None

        if best_distance <= 1e-12:
            direction_sin = 0.0
            direction_cos = 0.0
        else:
            inverse_distance = 1.0 / best_distance
            unit_x = best_dx * inverse_distance
            unit_y = best_dy * inverse_distance
            heading_cos = cos(heading)
            heading_sin = sin(heading)
            direction_cos = unit_x * heading_cos + unit_y * heading_sin
            direction_sin = unit_y * heading_cos - unit_x * heading_sin

        return (
            AcousticObservation(
                strength=best_strength,
                direction_sin=max(-1.0, min(1.0, direction_sin)),
                direction_cos=max(-1.0, min(1.0, direction_cos)),
                tone=best_signal.tone,
            ),
            best_signal,
        )

    @classmethod
    def _validated_signal(cls, signal: AcousticSignal) -> AcousticSignal:
        """Execute validated signal behavior.
        
        Parameters
        ----------
        signal
            Input used by this creature-domain operation.
        Returns
        -------
        AcousticSignal
            Result produced by this creature-domain operation.
        
        Raises
        ------
        ValueError
            If an input or restored value violates validation rules.
        """
        # Keep validated signal behavior explicit in its owning subsystem.
        if not isinstance(signal, AcousticSignal):
            raise ValueError(f"signal must be AcousticSignal, got {signal!r}.")
        if type(signal.emitter_id) is not int:
            raise ValueError(
                f"emitter_id must be an integer, got {signal.emitter_id!r}."
            )
        position = cls._validated_position(signal.position, "signal.position")
        strength = cls._validated_unit_value(signal.strength, "signal.strength")
        tone = signal.tone
        if (
            isinstance(tone, bool)
            or not isinstance(tone, (int, float))
            or not isfinite(tone)
            or not -1.0 <= tone <= 1.0
        ):
            raise ValueError(
                f"signal.tone must be finite and within [-1, 1], got {tone!r}."
            )
        return AcousticSignal(signal.emitter_id, position, strength, float(tone))

    @staticmethod
    def _validated_unit_value(value: object, name: str) -> float:
        """Execute validated unit value behavior.
        
        Parameters
        ----------
        value
            Input used by this creature-domain operation.
        name
            Input used by this creature-domain operation.
        Returns
        -------
        float
            Result produced by this creature-domain operation.
        
        Raises
        ------
        ValueError
            If an input or restored value violates validation rules.
        """
        # Keep validated unit value behavior explicit in its owning subsystem.
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"{name} must be finite and within [0, 1], got {value!r}.")
        return float(value)

    @staticmethod
    def _validated_position(
        position: object,
        name: str,
    ) -> tuple[float, float]:
        """Execute validated position behavior.
        
        Parameters
        ----------
        position
            Input used by this creature-domain operation.
        name
            Input used by this creature-domain operation.
        Returns
        -------
        tuple[float, float]
            Result produced by this creature-domain operation.
        
        Raises
        ------
        ValueError
            If an input or restored value violates validation rules.
        """
        # Keep validated position behavior explicit in its owning subsystem.
        if not isinstance(position, (tuple, list)) or len(position) != 2:
            raise ValueError(f"{name} must contain exactly two coordinates.")
        x, y = position
        if (
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not isfinite(x)
            or isinstance(y, bool)
            or not isinstance(y, (int, float))
            or not isfinite(y)
        ):
            raise ValueError(f"{name} coordinates must be finite, got {position!r}.")
        return float(x), float(y)


class PheromoneSystem:
    """Width-major three-channel stigmergy field.

    Axis 0 is world X, axis 1 is world Y, and axis 2 is RGB.  Image code must
    swap the first two axes rather than changing this simulation invariant.
    """

    _STABILITY_TOLERANCE = 1e-12

    def __init__(
        self,
        config: PheromoneConfig | CommunicationConfig,
        grid_width: int,
        grid_height: int,
        world_bounds: tuple[float, float, float, float],
    ) -> None:
        """Create a width-major RGB pheromone field.

        Parameters
        ----------
        config
            Pheromone physics and boundary configuration.
        grid_width
            Number of samples along world X.
        grid_height
            Number of samples along world Y.
        world_bounds
            World-space left, bottom, right, and top bounds.

        Returns
        -------
        None
            The initialized system owns an empty RGB tensor.

        Raises
        ------
        TypeError
            If the supplied configuration has the wrong type.
        ValueError
            If dimensions or bounds are invalid.
        """
        # Validate geometry once so every hot-path operation can stay vectorized.
        if isinstance(config, CommunicationConfig):
            config = config.pheromone
        if not isinstance(config, PheromoneConfig):
            raise TypeError("config must be a PheromoneConfig.")
        if type(grid_width) is not int or grid_width < 2:
            raise ValueError("grid_width must be an integer >= 2.")
        if type(grid_height) is not int or grid_height < 2:
            raise ValueError("grid_height must be an integer >= 2.")
        if not isinstance(world_bounds, (tuple, list)) or len(world_bounds) != 4:
            raise ValueError("world_bounds must contain left, bottom, right, and top.")
        bounds = tuple(float(value) for value in world_bounds)
        if not np.all(np.isfinite(bounds)):
            raise ValueError("world_bounds must be finite.")
        left, bottom, right, top = bounds
        if right <= left or top <= bottom:
            raise ValueError("world_bounds must have positive width and height.")

        self.config = config
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.world_bounds = bounds
        self.boundary_mode = PheromoneBoundaryMode(config.boundary_mode)
        self.cell_size_x = (right - left) / (grid_width - 1)
        self.cell_size_y = (top - bottom) / (grid_height - 1)
        self.inverse_cell_size_x_squared = 1.0 / self.cell_size_x**2
        self.inverse_cell_size_y_squared = 1.0 / self.cell_size_y**2
        self._diffusion_coefficient = float(config.diffusion_coefficient)
        self._decay_rate = float(config.decay_rate)
        self._maximum = float(config.max_concentration)
        inverse_geometry_sum = (
            self.inverse_cell_size_x_squared
            + self.inverse_cell_size_y_squared
        )
        self.maximum_stable_timestep = (
            float("inf")
            if self._diffusion_coefficient == 0.0
            else 0.5 / (self._diffusion_coefficient * inverse_geometry_sum)
        )
        self.field = np.zeros((grid_width, grid_height, 3), dtype=np.float32)
        self.update_count = 0

    def sample(self, x: float, y: float) -> np.ndarray:
        """Bilinearly sample RGB at one world position.

        Parameters
        ----------
        x
            World X coordinate.
        y
            World Y coordinate.

        Returns
        -------
        numpy.ndarray
            Detached RGB vector with shape ``(3,)``.
        """
        # Route scalar reads through the same interpolation path as batch reads.
        position = np.asarray([[self._finite_coordinate(x), self._finite_coordinate(y)]])
        return self._sample_positions(position)[0].copy()

    def deposit(self, x: float, y: float, color_vector: object) -> None:
        """Additively splat one RGB amount at a world position.

        Parameters
        ----------
        x
            World X coordinate.
        y
            World Y coordinate.
        color_vector
            Nonnegative Red, Green, and Blue deposit amounts.

        Returns
        -------
        None
            The field is updated in place.
        """
        # Share validation and edge handling with batched deposition.
        colors = self._validated_colors(color_vector, 1)
        self.deposit_many(
            np.asarray([[self._finite_coordinate(x), self._finite_coordinate(y)]]),
            colors,
        )

    def deposit_many(self, positions: object, color_vectors: object) -> None:
        """Bilinearly splat many RGB vectors without grid loops.

        Parameters
        ----------
        positions
            World positions with shape ``(N, 2)``.
        color_vectors
            RGB deposit amounts with shape ``(N, 3)``.

        Returns
        -------
        None
            Aggregated contributions are clipped into the field.
        """
        # Duplicate edge indices are intentionally accumulated by ``np.add.at``.
        points = self._validated_positions(positions, trailing_shape=(2,))
        colors = self._validated_colors(color_vectors, points.shape[0])
        if points.shape[0] == 0:
            return
        mapped, inside = self._map_positions(points)
        if self.boundary_mode is PheromoneBoundaryMode.ABSORB:
            mapped = mapped[inside]
            colors = colors[inside]
            if mapped.shape[0] == 0:
                return

        x0, x1, y0, y1, u, v = self._bilinear_components(mapped)
        cell_indices = np.stack(
            (
                x0 * self.grid_height + y0,
                x1 * self.grid_height + y0,
                x0 * self.grid_height + y1,
                x1 * self.grid_height + y1,
            ),
            axis=1,
        )
        weights = np.stack(
            (
                (1.0 - u) * (1.0 - v),
                u * (1.0 - v),
                (1.0 - u) * v,
                u * v,
            ),
            axis=1,
        )
        contributions = weights[..., None] * colors[:, None, :]
        np.add.at(
            self.field.reshape(-1, 3),
            cell_indices.reshape(-1),
            contributions.reshape(-1, 3),
        )
        np.clip(self.field, 0.0, self._maximum, out=self.field)

    def sense(
        self,
        positions: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ],
    ) -> PheromoneSnapshot:
        """Sample local, forward-left, and forward-right RGB probes.

        Parameters
        ----------
        positions
            Exactly three world-space probe positions.

        Returns
        -------
        PheromoneSnapshot
            Immutable RGB values in probe order.

        Raises
        ------
        ValueError
            If the probe collection does not have shape ``(3, 2)``.
        """
        # Preserve probe ordering for the sensor-gradient contract.
        points = self._validated_positions(positions, trailing_shape=(2,))
        if points.shape != (3, 2):
            raise ValueError("positions must contain exactly three probes.")
        values = self._sample_positions(points)
        return PheromoneSnapshot(
            local=tuple(float(value) for value in values[0]),
            forward_left=tuple(float(value) for value in values[1]),
            forward_right=tuple(float(value) for value in values[2]),
        )

    def sense_many(self, positions: object, out: np.ndarray | None = None) -> np.ndarray:
        """Sample three RGB probes for each creature in a batch.

        Parameters
        ----------
        positions
            Probe positions with shape ``(N, 3, 2)``.
        out
            Optional floating destination with shape ``(N, 3, 3)``.

        Returns
        -------
        numpy.ndarray
            Probe-major RGB samples with shape ``(N, 3, 3)``.

        Raises
        ------
        ValueError
            If positions or the optional destination have invalid shapes.
        """
        # Flatten only the probe dimension while retaining RGB as the last axis.
        points = self._validated_positions(positions, trailing_shape=(3, 2))
        count = points.shape[0]
        expected_shape = (count, 3, 3)
        if out is None:
            result = np.empty(expected_shape, dtype=np.float32)
        elif (
            not isinstance(out, np.ndarray)
            or out.shape != expected_shape
            or not np.issubdtype(out.dtype, np.floating)
        ):
            raise ValueError(f"out must be a floating array with shape {expected_shape}.")
        else:
            result = out
        if count:
            result[...] = self._sample_positions(points.reshape(-1, 2)).reshape(
                expected_shape
            )
        return result

    def advance(self, delta_time: float) -> None:
        """Advance diffusion-decay by one explicit fixed timestep.

        Parameters
        ----------
        delta_time
            Deterministic simulation timestep in seconds.

        Returns
        -------
        None
            The complete RGB tensor is advanced in place.

        Raises
        ------
        ValueError
            If the timestep is invalid or exceeds the stability bound.
        """
        # Roll X and Y independently; the channel axis is never diffused across.
        dt = self._validated_timestep(delta_time)
        if dt == 0.0:
            return
        if dt > self.maximum_stable_timestep + self._STABILITY_TOLERANCE:
            raise ValueError(
                "Pheromone timestep is unstable for the configured diffusion/grid: "
                f"{dt} > {self.maximum_stable_timestep}."
            )
        field = self.field
        x_minus = np.roll(field, 1, axis=0)
        x_plus = np.roll(field, -1, axis=0)
        y_minus = np.roll(field, 1, axis=1)
        y_plus = np.roll(field, -1, axis=1)
        if self.boundary_mode is PheromoneBoundaryMode.REFLECT:
            x_minus[0, :, :] = field[0, :, :]
            x_plus[-1, :, :] = field[-1, :, :]
            y_minus[:, 0, :] = field[:, 0, :]
            y_plus[:, -1, :] = field[:, -1, :]
        elif self.boundary_mode is PheromoneBoundaryMode.ABSORB:
            x_minus[0, :, :] = 0.0
            x_plus[-1, :, :] = 0.0
            y_minus[:, 0, :] = 0.0
            y_plus[:, -1, :] = 0.0
        laplacian = (
            (x_minus + x_plus - 2.0 * field)
            * self.inverse_cell_size_x_squared
            + (y_minus + y_plus - 2.0 * field)
            * self.inverse_cell_size_y_squared
        )
        updated = (field + self._diffusion_coefficient * laplacian * dt) * exp(
            -self._decay_rate * dt
        )
        np.clip(updated, 0.0, self._maximum, out=self.field)
        self.update_count += 1

    def state_metadata(self) -> dict[str, object]:
        """Describe the persisted tensor coordinate contract.

        Parameters
        ----------
        None
            This callable receives no external parameters.

        Returns
        -------
        dict[str, object]
            Shape, axis order, bounds, and boundary mode metadata.
        """
        # Persist axis semantics explicitly to reject accidental transposition.
        return {
            "grid_shape": self.field.shape,
            "axis_order": "xyc",
            "world_bounds": self.world_bounds,
            "boundary_mode": self.boundary_mode.value,
        }

    def restore(
        self,
        field: object,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Restore a validated width-major RGB field.

        Parameters
        ----------
        field
            Candidate tensor with shape ``(width, height, 3)``.
        metadata
            Optional saved coordinate and boundary contract.

        Returns
        -------
        None
            Validated values replace the current field.

        Raises
        ------
        ValueError
            If values or metadata are incompatible with this system.
        """
        # Validate every persisted invariant before mutating live state.
        restored = np.asarray(field, dtype=np.float32)
        if restored.shape != self.field.shape:
            raise ValueError(
                f"Saved pheromone field must have shape {self.field.shape}, "
                f"got {restored.shape}."
            )
        if not np.all(np.isfinite(restored)):
            raise ValueError("Saved pheromone field contains nonfinite values.")
        if np.any(restored < 0.0) or np.any(restored > self._maximum):
            raise ValueError("Saved pheromone field is outside its configured range.")
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise ValueError("pheromone metadata must be a mapping.")
            if tuple(metadata.get("grid_shape", ())) != self.field.shape:
                raise ValueError("Saved pheromone metadata has an incompatible shape.")
            if metadata.get("axis_order") != "xyc":
                raise ValueError("Saved pheromone metadata has an incompatible axis order.")
            saved_bounds = tuple(float(value) for value in metadata.get("world_bounds", ()))
            if saved_bounds != self.world_bounds:
                raise ValueError("Saved pheromone metadata has incompatible world bounds.")
            if PheromoneBoundaryMode(metadata.get("boundary_mode")) is not self.boundary_mode:
                raise ValueError("Saved pheromone metadata has incompatible boundaries.")
        np.copyto(self.field, restored)

    def _sample_positions(self, positions: np.ndarray) -> np.ndarray:
        """Interpolate RGB values for validated world positions.

        Parameters
        ----------
        positions
            Finite world positions with shape ``(N, 2)``.

        Returns
        -------
        numpy.ndarray
            RGB samples with shape ``(N, 3)``.
        """
        # Use advanced indexing for all positions and channels at once.
        mapped, inside = self._map_positions(positions)
        x0, x1, y0, y1, u, v = self._bilinear_components(mapped)
        result = (
            self.field[x0, y0] * ((1.0 - u) * (1.0 - v))[:, None]
            + self.field[x1, y0] * (u * (1.0 - v))[:, None]
            + self.field[x0, y1] * ((1.0 - u) * v)[:, None]
            + self.field[x1, y1] * (u * v)[:, None]
        ).astype(np.float32, copy=False)
        if self.boundary_mode is PheromoneBoundaryMode.ABSORB:
            result[~inside] = 0.0
        return result

    def _map_positions(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply the configured continuous boundary policy.

        Parameters
        ----------
        positions
            Finite world positions with shape ``(N, 2)``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Mapped positions and their original in-bounds mask.
        """
        # Keep the original mask so absorb mode can reject outside samples.
        left, bottom, right, top = self.world_bounds
        mapped = positions.astype(np.float64, copy=True)
        inside = (
            (mapped[:, 0] >= left)
            & (mapped[:, 0] <= right)
            & (mapped[:, 1] >= bottom)
            & (mapped[:, 1] <= top)
        )
        if self.boundary_mode is PheromoneBoundaryMode.WRAP:
            mapped[:, 0] = np.mod(mapped[:, 0] - left, right - left) + left
            mapped[:, 1] = np.mod(mapped[:, 1] - bottom, top - bottom) + bottom
        else:
            mapped[:, 0] = np.clip(mapped[:, 0], left, right)
            mapped[:, 1] = np.clip(mapped[:, 1], bottom, top)
        return mapped, inside

    def _bilinear_components(
        self, positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate safe bilinear indices and fractional weights.

        Parameters
        ----------
        positions
            Boundary-mapped world positions with shape ``(N, 2)``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Lower/upper X and Y indices followed by X and Y fractions.
        """
        # Clamp upper indices so valid edge splats aggregate rather than escape.
        left, bottom, right, top = self.world_bounds
        grid_x = (positions[:, 0] - left) / (right - left) * (self.grid_width - 1)
        grid_y = (positions[:, 1] - bottom) / (top - bottom) * (self.grid_height - 1)
        x0 = np.floor(grid_x).astype(np.intp)
        y0 = np.floor(grid_y).astype(np.intp)
        x1 = np.minimum(x0 + 1, self.grid_width - 1)
        y1 = np.minimum(y0 + 1, self.grid_height - 1)
        return x0, x1, y0, y1, grid_x - x0, grid_y - y0

    @staticmethod
    def _validated_positions(values: object, trailing_shape: tuple[int, ...]) -> np.ndarray:
        """Validate and normalize a position batch.

        Parameters
        ----------
        values
            Candidate numeric positions.
        trailing_shape
            Required dimensions after the batch axis.

        Returns
        -------
        numpy.ndarray
            Finite float64 positions with an explicit batch axis.

        Raises
        ------
        ValueError
            If values are nonnumeric, nonfinite, or incorrectly shaped.
        """
        # Normalize a single probe group into a one-item batch.
        try:
            positions = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("positions must be numeric.") from error
        expected_ndim = len(trailing_shape) + 1
        if positions.ndim == len(trailing_shape) and positions.shape == trailing_shape:
            positions = positions.reshape((1, *trailing_shape))
        if positions.ndim != expected_ndim or positions.shape[1:] != trailing_shape:
            raise ValueError(f"positions must have shape (N, {', '.join(map(str, trailing_shape))}).")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions must be finite.")
        return positions

    @staticmethod
    def _validated_colors(values: object, count: int) -> np.ndarray:
        """Validate nonnegative RGB deposit vectors.

        Parameters
        ----------
        values
            Candidate RGB values.
        count
            Required number of vectors.

        Returns
        -------
        numpy.ndarray
            Float64 RGB values with shape ``(count, 3)``.

        Raises
        ------
        ValueError
            If values are invalid, negative, or incorrectly shaped.
        """
        # Accept a lone RGB vector only for one-position scalar deposition.
        try:
            colors = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("color vectors must be numeric.") from error
        if colors.shape == (3,) and count == 1:
            colors = colors.reshape(1, 3)
        if colors.shape != (count, 3):
            raise ValueError(f"color vectors must have shape {(count, 3)}.")
        if not np.all(np.isfinite(colors)) or np.any(colors < 0.0):
            raise ValueError("color vectors must be finite and nonnegative.")
        return colors

    @staticmethod
    def _finite_coordinate(value: object) -> float:
        """Validate one finite world coordinate.

        Parameters
        ----------
        value
            Candidate coordinate.

        Returns
        -------
        float
            Validated coordinate.

        Raises
        ------
        ValueError
            If the coordinate is not a finite real number.
        """
        # Reject booleans even though Python treats them as integers.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("coordinates must be finite numbers.")
        coordinate = float(value)
        if not isfinite(coordinate):
            raise ValueError("coordinates must be finite numbers.")
        return coordinate

    @staticmethod
    def _validated_timestep(value: object) -> float:
        """Validate one nonnegative explicit timestep.

        Parameters
        ----------
        value
            Candidate timestep in seconds.

        Returns
        -------
        float
            Validated timestep.

        Raises
        ------
        ValueError
            If the timestep is nonnumeric, nonfinite, or negative.
        """
        # Explicit integration cannot accept reverse or undefined time.
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value < 0.0
        ):
            raise ValueError("delta_time must be finite and nonnegative.")
        return float(value)
