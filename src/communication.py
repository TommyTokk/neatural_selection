from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from math import ceil, cos, exp, floor, isfinite, sin, sqrt

import numpy as np

from configs.sim_config import CommunicationConfig, PheromoneBoundaryMode


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
    trail_here: float = 0.0
    trail_forward_left: float = 0.0
    trail_forward_right: float = 0.0
    alarm_here: float = 0.0
    alarm_forward_left: float = 0.0
    alarm_forward_right: float = 0.0


class PheromoneChannel(str, Enum):
    TRAIL = "trail"
    ALARM = "alarm"


_EMPTY_ACOUSTIC_OBSERVATION = AcousticObservation()
_EMPTY_ACOUSTIC_DEBUG = AcousticDebugInfo()


class AcousticSystem:
    """Instantaneous broadcasts indexed by emission-range-sized cells."""

    def __init__(self, config: CommunicationConfig) -> None:
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
        """Validate, filter, and index one current signal per emitter."""

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
        """Return only sensory information available to the receiver."""

        observation, _signal = self._sense(receiver_id, position, heading)
        return observation

    def sense_with_debug(
        self,
        receiver_id: int,
        position: tuple[float, float],
        heading: float,
    ) -> AcousticSenseResult:
        """Sense once and include privileged source metadata explicitly."""

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
    """Two float32 concentration fields using stable world-space diffusion."""

    _STABILITY_TOLERANCE = 1e-12

    def __init__(
        self,
        config: CommunicationConfig,
        grid_width: int,
        grid_height: int,
        world_bounds: tuple[float, float, float, float],
    ) -> None:
        if type(grid_width) is not int or grid_width < 2:
            raise ValueError(f"grid_width must be an integer >= 2, got {grid_width!r}.")
        if type(grid_height) is not int or grid_height < 2:
            raise ValueError(
                f"grid_height must be an integer >= 2, got {grid_height!r}."
            )
        if not isinstance(world_bounds, (tuple, list)) or len(world_bounds) != 4:
            raise ValueError("world_bounds must contain left, bottom, right, and top.")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            for value in world_bounds
        ):
            raise ValueError(f"world_bounds must be finite, got {world_bounds!r}.")
        left, bottom, right, top = (float(value) for value in world_bounds)
        if right <= left:
            raise ValueError(
                f"world_bounds require right > left, got left={left!r}, right={right!r}."
            )
        if top <= bottom:
            raise ValueError(
                f"world_bounds require top > bottom, got bottom={bottom!r}, top={top!r}."
            )

        self.config = config
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.world_bounds = (left, bottom, right, top)
        self.boundary_mode = PheromoneBoundaryMode(config.pheromone_boundary_mode)
        self.cell_size_x = (right - left) / (grid_width - 1)
        self.cell_size_y = (top - bottom) / (grid_height - 1)
        if (
            not isfinite(self.cell_size_x)
            or self.cell_size_x <= 0.0
            or not isfinite(self.cell_size_y)
            or self.cell_size_y <= 0.0
        ):
            raise ValueError("Pheromone cell sizes must be finite and positive.")
        self.inverse_cell_size_x_squared = 1.0 / (self.cell_size_x**2)
        self.inverse_cell_size_y_squared = 1.0 / (self.cell_size_y**2)
        self._diffusion_coefficient = float(config.pheromone_diffusion_coefficient)
        self._evaporation_rate = float(config.pheromone_evaporation_rate)
        self._maximum = float(config.pheromone_max_concentration)
        self._interval = float(config.pheromone_update_interval)
        self._max_updates_per_tick = config.pheromone_max_updates_per_tick
        inverse_geometry_sum = (
            self.inverse_cell_size_x_squared
            + self.inverse_cell_size_y_squared
        )
        self.maximum_stable_timestep = (
            float("inf")
            if self._diffusion_coefficient == 0.0
            else 0.5 / (self._diffusion_coefficient * inverse_geometry_sum)
        )

        shape = (grid_height, grid_width)
        self.trail = np.zeros(shape, dtype=np.float32)
        self.alarm = np.zeros(shape, dtype=np.float32)
        self.accumulator = 0.0
        self.update_count = 0
        self.diffusion_substep_count = 0
        self.last_processed_updates = 0
        self.last_dropped_updates = 0
        self.last_dropped_time = 0.0
        self.total_processed_updates = 0
        self.total_dropped_updates = 0
        self.total_dropped_time = 0.0
        self._padded = np.empty((grid_height + 2, grid_width + 2), dtype=np.float32)
        self._next = np.empty(shape, dtype=np.float32)
        self._work = np.empty(shape, dtype=np.float32)

    def deposit(
        self,
        position: tuple[float, float],
        trail_amount: float = 0.0,
        alarm_amount: float = 0.0,
    ) -> None:
        x, y = self._validate_position(position)
        trail_amount = self._validate_amount(trail_amount, "trail_amount")
        alarm_amount = self._validate_amount(alarm_amount, "alarm_amount")
        indices = self._bilinear_indices(x, y)
        if indices is None:
            return
        if trail_amount > 0.0:
            self._deposit_into(self.trail, indices, trail_amount)
        if alarm_amount > 0.0:
            self._deposit_into(self.alarm, indices, alarm_amount)

    def deposit_many(
        self,
        positions: np.ndarray,
        trail_amounts: np.ndarray | None = None,
        alarm_amounts: np.ndarray | None = None,
    ) -> None:
        """Deposit arrays of nonnegative amounts with one clip per channel."""

        positions_array = self._validated_positions_array(positions, 2)
        count = positions_array.shape[0]
        trail = self._validated_amounts_array(trail_amounts, count, "trail_amounts")
        alarm = self._validated_amounts_array(alarm_amounts, count, "alarm_amounts")
        if count == 0 or (trail is None and alarm is None):
            return

        x = positions_array[:, 0]
        y = positions_array[:, 1]
        left, bottom, right, top = self.world_bounds
        if self.boundary_mode is PheromoneBoundaryMode.REFLECT:
            x = np.clip(x, left, right)
            y = np.clip(y, bottom, top)
        elif self.boundary_mode is PheromoneBoundaryMode.WRAP:
            x = np.mod(x - left, right - left) + left
            y = np.mod(y - bottom, top - bottom) + bottom
        else:
            inside = (x >= left) & (x <= right) & (y >= bottom) & (y <= top)
            if not np.any(inside):
                return
            x = x[inside]
            y = y[inside]
            if trail is not None:
                trail = trail[inside]
            if alarm is not None:
                alarm = alarm[inside]

        grid_x = (x - left) / (right - left) * (self.grid_width - 1)
        grid_y = (y - bottom) / (top - bottom) * (self.grid_height - 1)
        column0 = np.floor(grid_x).astype(np.intp)
        row0 = np.floor(grid_y).astype(np.intp)
        column1 = np.minimum(column0 + 1, self.grid_width - 1)
        row1 = np.minimum(row0 + 1, self.grid_height - 1)
        u = grid_x - column0
        v = grid_y - row0
        one_minus_u = 1.0 - u
        one_minus_v = 1.0 - v
        weights = (
            one_minus_u * one_minus_v,
            u * one_minus_v,
            one_minus_u * v,
            u * v,
        )
        flat_indices = np.concatenate(
            (
                row0 * self.grid_width + column0,
                row0 * self.grid_width + column1,
                row1 * self.grid_width + column0,
                row1 * self.grid_width + column1,
            )
        )
        if trail is not None:
            self._deposit_many_into(self.trail, flat_indices, weights, trail)
        if alarm is not None:
            self._deposit_many_into(self.alarm, flat_indices, weights, alarm)

    def sample(
        self,
        position: tuple[float, float],
        channel: PheromoneChannel | str,
    ) -> float:
        try:
            validated_channel = PheromoneChannel(channel)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"channel must be 'trail' or 'alarm', got {channel!r}."
            ) from error
        return (
            self.sample_trail(position)
            if validated_channel is PheromoneChannel.TRAIL
            else self.sample_alarm(position)
        )

    def sample_trail(self, position: tuple[float, float]) -> float:
        return self._sample_channel(self.trail, position)

    def sample_alarm(self, position: tuple[float, float]) -> float:
        return self._sample_channel(self.alarm, position)

    def sense(
        self,
        positions: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ],
    ) -> PheromoneSnapshot:
        if not isinstance(positions, (tuple, list)) or len(positions) != 3:
            raise ValueError("positions must contain here, forward-left, and forward-right.")
        values: list[tuple[float, float]] = []
        for position in positions:
            x, y = self._validate_position(position)
            indices = self._bilinear_indices(x, y)
            values.append(
                (0.0, 0.0)
                if indices is None
                else (
                    self._sample_grid(self.trail, indices),
                    self._sample_grid(self.alarm, indices),
                )
            )
        return PheromoneSnapshot(
            trail_here=values[0][0],
            trail_forward_left=values[1][0],
            trail_forward_right=values[2][0],
            alarm_here=values[0][1],
            alarm_forward_left=values[1][1],
            alarm_forward_right=values[2][1],
        )

    def sense_many(
        self,
        positions: np.ndarray,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sample N triples of sensor positions into columns matching snapshots."""

        positions_array = self._validated_positions_array(positions, 3)
        count = positions_array.shape[0]
        if out is None:
            result = np.empty((count, 6), dtype=np.float32)
        else:
            if (
                not isinstance(out, np.ndarray)
                or out.shape != (count, 6)
                or not np.issubdtype(out.dtype, np.floating)
            ):
                raise ValueError(
                    f"out must be a floating array with shape {(count, 6)}."
                )
            result = out
        if count == 0:
            return result

        flattened = positions_array.reshape(-1, 2)
        trail = self._sample_many_grid(self.trail, flattened).reshape(count, 3)
        alarm = self._sample_many_grid(self.alarm, flattened).reshape(count, 3)
        result[:, :3] = trail
        result[:, 3:] = alarm
        return result

    def accumulate(self, delta_time: float) -> int:
        delta_time = self._validated_timestep(delta_time)
        if delta_time == 0.0:
            self._set_last_catch_up(0, 0, 0.0)
            return 0
        accumulated = self.accumulator + delta_time
        if not isfinite(accumulated):
            raise ValueError("delta_time makes the pheromone accumulator nonfinite.")
        full_updates = int(floor((accumulated + 1e-12) / self._interval))
        remainder = accumulated - full_updates * self._interval
        if remainder < 0.0 and abs(remainder) <= 1e-12:
            remainder = 0.0
        if remainder >= self._interval:
            full_updates += 1
            remainder -= self._interval
        processed = min(full_updates, self._max_updates_per_tick)
        dropped = full_updates - processed
        for _ in range(processed):
            self.advance(self._interval)
        self.accumulator = 0.0 if abs(remainder) <= 1e-12 else remainder
        dropped_time = dropped * self._interval
        self._set_last_catch_up(processed, dropped, dropped_time)
        self.total_processed_updates += processed
        self.total_dropped_updates += dropped
        self.total_dropped_time += dropped_time
        return processed

    def advance(self, delta_time: float) -> None:
        delta_time = self._validated_timestep(delta_time)
        if delta_time == 0.0:
            return
        if self._diffusion_coefficient == 0.0:
            substeps = 1
        else:
            ratio = delta_time / self.maximum_stable_timestep
            substeps = max(1, ceil(ratio - self._STABILITY_TOLERANCE))
        step_delta_time = delta_time / substeps
        for _ in range(substeps):
            self._advance_stable_step(step_delta_time)
        self.update_count += 1

    def state_metadata(self) -> dict[str, object]:
        return {
            "grid_shape": (self.grid_height, self.grid_width),
            "world_bounds": self.world_bounds,
            "boundary_mode": self.boundary_mode.value,
        }

    def restore(
        self,
        trail: np.ndarray,
        alarm: np.ndarray,
        accumulator: float,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        restored_trail = self._validated_restore_grid(trail, "trail")
        restored_alarm = self._validated_restore_grid(alarm, "alarm")
        if (
            isinstance(accumulator, bool)
            or not isinstance(accumulator, (int, float))
            or not isfinite(accumulator)
            or not 0.0 <= accumulator < self._interval
        ):
            raise ValueError(
                "pheromone accumulator must be finite and within "
                f"[0, {self._interval}), got {accumulator!r}."
            )
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise ValueError("pheromone metadata must be a mapping.")
            expected_shape = (self.grid_height, self.grid_width)
            if tuple(metadata.get("grid_shape", ())) != expected_shape:
                raise ValueError("Saved pheromone metadata has an incompatible grid shape.")
            try:
                saved_bounds = tuple(float(value) for value in metadata["world_bounds"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "Saved pheromone metadata has invalid world bounds."
                ) from error
            if saved_bounds != self.world_bounds:
                raise ValueError("Saved pheromone metadata has incompatible world bounds.")
            try:
                saved_mode = PheromoneBoundaryMode(metadata["boundary_mode"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "Saved pheromone metadata has an invalid boundary mode."
                ) from error
            if saved_mode is not self.boundary_mode:
                raise ValueError("Saved pheromone metadata has an incompatible boundary mode.")

        np.copyto(self.trail, restored_trail)
        np.copyto(self.alarm, restored_alarm)
        self.accumulator = float(accumulator)

    def _advance_stable_step(self, delta_time: float) -> None:
        rx = (
            self._diffusion_coefficient
            * delta_time
            * self.inverse_cell_size_x_squared
        )
        ry = (
            self._diffusion_coefficient
            * delta_time
            * self.inverse_cell_size_y_squared
        )
        if rx + ry > 0.5 + self._STABILITY_TOLERANCE:
            raise ValueError(
                "Internal pheromone step is unstable: "
                f"rx + ry = {rx + ry!r}."
            )
        decay = exp(-self._evaporation_rate * delta_time)
        self._advance_grid(self.trail, rx, ry, decay)
        self._advance_grid(self.alarm, rx, ry, decay)
        self.diffusion_substep_count += 1

    def _advance_grid(
        self,
        grid: np.ndarray,
        rx: float,
        ry: float,
        decay: float,
    ) -> None:
        padded = self._padded
        padded[1:-1, 1:-1] = grid
        if self.boundary_mode is PheromoneBoundaryMode.REFLECT:
            padded[0, 1:-1] = grid[0, :]
            padded[-1, 1:-1] = grid[-1, :]
            padded[1:-1, 0] = grid[:, 0]
            padded[1:-1, -1] = grid[:, -1]
        elif self.boundary_mode is PheromoneBoundaryMode.WRAP:
            padded[0, 1:-1] = grid[-1, :]
            padded[-1, 1:-1] = grid[0, :]
            padded[1:-1, 0] = grid[:, -1]
            padded[1:-1, -1] = grid[:, 0]
        else:
            padded[0, 1:-1] = 0.0
            padded[-1, 1:-1] = 0.0
            padded[1:-1, 0] = 0.0
            padded[1:-1, -1] = 0.0

        next_grid = self._next
        work = self._work
        np.copyto(next_grid, grid)
        np.add(padded[1:-1, :-2], padded[1:-1, 2:], out=work)
        np.subtract(work, grid, out=work)
        np.subtract(work, grid, out=work)
        np.multiply(work, rx, out=work)
        np.add(next_grid, work, out=next_grid)
        np.add(padded[:-2, 1:-1], padded[2:, 1:-1], out=work)
        np.subtract(work, grid, out=work)
        np.subtract(work, grid, out=work)
        np.multiply(work, ry, out=work)
        np.add(next_grid, work, out=next_grid)
        np.multiply(next_grid, decay, out=next_grid)
        np.clip(next_grid, 0.0, self._maximum, out=next_grid)
        np.copyto(grid, next_grid)

    def _sample_channel(
        self,
        grid: np.ndarray,
        position: tuple[float, float],
    ) -> float:
        x, y = self._validate_position(position)
        indices = self._bilinear_indices(x, y)
        return 0.0 if indices is None else self._sample_grid(grid, indices)

    def _bilinear_indices(
        self,
        x: float,
        y: float,
    ) -> tuple[int, int, int, int, float, float] | None:
        left, bottom, right, top = self.world_bounds
        if self.boundary_mode is PheromoneBoundaryMode.REFLECT:
            x = max(left, min(right, x))
            y = max(bottom, min(top, y))
        elif self.boundary_mode is PheromoneBoundaryMode.WRAP:
            x = (x - left) % (right - left) + left
            y = (y - bottom) % (top - bottom) + bottom
        elif x < left or x > right or y < bottom or y > top:
            return None
        grid_x = (x - left) / (right - left) * (self.grid_width - 1)
        grid_y = (y - bottom) / (top - bottom) * (self.grid_height - 1)
        column0 = int(floor(grid_x))
        row0 = int(floor(grid_y))
        column1 = min(self.grid_width - 1, column0 + 1)
        row1 = min(self.grid_height - 1, row0 + 1)
        return column0, column1, row0, row1, grid_x - column0, grid_y - row0

    def _deposit_into(
        self,
        grid: np.ndarray,
        indices: tuple[int, int, int, int, float, float],
        amount: float,
    ) -> None:
        column0, column1, row0, row1, u, v = indices
        if column0 == column1 and row0 == row1:
            grid[row0, column0] = min(
                self._maximum, float(grid[row0, column0]) + amount
            )
            return
        if column0 == column1:
            self._add_clipped(grid, row0, column0, amount * (1.0 - v))
            self._add_clipped(grid, row1, column0, amount * v)
            return
        if row0 == row1:
            self._add_clipped(grid, row0, column0, amount * (1.0 - u))
            self._add_clipped(grid, row0, column1, amount * u)
            return
        self._add_clipped(grid, row0, column0, amount * (1.0 - u) * (1.0 - v))
        self._add_clipped(grid, row0, column1, amount * u * (1.0 - v))
        self._add_clipped(grid, row1, column0, amount * (1.0 - u) * v)
        self._add_clipped(grid, row1, column1, amount * u * v)

    def _add_clipped(
        self,
        grid: np.ndarray,
        row: int,
        column: int,
        amount: float,
    ) -> None:
        grid[row, column] = min(
            self._maximum,
            float(grid[row, column]) + amount,
        )

    def _deposit_many_into(
        self,
        grid: np.ndarray,
        flat_indices: np.ndarray,
        interpolation_weights: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        amounts: np.ndarray,
    ) -> None:
        contributions = np.concatenate(
            tuple(amounts * weights for weights in interpolation_weights)
        )
        accumulated = np.bincount(
            flat_indices,
            weights=contributions,
            minlength=grid.size,
        )
        np.add(grid, accumulated.reshape(grid.shape), out=grid, casting="unsafe")
        np.clip(grid, 0.0, self._maximum, out=grid)

    @staticmethod
    def _sample_grid(
        grid: np.ndarray,
        indices: tuple[int, int, int, int, float, float],
    ) -> float:
        column0, column1, row0, row1, u, v = indices
        return float(
            grid[row0, column0] * (1.0 - u) * (1.0 - v)
            + grid[row0, column1] * u * (1.0 - v)
            + grid[row1, column0] * (1.0 - u) * v
            + grid[row1, column1] * u * v
        )

    def _sample_many_grid(
        self,
        grid: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        left, bottom, right, top = self.world_bounds
        x = positions[:, 0]
        y = positions[:, 1]
        outside: np.ndarray | None = None
        if self.boundary_mode is PheromoneBoundaryMode.REFLECT:
            x = np.clip(x, left, right)
            y = np.clip(y, bottom, top)
        elif self.boundary_mode is PheromoneBoundaryMode.WRAP:
            x = np.mod(x - left, right - left) + left
            y = np.mod(y - bottom, top - bottom) + bottom
        else:
            outside = (x < left) | (x > right) | (y < bottom) | (y > top)
            x = np.clip(x, left, right)
            y = np.clip(y, bottom, top)
        grid_x = (x - left) / (right - left) * (self.grid_width - 1)
        grid_y = (y - bottom) / (top - bottom) * (self.grid_height - 1)
        column0 = np.floor(grid_x).astype(np.intp)
        row0 = np.floor(grid_y).astype(np.intp)
        column1 = np.minimum(column0 + 1, self.grid_width - 1)
        row1 = np.minimum(row0 + 1, self.grid_height - 1)
        u = grid_x - column0
        v = grid_y - row0
        result = (
            grid[row0, column0] * (1.0 - u) * (1.0 - v)
            + grid[row0, column1] * u * (1.0 - v)
            + grid[row1, column0] * (1.0 - u) * v
            + grid[row1, column1] * u * v
        ).astype(np.float32, copy=False)
        if outside is not None:
            result[outside] = 0.0
        return result

    def _validated_restore_grid(self, values: object, name: str) -> np.ndarray:
        try:
            restored = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Saved {name} pheromone grid is not numeric.") from error
        expected_shape = (self.grid_height, self.grid_width)
        if restored.shape != expected_shape:
            raise ValueError(
                f"Saved {name} pheromone grid must have shape {expected_shape}, "
                f"got {restored.shape}."
            )
        if not np.all(np.isfinite(restored)):
            raise ValueError(f"Saved {name} pheromone grid contains nonfinite values.")
        if np.any(restored < 0.0):
            raise ValueError(f"Saved {name} pheromone grid contains negative values.")
        if np.any(restored > self._maximum):
            raise ValueError(
                f"Saved {name} pheromone grid exceeds maximum concentration "
                f"{self._maximum}."
            )
        return restored

    @staticmethod
    def _validate_position(position: object) -> tuple[float, float]:
        if not isinstance(position, (tuple, list)) or len(position) != 2:
            raise ValueError("position must contain exactly two coordinates.")
        x, y = position
        if (
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not isfinite(x)
            or isinstance(y, bool)
            or not isinstance(y, (int, float))
            or not isfinite(y)
        ):
            raise ValueError(f"position coordinates must be finite, got {position!r}.")
        return float(x), float(y)

    @staticmethod
    def _validate_amount(value: object, name: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value < 0.0
        ):
            raise ValueError(f"{name} must be finite and nonnegative, got {value!r}.")
        return float(value)

    @staticmethod
    def _validated_positions_array(values: object, sensors: int) -> np.ndarray:
        try:
            positions = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("positions must be a numeric array.") from error
        expected_tail = (2,) if sensors == 2 else (3, 2)
        if positions.ndim != len(expected_tail) + 1 or positions.shape[1:] != expected_tail:
            shape_text = "(N, 2)" if sensors == 2 else "(N, 3, 2)"
            raise ValueError(f"positions must have shape {shape_text}, got {positions.shape}.")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions must contain only finite values.")
        return positions

    @staticmethod
    def _validated_amounts_array(
        values: object | None,
        count: int,
        name: str,
    ) -> np.ndarray | None:
        if values is None:
            return None
        try:
            amounts = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a numeric array.") from error
        if amounts.shape != (count,):
            raise ValueError(f"{name} must have shape {(count,)}, got {amounts.shape}.")
        if not np.all(np.isfinite(amounts)) or np.any(amounts < 0.0):
            raise ValueError(f"{name} must contain finite nonnegative values.")
        return amounts

    @staticmethod
    def _validated_timestep(value: object) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value < 0.0
        ):
            raise ValueError(f"delta_time must be finite and nonnegative, got {value!r}.")
        return float(value)

    def _set_last_catch_up(
        self,
        processed: int,
        dropped: int,
        dropped_time: float,
    ) -> None:
        self.last_processed_updates = processed
        self.last_dropped_updates = dropped
        self.last_dropped_time = dropped_time
