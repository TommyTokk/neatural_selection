from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, floor, hypot, sin
from typing import Iterable

import numpy as np

from configs.sim_config import CommunicationConfig


@dataclass(frozen=True, slots=True)
class AcousticSignal:
    emitter_id: int
    position: tuple[float, float]
    strength: float
    tone: float


@dataclass(frozen=True, slots=True)
class AcousticSnapshot:
    strength: float = 0.0
    direction_sin: float = 0.0
    direction_cos: float = 0.0
    tone: float = 0.0
    source_id: int | None = None
    source_position: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class PheromoneSnapshot:
    trail_here: float = 0.0
    trail_forward_left: float = 0.0
    trail_forward_right: float = 0.0
    alarm_here: float = 0.0
    alarm_forward_left: float = 0.0
    alarm_forward_right: float = 0.0


class AcousticSystem:
    """A bounded set of instantaneous broadcasts, one per active emitter."""

    def __init__(self, config: CommunicationConfig) -> None:
        self.config = config
        self.signals: dict[int, AcousticSignal] = {}

    def replace_signals(self, signals: Iterable[AcousticSignal]) -> None:
        """
        Replace the current set of signals with a new set, filtering out any
        signals that are below the configured minimum emission strength.
        """

        # Get the minimum emission strength from the configuration, ensuring it's non-negative.
        minimum = max(0.0, self.config.acoustic_min_emission)

        # Create a new dictionary of signals, filtering out any signals that are below the minimum strength.
        self.signals = {
            signal.emitter_id: signal
            for signal in signals
            if signal.strength >= minimum
        }

    def remove_emitter(self, emitter_id: int) -> None:
        self.signals.pop(emitter_id, None)

    def sense(
        self,
        receiver_id: int,
        position: tuple[float, float],
        heading: float,
    ) -> AcousticSnapshot:
        """
        Determine the strongest audible signal at a given position and heading,
        excluding any signals emitted by the receiver itself. Returns an
        AcousticSnapshot containing the strength, direction, tone, and source
        information of the strongest signal, or an empty snapshot if no signals
        are audible.
        
        Args:
            receiver_id (int): The ID of the receiver to exclude from sensing.
            position (tuple[float, float]): The (x, y) position of the receiver.
            heading (float): The heading of the receiver in radians, where 0 is along the positive x-axis and pi/2 is along the positive y-axis.

        Returns:
            AcousticSnapshot: A snapshot of the strongest audible signal, or an empty snapshot if no signals are audible.
        """
        # Determine the maximum range for acoustic signals based on the configuration.
        max_range = max(0.0, self.config.acoustic_range)
        if max_range <= 0.0:
            return AcousticSnapshot()

        # Initialize a variable to keep track of the best signal found so far. The tuple contains:
        # (heard_strength, -emitter_id, signal, dx, dy)
        best: tuple[float, int, AcousticSignal, float, float] | None = None

        # Iterate through all signals in the system to find the strongest audible signal.
        for emitter_id, signal in self.signals.items():
            # Skip any signals emitted by the receiver itself.
            if emitter_id == receiver_id:
                continue

            # Calculate the distance from the receiver to the signal emitter.
            dx = signal.position[0] - position[0]
            dy = signal.position[1] - position[1]
            # Calculate the Euclidean distance to the signal emitter.
            distance = hypot(dx, dy)
            # If the distance exceeds the maximum range, skip this signal.
            if distance > max_range:
                continue

            # Calculate the attenuation of the signal based on the distance and maximum range.
            attenuation = max(0.0, 1.0 - distance / max_range) ** 2
            # Calculate the effective heard strength of the signal after applying attenuation.
            heard_strength = max(0.0, min(1.0, signal.strength)) * attenuation
            # If the heard strength is below the minimum emission threshold, skip this signal.
            if heard_strength < self.config.acoustic_min_emission:
                continue

            # Create a candidate tuple for comparison, using -emitter_id to break ties in favor of lower IDs.
            candidate = (heard_strength, -emitter_id, signal, dx, dy)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        # If no audible signals were found, return an empty AcousticSnapshot.
        if best is None:
            return AcousticSnapshot()

        # Unpack the best signal information and calculate the direction of the signal relative to the receiver's heading.
        heard_strength, _tie_breaker, signal, dx, dy = best
        distance = hypot(dx, dy)
        
        # If the distance is effectively zero, set the direction to zero to avoid division by zero. Otherwise, calculate the direction using dot products with the receiver's forward and left unit vectors.
        if distance <= 1e-12:
            direction_sin = 0.0
            direction_cos = 0.0
        else:
            # Dot products with the receiver's forward and left unit vectors
            # avoid an angular wrap and an unnecessary atan2 call.
            unit_x = dx / distance
            unit_y = dy / distance
            direction_cos = unit_x * cos(heading) + unit_y * sin(heading)
            direction_sin = unit_y * cos(heading) - unit_x * sin(heading)

        return AcousticSnapshot(
            strength=heard_strength,
            direction_sin=max(-1.0, min(1.0, direction_sin)),
            direction_cos=max(-1.0, min(1.0, direction_cos)),
            tone=max(-1.0, min(1.0, signal.tone)),
            source_id=signal.emitter_id,
            source_position=signal.position,
        )


class PheromoneSystem:
    """Two slowly changing, vectorized pheromone concentration fields."""

    def __init__(
        self,
        config: CommunicationConfig,
        grid_width: int,
        grid_height: int,
        world_bounds: tuple[float, float, float, float],
    ) -> None:
        self.config = config
        self.grid_width = max(2, int(grid_width))
        self.grid_height = max(2, int(grid_height))
        self.world_bounds = tuple(float(value) for value in world_bounds)
        self._validate_config()
        shape = (self.grid_height, self.grid_width)
        self.trail = np.zeros(shape, dtype=np.float32)
        self.alarm = np.zeros(shape, dtype=np.float32)
        self.accumulator = 0.0
        self.update_count = 0
        self._padded = np.empty(
            (self.grid_height + 2, self.grid_width + 2),
            dtype=np.float32,
        )
        self._next = np.empty(shape, dtype=np.float32)

    def _validate_config(self) -> None:
        interval = float(self.config.pheromone_update_interval)
        coefficient = float(self.config.pheromone_diffusion_coefficient)
        if interval <= 0.0:
            raise ValueError("Pheromone update interval must be positive.")
        if coefficient < 0.0:
            raise ValueError("Pheromone diffusion coefficient cannot be negative.")
        if coefficient * interval > 0.25 + 1e-12:
            raise ValueError(
                "Unstable pheromone diffusion: coefficient * interval must be <= 0.25."
            )

    def deposit(
        self,
        position: tuple[float, float],
        trail_amount: float = 0.0,
        alarm_amount: float = 0.0,
    ) -> None:
        indices = self._bilinear_indices(*position)
        if trail_amount > 0.0:
            self._deposit_into(self.trail, indices, trail_amount)
        if alarm_amount > 0.0:
            self._deposit_into(self.alarm, indices, alarm_amount)

    def sample(self, position: tuple[float, float], channel: str) -> float:
        grid = self.trail if channel == "trail" else self.alarm
        return self._sample_grid(grid, self._bilinear_indices(*position))

    def sense(
        self,
        positions: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ],
    ) -> PheromoneSnapshot:
        here, forward_left, forward_right = positions
        return PheromoneSnapshot(
            trail_here=self.sample(here, "trail"),
            trail_forward_left=self.sample(forward_left, "trail"),
            trail_forward_right=self.sample(forward_right, "trail"),
            alarm_here=self.sample(here, "alarm"),
            alarm_forward_left=self.sample(forward_left, "alarm"),
            alarm_forward_right=self.sample(forward_right, "alarm"),
        )

    def accumulate(self, delta_time: float) -> int:
        self.accumulator += max(0.0, float(delta_time))
        interval = self.config.pheromone_update_interval
        updates = 0
        while self.accumulator + 1e-12 >= interval:
            self.advance(interval)
            self.accumulator -= interval
            updates += 1
        if abs(self.accumulator) < 1e-12:
            self.accumulator = 0.0
        return updates

    def advance(self, delta_time: float) -> None:
        decay = exp(-max(0.0, self.config.pheromone_evaporation_rate) * delta_time)
        alpha = self.config.pheromone_diffusion_coefficient * delta_time
        self._advance_grid(self.trail, alpha, decay)
        self._advance_grid(self.alarm, alpha, decay)
        self.update_count += 1

    def restore(
        self,
        trail: np.ndarray,
        alarm: np.ndarray,
        accumulator: float,
    ) -> None:
        expected_shape = (self.grid_height, self.grid_width)
        restored_trail = np.asarray(trail, dtype=np.float32)
        restored_alarm = np.asarray(alarm, dtype=np.float32)
        if (
            restored_trail.shape != expected_shape
            or restored_alarm.shape != expected_shape
        ):
            raise ValueError("Saved pheromone grids do not match the configured grid shape.")
        np.copyto(self.trail, restored_trail)
        np.copyto(self.alarm, restored_alarm)
        self.accumulator = (
            max(0.0, float(accumulator))
            % self.config.pheromone_update_interval
        )

    def _advance_grid(self, grid: np.ndarray, alpha: float, decay: float) -> None:
        padded = self._padded
        padded[1:-1, 1:-1] = grid
        padded[0, 1:-1] = grid[0, :]
        padded[-1, 1:-1] = grid[-1, :]
        padded[1:-1, 0] = grid[:, 0]
        padded[1:-1, -1] = grid[:, -1]
        padded[0, 0] = grid[0, 0]
        padded[0, -1] = grid[0, -1]
        padded[-1, 0] = grid[-1, 0]
        padded[-1, -1] = grid[-1, -1]
        self._next[:] = grid + alpha * (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            - 4.0 * grid
        )
        self._next *= decay
        np.clip(
            self._next,
            0.0,
            max(0.0, self.config.pheromone_max_concentration),
            out=grid,
        )

    def _bilinear_indices(
        self,
        x: float,
        y: float,
    ) -> tuple[int, int, int, int, float, float]:
        left, bottom, right, top = self.world_bounds
        x_ratio = max(0.0, min(1.0, (x - left) / max(1e-12, right - left)))
        y_ratio = max(0.0, min(1.0, (y - bottom) / max(1e-12, top - bottom)))
        grid_x = x_ratio * (self.grid_width - 1)
        grid_y = y_ratio * (self.grid_height - 1)
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
        weights = (
            (row0, column0, (1.0 - u) * (1.0 - v)),
            (row0, column1, u * (1.0 - v)),
            (row1, column0, (1.0 - u) * v),
            (row1, column1, u * v),
        )
        maximum = max(0.0, self.config.pheromone_max_concentration)
        for row, column, weight in weights:
            grid[row, column] = min(maximum, float(grid[row, column]) + amount * weight)

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
