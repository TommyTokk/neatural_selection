from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from math import floor

import numpy as np
from opensimplex import OpenSimplex

from configs.sim_config import BiomeConfig


class Biome(IntEnum):
    PRAIRIE = 0
    BUSHES = 1
    FOREST = 2

    @property
    def label(self) -> str:
        if self is Biome.FOREST:
            return "Forest"
        if self is Biome.BUSHES:
            return "Bushes"
        return "Prairie"


@dataclass(frozen=True, slots=True)
class BiomeMap:
    biome_ids: np.ndarray
    render_rgba: np.ndarray
    world_bounds: tuple[float, float, float, float]
    area_shares: dict[Biome, float]
    spawn_weights: dict[Biome, float]
    uniform_spawn_chance: float
    max_spawn_attempts: int
    _fertility_grid: np.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        # Preserve the reference path's Python-float precision: even a float32
        # change in a continuous sensor can alter an evolved network's action.
        weights = np.asarray(
            [
                max(0.0, float(self.spawn_weights[Biome(index)]))
                for index in range(len(Biome))
            ],
            dtype=np.float64,
        )
        minimum = float(weights.min())
        maximum = float(weights.max())
        if maximum <= minimum:
            fertility_grid = np.ones_like(self.biome_ids, dtype=np.float64)
        else:
            fertility_grid = (
                weights[self.biome_ids] - minimum
            ) / (maximum - minimum)
            fertility_grid = np.asarray(fertility_grid, dtype=np.float64)
        fertility_grid.setflags(write=False)
        object.__setattr__(self, "_fertility_grid", fertility_grid)

    @property
    def grid_height(self) -> int:
        return int(self.biome_ids.shape[0])

    @property
    def grid_width(self) -> int:
        return int(self.biome_ids.shape[1])

    def biome_at(self, x: float, y: float) -> Biome:
        column, row = self._cell_for_world_position(x, y)
        return Biome(int(self.biome_ids[row, column]))

    def spawn_weight_at(self, x: float, y: float) -> float:
        return self.spawn_weights[self.biome_at(x, y)]

    def fertility_at(self, x: float, y: float) -> float:
        left, bottom, right, top = self.world_bounds
        cell_width = max(0.0001, right - left) / self.grid_width
        cell_height = max(0.0001, top - bottom) / self.grid_height
        grid_x = (x - left) / cell_width - 0.5
        grid_y = (y - bottom) / cell_height - 0.5

        column0 = floor(grid_x)
        row0 = floor(grid_y)
        column1 = column0 + 1
        row1 = row0 + 1
        u = grid_x - column0
        v = grid_y - row0

        column0 = max(0, min(self.grid_width - 1, column0))
        column1 = max(0, min(self.grid_width - 1, column1))
        row0 = max(0, min(self.grid_height - 1, row0))
        row1 = max(0, min(self.grid_height - 1, row1))

        c00 = float(self._fertility_grid[row0, column0])
        c10 = float(self._fertility_grid[row0, column1])
        c01 = float(self._fertility_grid[row1, column0])
        c11 = float(self._fertility_grid[row1, column1])
        normalized_fertility = (
            c00 * (1.0 - u) * (1.0 - v)
            + c10 * u * (1.0 - v)
            + c01 * (1.0 - u) * v
            + c11 * u * v
        )

        return max(0.0, min(1.0, normalized_fertility))

    def _cell_for_world_position(self, x: float, y: float) -> tuple[int, int]:
        left, bottom, right, top = self.world_bounds
        width = max(0.0001, right - left)
        height = max(0.0001, top - bottom)

        x_ratio = max(0.0, min(0.999999, (x - left) / width))
        y_ratio = max(0.0, min(0.999999, (y - bottom) / height))
        column = floor(x_ratio * self.grid_width)
        row = floor(y_ratio * self.grid_height)
        return column, row


class BiomeGenerationHandler:
    def __init__(self, config: BiomeConfig) -> None:
        self.config = config

    def generate(
        self,
        world_bounds: tuple[float, float, float, float],
    ) -> BiomeMap:
        noise_field = self._fractal_noise(world_bounds)
        biome_ids = self._classify_biomes(noise_field)
        return BiomeMap(
            biome_ids=biome_ids,
            render_rgba=self._render_rgba_for(biome_ids),
            world_bounds=world_bounds,
            area_shares=self._area_shares_for(biome_ids),
            spawn_weights={
                Biome.FOREST: max(0.0, self.config.forest_spawn_weight),
                Biome.BUSHES: max(0.0, self.config.bushes_spawn_weight),
                Biome.PRAIRIE: max(0.0, self.config.prairie_spawn_weight),
            },
            uniform_spawn_chance=max(
                0.0,
                min(1.0, self.config.uniform_spawn_chance),
            ),
            max_spawn_attempts=max(1, self.config.max_spawn_attempts),
        )

    def _fractal_noise(
        self,
        world_bounds: tuple[float, float, float, float],
    ) -> np.ndarray:
        left, bottom, right, top = world_bounds
        world_width = max(0.0001, right - left)
        world_height = max(0.0001, top - bottom)
        grid_width = max(2, self.config.grid_width)
        grid_height = max(2, self.config.grid_height)

        noise = OpenSimplex(self.config.seed)
        xs = np.linspace(0.0, world_width, grid_width, dtype=np.float32)
        ys = np.linspace(0.0, world_height, grid_height, dtype=np.float32)
        xs = (xs + left) / max(0.0001, self.config.noise_scale)
        ys = (ys + bottom) / max(0.0001, self.config.noise_scale)

        total = np.zeros((grid_height, grid_width), dtype=np.float32)
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0

        for _ in range(max(1, self.config.octaves)):
            layer = noise.noise2array(xs * frequency, ys * frequency).astype(np.float32)
            total += layer * amplitude
            max_value += amplitude
            amplitude *= self.config.persistence
            frequency *= self.config.lacunarity

        if max_value <= 0.0:
            return total
        return total / max_value

    def _classify_biomes(self, noise_field: np.ndarray) -> np.ndarray:
        prairie_share = max(0.0, self.config.prairie_target_share)
        bushes_share = max(0.0, self.config.bushes_target_share)
        forest_share = max(0.0, self.config.forest_target_share)
        total_share = max(0.0001, prairie_share + bushes_share + forest_share)
        prairie_quantile = prairie_share / total_share
        bushes_quantile = (prairie_share + bushes_share) / total_share

        prairie_threshold, bushes_threshold = np.quantile(
            noise_field,
            [prairie_quantile, bushes_quantile],
        )

        biome_ids = np.empty(noise_field.shape, dtype=np.uint8)
        biome_ids[noise_field < prairie_threshold] = int(Biome.PRAIRIE)
        biome_ids[
            (noise_field >= prairie_threshold) & (noise_field < bushes_threshold)
        ] = int(Biome.BUSHES)
        biome_ids[noise_field >= bushes_threshold] = int(Biome.FOREST)
        return biome_ids

    def _render_rgba_for(self, biome_ids: np.ndarray) -> np.ndarray:
        colors = np.array(
            [
                self.config.prairie_color,
                self.config.bushes_color,
                self.config.forest_color,
            ],
            dtype=np.uint8,
        )
        return colors[biome_ids]

    def _area_shares_for(self, biome_ids: np.ndarray) -> dict[Biome, float]:
        total_cells = max(1, int(biome_ids.size))
        counts = np.bincount(biome_ids.reshape(-1), minlength=len(Biome))
        return {biome: float(counts[int(biome)] / total_cells) for biome in Biome}
