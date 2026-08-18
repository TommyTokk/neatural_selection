from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from math import ceil, hypot
from typing import TYPE_CHECKING, Any

import numpy as np

from configs.sim_config import BiomeFoodClusterProfile, FoodClusterConfig, FoodConfig
from src.biome import Biome, BiomeMap
from src.food import Food

if TYPE_CHECKING:
    from src.world import World


@dataclass(slots=True)
class FoodPatch:
    id: int
    biome_type: Biome
    center_pos: tuple[float, float]
    configured_radius: float
    effective_radius: float
    target_food_count: int
    active_food_ids: set[int] = field(default_factory=set)
    is_depleted: bool = True
    cooldown_timer: int = 0
    has_spawned: bool = False
    relocated: bool = True


class FoodClusterManager:
    """Own biome-specific food patches alongside independent food spawning."""

    def __init__(
        self,
        config: FoodClusterConfig,
        food_config: FoodConfig,
        biome_map: BiomeMap,
        seed: int,
    ) -> None:
        self.config = config
        self.food_config = food_config
        self.biome_map = biome_map
        self.rng = np.random.default_rng(seed)
        self.patches: dict[int, FoodPatch] = {}
        self._next_patch_id = 1
        self._cluster_spawn_credit = 0.0
        self._low_food_burst_credit = 0.0
        self._emergency_burst_armed = False
        self._desired_patch_count_cache: dict[int, dict[Biome, int]] = {}
        self._biome_cells = {
            biome: np.argwhere(biome_map.biome_ids == int(biome))
            for biome in Biome
        }
        self._diagnostics = {
            "sampling_retries": 0,
            "radius_reductions": 0,
            "rejected_centers": 0,
            "deferred_patches": 0,
            "relocations": 0,
        }

    @staticmethod
    def mode_targets(total_items: int, cluster_share: float) -> tuple[int, int]:
        total = max(0, int(total_items))
        share = max(0.0, min(1.0, float(cluster_share)))
        independent = max(0, min(total, round(total * (1.0 - share))))
        return independent, total - independent

    def initialize_patches(self, world: World, cluster_item_count: int) -> None:
        self._ensure_patch_capacity(max(0, cluster_item_count))
        remaining = max(0, cluster_item_count)
        for patch in self.patches.values():
            if remaining <= 0:
                break
            patch.target_food_count = min(patch.target_food_count, remaining)
            foods = self._build_patch_food(patch, world)
            if foods is None:
                self._diagnostics["deferred_patches"] += 1
                continue
            world._add_foods(foods)
            patch.active_food_ids = {food.id for food in foods}
            patch.is_depleted = False
            patch.has_spawned = True
            remaining -= len(foods)

    def update(self, world: World, delta_ticks: int = 1) -> None:
        self._synchronize_membership(world)
        self._advance_cooldowns(max(0, int(delta_ticks)))
        self._deplete_sparse_patches(world)

        _, cluster_target = self.mode_targets(
            world.food_spawner.config.max_food_items,
            world.live_food_config.cluster_spawn_share,
        )
        self._ensure_patch_capacity(cluster_target)
        active_cluster_count = sum(
            1 for food in world.foods if food.cluster_id is not None
        )
        if active_cluster_count >= cluster_target:
            self._cluster_spawn_credit = min(
                self._cluster_spawn_credit,
                float(self._largest_patch_size()),
            )
            return

        pressure = world.food_spawner.food_regrowth_pressure(
            len(world.foods),
            world.food_spawner.food_capacity(),
        )
        spawn_rate = world.food_spawner._spawn_rate_per_second(
            world._active_species_count(),
            pressure,
        )
        self._cluster_spawn_credit += (
            world.fixed_timestep
            * spawn_rate
            * world.live_food_config.cluster_spawn_share
        )
        self._accrue_emergency_credit(world, cluster_target, active_cluster_count)

        active_patch_counts = Counter(
            patch.biome_type
            for patch in self.patches.values()
            if not patch.is_depleted
        )
        pending = [
            patch
            for patch in self.patches.values()
            if patch.is_depleted and patch.cooldown_timer <= 0
        ]
        while pending:
            patch = min(
                pending,
                key=lambda item: (
                    active_patch_counts[item.biome_type] > 0,
                    (
                        active_patch_counts[item.biome_type]
                        / max(
                            0.000001,
                            self._profile(item.biome_type).cluster_count_weight,
                        )
                    ),
                    item.biome_type is not Biome.FOREST,
                    item.id,
                ),
            )
            pending.remove(patch)
            count = patch.target_food_count
            if count <= 0 or self._cluster_spawn_credit + 1e-12 < count:
                continue
            global_slots = (
                world.food_spawner.config.max_food_items - len(world.foods)
            )
            cluster_slots = cluster_target - active_cluster_count
            if count > global_slots or count > cluster_slots:
                continue
            foods = self._build_patch_food(
                patch,
                world,
                energy_budget=world._available_biomass(),
            )
            if foods is None:
                self._diagnostics["deferred_patches"] += 1
                continue
            actual_energy = sum(food.max_energy or 0.0 for food in foods)
            if actual_energy > world._available_biomass() + 1e-12:
                continue
            world._add_foods(foods)
            patch.active_food_ids = {food.id for food in foods}
            patch.is_depleted = False
            patch.has_spawned = True
            self._cluster_spawn_credit -= len(foods)
            active_cluster_count += len(foods)
            active_patch_counts[patch.biome_type] += 1

    def on_food_consumed(self, food_id: int, eater_id: int, world: World) -> None:
        """Compatibility notification; committed depletion is handled on removal."""
        del eater_id
        if any(food.id == food_id for food in world.foods):
            return
        self.on_food_removed(food_id)

    def on_food_removed(self, food_id: int) -> None:
        for patch in self.patches.values():
            if food_id in patch.active_food_ids:
                patch.active_food_ids.discard(food_id)
                return

    def spawn_patch_food(self, patch: FoodPatch, world: World) -> list[Food]:
        foods = self._build_patch_food(patch, world)
        if foods is None:
            return []
        world._add_foods(foods)
        patch.active_food_ids = {food.id for food in foods}
        patch.is_depleted = False
        patch.has_spawned = True
        return foods

    def diagnostics(self, world: World | None = None) -> dict[str, Any]:
        result: dict[str, Any] = dict(self._diagnostics)
        result["spawn_credit"] = self._cluster_spawn_credit
        result["patch_states"] = {
            "active": sum(not patch.is_depleted for patch in self.patches.values()),
            "cooldown": sum(
                patch.is_depleted and patch.cooldown_timer > 0
                for patch in self.patches.values()
            ),
            "ready": sum(
                patch.is_depleted and patch.cooldown_timer <= 0
                for patch in self.patches.values()
            ),
        }
        result["by_biome"] = {
            biome.label: {
                "patches": sum(
                    patch.biome_type is biome for patch in self.patches.values()
                ),
                "active_patches": sum(
                    patch.biome_type is biome and not patch.is_depleted
                    for patch in self.patches.values()
                ),
                "active_pellets": sum(
                    len(patch.active_food_ids)
                    for patch in self.patches.values()
                    if patch.biome_type is biome
                ),
            }
            for biome in Biome
        }
        cluster_target = sum(
            patch.target_food_count for patch in self.patches.values()
        )
        if world is not None:
            _, cluster_target = self.mode_targets(
                world.food_spawner.config.max_food_items,
                world.live_food_config.cluster_spawn_share,
            )
        desired_counts = self._desired_patch_counts(cluster_target)
        for biome in Biome:
            result["by_biome"][biome.label]["desired_patches"] = (
                desired_counts.get(biome, 0)
            )
        bush_patches = result["by_biome"][Biome.BUSHES.label]["patches"]
        forest_patches = result["by_biome"][Biome.FOREST.label]["patches"]
        result["bush_to_forest_patch_ratio"] = (
            bush_patches / forest_patches if forest_patches > 0 else None
        )
        if world is not None:
            independent = sum(food.cluster_id is None for food in world.foods)
            clustered = len(world.foods) - independent
            result["independent_items"] = independent
            result["clustered_items"] = clustered
            total_items = max(1, len(world.foods))
            result["independent_share"] = independent / total_items
            result["clustered_share"] = clustered / total_items
            patches_by_id = self.patches
            for biome in Biome:
                biome_foods = [
                    food
                    for food in world.foods
                    if food.cluster_id in patches_by_id
                    and patches_by_id[food.cluster_id].biome_type is biome
                ]
                result["by_biome"][biome.label]["energy_remaining"] = sum(
                    food.energy_value for food in biome_foods
                )
                result["by_biome"][biome.label]["max_energy"] = sum(
                    food.original_energy_value for food in biome_foods
                )
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "next_patch_id": self._next_patch_id,
            "cluster_spawn_credit": self._cluster_spawn_credit,
            "low_food_burst_credit": self._low_food_burst_credit,
            "emergency_burst_armed": self._emergency_burst_armed,
            "rng_state": self.rng.bit_generator.state,
            "diagnostics": dict(self._diagnostics),
            "patches": [
                {
                    **asdict(patch),
                    "biome_type": int(patch.biome_type),
                    "active_food_ids": sorted(patch.active_food_ids),
                }
                for patch in self.patches.values()
            ],
        }

    def restore_state(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        self._next_patch_id = max(1, int(state.get("next_patch_id", 1)))
        self._cluster_spawn_credit = max(
            0.0, float(state.get("cluster_spawn_credit", 0.0))
        )
        self._low_food_burst_credit = max(
            0.0, float(state.get("low_food_burst_credit", 0.0))
        )
        self._emergency_burst_armed = bool(
            state.get("emergency_burst_armed", False)
        )
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state
        self._diagnostics.update(state.get("diagnostics", {}))
        self.patches = {}
        for value in state.get("patches", []):
            patch = FoodPatch(
                id=int(value["id"]),
                biome_type=Biome(int(value["biome_type"])),
                center_pos=tuple(value["center_pos"]),
                configured_radius=float(value["configured_radius"]),
                effective_radius=float(value["effective_radius"]),
                target_food_count=max(1, int(value["target_food_count"])),
                active_food_ids={int(item) for item in value["active_food_ids"]},
                is_depleted=bool(value["is_depleted"]),
                cooldown_timer=max(0, int(value["cooldown_timer"])),
                has_spawned=bool(value.get("has_spawned", True)),
                relocated=bool(value.get("relocated", True)),
            )
            self.patches[patch.id] = patch

    def _ensure_patch_capacity(self, target_items: int) -> None:
        """Reconcile inventory without disturbing healthy active patches."""
        target = max(0, int(target_items))
        desired = self._desired_patch_counts(target)
        capacity = sum(patch.target_food_count for patch in self.patches.values())
        counts = Counter(patch.biome_type for patch in self.patches.values())

        # Old prairie inventory and excess forest-heavy inventory retire only
        # after their active members naturally deplete.
        for biome in (Biome.PRAIRIE, Biome.FOREST, Biome.BUSHES):
            surplus = max(0, counts[biome] - desired.get(biome, 0))
            candidates = sorted(
                (
                    patch
                    for patch in self.patches.values()
                    if patch.biome_type is biome and patch.is_depleted
                ),
                key=lambda patch: patch.id,
                reverse=True,
            )
            for patch in candidates[:surplus]:
                self.patches.pop(patch.id, None)
                capacity -= patch.target_food_count
                counts[biome] -= 1

        # Live cap reductions also converge through depleted inventory only.
        if capacity > target:
            candidates = sorted(
                (patch for patch in self.patches.values() if patch.is_depleted),
                key=lambda patch: (
                    counts[patch.biome_type]
                    <= desired.get(patch.biome_type, 0),
                    -patch.id,
                ),
            )
            for patch in candidates:
                if capacity <= target:
                    break
                self.patches.pop(patch.id, None)
                capacity -= patch.target_food_count
                counts[patch.biome_type] -= 1

        remaining = target - capacity
        if remaining <= 0:
            return

        additions = self._plan_patch_additions(remaining, counts, desired)
        for biome, count in additions:
            profile = self._profile(biome)
            configured_radius = float(
                self.rng.uniform(*profile.spread_radius)
                if profile.spread_radius[1] > profile.spread_radius[0]
                else profile.spread_radius[0]
            )
            center = self._sample_center(biome, None, configured_radius)
            if center is None:
                self._diagnostics["deferred_patches"] += 1
                break
            patch = FoodPatch(
                id=self._next_patch_id,
                biome_type=biome,
                center_pos=center,
                configured_radius=configured_radius,
                effective_radius=configured_radius,
                target_food_count=count,
            )
            self.patches[patch.id] = patch
            self._next_patch_id += 1

    def _eligible_patch_biomes(self) -> tuple[Biome, ...]:
        return tuple(
            biome
            for biome in (Biome.BUSHES, Biome.FOREST)
            if self._profile(biome).cluster_count_weight > 0.0
            and len(self._biome_cells[biome]) > 0
        )

    def _desired_patch_counts(self, target_items: int) -> dict[Biome, int]:
        """Return the feasible patch-count mix nearest configured frequency."""
        target = max(0, int(target_items))
        cached = self._desired_patch_count_cache.get(target)
        if cached is not None:
            return dict(cached)
        eligible = self._eligible_patch_biomes()
        empty = {biome: 0 for biome in Biome}
        if not eligible or target <= 0:
            self._desired_patch_count_cache[target] = empty
            return dict(empty)

        required = (
            eligible
            if (
                Biome.BUSHES in eligible
                and Biome.FOREST in eligible
                and target
                >= (
                    self.config.bushes.pellets_per_cluster[0]
                    + self.config.forest.pellets_per_cluster[0]
                )
            )
            else ()
        )
        candidates = self._patch_count_candidates(target, eligible, required)
        if not candidates:
            self._desired_patch_count_cache[target] = empty
            return dict(empty)
        bushes, forests = min(
            candidates,
            key=lambda counts: self._patch_count_score(counts, target, eligible),
        )
        result = {
            Biome.PRAIRIE: 0,
            Biome.BUSHES: bushes,
            Biome.FOREST: forests,
        }
        self._desired_patch_count_cache[target] = result
        return dict(result)

    def _patch_count_candidates(
        self,
        target: int,
        eligible: tuple[Biome, ...],
        required: tuple[Biome, ...] = (),
    ) -> list[tuple[int, int]]:
        bush_low, bush_high = self.config.bushes.pellets_per_cluster
        forest_low, forest_high = self.config.forest.pellets_per_cluster
        max_bushes = target // bush_low if Biome.BUSHES in eligible else 0
        max_forests = target // forest_low if Biome.FOREST in eligible else 0
        candidates: list[tuple[int, int]] = []
        for bushes in range(max_bushes + 1):
            for forests in range(max_forests + 1):
                if bushes + forests == 0:
                    continue
                if Biome.BUSHES in required and bushes == 0:
                    continue
                if Biome.FOREST in required and forests == 0:
                    continue
                minimum = bushes * bush_low + forests * forest_low
                maximum = bushes * bush_high + forests * forest_high
                if minimum <= target <= maximum:
                    candidates.append((bushes, forests))
        return candidates

    def _patch_count_score(
        self,
        counts: tuple[int, int],
        target: int,
        eligible: tuple[Biome, ...],
    ) -> tuple[float, float, int]:
        bushes, forests = counts
        total_patches = bushes + forests
        bush_weight = (
            self.config.bushes.cluster_count_weight
            if Biome.BUSHES in eligible
            else 0.0
        )
        forest_weight = (
            self.config.forest.cluster_count_weight
            if Biome.FOREST in eligible
            else 0.0
        )
        total_weight = bush_weight + forest_weight
        desired_bush_share = (
            bush_weight / total_weight if total_weight > 0.0 else 0.0
        )
        frequency_error = abs((bushes / total_patches) - desired_bush_share)
        bush_midpoint = sum(self.config.bushes.pellets_per_cluster) * 0.5
        forest_midpoint = sum(self.config.forest.pellets_per_cluster) * 0.5
        expected_capacity = bushes * bush_midpoint + forests * forest_midpoint
        return frequency_error, abs(expected_capacity - target), total_patches

    def _plan_patch_additions(
        self,
        remaining: int,
        current_counts: Counter[Biome],
        desired_counts: dict[Biome, int],
    ) -> list[tuple[Biome, int]]:
        eligible = self._eligible_patch_biomes()
        required = tuple(
            biome
            for biome in eligible
            if desired_counts.get(biome, 0) > 0 and current_counts[biome] == 0
        )
        required_minimum = sum(
            self._profile(biome).pellets_per_cluster[0] for biome in required
        )
        if required_minimum > remaining:
            required = ()
        candidates = self._patch_count_candidates(remaining, eligible, required)
        if not candidates:
            return []

        total_target = remaining + sum(
            patch.target_food_count for patch in self.patches.values()
        )

        def addition_score(counts: tuple[int, int]) -> tuple[int, float, float, int]:
            bushes, forests = counts
            final_counts = (
                current_counts[Biome.BUSHES] + bushes,
                current_counts[Biome.FOREST] + forests,
            )
            count_error = (
                abs(final_counts[0] - desired_counts.get(Biome.BUSHES, 0))
                + abs(final_counts[1] - desired_counts.get(Biome.FOREST, 0))
            )
            frequency, expected, total = self._patch_count_score(
                final_counts,
                total_target,
                eligible,
            )
            return count_error, frequency, expected, total

        bush_count, forest_count = min(candidates, key=addition_score)
        biome_counts = {
            Biome.BUSHES: bush_count,
            Biome.FOREST: forest_count,
        }
        sizes = self._allocate_patch_sizes(remaining, biome_counts)
        ordered_biomes = self._frequency_order(biome_counts)
        sizes_by_biome = {
            biome: iter(sizes[biome])
            for biome in (Biome.BUSHES, Biome.FOREST)
        }
        return [
            (biome, next(sizes_by_biome[biome]))
            for biome in ordered_biomes
        ]

    def _allocate_patch_sizes(
        self,
        target: int,
        biome_counts: dict[Biome, int],
    ) -> dict[Biome, list[int]]:
        entries: list[tuple[Biome, int, int]] = []
        for biome in (Biome.BUSHES, Biome.FOREST):
            low, high = self._profile(biome).pellets_per_cluster
            entries.extend((biome, low, high) for _ in range(biome_counts[biome]))

        sizes = [low for _biome, low, _high in entries]
        remaining = target - sum(sizes)
        order = list(map(int, self.rng.permutation(len(entries))))
        remaining_capacity = sum(high - low for _biome, low, high in entries)
        for index in order:
            _biome, low, high = entries[index]
            capacity = high - low
            remaining_capacity -= capacity
            minimum_take = max(0, remaining - remaining_capacity)
            maximum_take = min(capacity, remaining)
            take = int(self.rng.integers(minimum_take, maximum_take + 1))
            sizes[index] += take
            remaining -= take

        result = {Biome.BUSHES: [], Biome.FOREST: []}
        for (biome, _low, _high), size in zip(entries, sizes, strict=True):
            result[biome].append(size)
        return result

    def _frequency_order(self, counts: dict[Biome, int]) -> list[Biome]:
        """Interleave patch types while preserving their planned count ratio."""
        remaining = dict(counts)
        placed = {Biome.BUSHES: 0, Biome.FOREST: 0}
        total = sum(remaining.values())
        ordered: list[Biome] = []
        while len(ordered) < total:
            next_index = len(ordered) + 1
            choices = [biome for biome, count in remaining.items() if count > 0]
            biome = max(
                choices,
                key=lambda item: (
                    counts[item] * next_index / total - placed[item],
                    item is Biome.FOREST,
                ),
            )
            ordered.append(biome)
            placed[biome] += 1
            remaining[biome] -= 1
        return ordered

    def _profile(self, biome: Biome) -> BiomeFoodClusterProfile:
        if biome is Biome.FOREST:
            return self.config.forest
        if biome is Biome.BUSHES:
            return self.config.bushes
        return self.config.prairie

    def _sample_center(
        self,
        biome: Biome,
        previous: tuple[float, float] | None,
        radius: float,
    ) -> tuple[float, float] | None:
        cells = self._biome_cells[biome]
        if len(cells) == 0:
            return None
        left, bottom, right, top = self.biome_map.world_bounds
        cell_width = (right - left) / self.biome_map.grid_width
        cell_height = (top - bottom) / self.biome_map.grid_height
        minimum_distance = max(
            self.config.minimum_relocation_distance,
            radius,
        )
        for _ in range(self.config.center_sampling_attempts):
            row, column = cells[int(self.rng.integers(0, len(cells)))]
            x = left + (float(column) + self.rng.random()) * cell_width
            y = bottom + (float(row) + self.rng.random()) * cell_height
            if previous is not None and hypot(x - previous[0], y - previous[1]) < minimum_distance:
                continue
            return x, y
        return None

    def _build_patch_food(
        self,
        patch: FoodPatch,
        world: World,
        *,
        energy_budget: float | None = None,
    ) -> list[Food] | None:
        profile = self._profile(patch.biome_type)
        count = patch.target_food_count
        base_radii = self.rng.uniform(
            self.food_config.min_food_radius,
            self.food_config.max_food_radius,
            size=count,
        )
        energy_multipliers = self.rng.uniform(
            profile.energy_multiplier[0],
            profile.energy_multiplier[1],
            size=count,
        )
        radii = base_radii * np.sqrt(energy_multipliers)
        required_energy = float(
            np.sum(np.pi * radii**2 * self.food_config.energy_density)
        )
        if energy_budget is not None and required_energy > energy_budget + 1e-12:
            return None
        bite_capacities = self.rng.integers(
            profile.bite_capacity[0],
            profile.bite_capacity[1] + 1,
            size=count,
        )

        previous_center = patch.center_pos if patch.has_spawned else None
        for _ in range(self.config.center_sampling_attempts):
            positions_and_radius = self._sample_positions(
                patch.center_pos,
                patch.biome_type,
                patch.configured_radius,
                radii,
            )
            if positions_and_radius is not None:
                positions, effective_radius = positions_and_radius
                patch.effective_radius = effective_radius
                foods = []
                for index, (x, y) in enumerate(positions):
                    radius = float(radii[index])
                    foods.append(
                        Food(
                            id=world.food_spawner._claim_food_id(),
                            x=float(x),
                            y=float(y),
                            radius=radius,
                            energy_density=self.food_config.energy_density,
                            cluster_id=patch.id,
                            bite_capacity=int(bite_capacities[index]),
                        )
                    )
                return foods
            self._diagnostics["rejected_centers"] += 1
            center = self._sample_center(
                patch.biome_type,
                previous_center,
                patch.configured_radius,
            )
            if center is None:
                return None
            patch.center_pos = center
        return None

    def _sample_positions(
        self,
        center: tuple[float, float],
        biome: Biome,
        configured_radius: float,
        food_radii: np.ndarray,
    ) -> tuple[np.ndarray, float] | None:
        count = len(food_radii)
        if configured_radius <= 0.0:
            points = np.repeat(np.asarray(center)[None, :], count, axis=0)
            if self._valid_points(points, food_radii, biome).all():
                return points, 0.0
            return None

        radius = configured_radius
        minimum = configured_radius * self.config.minimum_radius_ratio
        while radius + 1e-12 >= minimum:
            accepted: list[np.ndarray] = []
            for _ in range(self.config.max_sampling_attempts):
                remaining = count - len(accepted)
                if remaining <= 0:
                    return np.asarray(accepted), radius
                batch_size = max(16, remaining * 4)
                offsets = self.rng.normal(0.0, max(radius / 3.0, 1e-9), (batch_size, 2))
                radial_mask = np.einsum("ij,ij->i", offsets, offsets) <= radius * radius
                offsets = offsets[radial_mask]
                if len(offsets) == 0:
                    continue
                points = offsets + np.asarray(center)
                candidate_radii = np.full(
                    len(points),
                    float(np.max(food_radii)),
                    dtype=np.float64,
                )
                valid = self._valid_points(points, candidate_radii, biome)
                for point in points[valid]:
                    accepted.append(point)
                    if len(accepted) == count:
                        return np.asarray(accepted), radius
                self._diagnostics["sampling_retries"] += 1
            radius *= self.config.radius_shrink_factor
            self._diagnostics["radius_reductions"] += 1
        return None

    def _valid_points(
        self,
        points: np.ndarray,
        radii: np.ndarray,
        biome: Biome,
    ) -> np.ndarray:
        left, bottom, right, top = self.biome_map.world_bounds
        valid = (
            (points[:, 0] >= left + radii)
            & (points[:, 0] <= right - radii)
            & (points[:, 1] >= bottom + radii)
            & (points[:, 1] <= top - radii)
        )
        cell_width = max(0.0001, right - left) / self.biome_map.grid_width
        cell_height = max(0.0001, top - bottom) / self.biome_map.grid_height
        columns = np.clip(
            ((points[:, 0] - left) / cell_width).astype(np.int64),
            0,
            self.biome_map.grid_width - 1,
        )
        rows = np.clip(
            ((points[:, 1] - bottom) / cell_height).astype(np.int64),
            0,
            self.biome_map.grid_height - 1,
        )
        return valid & (self.biome_map.biome_ids[rows, columns] == int(biome))

    def _synchronize_membership(self, world: World) -> None:
        live_ids = {food.id for food in world.foods}
        for patch in self.patches.values():
            patch.active_food_ids.intersection_update(live_ids)

    def _deplete_sparse_patches(self, world: World) -> None:
        for patch in self.patches.values():
            if patch.is_depleted or not patch.has_spawned:
                continue
            threshold = int(patch.target_food_count * self.config.depletion_ratio)
            if len(patch.active_food_ids) > threshold:
                continue
            residual_ids = set(patch.active_food_ids)
            for food in list(world.foods):
                if food.id in residual_ids:
                    world._remove_food(food)
            patch.active_food_ids.clear()
            patch.is_depleted = True
            patch.relocated = False
            low, high = self.config.cooldown_ticks
            patch.cooldown_timer = int(self.rng.integers(low, high + 1))

    def _advance_cooldowns(self, ticks: int) -> None:
        for patch in self.patches.values():
            if not patch.is_depleted or patch.relocated:
                continue
            patch.cooldown_timer = max(0, patch.cooldown_timer - ticks)
            if patch.cooldown_timer > 0:
                continue
            previous = patch.center_pos
            relocated = self._sample_center(
                patch.biome_type,
                previous,
                patch.effective_radius,
            )
            if relocated is None:
                patch.cooldown_timer = 1
                self._diagnostics["deferred_patches"] += 1
                continue
            patch.center_pos = relocated
            patch.relocated = True
            self._diagnostics["relocations"] += 1

    def _accrue_emergency_credit(
        self,
        world: World,
        cluster_target: int,
        active_cluster_count: int,
    ) -> None:
        capacity = max(1, world.food_spawner.food_capacity())
        if len(world.foods) / capacity > world.food_spawner.config.critical_food_ratio:
            self._low_food_burst_credit = 0.0
            self._emergency_burst_armed = False
            return
        if self._emergency_burst_armed:
            return
        shortage = max(0, cluster_target - active_cluster_count)
        desired = min(
            shortage,
            ceil(
                world.food_spawner.config.low_food_burst_items
                * world.live_food_config.cluster_spawn_share
            ),
        )
        if desired <= 0:
            return
        self._cluster_spawn_credit += desired
        self._low_food_burst_credit += desired
        self._emergency_burst_armed = True

    def _largest_patch_size(self) -> int:
        return max(
            (profile.pellets_per_cluster[1] for profile in (
                self.config.prairie,
                self.config.bushes,
                self.config.forest,
            )),
            default=1,
        )
