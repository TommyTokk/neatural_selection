from __future__ import annotations

from math import ceil
from random import Random
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, PropertyMock, patch

from configs.sim_config import SimConfig
from src.creature.metabolism import calculate_reproduction_energy_transfer
from src.creature.neat.brain import NeatBrain
from src.creature.neat.controller import NeatBrainController
from src.world import ReproductionRequest
from src.world import World


class ThermodynamicReproductionTest(unittest.TestCase):
    def test_energy_transfer_conserves_parent_investment(self) -> None:
        transfer = calculate_reproduction_energy_transfer(0.8, 0.45, 0.9)

        self.assertAlmostEqual(transfer.parent_investment, 0.36)
        self.assertAlmostEqual(transfer.child_endowment, 0.324)
        self.assertAlmostEqual(
            0.8,
            0.8 - transfer.parent_investment
            + transfer.child_endowment / 0.9,
        )

    def test_infant_runway_exceeds_worst_case_idle_burn(self) -> None:
        config = SimConfig()
        maximum_initial_connections = ceil(43 * 15 * 0.15)
        minimum_endowment = (
            config.metabolism.max_energy
            * config.population.reproduction_energy_fraction
            * config.population.child_energy_investment_fraction
            * config.population.birth_conversion_efficiency
        )
        maximum_idle_rate = (
            config.metabolism.basic_metabolism_rate
            + config.trait.body_metabolism_cost_factor
            + config.vision.base_energy_cost
            + config.vision.area_energy_cost_factor
            + maximum_initial_connections
            * config.metabolism.brain_upkeep_per_enabled_connection
        )

        self.assertGreater(
            minimum_endowment,
            maximum_idle_rate
            * config.population.maturity_age_seconds
            * 1.2,
        )

    def test_startup_rejects_an_infant_runway_deficit(self) -> None:
        world = object.__new__(World)
        world.config = SimConfig()
        world.config.population.maturity_age_seconds = 100.0
        world.neat_controller = SimpleNamespace(
            config=SimpleNamespace(
                genome_config=SimpleNamespace(
                    num_inputs=46,
                    num_outputs=16,
                    connection_fraction=0.15,
                )
            )
        )

        with self.assertRaisesRegex(ValueError, "Infant endowment"):
            world._validate_infant_runway()

    def test_reproduction_intent_domains_are_equivalent_and_quiescent(self) -> None:
        raw_threshold = 0.6
        centered_threshold = 0.2
        offset_threshold = 0.1

        self.assertAlmostEqual(2.0 * (raw_threshold - 0.5), centered_threshold)
        self.assertAlmostEqual(raw_threshold - 0.5, offset_threshold)
        founder_raw = 1.0 / (1.0 + 2.718281828459045)
        founder_centered = NeatBrain._center_output_value(
            founder_raw,
            "sigmoid",
        )
        self.assertLess(founder_centered, centered_threshold)

    def test_founder_reproduction_output_is_sigmoid_with_negative_bias(self) -> None:
        controller = object.__new__(NeatBrainController)
        controller.config = SimpleNamespace(
            genome_config=SimpleNamespace(output_keys=(-1, -2, -3))
        )
        node = SimpleNamespace(activation="tanh", bias=0.0)
        genome = SimpleNamespace(nodes={-3: node})

        controller._enforce_reproduction_output_contract(genome, founder=True)

        self.assertEqual(node.activation, "sigmoid")
        self.assertEqual(node.bias, -1.0)

    def test_blocked_child_placement_does_not_mutate_parent(self) -> None:
        world = object.__new__(World)
        world.config = SimConfig()
        world.rng = Random(7)
        parent = SimpleNamespace(
            position=(5.0, 5.0),
            radius=16.0,
            heading=0.0,
            energy=0.8,
            last_birth_time=0.0,
            lifetime_offspring_count=0,
        )
        world.creatures = [parent]
        world.foods = []
        state_before = (
            parent.energy,
            parent.last_birth_time,
            parent.lifetime_offspring_count,
            world.rng.getstate(),
        )

        with patch.object(
            World,
            "environment_world_bounds",
            new_callable=PropertyMock,
            return_value=(0.0, 0.0, 10.0, 10.0),
        ):
            position = world._child_spawn_position(parent, 16.0)

        self.assertIsNone(position)
        self.assertEqual(
            state_before[:3],
            (
                parent.energy,
                parent.last_birth_time,
                parent.lifetime_offspring_count,
            ),
        )

    def test_failed_staging_rolls_back_rng_ids_and_neural_state(self) -> None:
        world = object.__new__(World)
        world.config = SimConfig()
        world.rng = Random(7)
        world._next_creature_id_value = 10
        world.neat_controller = SimpleNamespace(marker=["unchanged"])
        world._child_spawn_position = MagicMock(return_value=None)
        parent = SimpleNamespace(creature_id=1)
        request = ReproductionRequest(parent, 0.36, 0.324)
        rng_state = world.rng.getstate()

        staged, _shadow, _shadow_rng_state = world._stage_final_reproductions(
            [request]
        )

        self.assertEqual(staged, [])
        self.assertEqual(world.rng.getstate(), rng_state)
        self.assertEqual(world._next_creature_id_value, 10)
        self.assertEqual(world.neat_controller.marker, ["unchanged"])

    def test_full_capacity_does_not_consume_rng_or_parent_state(self) -> None:
        world = object.__new__(World)
        world.config = SimConfig()
        world.config.population.max_creatures = 1
        world.rng = Random(7)
        parent = SimpleNamespace(
            energy=0.8,
            last_birth_time=0.0,
            lifetime_offspring_count=0,
        )
        world.creatures = [parent]
        rng_state = world.rng.getstate()

        requests = world._prepare_reproduction_requests({})

        self.assertEqual(requests, [])
        self.assertEqual(world.rng.getstate(), rng_state)
        self.assertEqual(parent.energy, 0.8)
        self.assertEqual(parent.last_birth_time, 0.0)
        self.assertEqual(parent.lifetime_offspring_count, 0)

    def test_empty_species_archive_uses_procedural_founders(self) -> None:
        world = object.__new__(World)
        world.config = SimConfig()
        world.creatures = []
        world._trait_archive_by_genome_id = {}
        world.neat_controller = SimpleNamespace(
            population=SimpleNamespace(population={})
        )
        world._recover_with_procedural_founders = MagicMock()

        world._recover_extinct_population()

        world._recover_with_procedural_founders.assert_called_once_with(
            min(
                world.config.population.extinction_recovery_creatures,
                world.config.population.max_creatures,
            )
        )

    def test_extinction_parent_pool_samples_species_before_genomes(self) -> None:
        world = object.__new__(World)
        world.rng = Random(7)
        archives = {
            1: [SimpleNamespace(key=11), SimpleNamespace(key=12)],
            2: [SimpleNamespace(key=21)],
            3: [SimpleNamespace(key=31), SimpleNamespace(key=32)],
        }

        parents = world._unranked_recovery_parent_genomes(archives, 3)

        species_by_key = {11: 1, 12: 1, 21: 2, 31: 3, 32: 3}
        self.assertEqual(
            {species_by_key[parent.key] for parent in parents},
            {1, 2, 3},
        )


if __name__ == "__main__":
    unittest.main()
