from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
import sys
import unittest

for optional_module in ("neat", "pymunk"):
    try:
        __import__(optional_module)
    except ModuleNotFoundError:
        sys.modules[optional_module] = ModuleType(optional_module)

from configs.sim_config import PopulationConfig
from src.creature import LineageInfo
from src.fitness import CreatureFitness, CreatureTelemetry
from src.rt_neat import RtNeatManager


@dataclass(slots=True)
class FakeCreature:
    creature_id: int
    energy: float = 1.0
    age_seconds: float = 30.0
    last_birth_time: float = -1_000_000.0
    lineage: LineageInfo = field(default_factory=LineageInfo)


class FakeBrainController:
    def __init__(self) -> None:
        """Create two deterministic fake neural genomes.

        Parameters
        ----------
        None
            This fixture receives no external parameters.

        Returns
        -------
        None
            The fake brain registry is initialized in place.
        """
        # Provide different enabled-connection counts for aggregate diagnostics.
        self.brains = {
            1: SimpleNamespace(
                genome=SimpleNamespace(
                    nodes={1: object(), 2: object(), 3: object()},
                    connections={
                        1: SimpleNamespace(enabled=True),
                        2: SimpleNamespace(enabled=False),
                    },
                )
            ),
            2: SimpleNamespace(
                genome=SimpleNamespace(
                    nodes={1: object(), 2: object(), 3: object(), 4: object()},
                    connections={
                        1: SimpleNamespace(enabled=True),
                        2: SimpleNamespace(enabled=True),
                        3: SimpleNamespace(enabled=True),
                    },
                )
            ),
        }

    def brain_for(self, creature_id: int) -> object | None:
        """Return the fake brain registered for one creature.

        Parameters
        ----------
        creature_id
            Stable fake creature identity.

        Returns
        -------
        object | None
            Matching fake brain when it exists.
        """
        # Mirror the production controller lookup contract.
        return self.brains.get(creature_id)


class RtNeatTelemetryTest(unittest.TestCase):
    def test_update_stats_reports_passive_energy_balances(self) -> None:
        """Verify aggregate statistics use only passive energy diagnostics.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Assertions verify the passive aggregate values.
        """
        # Give the two ledgers distinct ingestion and expenditure histories.
        manager = RtNeatManager(brain_controller=None)
        first = CreatureTelemetry(age_seconds=10.0)
        first.record_energy_transaction(ingested=4.0, spent=3.0)
        second = CreatureTelemetry(age_seconds=20.0)
        second.record_energy_transaction(ingested=9.0, spent=2.0)

        manager.update_stats(
            creatures=[FakeCreature(1), FakeCreature(2)],
            fitness_by_creature_id={1: first, 2: second},
            population_config=PopulationConfig(),
        )

        self.assertEqual(manager.eligible_parent_ids, [1, 2])
        self.assertEqual(manager.stats.eligible_parent_count, 2)
        self.assertAlmostEqual(manager.stats.best_net_energy_balance, 7.0)
        self.assertAlmostEqual(manager.stats.average_net_energy_balance, 4.0)
        self.assertAlmostEqual(manager.stats.worst_net_energy_balance, 1.0)
        self.assertAlmostEqual(manager.stats.average_net_metabolic_rate, 0.225)

    def test_species_size_does_not_modify_telemetry_or_eligibility(self) -> None:
        """Verify clade size does not share or rescale eligibility.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Assertions compare eligibility across unequal clade sizes.
        """
        # Construct one large clade and one singleton without ranking either.
        manager = RtNeatManager(brain_controller=None)
        creatures = [
            FakeCreature(creature_id, lineage=LineageInfo(species_id=1))
            for creature_id in range(1, 11)
        ]
        creatures.append(FakeCreature(11, lineage=LineageInfo(species_id=2)))
        telemetry = {
            creature.creature_id: CreatureTelemetry(age_seconds=30.0)
            for creature in creatures
        }

        manager.update_stats(creatures, telemetry, PopulationConfig())

        self.assertEqual(manager.eligible_parent_ids, list(range(1, 12)))
        self.assertEqual(manager.stats.eligible_parent_count, 11)

    def test_eligibility_uses_physiological_energy_age_and_cooldown(self) -> None:
        """Verify reproduction gates read live physiological state.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Assertions cover energy, maturity, and cooldown boundaries.
        """
        # Give telemetry a conflicting age so physiology is authoritative.
        manager = RtNeatManager(brain_controller=None)
        config = PopulationConfig()
        telemetry = CreatureTelemetry(age_seconds=999.0)
        eligible = FakeCreature(1, energy=0.75, age_seconds=10.0)
        too_young = FakeCreature(2, energy=1.0, age_seconds=9.99)
        cooling_down = FakeCreature(
            3,
            energy=1.0,
            age_seconds=10.0,
            last_birth_time=6.0,
        )

        self.assertTrue(
            manager.is_reproduction_eligible(eligible, telemetry, config)
        )
        self.assertFalse(
            manager.is_reproduction_eligible(too_young, telemetry, config)
        )
        self.assertFalse(
            manager.is_reproduction_eligible(cooling_down, telemetry, config)
        )

    def test_no_comparative_parent_selector_or_scalar_score_exists(self) -> None:
        """Verify comparative parent selection APIs have been removed.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Assertions reject obsolete selection interfaces.
        """
        # Check both the real-time manager and compatibility telemetry alias.
        manager = RtNeatManager(brain_controller=None)

        self.assertFalse(hasattr(manager, "select_parent"))
        self.assertFalse(hasattr(CreatureFitness(), "score"))

    def test_update_stats_tracks_trend_metrics(self) -> None:
        """Verify non-selection population trend diagnostics remain available.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Assertions cover event rates, lifespan, motion, and brain size.
        """
        # Populate independent counters before refreshing the live snapshot.
        manager = RtNeatManager(brain_controller=FakeBrainController())
        manager.record_normal_replacement()
        manager.record_extinction_replacements(2)
        manager.record_death(CreatureTelemetry(age_seconds=20.0))
        manager.record_death(CreatureTelemetry(age_seconds=40.0))

        manager.update_stats(
            creatures=[FakeCreature(1), FakeCreature(2)],
            fitness_by_creature_id={
                1: CreatureTelemetry(age_seconds=10.0, distance_traveled=50.0),
                2: CreatureTelemetry(age_seconds=20.0, distance_traveled=150.0),
            },
            population_config=PopulationConfig(),
            elapsed_time=120.0,
        )

        self.assertEqual(manager.stats.births, 3)
        self.assertEqual(manager.stats.normal_replacements, 1)
        self.assertEqual(manager.stats.extinction_replacements, 2)
        self.assertEqual(manager.stats.deaths, 2)
        self.assertAlmostEqual(manager.stats.average_lifespan_at_death, 30.0)
        self.assertAlmostEqual(manager.stats.average_speed, 6.25)
        self.assertAlmostEqual(manager.stats.average_distance_traveled, 100.0)
        self.assertAlmostEqual(manager.stats.average_brain_nodes, 3.5)
        self.assertAlmostEqual(manager.stats.average_brain_enabled_connections, 2.0)
        self.assertAlmostEqual(manager.stats.average_brain_connections, 2.5)
        self.assertAlmostEqual(manager.stats.births_per_minute, 1.5)
        self.assertAlmostEqual(manager.stats.deaths_per_minute, 1.0)


if __name__ == "__main__":
    unittest.main()
