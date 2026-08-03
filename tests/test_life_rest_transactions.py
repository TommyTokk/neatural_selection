from __future__ import annotations

from types import SimpleNamespace
import unittest

from configs.sim_config import ActionConfig, MetabolismConfig, TraitConfig, VisionConfig
from src.action import Action
from src.creature import LedgerDiagnostics
from src.persistence import PersistenceManager
from src.metabolism import (
    ActivityResult,
    DigestionResult,
    ENERGY_EPSILON,
    EnergyCostBreakdown,
    Metabolism,
    ResourceCandidate,
    calculate_digestion,
    calculate_weighted_activity,
    movement_life_penalty_multiplier,
)
from src.vision import VisionSystem
from src.world import (
    AcceptedNursingTransfer,
    NursingRequest,
    ReproductionRequest,
    TransactionResolution,
    World,
)


class DigestionLedgerTest(unittest.TestCase):
    def test_pure_digestion_reports_all_four_units(self) -> None:
        result = calculate_digestion(
            stomach_contents=0.5,
            digestion_rate=0.2,
            delta_time=1.0,
            trait_efficiency=0.8,
            rest_digestion_efficiency_bonus=0.0,
            effective_rest=0.0,
            processing_fraction=0.25,
            total_energy_demand=0.1,
            max_energy=1.0,
            starting_energy=0.9,
        )

        self.assertAlmostEqual(result.stomach_consumed, 0.2)
        self.assertAlmostEqual(result.gross_energy, 0.16)
        self.assertAlmostEqual(result.processing_cost, 0.04)
        self.assertAlmostEqual(result.net_energy, 0.12)

    def test_full_energy_without_demand_preserves_stomach(self) -> None:
        result = calculate_digestion(
            stomach_contents=1.0,
            digestion_rate=1.0,
            delta_time=1.0,
            trait_efficiency=1.0,
            rest_digestion_efficiency_bonus=0.1,
            effective_rest=1.0,
            processing_fraction=0.1,
            total_energy_demand=0.0,
            max_energy=1.0,
            starting_energy=1.0,
        )

        self.assertEqual(result.stomach_consumed, 0.0)

    def test_invalid_processing_fraction_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_digestion(
                stomach_contents=1.0,
                digestion_rate=1.0,
                delta_time=1.0,
                trait_efficiency=1.0,
                rest_digestion_efficiency_bonus=0.0,
                effective_rest=0.0,
                processing_fraction=1.01,
                total_energy_demand=0.0,
                max_energy=1.0,
                starting_energy=0.0,
            )


class LifeCandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        config = MetabolismConfig(
            basic_metabolism_rate=0.0,
            life_damage_per_energy_deficit=0.25,
        )
        vision = VisionSystem(VisionConfig(), max_life=config.max_life)
        self.metabolism = Metabolism(config, vision, TraitConfig())
        self.creature = SimpleNamespace(
            energy=0.0,
            life=1.0,
            pending_direct_life_damage=0.1,
            stomach_energy=0.0,
            stomach_difficulty_load=0.0,
            physical_traits=SimpleNamespace(
                digestion_rate=0.2,
                digestion_efficiency=0.9,
            ),
        )

    def test_zero_energy_alone_does_not_kill(self) -> None:
        self.creature.pending_direct_life_damage = 0.0

        candidate = self.metabolism.evaluate_candidate(
            self.creature,
            1.0,
            total_energy_demand=0.0,
        )

        self.assertTrue(candidate.survives)
        self.assertEqual(candidate.final_energy, 0.0)
        self.assertEqual(candidate.final_life, 1.0)

    def test_direct_and_deficit_damage_are_combined_once(self) -> None:
        candidate = self.metabolism.evaluate_candidate(
            self.creature,
            1.0,
            total_energy_demand=0.4,
        )

        self.assertAlmostEqual(candidate.unmet_energy_demand, 0.4)
        self.assertAlmostEqual(candidate.life_damage_from_deficit, 0.1)
        self.assertAlmostEqual(candidate.final_life, 0.8)

    def test_movement_penalty_curve_is_quadratic(self) -> None:
        self.assertEqual(movement_life_penalty_multiplier(1.0, 1.0, 4.0), 1.0)
        self.assertAlmostEqual(
            movement_life_penalty_multiplier(0.5, 1.0, 4.0),
            1.75,
        )
        self.assertEqual(movement_life_penalty_multiplier(0.0, 1.0, 4.0), 4.0)

    def test_funded_powered_movement_does_not_damage_life(self) -> None:
        self.creature.pending_direct_life_damage = 0.0
        self.creature.energy = 0.4

        candidate = self.metabolism.evaluate_candidate(
            self.creature,
            1.0,
            total_energy_demand=0.4,
            powered_movement_energy_demand=0.4,
        )

        self.assertEqual(candidate.unmet_powered_movement_demand, 0.0)
        self.assertEqual(candidate.movement_life_damage, 0.0)
        self.assertEqual(candidate.final_life, 1.0)

    def test_only_exact_unpaid_movement_receives_penalty(self) -> None:
        self.creature.pending_direct_life_damage = 0.0
        self.creature.energy = 0.2

        candidate = self.metabolism.evaluate_candidate(
            self.creature,
            1.0,
            total_energy_demand=0.5,
            powered_movement_energy_demand=0.3,
        )

        self.assertAlmostEqual(candidate.unmet_other_energy_demand, 0.0)
        self.assertAlmostEqual(candidate.unmet_powered_movement_demand, 0.3)
        self.assertAlmostEqual(candidate.movement_life_damage, 0.075)
        self.assertAlmostEqual(candidate.final_life, 0.925)

    def test_ordinary_damage_precedes_low_life_movement_multiplier(self) -> None:
        self.creature.pending_direct_life_damage = 0.0
        self.creature.life = 0.5

        candidate = self.metabolism.evaluate_candidate(
            self.creature,
            1.0,
            total_energy_demand=0.5,
            powered_movement_energy_demand=0.2,
        )

        expected_multiplier = 1.0 + 3.0 * (1.0 - 0.425) ** 2
        self.assertAlmostEqual(candidate.unmet_other_energy_demand, 0.3)
        self.assertAlmostEqual(candidate.unmet_powered_movement_demand, 0.2)
        self.assertAlmostEqual(
            candidate.movement_life_penalty_multiplier,
            expected_multiplier,
        )
        self.assertAlmostEqual(
            candidate.movement_life_damage,
            0.2 * 0.25 * expected_multiplier,
        )

    def test_passive_demand_remains_linear_at_low_life(self) -> None:
        self.creature.pending_direct_life_damage = 0.0
        self.creature.life = 0.2

        candidate = self.metabolism.evaluate_candidate(
            self.creature,
            1.0,
            total_energy_demand=0.4,
            powered_movement_energy_demand=0.0,
        )

        self.assertEqual(candidate.movement_life_damage, 0.0)
        self.assertAlmostEqual(candidate.life_damage_from_deficit, 0.1)
        self.assertAlmostEqual(candidate.final_life, 0.1)

    def test_low_life_powered_movement_can_kill(self) -> None:
        self.creature.pending_direct_life_damage = 0.0
        self.creature.life = 0.05

        candidate = self.metabolism.evaluate_candidate(
            self.creature,
            1.0,
            total_energy_demand=0.2,
            powered_movement_energy_demand=0.2,
        )

        self.assertFalse(candidate.survives)
        self.assertEqual(candidate.final_life, 0.0)


class EffectiveActionTest(unittest.TestCase):
    @staticmethod
    def action() -> Action:
        return Action(
            accelerate=1.0,
            rotate=-0.8,
            want_reproduce=1.0,
            want_eat=1.0,
            reset_chronometer=1.0,
            want_grab=1.0,
            want_release=1.0,
            want_nurse=1.0,
            flee_panic_intensity=1.0,
            herding=1.0,
            emit_sound=1.0,
            sound_tone=0.4,
            emit_trail_pheromone=1.0,
            emit_alarm_pheromone=1.0,
            rest=1.0,
        )

    def test_depletion_preserves_locomotion_and_gates_other_outputs(self) -> None:
        creature = SimpleNamespace(energy=ENERGY_EPSILON)
        raw = self.action()

        effective = World._effective_action_for(creature, raw)

        self.assertEqual(effective.accelerate, raw.accelerate)
        self.assertEqual(effective.rotate, raw.rotate)
        self.assertEqual(effective.flee_panic_intensity, raw.flee_panic_intensity)
        self.assertEqual(effective.herding, raw.herding)
        self.assertEqual(effective.want_reproduce, 0.0)
        self.assertEqual(effective.want_grab, 0.0)
        self.assertEqual(effective.want_nurse, 0.0)
        self.assertEqual(effective.emit_sound, 0.0)
        self.assertEqual(effective.emit_trail_pheromone, 0.0)
        self.assertEqual(effective.emit_alarm_pheromone, 0.0)
        self.assertEqual(effective.want_eat, raw.want_eat)
        self.assertEqual(effective.want_release, raw.want_release)
        self.assertEqual(effective.reset_chronometer, raw.reset_chronometer)
        self.assertEqual(effective.rest, raw.rest)
        self.assertEqual(raw.emit_sound, 1.0)

    def test_cached_raw_action_is_rederived_after_recovery(self) -> None:
        creature = SimpleNamespace(energy=0.0)
        raw = self.action()
        depleted = World._effective_action_for(creature, raw)

        creature.energy = ENERGY_EPSILON * 2.0
        recovered = World._effective_action_for(creature, raw)

        self.assertEqual(depleted.want_reproduce, 0.0)
        self.assertIs(recovered, raw)


class PoweredMovementDemandTest(unittest.TestCase):
    @staticmethod
    def make_world(effort: float, sprint: float) -> object:
        world = object.__new__(World)
        world.MAX_SPEED = 100.0
        creature = SimpleNamespace(
            creature_id=1,
            effective_voluntary_motor_effort=effort,
        )
        world.creatures = [creature]
        world._effective_actions = {
            1: Action(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      flee_panic_intensity=sprint)
        }
        world._apply_infant_movement_penalties = lambda: []
        world._restore_movement_multipliers = lambda _items: None
        world._creature_age_seconds = lambda _creature: 10.0
        world._senescence_factor_for = lambda _creature: 1.0
        def breakdown(*_args: object, **kwargs: float) -> EnergyCostBreakdown:
            return EnergyCostBreakdown(
                base=0.2,
                movement=0.3,
                vision=0.1,
                body=0.1,
                trait=0.0,
                sprint=0.4 * kwargs["sprint_intensity"],
            )

        world.metabolism = SimpleNamespace(
            energy_cost_breakdown_per_second=breakdown
        )
        return world

    def test_voluntary_movement_and_sprint_are_powered(self) -> None:
        world = self.make_world(effort=1.0, sprint=1.0)

        total, powered = world._energy_demands_for(2.0)

        self.assertAlmostEqual(total[1], 2.2)
        self.assertAlmostEqual(powered[1], 1.4)

    def test_passive_speed_is_not_powered_movement(self) -> None:
        world = self.make_world(effort=0.0, sprint=0.0)

        _total, powered = world._energy_demands_for(2.0)

        self.assertEqual(powered[1], 0.0)


class ActivityLedgerTest(unittest.TestCase):
    def test_weights_and_external_speed_contribution_are_exact(self) -> None:
        result = calculate_weighted_activity(
            voluntary_motor_effort=1.0,
            normalized_speed=1.0,
            turn_command=1.0,
            normalized_angular_speed=1.0,
            communication_cost=1.0,
            reproduction_selected=False,
            nursing_transfer=0.0,
        )

        self.assertAlmostEqual(result.total, 0.80)
        self.assertAlmostEqual(0.10 * result.normalized_speed, 0.10)

    def test_positive_pending_action_applies_activity_floor(self) -> None:
        result = calculate_weighted_activity(
            voluntary_motor_effort=0.0,
            normalized_speed=0.0,
            turn_command=0.0,
            normalized_angular_speed=0.0,
            communication_cost=0.0,
            reproduction_selected=False,
            nursing_transfer=0.01,
        )

        self.assertEqual(result.total, 1.0)


class TransactionPerformanceContractTest(unittest.TestCase):
    @staticmethod
    def candidate(
        *,
        energy: float = 1.0,
        life: float = 1.0,
        demand: float = 0.0,
    ) -> ResourceCandidate:
        return ResourceCandidate(
            digestion=DigestionResult(0.0, 0.0, 0.0, 0.0),
            total_energy_demand=demand,
            final_stomach_energy=0.0,
            final_stomach_difficulty_load=0.0,
            available_energy=energy,
            remaining_energy=energy,
            unmet_energy_demand=0.0,
            life_damage_from_deficit=0.0,
            direct_life_damage=0.0,
            final_energy=energy,
            final_life=life,
        )

    def make_world(
        self,
        reproduction_requests: list[ReproductionRequest],
    ) -> object:
        world = object.__new__(World)
        world.config = SimpleNamespace(
            population=SimpleNamespace(max_creatures=10),
            metabolism=SimpleNamespace(max_energy=1.0),
        )
        world.creatures = [
            SimpleNamespace(creature_id=index, smoothed_rest=0.0)
            for index in (1, 2, 3)
        ]
        world._prepare_reproduction_requests = lambda: reproduction_requests
        world._prepare_nursing_requests = lambda _delta: []
        world._energy_demands_for = lambda _delta: (
            {creature.creature_id: 0.0 for creature in world.creatures},
            {creature.creature_id: 0.0 for creature in world.creatures},
        )
        world._activity_for = lambda _creature, **kwargs: ActivityResult(
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 if kwargs else 0.0,
            0.0,
            1.0 if kwargs else 0.0,
        )

        def evaluate(
            creature: object,
            _delta: float,
            **kwargs: float,
        ) -> ResourceCandidate:
            demand = kwargs["total_energy_demand"]
            if creature.creature_id == 1 and demand > 0.0:
                return self.candidate(energy=0.0, life=0.0, demand=demand)
            return self.candidate(demand=demand)

        world.metabolism = SimpleNamespace(evaluate_candidate=evaluate)
        return world

    def test_empty_request_path_returns_baselines_without_resolution_pass(
        self,
    ) -> None:
        world = self.make_world([])
        world._resolve_transaction_pass = lambda *_args, **_kwargs: self.fail(
            "idle transaction path must not resolve or copy baselines"
        )

        resolution = world._resolve_resource_transactions(1.0 / 60.0)

        self.assertEqual(set(resolution.candidates), {1, 2, 3})
        self.assertEqual(set(resolution.activities), {1, 2, 3})
        self.assertEqual(resolution.reproductions, [])
        self.assertEqual(resolution.nursing_transfers, [])

    def test_reproduction_capacity_is_one_and_failed_parent_promotes_next(
        self,
    ) -> None:
        creatures = [
            SimpleNamespace(creature_id=index, smoothed_rest=0.0)
            for index in (1, 2, 3)
        ]
        requests = [
            ReproductionRequest(creature, rank, 0.4)
            for rank, creature in enumerate(creatures)
        ]
        world = self.make_world(requests)
        world.creatures = creatures

        resolution = world._resolve_resource_transactions(1.0 / 60.0)

        self.assertEqual(len(resolution.reproductions), 1)
        self.assertEqual(
            resolution.reproductions[0].parent.creature_id,
            2,
        )
        self.assertEqual(
            resolution.candidates[1].total_energy_demand,
            0.0,
        )
        self.assertAlmostEqual(
            resolution.candidates[2].total_energy_demand,
            0.4,
        )


class SelectedDiagnosticsTest(unittest.TestCase):
    @staticmethod
    def candidate() -> ResourceCandidate:
        return ResourceCandidate(
            digestion=DigestionResult(0.2, 0.18, 0.02, 0.16),
            total_energy_demand=0.3,
            final_stomach_energy=0.4,
            final_stomach_difficulty_load=0.5,
            available_energy=0.8,
            remaining_energy=0.5,
            unmet_energy_demand=0.0,
            life_damage_from_deficit=0.0,
            direct_life_damage=0.1,
            final_energy=0.5,
            final_life=0.9,
        )

    def test_core_state_commits_without_detailed_ledger_writes(self) -> None:
        creature = SimpleNamespace(
            energy=0.0,
            life=0.0,
            stomach_energy=0.0,
            stomach_difficulty_load=0.0,
            pending_direct_life_damage=0.1,
            ledger_diagnostics=LedgerDiagnostics(total_energy_demand=99.0),
        )

        Metabolism.commit_candidate(
            object.__new__(Metabolism),
            creature,
            self.candidate(),
            record_diagnostics=False,
        )

        self.assertEqual(creature.energy, 0.5)
        self.assertEqual(creature.life, 0.9)
        self.assertEqual(creature.stomach_energy, 0.4)
        self.assertEqual(creature.pending_direct_life_damage, 0.0)
        self.assertEqual(creature.ledger_diagnostics.total_energy_demand, 99.0)

    def test_activity_core_updates_every_tick_but_components_are_selective(
        self,
    ) -> None:
        creature = SimpleNamespace(
            smoothed_rest=0.5,
            activity=0.0,
            effective_rest=0.0,
            ledger_diagnostics=LedgerDiagnostics(),
        )
        creature.ledger_diagnostics.activity.weighted_total = 0.75
        activity = ActivityResult(0.2, 0.3, 0.4, 0.5, 0.0, 0.0, 0.6)

        World._commit_activity_diagnostics(
            creature,
            activity,
            record_diagnostics=False,
        )

        self.assertEqual(creature.activity, 0.6)
        self.assertAlmostEqual(creature.effective_rest, 0.2)
        self.assertEqual(
            creature.ledger_diagnostics.activity.weighted_total,
            0.75,
        )

        World._commit_activity_diagnostics(creature, activity)
        self.assertEqual(
            creature.ledger_diagnostics.activity.weighted_total,
            0.6,
        )

    def test_world_commit_marks_actions_and_records_only_selected_details(
        self,
    ) -> None:
        creatures = [
            SimpleNamespace(
                creature_id=creature_id,
                energy=0.5,
                life=1.0,
                stomach_energy=0.0,
                stomach_difficulty_load=0.0,
                pending_direct_life_damage=0.0,
                smoothed_rest=0.5,
                activity=0.0,
                effective_rest=0.0,
                ledger_diagnostics=LedgerDiagnostics(
                    transaction_status="unchanged"
                ),
            )
            for creature_id in (1, 2, 3)
        ]
        reproduction = ReproductionRequest(creatures[0], 0, 0.2)
        nursing = AcceptedNursingTransfer(
            NursingRequest(creatures[1], creatures[2], 0.1),
            0.1,
        )
        candidate = self.candidate()
        activity = ActivityResult(0.2, 0.3, 0.4, 0.5, 0.0, 0.0, 0.6)
        resolution = TransactionResolution(
            candidates={creature.creature_id: candidate for creature in creatures},
            activities={creature.creature_id: activity for creature in creatures},
            reproductions=[reproduction],
            nursing_transfers=[nursing],
        )
        world = object.__new__(World)
        world.creatures = creatures
        world.selected_creature_id = 2
        world.config = SimpleNamespace(
            metabolism=SimpleNamespace(max_energy=1.0)
        )
        world._resolve_resource_transactions = lambda _delta: resolution
        world._stage_final_reproductions = lambda _requests: ([], None, None)
        world._last_digestion_processing_costs_per_second = {}
        world.foods = []
        world.fitness = {}
        world._recover_extinct_population = lambda: None
        world._reset_behavior_focus = lambda _creature_id: None
        committed: dict[int, tuple[str, bool]] = {}
        metabolism = object.__new__(Metabolism)

        def commit(
            creature: object,
            resource: ResourceCandidate,
            *,
            transaction_status: str,
            record_diagnostics: bool,
        ) -> None:
            committed[creature.creature_id] = (
                transaction_status,
                record_diagnostics,
            )
            Metabolism.commit_candidate(
                metabolism,
                creature,
                resource,
                transaction_status=transaction_status,
                record_diagnostics=record_diagnostics,
            )

        world.metabolism = SimpleNamespace(
            evaluate_candidate=lambda *_args, **_kwargs: candidate,
            commit_candidate=commit,
        )

        world._update_metabolism(1.0)

        self.assertEqual(
            committed,
            {
                1: ("action_committed", False),
                2: ("action_committed", True),
                3: ("baseline_committed", False),
            },
        )
        self.assertEqual(
            creatures[0].ledger_diagnostics.transaction_status,
            "unchanged",
        )
        self.assertEqual(
            creatures[1].ledger_diagnostics.transaction_status,
            "action_committed",
        )
        self.assertEqual(
            creatures[2].ledger_diagnostics.transaction_status,
            "unchanged",
        )


class CompatibilityAndValidationTest(unittest.TestCase):
    def test_legacy_contract_is_the_only_append_migration(self) -> None:
        self.assertTrue(
            PersistenceManager._is_life_rest_append_migration(5, 1, 43, 14)
        )
        self.assertFalse(
            PersistenceManager._is_life_rest_append_migration(4, 1, 43, 14)
        )

    def test_rest_output_migration_is_disconnected_zero_bias_sigmoid(self) -> None:
        class Genome:
            def __init__(self) -> None:
                self.nodes = {}
                self.connections = {("old", "edge"): object()}

            def create_node(self, _config: object, key: int) -> object:
                return SimpleNamespace(key=key, bias=1.0, activation="tanh")

        genome = Genome()
        PersistenceManager._append_rest_output_node({1: genome}, object())

        self.assertEqual(genome.nodes[14].bias, 0.0)
        self.assertEqual(genome.nodes[14].activation, "sigmoid")
        self.assertEqual(list(genome.connections), [("old", "edge")])

    def test_runtime_mutation_of_rest_config_is_revalidated(self) -> None:
        config = ActionConfig()
        config.rest_response_rate = float("nan")

        with self.assertRaises(ValueError):
            config.validate()

    def test_movement_life_penalty_multiplier_is_validated(self) -> None:
        config = MetabolismConfig(movement_life_penalty_max_multiplier=0.99)
        vision = VisionSystem(VisionConfig(), max_life=config.max_life)

        with self.assertRaises(ValueError):
            Metabolism(config, vision, TraitConfig())


class NursingAllocationTest(unittest.TestCase):
    @staticmethod
    def candidate(energy: float, life: float = 1.0) -> ResourceCandidate:
        return ResourceCandidate(
            digestion=DigestionResult(0.0, 0.0, 0.0, 0.0),
            total_energy_demand=0.0,
            final_stomach_energy=0.0,
            final_stomach_difficulty_load=0.0,
            available_energy=energy,
            remaining_energy=energy,
            unmet_energy_demand=0.0,
            life_damage_from_deficit=0.0,
            direct_life_damage=0.0,
            final_energy=energy,
            final_life=life,
        )

    def make_world(self, *, first_donor_dies: bool = False) -> object:
        world = object.__new__(World)
        world.config = SimpleNamespace(
            metabolism=SimpleNamespace(max_energy=1.0)
        )
        world.creatures = []
        world._activity_for = lambda *_args, **_kwargs: ActivityResult(
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0
        )

        def evaluate(creature: object, _dt: float, **kwargs: float) -> ResourceCandidate:
            demand = kwargs["total_energy_demand"]
            if first_donor_dies and creature.creature_id == 1 and demand > 0.0:
                return self.candidate(0.0, 0.0)
            return self.candidate(max(0.0, 1.0 - demand))

        world.metabolism = SimpleNamespace(evaluate_candidate=evaluate)
        return world

    @staticmethod
    def creature(creature_id: int, generation: int) -> object:
        return SimpleNamespace(
            creature_id=creature_id,
            lineage=SimpleNamespace(generation=generation),
            smoothed_rest=0.0,
        )

    def test_multiple_donors_use_id_order_and_never_exceed_headroom(self) -> None:
        world = self.make_world()
        first = self.creature(1, 1)
        second = self.creature(2, 1)
        target = self.creature(10, 2)
        world.creatures = [second, target, first]
        baseline = {
            1: self.candidate(1.0),
            2: self.candidate(1.0),
            10: self.candidate(0.8),
        }
        activity = {
            creature_id: ActivityResult(0, 0, 0, 0, 0, 0, 0)
            for creature_id in baseline
        }

        resolution, _failed = world._resolve_transaction_pass(
            1.0,
            [],
            [
                NursingRequest(second, target, 0.15),
                NursingRequest(first, target, 0.15),
            ],
            {1: 0.0, 2: 0.0, 10: 0.0},
            baseline,
            activity,
        )

        self.assertEqual(
            [item.request.donor.creature_id for item in resolution.nursing_transfers],
            [1, 2],
        )
        self.assertAlmostEqual(
            sum(item.allocated_transfer for item in resolution.nursing_transfers),
            0.2,
        )

    def test_dead_donor_releases_all_headroom_to_next_donor(self) -> None:
        world = self.make_world(first_donor_dies=True)
        first = self.creature(1, 1)
        second = self.creature(2, 1)
        target = self.creature(10, 2)
        world.creatures = [first, second, target]
        baseline = {
            1: self.candidate(1.0),
            2: self.candidate(1.0),
            10: self.candidate(0.8),
        }
        activity = {
            creature_id: ActivityResult(0, 0, 0, 0, 0, 0, 0)
            for creature_id in baseline
        }

        resolution, _failed = world._resolve_transaction_pass(
            1.0,
            [],
            [
                NursingRequest(first, target, 0.2),
                NursingRequest(second, target, 0.2),
            ],
            {1: 0.0, 2: 0.0, 10: 0.0},
            baseline,
            activity,
        )

        self.assertEqual(len(resolution.nursing_transfers), 1)
        self.assertEqual(
            resolution.nursing_transfers[0].request.donor.creature_id,
            2,
        )
        self.assertAlmostEqual(
            resolution.nursing_transfers[0].allocated_transfer,
            0.2,
        )


if __name__ == "__main__":
    unittest.main()
