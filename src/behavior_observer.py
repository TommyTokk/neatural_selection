"""Spawn-safe automatic and focal temporal behaviour observation.

The classes in this module deliberately depend only on primitive, picklable
data.  Behaviour labels are derived from realized world/action history, never
from NEAT outputs or named neural intents.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
import logging
from math import cos, hypot, sin
import multiprocessing
from queue import Empty, Full
from time import monotonic
from typing import Any

from configs.sim_config import (
    BehaviorHistoryConfig,
    BehaviorObserverConfig,
    CounterfactualWhyConfig,
)
from src.behavior_history import (
    BehaviorEvidenceAccumulator,
    BehaviorOutcome,
    BehaviorTermination,
    CompletedBehaviorBoutDraft,
)


LOGGER = logging.getLogger(__name__)


class BehaviorKind(str, Enum):
    FOOD_ORIENTATION = "food_orientation"
    FOOD_APPROACH = "food_approach"
    FEEDING = "feeding"
    RESTING = "resting"
    COHESION = "cohesion"
    ALARM_RETREAT = "alarm_retreat"


class BoutStatus(str, Enum):
    EMERGING = "emerging"
    ACTIVE = "active"


class ObservationMode(str, Enum):
    AUTOMATIC = "automatic"
    FOCAL = "focal"


@dataclass(frozen=True, slots=True)
class BehaviorObservation:
    creature_id: int
    selection_generation: int
    simulation_time: float
    x: float
    y: float
    heading: float
    angular_velocity: float
    velocity_x: float
    velocity_y: float
    speed: float
    nearest_food_id: int | None = None
    food_visible: bool = False
    food_distance: float | None = None
    food_relative_angle: float | None = None
    compatible_group_visible: bool = False
    compatible_group_count: float = 0.0
    compatible_group_distance: float | None = None
    compatible_group_direction: float | None = None
    group_velocity_x: float = 0.0
    group_velocity_y: float = 0.0
    personal_space_occupied: bool = False
    alarm_here: float = 0.0
    alarm_forward_left: float = 0.0
    alarm_forward_right: float = 0.0
    carrying_food: bool = False
    food_consumption_count: int = 0
    food_consumed_energy_total: float = 0.0


@dataclass(frozen=True, slots=True)
class BehaviorObservationBatch:
    observations: tuple[BehaviorObservation, ...]


@dataclass(frozen=True, slots=True)
class BehaviorEvidence:
    key: str
    label: str
    value: float
    unit: str | None
    passed: bool


@dataclass(frozen=True, slots=True)
class BehaviorStateSnapshot:
    behavior: BehaviorKind
    status: BoutStatus
    evidence_score: float
    duration_seconds: float
    evidence: tuple[BehaviorEvidence, ...]
    bout_id: int = 0
    target_id: int | None = None


@dataclass(frozen=True, slots=True)
class BehaviorSnapshot:
    creature_id: int
    selection_generation: int
    simulation_time: float
    behaviors: tuple[BehaviorStateSnapshot, ...]
    observations_processed: int
    produced_monotonic: float
    result_drops: int = 0


@dataclass(frozen=True, slots=True)
class BehaviorSnapshotBatch:
    snapshots: tuple[BehaviorSnapshot, ...]
    observations_processed: int
    produced_monotonic: float
    result_drops: int = 0


@dataclass(frozen=True, slots=True)
class BehaviorWorkerError:
    creature_id: int | None
    selection_generation: int | None
    message: str


@dataclass(frozen=True, slots=True)
class FinalizeBehaviorFocus:
    creature_id: int
    selection_generation: int
    termination: BehaviorTermination


@dataclass(frozen=True, slots=True)
class BehaviorHistoryWorkerStatus:
    completion_outbox_depth: int
    completion_outbox_high_water: int
    completion_outbox_warning: bool
    completion_recording_suspended: bool
    history_completions_not_recorded: int


@dataclass(frozen=True, slots=True)
class BehaviorObserverDiagnostics:
    samples_produced: int = 0
    samples_dropped: int = 0
    observations_processed: int = 0
    results_dropped: int = 0
    result_latency_ms: float | None = None
    input_queue_size: int | None = None
    worker_health: str = "idle"
    last_error: str | None = None
    completion_outbox_depth: int = 0
    completion_outbox_high_water: int = 0
    completion_outbox_warning: bool = False
    completion_recording_suspended: bool = False
    history_completions_not_recorded: int = 0


@dataclass(frozen=True, slots=True)
class _RuleEvidence:
    present: bool
    score: float
    evidence: tuple[BehaviorEvidence, ...]
    target_id: int | None = None


@dataclass(slots=True)
class _BoutState:
    status: BoutStatus | None = None
    candidate_since: float | None = None
    active_since: float | None = None
    last_evidence_time: float | None = None
    bout_id: int = 0
    target_id: int | None = None
    evidence_accumulator: BehaviorEvidenceAccumulator | None = None
    start_consumption_count: int = 0
    start_alarm_level: float = 0.0

    def reset(self) -> None:
        self.status = None
        self.candidate_since = None
        self.active_since = None
        self.last_evidence_time = None
        self.target_id = None
        self.evidence_accumulator = None
        self.start_consumption_count = 0
        self.start_alarm_level = 0.0


class TemporalBehaviorAnalyzer:
    """Analyze one focal creature and reset completely when focus changes."""

    def __init__(
        self,
        config: BehaviorObserverConfig,
        history_config: BehaviorHistoryConfig | None = None,
    ) -> None:
        self.config = config
        self.history_config = history_config or BehaviorHistoryConfig()
        self.history: deque[BehaviorObservation] = deque()
        self.creature_id: int | None = None
        self.selection_generation: int | None = None
        self.bouts = {
            behavior: _BoutState()
            for behavior in BehaviorKind
        }
        self.observations_processed = 0
        self._last_consumption_count = 0
        self._last_consumed_energy = 0.0
        self._last_feeding_rule = _empty_rule()
        self._speed_sum = 0.0
        self._low_speed_count = 0
        self._last_observation: BehaviorObservation | None = None
        self._completed_bouts: deque[CompletedBehaviorBoutDraft] = deque()

    def reset(
        self,
        creature_id: int | None = None,
        selection_generation: int | None = None,
    ) -> None:
        self.history.clear()
        self.creature_id = creature_id
        self.selection_generation = selection_generation
        self._last_consumption_count = 0
        self._last_consumed_energy = 0.0
        self._last_feeding_rule = _empty_rule()
        self._speed_sum = 0.0
        self._low_speed_count = 0
        self._last_observation = None
        for bout in self.bouts.values():
            bout.reset()
            bout.bout_id = 0

    def process(self, observation: BehaviorObservation) -> BehaviorSnapshot:
        identity = (
            observation.creature_id,
            observation.selection_generation,
        )
        if identity != (self.creature_id, self.selection_generation):
            self.force_finalize(BehaviorTermination.FOCUS_CHANGED)
            self.reset(*identity)

        self._last_observation = observation
        self.history.append(observation)
        self._speed_sum += observation.speed
        if observation.speed <= self.config.rest_speed_threshold:
            self._low_speed_count += 1
        cutoff = observation.simulation_time - self.config.window_seconds
        while self.history and self.history[0].simulation_time < cutoff:
            expired = self.history.popleft()
            self._speed_sum -= expired.speed
            if expired.speed <= self.config.rest_speed_threshold:
                self._low_speed_count -= 1
        self.observations_processed += 1

        rules = {
            BehaviorKind.FOOD_ORIENTATION: self._food_orientation(),
            BehaviorKind.FOOD_APPROACH: self._food_approach(),
            BehaviorKind.RESTING: self._resting(),
            BehaviorKind.COHESION: self._cohesion(),
            BehaviorKind.ALARM_RETREAT: self._alarm_retreat(),
        }
        feeding = self._feeding(observation)

        states: list[BehaviorStateSnapshot] = []
        for behavior in (
            BehaviorKind.FOOD_ORIENTATION,
            BehaviorKind.FOOD_APPROACH,
            BehaviorKind.RESTING,
            BehaviorKind.COHESION,
            BehaviorKind.ALARM_RETREAT,
        ):
            state = self._update_bout(
                behavior,
                rules[behavior],
                observation.simulation_time,
                observation,
            )
            if state is not None:
                states.append(state)
        feeding_state = self._update_feeding(
            feeding,
            observation.simulation_time,
            observation,
        )
        if feeding_state is not None:
            # Keep the enum's stable display order rather than event timing.
            states.append(feeding_state)
        states.sort(key=lambda item: list(BehaviorKind).index(item.behavior))

        return BehaviorSnapshot(
            creature_id=observation.creature_id,
            selection_generation=observation.selection_generation,
            simulation_time=observation.simulation_time,
            behaviors=tuple(states),
            observations_processed=self.observations_processed,
            produced_monotonic=monotonic(),
        )

    def drain_completed_bouts(self) -> tuple[CompletedBehaviorBoutDraft, ...]:
        completed = tuple(self._completed_bouts)
        self._completed_bouts.clear()
        return completed

    def force_finalize(self, termination: BehaviorTermination) -> None:
        observation = self._last_observation
        if observation is None:
            return
        for behavior, bout in self.bouts.items():
            if bout.status is BoutStatus.ACTIVE:
                self._finalize_bout(
                    behavior,
                    bout,
                    observation.simulation_time,
                    observation,
                    termination,
                )
            bout.reset()

    def _update_bout(
        self,
        behavior: BehaviorKind,
        rule: _RuleEvidence,
        now: float,
        observation: BehaviorObservation,
    ) -> BehaviorStateSnapshot | None:
        bout = self.bouts[behavior]
        if rule.present:
            bout.last_evidence_time = now
            if bout.status is None:
                bout.bout_id += 1
                bout.status = BoutStatus.EMERGING
                bout.candidate_since = now
                bout.target_id = rule.target_id
                bout.evidence_accumulator = BehaviorEvidenceAccumulator(
                    self.history_config.active_metric_sample_capacity
                )
                bout.start_consumption_count = (
                    observation.food_consumption_count
                )
                bout.start_alarm_level = observation.alarm_here
            if bout.evidence_accumulator is not None:
                bout.evidence_accumulator.add(rule.evidence)
            if rule.target_id is not None:
                bout.target_id = rule.target_id
            if (
                bout.status is BoutStatus.EMERGING
                and bout.candidate_since is not None
                and now - bout.candidate_since
                >= self.config.bout_start_seconds
            ):
                bout.status = BoutStatus.ACTIVE
                bout.active_since = bout.candidate_since
        elif bout.status is BoutStatus.EMERGING:
            bout.reset()
        elif bout.status is BoutStatus.ACTIVE:
            if bout.evidence_accumulator is not None:
                bout.evidence_accumulator.add(rule.evidence)
            last = bout.last_evidence_time
            if (
                last is None
                or now - last > self.config.bout_end_grace_seconds
            ):
                self._finalize_bout(
                    behavior,
                    bout,
                    now,
                    observation,
                    BehaviorTermination.NATURAL,
                )
                bout.reset()

        if bout.status is None:
            return None
        started = (
            bout.candidate_since
            if bout.status is BoutStatus.EMERGING
            else bout.active_since
        )
        return BehaviorStateSnapshot(
            behavior=behavior,
            status=bout.status,
            evidence_score=_clamp01(rule.score),
            duration_seconds=max(
                0.0,
                now - (now if started is None else started),
            ),
            evidence=rule.evidence,
            bout_id=bout.bout_id,
            target_id=rule.target_id,
        )

    def _update_feeding(
        self,
        rule: _RuleEvidence,
        now: float,
        observation: BehaviorObservation,
    ) -> BehaviorStateSnapshot | None:
        bout = self.bouts[BehaviorKind.FEEDING]
        if rule.present:
            self._last_feeding_rule = rule
            if bout.status is None:
                bout.bout_id += 1
                bout.active_since = now
                bout.candidate_since = now
                bout.evidence_accumulator = BehaviorEvidenceAccumulator(
                    self.history_config.active_metric_sample_capacity
                )
                bout.start_consumption_count = (
                    observation.food_consumption_count
                )
                bout.start_alarm_level = observation.alarm_here
            if bout.evidence_accumulator is not None:
                bout.evidence_accumulator.add(rule.evidence)
            bout.status = BoutStatus.ACTIVE
            bout.last_evidence_time = now
        elif (
            bout.status is BoutStatus.ACTIVE
            and (
                bout.last_evidence_time is None
                or now - bout.last_evidence_time
                > self.config.feeding_display_seconds
            )
        ):
            if bout.evidence_accumulator is not None:
                bout.evidence_accumulator.add(rule.evidence)
            self._finalize_bout(
                BehaviorKind.FEEDING,
                bout,
                now,
                observation,
                BehaviorTermination.NATURAL,
            )
            bout.reset()
        if bout.status is None:
            return None
        displayed_rule = rule if rule.present else self._last_feeding_rule
        return BehaviorStateSnapshot(
            behavior=BehaviorKind.FEEDING,
            status=BoutStatus.ACTIVE,
            evidence_score=displayed_rule.score,
            duration_seconds=max(
                0.0,
                now
                - (
                    now
                    if bout.active_since is None
                    else bout.active_since
                ),
            ),
            evidence=displayed_rule.evidence,
            bout_id=bout.bout_id,
        )

    def _finalize_bout(
        self,
        behavior: BehaviorKind,
        bout: _BoutState,
        end_time: float,
        observation: BehaviorObservation,
        termination: BehaviorTermination,
    ) -> None:
        if (
            self.creature_id is None
            or self.selection_generation is None
            or bout.active_since is None
        ):
            return
        outcome = self._outcome_for(
            behavior,
            bout,
            observation,
            termination,
            end_time,
        )
        summaries = (
            ()
            if bout.evidence_accumulator is None
            else bout.evidence_accumulator.summaries()
        )
        self._completed_bouts.append(
            CompletedBehaviorBoutDraft(
                creature_id=self.creature_id,
                selection_generation=self.selection_generation,
                behavior=behavior,
                local_bout_id=bout.bout_id,
                start_time=bout.active_since,
                end_time=end_time,
                duration=max(0.0, end_time - bout.active_since),
                evidence_summary=summaries,
                outcome=outcome,
                termination=termination,
            )
        )

    def _outcome_for(
        self,
        behavior: BehaviorKind,
        bout: _BoutState,
        observation: BehaviorObservation,
        termination: BehaviorTermination,
        end_time: float,
    ) -> BehaviorOutcome | None:
        if behavior is BehaviorKind.FEEDING:
            return BehaviorOutcome.CONSUMPTION_EVENT
        if termination is not BehaviorTermination.NATURAL:
            if behavior in {
                BehaviorKind.FOOD_ORIENTATION,
                BehaviorKind.FOOD_APPROACH,
                BehaviorKind.ALARM_RETREAT,
            }:
                return BehaviorOutcome.INTERRUPTED
            return None
        if behavior is BehaviorKind.FOOD_APPROACH:
            if (
                observation.food_consumption_count
                > bout.start_consumption_count
            ):
                return BehaviorOutcome.FOOD_CONSUMED
            if (
                not observation.food_visible
                or observation.nearest_food_id is None
                or (
                    bout.target_id is not None
                    and observation.nearest_food_id != bout.target_id
                )
            ):
                return BehaviorOutcome.TARGET_LOST
            return BehaviorOutcome.ABANDONED
        if behavior is BehaviorKind.FOOD_ORIENTATION:
            approach = self.bouts[BehaviorKind.FOOD_APPROACH]
            if (
                approach.status is BoutStatus.ACTIVE
                and (
                    bout.target_id is None
                    or approach.target_id == bout.target_id
                )
            ):
                return BehaviorOutcome.APPROACH_STARTED
            if (
                not observation.food_visible
                or observation.nearest_food_id is None
                or (
                    bout.target_id is not None
                    and observation.nearest_food_id != bout.target_id
                )
            ):
                return BehaviorOutcome.TARGET_LOST
            return BehaviorOutcome.ENDED_WITHOUT_APPROACH
        if behavior is BehaviorKind.ALARM_RETREAT:
            start_time = (
                end_time
                if bout.active_since is None
                else bout.active_since
            )
            duration = max(0.0, end_time - start_time)
            required_drop = self.config.alarm_min_temporal_drop * duration
            if observation.alarm_here <= bout.start_alarm_level - required_drop:
                return BehaviorOutcome.ALARM_EXPOSURE_REDUCED
        return None

    def _food_segment(self) -> list[BehaviorObservation]:
        if not self.history:
            return []
        current = self.history[-1]
        target_id = current.nearest_food_id
        if (
            target_id is None
            or not current.food_visible
            or current.food_distance is None
            or current.food_relative_angle is None
        ):
            return []
        segment: list[BehaviorObservation] = []
        for sample in reversed(self.history):
            if (
                not sample.food_visible
                or sample.nearest_food_id != target_id
                or sample.food_distance is None
                or sample.food_relative_angle is None
            ):
                break
            segment.append(sample)
        segment.reverse()
        return segment

    def _food_orientation(self) -> _RuleEvidence:
        segment = self._food_segment()
        if len(segment) < 3:
            return _missing_rule(
                "target_persistent",
                "Same food target",
                len(segment),
                "samples",
            )
        duration = segment[-1].simulation_time - segment[0].simulation_time
        if duration <= 0.0:
            return _empty_rule()
        visibility_share = len(segment) / len(self.history)
        errors = [abs(float(sample.food_relative_angle)) for sample in segment]
        reduction = (errors[0] - errors[-1]) / duration
        improvements = sum(
            current < previous - 1e-9
            for previous, current in zip(errors, errors[1:])
        )
        consistency = improvements / max(1, len(errors) - 1)
        current = segment[-1]
        relative_angle = float(current.food_relative_angle)
        correct_turn = (
            abs(current.angular_velocity)
            >= self.config.orientation_min_turn_rate
            and relative_angle * current.angular_velocity > 0.0
        )
        passed = (
            visibility_share >= self.config.food_visibility_ratio
            and reduction >= self.config.orientation_min_error_reduction
            and consistency >= self.config.trend_consistency_ratio
            and correct_turn
        )
        evidence = (
            _evidence(
                "food_visibility",
                "Stable target visibility",
                visibility_share,
                None,
                visibility_share >= self.config.food_visibility_ratio,
            ),
            _evidence(
                "target_persistent",
                "Same food target",
                len(segment),
                "samples",
                len(segment) >= 3,
            ),
            _evidence(
                "heading_error_reduction",
                "Heading error reduction",
                reduction,
                "rad/s",
                reduction >= self.config.orientation_min_error_reduction,
            ),
            _evidence(
                "alignment_consistency",
                "Alignment consistency",
                consistency,
                None,
                consistency >= self.config.trend_consistency_ratio,
            ),
            _evidence(
                "realized_turn_toward_food",
                "Realized turn toward food",
                current.angular_velocity,
                "rad/s",
                correct_turn,
            ),
        )
        score = _mean(
            visibility_share,
            min(1.0, duration / self.config.bout_start_seconds),
            reduction / self.config.orientation_min_error_reduction,
            consistency,
            1.0 if correct_turn else 0.0,
        )
        return _RuleEvidence(
            passed,
            score,
            evidence,
            target_id=current.nearest_food_id,
        )

    def _food_approach(self) -> _RuleEvidence:
        segment = self._food_segment()
        if len(segment) < 3:
            return _missing_rule(
                "target_persistent",
                "Same visible food target",
                len(segment),
                "samples",
            )
        duration = segment[-1].simulation_time - segment[0].simulation_time
        if duration <= 0.0:
            return _empty_rule()
        visibility_share = len(segment) / len(self.history)
        distances = [float(sample.food_distance) for sample in segment]
        closing_speed = (distances[0] - distances[-1]) / duration
        closing_steps = sum(
            current < previous - 1e-9
            for previous, current in zip(distances, distances[1:])
        )
        consistency = closing_steps / max(1, len(distances) - 1)
        current = segment[-1]
        alignment = _relative_direction_alignment(
            current,
            float(current.food_relative_angle),
        )
        passed = (
            visibility_share >= self.config.food_visibility_ratio
            and closing_speed >= self.config.approach_min_closing_speed
            and consistency >= self.config.trend_consistency_ratio
            and alignment >= self.config.movement_alignment_threshold
        )
        evidence = (
            _evidence(
                "food_visibility",
                "Stable target visibility",
                visibility_share,
                None,
                visibility_share >= self.config.food_visibility_ratio,
            ),
            _evidence(
                "target_persistent",
                "Same visible food target",
                len(segment),
                "samples",
                len(segment) >= 3,
            ),
            _evidence(
                "closing_speed",
                "Food closing speed",
                closing_speed,
                "px/s",
                closing_speed >= self.config.approach_min_closing_speed,
            ),
            _evidence(
                "closing_consistency",
                "Distance decrease consistency",
                consistency,
                None,
                consistency >= self.config.trend_consistency_ratio,
            ),
            _evidence(
                "movement_toward_food",
                "Realized movement toward food",
                alignment,
                "cos",
                alignment >= self.config.movement_alignment_threshold,
            ),
        )
        score = _mean(
            visibility_share,
            min(1.0, duration / self.config.bout_start_seconds),
            closing_speed / self.config.approach_min_closing_speed,
            consistency,
            max(0.0, alignment),
        )
        return _RuleEvidence(
            passed,
            score,
            evidence,
            target_id=current.nearest_food_id,
        )

    def _feeding(self, observation: BehaviorObservation) -> _RuleEvidence:
        count_delta = (
            observation.food_consumption_count
            - self._last_consumption_count
        )
        energy_delta = (
            observation.food_consumed_energy_total
            - self._last_consumed_energy
        )
        self._last_consumption_count = observation.food_consumption_count
        self._last_consumed_energy = observation.food_consumed_energy_total
        occurred = count_delta > 0 and energy_delta > 0.0
        return _RuleEvidence(
            occurred,
            1.0 if occurred else 0.0,
            (
                _evidence(
                    "food_consumption_event",
                    "Food consumption events",
                    max(0, count_delta),
                    "events",
                    occurred,
                ),
                _evidence(
                    "energy_swallowed",
                    "Energy swallowed",
                    max(0.0, energy_delta),
                    "energy",
                    energy_delta > 0.0,
                ),
            ),
        )

    def _resting(self) -> _RuleEvidence:
        if not self.history:
            return _empty_rule()
        samples = list(self.history)
        threshold = self.config.rest_speed_threshold
        low_share = self._low_speed_count / len(samples)
        current_speed = samples[-1].speed
        mean_speed = self._speed_sum / len(samples)
        passed = current_speed <= threshold and low_share >= 0.80
        evidence = (
            _evidence(
                "current_speed",
                "Current realized speed",
                current_speed,
                "px/s",
                current_speed <= threshold,
            ),
            _evidence(
                "low_speed_share",
                "Low-speed sample share",
                low_share,
                None,
                low_share >= 0.80,
            ),
            _evidence(
                "mean_speed",
                "Window mean speed",
                mean_speed,
                "px/s",
                mean_speed <= threshold,
            ),
        )
        score = _mean(
            1.0 - current_speed / threshold,
            low_share,
            1.0 - mean_speed / threshold,
        )
        return _RuleEvidence(passed, score, evidence)

    def _cohesion(self) -> _RuleEvidence:
        samples = list(self.history)
        if len(samples) < 3:
            return _empty_rule()
        visible = [sample for sample in samples if sample.compatible_group_visible]
        visible_share = len(visible) / len(samples)
        if len(visible) < 3:
            return _RuleEvidence(
                False,
                visible_share,
                (
                    _evidence(
                        "compatible_group_visible",
                        "Compatible group visible",
                        visible_share,
                        None,
                        False,
                    ),
                ),
            )
        duration = visible[-1].simulation_time - visible[0].simulation_time
        if duration <= 0.0:
            return _empty_rule()
        first_distance = visible[0].compatible_group_distance
        last_distance = visible[-1].compatible_group_distance
        if first_distance is None or last_distance is None:
            return _empty_rule()
        closing_speed = (first_distance - last_distance) / duration
        current = visible[-1]
        center_alignment = _relative_direction_alignment(
            current,
            float(current.compatible_group_direction or 0.0),
        )
        velocity_alignment = _vector_alignment(
            current.velocity_x,
            current.velocity_y,
            current.group_velocity_x,
            current.group_velocity_y,
        )
        separation_rate = (last_distance - first_distance) / duration
        outside_share = (
            sum(not sample.personal_space_occupied for sample in visible)
            / len(visible)
        )
        closing_branch = (
            closing_speed >= self.config.cohesion_min_closing_speed
            and center_alignment >= self.config.movement_alignment_threshold
        )
        following_branch = (
            velocity_alignment >= self.config.cohesion_min_velocity_alignment
            and abs(separation_rate)
            <= self.config.cohesion_min_closing_speed
        )
        passed = (
            visible_share >= 0.60
            and outside_share >= 0.80
            and (closing_branch or following_branch)
        )
        evidence = (
            _evidence(
                "compatible_group_visible",
                "Compatible group visibility",
                visible_share,
                None,
                visible_share >= 0.60,
            ),
            _evidence(
                "outside_personal_space",
                "Outside collision-separation range",
                outside_share,
                None,
                outside_share >= 0.80,
            ),
            _evidence(
                "group_closing_speed",
                "Group-center closing speed",
                closing_speed,
                "px/s",
                closing_branch,
            ),
            _evidence(
                "group_velocity_alignment",
                "Realized group velocity alignment",
                velocity_alignment,
                "cos",
                following_branch,
            ),
        )
        motion_score = max(
            closing_speed / self.config.cohesion_min_closing_speed,
            max(0.0, velocity_alignment),
        )
        score = _mean(visible_share, outside_share, motion_score)
        return _RuleEvidence(passed, score, evidence)

    def _alarm_retreat(self) -> _RuleEvidence:
        samples = list(self.history)
        if len(samples) < 3:
            return _empty_rule()
        current = samples[-1]
        duration = current.simulation_time - samples[0].simulation_time
        if duration <= 0.0:
            return _empty_rule()
        temporal_drop = (samples[0].alarm_here - current.alarm_here) / duration
        drop_steps = sum(
            later.alarm_here < earlier.alarm_here - 1e-9
            for earlier, later in zip(samples, samples[1:])
        )
        consistency = drop_steps / max(1, len(samples) - 1)
        forward_alarm = (
            current.alarm_forward_left + current.alarm_forward_right
        ) / 2.0
        spatial_drop = current.alarm_here - forward_alarm
        forward_speed = (
            current.velocity_x * cos(current.heading)
            + current.velocity_y * sin(current.heading)
        )
        passed = (
            current.alarm_here >= self.config.alarm_min_level
            and spatial_drop >= self.config.alarm_min_spatial_gradient
            and temporal_drop >= self.config.alarm_min_temporal_drop
            and consistency >= self.config.trend_consistency_ratio
            and forward_speed >= self.config.alarm_retreat_min_speed
        )
        evidence = (
            _evidence(
                "alarm_level",
                "Local alarm level",
                current.alarm_here,
                None,
                current.alarm_here >= self.config.alarm_min_level,
            ),
            _evidence(
                "down_alarm_gradient",
                "Forward alarm decrease",
                spatial_drop,
                None,
                spatial_drop >= self.config.alarm_min_spatial_gradient,
            ),
            _evidence(
                "alarm_exposure_drop",
                "Alarm exposure decrease",
                temporal_drop,
                "/s",
                temporal_drop >= self.config.alarm_min_temporal_drop,
            ),
            _evidence(
                "realized_retreat_speed",
                "Realized forward retreat speed",
                forward_speed,
                "px/s",
                forward_speed >= self.config.alarm_retreat_min_speed,
            ),
        )
        score = _mean(
            current.alarm_here / self.config.alarm_min_level,
            spatial_drop / self.config.alarm_min_spatial_gradient,
            temporal_drop / self.config.alarm_min_temporal_drop,
            consistency,
            forward_speed / self.config.alarm_retreat_min_speed,
        )
        return _RuleEvidence(passed, score, evidence)


class BehaviorObserverService:
    """Main-process owner of the lazy worker and bounded IPC queues."""

    def __init__(
        self,
        config: BehaviorObserverConfig,
        why_config: CounterfactualWhyConfig | None = None,
        history_config: BehaviorHistoryConfig | None = None,
    ) -> None:
        self.config = config
        self.why_config = (
            why_config
            if why_config is not None
            else CounterfactualWhyConfig(enabled=False)
        )
        self.history_config = history_config or BehaviorHistoryConfig()
        self.latest_snapshot: BehaviorSnapshot | None = None
        self.latest_snapshots: dict[tuple[int, int], BehaviorSnapshot] = {}
        self._progress_snapshots: dict[
            tuple[int, int],
            BehaviorSnapshot,
        ] = {}
        self.latest_why_snapshots: tuple[Any, ...] = ()
        self._focus: tuple[int, int] | None = None
        self._subjects: set[tuple[int, int]] = set()
        self._why_focus: tuple[int, int, int] | None = None
        self._context: Any = None
        self._input_queue: Any = None
        self._result_queue: Any = None
        self._why_control_queue: Any = None
        self._why_probe_queue: Any = None
        self._why_result_queue: Any = None
        self._lifecycle_queue: Any = None
        self._completion_queue: Any = None
        self._history_status_queue: Any = None
        self._stop_event: Any = None
        self._process: Any = None
        self._closed = False
        self._samples_produced = 0
        self._samples_dropped = 0
        self._observations_processed = 0
        self._results_dropped = 0
        self._result_latency_ms: float | None = None
        self._last_error: str | None = None
        self._why_probe_requests = 0
        self._why_probe_requests_dropped = 0
        self._why_probes_superseded = 0
        self._why_evaluations_performed = 0
        self._why_results_dropped = 0
        self._why_result_latency_ms: float | None = None
        self._why_latest_produced_monotonic: float | None = None
        self._why_last_error: str | None = None
        self._why_started_monotonic = monotonic()
        self._completed_bouts: deque[CompletedBehaviorBoutDraft] = deque()
        self._completion_outbox_depth = 0
        self._completion_outbox_high_water = 0
        self._completion_outbox_warning = False
        self._completion_recording_suspended = False
        self._history_completions_not_recorded = 0

    def set_focus(
        self,
        creature_id: int | None,
        selection_generation: int,
    ) -> None:
        focus = (
            None
            if creature_id is None
            else (creature_id, selection_generation)
        )
        if (
            focus == self._focus
            and self._subjects == ({focus} if focus else set())
        ):
            return
        self.set_subjects(() if focus is None else (focus,))
        self._focus = focus
        self.latest_snapshot = (
            None if focus is None else self.latest_snapshots.get(focus)
        )
        self.latest_why_snapshots = ()
        self._why_focus = None

    def set_subjects(
        self,
        subjects: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    ) -> None:
        """Replace temporal subjects while preserving independent analyzers."""
        normalized = {
            (int(creature_id), int(generation))
            for creature_id, generation in subjects
        }
        if normalized == self._subjects:
            return
        removed = self._subjects - normalized
        for identity in removed:
            self.finalize_subject(identity, BehaviorTermination.MODE_SWITCHED)
        self._subjects = normalized
        self.latest_snapshots = {
            identity: snapshot
            for identity, snapshot in self.latest_snapshots.items()
            if identity in normalized
        }
        if self._focus not in normalized:
            self._focus = None
            self.latest_snapshot = None
        if normalized and self.config.enabled:
            self._start()

    def snapshot_for(
        self,
        creature_id: int,
        selection_generation: int,
    ) -> BehaviorSnapshot | None:
        return self.latest_snapshots.get(
            (int(creature_id), int(selection_generation))
        )

    def drain_progress_snapshots(self) -> tuple[BehaviorSnapshot, ...]:
        snapshots = tuple(self._progress_snapshots.values())
        self._progress_snapshots.clear()
        return snapshots

    def set_focal_brain(self, update: Any) -> bool:
        """Send one complete latest-wins focal evaluator update."""
        if not self.why_config.enabled or self._closed:
            return False
        if (
            getattr(update, "creature_id", None) is None
            and self._process is None
        ):
            self._why_focus = None
            self.latest_why_snapshots = ()
            return True
        self._start()
        if self._why_control_queue is None:
            return False
        revision = getattr(update, "brain_revision", None)
        creature_id = getattr(update, "creature_id", None)
        generation = int(getattr(update, "selection_generation", 0))
        self._why_focus = (
            None
            if creature_id is None or revision is None
            else (int(creature_id), generation, int(revision))
        )
        self.latest_why_snapshots = ()
        return self._put_main_latest(
            self._why_control_queue,
            update,
            count_probe_drop=False,
        )

    def submit_why(self, probe: Any) -> bool:
        """Submit explanatory work without ever blocking the caller."""
        if not self.why_config.enabled or self._closed:
            return False
        self._start()
        if self._why_probe_queue is None:
            return False
        self._why_probe_requests += 1
        return self._put_main_latest(
            self._why_probe_queue,
            probe,
            count_probe_drop=True,
        )

    def finalize_focus(self, termination: BehaviorTermination) -> bool:
        focus = self._focus
        if focus is None or self._closed:
            return False
        return self.finalize_subject(focus, termination)

    def finalize_subject(
        self,
        identity: tuple[int, int],
        termination: BehaviorTermination,
    ) -> bool:
        if self._closed:
            return False
        creature_id, generation = identity
        self._start()
        if self._lifecycle_queue is None:
            return False
        request = FinalizeBehaviorFocus(
            creature_id=creature_id,
            selection_generation=generation,
            termination=termination,
        )
        try:
            self._lifecycle_queue.put_nowait(request)
            return True
        except Full:
            self._record_error(
                "Behaviour lifecycle queue is full; focal finalization "
                "could not be scheduled."
            )
            return False
        except (OSError, ValueError) as error:
            self._record_error(
                f"Could not enqueue focal finalization: {error}"
            )
            return False

    def drain_completed_bouts(
        self,
    ) -> tuple[CompletedBehaviorBoutDraft, ...]:
        self._poll_completed_bouts()
        completed = tuple(self._completed_bouts)
        self._completed_bouts.clear()
        return completed

    def _put_main_latest(
        self,
        queue: Any,
        value: Any,
        *,
        count_probe_drop: bool,
    ) -> bool:
        try:
            queue.put_nowait(value)
            return True
        except Full:
            try:
                queue.get_nowait()
                if count_probe_drop:
                    self._why_probe_requests_dropped += 1
            except Empty:
                pass
            try:
                queue.put_nowait(value)
                return True
            except Full:
                if count_probe_drop:
                    self._why_probe_requests_dropped += 1
                return False
        except (OSError, ValueError) as error:
            if count_probe_drop:
                self._why_probe_requests_dropped += 1
            self._record_why_error(
                f"Could not enqueue counterfactual work: {error}"
            )
            return False

    def submit(self, observation: BehaviorObservation) -> bool:
        return self._submit_observation_value(observation, 1)

    def submit_batch(
        self,
        observations: (
            tuple[BehaviorObservation, ...] | list[BehaviorObservation]
        ),
    ) -> bool:
        if not self.config.enabled or self._closed:
            return False
        batch = BehaviorObservationBatch(tuple(observations))
        if not batch.observations:
            return True
        return self._submit_observation_value(
            batch,
            len(batch.observations),
        )

    def _submit_observation_value(
        self,
        value: BehaviorObservation | BehaviorObservationBatch,
        sample_count: int,
    ) -> bool:
        if not self.config.enabled or self._closed:
            return False
        self._start()
        if self._input_queue is None:
            return False
        self._samples_produced += sample_count
        try:
            self._input_queue.put_nowait(value)
            return True
        except Full:
            try:
                dropped = self._input_queue.get_nowait()
                self._samples_dropped += len(
                    getattr(dropped, "observations", (dropped,))
                )
            except Empty:
                pass
            try:
                self._input_queue.put_nowait(value)
                return True
            except Full:
                self._samples_dropped += sample_count
                return False
        except (OSError, ValueError) as error:
            self._record_error(f"Could not enqueue behaviour sample: {error}")
            self._samples_dropped += sample_count
            return False

    def poll(self) -> BehaviorSnapshot | None:
        queue = self._result_queue
        if queue is not None:
            while True:
                try:
                    result = queue.get_nowait()
                except Empty:
                    break
                except (OSError, ValueError) as error:
                    self._record_error(
                        f"Could not read behaviour result: {error}"
                    )
                    break
                if isinstance(result, BehaviorWorkerError):
                    self._record_error(result.message)
                    continue
                if isinstance(result, BehaviorSnapshot):
                    batch = BehaviorSnapshotBatch(
                        snapshots=(result,),
                        observations_processed=result.observations_processed,
                        produced_monotonic=result.produced_monotonic,
                        result_drops=result.result_drops,
                    )
                elif isinstance(result, BehaviorSnapshotBatch):
                    batch = result
                else:
                    self._record_error(
                        f"Unexpected behaviour result {type(result).__name__}."
                    )
                    continue
                self._observations_processed = batch.observations_processed
                self._results_dropped = batch.result_drops
                self._result_latency_ms = max(
                    0.0,
                    (monotonic() - batch.produced_monotonic) * 1000.0,
                )
                for snapshot in batch.snapshots:
                    identity = (
                        snapshot.creature_id,
                        snapshot.selection_generation,
                    )
                    if snapshot.result_drops != batch.result_drops:
                        snapshot = replace(
                            snapshot,
                            result_drops=batch.result_drops,
                        )
                    self._progress_snapshots[identity] = snapshot
                    if (
                        identity not in self._subjects
                        and identity != self._focus
                    ):
                        continue
                    self.latest_snapshots[identity] = snapshot
                    if self._focus == identity:
                        self.latest_snapshot = snapshot
        self._poll_history_status()
        self._poll_why_results()
        process = self._process
        if (
            process is not None
            and process.exitcode is not None
            and not self._closed
            and self._stop_event is not None
            and not self._stop_event.is_set()
        ):
            self._record_error(
                f"Behaviour observer exited unexpectedly "
                f"(code {process.exitcode})."
            )
        return self.latest_snapshot

    def _poll_history_status(self) -> None:
        queue = self._history_status_queue
        if queue is None:
            return
        while True:
            try:
                status = queue.get_nowait()
            except Empty:
                break
            except (OSError, ValueError) as error:
                self._record_error(
                    f"Could not read behaviour history status: {error}"
                )
                break
            if not isinstance(status, BehaviorHistoryWorkerStatus):
                self._record_error(
                    "Unexpected behaviour history status "
                    f"{type(status).__name__}."
                )
                continue
            self._completion_outbox_depth = status.completion_outbox_depth
            self._completion_outbox_high_water = max(
                self._completion_outbox_high_water,
                status.completion_outbox_high_water,
            )
            self._completion_outbox_warning = (
                status.completion_outbox_warning
            )
            self._completion_recording_suspended = (
                status.completion_recording_suspended
            )
            self._history_completions_not_recorded = max(
                self._history_completions_not_recorded,
                status.history_completions_not_recorded,
            )

    def _poll_completed_bouts(self) -> None:
        queue = self._completion_queue
        if queue is None:
            return
        max_batch = (
            self.history_config.completion_outbox_hard_capacity
            + self.history_config.completion_queue_capacity
        )
        for _index in range(max_batch):
            try:
                result = queue.get_nowait()
            except Empty:
                break
            except (OSError, ValueError) as error:
                self._record_error(
                    f"Could not read completed behaviour bout: {error}"
                )
                break
            if isinstance(result, CompletedBehaviorBoutDraft):
                self._completed_bouts.append(result)
            else:
                self._record_error(
                    "Unexpected completed behaviour result "
                    f"{type(result).__name__}."
                )

    def _poll_why_results(self) -> None:
        queue = self._why_result_queue
        if queue is None or not self.why_config.enabled:
            return
        from src.counterfactual_neat import (
            CounterfactualWorkerError,
            WhyBatchResult,
        )

        while True:
            try:
                result = queue.get_nowait()
            except Empty:
                break
            except (OSError, ValueError) as error:
                self._record_why_error(
                    f"Could not read counterfactual result: {error}"
                )
                break
            if isinstance(result, CounterfactualWorkerError):
                self._record_why_error(result.message)
                continue
            if not isinstance(result, WhyBatchResult):
                self._record_why_error(
                    f"Unexpected WHY result {type(result).__name__}."
                )
                continue
            self._why_evaluations_performed = result.evaluations_performed
            self._why_probes_superseded = result.probes_superseded
            self._why_results_dropped = result.result_drops
            self._why_latest_produced_monotonic = result.produced_monotonic
            self._why_result_latency_ms = max(
                0.0,
                (monotonic() - result.produced_monotonic) * 1000.0,
            )
            if self._why_focus == (
                result.creature_id,
                result.selection_generation,
                result.brain_revision,
            ):
                self.latest_why_snapshots = result.snapshots

    @property
    def diagnostics(self) -> BehaviorObserverDiagnostics:
        if not self.config.enabled:
            health = "disabled"
        elif self._last_error is not None:
            health = "error"
        elif self._process is None:
            health = "idle"
        elif self._process.is_alive():
            health = "running"
        else:
            health = "stopped"
        return BehaviorObserverDiagnostics(
            samples_produced=self._samples_produced,
            samples_dropped=self._samples_dropped,
            observations_processed=self._observations_processed,
            results_dropped=self._results_dropped,
            result_latency_ms=self._result_latency_ms,
            input_queue_size=_safe_qsize(self._input_queue),
            worker_health=health,
            last_error=self._last_error,
            completion_outbox_depth=self._completion_outbox_depth,
            completion_outbox_high_water=(
                self._completion_outbox_high_water
            ),
            completion_outbox_warning=self._completion_outbox_warning,
            completion_recording_suspended=(
                self._completion_recording_suspended
            ),
            history_completions_not_recorded=(
                self._history_completions_not_recorded
            ),
        )

    @property
    def counterfactual_diagnostics(self) -> Any:
        from src.counterfactual_neat import CounterfactualDiagnostics

        if not self.why_config.enabled:
            health = "disabled"
        elif self._why_last_error is not None:
            health = "error"
        elif self._process is None:
            health = "idle"
        elif self._process.is_alive():
            health = "running"
        else:
            health = "stopped"
        now = monotonic()
        age = (
            None
            if self._why_latest_produced_monotonic is None
            else max(
                0.0,
                (now - self._why_latest_produced_monotonic) * 1000.0,
            )
        )
        elapsed = max(1e-12, now - self._why_started_monotonic)
        return CounterfactualDiagnostics(
            probe_requests=self._why_probe_requests,
            probe_requests_dropped=self._why_probe_requests_dropped,
            probes_superseded=self._why_probes_superseded,
            evaluations_performed=self._why_evaluations_performed,
            result_drops=self._why_results_dropped,
            result_latency_ms=self._why_result_latency_ms,
            latest_result_age_ms=age,
            evaluations_per_second=(
                self._why_evaluations_performed / elapsed
            ),
            probe_queue_size=_safe_qsize(self._why_probe_queue),
            worker_health=health,
            last_error=self._why_last_error,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._stop_event is not None:
            self._stop_event.set()
        process = self._process
        if process is not None:
            process.join(timeout=2.0)
            if process.is_alive():
                LOGGER.warning(
                    "Behaviour observer did not stop in time; terminating it."
                )
                process.terminate()
                process.join(timeout=0.5)
        for queue in (
            self._input_queue,
            self._result_queue,
            self._why_control_queue,
            self._why_probe_queue,
            self._why_result_queue,
            self._lifecycle_queue,
            self._completion_queue,
            self._history_status_queue,
        ):
            if queue is None:
                continue
            try:
                queue.close()
                queue.join_thread()
            except (OSError, ValueError):
                pass

    def _start(self) -> None:
        if (
            not (self.config.enabled or self.why_config.enabled)
            or self._closed
            or (
                self._process is not None
                and self._process.is_alive()
            )
        ):
            return
        if self._process is not None and self._process.exitcode is not None:
            return
        try:
            self._context = multiprocessing.get_context("spawn")
            self._input_queue = self._context.Queue(
                maxsize=self.config.input_queue_capacity
            )
            self._result_queue = self._context.Queue(
                maxsize=self.config.result_queue_capacity
            )
            self._why_control_queue = self._context.Queue(
                maxsize=self.why_config.control_queue_capacity
            )
            self._why_probe_queue = self._context.Queue(
                maxsize=self.why_config.probe_queue_capacity
            )
            self._why_result_queue = self._context.Queue(
                maxsize=self.why_config.result_queue_capacity
            )
            # A mode switch can finalize every creature in a maximally sized
            # automatic cohort at once.
            self._lifecycle_queue = self._context.Queue(maxsize=128)
            self._completion_queue = self._context.Queue(
                maxsize=self.history_config.completion_queue_capacity
            )
            self._history_status_queue = self._context.Queue(maxsize=1)
            self._stop_event = self._context.Event()
            self._process = self._context.Process(
                target=_behavior_worker_main,
                args=(
                    self.config,
                    self.why_config,
                    self.history_config,
                    self._input_queue,
                    self._result_queue,
                    self._why_control_queue,
                    self._why_probe_queue,
                    self._why_result_queue,
                    self._lifecycle_queue,
                    self._completion_queue,
                    self._history_status_queue,
                    self._stop_event,
                ),
                name="behavior-observer",
            )
            self._process.start()
        except (OSError, RuntimeError) as error:
            self._record_error(
                f"Could not start behaviour observer: {error}"
            )

    def _record_error(self, message: str) -> None:
        if message == self._last_error:
            return
        self._last_error = message
        LOGGER.error(message)

    def _record_why_error(self, message: str) -> None:
        if message == self._why_last_error:
            return
        self._why_last_error = message
        LOGGER.error(message)


def _behavior_worker_main(
    config: BehaviorObserverConfig,
    why_config: CounterfactualWhyConfig,
    history_config: BehaviorHistoryConfig,
    input_queue: Any,
    result_queue: Any,
    why_control_queue: Any,
    why_probe_queue: Any,
    why_result_queue: Any,
    lifecycle_queue: Any,
    completion_queue: Any,
    history_status_queue: Any,
    stop_event: Any,
) -> None:
    import pickle

    from src.counterfactual_neat import (
        CounterfactualBoutAggregator,
        CounterfactualProbeInput,
        CounterfactualProbeJob,
        CounterfactualWorkerError,
        FocalBrainUpdate,
        PureNeatEvaluator,
        WhyBatchResult,
        validate_probe,
    )

    analyzers: dict[tuple[int, int], TemporalBehaviorAnalyzer] = {}
    observations_processed_total = 0
    result_drops = 0
    why_result_drops = 0
    why_evaluations = 0
    why_superseded = 0
    evaluator: PureNeatEvaluator | None = None
    why_focus: tuple[int, int, int] | None = None
    pending_job: CounterfactualProbeJob | None = None
    aggregator = CounterfactualBoutAggregator(
        why_config.history_capacity,
        why_config.target_center_dead_zone_radians,
    )
    completion_outbox: deque[CompletedBehaviorBoutDraft] = deque()
    completion_outbox_high_water = 0
    completion_recording_suspended = False
    history_completions_not_recorded = 0
    outbox_warning_latched = False
    outbox_error_latched = False
    last_published_history_status: (
        tuple[int, int, bool, bool, int] | None
    ) = None

    def publish_history_status() -> None:
        nonlocal last_published_history_status
        status_tuple = (
            len(completion_outbox),
            completion_outbox_high_water,
            outbox_warning_latched,
            completion_recording_suspended,
            history_completions_not_recorded,
        )
        if status_tuple == last_published_history_status:
            return
        last_published_history_status = status_tuple
        _put_latest(
            history_status_queue,
            BehaviorHistoryWorkerStatus(
                completion_outbox_depth=status_tuple[0],
                completion_outbox_high_water=status_tuple[1],
                completion_outbox_warning=status_tuple[2],
                completion_recording_suspended=status_tuple[3],
                history_completions_not_recorded=status_tuple[4],
            ),
            0,
        )

    def queue_completed_bouts(analyzer: TemporalBehaviorAnalyzer) -> None:
        nonlocal completion_outbox_high_water
        nonlocal completion_recording_suspended
        nonlocal history_completions_not_recorded
        nonlocal outbox_warning_latched
        nonlocal outbox_error_latched

        for draft in analyzer.drain_completed_bouts():
            why_summary = aggregator.finalize_completed_bout(
                draft.creature_id,
                draft.selection_generation,
                draft.behavior,
                draft.local_bout_id,
            )
            completed = replace(draft, why_summary=why_summary)
            if (
                completion_recording_suspended
                or len(completion_outbox)
                >= history_config.completion_outbox_hard_capacity
            ):
                completion_recording_suspended = True
                history_completions_not_recorded += 1
                if not outbox_error_latched:
                    LOGGER.error(
                        "Completed behaviour history outbox reached its hard "
                        "capacity; long-term recording is suspended until "
                        "the history consumer recovers."
                    )
                    outbox_error_latched = True
                continue
            completion_outbox.append(completed)
            completion_outbox_high_water = max(
                completion_outbox_high_water,
                len(completion_outbox),
            )
            if (
                len(completion_outbox)
                >= history_config.completion_outbox_soft_capacity
                and not outbox_warning_latched
            ):
                LOGGER.warning(
                    "Completed behaviour history outbox crossed its soft "
                    "capacity; records remain queued."
                )
                outbox_warning_latched = True
            if (
                len(completion_outbox)
                >= history_config.completion_outbox_hard_capacity
            ):
                completion_recording_suspended = True
                if not outbox_error_latched:
                    LOGGER.error(
                        "Completed behaviour history outbox reached its hard "
                        "capacity; long-term recording is suspended until "
                        "the history consumer recovers."
                    )
                    outbox_error_latched = True
        publish_history_status()

    def flush_completion_outbox() -> None:
        nonlocal completion_recording_suspended
        nonlocal outbox_warning_latched
        nonlocal outbox_error_latched

        while completion_outbox:
            try:
                completion_queue.put_nowait(completion_outbox[0])
            except Full:
                break
            except (OSError, ValueError):
                break
            completion_outbox.popleft()
        if (
            completion_recording_suspended
            and len(completion_outbox)
            <= history_config.completion_outbox_recovery_capacity
        ):
            completion_recording_suspended = False
            outbox_warning_latched = False
            outbox_error_latched = False
            LOGGER.info(
                "Completed behaviour history consumer recovered; "
                "long-term recording resumed."
            )
        publish_history_status()

    def process_observation_batch(value: Any) -> None:
        nonlocal observations_processed_total
        nonlocal result_drops
        observations = (
            value.observations
            if isinstance(value, BehaviorObservationBatch)
            else (value,)
        )
        snapshots: list[BehaviorSnapshot] = []
        for observation in observations:
            identity = (
                getattr(observation, "creature_id", None),
                getattr(observation, "selection_generation", None),
            )
            try:
                if not isinstance(observation, BehaviorObservation):
                    raise TypeError(
                        f"Unexpected observation {type(observation).__name__}."
                    )
                analyzer = analyzers.get(identity)
                if analyzer is None:
                    analyzer = TemporalBehaviorAnalyzer(config, history_config)
                    analyzers[identity] = analyzer
                snapshots.append(analyzer.process(observation))
                observations_processed_total += 1
                queue_completed_bouts(analyzer)
            except Exception as error:  # Worker must report and continue.
                result_drops = _put_latest(
                    result_queue,
                    BehaviorWorkerError(
                        creature_id=identity[0],
                        selection_generation=identity[1],
                        message=(
                            "Behaviour analysis failed: "
                            f"{type(error).__name__}: {error}"
                        ),
                    ),
                    result_drops,
                )
        if not snapshots:
            return
        batch = BehaviorSnapshotBatch(
            snapshots=tuple(snapshots),
            observations_processed=observations_processed_total,
            produced_monotonic=monotonic(),
            result_drops=result_drops,
        )
        result_drops = _put_latest(result_queue, batch, result_drops)

    while not stop_event.is_set():
        # Temporal observations are the primary signal. Process every item
        # already pending before touching explanatory work.
        try:
            observation = input_queue.get(timeout=0.005)
        except Empty:
            observation = None
        except (OSError, EOFError, ValueError):
            return
        if observation is not None:
            process_observation_batch(observation)
            while True:
                try:
                    process_observation_batch(input_queue.get_nowait())
                except Empty:
                    break
                except (OSError, EOFError, ValueError):
                    return

        # Forced focus/death finalization is ordered after all pending
        # observations and before explanatory work.
        while True:
            try:
                lifecycle = lifecycle_queue.get_nowait()
            except Empty:
                break
            except (OSError, EOFError, ValueError):
                return
            if not isinstance(lifecycle, FinalizeBehaviorFocus):
                continue
            identity = (
                lifecycle.creature_id,
                lifecycle.selection_generation,
            )
            analyzer = analyzers.pop(identity, None)
            if analyzer is not None:
                analyzer.force_finalize(lifecycle.termination)
                queue_completed_bouts(analyzer)

        # Completed records are lossless below the hard emergency bound and
        # never block the high-priority temporal analyzer.
        flush_completion_outbox()

        # Apply only the newest complete focal evaluator update.
        control, _discarded_controls = _drain_latest(why_control_queue)
        if control is not None:
            pending_job = None
            aggregator.reset()
            try:
                if not isinstance(control, FocalBrainUpdate):
                    raise TypeError(
                        f"Unexpected control {type(control).__name__}."
                    )
                if (
                    control.creature_id is None
                    or control.brain_revision is None
                    or control.evaluator_payload is None
                ):
                    evaluator = None
                    why_focus = None
                else:
                    restored = pickle.loads(control.evaluator_payload)
                    if not isinstance(restored, PureNeatEvaluator):
                        raise TypeError("Invalid focal evaluator payload.")
                    evaluator = restored
                    why_focus = (
                        control.creature_id,
                        control.selection_generation,
                        control.brain_revision,
                    )
            except Exception as error:
                evaluator = None
                why_focus = None
                why_result_drops = _put_latest(
                    why_result_queue,
                    CounterfactualWorkerError(
                        creature_id=getattr(control, "creature_id", None),
                        selection_generation=getattr(
                            control,
                            "selection_generation",
                            None,
                        ),
                        brain_revision=getattr(
                            control,
                            "brain_revision",
                            None,
                        ),
                        message=(
                            "Counterfactual brain update failed: "
                            f"{type(error).__name__}: {error}"
                        ),
                    ),
                    why_result_drops,
                )

        # New factual state supersedes queued or partially evaluated WHY work.
        newest_probe, discarded_probes = _drain_latest(why_probe_queue)
        why_superseded += discarded_probes
        if newest_probe is not None:
            if pending_job is not None:
                why_superseded += 1
            pending_job = None
            try:
                if not isinstance(newest_probe, CounterfactualProbeInput):
                    raise TypeError(
                        f"Unexpected probe {type(newest_probe).__name__}."
                    )
                validate_probe(newest_probe)
                identity = (
                    newest_probe.creature_id,
                    newest_probe.selection_generation,
                    newest_probe.brain_revision,
                )
                if evaluator is not None and identity == why_focus:
                    pending_job = CounterfactualProbeJob(
                        newest_probe,
                        evaluator,
                    )
            except Exception as error:
                why_result_drops = _put_latest(
                    why_result_queue,
                    CounterfactualWorkerError(
                        creature_id=getattr(
                            newest_probe,
                            "creature_id",
                            None,
                        ),
                        selection_generation=getattr(
                            newest_probe,
                            "selection_generation",
                            None,
                        ),
                        brain_revision=getattr(
                            newest_probe,
                            "brain_revision",
                            None,
                        ),
                        message=(
                            "Counterfactual probe failed: "
                            f"{type(error).__name__}: {error}"
                        ),
                    ),
                    why_result_drops,
                )

        # Best effort: one activation, then immediately re-check behavior work.
        if pending_job is not None:
            try:
                completed = pending_job.advance()
                why_evaluations += 1
                if completed:
                    probe = pending_job.probe
                    snapshots = aggregator.complete_job(pending_job)
                    produced = monotonic()
                    batch = WhyBatchResult(
                        creature_id=probe.creature_id,
                        selection_generation=probe.selection_generation,
                        brain_revision=probe.brain_revision,
                        simulation_time=probe.simulation_time,
                        snapshots=snapshots,
                        evaluations_performed=why_evaluations,
                        probes_superseded=why_superseded,
                        result_drops=why_result_drops,
                        produced_monotonic=produced,
                    )
                    why_result_drops = _put_latest(
                        why_result_queue,
                        batch,
                        why_result_drops,
                    )
                    pending_job = None
            except Exception as error:
                probe = pending_job.probe
                pending_job = None
                why_result_drops = _put_latest(
                    why_result_queue,
                    CounterfactualWorkerError(
                        creature_id=probe.creature_id,
                        selection_generation=probe.selection_generation,
                        brain_revision=probe.brain_revision,
                        message=(
                            "Counterfactual evaluation failed: "
                            f"{type(error).__name__}: {error}"
                        ),
                    ),
                    why_result_drops,
                )


def _drain_latest(queue: Any) -> tuple[Any | None, int]:
    latest = None
    discarded = 0
    while True:
        try:
            value = queue.get_nowait()
        except Empty:
            return latest, discarded
        except (OSError, ValueError):
            return latest, discarded
        if latest is not None:
            discarded += 1
        latest = value


def _put_latest(queue: Any, value: object, drops: int) -> int:
    try:
        queue.put_nowait(value)
        return drops
    except Full:
        try:
            queue.get_nowait()
            drops += 1
        except Empty:
            pass
        if hasattr(value, "result_drops"):
            value = replace(value, result_drops=drops)
        try:
            queue.put_nowait(value)
        except Full:
            drops += 1
        return drops
    except (OSError, ValueError):
        return drops + 1


def _safe_qsize(queue: Any) -> int | None:
    if queue is None:
        return None
    try:
        return int(queue.qsize())
    except (AttributeError, NotImplementedError, OSError):
        return None


def _relative_direction_alignment(
    observation: BehaviorObservation,
    relative_angle: float,
) -> float:
    realized_speed = hypot(
        observation.velocity_x,
        observation.velocity_y,
    )
    if realized_speed <= 1e-12:
        return -1.0
    target_angle = observation.heading + relative_angle
    return (
        observation.velocity_x * cos(target_angle)
        + observation.velocity_y * sin(target_angle)
    ) / realized_speed


def _vector_alignment(
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    first = hypot(ax, ay)
    second = hypot(bx, by)
    if first <= 1e-12 or second <= 1e-12:
        return -1.0
    return (ax * bx + ay * by) / (first * second)


def _evidence(
    key: str,
    label: str,
    value: float,
    unit: str | None,
    passed: bool,
) -> BehaviorEvidence:
    return BehaviorEvidence(
        key=key,
        label=label,
        value=float(value),
        unit=unit,
        passed=bool(passed),
    )


def _missing_rule(
    key: str,
    label: str,
    value: float,
    unit: str | None,
) -> _RuleEvidence:
    return _RuleEvidence(
        False,
        0.0,
        (_evidence(key, label, value, unit, False),),
    )


def _empty_rule() -> _RuleEvidence:
    return _RuleEvidence(False, 0.0, ())


def _mean(*values: float) -> float:
    if not values:
        return 0.0
    return sum(_clamp01(value) for value in values) / len(values)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
