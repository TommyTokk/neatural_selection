"""Spawn-safe focal temporal behaviour observation.

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

from configs.sim_config import BehaviorObserverConfig


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
class BehaviorWorkerError:
    creature_id: int | None
    selection_generation: int | None
    message: str


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


@dataclass(frozen=True, slots=True)
class _RuleEvidence:
    present: bool
    score: float
    evidence: tuple[BehaviorEvidence, ...]


@dataclass(slots=True)
class _BoutState:
    status: BoutStatus | None = None
    candidate_since: float | None = None
    active_since: float | None = None
    last_evidence_time: float | None = None

    def reset(self) -> None:
        self.status = None
        self.candidate_since = None
        self.active_since = None
        self.last_evidence_time = None


class TemporalBehaviorAnalyzer:
    """Analyze one focal creature and reset completely when focus changes."""

    def __init__(self, config: BehaviorObserverConfig) -> None:
        self.config = config
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
        for bout in self.bouts.values():
            bout.reset()

    def process(self, observation: BehaviorObservation) -> BehaviorSnapshot:
        identity = (
            observation.creature_id,
            observation.selection_generation,
        )
        if identity != (self.creature_id, self.selection_generation):
            self.reset(*identity)

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
            )
            if state is not None:
                states.append(state)
        feeding_state = self._update_feeding(
            feeding,
            observation.simulation_time,
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

    def _update_bout(
        self,
        behavior: BehaviorKind,
        rule: _RuleEvidence,
        now: float,
    ) -> BehaviorStateSnapshot | None:
        bout = self.bouts[behavior]
        if rule.present:
            bout.last_evidence_time = now
            if bout.status is None:
                bout.status = BoutStatus.EMERGING
                bout.candidate_since = now
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
            last = bout.last_evidence_time
            if (
                last is None
                or now - last > self.config.bout_end_grace_seconds
            ):
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
        )

    def _update_feeding(
        self,
        rule: _RuleEvidence,
        now: float,
    ) -> BehaviorStateSnapshot | None:
        bout = self.bouts[BehaviorKind.FEEDING]
        if rule.present:
            self._last_feeding_rule = rule
            if bout.status is None:
                bout.active_since = now
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
        )

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
        return _RuleEvidence(passed, score, evidence)

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
        return _RuleEvidence(passed, score, evidence)

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

    def __init__(self, config: BehaviorObserverConfig) -> None:
        self.config = config
        self.latest_snapshot: BehaviorSnapshot | None = None
        self._focus: tuple[int, int] | None = None
        self._context: Any = None
        self._input_queue: Any = None
        self._result_queue: Any = None
        self._stop_event: Any = None
        self._process: Any = None
        self._closed = False
        self._samples_produced = 0
        self._samples_dropped = 0
        self._observations_processed = 0
        self._results_dropped = 0
        self._result_latency_ms: float | None = None
        self._last_error: str | None = None

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
        if focus == self._focus:
            return
        self._focus = focus
        self.latest_snapshot = None
        if focus is not None and self.config.enabled:
            self._start()

    def submit(self, observation: BehaviorObservation) -> bool:
        if not self.config.enabled or self._closed:
            return False
        self._start()
        if self._input_queue is None:
            return False
        self._samples_produced += 1
        try:
            self._input_queue.put_nowait(observation)
            return True
        except Full:
            try:
                self._input_queue.get_nowait()
                self._samples_dropped += 1
            except Empty:
                pass
            try:
                self._input_queue.put_nowait(observation)
                return True
            except Full:
                self._samples_dropped += 1
                return False
        except (OSError, ValueError) as error:
            self._record_error(f"Could not enqueue behaviour sample: {error}")
            self._samples_dropped += 1
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
                if not isinstance(result, BehaviorSnapshot):
                    self._record_error(
                        f"Unexpected behaviour result {type(result).__name__}."
                    )
                    continue
                self._observations_processed = result.observations_processed
                self._results_dropped = result.result_drops
                self._result_latency_ms = max(
                    0.0,
                    (monotonic() - result.produced_monotonic) * 1000.0,
                )
                if self._focus == (
                    result.creature_id,
                    result.selection_generation,
                ):
                    self.latest_snapshot = result
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
        for queue in (self._input_queue, self._result_queue):
            if queue is None:
                continue
            try:
                queue.close()
                queue.join_thread()
            except (OSError, ValueError):
                pass

    def _start(self) -> None:
        if (
            not self.config.enabled
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
            self._stop_event = self._context.Event()
            self._process = self._context.Process(
                target=_behavior_worker_main,
                args=(
                    self.config,
                    self._input_queue,
                    self._result_queue,
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


def _behavior_worker_main(
    config: BehaviorObserverConfig,
    input_queue: Any,
    result_queue: Any,
    stop_event: Any,
) -> None:
    analyzer = TemporalBehaviorAnalyzer(config)
    result_drops = 0
    while not stop_event.is_set():
        try:
            observation = input_queue.get(timeout=0.1)
        except Empty:
            continue
        except (OSError, EOFError, ValueError):
            return
        try:
            snapshot = analyzer.process(observation)
        except Exception as error:  # Worker must report and continue.
            message = BehaviorWorkerError(
                creature_id=getattr(observation, "creature_id", None),
                selection_generation=getattr(
                    observation,
                    "selection_generation",
                    None,
                ),
                message=(
                    "Behaviour analysis failed: "
                    f"{type(error).__name__}: {error}"
                ),
            )
            result_drops = _put_latest(
                result_queue,
                message,
                result_drops,
            )
            continue
        snapshot = replace(snapshot, result_drops=result_drops)
        result_drops = _put_latest(
            result_queue,
            snapshot,
            result_drops,
        )


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
        if isinstance(value, BehaviorSnapshot):
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
