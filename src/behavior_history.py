"""Compact completed-behaviour history and bout-level aggregation.

This module deliberately stores only immutable, finalized records.  Active
observer windows and counterfactual probe samples remain owned by the live
observer worker and never enter the long-term store.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field, replace
from enum import Enum
from math import floor, isfinite
from statistics import median
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from src.behavior_observer import BehaviorEvidence, BehaviorKind
    from src.counterfactual_neat import (
        EffectDirection,
        InfluenceLabel,
        SemanticIntervention,
    )


class BehaviorTermination(str, Enum):
    NATURAL = "natural"
    FOCUS_CHANGED = "focus_changed"
    CREATURE_DIED = "creature_died"
    MODE_SWITCHED = "mode_switched"


class BehaviorOutcome(str, Enum):
    FOOD_CONSUMED = "food_consumed"
    ABANDONED = "abandoned"
    TARGET_LOST = "target_lost"
    APPROACH_STARTED = "approach_started"
    ENDED_WITHOUT_APPROACH = "ended_without_approach"
    CONSUMPTION_EVENT = "consumption_event"
    ALARM_EXPOSURE_REDUCED = "alarm_exposure_reduced"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class BehaviorEvidenceSummary:
    key: str
    label: str
    unit: str | None
    sample_count: int
    passed_count: int
    median_value: float
    p25: float | None
    p75: float | None
    first_value: float
    last_value: float
    quantiles_estimated: bool = False
    total_value: float = 0.0

    @property
    def passed_fraction(self) -> float:
        return (
            0.0
            if self.sample_count <= 0
            else self.passed_count / self.sample_count
        )


@dataclass(frozen=True, slots=True)
class EffectDirectionCounts:
    supportive: int = 0
    suppressive: int = 0
    reversing: int = 0
    mixed: int = 0
    minimal: int = 0

    @property
    def total(self) -> int:
        return (
            self.supportive
            + self.suppressive
            + self.reversing
            + self.mixed
            + self.minimal
        )


@dataclass(frozen=True, slots=True)
class CompletedOutputEffectSummary:
    output_name: str
    sample_count: int
    median_factual: float
    median_counterfactual: float
    median_delta: float
    delta_p25: float | None = None
    delta_p75: float | None = None
    quantiles_estimated: bool = False


@dataclass(frozen=True, slots=True)
class CompletedSemanticEffect:
    intervention: SemanticIntervention
    sample_count: int
    median_influence: float
    p25: float | None
    p75: float | None
    influence_label: InfluenceLabel
    effect_direction: EffectDirection
    direction_counts: EffectDirectionCounts
    output_summaries: tuple[CompletedOutputEffectSummary, ...]
    quantiles_estimated: bool = False


@dataclass(frozen=True, slots=True)
class CompletedWhyExplanation:
    behavior: BehaviorKind
    bout_id: int
    effects: tuple[CompletedSemanticEffect, ...]


@dataclass(frozen=True, slots=True)
class CompletedBehaviorBout:
    creature_id: int
    behavior: BehaviorKind
    bout_id: int
    start_time: float
    end_time: float
    duration: float
    evidence_summary: tuple[BehaviorEvidenceSummary, ...]
    outcome: BehaviorOutcome | None
    termination: BehaviorTermination
    why_summary: CompletedWhyExplanation | None = None


@dataclass(frozen=True, slots=True)
class CompletedBehaviorBoutDraft:
    """Worker-local completion before the store assigns a persistent bout id."""

    creature_id: int
    selection_generation: int
    behavior: BehaviorKind
    local_bout_id: int
    start_time: float
    end_time: float
    duration: float
    evidence_summary: tuple[BehaviorEvidenceSummary, ...]
    outcome: BehaviorOutcome | None
    termination: BehaviorTermination
    why_summary: CompletedWhyExplanation | None = None


@dataclass(frozen=True, slots=True)
class BehaviorLifetimeWhySummary:
    intervention: SemanticIntervention
    behavior_bout_count: int
    contributing_bout_count: int
    median_bout_influence: float
    p25: float | None
    p75: float | None
    influence_label: InfluenceLabel
    direction_counts: EffectDirectionCounts
    quantiles_estimated: bool = False


@dataclass(frozen=True, slots=True)
class BehaviorLifetimeSummary:
    behavior: BehaviorKind
    completed_bout_count: int
    total_duration: float
    median_duration: float
    outcome_counts: tuple[tuple[BehaviorOutcome, int], ...]
    why_summaries: tuple[BehaviorLifetimeWhySummary, ...]


@dataclass(frozen=True, slots=True)
class CreatureBehaviorSummary:
    creature_id: int
    completed_bout_count: int
    behaviors: tuple[BehaviorLifetimeSummary, ...]
    stable_pattern_threshold: int


@dataclass(frozen=True, slots=True)
class CreatureHistoryIndexEntry:
    creature_id: int
    creature_name: str
    deceased: bool
    last_observed_time: float
    completed_bout_count: int
    species_id: int | None = None
    total_observation_seconds: float = 0.0
    observation_session_count: int = 0
    last_observation_mode: str | None = None
    active: bool = False


@dataclass(frozen=True, slots=True)
class SpeciesBehaviorSummary:
    behavior: BehaviorKind
    completed_bout_count: int
    total_duration: float
    median_duration: float
    bouts_per_creature_hour: float


@dataclass(frozen=True, slots=True)
class SpeciesBehaviorReport:
    species_id: int | None
    observed_creature_count: int
    total_observation_seconds: float
    completed_bout_count: int
    behaviors: tuple[SpeciesBehaviorSummary, ...]
    creatures: tuple[CreatureHistoryIndexEntry, ...]
    alive_population: int = 0
    monitored_count: int = 0


@dataclass(frozen=True, slots=True)
class SpeciesBehaviorIndexEntry:
    species_id: int | None
    alive_population: int
    monitored_count: int
    observed_creature_count: int
    total_observation_seconds: float
    completed_bout_count: int
    active: bool


@dataclass(frozen=True, slots=True)
class CreatureBehaviorReport:
    creature_id: int
    creature_name: str
    deceased: bool
    completed_bouts: tuple[CompletedBehaviorBout, ...]
    summary: CreatureBehaviorSummary
    history_incomplete: bool
    history_completions_not_recorded: int
    detailed_bouts_dropped: int
    species_id: int | None = None
    total_observation_seconds: float = 0.0
    observation_session_count: int = 0
    last_observation_mode: str | None = None


@dataclass(frozen=True, slots=True)
class BehaviorHistoryDiagnostics:
    bout_finalizations: int = 0
    why_summaries_finalized: int = 0
    completed_bouts_stored: int = 0
    detailed_bouts_dropped: int = 0
    remembered_creatures: int = 0
    creatures_evicted: int = 0
    duplicate_completions_ignored: int = 0
    history_incomplete: bool = False
    history_completions_not_recorded: int = 0


class BoundedMetricAccumulator:
    """Exact short-bout samples with deterministic long-bout compaction."""

    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity < 4:
            raise ValueError("Metric sample capacity must be at least four.")
        self.capacity = capacity
        self.total_count = 0
        self.passed_count = 0
        self.value_total = 0.0
        self.first_value: float | None = None
        self.last_value: float | None = None
        self._stride = 1
        self._samples: list[tuple[int, float]] = []
        self._compacted = False

    @property
    def compacted(self) -> bool:
        return self._compacted

    def add(self, value: float, passed: bool = True) -> None:
        numeric = float(value)
        if not isfinite(numeric):
            return
        index = self.total_count
        self.total_count += 1
        self.passed_count += int(bool(passed))
        self.value_total += numeric
        if self.first_value is None:
            self.first_value = numeric
        self.last_value = numeric

        # The most recent non-grid sample is a provisional exact-last value.
        # Replace it as time advances, then retain the new exact latest value.
        if self._samples:
            previous_index = self._samples[-1][0]
            if (
                previous_index != 0
                and previous_index % self._stride != 0
            ):
                self._samples.pop()
        if index == 0 or index % self._stride == 0:
            self._samples.append((index, numeric))
        if not self._samples or self._samples[-1][0] != index:
            self._samples.append((index, numeric))
        self._compact_if_needed()

    def _compact_if_needed(self) -> None:
        while len(self._samples) > self.capacity:
            self._compacted = True
            self._stride *= 2
            first = self._samples[0]
            last = self._samples[-1]
            interior = [
                item
                for item in self._samples[1:-1]
                if item[0] % self._stride == 0
            ]
            self._samples = [first, *interior, last]

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.add(value)

    def summary_values(
        self,
    ) -> tuple[float, float | None, float | None, bool]:
        if not self._samples:
            return 0.0, None, None, self._compacted
        values = sorted(value for _index, value in self._samples)
        return (
            float(median(values)),
            _percentile(values, 0.25),
            _percentile(values, 0.75),
            self._compacted,
        )


class BehaviorEvidenceAccumulator:
    """Accumulate the analyzer's existing operational evidence values."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._metrics: OrderedDict[
            str,
            tuple[str, str | None, BoundedMetricAccumulator],
        ] = OrderedDict()

    def add(self, evidence_items: Iterable[BehaviorEvidence]) -> None:
        for evidence in evidence_items:
            metric = self._metrics.get(evidence.key)
            if metric is None:
                metric = (
                    evidence.label,
                    evidence.unit,
                    BoundedMetricAccumulator(self.capacity),
                )
                self._metrics[evidence.key] = metric
            metric[2].add(evidence.value, evidence.passed)

    def summaries(self) -> tuple[BehaviorEvidenceSummary, ...]:
        summaries: list[BehaviorEvidenceSummary] = []
        for key, (label, unit, accumulator) in self._metrics.items():
            if accumulator.total_count <= 0:
                continue
            median_value, p25, p75, estimated = (
                accumulator.summary_values()
            )
            summaries.append(
                BehaviorEvidenceSummary(
                    key=key,
                    label=label,
                    unit=unit,
                    sample_count=accumulator.total_count,
                    passed_count=accumulator.passed_count,
                    median_value=median_value,
                    p25=p25,
                    p75=p75,
                    first_value=float(
                        0.0
                        if accumulator.first_value is None
                        else accumulator.first_value
                    ),
                    last_value=float(
                        0.0
                        if accumulator.last_value is None
                        else accumulator.last_value
                    ),
                    quantiles_estimated=estimated,
                    total_value=accumulator.value_total,
                )
            )
        return tuple(summaries)


@dataclass(slots=True)
class _CreatureHistory:
    creature_id: int
    creature_name: str
    deceased: bool
    last_observed_time: float
    next_bout_id: int
    bouts: deque[CompletedBehaviorBout]
    summary: CreatureBehaviorSummary
    detailed_bouts_dropped: int = 0
    species_id: int | None = None
    total_observation_seconds: float = 0.0
    observation_session_count: int = 0
    last_observation_mode: str | None = None
    active: bool = False
    progress_by_generation: OrderedDict[
        int,
        tuple[int, float | None],
    ] = field(default_factory=OrderedDict)


class CreatureBehaviorHistoryStore:
    """Bounded main-process store of immutable finalized bouts."""

    def __init__(
        self,
        *,
        max_completed_bouts_per_creature: int,
        max_remembered_creatures: int,
        minimum_stable_bouts: int,
    ) -> None:
        self.max_completed_bouts_per_creature = (
            max_completed_bouts_per_creature
        )
        self.max_remembered_creatures = max_remembered_creatures
        self.minimum_stable_bouts = minimum_stable_bouts
        self._creatures: OrderedDict[int, _CreatureHistory] = OrderedDict()
        self._seen_sources: OrderedDict[
            tuple[int, int, Any, int],
            None,
        ] = OrderedDict()
        self._seen_source_capacity = (
            max_completed_bouts_per_creature * max_remembered_creatures
        )
        self._creatures_evicted = 0
        self._bout_finalizations = 0
        self._why_summaries_finalized = 0
        self._duplicate_completions_ignored = 0
        self._history_incomplete = False
        self._history_completions_not_recorded = 0

    def register_creature(
        self,
        creature_id: int,
        creature_name: str,
        simulation_time: float,
        *,
        species_id: int | None = None,
        observation_mode: object | None = None,
        observation_generation: int | None = None,
        active: bool = False,
    ) -> None:
        mode_value = (
            None
            if observation_mode is None
            else str(getattr(observation_mode, "value", observation_mode))
        )
        record = self._creatures.get(creature_id)
        if record is None:
            record = _CreatureHistory(
                creature_id=creature_id,
                creature_name=str(creature_name),
                deceased=False,
                last_observed_time=float(simulation_time),
                next_bout_id=1,
                bouts=deque(),
                summary=_build_creature_summary(
                    creature_id,
                    (),
                    self.minimum_stable_bouts,
                ),
                species_id=species_id,
                observation_session_count=int(active),
                last_observation_mode=mode_value,
                active=active,
            )
            if observation_generation is not None:
                record.progress_by_generation[observation_generation] = (
                    0,
                    None,
                )
            self._creatures[creature_id] = record
        else:
            record.creature_name = str(creature_name)
            record.deceased = False
            record.last_observed_time = float(simulation_time)
            if species_id is not None:
                record.species_id = int(species_id)
            if active and (
                not record.active
                or observation_generation
                not in record.progress_by_generation
            ):
                record.observation_session_count += 1
            if mode_value is not None:
                record.last_observation_mode = mode_value
            record.active = active
            if (
                observation_generation is not None
                and observation_generation
                not in record.progress_by_generation
            ):
                record.progress_by_generation[observation_generation] = (
                    0,
                    None,
                )
            self._creatures.move_to_end(creature_id)
        while len(record.progress_by_generation) > 64:
            record.progress_by_generation.popitem(last=False)
        self._evict_creatures()

    def set_active_creatures(self, creature_ids: set[int]) -> None:
        for creature_id, record in self._creatures.items():
            record.active = creature_id in creature_ids
        self._evict_creatures()

    def record_observation_progress(
        self,
        creature_id: int,
        generation: int,
        simulation_time: float,
        observations_processed: int,
    ) -> None:
        """Accumulate processed temporal coverage without double counting."""
        record = self._creatures.get(creature_id)
        if record is None:
            return
        current_time = float(simulation_time)
        current_count = max(0, int(observations_processed))
        progress = record.progress_by_generation.get(generation)
        if progress is None:
            record.progress_by_generation[generation] = (
                current_count,
                current_time,
            )
            return
        previous_count, previous_time = progress
        if current_count <= previous_count:
            return
        if previous_time is not None:
            record.total_observation_seconds += max(
                0.0,
                current_time - previous_time,
            )
        record.progress_by_generation[generation] = (
            current_count,
            current_time,
        )
        record.progress_by_generation.move_to_end(generation)
        while len(record.progress_by_generation) > 64:
            record.progress_by_generation.popitem(last=False)
        record.last_observed_time = max(record.last_observed_time, current_time)

    def mark_deceased(self, creature_id: int, simulation_time: float) -> None:
        record = self._creatures.get(creature_id)
        if record is None:
            return
        record.deceased = True
        record.active = False
        record.last_observed_time = float(simulation_time)
        self._creatures.move_to_end(creature_id)
        self._evict_creatures()

    def append_draft(
        self,
        draft: CompletedBehaviorBoutDraft,
    ) -> CompletedBehaviorBout | None:
        source = (
            draft.creature_id,
            draft.selection_generation,
            draft.behavior,
            draft.local_bout_id,
        )
        if source in self._seen_sources:
            self._duplicate_completions_ignored += 1
            return None
        self._seen_sources[source] = None
        while len(self._seen_sources) > self._seen_source_capacity:
            self._seen_sources.popitem(last=False)
        record = self._creatures.get(draft.creature_id)
        if record is None:
            self.register_creature(
                draft.creature_id,
                f"Creature {draft.creature_id}",
                draft.end_time,
            )
            record = self._creatures[draft.creature_id]
        persistent_id = record.next_bout_id
        record.next_bout_id += 1
        why = (
            None
            if draft.why_summary is None
            else replace(draft.why_summary, bout_id=persistent_id)
        )
        completed = CompletedBehaviorBout(
            creature_id=draft.creature_id,
            behavior=draft.behavior,
            bout_id=persistent_id,
            start_time=draft.start_time,
            end_time=draft.end_time,
            duration=draft.duration,
            evidence_summary=draft.evidence_summary,
            outcome=draft.outcome,
            termination=draft.termination,
            why_summary=why,
        )
        if len(record.bouts) >= self.max_completed_bouts_per_creature:
            record.bouts.popleft()
            record.detailed_bouts_dropped += 1
        record.bouts.append(completed)
        self._bout_finalizations += 1
        self._why_summaries_finalized += int(why is not None)
        record.summary = _build_creature_summary(
            record.creature_id,
            tuple(record.bouts),
            self.minimum_stable_bouts,
        )
        record.last_observed_time = max(
            record.last_observed_time,
            completed.end_time,
        )
        self._creatures.move_to_end(draft.creature_id)
        self._evict_creatures()
        return completed

    def mark_incomplete(self, completions_not_recorded: int) -> None:
        count = max(0, int(completions_not_recorded))
        self._history_incomplete = (
            self._history_incomplete or count > 0
        )
        self._history_completions_not_recorded = max(
            self._history_completions_not_recorded,
            count,
        )

    def record_skipped_completions(self, count: int) -> None:
        increment = max(0, int(count))
        if increment <= 0:
            return
        self._history_incomplete = True
        self._history_completions_not_recorded += increment

    def _evict_creatures(self) -> None:
        while sum(not record.active for record in self._creatures.values()) > (
            self.max_remembered_creatures
        ):
            inactive = {
                creature_id: record
                for creature_id, record in self._creatures.items()
                if not record.active
            }
            if not inactive:
                break
            creature_id = min(
                inactive,
                key=lambda candidate_id: (
                    inactive[candidate_id].last_observed_time
                ),
            )
            self._creatures.pop(creature_id)
            self._seen_sources = OrderedDict(
                (source, None)
                for source in self._seen_sources
                if source[0] != creature_id
            )
            self._creatures_evicted += 1

    @property
    def index(self) -> tuple[CreatureHistoryIndexEntry, ...]:
        ordered_records = tuple(self._creatures.values())
        recent_first = (
            record
            for _position, record in sorted(
                enumerate(ordered_records),
                key=lambda item: (
                    item[1].last_observed_time,
                    item[0],
                ),
                reverse=True,
            )
        )
        return tuple(
            CreatureHistoryIndexEntry(
                creature_id=record.creature_id,
                creature_name=record.creature_name,
                deceased=record.deceased,
                last_observed_time=record.last_observed_time,
                completed_bout_count=len(record.bouts),
                species_id=record.species_id,
                total_observation_seconds=record.total_observation_seconds,
                observation_session_count=record.observation_session_count,
                last_observation_mode=record.last_observation_mode,
                active=record.active,
            )
            for record in recent_first
        )

    def report_for(self, creature_id: int) -> CreatureBehaviorReport | None:
        record = self._creatures.get(creature_id)
        if record is None:
            return None
        bouts = tuple(record.bouts)
        return CreatureBehaviorReport(
            creature_id=record.creature_id,
            creature_name=record.creature_name,
            deceased=record.deceased,
            completed_bouts=bouts,
            summary=record.summary,
            history_incomplete=self._history_incomplete,
            history_completions_not_recorded=(
                self._history_completions_not_recorded
            ),
            detailed_bouts_dropped=record.detailed_bouts_dropped,
            species_id=record.species_id,
            total_observation_seconds=record.total_observation_seconds,
            observation_session_count=record.observation_session_count,
            last_observation_mode=record.last_observation_mode,
        )

    def species_report(
        self,
        species_id: int | None,
    ) -> SpeciesBehaviorReport:
        records = tuple(
            record
            for record in self._creatures.values()
            if record.species_id == species_id
        )
        entries_by_id = {entry.creature_id: entry for entry in self.index}
        entries = tuple(
            entries_by_id[record.creature_id]
            for record in records
            if record.creature_id in entries_by_id
        )
        bouts = tuple(bout for record in records for bout in record.bouts)
        observation_seconds = sum(
            record.total_observation_seconds for record in records
        )
        from src.behavior_observer import BehaviorKind

        summaries: list[SpeciesBehaviorSummary] = []
        for behavior in BehaviorKind:
            matching = tuple(
                bout for bout in bouts if bout.behavior is behavior
            )
            if not matching:
                continue
            durations = sorted(bout.duration for bout in matching)
            summaries.append(
                SpeciesBehaviorSummary(
                    behavior=behavior,
                    completed_bout_count=len(matching),
                    total_duration=sum(durations),
                    median_duration=float(median(durations)),
                    bouts_per_creature_hour=(
                        0.0
                        if observation_seconds <= 0.0
                        else len(matching) * 3600.0 / observation_seconds
                    ),
                )
            )
        return SpeciesBehaviorReport(
            species_id=species_id,
            observed_creature_count=len(records),
            total_observation_seconds=observation_seconds,
            completed_bout_count=len(bouts),
            behaviors=tuple(summaries),
            creatures=entries,
        )

    def assign_missing_species(
        self,
        species_by_creature_id: dict[int, int],
    ) -> None:
        for creature_id, record in self._creatures.items():
            if (
                record.species_id is None
                and creature_id in species_by_creature_id
            ):
                record.species_id = int(species_by_creature_id[creature_id])

    @property
    def diagnostics(self) -> BehaviorHistoryDiagnostics:
        return BehaviorHistoryDiagnostics(
            bout_finalizations=self._bout_finalizations,
            why_summaries_finalized=self._why_summaries_finalized,
            completed_bouts_stored=sum(
                len(record.bouts) for record in self._creatures.values()
            ),
            detailed_bouts_dropped=sum(
                record.detailed_bouts_dropped
                for record in self._creatures.values()
            ),
            remembered_creatures=len(self._creatures),
            creatures_evicted=self._creatures_evicted,
            duplicate_completions_ignored=(
                self._duplicate_completions_ignored
            ),
            history_incomplete=self._history_incomplete,
            history_completions_not_recorded=(
                self._history_completions_not_recorded
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "history_incomplete": self._history_incomplete,
            "history_completions_not_recorded": (
                self._history_completions_not_recorded
            ),
            "creatures_evicted": self._creatures_evicted,
            "bout_finalizations": self._bout_finalizations,
            "why_summaries_finalized": self._why_summaries_finalized,
            "duplicate_completions_ignored": (
                self._duplicate_completions_ignored
            ),
            "creatures": [
                {
                    "creature_id": record.creature_id,
                    "creature_name": record.creature_name,
                    "deceased": record.deceased,
                    "last_observed_time": record.last_observed_time,
                    "next_bout_id": record.next_bout_id,
                    "detailed_bouts_dropped": record.detailed_bouts_dropped,
                    "species_id": record.species_id,
                    "total_observation_seconds": (
                        record.total_observation_seconds
                    ),
                    "observation_session_count": (
                        record.observation_session_count
                    ),
                    "last_observation_mode": record.last_observation_mode,
                    "bouts": tuple(record.bouts),
                }
                for record in self._creatures.values()
            ],
        }

    def restore_state(self, state: object) -> None:
        if not isinstance(state, dict):
            return
        restored: OrderedDict[int, _CreatureHistory] = OrderedDict()
        for raw in state.get("creatures", ()):
            if not isinstance(raw, dict):
                continue
            creature_id = int(raw["creature_id"])
            bouts = deque(
                (
                    bout
                    for bout in raw.get("bouts", ())
                    if isinstance(bout, CompletedBehaviorBout)
                ),
            )
            restored_drop_count = 0
            while len(bouts) > self.max_completed_bouts_per_creature:
                bouts.popleft()
                restored_drop_count += 1
            completed_bouts = tuple(bouts)
            restored[creature_id] = _CreatureHistory(
                creature_id=creature_id,
                creature_name=str(raw.get("creature_name", creature_id)),
                deceased=bool(raw.get("deceased", False)),
                last_observed_time=float(
                    raw.get("last_observed_time", 0.0)
                ),
                next_bout_id=max(
                    int(raw.get("next_bout_id", 1)),
                    max((bout.bout_id for bout in bouts), default=0) + 1,
                ),
                bouts=bouts,
                summary=_build_creature_summary(
                    creature_id,
                    completed_bouts,
                    self.minimum_stable_bouts,
                ),
                detailed_bouts_dropped=max(
                    0,
                    int(raw.get("detailed_bouts_dropped", 0)),
                )
                + restored_drop_count,
                species_id=(
                    None
                    if raw.get("species_id") is None
                    else int(raw["species_id"])
                ),
                total_observation_seconds=max(
                    0.0,
                    float(raw.get("total_observation_seconds", 0.0)),
                ),
                observation_session_count=max(
                    0,
                    int(raw.get("observation_session_count", 0)),
                ),
                last_observation_mode=raw.get("last_observation_mode"),
            )
        self._creatures = restored
        self._seen_sources.clear()
        self._history_incomplete = bool(
            state.get("history_incomplete", False)
        )
        self._history_completions_not_recorded = max(
            0,
            int(state.get("history_completions_not_recorded", 0)),
        )
        self._creatures_evicted = max(
            0,
            int(state.get("creatures_evicted", 0)),
        )
        retained_and_dropped = sum(
            len(record.bouts) + record.detailed_bouts_dropped
            for record in self._creatures.values()
        )
        self._bout_finalizations = max(
            retained_and_dropped,
            int(state.get("bout_finalizations", retained_and_dropped)),
        )
        retained_why = sum(
            bout.why_summary is not None
            for record in self._creatures.values()
            for bout in record.bouts
        )
        self._why_summaries_finalized = max(
            retained_why,
            int(state.get("why_summaries_finalized", retained_why)),
        )
        self._duplicate_completions_ignored = max(
            0,
            int(state.get("duplicate_completions_ignored", 0)),
        )
        self._evict_creatures()


def _build_creature_summary(
    creature_id: int,
    bouts: tuple[CompletedBehaviorBout, ...],
    stable_threshold: int,
) -> CreatureBehaviorSummary:
    from src.behavior_observer import BehaviorKind
    from src.counterfactual_neat import (
        EffectDirection,
        SemanticIntervention,
        influence_label,
    )

    behavior_summaries: list[BehaviorLifetimeSummary] = []
    for behavior in BehaviorKind:
        matching = tuple(bout for bout in bouts if bout.behavior is behavior)
        if not matching:
            continue
        durations = sorted(bout.duration for bout in matching)
        outcomes = Counter(
            bout.outcome for bout in matching if bout.outcome is not None
        )
        why_summaries: list[BehaviorLifetimeWhySummary] = []
        for intervention in SemanticIntervention:
            effects = [
                effect
                for bout in matching
                if bout.why_summary is not None
                for effect in bout.why_summary.effects
                if effect.intervention is intervention
            ]
            if not effects:
                continue
            influences = sorted(effect.median_influence for effect in effects)
            lifetime_median = float(median(influences))
            directions = Counter(effect.effect_direction for effect in effects)
            why_summaries.append(
                BehaviorLifetimeWhySummary(
                    intervention=intervention,
                    behavior_bout_count=len(matching),
                    contributing_bout_count=len(effects),
                    median_bout_influence=lifetime_median,
                    p25=_percentile(influences, 0.25),
                    p75=_percentile(influences, 0.75),
                    influence_label=influence_label(lifetime_median),
                    direction_counts=EffectDirectionCounts(
                        supportive=directions[EffectDirection.SUPPORTIVE],
                        suppressive=directions[EffectDirection.SUPPRESSIVE],
                        reversing=directions[EffectDirection.REVERSING],
                        mixed=directions[EffectDirection.MIXED],
                        minimal=directions[EffectDirection.MINIMAL],
                    ),
                    quantiles_estimated=any(
                        effect.quantiles_estimated for effect in effects
                    ),
                )
            )
        why_summaries.sort(
            key=lambda item: (
                -item.median_bout_influence,
                list(SemanticIntervention).index(item.intervention),
            )
        )
        behavior_summaries.append(
            BehaviorLifetimeSummary(
                behavior=behavior,
                completed_bout_count=len(matching),
                total_duration=sum(durations),
                median_duration=float(median(durations)),
                outcome_counts=tuple(
                    sorted(
                        outcomes.items(),
                        key=lambda item: item[0].value,
                    )
                ),
                why_summaries=tuple(why_summaries),
            )
        )
    return CreatureBehaviorSummary(
        creature_id=creature_id,
        completed_bout_count=len(bouts),
        behaviors=tuple(behavior_summaries),
        stable_pattern_threshold=stable_threshold,
    )


def _percentile(sorted_values: list[float], quantile: float) -> float | None:
    if len(sorted_values) < 2:
        return None
    position = (len(sorted_values) - 1) * quantile
    lower = floor(position)
    upper = min(len(sorted_values) - 1, lower + 1)
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )
