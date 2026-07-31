# Long-Term Behaviour History

The long-term report is a finalized layer above the existing Temporal
Behaviour Observer and counterfactual NEAT WHY system.

## Timescales

- `BehaviorSnapshot` and `WhySnapshot` remain live, changing views.
- `CompletedBehaviorBout` and `CompletedWhyExplanation` are immutable records
  created only after an active bout ends or focal observation is interrupted.
- `CreatureBehaviorReport` aggregates retained completed bouts. WHY influence
  is aggregated as one median value per bout, so long bouts do not receive
  extra weight.

Only selected creatures receive detailed observation. Completed records remain
available after deselection and death, subject to the configured per-creature
and remembered-creature bounds.

## Bounded collection

Operational evidence uses the same keys, values, and units as the live
observer. Up to `active_metric_sample_capacity` values are exact. Longer bouts
use deterministic, time-distributed compaction while preserving exact counts,
pass fractions, value totals for event evidence, and first/last values.
Median and IQR fields carry an explicit `quantiles_estimated` marker only after
compaction; at the default capacity, bouts through 512 samples are exact.

Completed records travel through a non-blocking FIFO worker outbox. Temporary
backpressure is lossless. The outbox has soft, hard, and recovery thresholds;
at the hard threshold the live observer continues, new long-term completions
are explicitly counted as unrecorded, and reports are permanently marked
incomplete.

The main-process store also bounds recent completion identities used for
duplicate protection. Prepared lifetime summaries are rebuilt only when a
completed bout is appended, removed by retention, or restored—not on every UI
frame. Diagnostics expose finalization and finalized-WHY counts, dropped
detail, MRU creature eviction, ignored duplicates, outbox warning/high-water
state, and skipped completions.

## Behaviour Report

The separate Behaviour Report is available from the Stats panel's
`View Behaviour History` action and the selected-creature Report action. It
keeps the live `NODE | BEHAVIOURS | WHY` inspector unchanged and provides:

- `TIMELINE`: overlapping behavior lanes scaled to the retained history, with
  clickable immutable bout details;
- `SUMMARY`: completed-bout counts, total/median duration, and outcomes;
- `WHY`: one lifetime value per completed-bout median, intervention-specific
  availability denominators, IQR, centralized influence labels, and
  bout-direction counts.

Stable wording such as “Typical” is used only after the configured number of
WHY-contributing completed bouts. An incomplete-history warning remains visible
after any hard-bound skip, even after the worker has recovered.

## Persistence

Checkpoint version 16 persists only:

- finalized completed bouts and compact WHY summaries;
- persistent per-creature bout counters;
- retained-creature metadata and capacity diagnostics;
- the incomplete-history marker and unrecorded-completion count.

Active observation windows, live snapshots, pending IPC/outbox records,
temporary metric samples, and raw counterfactual probes are never persisted.
Version 15 and older checkpoints restore with an empty, complete history.

## Benchmark

The observer benchmark includes a long-run finalized-history path and checks
the configured detail and duplicate-memory bounds:

```text
python benchmarks/benchmark_behavior_observer.py \
  --samples 5000 --history-bouts 10000
```
