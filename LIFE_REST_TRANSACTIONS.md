# Life, Rest, and Transactional Resources

## Runtime contract

`life` is the durable survival reserve. Energy may remain at zero; metabolic
death occurs only when `life <= 0`. Manual kill remains immediate. The current
brain contract is sensing schema 6 and action schema 2: input `-44` is
`life_normalized`, and output `14` is the continuous sigmoid `rest` action.

Rest is smoothed on every fixed step, including cached-brain steps:

```text
alpha = 1 - exp(-rest_response_rate * dt)
smoothed += (intent - smoothed) * alpha
```

Voluntary translation is multiplied by
`1 - smoothed_rest ** rest_movement_exponent`; neural turning is multiplied by
`1 - rest_rotation_inhibition * smoothed_rest`. Collision avoidance is not
inhibited. Post-physics planar speed receives the additional factor
`exp(-rest_braking_strength * smoothed_rest * dt)`.

Activity weights are motor effort 0.40, actual speed 0.10, blended turn 0.15,
actual communication cost 0.15, reproduction 0.10, and nursing 0.10. An
accepted reproduction or positive nursing transfer sets activity to 1.0.
`effective_rest = smoothed_rest * (1 - activity)`.

## Digestion and the ledger

Stomach contents are stored in raw edible-energy units. Digestive efficiency
converts that quantity to gross energy; the processing fraction is a
dimensionless share of gross energy:

```text
gross_energy = stomach_consumed * effective_efficiency
processing_cost = gross_energy * processing_fraction
net_energy = gross_energy - processing_cost
```

The pure digestion calculation limits stomach use to the current rate and to
the net energy that can cover same-step demand plus final energy headroom. A
small numerical tolerance prevents loss when headroom is effectively zero.

Each resource candidate is evaluated once from its starting snapshot:

```text
available = starting_energy + net_digestion
after_ordinary = max(0, available - ordinary_demand)
ordinary_deficit = max(0, ordinary_demand - available)
paid_powered_movement = min(powered_movement_demand, after_ordinary)
unpaid_powered_movement = powered_movement_demand - paid_powered_movement
remaining = after_ordinary - paid_powered_movement
final_energy = min(max_energy, remaining)
final_life = clamp(
    starting_life - ordinary_damage - movement_damage - direct_damage,
    0,
    max_life,
)
```

At depleted energy, locomotion remains available while communication, new
grabs, nursing, and reproduction are gated from the effective action. Raw brain
outputs remain unchanged. The ledger pays ordinary demand first and then
powered movement. Any powered-movement shortfall draws on life using:

```text
life_ratio = clamp(life_before_movement / max_life, 0, 1)
movement_multiplier = 1 + (max_multiplier - 1) * (1 - life_ratio)^2
movement_damage = unpaid_powered_movement * deficit_damage_rate * multiplier
```

The default maximum multiplier is 4.0. Passive coasting, collision displacement,
and mandatory avoidance remain ordinary linear demand rather than powered
movement.

Processing loss is never charged again as demand. Pending direct damage is
cleared with the single candidate commit.

## Transaction allocation

Nursing and reproduction preparation is side-effect free. The coordinator
first reserves reproduction capacity by rtNEAT eligibility rank and creature
ID, with at most one accepted birth per reproduction interval. If that bundle
fails, the next eligible request is promoted deterministically. Nursing targets
only existing infants and is resolved by descending target generation, target
ID, then donor ID. Each target's post-ledger, pre-transfer headroom is allocated
exactly; a donor whose bundled action candidate dies falls back to baseline,
pays nothing, and releases its allocation.

Every creature has a baseline candidate and, when globally selected, one exact
action candidate. Reproduction and nursing by the same creature form one
bundle. A failed bundle selects baseline and releases all reservations, which
promotes the next reproduction request deterministically.

Accepted resource candidates commit for every creature before nursing.
Transfers then commit in target/donor order with a clamp after every transfer.
Final offspring are staged against shadow RNG, allocator, genome, and species
state. Shadow staging copies mutable containers and allocator state while
reusing live genomes and brains that are read-only during the transaction; only
the selected parent genome is deep-copied for mutation. Only the final surviving
request advances live state. Dead creatures are removed once, and eating runs
afterward. Detailed ledger diagnostics are refreshed only for the selected
creature, while biological activity and resource state commit for every
creature on every fixed step.

## Persistence migration

Checkpoint version 18 persists `life` but no rest/action cache state. Creatures
from older checkpoints receive configured initial life. A version-17
schema-5/action-1 checkpoint with 43 inputs and 14 outputs migrates append-only:
existing genes, fitness, species, innovations, and allocator positions are
preserved; input key `-44` remains disconnected and output node `14` is added as
a disconnected sigmoid with zero bias. Other incompatible contracts continue
to require the explicit brain-reset load path.
