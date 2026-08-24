# Creature Modelling and Functioning in an Evolving Artificial-Life System

## Abstract

This project models a population of autonomous herbivorous creatures whose morphology, metabolism, perception, social tendencies, and neural controllers change through continuous asexual reproduction and mutation. A creature is not represented by a single state vector alone: the implementation separates inherited genotype, transient physiological state, rigid-body dynamics, recurrent neural decision state, ancestry, and observational records. Behaviour emerges from perception, recurrent NEAT activation, action execution, and thermodynamic resource accounting. There is no explicit parent ranking: a lineage propagates only when an individual survives long enough, accumulates sufficient post-upkeep energy, expresses reproductive intent, and funds its child directly.

This report describes the current computational model, its principal mathematical formulations, and the mechanisms used to analyse emergent behaviour. It includes the generated biome field, mixed food ecology, conserved biomass inventory, sensory transformations, physical and physiological transactions, communication fields, evolution, and passive scientific analysis.

_Primary implementation: [creature domain package](src/creature/) · [simulation integration](src/world.py) · [simulation parameters](configs/sim_config.py)_

## Table of contents

1. [Modelling scope and execution cycle](#1-modelling-scope-and-execution-cycle)
2. [World, biomes, and food ecology](#2-world-biomes-and-food-ecology)
3. [Creature state and genotype](#3-creature-state-and-genotype)
4. [Perception and sensor formulation](#4-perception-and-sensor-formulation)
5. [Neural decision model and actions](#5-neural-decision-model-and-actions)
6. [Locomotion and social dynamics](#6-locomotion-and-social-dynamics)
7. [Communication](#7-communication)
8. [Feeding, metabolism, rest, and vitality](#8-feeding-metabolism-rest-and-vitality)
9. [Reproduction, development, and evolution](#9-reproduction-development-and-evolution)
10. [Passive telemetry and speciation](#10-passive-telemetry-and-speciation)
11. [Behavioural analysis and explainability](#11-behavioural-analysis-and-explainability)
12. [Concluding interpretation](#12-concluding-interpretation)

## Notation

The operator

$$
\operatorname{clip}_{[a,b]}(x)=\min(b,\max(a,x))
$$

is used throughout. Bold lower-case symbols denote two-dimensional vectors; $\lVert\mathbf{x}\rVert$ is the Euclidean norm; $\Delta t$ is simulated elapsed time; and normalized intensities lie in $[0,1]$ unless stated otherwise. Configuration values quoted below are the current defaults rather than universal constants of the model.

_Implementation: [common numerical helpers](src/creature/common.py) · [configuration](configs/sim_config.py)_

---

## 1. Modelling scope and execution cycle

### 1.1 Artificial-life formulation

Each creature is an embodied, autonomous agent. Its body occupies continuous two-dimensional space; its sensors transform local physical and biological conditions into a fixed neural input vector; its recurrent neural genome combines that vector with its private activation history to produce action intensities; and the resulting motion and biological transactions alter both the creature and its surroundings. Reproduction is asexual: any physiologically eligible creature may fund a child from its own post-upkeep energy, after which inherited traits and the neural genome mutate. The population therefore implements a continuous evolutionary process rather than a sequence of isolated generations.

_Implementation: [creature entity](src/creature/model.py) · [evolution coordinator](src/creature/evolution.py) · [real-time NEAT manager](src/creature/neat/rt_neat.py)_

The model distinguishes four kinds of state:

1. **Hereditary state:** vision, morphology, digestive traits, flocking genes, social tags, colour, and the NEAT genome.
2. **Physical and physiological state:** position, velocity, heading, energy, life, stomach contents, age, and carried food.
3. **Controller state:** the instantiated neural network, smoothed action values, internal chronometers, and cached social intention.
4. **Historical state:** lineage, passive lifetime measurements, completed behavioural bouts, species records, and archived genomes.

This separation prevents temporary conditions—such as infant movement penalties or senescence—from accidentally changing inherited traits.

_Implementation: [genotype](src/creature/genotype.py) · [live state](src/creature/model.py) · [runtime services](src/creature/runtime/) · [historical telemetry](src/creature/fitness.py)_

### 1.2 Deterministic multi-rate cycle

The simulation uses a fixed physical frequency of $60\,\mathrm{Hz}$. Neural decisions and biological updates occur every three physical steps, hence at $20\,\mathrm{Hz}$, while aggregate statistics are refreshed every twelve steps, hence at $5\,\mathrm{Hz}$. A fixed-step accumulator limits work per external update without changing the duration of an accepted simulation step. This is important biologically: forces, bite exposures, pheromone evolution, ageing, digestion, reproduction, and energy expenditure are calculated from simulated time rather than from the rate at which external frames arrive.

| Process | Default frequency | Default period |
|---|---:|---:|
| Rigid-body integration | $60\,\mathrm{Hz}$ | $1/60\,\mathrm{s}$ |
| Neural decisions | $20\,\mathrm{Hz}$ | $3$ physical steps |
| Biological/resource update | $20\,\mathrm{Hz}$ | $3$ physical steps |
| Population statistics | $5\,\mathrm{Hz}$ | $12$ physical steps |

_Implementation: [scheduler configuration](configs/sim_config.py) · [fixed-step execution](src/world.py) · [scheduler validation](tests/test_scheduler_validation.py)_

At a conceptual level, one cycle is

$$
\text{sense}\rightarrow\text{decide}\rightarrow\text{apply intent}
\rightarrow\text{integrate physics}\rightarrow\text{resolve biology}.
$$

Perception and decisions are staggered deterministically across creature identities, but cached actions remain active between decisions. Mouth contacts are accumulated at the physical rate and resolved chronologically before the next biological ledger. Consequently, short-lived contacts are not lost merely because biology runs at a lower frequency.

_Implementation: [decision and action phases](src/world.py) · [perception service](src/creature/runtime/perception.py) · [mouth-exposure transaction](src/world.py)_

---

## 2. World, biomes, and food ecology

### 2.1 Fractal biome generation

The environment is a rectangular continuous world backed by a $64\times44$ biome grid. A seeded OpenSimplex field is evaluated over that grid at three octaves. If $N$ denotes two-dimensional simplex noise, $s$ is the spatial noise scale, $p$ is persistence, and $\ell$ is lacunarity, the sampled field is

$$
F(x,y)=\frac{1}{A}\sum_{o=0}^{O-1}p^o
N\!\left(\ell^o\frac{x}{s},\ell^o\frac{y}{s}\right),
\qquad
A=\sum_{o=0}^{O-1}p^o.
$$

The defaults are $O=3$, $s=800$, $p=0.5$, and $\ell=2$. The sampled field is min--max normalized over the generated map. Let the requested prairie, bushes, and forest shares be $a_P$, $a_B$, and $a_F$. Their cumulative classification quantiles are

$$
q_P=\frac{a_P}{a_P+a_B+a_F},\qquad
q_B=\frac{a_P+a_B}{a_P+a_B+a_F}.
$$

Values below the $q_P$ quantile become prairie, values between the two thresholds become bushes, and values at or above the $q_B$ quantile become forest. The default target shares are $0.35$, $0.40$, and $0.25$, respectively. Explicit prairie and bush thresholds from the food-patch configuration can replace the quantiles; startup rejects thresholds that are unordered. Because classification is grid-based, realized area shares are measured from cell counts and can differ slightly from the targets.

_Implementation: [biome generation and classification](src/biome.py) · [biome defaults](configs/sim_config.py) · [generation tests](tests/test_biome.py)_

### 2.2 Biome richness field

Each biome has an ordinary-food spawn weight $w_b$: forest $2.75$, bushes $1.25$, and prairie $0.25$ by default. With uniform-spawn probability $p_u=0.10$ and $w_{\max}=\max_b w_b$, the expected ordinary-food density assigned to a biome cell is

$$
d_b=p_u+(1-p_u)\frac{w_b}{w_{\max}}.
$$

Thus the default cell values are $1$ in forest, approximately $0.509$ in bushes, and approximately $0.182$ in prairie. The positive uniform component deliberately prevents prairie from becoming an absolute zero-food region. If every biome weight is zero, the expected-density grid falls back to one everywhere because actual placement also falls back to uniform sampling.

At an arbitrary world position, density is interpolated from the four surrounding cell-centre values. For fractional cell coordinates $(u,v)$,

$$
d(x,y)=(1-u)(1-v)d_{00}+u(1-v)d_{10}
 +(1-u)vd_{01}+uvd_{11}.
$$

This continuous field is both an analytical prediction of the ordinary spawn process and the quantity exposed to biome sensors. It is not a count of currently present food. Biomes otherwise have no direct effect on locomotion, life, or metabolic cost; they influence creatures indirectly through resource placement and sensed resource expectation.

_Implementation: [expected-density cache and interpolation](src/biome.py) · [richness sensing tests](tests/test_biome_sensing.py)_

### 2.3 Independent food placement

Independent pellets use a mixture sampler. With probability $p_u$, a position is drawn uniformly from the world interior. Otherwise, a uniformly proposed position in biome $b$ is accepted with probability

$$
P(\text{accept}\mid b)=\frac{w_b}{w_{\max}}.
$$

The pellet radius is respected when defining the admissible world interior. Weighted sampling makes at most 32 proposals by default and then falls back to a uniform position, so configuration or an unfortunate random sequence cannot block food creation. At bootstrap, if at least two independent items are requested, bounded cell enumeration first guarantees one ordinary pellet in bushes and one in forest when those biomes exist; remaining pellets follow the mixture sampler.

_Implementation: [independent placement](src/food_spawner.py) · [biome placement tests](tests/test_food_spawner.py)_

### 2.4 Food energy and physical representation

An ordinary food pellet with radius $r$ and energy density $\rho$ contains

$$
E_f=\pi r^2\rho,
$$

where $r\in[6,10]$ and $\rho=0.002$ by default. After partial consumption, its radius follows the remaining energy,

$$
r=\sqrt{\frac{E_f}{\pi\rho}},
$$

and its physical mass is recomputed as

$$
m_f=0.9(0.2+0.035r).
$$

Ordinary partial consumption removes the committed energy and then updates energy, radius, mass, moment, and collision radius together. The runtime's $10\%$ micro-food tolerance belongs to bite formation rather than resizing: a whole pellet may be swallowed when it fits in the remaining stomach and is no more than $10\%$ larger than the interval's ordinary bite limit. Every pellet also has a bite capacity: at most that many chronological claims may consume it during one physical step. Independent pellets default to capacity one; biome patches may generate higher capacities.

_Implementation: [food energy and resizing](src/food.py) · [food energy tests](tests/test_food_spawner.py)_

### 2.5 Conserved biomass budget

Food regeneration is limited by an explicit world energy inventory. Let $E_i$ and $S_i$ be usable and stomach energy for live creature $i$, and let $E_{f,j}$ be the remaining energy of food $j$. Available biomass is

$$
B_{\mathrm{available}}=\max\left(0,
B_{\mathrm{total}}-\sum_i(E_i+S_i)-\sum_jE_{f,j}\right).
$$

Unless `total_biomass_energy` is configured explicitly, $B_{\mathrm{total}}$ is initialized from all creature usable/stomach energy plus all plant energy immediately after world bootstrap. Energy lost to upkeep, digestion inefficiency, or birth conversion therefore re-enters the unallocated biomass pool; it can become food again only through the bounded spawn mechanisms below. Life reserve is not included in this biomass ledger.

For independent spawning, the budget is converted into slots using the energy of a pellet at the midpoint radius,

$$
\bar E_f=\pi\left(\frac{r_{\min}+r_{\max}}{2}\right)^2\rho,
\qquad
N_B=\left\lfloor\frac{B_{\mathrm{available}}}{\bar E_f}\right\rfloor.
$$

Clustered spawning instead constructs a complete candidate patch and checks its exact total energy before committing it. Both modes are additionally bounded by the global food-item capacity.

_Implementation: [world biomass accounting](src/world.py) · [spawn budget](src/food_spawner.py) · [cluster budget](src/food_clustering.py)_

### 2.6 Mixed independent and clustered food

The food capacity is divided between independent pellets and biome-specific patches. For maximum item count $N_{\max}$ and cluster share $s_c$,

$$
N_{\mathrm{ind}}=\operatorname{round}(N_{\max}(1-s_c)),
\qquad
N_{\mathrm{cluster}}=N_{\max}-N_{\mathrm{ind}}.
$$

The default initial and maximum counts are both $363$, with $s_c=0.30$. Patch frequency is independent of biome area and ordinary-food weights. The manager chooses a feasible number of bush and forest patches whose patch-count ratio is nearest the configured $3{:}1$ weight while exactly accommodating the clustered item target within each profile's pellet-count bounds.

| Biome | Patch-frequency weight | Pellets per patch | Spread radius | Energy multiplier | Bite capacity |
|---|---:|---:|---:|---:|---:|
| Prairie | $0$ | $1$ | $0$ | $1$ | $1$ |
| Bushes | $3$ | $3$--$6$ | $20$--$40$ | $1$ | $1$--$2$ |
| Forest | $1$ | $12$--$25$ | $50$--$100$ | $2$--$5$ | $2$--$6$ |

Prairie patches are disabled by their zero frequency weight. For each active patch, a centre is sampled inside its assigned biome. Pellet offsets follow a two-dimensional Gaussian with standard deviation one third of the patch radius and are rejected outside that radius, the world, or the selected biome. Repeated placement failure shrinks the effective patch radius by $0.80$ down to $25\%$ of its configured radius before a new centre is tried.

A patch energy multiplier $m_e$ is implemented by scaling the sampled base radius by $\sqrt{m_e}$, which multiplies area and energy by $m_e$. When a patch falls to at most $\lfloor0.10N_p\rfloor$ active pellets, its residual pellets are removed, it enters a random $600$--$1200$ tick cooldown, and it is relocated within the same biome before regrowth. Existing healthy patches are not destroyed merely because live capacity settings change; depleted inventory converges to the new target.

_Implementation: [food-patch lifecycle](src/food_clustering.py) · [patch profiles](configs/sim_config.py) · [cluster tests](tests/test_food_clustering.py)_

### 2.7 Regrowth pressure and shortage recovery

Let $x=\operatorname{clip}_{[0,1]}(N/N_{\mathrm{cap}})$ be the relevant food ratio. The independent spawner uses independent-pellet count and its mode target; the patch manager uses total food count and global capacity. Regular regrowth uses

$$
P_f=\max\left(4x(1-x),\;0.05\,\mathbb{1}_{N=0}\right).
$$

The logistic term is zero at empty and full capacity and maximal at half capacity; the explicit $0.05$ seed prevents an empty world from being an absorbing state. Active species count $S_a$ scales the maximum rate through

$$
M_s=0.5+\frac{1.5}{1+\exp[-0.6(S_a-4)]},
\qquad
R=R_{\max}M_sP_f,
$$

where $R_{\max}=10$ biomass spawns per second. Independent and clustered systems accumulate fractional spawn credit at shares $1-s_c$ and $s_c$ of this rate. Integer credit can create items only when mode capacity, global capacity, placement, and biomass checks all succeed.

Critical shortage recovery is separate from regular logistic growth. With low-food threshold $\theta=0.50$,

$$
q=\operatorname{clip}_{[0,1]}
\left(\frac{\theta-x}{\max(0.001,\theta)}\right)
$$

measures shortage severity. At or below the critical ratio $x_c=0.15$, the independent spawner immediately queues enough pellets to approach $\lceil x_cN_{\mathrm{cap}}\rceil$, subject to the configured $215$-item burst budget and a per-update release cap of one quarter of that budget. If a critical shortage persists, additional burst credit accrues at $q/0.75$ events per second. The cluster manager similarly grants a one-shot emergency credit for its missing share. Emergency creation remains constrained by biomass and capacity; it cannot mint energy.

_Implementation: [regular and emergency regrowth](src/food_spawner.py) · [cluster emergency credit](src/food_clustering.py) · [regrowth tests](tests/test_food_spawner.py)_

---

## 3. Creature state and genotype

### 3.1 Embodied state

A live creature owns a unit-mass Pymunk rigid body and a circular collision shape. The body's moment of inertia is that of a solid circle with inherited radius $r$. Its authoritative position, heading, linear velocity, and angular velocity are held by the physics body. The convenience quantities used by other subsystems are

$$
\mathbf{p}=(x,y),\qquad \theta=\text{body angle},\qquad
v=\lVert\mathbf{v}\rVert.
$$

The collision shape has elasticity $0.15$ and zero friction; controlled anisotropic drag is applied explicitly instead of relying on surface friction. A newly created founder receives a random heading and an energy reserve sampled uniformly from $[0.55,0.95]$.

_Implementation: [creature model](src/creature/model.py) · [creature factory](src/creature/factory.py) · [collision categories](src/collision.py)_

The principal live physiological variables are usable energy $E$, life reserve $L$, stomach energy $S$, weighted stomach difficulty load $D_S$, lifetime gathered energy, resting intent, recent activity, and any direct damage awaiting the next resource transaction. Energy and life are distinct: exhaustion first creates unmet energetic demand, which is then converted into damage; a creature dies when life reaches zero, not merely when usable energy is temporarily empty.

_Implementation: [live physiological fields and ledger diagnostics](src/creature/model.py) · [resource ledger](src/creature/metabolism.py)_

### 3.2 Aggregate genotype

The non-neural genotype is the tuple

$$
G=(G_V,G_P,G_F,C),
$$

where $G_V$ contains vision traits, $G_P$ contains physical and digestive traits, $G_F$ contains social traits, and $C$ is inherited colour. The neural genome is managed separately because it follows NEAT's graph mutation and innovation bookkeeping. Both parts are coordinated atomically when an offspring is planned.

_Implementation: [genotype records](src/creature/genotype.py) · [neural brain](src/creature/neat/brain.py) · [evolution transaction](src/creature/evolution.py)_

| Trait | Meaning | Initial/default domain |
|---|---|---:|
| Vision range $R$ | Maximum visual reach | $[100,200]$ |
| Vision angle $\Phi$ | Total field of view | $[0.35,\pi]$ rad |
| Radius $r$ | Body and mouth scale | $[12,22]$ |
| Movement multiplier $m$ | Inherited locomotion cost factor | $[0.75,1.35]$ |
| Stomach capacity $K$ | Maximum stored food energy | $[0.8,2.6]$ |
| Digestion rate $q$ | Maximum stomach processing per second | $[0.05,0.40]$ |
| Digestion efficiency $\eta$ | Gross conversion fraction | $[0.55,0.98]$ |
| Separation, alignment, cohesion | Social steering genes | each in $[0,1]$ |
| Social tag $(t_x,t_y)$ | Heritable compatibility coordinates | each in $[0,1]$ |

Founder physical and flocking values are Gaussian perturbations around configured defaults, clipped to their admissible intervals; founder vision is uniformly sampled over its allowed ranges. All flocking genes and tags are normalized when constructed, so downstream social calculations can assume a valid unit interval.

_Implementation: [genotype initialization and bounds](src/creature/genotype.py) · [trait and vision defaults](configs/sim_config.py)_

### 3.3 Lineage and diagnostics

Every creature has a stable integer identity and a `LineageInfo` record containing its parent identity, generation, species identity, and the effective bounded change of every mutable non-neural trait. The recorded mutation is the difference after clipping, not the unbounded random proposal; it therefore describes the phenotype actually inherited by the child.

_Implementation: [lineage and mutation deltas](src/creature/genotype.py) · [offspring planning](src/creature/evolution.py)_

Resource diagnostics retain the latest complete transaction: digestive conversion, the explicitly zero rest-generation term, paid healing, total demand, movement-powered demand, deficits, direct damage, and final energy and life. These records do not determine biology; they expose the already resolved ledger for scientific inspection.

_Implementation: [diagnostic records](src/creature/model.py) · [ledger commit](src/creature/metabolism.py)_

---

## 4. Perception and sensor formulation

### 4.1 Sensor contract

The current sensing schema is version 7 and contains exactly 43 ordered inputs. This ordering is a formal interface: it must match the `num_inputs = 43` declaration used to construct every NEAT network. A schema change therefore represents a change in the meaning of evolved genomes, not a cosmetic renaming.

_Implementation: [sensor contract and serialization](src/creature/vision.py) · [NEAT input declaration](configs/neat_herbivore.ini) · [controller contract validation](src/creature/neat/controller.py)_

| Indices | Functional group | Ordered signals |
|---:|---|---|
| 1–4 | Endogenous resources | constant, feeding drive, reproductive readiness, energy percentage |
| 5–7 | Motion and local counts | speed, creature count, food count |
| 8–10 | Internal time | alternating clock, resettable chronometer, normalized age |
| 11–16 | Nearest visible targets | food proximity/angle, creature proximity/angle, wall proximity/angle |
| 17–20 | Manipulation and resource field | carrying state, local richness, lateral richness gradient, forward richness gradient |
| 21–22 | Offspring | own-infant proximity and angle |
| 23–31 | Compatible group | presence, effective count, centre in body coordinates, relative velocity, long-range intensity and direction |
| 32 | Digestion | stomach fullness |
| 33–36 | Acoustic communication | strength, direction sine, direction cosine, tone |
| 37–42 | Chemical communication | trail and alarm concentrations at here, forward-left, and forward-right probes |
| 43 | Vitality | normalized remaining life |

_Implementation: [input names and writer](src/creature/vision.py) · [sensor construction](src/world.py)_

### 4.2 Endogenous normalization

Let normalized energy and stomach fullness be

$$
e=\operatorname{clip}_{[0,1]}(E/E_{\max}),\qquad
s=\operatorname{clip}_{[0,1]}(S/K).
$$

The feeding drive is not simple hunger. It is suppressed when the stomach is full:

$$
h=(1-e)(1-s).
$$

Thus a low-energy creature with no stomach space cannot obtain a strong feeding-drive input until digestion creates capacity. Speed is normalized by the world's maximum speed; visible creature and food counts are normalized as $\min(n_c/5,1)$ and $\min(n_f/10,1)$. Stomach fullness and life reserve are independently exposed as $s$ and $\operatorname{clip}_{[0,1]}(L/L_{\max})$, so the network can distinguish stored nutrients, immediately usable energy, and remaining vitality.

_Implementation: [sensor input formulation](src/creature/vision.py) · [stomach fullness](src/creature/vision.py) · [flocking defaults](configs/sim_config.py)_

The input named reproductive readiness is specifically a maturity ramp,

$$
r_m=\min\left(\frac{t_{\mathrm{age}}}{t_{\mathrm{maturity}}},1\right),
$$

not a promise that reproduction will succeed. Post-upkeep energy, survival, cooldown, population capacity, and neural intent remain authoritative external gates. Internal time consists of a signal alternating between zero and one every integer second, a resettable chronometer clipped after $20$ seconds, and age clipped after $120$ seconds. Together with the recurrent activation state, these inputs allow periodic and interval-dependent policies without exposing global simulation time.

_Implementation: [sensor snapshot assembly](src/world.py) · [reproductive eligibility](src/creature/neat/rt_neat.py) · [chronometer update](src/world.py)_

### 4.3 Visual geometry

The visual field is a circular sector parameterized by inherited range $R$ and angle $\Phi$. Candidates are compared by surface distance and relative bearing, so the physical sizes of observer and target influence visibility. Relative angles are normalized to the signed interval $[-1,1]$: zero denotes the forward axis, negative and positive values denote opposite sides of the field, and the magnitudes approach one near the field boundaries. Proximity increases from zero for absent or distant targets toward one as the target approaches.

_Implementation: [visibility candidates and angle normalization](src/creature/vision.py) · [vision traits](src/creature/genotype.py)_

Visible targets are sorted deterministically and nearer creatures can occlude angular intervals occupied by more distant food or creatures. Wall sensing uses ray–segment intersections against environmental boundaries. Mouth contact is handled separately from ordinary visibility so a food item physically touching the mouth can be processed even in close geometric edge cases.

_Implementation: [occlusion and blocked intervals](src/creature/vision.py) · [wall rays](src/creature/vision.py) · [mouth-contact candidate](src/creature/vision.py)_

Larger sensory fields are metabolically costly. With maximum configured range $R_{\max}$ and angle $\Phi_{\max}$, the per-second vision cost is

$$
C_{\mathrm{vision}}
=C_0+C_A\left(\frac{\Phi}{\Phi_{\max}}\right)
\left(\frac{R}{R_{\max}}\right)^2,
$$

where $C_0=0.001$ and $C_A=0.005$ by default. The quadratic range term approximates the growth of sensed sector area and creates an evolutionary trade-off between information and upkeep.

_Implementation: [vision energy model](src/creature/vision.py) · [vision configuration](configs/sim_config.py)_

### 4.4 Biome probes and resource gradients

Biome and pheromone sensing share three body-relative probe positions. For creature position $\mathbf{p}$ and heading $\theta$, define forward and left unit vectors

$$
\mathbf{h}=(\cos\theta,\sin\theta),\qquad
\mathbf{l}=(-\sin\theta,\cos\theta).
$$

With forward distance $f=96$ and lateral offset $s_b=48$, the probes are

$$
\mathbf{p}_0=\mathbf{p},\qquad
\mathbf{p}_L=\mathbf{p}+f\mathbf{h}+s_b\mathbf{l},\qquad
\mathbf{p}_R=\mathbf{p}+f\mathbf{h}-s_b\mathbf{l}.
$$

Let $d_0$, $d_L$, and $d_R$ be the interpolated expected food densities at those locations. The three neural inputs are

$$
\begin{aligned}
r_{\mathrm{local}}&=d_0,\\
g_{\mathrm{lat}}&=\operatorname{clip}_{[-1,1]}
\left(\frac{d_L-d_R}{d_0+\epsilon}\right),\\
g_{\mathrm{fwd}}&=\operatorname{clip}_{[-1,1]}
\left(\frac{(d_L+d_R)/2-d_0}{d_0+\epsilon}\right),
\end{aligned}
\qquad \epsilon=0.001.
$$

A positive lateral gradient means the expected resource field is richer on the left; a positive forward gradient means the average of the two forward probes is richer than the current position. Division by local richness makes the gradient approximately scale-invariant, while $\epsilon$ keeps zero-richness boundaries finite. These signals predict ordinary-food spawn propensity. They do not report live food, clustered-patch occupancy, or food hidden behind another body; those facts enter only through visual and contact mechanisms.

_Implementation: [probe geometry and richness sampling](src/world.py) · [gradient formulation](src/creature/vision.py) · [biome sensing tests](tests/test_biome_sensing.py)_

### 4.5 Family and compatible-group sensing

Family sensing searches the lineage index only for the focal creature's own children whose age is below the maturity threshold. The nearest visible infant contributes the same surface-proximity and signed-angle representation used by other visual targets. Unrelated infants and mature offspring do not enter these two inputs, which allows nursing policies without providing a general kinship oracle.

Social sensing is geometrically separate from ordinary visual occlusion. Within the configured $150$-unit social radius, each neighbour contributes with compatibility $c_j\in[0,1]$. The effective flock count is

$$
n_{\mathrm{eff}}=\sum_j c_j,
$$

and is exposed as $\operatorname{clip}_{[0,1]}(n_{\mathrm{eff}}/4)$. Compatible displacements and velocities are weighted by $c_j$, averaged by $n_{\mathrm{eff}}$, rotated into the focal creature's forward/right frame, normalized by social radius and twice maximum speed, and clipped to $[-1,1]$. Presence is $\operatorname{clip}_{[0,1]}(n_{\mathrm{eff}})$.

The default compatibility mode compares inherited two-dimensional social tags $\mathbf{t}_i$ using a Gaussian kernel,

$$
c_{ij}=\exp\left(-\frac{\lVert\mathbf{t}_i-\mathbf{t}_j\rVert^2}
{2\sigma_t^2}\right),\qquad \sigma_t=0.35.
$$

Species mode instead returns one only for matching known species; legacy mode derives a graded value from composite evolutionary distance. Optional long-range sensing, disabled by default, weights compatible neighbours by

$$
w_j=c_j\operatorname{clip}_{[0,1]}
\left(1-\frac{d_j}{R_L}\right)s_L,
$$

then exposes clipped total intensity and the normalized resultant direction in body coordinates. No neighbour identities are supplied to the brain.

_Implementation: [family and flock snapshots](src/creature/vision.py) · [family index](src/world.py) · [compatibility resolver](src/creature/flocking.py)_

### 4.6 Acoustic and chemical sensing

Acoustic sensing selects the strongest audible non-self emission after distance attenuation, with emitter identity used only as a deterministic tie-breaker. The neural inputs are received strength, tone, and the sine/cosine of direction in the listener's body frame; emitter identity and semantic meaning are not exposed.

Chemical sensing bilinearly samples trail and alarm fields at $\mathbf{p}_0$, $\mathbf{p}_L$, and $\mathbf{p}_R$. All six concentrations are passed directly to the network rather than being collapsed into engineered gradients. The recurrent controller may therefore learn a difference such as left minus right, compare alarm with trail, or ignore either channel. Pheromone values identify neither the emitter nor the age of a deposit, and the identical probe geometry does not imply that biome richness and pheromone concentration share a physical process.

_Implementation: [acoustic and pheromone observations](src/creature/communication.py) · [shared probe positions](src/world.py)_

---

## 5. Neural decision model and actions

### 5.1 Recurrent NEAT controller

Each creature owns a NEAT genome and a discrete recurrent network instantiated from it. Founder genomes begin without hidden nodes and with a sparse random sample—currently $15\%$—of possible direct input-to-output connections. Recurrence is enabled at the genome level, so evolution may later add or delete nodes and connections, including cycles and self-loops, change connection weights, toggle connections, and occasionally change activation or aggregation functions. Every creature retains its own two-buffer activation state across decision ticks; newly created brains start at zero. Activation mutation is restricted to bounded `sigmoid`, `tanh`, and `clamped` functions so evolved feedback cannot select unbounded activations.

_Implementation: [brain representation](src/creature/neat/brain.py) · [brain controller](src/creature/neat/controller.py) · [NEAT genome parameters](configs/neat_herbivore.ini)_

The network computes

$$
\mathbf{z}=N_{G_N}(\mathbf{x}),
$$

where $\mathbf{x}\in\mathbb{R}^{43}$ is the ordered sensor vector and $G_N$ is the neural genome. Output centering is activation-aware. For example, a sigmoid result is transformed by $y=2\operatorname{clip}_{[0,1]}(z)-1$, while $tanh$ and clamped outputs are clipped directly to $[-1,1]$; unsupported finite activation outputs pass through $tanh$. Invalid or missing outputs become zero. This centering makes a neutral sigmoid value of $0.5$ correspond to zero action evidence.

_Implementation: [network evaluation and normalization](src/creature/neat/brain.py) · [initial activation choices](configs/neat_herbivore.ini)_

If a brain is missing or a decision cannot be obtained, the controller returns a neutral action with every intent disabled. Sensor and output counts are validated when the controller is built, preventing a network with an incompatible contract from silently operating.

_Implementation: [controller fallback](src/creature/neat/controller.py) · [neutral action](src/creature/action.py)_

### 5.2 Action contract

The action schema contains exactly 15 ordered outputs:

| Index | Action | Range | Interpretation |
|---:|---|---:|---|
| 1 | `accelerate` | $[-1,1]$ | forward/backward thrust |
| 2 | `rotate` | $[-1,1]$ | signed turn command |
| 3 | `want_reproduce` | $[0,1]$ | reproductive intent |
| 4 | `want_eat` | $[0,1]$ | ingestion intent |
| 5 | `reset_chronometer` | $[0,1]$ | reset internal interval timer |
| 6 | `want_grab` | $[0,1]$ | pick up nearby food |
| 7 | `want_release` | $[0,1]$ | release carried food |
| 8 | `want_nurse` | $[0,1]$ | transfer energy to own infant |
| 9 | `flee_panic_intensity` | $[0,1]$ | sprint/panic intensity |
| 10 | `herding` | $[0,1]$ | social engagement drive |
| 11 | `emit_sound` | $[0,1]$ | acoustic emission strength |
| 12 | `sound_tone` | $[-1,1]$ | signed acoustic tone |
| 13 | `emit_trail_pheromone` | $[0,1]$ | trail deposition intensity |
| 14 | `emit_alarm_pheromone` | $[0,1]$ | alarm deposition intensity |
| 15 | `rest` | $[0,1]$ | continuous resting intent |

Separation, alignment, and cohesion are not actions: they are inherited traits that modulate social steering. Most positive discrete intents become active only when their value exceeds $0.1$. Reproduction deliberately uses the stricter centered threshold $0.2$, equivalent to raw sigmoid output above $0.6$, while eating, carrying, nursing, and chronometer resets retain the shared $0.1$ action threshold.

_Implementation: [action ordering, ranges, and threshold](src/creature/action.py) · [NEAT output declaration](configs/neat_herbivore.ini)_

Herding has additional elapsed-time smoothing. If the reference update coefficient is $\alpha$ at $30\,\mathrm{Hz}$, its continuous-equivalent response rate is

$$
\lambda=-\frac{\ln(1-\alpha)}{1/30},\qquad
\alpha_{\Delta t}=1-e^{-\lambda\Delta t},
$$

and the state is updated by linear interpolation with $\alpha_{\Delta t}$. Motion commands and rest are also smoothed before physical execution, reducing discontinuities while preserving approximately the same response under different fixed cadences.

_Implementation: [herding filter](src/creature/neat/brain.py) · [action and rest smoothing](src/world.py)_

---

## 6. Locomotion and social dynamics

### 6.1 Propulsion and turning

For heading $\theta$ and signed acceleration $a$, the direct neural force is

$$
\mathbf{F}_N=
\begin{cases}
aF_{+}(\cos\theta,\sin\theta), & a\ge 0,\\
aF_{-}(\cos\theta,\sin\theta), & a<0,
\end{cases}
$$

with default forward and backward maxima $F_+=125$ and $F_-=70$. Panic $p$ increases force and speed limits through the multiplier $1+0.5p$. The command is smoothed at the physical rate before application.

_Implementation: [force-vector formulation](src/creature/action.py) · [action application](src/world.py) · [motion parameters](configs/sim_config.py)_

Turning is controlled as a target angular velocity rather than an accumulated torque. For normalized turn command $u$, the target is $\omega^*=u\omega_{\max}$ and the current velocity approaches it by

$$
\omega' = \omega + \rho(\omega^*-\omega),
$$

where the response coefficient $\rho$ is adjusted for elapsed physical time. A dead zone removes very small commands, a different damping response is used when the target is zero, and both linear and angular velocities are clamped to their current limits.

_Implementation: [turn control and motion limits](src/world.py) · [turn parameters](configs/sim_config.py)_

### 6.2 Planar drag and rest inhibition

Velocity is decomposed into forward and lateral body axes. At each step the components are multiplied by cadence-adjusted retentions, currently $0.992$ and $0.72$ per reference physical step. This strong lateral damping produces top-down locomotion with directional inertia. Rest adds exponential braking

$$
b_{\mathrm{rest}}=\exp(-k_b\,r_s\,\Delta t),
$$

where $r_s$ is smoothed rest and $k_b=2.5$. Voluntary force is additionally scaled by $1-r_s^\gamma$ with default exponent $\gamma=2$, and neural turning is reduced by $1-0.5r_s$.

_Implementation: [planar drag and rest braking](src/world.py) · [rest parameters](configs/sim_config.py)_

### 6.3 Collision avoidance and force priority

Collision avoidance is mandatory and receives first access to a finite force budget. Any lower-priority force component that opposes accepted avoidance is projected out. If $\mathbf{a}$ is the avoidance vector and $\mathbf{v}\cdot\mathbf{a}<0$, the opposing component is removed as

$$
\mathbf{v}'=\mathbf{v}
-\mathbf{a}\frac{\mathbf{v}\cdot\mathbf{a}}{\lVert\mathbf{a}\rVert^2}.
$$

Requested forces are then magnitude-limited to the remaining budget. This ordering guarantees that social or neural steering cannot spend force already required for immediate collision avoidance.

_Implementation: [collision avoidance](src/world.py) · [opposition removal and force allocation](src/creature/flocking.py)_

### 6.4 Flocking formulation

For herding drive $h$, panic $p$, compatible-social presence $s$, personal-space presence $q$, minimum engagement $e_0$, panic suppression $\kappa$, and inherited genes $(g_s,g_a,g_c)$, the effective flocking weights are

$$
e=s\left[e_0+(1-e_0)h\right],\qquad
\pi=1-\kappa p,
$$

$$
w_s=qg_s,\qquad w_a=eg_a\pi,\qquad w_c=eg_c\pi.
$$

The defaults are $e_0=0.25$ and $\kappa=0.5$. Separation remains responsive to immediate personal-space violation, whereas panic attenuates alignment and cohesion.

_Implementation: [flocking weights](src/creature/flocking.py) · [flocking parameters](configs/sim_config.py)_

Let $\mathbf{v}_s$, $\mathbf{v}_a$, and $\mathbf{v}_c$ be the desired separation, alignment, and cohesion velocities. The social target is

$$
\mathbf{v}_{S}=
\frac{w_s\mathbf{v}_s+w_a\mathbf{v}_a+w_c\mathbf{v}_c}
{w_s+w_a+w_c},
$$

bounded by maximum speed. Confidence combines the strongest active weight with compatible group size. Neural and social desired velocities are blended with influence $\beta$, where $0\le\beta\le0.35$ by default:

$$
\mathbf{v}_{B}=(1-\beta)\mathbf{v}_{N}+\beta\mathbf{v}_{S}.
$$

_Implementation: [social intention and blending](src/creature/flocking.py) · [world social-runtime integration](src/world.py)_

Compatibility can be derived from social-tag distance, species membership, or the same composite evolutionary distance used by speciation. In the composite case, compatibility decreases linearly from one at zero distance to zero at the current species threshold. Pairwise distances are cached by immutable neural genome identities.

_Implementation: [compatibility resolver](src/creature/flocking.py) · [composite live compatibility](src/creature/evolution.py)_

---

## 7. Communication

### 7.1 Acoustic channel

A creature may emit a signal with strength $s_e\in[0,1]$ and tone $\tau\in[-1,1]$. Emissions below $0.05$ are omitted. For source–receiver distance $d$ within acoustic range $R_a=480$, perceived strength is

$$
s_h=s_e\left(1-\frac{d}{R_a}\right)^2.
$$

Signals below the hearing threshold $0.05$ are ignored. Among audible candidates, the strongest attenuated signal is selected, with emitter identity used only as a deterministic tie-breaker. The receiver obtains no emitter identity; it receives strength, tone, and body-relative direction represented by sine and cosine.

_Implementation: [acoustic sensing](src/creature/communication.py) · [communication defaults](configs/sim_config.py) · [signal commitment](src/world.py)_

Acoustic energetic cost is quadratic in emitted strength:

$$
C_{\mathrm{sound}}=c_a s_e^2,
$$

with $c_a=0.006$ energy units per second. The quadratic term makes strong long-range signalling disproportionately costly.

_Implementation: [communication energy demand](src/creature/metabolism.py) · [communication configuration](configs/sim_config.py)_

### 7.2 Pheromone fields

Trail and alarm pheromones are independent `float32` concentration fields with the same $64\times44$ dimensions and world bounds as the biome map. They share numerical machinery but never transform into one another. For emission intensity $u\in[0,1]$, configured deposition rate $r_p=0.75$, and elapsed time $\Delta t$, the deposited amount is

$$
a_p=u r_p\Delta t.
$$

If the emitter lies at fractional grid coordinates $(u_g,v_g)$ between columns $i_0,i_1$ and rows $j_0,j_1$, the four additions are

$$
\begin{array}{c|c}
\text{grid node} & \text{added amount}\\ \hline
(i_0,j_0) & a_p(1-u_g)(1-v_g)\\
(i_1,j_0) & a_pu_g(1-v_g)\\
(i_0,j_1) & a_p(1-u_g)v_g\\
(i_1,j_1) & a_pu_gv_g
\end{array}
$$

Concurrent deposits are accumulated per channel and concentrations are clipped to $P_{\max}=1$. Bilinear sampling uses the same four weights in reverse to avoid discontinuous sensor jumps at cell boundaries.

_Implementation: [pheromone deposition](src/creature/communication.py) · [batched communication intents](src/world.py) · [pheromone defaults](configs/sim_config.py)_

Each field follows diffusion with first-order evaporation,

$$
\frac{\partial P}{\partial t}=D\nabla^2P-\lambda P,
$$

where the defaults are $D=390$ world-distance squared per simulated second and $\lambda=0.08\,\mathrm{s}^{-1}$. For grid spacing $\Delta x,\Delta y$ and numerical substep $\delta t$, define

$$
r_x=\frac{D\delta t}{\Delta x^2},\qquad
r_y=\frac{D\delta t}{\Delta y^2}.
$$

The explicit update is

$$
P_{i,j}^{n+1}=\operatorname{clip}_{[0,P_{\max}]}
\left(
\left[
P_{i,j}^{n}
+r_x(P_{i-1,j}^{n}-2P_{i,j}^{n}+P_{i+1,j}^{n})
+r_y(P_{i,j-1}^{n}-2P_{i,j}^{n}+P_{i,j+1}^{n})
\right]e^{-\lambda\delta t}
\right).
$$

Stability requires $r_x+r_y\le 0.5$. The solver therefore derives

$$
\delta t_{\max}=
\frac{0.5}{D(\Delta x^{-2}+\Delta y^{-2})}
$$

and splits a requested advance into $\lceil\Delta t/\delta t_{\max}\rceil$ stable substeps. The default reflective boundary copies edge concentrations into ghost cells, giving zero outward gradient. Wrap mode uses the opposite edge; absorbing mode uses zero-valued ghost cells and returns zero for samples outside the domain.

Field time accumulates independently of rendering and is normally advanced every $0.25\,\mathrm{s}$. At most four complete field updates are processed in one external tick. Excess full intervals after an abnormally large time increment are recorded and dropped while the sub-interval remainder is retained, bounding catch-up work explicitly.

_Implementation: [pheromone solver](src/creature/communication.py) · [boundary modes and numerical parameters](configs/sim_config.py)_

Each creature receives trail and alarm concentrations at the body centre, forward-left probe, and forward-right probe described in Section 4.4. These six raw inputs expose local level and direction without revealing emitter identity or deposit age. Trail and alarm emission costs are linear in intensity:

$$
C_{\mathrm{pheromone}}=c_p(u_{\mathrm{trail}}+u_{\mathrm{alarm}}),
$$

where $c_p=0.002$ per second. These costs enter the same resource ledger as locomotion and neural upkeep.

_Implementation: [pheromone sensing](src/creature/communication.py) · [communication energy accounting](src/creature/metabolism.py)_

---

## 8. Feeding, metabolism, rest, and vitality

### 8.1 Ingestion and stomach state

Eating requires active intent and geometric overlap between a food item and the creature's forward mouth position. Let $K-S$ be remaining stomach space, $b_{\max}=0.5$ the default bite rate, $E_f$ remaining food energy, and $\tau=0.10$ the micro-food tolerance. At a physical contact, the committed bite is

$$
B=
\begin{cases}
E_f,
& E_f\le K-S\ \text{and}\ E_f\le b_{\max}\Delta t(1+\tau),\\
\min(K-S,\;b_{\max}\Delta t,\;E_f), & \text{otherwise}.
\end{cases}
$$

The exceptional branch consumes a pellet that is only slightly larger than the time-scaled bite limit, provided the complete energy fits in the stomach. Bite claims are ordered by physical step and stable identities, and per-food bite capacity prevents unlimited simultaneous consumption.

_Implementation: [mouth geometry and eating](src/creature/metabolism.py) · [food energy removal](src/food.py) · [chronological exposure resolution](src/world.py)_

Larger original food radii map linearly to a difficulty multiplier in $[0.75,1.25]$. The stomach stores both swallowed energy and its difficulty-weighted load. This preserves the composition of mixed bites without retaining every swallowed item. A carried item is kept ahead of the creature and cannot be eaten as though it were an unrelated nearby object.

_Implementation: [food difficulty and stomach load](src/creature/metabolism.py) · [food carrying](src/world.py)_

### 8.2 Digestion

During a biological interval, the maximum processable stomach amount is

$$
S_{\max}=\min(S,q\Delta t),
$$

where $q$ is the inherited digestion rate. Effective conversion efficiency is increased by effective rest $r_e$:

$$
\eta_e=\operatorname{clip}_{[0,1]}
(\eta+b_r r_e),
$$

with inherited efficiency $\eta$ and default rest bonus $b_r=0.10$. Processing difficulty and rapid digestion impose the fraction

$$
f_p=\min\left(f_{\max},
f_0d\left[1+k_q\left(\frac{q}{q_{\max}}\right)^2\right]\right),
$$

where $d$ is mean stomach difficulty, $f_0=0.08$, $k_q=1.5$, and $f_{\max}=0.5$.

_Implementation: [digestion and processing fraction](src/creature/metabolism.py) · [metabolic parameters](configs/sim_config.py)_

For stomach amount $S_c$ actually consumed,

$$
E_{\mathrm{gross}}=S_c\eta_e,\qquad
C_{\mathrm{process}}=E_{\mathrm{gross}}f_p,\qquad
E_{\mathrm{net}}=E_{\mathrm{gross}}-C_{\mathrm{process}}.
$$

$S_c$ is reduced when neither current demand nor remaining energy capacity can use the resulting net energy. Digestion therefore does not destroy stomach contents merely because the usable energy reserve is already full.

_Implementation: [pure digestion calculation](src/creature/metabolism.py) · [digestion tests](tests/creature/test_metabolism_traits.py)_

### 8.3 Energy demand

The per-second demand is the sum

$$
C=C_{\mathrm{base}}+C_{\mathrm{brain}}+C_{\mathrm{move}}
+C_{\mathrm{sprint}}+C_{\mathrm{vision}}+C_{\mathrm{body}}
+C_{\mathrm{sound}}+C_{\mathrm{pheromone}}+C_{\mathrm{digestive}}.
$$

Default basal demand is $0.005$. Movement demand is quadratic, $0.02(v/v_{\max})^2m$, where $m$ is the inherited movement-cost multiplier; panic adds up to $0.04$ per second. Maximum normalized body cost is $0.004(r/r_{\max})^2$. Vision contributes between $0.001$ and $0.006$ per second, and neural upkeep charges $0.00008$ per enabled connection.

_Implementation: [energy cost breakdown](src/creature/metabolism.py) · [body and brain upkeep](src/creature/metabolism.py) · [metabolism and trait defaults](configs/sim_config.py)_

Digestive upkeep penalizes capacity, rate, and efficiency relative to their defaults:

$$
C_{\mathrm{digestive}}=
\min\left(C_{\max},C_d\left[
0.40\left(\frac{K}{K_0}\right)^2+
0.35\left(\frac{q}{q_0}\right)^2+
0.25\left(\frac{\eta}{\eta_0}\right)^2
\right]\right),
$$

with $C_d=0.004$ and $C_{\max}=0.012$. Hence no digestive trait is unconditionally advantageous.

_Implementation: [digestive upkeep](src/creature/metabolism.py) · [digestive defaults](configs/sim_config.py)_

### 8.4 Activity-gated rest

Resting effectiveness depends on realized activity, not merely on the neural rest output. Motor effort, normalized speed, turning, communication, reproduction, and nursing produce

$$
A=\min\left(1,
0.40a_m+0.10a_v+0.15a_\omega+0.15a_c
+0.10a_r+0.10a_n\right).
$$

Reproduction or nursing forces $A=1$. Effective rest is

$$
r_e=r_s(1-A),
$$

where $r_s$ is smoothed neural rest intent. A creature cannot obtain full digestive or paid-healing benefits while simultaneously performing costly behaviour.

_Implementation: [activity formulation](src/creature/metabolism.py) · [effective-rest commit](src/world.py)_

Rest does not create usable energy. It can improve digestion and heal life at up to $0.01r_e$ life units per second, provided the creature pays one unit of stored energy per life unit restored. Healing never revives a creature whose life has already reached zero.

_Implementation: [rest digestion and paid healing](src/creature/metabolism.py) · [rest defaults](configs/sim_config.py)_

### 8.5 Deficit damage, senescence, and death

Available energy first pays non-movement demand and then powered voluntary movement. Unmet ordinary demand produces life damage at $0.25$ life units per missing energy unit. Unmet movement demand is more dangerous at low life. Its multiplier is

$$
M_L=1+(M_{\max}-1)\left(1-\frac{L}{L_{\max}}\right)^2,
$$

with $M_{\max}=4$. The quadratic increase prevents indefinite high-speed motion powered only by life when energy is depleted.

_Implementation: [resource candidate and movement penalty](src/creature/metabolism.py) · [resource-ledger tests](tests/test_life_rest_transactions.py)_

After the default senescence age of $200$ seconds, all per-interval energy costs are multiplied by

$$
M_{\mathrm{age}}=1+0.05(t-200).
$$

Direct collision or other queued damage and deficit damage are committed in the same candidate transaction. A creature survives exactly when its candidate final life is positive; otherwise it is removed and its neural and trait state can be archived for evolutionary recovery.

_Implementation: [senescence factor and death processing](src/world.py) · [survival predicate](src/creature/metabolism.py) · [population defaults](configs/sim_config.py)_

---

## 9. Reproduction, development, and evolution

### 9.1 Autonomous eligibility

At each $20\,\mathrm{Hz}$ biology boundary the ledger credits digestion, deducts ordinary upkeep, resolves deficit damage, and then evaluates reproduction. A surviving creature is eligible when its post-upkeep energy is at least $0.75E_{\max}$, its age is at least ten seconds, its five-second cooldown is complete, and its centered neural intent exceeds $0.2$.

The sub-tick transaction order is fixed: (1) credit digested nutrients to usable energy; (2) deduct basal, visual, locomotor, neural, and other ordinary upkeep; (3) evaluate energy, maturity, cooldown, survival, and intent against that post-upkeep state; and (4) reserve the $45\%$ birth investment. Thus the birth reservation never runs ahead of upkeep or turns ordinary birth-frame upkeep into an unpaid life deficit.

The reproduction neuron is pinned to logistic sigmoid with founder bias $-1$. Intent thresholds must be interpreted in the output's declared domain: raw $\sigma(z)\in[0,1]$ uses $\sigma(z)>0.6$; symmetric centering $y=2(\sigma(z)-0.5)\in[-1,1]$ uses $y>0.2$; and offset centering $y=\sigma(z)-0.5\in[-0.5,0.5]$ uses $y>0.1$. The runtime uses the symmetric form. Eligible requests are shuffled uniformly; gathered energy, age ranking, neural complexity, species size, and identity never prioritize a parent. Capacity-deferred requests pay nothing and remain immediately eligible when a slot opens.

_Implementation: [physiological eligibility](src/world.py) · [neural output contract](src/creature/neat/controller.py) · [population defaults](configs/sim_config.py)_

### 9.2 Atomic birth transaction

For post-upkeep energy $E_p$, the parent reserves $I=0.45E_p$ and the child receives $E_c=0.90I$; the remaining $0.10I$ is conversion loss. Energy, cooldown, offspring count, genotype mutation, neural mutation, species assignment, allocators, and random-number state become observable only after the complete staged transaction succeeds.

_Implementation: [resource transaction resolution](src/world.py) · [evolution shadow transaction](src/creature/evolution.py) · [transaction tests](tests/test_world_reproduction.py)_

Placement tries up to sixteen randomized angular and radial offsets and rejects boundaries or overlap with creatures, food, solid geometry, and other staged offspring. If every position is blocked, the shadow transaction is discarded: the parent loses no energy, cooldown, offspring count, allocator position, species state, or RNG state. A successful child inherits its parent's heading, generation $g_p+1$, and parent identity.

_Implementation: [offspring commit and placement](src/world.py) · [lineage planning](src/creature/evolution.py)_

### 9.3 Non-neural mutation

Bounded continuous traits mutate in latent logit space rather than by additive physical-space noise. For $x\in[a,b]$, define

$$
z=\operatorname{clip}_{[\epsilon,1-\epsilon]}
\left(\frac{x-a}{b-a}\right),\qquad
u=\log\frac{z}{1-z},
$$

then sample

$$
u'=u+\mathcal{N}(0,\sigma_u^2),\qquad
x'=a+(b-a)\frac{1}{1+e^{-u'}}.
$$

The implementation uses $\epsilon=10^{-6}$ only when mapping the parent into latent space and keeps the result strictly inside the configured interval. This avoids boundary point masses from repeated clipping and makes mutations naturally smaller in physical units near a trait limit.

| Trait | Mutation gate | Latent $\sigma_u$ |
|---|---:|---:|
| Vision range | always | $0.32$ |
| Vision angle | always | $0.17$ |
| Radius | always | $0.42$ |
| Movement-cost multiplier | always | $0.27$ |
| Stomach capacity | $0.15$ | $0.27$ |
| Digestion rate | $0.15$ | $0.23$ |
| Digestion efficiency | $0.15$ | $0.23$ |

Every digestive trait receives its own mutation-gate draw; a failed gate inherits the bounded parent value unchanged. Recorded lineage deltas are the realized physical-space differences after transformation.

_Implementation: [vision and physical mutation](src/creature/genotype.py) · [mutation parameters](configs/sim_config.py)_

Each flocking gene has probability $0.005$ of replacement by a uniform unit value and a mutually exclusive probability $0.05$ of latent-logit mutation with $\sigma_u=0.20$. Social-tag coordinates follow the same defaults only when social-tag compatibility is enabled; disabling tag mode consumes no tag-mutation random draws. Colour undergoes a small HSV mutation and is kept away from the configured food-colour neighbourhood; colour has no direct energetic or behavioural effect.

_Implementation: [flocking, tag, and colour mutation](src/creature/genotype.py) · [trait configuration](configs/sim_config.py)_

### 9.4 Neural mutation and real-time evolution

The child's neural genome is a mutated copy of its parent's genome. Current NEAT parameters include connection addition probability $0.5$, connection deletion $0.05$, node addition $0.1$, node deletion $0.05$, weight mutation $0.8$, and weight replacement $0.02$. Innovation and allocator state are retained across births and persistence boundaries so structurally homologous changes remain comparable.

_Implementation: [neural child creation](src/creature/neat/controller.py) · [NEAT mutation parameters](configs/neat_herbivore.ini) · [evolution coordinator](src/creature/evolution.py)_

### 9.5 Infancy, nursing, and extinction recovery

A creature is an infant until age ten seconds. Infant movement cost is multiplied by three as a runtime penalty without modifying the inherited movement gene. Nursing candidates are restricted to the donor's own infants within $2.5$ donor radii, with distance and infant identity providing deterministic ordering. The requested transfer is

$$
E_n=0.05\Delta t.
$$

Actual allocation is limited by the infant's remaining energy capacity and accepted only if the donor survives and can fund it from the same post-upkeep resource transaction. Nursing therefore creates neither energy nor an unpaid donor deficit. Startup validation requires the minimum child endowment to exceed worst-case idle burn over the maturity window by at least $20\%$.

_Implementation: [infancy and nursing](src/world.py) · [family lineage](src/creature/genotype.py) · [population configuration](configs/sim_config.py)_

If no creatures remain, recovery samples preserved species uniformly before sampling an unranked genome within each species. Archived traits are mutated through the ordinary offspring path and re-evaluated for taxonomy. If extinction occurs before any species is archived, the model procedurally creates a fresh founder cohort and registers a valid root species.

_Implementation: [extinction recovery](src/world.py) · [aligned archive pruning](src/creature/evolution.py) · [unranked neural archive](src/creature/neat/controller.py)_

---

## 10. Passive telemetry and speciation

### 10.1 Thermodynamic telemetry

There is no scalar selection score. `CreatureTelemetry` passively records lifetime ingestion $E_i$, realized expenditure $E_s$, offspring, age, food contacts, movement, and behavioural diagnostics. Its energy diagnostics are

$$
E_{\mathrm{net}}=E_i-E_s,\qquad
R_{\mathrm{net}}=\frac{E_{\mathrm{net}}}{\max(t_{\mathrm{age}},1)}.
$$

These values are displayed and persisted but never enter reproduction, genome retention, speciation, or extinction recovery. `CreatureFitness` remains only as a compatibility alias for older imports and checkpoints.

_Implementation: [passive telemetry ledger](src/creature/fitness.py) · [real-time diagnostics](src/creature/neat/rt_neat.py)_

An optional flocking benchmark measures group presence, heading alignment, spacing, and movement:

$$
Q=Q_gQ_aQ_sQ_m,
$$

$$
Q_s=\exp\left[-\left(\frac{d-d^*}{\sigma_d}\right)^2\right].
$$

The current defaults use target group size four, target spacing $60$, tolerance $30$, and reference speed $50$. Its bounded accumulated reward is diagnostic and does not affect reproduction.

_Implementation: [flocking benchmark](src/creature/fitness.py) · [benchmark configuration](configs/sim_config.py)_

### 10.2 Phenotypic distance

For any bounded trait $x\in[x_{\min},x_{\max}]$, the normalized difference between child $c$ and representative $r$ is

$$
\delta_x=
\frac{|\operatorname{clip}(x_c)-\operatorname{clip}(x_r)|}
{x_{\max}-x_{\min}}.
$$

The phenotypic distance is

$$
D_P=\delta_r+\delta_R+\delta_\Phi+\delta_m
+\frac{\delta_K+\delta_q+\delta_\eta}{3}.
$$

Digestive traits are averaged into one component so that adding three closely related digestive dimensions does not triple their total weight.

_Implementation: [phenotypic components and distance](src/creature/speciation.py) · [trait bounds](configs/sim_config.py)_

The inherited flocking-trait distance is

$$
D_F=\frac{|g_{s,c}-g_{s,r}|+|g_{a,c}-g_{a,r}|+|g_{c,c}-g_{c,r}|}{3}.
$$

Social tags affect live compatibility in tag mode but are not part of this speciation term.

_Implementation: [flocking-trait distance](src/creature/speciation.py) · [live social-tag compatibility](src/creature/flocking.py)_

### 10.3 Composite species criterion

Let $D_N$ be neat-python's genomic compatibility distance. The complete distance is

$$
D=D_N+w_PD_P+w_FD_F,
$$

with current defaults $w_P=2$ and $w_F=1$. A child remains in its parent's species if $D\le T$; if $D>T$, it becomes the representative and founder of a new species. The default initial threshold is $T=3.5$.

Species labels never share or rescale reproductive success. They remain operational for three separate purposes: taxonomic assignment by composite distance; active-clade management, where a species leaves the living set when its final member dies even if an archival representative is retained; and diversity-preserving extinction recovery, which samples preserved species before choosing an unranked genome within each selected species.

_Implementation: [composite compatibility and species evaluation](src/creature/speciation.py) · [speciation defaults](configs/sim_config.py)_

Every five seconds, the threshold is adjusted toward a target of five active species. It decreases by $0.05$ per elapsed adjustment interval when too few species exist and increases by the same amount when too many exist, remaining in $[2,7]$. This feedback stabilizes taxonomic resolution as the population evolves.

_Implementation: [adaptive threshold](src/world.py) · [threshold configuration](configs/sim_config.py)_

When a new species is formed, the system records founder traits, raw trait changes, every normalized distance component, the compatibility threshold, and enabled neural connection additions, removals, or weight changes. These records permit later reconstruction of why the offspring crossed the species boundary.

_Implementation: [speciation result and neural shifts](src/creature/speciation.py) · [species telemetry](src/telemetry.py)_

---

## 11. Behavioural analysis and explainability

### 11.1 Behaviour from realized evidence

The behavioural analyser deliberately does not infer behaviour from named NEAT outputs. It receives primitive observations of realized state and trajectory, then applies temporal rules to six categories:

- orientation toward food;
- approach toward food;
- feeding;
- resting;
- group cohesion;
- retreat from alarm pheromone.

This distinction is methodologically important. A positive `want_eat` output is an intention; feeding is recognized only from an actual consumption event. Likewise, turning output alone is not evidence of food orientation unless the creature's realized direction changes consistently toward a visible food target.

_Implementation: [behaviour observation contract and analyser](src/behavior_observer.py) · [world observation sampling](src/world.py)_

The default analyser samples at $10\,\mathrm{Hz}$ over a $2.5$-second sliding window. Candidate evidence must persist for $0.5$ seconds before an ordinary behaviour becomes active; a $0.3$-second grace period prevents momentary evidence loss from splitting one bout. Feeding is event-driven and remains associated with its event for $0.75$ seconds.

_Implementation: [temporal bout state machine](src/behavior_observer.py) · [observer parameters](configs/sim_config.py)_

### 11.2 Evidence and completed bouts

Every active rule emits named evidence with a value, unit, pass/fail result, and explanatory label. A completed bout stores start and end times, duration, evidence summaries, termination reason, and an outcome appropriate to the behaviour, such as food consumed, target lost, approach started, alarm exposure reduced, or interruption.

_Implementation: [behaviour evidence and outcomes](src/behavior_observer.py) · [completed-bout records](src/behavior_history.py)_

Evidence summaries retain sample count, pass count, first and last values, median, quartiles, and total. Sampling memory is bounded. Once capacity is exceeded, deterministic compaction increases the retained stride; consequently long bouts retain exact counts and totals while quantiles are explicitly marked as estimated.

_Implementation: [bounded metric and evidence accumulators](src/behavior_history.py) · [history capacity configuration](configs/sim_config.py)_

Completed bouts are aggregated into per-creature lifetime summaries and species-normalized reports. Species reports include observed creature count, observation time, total bouts, median duration, and bouts per creature-hour, preventing raw bout count from being mistaken for a rate when observation effort differs.

_Implementation: [creature and species history reports](src/behavior_history.py) · [history integration](src/world.py)_

### 11.3 Counterfactual NEAT explanations

Counterfactual analysis evaluates an isolated copy of the focal recurrent network. Starting from the factual sensor vector $\mathbf{x}$ and the exact pre-decision recurrent state, one semantic group of inputs is replaced by neutral values to obtain $\mathbf{x}^{(-I)}$. The worker retains one compiled evaluator per focal brain, queues only shallow-copied state buffers with each probe, and restores fresh buffer dictionaries before every intervention. This preserves intervention independence without repeatedly cloning or serializing the compiled topology, and never mutates the live brain:

$$
\mathbf{y}=N(\mathbf{x}),\qquad
\mathbf{y}^{(-I)}=N(\mathbf{x}^{(-I)}),\qquad
\Delta\mathbf{y}=\mathbf{y}-\mathbf{y}^{(-I)}.
$$

Available interventions remove visible-food cues, resource gradients, hunger through a satiated state, social cues, offspring cues, acoustic cues, trail pheromone, alarm pheromone, or temporal cues. Because all other inputs and the genome remain fixed, the difference estimates local neural dependence on that semantic information group; it is not a claim of causal effect on the complete future world trajectory.

_Implementation: [semantic interventions and pure evaluator](src/counterfactual_neat.py) · [pure brain evaluation](src/creature/neat/brain.py)_

For an output with admissible span $s_y$, influence is normalized as

$$
I_y=\operatorname{clip}_{[0,1]}
\left(\frac{|y-y^{(-I)}|}{s_y}\right).
$$

Behaviour-level influence aggregates only outputs relevant to that behaviour. Direction is interpreted relative to the factual target: an output can be supportive, suppressive, reversing, mixed, or minimal. Influence labels use thresholds $0.10$, $0.30$, and $0.60$ for minimal, weak, moderate, and strong effects.

_Implementation: [output and semantic effects](src/counterfactual_neat.py) · [bout-level counterfactual aggregation](src/counterfactual_neat.py)_

Counterfactual samples belonging to a completed bout are summarized by medians, quartiles, direction counts, and per-output factual/counterfactual values. This joins mechanistic neural sensitivity with independently observed behaviour while keeping the two evidence sources conceptually separate.

_Implementation: [counterfactual bout aggregator](src/counterfactual_neat.py) · [completed explanation records](src/behavior_history.py)_

### 11.4 Species-level scientific profiles

Species analysis compares founder morphology with the parent species, reporting percentage change as

$$
100\frac{x_c-x_p}{|x_p|}.
$$

It also estimates idle and active metabolic costs from founder traits, separates direct sensor-to-action neural changes from hidden-node integration changes, and queries descendant count and average lifespan. These profiles interpret recorded model state; they do not modify creatures or selection.

_Implementation: [morphology, metabolism, and neuroethology profiles](src/analysis.py) · [species records](src/creature/speciation.py)_

Creature and species telemetry stores births, deaths, parent-selection attempts, resource and population metrics, flocking measurements, founder traits, compatibility components, and neural shifts in a relational database. Persistent flock groups are matched over time by membership overlap, enabling group duration and stability measurements without adding a “group identity” sensor to creatures.

_Implementation: [telemetry database](src/telemetry.py) · [persistent flock tracking](src/flocking_telemetry.py)_

---

## 12. Concluding interpretation

The model couples ecology and evolution across several levels. A seeded biome map shapes ordinary resource probability and biome-specific food patches; a conserved biomass inventory limits how much plant and creature energy can coexist. Creatures do not receive biome labels or privileged food maps. Instead, inherited vision observes concrete nearby targets while body-relative probes expose a continuous prediction of ordinary-food richness and local pheromone concentrations. Inherited radius, digestion, movement efficiency, and social genes determine both opportunities and energetic liabilities, and the recurrent NEAT genome transforms the resulting local sensory contract into continuous intentions. Physical integration, collision avoidance, resource availability, and transaction rules determine which intentions become realized behaviour.

Lineages persist only through survival and autonomous energy-funded births; no comparative ranking or parsimony score selects parents. Speciation combines neural and phenotypic change, so neither morphology nor neural topology alone defines evolutionary divergence. The separation between predicted richness, realized food encounters, neural intentions, completed biological transactions, and passive behavioural evidence is central to the model: none of these layers is treated as a substitute for another.

_Implementation synthesis: [genotype](src/creature/genotype.py) · [brain](src/creature/neat/brain.py) · [metabolism](src/creature/metabolism.py) · [evolution and speciation](src/creature/evolution.py)_

Several limitations follow from the formulation. The world is planar; creature bodies are circular and have equal mass; biomes and pheromones are discretized fields; the richness signal predicts the ordinary spawn distribution rather than current food occupancy; and clustered resources are not represented in that prediction. Reproduction is asexual, sensory channels are engineered summaries rather than raw physical receptor arrays, recurrent updates are discrete rather than continuous-time, and compatibility and species thresholds are computational constructs. The conserved budget tracks usable energy, stomach contents, and food energy rather than a complete material or life-reserve chemistry. Passive energy and behavioural measurements describe outcomes but do not define a selection objective. Behavioural rules and counterfactual probes improve interpretability but remain operational definitions, not proof of subjective intention. These simplifications create a tractable experimental system in which ecology, morphology, physiology, neural structure, social interaction, and evolutionary history can be measured under one deterministic simulation contract.

_Sources and validation: [creature architecture tests](tests/creature/test_architecture.py) · [genotype determinism tests](tests/creature/test_genotype_determinism.py) · [scheduler validation](tests/test_scheduler_validation.py) · [behaviour observer tests](tests/test_behavior_observer.py)_
