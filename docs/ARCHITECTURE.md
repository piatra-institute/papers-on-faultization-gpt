# MorphoGPT

## Emergent Competencies in a Minimal Transformer Under Perturbation

### A Levin-Inspired Framework for Understanding What the GPT Algorithm Really Is

> **Note:** This is the original design document written before experiments were run.
> It uses philosophical framing (Nancy, Levin) that motivated the methodology.
> For scientific results with neutral terminology, see `PAPER.md` and `FINDINGS.md`.


---


## 1. Overview

MorphoGPT applies Michael Levin's morphogenetic perturbation methodology to Andrej Karpathy's microgpt — a minimal, fully transparent GPT implementation — in order to discover emergent competencies, fault tolerance, and collective behaviors that are not explicitly encoded in the algorithm.

The philosophical frame comes from Jean-Luc Nancy: we perform **désœuvrement** (unworking) on the GPT loop. We do not merely ablate or benchmark. We interrupt the work of next-token prediction to reveal the **being-singular-plural** of the transformer's components — the community that was always already there but invisible when everything was "working."

The claim: **to understand a collective intelligence, you must unwork it.**


---


## 2. Philosophical Foundation

### 2.1 Nancy's Désœuvrement (Unworking)

In *The Inoperative Community* (La communauté désœuvrée), Nancy argues that community is not a work to be produced — not a project, not an achievement, not a fusion into unity. Community happens through the **interruption** of work. When the work is suspended, what becomes visible is the being-together of singularities: exposed, finite, sharing nothing except their exposition to one another.

Désœuvrement is not destruction. It is not the failure of the work. It is the **making-inoperative** of the work's claim to totality, which reveals what the work was always dependent on but could not represent: the relation between its parts.

**For MorphoGPT**: The "work" (oeuvre) of the GPT is next-token prediction. When the system works perfectly — loss decreasing, names generating — the collective nature of the computation is invisible. Each head, each layer, each parameter appears to be a cog in a machine. But when we freeze a head, corrupt the KV cache, cut the gradient flow, mix incompatible objectives — when we perform désœuvrement — the system's response reveals something: the distributed, redundant, reroutable community of components that constitutes the actual computation. The unworking does not destroy the GPT; it exposes the relations that make the GPT possible.

### 2.2 Nancy's Being Singular Plural

In *Being Singular Plural* (Être singulier pluriel), Nancy's thesis is: **existence is always co-existence**. There is no isolated being; there is only being-with (être-avec). The singular is always already plural — not as an aggregate, not as a fusion, but as a constitutive spacing between singularities that are only what they are through their relation to others.

Key concepts:
- **Comparution** (appearing-together): Singularities do not first exist and then enter into relation. They appear together, are constituted by their appearing-together.
- **Partage** (sharing/dividing): The French word means both sharing and partitioning. Every relation is simultaneously a sharing and a division.
- **Exposition**: To exist is to be exposed — turned outward toward others, vulnerable, not self-enclosed.

**For MorphoGPT**: Each attention head, each MLP neuron, each embedding dimension is a singularity that only exists as what it is through its relation to all others via the **residual stream** — which is the site of partage. The residual stream is both shared (all components read from and write to it) and divided (each component contributes its own vector). Attention is literally comparution: tokens appear to each other, and what they are (their representation) is constituted by this appearing-together. When we freeze or damage a component, we do not remove an isolated part — we alter the fabric of relations, and the system's response reveals the depth of its being-singular-plural.

### 2.3 The Methodological Claim

Nancy and Levin converge on a single methodological insight:

> **You cannot understand a collective system by examining its parts in isolation. You must perturb the collective and observe its response as a trajectory through a problem space.**

Levin: "even familiar, simple algorithms have the surprising ability to deal with perturbations in order to meet the algorithmically specified goals, and also exhibit novel behaviors that are not directly encoded in the algorithm."

Nancy: community "is revealed in the interruption of this work" — not in the work itself.

MorphoGPT operationalizes this: we build a perturbation testbed around microgpt, inject systematic damage, and measure not just degradation but **emergent competencies** — rerouting, delayed gratification, aggregation, and behaviors we do not yet know to look for.


---


## 3. The Mapping: Sorting Algorithms → GPT

### 3.1 Structural Analogy

| Levin (Sorting) | MorphoGPT (GPT) | Nancy |
|---|---|---|
| Array of cells | Model: parameters, heads, layers, embeddings, KV cache | The community |
| Cell value | Parameter value / activation value | The singular |
| Cell-view algorithm | Local component policy (what a head or neuron "does") | Being-with |
| Algotype | Functional role of a component (emergent, observed across contexts) | Comparution |
| Frozen cell | Frozen/damaged component (zeroed head, locked params, corrupted cache) | Désœuvrement |
| Sortedness | Competence metric (loss trajectory, task success, constraint satisfaction) | — |
| Sorting process | Training loop + inference loop | The work (oeuvre) |
| Top-down controller | Global backpropagation + Adam optimizer | The project of community |
| Cell-view sorting | Local learning rules, per-component objectives | Being singular plural |
| Chimera (mixed algotypes) | Mixed architecture: different layer types, mixed objectives, interleaved blocks | Partage |
| Delayed gratification | Temporary loss increase followed by recovery after damage | Rerouting around finitude |
| Aggregation | Emergent specialization/clustering of components | Unexpected community |

### 3.2 What Levin Broke (and What We Break)

Levin broke two assumptions:
1. **Top-down control** → replaced with cell-view (distributed) algorithms
2. **Reliable hardware** → introduced frozen/damaged cells

We break the same two, plus a third:
1. **Top-down control** → replace global backprop with local learning rules
2. **Reliable hardware** → freeze/damage heads, layers, KV cache, gradients, embeddings
3. **Homogeneous policy** → create chimeric GPTs with mixed architectures/objectives

### 3.3 What "Sortedness" Becomes

In Levin's framework, Sortedness is the progress coordinate through problem space. For GPT, we need multiple competence metrics:

- **Loss trajectory**: the primary analog of Sortedness — lower is "more sorted"
- **Task-specific accuracy**: for structured tasks (balanced brackets, copy, name generation quality)
- **Constraint satisfaction rate**: for tasks with hard constraints
- **Perplexity on held-out data**: generalization competence

Crucially, following Levin, we track **trajectories**, not just endpoints. A system that temporarily worsens then recovers (delayed gratification) is exhibiting a different competency than one that degrades monotonically.


---


## 4. Experiment Design

### 4.1 Désœuvrement I: Frozen Components (Damage Injection)

**Directly mirrors Levin's frozen cells.**

We systematically freeze parts of the GPT and measure competence trajectories.

#### 4.1.1 Damage Types

| Damage Type | What It Does | Analog in Levin |
|---|---|---|
| **Freeze parameters** | Stop updating selected params (zero their gradients) | Frozen cell (movable) |
| **Zero head output** | Set a head's output to zero during forward pass | Frozen cell (immovable) |
| **Lock KV cache entries** | Replace cache entries with fixed/stale values | Frozen cell in sorting space |
| **Corrupt activations** | Add Gaussian noise to activations at hook points | Damaged cell |
| **Drop layer** | Skip a layer's computation entirely | Removed cell |
| **Quantize weights** | Coarsely round parameter values | Degraded cell |
| **Random embedding** | Replace token/position embeddings with random vectors for some tokens | Sensory damage |

#### 4.1.2 Damage Targets (Granularity)

- Per-parameter (scalar level — maximally fine-grained, feasible in microgpt)
- Per-head (freeze 1 of 4 heads)
- Per-layer (freeze the entire single layer, or in multi-layer configs, freeze specific layers)
- Per-component-type (freeze all attention but not MLP, or vice versa)
- Per-position (damage only affects certain token positions)
- Per-step (damage appears/disappears on a schedule)

#### 4.1.3 Damage Schedules

Following Levin's distinction between chronic and acute perturbation:

- **Chronic**: damage present throughout training (or throughout inference)
- **Acute**: damage injected at a specific training step window, then removed
- **Progressive**: damage increases over time (stress test)
- **Stochastic**: damage applied with probability p at each step

#### 4.1.4 Measurements

For each damage configuration:
- Loss trajectory (the "Sortedness curve")
- Final loss vs. baseline (competence degradation)
- Recovery dynamics after acute damage removal
- Which undamaged components change most (gradient norms, update magnitudes) — evidence of rerouting
- Inference quality (generated samples)

### 4.2 Désœuvrement II: Cell-View GPT (Decentralized Learning)

**Mirrors Levin's cell-view sorting algorithms.**

In microgpt, learning is fully centralized: a single loss function, global backprop through the entire computation graph, a single Adam optimizer. This is the "traditional sorting algorithm" — a top-down controller.

We create a "cell-view GPT" where components learn locally.

#### 4.2.1 Local Learning Variants

| Variant | Description | What It Tests |
|---|---|---|
| **Block-local loss** | Each layer gets its own linear probe → next-token prediction loss from its local state | Can layers learn useful representations without end-to-end backprop? |
| **Stop-gradient boundaries** | Cut gradient flow at residual connections between layers | How much does credit assignment across layers matter? |
| **Per-component optimizers** | Different learning rates, different Adam betas, or even different algorithms (SGD vs Adam) per layer/head | Does homogeneous optimization matter? |
| **Hebbian attention** | Replace backprop-trained attention with a Hebbian update rule: strengthen connections between co-active query-key pairs | Can attention self-organize without gradient signal? |
| **Predictive coding** | Each layer predicts the next layer's output; update to minimize prediction error locally | Alternative to backprop that is more biologically plausible |

#### 4.2.2 The Question

Levin found that cell-view sorting algorithms are **more efficient** than traditional versions (for Bubble and Insertion sort) and **more fault-tolerant**. The key question for GPT:

> Does decentralized learning produce a GPT that is more robust to damage? Does it develop different internal structure? Does it exhibit competencies that centralized learning does not?

This directly tests Nancy's claim: if the components are truly being-singular-plural — if their existence is constituted by their relations rather than imposed from above — then removing top-down control should not destroy the system but reveal a different mode of collective computation.

### 4.3 Chimeric GPT (Mixed Algotypes)

**Mirrors Levin's chimeric arrays.**

In Levin's chimeras, cells in the same array follow different sorting algorithms. The system still sorts, but exhibits unexpected aggregation behavior.

#### 4.3.1 Chimera Types

| Chimera Type | Description |
|---|---|
| **Mixed layer types** | Some layers use standard softmax attention; others use linear attention, short-range (windowed) attention, or MLP-only (no attention) |
| **Mixed objectives** | Some components trained on next-token prediction; others on auxiliary tasks (masked token prediction, reconstruction) |
| **Mixed pretrained blocks** | Interleave layers from two separately-trained microgpt instances (different random seeds, different data orderings, different hyperparameters) |
| **Opposing objectives** | Half the components trained to predict next token; half trained to predict *previous* token (directly mirrors Levin's increasing vs. decreasing sort experiment) |
| **Mixed precision** | Some components at full precision, others quantized to different levels |

#### 4.3.2 Measurements

- Can the chimera still learn? (Competence trajectory)
- Efficiency: how does the chimera compare to pure-type models?
- **Aggregation**: do components of the same "algotype" develop similar representations or functional roles? This is the GPT analog of Levin's most surprising finding.
  - Measure: representational similarity (cosine similarity of activations) between same-type vs. different-type components
  - Measure: gradient correlation between same-type vs. different-type components
  - Measure: attention pattern similarity
- **Opposing objectives equilibrium**: when components have conflicting goals, what state does the system settle into? (Mirrors Levin's increasing-vs-decreasing sort experiment where Sortedness reaches ~42-74% equilibrium)

### 4.4 Delayed Gratification Detection

**Mirrors Levin's delayed gratification metric.**

Delayed gratification = the ability to temporarily move *away* from the goal in order to make progress later. Levin showed this is context-dependent (more frozen cells → more backtracking) and is not just random noise.

#### 4.4.1 In Training

- Track the loss trajectory step-by-step
- Identify episodes where loss increases then decreases to below the pre-increase level
- Compute the Delayed Gratification index: D = (ΔS_increasing - ΔS_decreasing) / ΔS_decreasing
- Compare D across: undamaged baseline, various damage levels, cell-view vs. centralized
- **Key test**: Does D increase with more damage? If yes, the system is rerouting (genuine delayed gratification), not just fluctuating.

#### 4.4.2 In Inference

- Tasks where the locally most probable token is globally wrong (balanced brackets, long-range dependencies)
- Compare greedy decoding vs. sampling: does the model "know" when to take a non-greedy path?
- This separates competence-in-weights from competence-in-decoding

### 4.5 Emergent Behaviors (What We Don't Know to Look For)

Levin's most important finding — aggregation — was not predicted from the algorithm. His paper emphasizes that "other interesting things could be happening that we do not yet know to test for."

We should:
- Record rich telemetry (all activations, all gradients, all attention patterns, all KV cache states) for post-hoc analysis
- Look for: spontaneous specialization, phase transitions, oscillatory dynamics, symmetry breaking
- Use dimensionality reduction (PCA/t-SNE on activation trajectories) to visualize the system's path through "representation space" — the analog of Levin's sorting-space visualization


---


## 5. Metrics and Instrumentation

### 5.1 Competence Metrics (Analogs of Sortedness)

| Metric | Definition | When Used |
|---|---|---|
| **Training loss** | Cross-entropy loss averaged over document | Training trajectory |
| **Held-out loss** | Loss on unseen documents | Generalization |
| **Name validity** | % of generated names that are plausible (character distribution, length) | Inference quality |
| **Bracket balance** | % of generated bracket sequences that are valid | Structured task competence |
| **Copy accuracy** | Accuracy on input-copying tasks | Long-range dependency |
| **Constraint satisfaction** | General: % of outputs meeting task-specific hard constraints | Task-specific competence |

### 5.2 Trajectory Metrics (How the System Moves Through Problem Space)

| Metric | Definition |
|---|---|
| **Loss slope** | Rate of loss decrease (or increase) over a window |
| **Loss variance** | Stability of training |
| **Recovery time** | Steps to return to pre-damage loss level after acute damage |
| **Delayed Gratification index** | D = (gain after temporary loss) / (temporary loss magnitude) |
| **Rerouting score** | Change in gradient distribution across components after damage (which components "take over") |

### 5.3 Structural Metrics (What the System Looks Like Inside)

| Metric | Definition |
|---|---|
| **Head contribution** | Cosine similarity between each head's output and the residual stream delta |
| **Attention entropy** | Shannon entropy of attention weights per head (sharp vs. diffuse) |
| **Gradient norm per component** | How much gradient flows to each head/layer/embedding |
| **Update magnitude per component** | How much each component changes per step |
| **Representation similarity** | Cosine similarity between activation vectors at corresponding positions across runs/conditions |
| **Dead neuron fraction** | % of ReLU neurons that are always zero |
| **Effective rank** | Rank of activation matrices (measure of representational diversity) |

### 5.4 Chimera-Specific Metrics

| Metric | Definition |
|---|---|
| **Aggregation index** | Analog of Levin's aggregation: representational similarity between same-type vs. different-type components (higher = more clustering) |
| **Dominance** | In opposing-objective chimeras: which objective "wins" (what % sorted in each direction) |
| **Equilibrium state** | Final stable loss/accuracy when opposing objectives reach stasis |


---


## 6. Competence Tasks

### 6.1 Primary Task: Name Generation (from microgpt)

The default microgpt task: learn to generate plausible names from a dataset of ~32k names. Character-level, short sequences (max 15 chars). Good for: basic competence testing, fast iteration.

### 6.2 Structured Tasks (for Delayed Gratification / Long-Range Competence)

| Task | Description | Why It Matters |
|---|---|---|
| **Balanced brackets** | Generate sequences of `(` and `)` that are balanced | Requires counting / long-range constraint. Locally, `(` and `)` are equally likely, but globally they must balance. Tests delayed gratification at inference. |
| **Copy/repeat** | Input: `abc|` → Output: `abc` (repeat the prefix) | Tests long-range memory through attention/KV cache |
| **Reverse** | Input: `abc>` → Output: `cba` | Harder than copy; requires reordering |
| **Simple formal languages** | `a^n b^n` — n a's followed by n b's | Tests counting competence |
| **Palindromes** | Generate palindromic strings | Tests bidirectional structure awareness |

### 6.3 Task Selection for Experiments

- Use **name generation** as the default for all damage/chimera/cell-view experiments (fast, meaningful, matches microgpt baseline)
- Use **structured tasks** specifically for delayed gratification and inference-time competence probes
- Each task should have a clear **competence metric** (validity %, accuracy) that serves as the Sortedness analog


---


## 7. Implementation Architecture

### 7.1 Code Structure

```
morphogpt/
├── ARCHITECTURE.md           # This document
├── microgpt.py               # Karpathy's original microgpt (unmodified reference)
├── morphogpt.py              # microgpt + hook system + instrumentation
├── perturbations.py          # Damage types, chimera configurations, schedules
├── metrics.py                # All metric computations
├── experiments.py            # Experiment runner: sweeps, configs, seeds
├── tasks.py                  # Task definitions (names, brackets, copy, etc.)
├── analysis.py               # Post-hoc analysis, plotting, phase diagrams
├── results/                  # Experiment output data
│   ├── trajectories/         # Loss/metric trajectories per experiment
│   ├── telemetry/            # Full activation/gradient snapshots
│   └── plots/                # Generated visualizations
└── paper/                    # Paper drafts and figures
```

### 7.2 Core: morphogpt.py (microgpt + Hooks)

The central file. Takes microgpt's code and adds a **hook system** at named points in the computation.

#### Hook Points

```python
# Embedding
H_EMB_TOKEN(token_id, pos_id, embedding)     # after token embedding lookup
H_EMB_POS(token_id, pos_id, embedding)        # after position embedding lookup
H_EMB_COMBINED(x)                              # after tok + pos

# Per-layer
H_PRE_ATTN_NORM(layer, x)                     # before attention block norm
H_QKV(layer, q, k, v)                         # after Q, K, V projection
H_KV_CACHE(layer, keys, values)               # after appending to KV cache
H_ATTN_WEIGHTS(layer, head, weights)           # after softmax in attention
H_HEAD_OUT(layer, head, output)                # per-head output
H_ATTN_OUT(layer, x)                          # after Wo projection
H_POST_ATTN_RESIDUAL(layer, x)                # after attention residual add
H_PRE_MLP_NORM(layer, x)                      # before MLP block norm
H_MLP_HIDDEN(layer, x)                        # after fc1 + relu
H_MLP_OUT(layer, x)                           # after fc2
H_POST_MLP_RESIDUAL(layer, x)                 # after MLP residual add

# Output
H_LOGITS(logits)                               # before final softmax

# Backward
H_GRAD(param_name, param, grad)                # after backward, before optimizer
H_UPDATE(param_name, param, delta)             # during Adam step
```

Each hook is a function `(hook_point_name, **kwargs) -> modified_values_or_None`. If a hook returns None, values pass through unchanged. If it returns modified values, those replace the originals.

Hooks are registered via:
```python
model.register_hook("H_HEAD_OUT", freeze_head_hook(layer=0, head=2))
```

#### Instrumentation

A `Probe` object (following Levin's terminology) is passed to each training/inference run. It records:
- Loss at every step
- All metric values at configurable intervals
- Optionally: full activation snapshots for post-hoc analysis

### 7.3 Perturbation Engine: perturbations.py

Each perturbation is a callable that can be registered as a hook.

```python
# Damage types
def freeze_head(layer, head):
    """Zero out a specific head's output."""

def freeze_params(param_names):
    """Zero gradients for specified parameters."""

def corrupt_kv_cache(layer, noise_std):
    """Add Gaussian noise to KV cache entries."""

def drop_layer(layer):
    """Skip a layer's computation (output = input)."""

def corrupt_activations(hook_point, noise_std, prob):
    """With probability prob, add noise at a hook point."""

def quantize_weights(param_names, bits):
    """Round parameter values to simulated low precision."""

# Schedules
def chronic(perturbation):
    """Apply perturbation at every step."""

def acute(perturbation, start_step, end_step):
    """Apply perturbation only during [start_step, end_step]."""

def stochastic(perturbation, prob):
    """Apply perturbation with probability prob at each step."""

def progressive(perturbation, schedule_fn):
    """Apply perturbation with intensity given by schedule_fn(step)."""
```

### 7.4 Chimera System

For chimeric GPTs, we need to support:

1. **Per-layer architecture variation**: replace attention or MLP with alternatives
2. **Per-component objective variation**: route different components' outputs to different loss functions
3. **Mixed pretrained weights**: load different layers from different trained checkpoints

Implementation: each layer in the model can have a `layer_type` attribute that determines its forward pass behavior.

### 7.5 Experiment Runner: experiments.py

```python
@dataclass
class ExperimentConfig:
    task: str                     # "names", "brackets", "copy", ...
    perturbations: list           # list of (perturbation, schedule) pairs
    n_layer: int = 1
    n_embd: int = 16
    n_head: int = 4
    block_size: int = 16
    num_steps: int = 1000
    learning_rate: float = 0.01
    seed: int = 42
    chimera_config: dict = None   # if chimeric
    local_learning: dict = None   # if cell-view
    probe_interval: int = 10      # how often to record metrics
    telemetry: bool = False       # record full activations?

def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run a single experiment and return trajectory + metrics."""

def run_sweep(configs: list[ExperimentConfig]) -> list[ExperimentResult]:
    """Run a batch of experiments."""
```


---


## 8. Experiment Plan (Phases)

### Phase 1: Baseline + Hook System

1. Port microgpt to morphogpt.py with hook system
2. Verify baseline: train on names, confirm loss curve matches original
3. Implement Probe and basic metrics (loss trajectory, generated samples)
4. **Deliverable**: morphogpt.py that reproduces microgpt behavior exactly when no hooks are active

### Phase 2: Frozen Components (Désœuvrement I)

1. Implement damage hooks: freeze_head, freeze_params, corrupt_kv_cache, drop_layer
2. Run systematic experiments:
   - Freeze 1, 2, 3, 4 heads → loss trajectory for each
   - Freeze attention vs. MLP vs. embeddings → compare
   - Chronic vs. acute damage → recovery dynamics
   - Stochastic damage with varying p → robustness curve
3. Compute: delayed gratification index, rerouting score, recovery time
4. **Deliverable**: "Robustness curves" and "phase diagrams" — competence vs. damage rate, broken down by damage target. The first Levin-style results for GPT.

### Phase 3: Cell-View GPT (Désœuvrement II)

1. Implement local learning variants:
   - Block-local loss (per-layer probe heads)
   - Stop-gradient boundaries
   - Per-component optimizers
2. Compare with centralized baseline: loss trajectory, final competence, internal structure
3. Combine with Phase 2: does cell-view GPT handle damage better?
4. **Deliverable**: Evidence for or against the hypothesis that decentralized learning produces more robust computation.

### Phase 4: Chimeric GPT

1. Implement mixed architecture layers (attention vs. linear attention vs. MLP-only)
2. Implement mixed pretrained blocks (interleave from different training runs)
3. Implement opposing objectives (next-token vs. previous-token)
4. Measure: competence, efficiency, aggregation index, equilibrium states
5. **Deliverable**: Chimeric competence maps. Evidence for or against emergent aggregation.

### Phase 5: Structured Tasks + Delayed Gratification

1. Implement bracket balancing, copy, and formal language tasks
2. Run baseline + damage experiments on these tasks
3. Analyze delayed gratification specifically in the context of hard constraints
4. **Deliverable**: Task-specific competence under perturbation. Evidence for delayed gratification as a context-dependent strategy.

### Phase 6: Analysis + Paper

1. Compile all results into phase diagrams, trajectory plots, aggregation curves
2. Write paper connecting results to Levin's findings and Nancy's framework
3. **Deliverable**: Paper + code repository


---


## 9. Paper Outline

### Title
"MorphoGPT: Désœuvrement of the Transformer Loop — Emergent Competencies Under Perturbation in a Minimal GPT"

### Abstract
We apply Michael Levin's morphogenetic perturbation methodology to a minimal GPT implementation, breaking assumptions of reliable hardware, top-down control, and homogeneous policy. Guided by Jean-Luc Nancy's concept of désœuvrement (unworking), we systematically damage, decentralize, and chimerize the GPT training loop to discover emergent competencies not explicitly encoded in the algorithm. We find [results].

### Sections
1. **Introduction**: The GPT as a collective intelligence. Why perturbation reveals more than analysis.
2. **Philosophical Framework**: Nancy's désœuvrement and being-singular-plural as methodological principles.
3. **Background**: Levin et al. on sorting algorithms as morphogenesis. Karpathy's microgpt as the transparent substrate.
4. **Methods**: Hook system, perturbation engine, metrics, competence tasks.
5. **Results I — Frozen Components**: Robustness curves, recovery dynamics, delayed gratification.
6. **Results II — Cell-View GPT**: Local learning, decentralized competence.
7. **Results III — Chimeric GPT**: Mixed architectures, aggregation, opposing objectives.
8. **Discussion**: What the GPT algorithm "really is." Implications for Diverse Intelligence, AI safety, interpretability.
9. **Conclusion**: Désœuvrement as a general methodology for understanding collective computational systems.


---


## 10. Key Hypotheses (What We Expect to Find)

### H1: Graceful Degradation
The GPT will show graceful (not catastrophic) degradation under component damage, similar to Levin's cell-view algorithms showing better error tolerance than traditional versions. Attention heads are known to be partially redundant.

### H2: Rerouting / Delayed Gratification
When components are frozen during training, the system will exhibit temporary loss increases followed by recovery — genuine rerouting around damage, not just noise. The delayed gratification index will increase with damage level (context-sensitivity).

### H3: Cell-View Robustness
GPTs trained with local learning rules will be MORE fault-tolerant than GPTs trained with global backprop, analogous to Levin's finding that cell-view sorting is more error-tolerant.

### H4: Chimeric Competence
Chimeric GPTs (mixed layer types, mixed objectives) will still learn, with efficiency between the pure types (as in Levin's chimeric sorting).

### H5: Emergent Aggregation
In chimeric GPTs, components of the same "algotype" will develop similar functional roles or representations, even though they have no explicit mechanism for detecting each other's type. This would be the GPT analog of Levin's most surprising finding.

### H6: Opposing Objective Equilibrium
When components have conflicting objectives (next-token vs. previous-token prediction), the system will reach a dynamic equilibrium — not fully competent at either task, but stable.

### H7: Unknown Unknowns
Following Levin's emphasis on "novel behaviors that are not directly encoded in the algorithm" and Nancy's insistence that the community is always more than the work — we expect to find emergent behaviors we cannot currently predict.


---


## 11. Connections to Existing Work

This project sits at the intersection of:

- **Mechanistic interpretability** (Olah, Elhage et al.): but we go beyond circuit analysis to perturbation-based competence assays
- **Lottery ticket hypothesis** (Frankle & Carlin): sparse subnetworks carry function — our frozen-component experiments test this at a more granular level
- **Dropout / stochastic depth** (Srivastava et al., Huang et al.): training with random component removal produces robust networks — we study this as an emergent competency rather than a training technique
- **Head pruning** (Michel et al., Voita et al.): attention heads can be removed with limited impact — we extend this from static ablation to dynamic trajectory analysis
- **Local learning rules** (Hinton, Lillicrap et al.): alternatives to backprop — we contextualize these as the "cell-view" vs. "traditional" comparison
- **Diverse Intelligence** (Levin et al.): this is a direct contribution to the field, extending the morphogenetic framework to neural network training

What distinguishes MorphoGPT:
1. **Trajectory analysis**, not just endpoint measurement
2. **Philosophical frame** (Nancy) that motivates qualitatively different questions
3. **Chimeric experiments** — unprecedented for neural networks in this specific form
4. **Fully transparent substrate** (microgpt's scalar autograd) — no hidden complexity


---


## 12. On Transparency and the Choice of microgpt

A critical advantage of microgpt as the substrate: **there is nowhere for complexity to hide**. Every parameter is a named scalar. Every operation is explicit. The computation graph is fully traceable.

This directly parallels Levin's choice of sorting algorithms: "the benefit of these sorting algorithms is precisely that they are simple, easy to understand, and offer no place for additional complexity to hide (unlike in real cells)."

Nancy's désœuvrement requires a work to unwork. microgpt is that work: small enough to be fully comprehended, yet complex enough (4,192 parameters, attention + MLP + embeddings + KV cache + backprop + Adam) to exhibit non-trivial collective behavior.

The question is not "can a 4,192-parameter model do impressive things?" It is: "what competencies does this fully transparent collective system possess that are not apparent from its algorithm?"
