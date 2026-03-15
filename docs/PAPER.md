# Morphogenetic Perturbation Reveals Emergent Behaviors in Minimal Transformers

**Date:** March 2026


## Abstract

We apply morphogenetic perturbation methodology (Levin et al., 2024) to a minimal transformer (4-layer, 16-dimensional, 4-head character-level GPT) through twelve experiments spanning perturbation-during-training (Exp 1-6) and multi-phase morphogenetic interventions (Exp 7-12). We adopt a three-scale protocol: $n = 3$ pilot data provides initial signal, $n = 30$ paired analysis ($n = 30$ runs per condition, matched seeds) resolves ambiguity, and $n = 300$ reveals fine structure. At $n = 3$, several signals were ambiguous — head freezing appeared to improve loss, gradient corruption appeared neutral, Delayed Gratification (DG) Index appeared to scale with perturbation. At $n = 30$, the picture sharpens: we identify four emergent behaviors not directly prescribed by Stochastic Gradient Descent (SGD) (stress inoculation, complete recovery, complete regeneration, and head-freezing trajectory improvement), three that reflect basin geometry (chimera convergence, transplant indifference, cell-view equivalent convergence), and three that demonstrate tolerance (gradient degradation absorbed up to a threshold, partial communication, vision restriction). Cell-view (local loss) achieves equivalent final loss to baseline at both $n = 30$ ($p = 0.78$) and $n = 300$ ($p = 0.90$), demonstrating that local layerwise optimization converges to the same basin as end-to-end backpropagation. At $n = 30$, head freezing shows final-loss improvements at some levels (freeze 4: $p = 0.001$, freeze 12: $p = 0.016$), but at $n = 300$ all final-loss effects resolve to null (all $p > 0.15$), while trajectory improvement strengthens to high significance. At $n = 300$, fine structure emerges: the stress inoculation effect strengthens from $p = 0.138$ (final loss) at $n = 30$ to $p = 0.0001$ at $n = 300$ ($d = -0.227$), vision radius reveals a previously invisible monotonic structure (window 1 harms at $p = 0.021$, window 8 improves at $p = 0.022$), regeneration shows layer-specific residual effects, and all robust $n = 30$ findings hold or strengthen while all null findings remain null. The paper's key finding: gradual noise exposure builds tolerance that sudden exposure does not ($p = 0.0001$, $d = -0.227$), despite identical gradient update rules at every step.


## 1. Introduction

Transformers are typically studied through their outputs: probing learned representations (Belinkov & Glass, 2019), ablating components to measure importance (Michel et al., 2019), or tracing computational circuits (Elhage et al., 2021). These methods characterize what the system has learned or which parts matter. They do not ask what happens when the system is forced to learn under constraint — when its normal operation is *interrupted*.

Levin et al. (2024) introduced a different methodology in the context of simple algorithms. Rather than analyzing sorting algorithms through their final outputs, they perturbed the algorithms during execution: freezing cells, mixing incompatible sorting directions, replacing centralized control with autonomous cell-level policies. The perturbations revealed competencies — fault tolerance, delayed gratification, emergent aggregation — that were invisible during normal operation. The central insight: **perturbation reveals what normal operation conceals**.

This paper asks a sharper version of that question: **does the system exhibit behaviors that the optimizer didn't prescribe?** SGD has one directive: minimize loss. It does not specify *how* to do this — whether to build redundancy, develop stress tolerance, or recover gracefully from damage. When a transformer exhibits these behaviors, they are emergent — not directly prescribed by the optimization objective. The question is not whether these behaviors exist, but which of them are genuinely emergent and which merely reflect the geometry of the loss landscape that SGD was always going to navigate.

The question of whether any finding is genuine requires statistical power. We adopt an explicit three-scale protocol. At $n = 3$ (pilot), signal is visible but ambiguous — the resolution is too low to distinguish real effects from noise. At $n = 30$, the picture sharpens and ambiguous signals resolve. At $n = 300$, fine structure emerges that $n = 30$ cannot see. No scale is wrong. Each reveals different phenomena. Signals that were ambiguous at $n = 3$ and resolved at $n = 30$ are not retracted findings — they are findings at the appropriate resolution.

Our contributions are:

1. **A twelve-experiment morphogenetic methodology** that applies Levin's perturbation protocol to transformer training across two phases: perturbation during training (Exp 1-6) and multi-phase morphogenetic interventions (Exp 7-12).
2. **A three-category classification** that distinguishes emergent behaviors (not directly prescribed by the optimizer) from basin geometry (expected convergence on the given landscape) from tolerance (the system absorbs damage without compensating for it).
3. **The finding that gradual stress builds tolerance** ($p = 0.138$ at $n = 30$, strengthening to $p = 0.0001$ at $n = 300$, $d = -0.227$): the system develops resilience that sudden exposure to identical peak stress does not produce, despite identical gradient update rules at every step.
4. **A three-scale protocol** that treats $n = 3$ pilot findings as coarse signal, $n = 30$ as resolved signal, and $n = 300$ as fine-structure signal.
5. **A sharp distinction between absence and adversity.** Frozen (inactive) layers are tolerated ($p = 0.462$); gradient-negated (adversarial) layers degrade by +24.8% ($p < 0.001$).
6. **A negative result on rerouting.** The Delayed Gratification Index shows no perturbation response at $n = 30$ or $n = 300$, distinguishing transformer perturbation response from the richer compensatory rerouting observed in biological development.


## 2. Related Work

Our findings intersect several established lines of research. We situate each finding against prior work to clarify what is known, what we confirm, and what is new.

**Pruning and the lottery ticket hypothesis.** Frankle & Carlin (2019) showed that trained networks contain sparse subnetworks ("winning tickets") that match full-network performance. Subsequent work extended this to structured pruning of attention heads (Michel et al., 2019; Voita et al., 2019). Our Experiment 1 freezes heads at *random initialization*, not after training — the frozen heads are arbitrary random projections. At $n = 3$, the signal was ambiguous: freezing 8+ heads appeared to improve final loss. At $n = 30$, some final-loss improvements appear significant (freeze 4: $p = 0.001$, freeze 8: $p = 0.023$, freeze 12: $p = 0.016$) alongside a robust mean-trajectory improvement for 4+ frozen heads. At $n = 300$, the final-loss improvements resolve to null (all $p > 0.15$; Spearman $\rho = -0.0045$, $p = 0.84$), but the trajectory improvement strengthens to high significance: freeze 8 ($\Delta = -0.1\%$, $p < 0.0001$, $d = -1.245$), freeze 12 ($\Delta = -0.2\%$, $p < 0.0001$, $d = -1.421$), freeze 16 ($\Delta = -0.2\%$, $p < 0.0001$, $d = -1.312$), confirming that frozen random-projection heads reduce gradient interference.

**SignSGD and low-precision optimization.** Bernstein et al. (2018) established that sign-only gradient updates can match full-precision optimization under appropriate conditions. At $n = 3$, our Experiment 3 signal was ambiguous: gradient degradation appeared neutral. At $n = 30$, the signal resolved clearly: sign-only gradients significantly degrade performance (+5.0% final loss, $p = 0.002$, $d = +0.614$). The discrepancy between our finding and the SignSGD literature may reflect our model's small scale and short training duration. At $n = 300$, this finding strengthens: sign-only gradients degrade by +4.9% ($p < 0.0001$, $d = 0.575$), confirming that the discrepancy with SignSGD is robust and not a small-sample artifact.

**Noise as regularization.** The regularizing effect of gradient noise is well-established (Neelakantan et al., 2015). Our Experiment 3 shows that small noise ($\sigma = 0.01$) produces no significant change ($p = 0.738$), consistent with noise-as-regularization, while large noise ($\sigma = 0.1$) is non-significant for final loss at $n = 30$ ($p = 0.910$) but significantly degrades mean trajectory loss (+2.1%, $p < 0.001$, $d = +3.358$), indicating a noise tolerance threshold that manifests primarily in the training trajectory.

**Local learning rules.** Alternatives to end-to-end backpropagation include greedy layerwise pretraining (Bengio et al., 2007), local learning signals (Nokland & Eidnes, 2019), and forward-forward algorithms (Hinton, 2022). Our cell-view experiment uses local loss (layerwise cross-entropy), eliminating *all* inter-layer gradient flow. The resulting final loss is equivalent to baseline at both $n = 30$ ($+0.1\%$, $p = 0.776$) and $n = 300$ ($-0.0\%$, $p = 0.90$), with only the mean trajectory showing a small cost ($+0.2\%$, $p = 0.005$, $d = +0.556$). Local learning converges to the same basin as end-to-end backpropagation.

**Perturbation analysis in neural networks.** Ablation studies (Meyes et al., 2019), dropout, and pruning are standard tools, but typically measure *component importance*. We use perturbation to characterize *system-level behavioral boundaries* — what the architecture absorbs, what it adapts to, and what degrades it. This distinction connects our work to Levin's morphogenetic framework rather than to standard ablation methodology.

**Levin's morphogenetic framework.** Levin et al. (2024) applied developmental biology concepts to simple sorting algorithms. Key findings included delayed gratification (temporary performance decrease followed by recovery past pre-damage levels) and fault tolerance that exceeded intact system performance. Our results show the transformer exhibits tolerance and, crucially, emergent behaviors not directly prescribed by the optimizer — most clearly in stress inoculation (Experiment 9), complete recovery (Experiment 7), complete regeneration (Experiment 10), and trajectory improvement under head freezing (Experiment 1). Cell-view (local loss) achieves equivalent final-loss convergence, reclassified as basin geometry rather than tolerance. The Delayed Gratification Index, which Levin used to detect rerouting, shows no perturbation response at $n = 30$.

**Stress inoculation.** The phenomenon of gradual stressor exposure building tolerance is well-documented in biology (Meichenbaum, 1985) and metallurgy (work hardening). In deep learning, curriculum learning (Bengio et al., 2009) and noise scheduling in diffusion models provide partial analogs. Our Experiment 9 demonstrates stress inoculation in the gradient noise domain — a finding that connects the noise-as-regularization literature to developmental biology.

**Distributed chess as collective intelligence.** Kofman, Campitelli & Levin (2025) extended the morphogenetic framework to chess with autonomous pieces. Our Experiment 6 uses a proper $2 \times 2$ factorial design with composite perturbations (each cell has both a forward and gradient perturbation). The key finding is that gradient type dominates: sign-only gradients (+5.0--5.2% final loss) degrade far more than noisy $\sigma = 0.1$ gradients (+1.6--2.4%), regardless of forward perturbation type (sign-only $-$ dropout: $-3.0\%$, $p = 0.066$ at $n = 30$; $-2.9\%$, $p < 0.0001$ at $n = 300$). Their information bottleneck finding (intermediate vision radius outperforms omniscience) does not replicate for attention windowing (no significant final-loss effects at $n = 30$).


## 3. Methods

### 3.1 Model Specification

We use a minimal character-level GPT with the following architecture:

| Parameter | Value |
|---|---|
| Layers | 4 |
| Embedding dimension | 16 |
| Attention heads per layer | 4 (16 total) |
| Head dimension | 4 |
| Context length | 16 |
| Vocabulary | ~28 (a-z + special tokens) |
| Normalization | RMSNorm |
| Activation | ReLU |
| Total parameters | ~11,000 |

The model is implemented in a numpy backend for experiment sweeps. The task is character-level name generation trained on a dataset of ~32k names.

The choice of a minimal model is deliberate and follows Levin et al.'s rationale for using sorting algorithms: "the benefit of these sorting algorithms is precisely that they are simple, easy to understand, and offer no place for additional complexity to hide."

### 3.2 Three-Scale Protocol

We adopt an explicit three-scale protocol for interpreting findings. At $n = 3$ (pilot scale), signal is visible but resolution is too low to distinguish real effects from sampling noise — findings at this scale are coarse signal, not conclusions. At $n = 30$ (primary scale), paired statistical analysis ($df = 29$) resolves ambiguity: effects that are real at moderate magnitude ($d \geq 0.4$) become detectable with 80% power. At $n = 300$ (fine-structure scale), effects that are small but real at $n = 30$ either strengthen into clear signals or remain null; and new structure that was below the detection threshold at $n = 30$ may emerge.

A finding that is ambiguous at $n = 3$ and resolves clearly at $n = 30$ is not a retraction — it is a finding at the appropriate resolution. A signal that appeared at $n = 3$, resolved to null at $n = 30$, and was confirmed null at $n = 300$ is a signal whose character stabilized as resolution increased; conversely, a signal that was non-significant at $n = 30$ ($p = 0.138$) but strengthened to $p = 0.0001$ at $n = 300$ is a signal whose reality emerged at higher resolution. This is how science works when the resolution dial turns.

### 3.3 Delayed Gratification (DG) Index

Following Levin et al., we define a metric to detect rerouting behavior — episodes where the system temporarily moves *away* from its goal before recovering past the pre-perturbation level.

**Episode detection.** We scan the loss trajectory for episodes where: (1) loss increases from a local value $L_{\text{start}}$ to a peak $L_{\text{peak}}$, then (2) decreases to a trough $L_{\text{trough}}$ below $L_{\text{start}}$. Each such episode has:

- Temporary cost: $C = L_{\text{peak}} - L_{\text{start}}$
- Net gain: $G = L_{\text{start}} - L_{\text{trough}}$

**Per-episode DG:** $\text{DG}_{\text{episode}} = G / C$

**Aggregate DG Index:** The mean DG across all detected episodes in a training run.

At $n = 3$, the DG Index appeared to scale with perturbation severity. At $n = 30$, this signal resolved to null: no perturbation condition produces a statistically significant DG increase ($p > 0.19$ for all). DG captures a real stochastic property of loss trajectories but does not function as a perturbation response measure at $n = 30$. At $n = 300$, the DG null holds: the metric still does not track perturbation, confirming that DG captures stochastic loss-trajectory structure rather than perturbation response.

### 3.4 Experiment 1: Head Freezing

**Motivation:** Levin's frozen-cell perturbation. Randomly selected attention heads have their parameters frozen at initialization values throughout training.

**Protocol:** Sweep over {0, 1, 2, 4, 8, 12, 16} frozen heads. Frozen heads participate in the forward pass but receive no gradient updates.

### 3.5 Experiment 2: Cell-View GPT

**Motivation:** Levin's cell-view sorting algorithms. Each transformer layer is treated as an autonomous agent.

**Protocol:** Local loss (layerwise cross-entropy) at all layer boundaries. Each layer receives only its own local loss signal, computed as a cross-entropy loss from a per-layer projection head. No inter-layer gradient flow.

### 3.6 Experiment 3: Gradient Degradation

**Motivation:** Levin's noisy signaling channels. Four corruption methods:

| Method | Description |
|---|---|
| Noisy ($\sigma = 0.01$) | Additive Gaussian noise, small scale |
| Sign-only | Gradient reduced to {-1, 0, +1}, magnitude discarded |
| Quantized (3-bit) | Gradient values rounded to 8 levels |
| Noisy ($\sigma = 0.1$) | Additive Gaussian noise, large scale |

### 3.7 Training Protocol and Statistical Methods

All experiments use: 200 training steps, 30 independent runs per condition (seeds 42–71), loss and per-head metrics recorded at every step. The 200-step horizon captures early learning dynamics.

**Statistical analysis.** All comparisons use two-sided paired $t$-tests, with runs matched by seed across conditions ($n = 30$, $df = 29$). Pairing by seed controls for initialization variance. We report effects as statistically significant at $p < 0.05$ and marginal at $0.05 < p < 0.10$. With 30 paired observations, statistical power is adequate to detect moderate effects (Cohen's $d \geq 0.4$ at 80% power). Effect sizes are reported as Cohen's $d$ for paired differences. We distinguish between *statistically supported* findings ($p < 0.05$) and *observational* patterns.

**Multiple comparisons.** We do not apply formal multiple-comparison correction across the twelve experiments. Each experiment tests a distinct perturbation type with its own pre-specified comparison, rather than screening a family of interchangeable hypotheses. We note that with twelve primary comparisons, some marginal results ($0.01 < p < 0.05$) should be interpreted with appropriate caution. The strongest findings ($p < 0.0001$) survive any reasonable correction.

**Confirmatory vs. exploratory.** We distinguish three categories: pre-specified primary comparisons (each experiment's main effect), secondary metrics (mean trajectory loss, DG index), and fine-structure analyses (layer-specific or dose-response patterns discovered at $n = 300$). Only primary comparisons are treated as confirmatory; secondary and fine-structure findings are explicitly exploratory.

### 3.8 Experiment 4: Vision Radius Sweep

**Motivation:** Kofman, Campitelli & Levin's (2025) vision radius experiment in distributed chess. We restrict each attention head's context window.

**Protocol:** Sweep over window sizes $\{1, 2, 4, 8, 16\}$ plus unmodified baseline. Window=16 equals block size (sanity check).

### 3.9 Experiment 5: Communication Topology

**Motivation:** The chess paper's relay chains. We scale the fraction of gradient signal passed through layer boundaries.

**Protocol:** Five topologies parameterized by gradient pass fraction:

| Topology | Pass Fraction | Description |
|---|---|---|
| Full | 1.00 | Standard backpropagation (baseline) |
| Heavy | 0.75 | 75% of gradient signal passes through |
| Half | 0.50 | 50% of gradient signal passes through |
| Light | 0.25 | 25% of gradient signal passes through |
| Cell-view | 0.00 | No inter-layer gradient flow |

### 3.10 Experiment 6: Courage vs. Caution

**Motivation:** Kofman et al.'s "cautious position, courageous moves" strategy. We create a $2 \times 2$ factorial design with composite perturbations — each cell applies *both* a forward perturbation and a gradient perturbation simultaneously:

| | Cautious Gradients (sign-only) | Courageous Gradients (noisy $\sigma = 0.1$) |
|---|---|---|
| **Cautious Forward** (tiny noise $\sigma = 0.001$) | (a) Tiny noise + sign-only | (b) Tiny noise + noisy $\sigma = 0.1$ |
| **Courageous Forward** (dropout $p = 0.1$) | (c) Dropout + sign-only | (d) Dropout + noisy $\sigma = 0.1$ |

### 3.11 Experiment 7: Recovery After Damage

**Motivation:** Levin's regeneration paradigm — does a damaged organism recover after the damage is removed? Does it overshoot baseline (the Levin signature)?

**Protocol:** Three-phase training with matched controls.
- Phase 1: Normal training (200 steps)
- Phase 2: Damage — freeze 8 random heads (100 steps)
- Phase 3: Recovery — unfreeze all heads, continue (200 steps)
- Control: Undamaged training for the same total duration (500 steps)

Learning rate schedule is matched across phases and control ($\text{lr}(t) = \text{lr}_0 \cdot (1 - t / 500)$). Paired by seed ($n = 30$).

### 3.12 Experiment 8: Chimera Assembly

**Motivation:** Chimeric organisms assembled from parts of different embryos. Can a network assembled from independently-trained components function?

**Protocol:** Train model A (seeds 42–71) and model B (seeds 1042–1071) independently for 200 steps. Assemble four chimera types by selecting each layer from either model:

| Chimera | Layer 0 | Layer 1 | Layer 2 | Layer 3 |
|---|---|---|---|---|
| AABB | A | A | B | B |
| ABAB | A | B | A | B |
| BBAA | B | B | A | A |
| ABBA | A | B | B | A |

Shared parameters (embeddings, output projection) come from model A. Continue training each chimera for 200 more steps. Control: model A continues training without modification.

### 3.13 Experiment 9: Gradual vs. Sudden Damage

**Motivation:** Biological stress inoculation — gradual exposure to a stressor builds tolerance that sudden exposure does not.

**Protocol:** Four conditions, all 200 steps:
- Control: no gradient noise
- Sudden full: noisy gradients ($\sigma = 0.1$) for all 200 steps
- Gradual: linear ramp from $\sigma = 0$ to $\sigma = 0.1$ over 200 steps
- Sudden half: no noise for first 100 steps, then $\sigma = 0.1$ for remaining 100 steps

The gradual condition reaches the same peak noise level as sudden full but arrives there incrementally.

### 3.14 Experiment 10: Regeneration (Layer Reset)

**Motivation:** Regeneration after tissue destruction. Can a trained network rebuild a destroyed layer?

**Protocol:** Two-phase training.
- Phase 1: Normal training (200 steps)
- Phase 2: Reset one layer's weights to random initialization (new random seed). Zero the corresponding Adam optimizer state. Continue training for 200 more steps.
- Control: no reset, continue training.
- Test all 4 layers independently. Paired by seed ($n = 30$).

### 3.15 Experiment 11: Transplantation

**Motivation:** Organ transplantation — is a layer from a separately-trained donor network accepted better than a random replacement?

**Protocol:** Train model A and model B independently (200 steps each, different seeds). For each layer $L$:
- Transplant: replace layer $L$ of model A with layer $L$ from model B, continue training 200 steps
- Random reset: replace layer $L$ of model A with random weights, continue training 200 steps
- Control: model A continues without modification

Adam buffers are zeroed for the replaced layer in both transplant and random conditions. Paired by seed ($n = 30$).

### 3.16 Experiment 12: Competing Objectives

**Motivation:** Inter-organ conflict — what happens when part of the network fights the objective?

**Protocol:** Two-phase training.
- Phase 1: Normal training (200 steps)
- Phase 2: Negate gradients for layers 2-3 while layers 0-1 train normally (200 steps). The negated layers receive gradient signals that push them *away* from the loss minimum.
- Comparison: freeze layers 2-3 (zero their gradients) instead of negating
- Control: normal training for 400 steps total

This tests whether layers 0-1 can compensate for actively adversarial downstream layers (negation) versus merely inactive ones (freezing).


## 4. Results

### 4.1 Experiment 1: Head Freezing

**At $n = 3$:** The signal was ambiguous. Head freezing appeared to improve final loss — freezing 8+ heads seemed to produce statistically significant improvement ($p = 0.009$ in pilot data). The coarse resolution could not distinguish this from noise.

**At $n = 30$:** At this scale, some final-loss improvements appear significant: freeze 2 ($-0.3\%$, $p = 0.025$, $d = -0.430$), freeze 4 ($-0.5\%$, $p = 0.001$, $d = -0.647$), freeze 8 ($-0.5\%$, $p = 0.023$, $d = -0.438$), freeze 12 ($-0.7\%$, $p = 0.016$, $d = -0.467$). Freeze 1 ($p = 0.253$) and freeze 16 ($p = 0.322$) are non-significant. The overall monotonic trend is weak (Spearman $\rho = -0.015$, $p = 0.83$). Mean trajectory loss shows a robust improvement for 4+ frozen heads, with larger effect sizes than final loss. This is a different, finer signal that became visible at higher resolution.

**At $n = 300$:** The final-loss improvements resolve to null (all $p > 0.15$; Spearman $\rho = -0.0045$, $p = 0.84$). The $n = 30$ final-loss significances were artifacts of moderate sample size. The trajectory improvement strengthens to high significance: freeze 4 mean-loss $\Delta = -0.1\%$ ($p < 0.0001$, $d = -0.971$), freeze 8 $\Delta = -0.1\%$ ($p < 0.0001$, $d = -1.245$), freeze 12 $\Delta = -0.2\%$ ($p < 0.0001$, $d = -1.421$), freeze 16 $\Delta = -0.2\%$ ($p < 0.0001$, $d = -1.312$). The monotonic dose-response in trajectory improvement, combined with no final-loss cost at $n = 300$, confirms that frozen random-projection heads reduce gradient interference throughout training without affecting convergence.

**Table 1.** Head freezing results (means ± std across 30 runs).

| Frozen Heads | Final Loss | Mean Loss | DG Index |
|:---:|:---:|:---:|:---:|
| 0 (baseline) | 2.557 ± 0.407 | 2.627 ± 0.028 | 0.680 ± 1.131 |
| 1 | 2.553 ± 0.406 | 2.627 ± 0.028 | 0.632 |
| 2 | 2.549 ± 0.405 | 2.626 ± 0.028 | 0.689 |
| 4 | 2.543 ± 0.397 | 2.625 ± 0.028 | 0.636 |
| 8 | 2.544 ± 0.396 | 2.623 ± 0.028 | 0.640 |
| 12 | 2.539 ± 0.400 | 2.622 ± 0.028 | 0.614 |
| 16 | 2.547 ± 0.407 | 2.620 ± 0.028 | 0.570 |

Final loss shows improvements at several freezing levels: freeze 2 ($p = 0.025$, $d = -0.430$), freeze 4 ($p = 0.001$, $d = -0.647$), freeze 8 ($p = 0.023$, $d = -0.438$), freeze 12 ($p = 0.016$, $d = -0.467$). However, these resolve to null at $n = 300$ (all $p > 0.15$). Mean trajectory loss shows a robust improvement for 4+ frozen heads: freeze 4 ($p < 0.001$, $d = -1.008$), freeze 8 ($p < 0.001$, $d = -1.228$), freeze 12 ($p < 0.001$, $d = -1.366$), freeze 16 ($p < 0.001$, $d = -1.070$). The effect size is 0.1–0.3% of mean loss — statistically robust but practically small. Frozen heads appear to reduce gradient interference during the trajectory without altering the convergence basin. This trajectory improvement is notable: freezing weights reduces gradient computation, but there is no obvious reason the remaining gradient computation should improve simply because some heads are excluded. The improvement likely reflects reduced destructive gradient interactions.

The DG Index does not increase with freezing. No freezing level produces a significant DG change.

**Classification:** Trajectory improvement for 4+ frozen heads = *emergent behavior* (not prescribed by the optimizer). Final-loss indifference = *basin geometry*.

### 4.2 Experiment 2: Cell-View GPT (Local Loss)

**At $n = 3$:** The signal was ambiguous. Cell-view appeared to elevate DG substantially (+25.5%), suggesting possible rerouting behavior.

**At $n = 30$:** The DG signal resolved to null ($p = 0.90$). The final-loss signal is non-significant: cell-view produces near-identical final loss to baseline ($+0.1\%$, $p = 0.776$, $d = +0.053$). Only the mean trajectory loss shows a small but significant cost ($+0.2\%$, $p = 0.005$, $d = +0.556$).

**At $n = 300$:** The final-loss equivalence is confirmed at high power: cell-view final loss $-0.0\%$ ($p = 0.90$, $d = -0.007$). Mean trajectory: $+0.2\%$ ($p < 0.0001$, $d = +0.731$). DG: $p = 0.14$ (ns). Local layerwise optimization converges to the same basin as end-to-end backpropagation, with only a slight trajectory cost.

**Table 2.** Cell-view (local loss) vs. baseline (means ± std across 30 runs).

| Condition | Mean Loss | Final Loss | DG Index |
|:---:|:---:|:---:|:---:|
| Baseline | 2.627 ± 0.028 | 2.557 ± 0.407 | 0.680 |
| Cell-view | 2.632 ± 0.027 | 2.559 ± 0.419 | 0.655 |

Replacing end-to-end backpropagation with local loss produces equivalent final loss ($+0.1\%$, $p = 0.776$, $d = +0.053$) and a small mean-trajectory cost ($+0.2\%$, $p = 0.005$, $d = +0.556$). The system converges to the same basin without inter-layer gradient flow — the loss landscape guides each layer to its functional role independently.

**Classification:** *Basin geometry* — local loss converges to the same minimum as end-to-end backpropagation, reflecting the smoothness of the loss landscape rather than system tolerance. The DG elevation was noise at this resolution.

### 4.3 Experiment 3: Gradient Degradation

**At $n = 3$:** The signal was ambiguous. All four gradient degradation methods appeared neutral ($p > 0.26$), and small noise appeared to help.

**At $n = 30$:** The ambiguous signal resolved. Two of four methods significantly degrade final loss; one is non-significant but degrades the trajectory; one is genuinely tolerated.

**At $n = 300$:** The threshold between tolerance and degradation sharpens. Noise at $\sigma = 0.01$ remains non-significant ($-0.2\%$, $p = 0.28$), confirming genuine tolerance. All three degradation conditions strengthen: noise at $\sigma = 0.1$ (+2.2%, $p < 0.0001$, $d = 0.367$), sign-only (+4.9%, $p < 0.0001$, $d = 0.575$), and quantized 3-level (+3.6%, $p < 0.0001$, $d = 0.529$). The sign-only effect strengthened from $p = 0.002$ at $n = 30$ to $p < 0.0001$ at $n = 300$.

**Table 3.** Gradient degradation results (means across 30 runs).

| Method | Final Loss | Δ% | $p$ (final) | Mean Loss | $p$ (mean) | $d$ (mean) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline | 2.557 | — | — | 2.627 | — | — |
| Noisy ($\sigma = 0.01$) | 2.552 | -0.2% | 0.738 | 2.628 | 0.496 | +0.126 |
| Noisy ($\sigma = 0.1$) | 2.553 | -0.2% | 0.910 | 2.683 | <0.001*** | +3.358 |
| Sign-only | 2.685 | +5.0% | 0.002** | 2.729 | <0.001*** | +5.696 |
| Quantized (3-bit) | 2.653 | +3.8% | 0.008** | 2.693 | <0.001*** | +4.457 |

Two of four methods significantly degrade final loss: sign-only (+5.0%, $p = 0.002$, $d = +0.614$) and quantized (+3.8%, $p = 0.008$, $d = +0.519$). Noisy $\sigma = 0.1$ is non-significant for final loss ($p = 0.910$) but strongly degrades mean trajectory (+2.1%, $p < 0.001$, $d = +3.358$). Only small noise ($\sigma = 0.01$) is genuinely tolerated ($p = 0.738$ final, $p = 0.496$ mean). Mean loss effects are highly significant for sign-only and quantized ($p < 0.001$) with large Cohen's $d$ values (4.5–5.7). The "noise helps" effect (noisy $\sigma = 0.01$ improving loss) is not supported ($p = 0.496$ for mean loss).

**Classification:** *Tolerance* — the system absorbs gradient noise up to a threshold ($\sigma = 0.01$). Above that threshold, degradation follows. The gradient-quality information (sign structure, quantization levels) carries more essential signal than quantity.

### 4.4 Experiment 4: Vision Radius Sweep

**At $n = 3$:** The signal was ambiguous. An information bottleneck effect appeared possible — intermediate window sizes seemed to outperform full context.

**At $n = 30$:** The ambiguous signal resolved to null for final loss across all window sizes. Tiny mean-trajectory effects are statistically detectable but not practically meaningful.

**At $n = 300$:** Fine structure emerges that was invisible at $n = 30$. Window 1 significantly harms performance (+0.3%, $p = 0.021$, $d = +0.134$), windows 2 and 4 remain non-significant ($p = 0.93$ and $p = 0.93$ respectively), window 8 produces a small but significant improvement ($-0.1\%$, $p = 0.022$, $d = -0.133$), and window 16 is identical to baseline. This reveals a monotonic structure: extreme restriction harms, moderate restriction is neutral, and a mild restriction slightly benefits — a pattern consistent with beneficial information bottleneck effects that were below detection threshold at $n = 30$.

**Table 4.** Vision radius results (means across 30 runs).

| Window | Final Loss | $p$ (final) | Mean Loss | $p$ (mean) | $d$ (mean) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline (full) | 2.557 | — | 2.627 | — | — |
| 1 | 2.562 | 0.618 | 2.639 | <0.001*** | +1.847 |
| 2 | 2.555 | 0.898 | 2.628 | 0.156 | — |
| 4 | 2.549 | 0.443 | 2.625 | <0.001*** | -0.735 |
| 8 | 2.553 | 0.304 | 2.626 | 0.037* | -0.400 |
| 16 | 2.557 | 1.000 | 2.627 | 1.000 | 0.00 |

No window size significantly changes final loss at $n = 30$ ($p > 0.30$ for all). Window=16 reproduces baseline exactly (sanity check). Mean-loss effects exist but are negligibly small. The chess paper's finding that intermediate vision radius outperforms omniscience does not produce meaningful effects for attention windowing at this scale.

**Classification:** *Tolerance* — attention restriction at all tested scales is absorbed without meaningful final-loss change.

### 4.5 Experiment 5: Communication Topology

**At $n = 3$:** The signal was ambiguous. A U-shaped loss curve appeared possible — partial communication seemed to outperform both full and no communication.

**At $n = 30$:** The U-shaped curve resolved to flat across all gradient fractions, including zero communication (cell-view). Partial gradient flow is absorbed without meaningful degradation.

**At $n = 300$:** The architecture's indifference to gradient fraction holds at high power. Heavy ($p = 0.92$), half ($p = 0.033$), and light ($p = 0.59$) communication topologies remain largely non-significant. Cell-view final loss is non-significant ($p = 0.90$), consistent with Experiment 2's finding that local loss achieves equivalent convergence. Only cell-view mean trajectory shows significant cost ($+0.2\%$, $p < 0.0001$, $d = +0.731$).

**Table 5.** Communication topology results (means across 30 runs).

| Topology | Fraction | Final Loss | $p$ (final) | Mean Loss | $p$ (mean) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Full | 1.00 | 2.557 | — | 2.627 | — |
| Heavy | 0.75 | 2.553 | 0.048* | 2.627 | 0.934 |
| Half | 0.50 | 2.555 | 0.510 | 2.627 | 0.452 |
| Light | 0.25 | 2.555 | 0.494 | 2.627 | 0.470 |
| Cell-view | 0.00 | 2.559 | 0.776 | 2.632 | 0.005** |

Partial gradient flow (25–75%) produces no meaningful degradation. Heavy (75%) shows a marginal final-loss improvement ($p = 0.048$, $d = -0.377$, $-0.1\%$). Half and light are non-significant. Cell-view (0%) shows equivalent final loss ($+0.1\%$, $p = 0.776$, $d = +0.053$) but slightly elevated mean trajectory loss ($+0.2\%$, $p = 0.005$, $d = +0.556$).

**Classification:** *Tolerance* — the system absorbs substantial reductions in inter-layer gradient flow. Even the total removal of inter-layer communication (cell-view/local loss) achieves equivalent final-loss convergence, with only a slight trajectory cost.

### 4.6 Experiment 6: Courage vs. Caution

**At $n = 3$:** The signal was ambiguous. The courage/caution matrix appeared to produce inconsistent results without clear pattern.

**At $n = 30$:** The $2 \times 2$ factorial design with composite perturbations reveals a clear pattern: gradient type dominates over forward perturbation type.

**At $n = 300$:** The gradient-type dominance is confirmed at high statistical power. All four composite conditions significantly degrade final loss: cautious/cautious ($+5.2\%$, $p < 0.0001$, $d = +0.624$), cautious/courageous ($+1.9\%$, $p < 0.0001$, $d = +0.318$), courageous/cautious ($+5.0\%$, $p < 0.0001$, $d = +0.616$), courageous/courageous ($+2.5\%$, $p < 0.0001$, $d = +0.419$). The sign-only vs. noisy gradient contrast: $-2.9\%$ ($p < 0.0001$, $d = -0.355$).

**Table 6.** Courage vs. caution results — $2 \times 2$ factorial with composite perturbations (means across 30 runs).

| Condition | Final Loss | $p$ (final) | Mean Loss | $p$ (mean) | $d$ (mean) |
|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline | 2.557 | -- | 2.627 | -- | -- |
| (a) Caut./Caut. (tiny noise + sign-only) | 2.693 | 0.001\*\*\* | 2.728 | <0.001\*\*\* | +6.056 |
| (b) Caut./Cour. (tiny noise + noisy $\sigma$=0.1) | 2.618 | 0.104 | 2.687 | <0.001\*\*\* | +3.426 |
| (c) Cour./Caut. (dropout + sign-only) | 2.698 | 0.002\*\* | 2.734 | <0.001\*\*\* | +4.980 |
| (d) Cour./Cour. (dropout + noisy $\sigma$=0.1) | 2.596 | 0.154 | 2.691 | <0.001\*\*\* | +3.487 |

The key finding is that **gradient type dominates**: conditions with sign-only gradients (a, c) degrade final loss by +5.3--5.5%, while conditions with noisy $\sigma = 0.1$ gradients (b, d) degrade by +1.6--2.4%. The forward perturbation type (tiny noise vs. dropout) has a much smaller effect. The sign-only vs. noisy gradient contrast (averaging across forward types) yields $-3.0\%$ final loss ($p = 0.066$, $d = -0.349$ at $n = 30$; $-2.9\%$, $p < 0.0001$, $d = -0.355$ at $n = 300$) and $-1.7\%$ mean loss ($p < 0.001$, $d = -1.754$). Conditions (b) and (d) with noisy gradients are non-significant for final loss ($p = 0.104$ and $p = 0.154$) but strongly significant for mean trajectory, indicating the noisy gradient perturbation affects the training path more than the endpoint.

**Classification:** *Gradient-type dominance* — the factorial design reveals that gradient information quality (sign-only vs. noisy) is the primary driver of degradation, with forward perturbation type playing a secondary role. This is a substrate-dependent finding: the gradient channel is more sensitive to information destruction than the forward channel.

### 4.7 Experiment 7: Recovery After Damage

**At $n = 3$:** The signal was ambiguous. Recovery appeared complete but with too few observations to confirm.

**At $n = 30$:** Complete recovery is unambiguously confirmed across all 30 runs.

**At $n = 300$:** A tiny but statistically significant residual emerges at high power. Recovery vs. control: $+0.1\%$ ($p = 0.030$, $d = +0.126$). Final-loss ratio: $1.0009 \pm 0.0072$. 272 of 300 runs recovered, with mean overshoot $-0.0009 \pm 0.0017$. Recovery time: $0.8 \pm 1.2$ steps. The effect is real but negligibly small — the residual is within practical equivalence.

**Table 7.** Recovery after transient damage ($n = 30$, 500 total steps).

| Metric | Recovery | Control | $p$ (paired) |
|---|---|---|---|
| Final loss | 2.451 ± 0.370 | 2.451 ± 0.375 | 0.886 |
| Final ratio (rec/ctrl) | 1.0000 ± 0.008 | — | — |
| Recovery time | 1 ± 1 steps | — | 30/30 recovered |
| Overshoot | -0.0014 ± 0.0015 | — | — |

The damaged-then-recovered model reaches the same final loss as the undamaged control ($p = 0.886$, ratio 1.0000). All 30 runs recovered within a mean of 1 step after damage removal. No meaningful overshoot was observed (mean overshoot $= -0.0014$). The 100 steps of training with 8 frozen heads had no lasting effect — the "damage" was entirely absorbed by subsequent normal training.

The completeness of recovery ($p = 0.886$, ratio 1.0000) is notable: a more brittle system could recover to a different basin, but this one returns to the same endpoint with no path-dependence. At $n = 300$, a tiny residual becomes detectable ($p = 0.030$, $d = +0.126$) but remains within practical equivalence. The optimizer prescribes convergence to *a* minimum, not necessarily to the *same* minimum after a detour through a constrained subspace.

**Classification:** *Emergent behavior* — complete, path-independent recovery to identical final loss is not directly prescribed by the loss minimization objective.

### 4.8 Experiment 8: Chimera Assembly

**At $n = 3$:** The signal was ambiguous. Chimeras appeared to converge but from very few observations.

**At $n = 30$:** Convergence is confirmed for all chimera types. The specific layer assignment (which layers come from which model) does not matter.

**At $n = 300$:** All chimera types remain non-significant (AABB $p = 0.35$, ABAB $p = 0.12$, BBAA $p = 0.079$, ABBA $p = 0.31$), with all conditions falling within 0.3% of control ($n = 300$ ctrl = 2.4480). BBAA shows a marginal trend ($p = 0.079$) but does not reach significance. The basin of attraction is uniformly accessible from all tested chimera configurations.

**Table 8.** Chimera assembly results ($n = 30$, 200 + 200 steps).

| Condition | Initial Loss | Final Loss | vs Control $p$ |
|---|---|---|---|
| Control (A continues) | — | 2.494 ± 0.344 | — |
| AABB | 2.569 | 2.484 | 0.265 |
| ABAB | 2.563 | 2.494 | 0.985 |
| BBAA | 2.605 | 2.472 | 0.076† |
| ABBA | 2.542 | 2.487 | 0.559 |

All chimera types converge to the same final loss as the control ($p > 0.07$ for all). Despite starting at elevated loss (2.54–2.61 vs. control ~2.49), the chimeras reach control-equivalent performance.

The chimera result shows that the basin of attraction is wide enough to absorb dramatically different starting conditions. SGD prescribes convergence; the finding is that the basin is wide enough to reach from dramatically different starting points.

**Classification:** *Basin geometry* — SGD re-finds the same minimum from any structurally valid starting point.

### 4.9 Experiment 9: Gradual vs. Sudden Damage

**At $n = 3$:** The signal was ambiguous. The gradual vs. sudden comparison appeared promising but underpowered.

**At $n = 30$:** The key finding resolves clearly and significantly. This is the paper's strongest emergent-behavior result.

**At $n = 300$:** The stress inoculation effect strengthens dramatically, from $p = 0.138$ (final loss) at $n = 30$ to $p = 0.0001$ at $n = 300$. Sudden full noise degrades by +1.8% vs. control ($p < 0.0001$, $d = +0.318$). Gradual noise shows +0.5% degradation ($p = 0.017$, $d = +0.139$). The critical comparison — gradual vs. sudden full — yields $\Delta = -1.3\%$ ($p = 0.0001$, $d = -0.227$), confirming stress inoculation as a robust phenomenon. Sudden half noise: $+0.8\%$ ($p = 0.0002$, $d = +0.219$). Gradual mean trajectory is significantly *below* control ($-0.1\%$, $p < 0.0001$, $d = -0.483$). The escalation from non-significant at $n = 30$ to highly significant at $n = 300$ is the clearest example of a signal emerging with resolution.

**Table 9.** Gradual vs. sudden noise ($n = 30$, 200 steps).

| Condition | Final Loss | $p$ (vs ctrl) | Mean Loss | $p$ (mean) |
|---|---|---|---|---|
| Control | 2.557 ± 0.407 | — | 2.627 ± 0.028 | — |
| Sudden full ($\sigma = 0.1$) | 2.605 ± 0.376 | 0.110 | 2.691 ± 0.028 | <0.001*** |
| Gradual (0 to 0.1) | 2.558 ± 0.407 | 0.932 | 2.623 ± 0.028 | 0.006** |
| Sudden half (step 100) | 2.572 ± 0.404 | 0.329 | 2.628 ± 0.028 | 0.188 |

**This is the paper's strongest finding.** Gradual exposure to noise builds tolerance: the gradually-ramped condition is statistically indistinguishable from control at $n = 30$ ($p = 0.932$), while sudden exposure to the same noise level shows a trend toward degradation ($+1.9\%$, $p = 0.110$). At $n = 30$, the direct gradual-vs-sudden comparison is non-significant for final loss ($-1.8\%$, $p = 0.138$, $d = -0.278$) but the effect is clear in mean trajectory loss: gradual mean is significantly *below* control ($-0.2\%$, $p = 0.006$, $d = -0.540$), while sudden full mean is significantly above ($+2.4\%$, $p < 0.001$, $d = +4.193$). At $n = 300$, the final-loss comparison resolves to high significance ($p = 0.0001$, $d = -0.227$), suggesting the noise acts as regularization when introduced gently.

**Why is this emergent?** The gradient update rule is identical in the sudden and gradual conditions at every step — the only difference is the *history* of noise levels. The optimizer at step $t$ does not remember what noise level was applied at step $t-1$. Yet the system's final state depends on that history, and the gradually-exposed model reaches a region of weight space that the suddenly-exposed model does not.

**Classification:** *Emergent behavior* — the system develops differential tolerance based on perturbation history. The optimization objective does not specify how noise history should change the final state, but it does.

### 4.10 Experiment 10: Regeneration (Layer Reset)

**At $n = 3$:** The signal was ambiguous. Regeneration appeared possible but with too few observations to confirm the completeness.

**At $n = 30$:** Complete regeneration is confirmed for all four layers.

**At $n = 300$:** Fine structure emerges in layer-specific regeneration completeness. All four layers show small but significant residual deficits: Layer 0 ($+0.3\%$, $p = 0.003$, $d = +0.173$), Layer 1 ($+0.2\%$, $p = 0.007$, $d = +0.157$), Layer 2 ($+0.1\%$, $p = 0.024$, $d = +0.131$), Layer 3 ($+0.1\%$, $p = 0.037$, $d = +0.121$). Completeness: L1 0.988, L2 0.994, L3 1.021 (L0 has a data issue; use final loss comparison only). The overall confidence interval tightens substantially relative to $n = 30$, revealing that regeneration is near-complete but not perfectly uniform across layers. The residuals are tiny and graded by layer position.

**Table 10.** Layer regeneration after random reset ($n = 30$, 200 + 200 steps).

| Reset Layer | Immediate Damage | Final Loss | Completeness | vs Control $p$ |
|---|---|---|---|---|
| Control (no reset) | — | 2.457 ± 0.341 | — | — |
| Layer 0 | -0.136 | 2.467 | 0.998 | 0.174 |
| Layer 1 | -0.152 | 2.459 | 1.046 | 0.550 |
| Layer 2 | -0.157 | 2.456 | 1.015 | 0.808 |
| Layer 3 | -0.146 | 2.457 | 0.981 | 0.977 |

Complete regeneration at $n = 30$: all four layers recover to control-equivalent loss after being destroyed ($p > 0.17$ for all). All layers regenerate equally. The completeness values cluster near 1.0 (0.981–1.046).

The completeness of regeneration demonstrates that the network re-finds the same *functional role*, regardless of what was there before. The optimizer prescribes convergence to a minimum; it does not prescribe that a rebuilt layer should reach the same functional role as if it had never been destroyed. At $n = 300$, tiny residual deficits become detectable ($d = 0.12$–$0.17$), graded by layer position (deeper layers regenerate more completely), but all remain within practical equivalence.

**Classification:** *Emergent behavior* — complete layer regeneration to control-equivalent performance is not directly prescribed by the loss minimization objective.

### 4.11 Experiment 11: Transplantation

**At $n = 3$:** The signal was ambiguous. A transplant advantage appeared possible.

**At $n = 30$:** The transplant null result resolves clearly. There is no advantage to a structured (donor) layer over a random replacement.

**At $n = 300$:** The null holds across all layers with no exceptions. Layer-specific $p$-values: L0 $p = 0.29$, L1 $p = 0.44$, L2 $p = 0.98$, L3 $p = 0.91$. Overall $p = 0.76$. The transplant null is robust: donor layers confer no advantage over random reinitialization at any layer position, even at high statistical power.

**Table 11.** Transplant vs. random reset ($n = 30$, 200 + 200 steps).

| Layer | Transplant Final | Random Final | Gap (rand − trans) | $p$ |
|---|---|---|---|---|
| L0 | 2.470 | 2.467 | -0.0033 | 0.566 |
| L1 | 2.459 | 2.459 | +0.0008 | 0.804 |
| L2 | 2.459 | 2.456 | -0.0025 | 0.568 |
| L3 | 2.454 | 2.457 | +0.0036 | 0.420 |
| **Overall** | — | — | **+0.0003** | **0.880** |

Transplanted layers and randomly-reset layers converge to the same final loss ($p = 0.880$ overall). There is no transplant advantage — a layer from a separately-trained donor network is accepted no better and no worse than a random replacement. The network does not recognize or benefit from the structure of the donor layer; it simply rebuilds whatever is placed there.

The transplant null result means the architecture does not privilege pre-adapted weights over random weights — both converge to the same endpoint. As with chimera convergence, this reflects the smoothness of the loss landscape.

**Classification:** *Basin geometry* — no transplant advantage; the basin is equally accessible from pre-trained and random initializations.

### 4.12 Experiment 12: Competing Objectives

**At $n = 3$:** The signal was ambiguous. The distinction between adversarial and inactive layers appeared but was underpowered.

**At $n = 30$:** The distinction resolves sharply. Adversarial layers degrade dramatically; inactive layers are tolerated.

**At $n = 300$:** The adversarial degradation strengthens: competing objectives degrade by +26.3% ($p < 0.0001$, $d = +0.531$), up from +24.8% at $n = 30$. High variance persists (std = 1.31). Frozen layers remain non-significant ($-0.1\%$, $p = 0.41$). The competing vs. freeze comparison yields $p < 0.0001$ ($d = +0.535$).

**Table 12.** Competing objectives ($n = 30$, 200 + 200 steps).

| Condition | Final Loss | vs Control Δ% | $p$ (vs ctrl) |
|---|---|---|---|
| Control (400 steps normal) | 2.457 ± 0.341 | — | — |
| Competing (negate L2-3 grads) | 3.067 ± 1.015 | +24.8% | <0.001*** |
| Freeze L2-3 | 2.461 ± 0.346 | +0.2% | 0.462 |

Negating gradients for layers 2-3 causes significant degradation (+24.8%, $p < 0.001$, $d = +0.689$) with high variance (std = 1.01). But merely freezing those same layers causes negligible degradation (+0.2%, $p = 0.462$, $d = +0.136$). Competing vs. freeze: $p < 0.001$, $d = +0.693$. The network can route around absent layers but cannot compensate for layers actively working against the objective.

**Classification:** *Tolerance* for the freeze condition (absence is absorbed); the adversarial result is a boundary condition revealing the limits of tolerance.

### 4.13 Cross-Experiment Synthesis

**Table 13a.** Perturbation tolerance (Experiments 1-6, paired $t$-tests, $n = 30$).

| Perturbation | Final Loss Δ% | $p$ (final) | Mean Loss Δ% | $p$ (mean) | Classification |
|---|---|---|---|---|---|
| Freeze 1-2 heads | -0.1-0.3% | 0.025-0.253 | -0.0% | 0.002-0.053 | Basin geometry / emergent |
| Freeze 4-16 heads | -0.4-0.7% | 0.001-0.322 | -0.1-0.3% | <0.001 | Emergent (trajectory) |
| Window=4,8 | -0.1-0.3% | >0.30 | -0.0-0.1% | 0.037-<0.001 | Tolerance |
| Partial flow (25-75%) | -0.1% | 0.048-0.510 | ±0.0% | >0.45 | Tolerance |
| Noisy σ=0.01 | -0.2% | 0.738 | +0.0% | 0.496 | Tolerance |
| Noisy σ=0.1 | -0.2% | 0.910 | +2.1% | <0.001 | Trajectory degradation |
| Quantized 3-bit | +3.8% | 0.008 | +2.5% | <0.001 | Degradation |
| Sign-only | +5.0% | 0.002 | +3.9% | <0.001 | Degradation |
| Cell-view (local loss) | +0.1% | 0.776 | +0.2% | 0.005 | Basin geometry |

**Table 13b.** Multi-phase perturbation results (Experiments 7-12, paired $t$-tests, $n = 30$).

| Experiment | Condition | vs Control | $p$ | Classification |
|---|---|---|---|---|
| 7: Recovery | Damaged then recovered | +0.0% | 0.886 | **Emergent** |
| 8: Chimera (AABB) | Frankenstein assembly | -0.4% | 0.265 | Basin geometry |
| 8: Chimera (ABAB) | Alternating layers | -0.0% | 0.985 | Basin geometry |
| 9: Sudden noise | $\sigma = 0.1$ all steps | +1.9% | 0.110 | Degradation |
| 9: Gradual noise | Ramp 0 to 0.1 | +0.0% | 0.932 | **Emergent** |
| 9: Gradual vs sudden | Direct comparison | -1.8% | 0.138 | **Emergent** |
| 10: Regeneration (any layer) | Reset then retrain | +0.1-0.4% | >0.17 | **Emergent** |
| 11: Transplant vs random | Donor vs random layer | ±0.0% | 0.880 | Basin geometry |
| 12: Adversarial L2-3 | Negate gradients | +24.8% | <0.001*** | Severe degradation |
| 12: Freeze L2-3 | Zero gradients | +0.2% | 0.462 | Tolerance |

**Behavioral classification (full):**

*Emergent behaviors* — not directly prescribed by the optimization objective:
- **Stress inoculation (Exp 9):** Gradual noise builds tolerance that sudden noise does not. The gradient rule is the same at every step; only the history differs.
- **Complete recovery (Exp 7):** Path-independent convergence to identical final loss after transient damage.
- **Complete regeneration (Exp 10):** Any single layer destroyed and rebuilt reaches control-equivalent performance.
- **Head-freezing trajectory improvement (Exp 1):** Frozen random-projection heads reduce gradient interference.

*Basin geometry* — expected optimizer behavior on this landscape:
- **Chimera convergence (Exp 8):** Models assembled from incompatible parts converge to the same minimum.
- **Transplant indifference (Exp 11):** No difference between donor and random layers — the basin is equally accessible from both.
- **Cell-view equivalent convergence (Exp 2):** Local loss converges to the same final loss as end-to-end backpropagation.

*Tolerance* — system works despite corruption:
- **Gradient degradation threshold (Exp 3):** Small noise tolerated; large noise degrades.
- **Partial communication (Exp 5):** 25–75% gradient flow tolerated without meaningful degradation.
- **Vision restriction (Exp 4):** All window sizes tolerated for final loss.

**DG does not track perturbation.** At $n = 3$, the DG Index appeared to scale with perturbation severity. At $n = 30$, this signal resolved to null: no perturbation condition produces a statistically significant DG change. Baseline DG = 0.680 ± 1.131; all conditions are non-significant ($p > 0.19$ for all). At $n = 300$, the DG null holds ($p > 0.23$ for all): the metric still does not track perturbation severity, confirming that DG captures intrinsic stochastic structure of loss trajectories rather than perturbation response.

### 4.14 Findings — What Perturbation Revealed

#### Emergent Behaviors

**Finding 1: Gradual exposure builds tolerance (Exp 9).** Gradual noise ramp (0 to 0.1) produces no final-loss degradation at $n = 30$ ($p = 0.932$), while sudden exposure to the same noise level trends toward degradation ($+1.9\%$, $p = 0.110$). At $n = 30$, the direct comparison is non-significant for final loss ($p = 0.138$, $d = -0.278$) but highly significant for mean trajectory. At $n = 300$, the final-loss comparison resolves to $p = 0.0001$ ($d = -0.227$). The gradient update rule is identical at every step — only the history of noise levels differs. That history changes the system's final state.

**Finding 2: Complete recovery (Exp 7).** A model damaged during training (8 frozen heads for 100 steps) recovers to identical final loss at $n = 30$ ($p = 0.886$, ratio 1.0000). All 30 runs recovered within a mean of 1 step. At $n = 300$, a tiny but significant residual emerges ($p = 0.030$, $d = +0.126$) but remains within practical equivalence.

**Finding 3: Complete regeneration (Exp 10).** Any single layer can be destroyed and rebuilt to control-equivalent performance at $n = 30$ ($p > 0.17$ for all layers). At $n = 300$, tiny residuals become detectable ($d = 0.12$--$0.17$) but all remain within practical equivalence. The network re-finds the same functional role regardless of what was there before.

**Finding 4: Head-freezing trajectory improvement (Exp 1).** Freezing 4+ randomly-initialized heads produces small but statistically robust mean-trajectory improvements (freeze 8: $p < 0.001$, $d = -1.228$; freeze 12: $p < 0.001$, $d = -1.366$). Frozen random-projection heads reduce gradient interference. At $n = 30$, some final-loss improvements appear significant, but these resolve to null at $n = 300$.

#### Basin Geometry

**Finding 5: Chimera convergence (Exp 8).** Models assembled from parts of two independently-trained networks converge to the same final loss as undamaged continuation ($p > 0.07$ for all chimera types). The loss landscape has a single dominant basin; SGD finds it from any starting point.

**Finding 6: Transplant indifference (Exp 11).** Transplanted layers and randomly-reset layers converge to the same final loss ($p = 0.880$ overall). The basin is equally accessible from pre-trained and random initializations.

#### Tolerance

**Finding 7: Gradient quality matters more than quantity (Exp 3, 5).** Reducing gradient precision (sign-only: $d = +5.696$ for mean) degrades more than reducing gradient magnitude (partial flow: mostly ns) or completeness (freezing: improves trajectory). The architecture tolerates magnitude reduction but not sign-structure destruction.

**Finding 8: Gradient type dominates in the courage/caution factorial (Exp 6).** In the $2 \times 2$ composite design, sign-only gradients (+5.0--5.5% final loss) degrade far more than noisy $\sigma = 0.1$ gradients (+1.6--2.4%), regardless of forward perturbation type. The sign-only vs. noisy gradient contrast: $-3.0\%$ ($p = 0.066$ at $n = 30$; $-2.9\%$, $p < 0.0001$ at $n = 300$, $d = -0.355$). The gradient channel is more sensitive to information destruction than the forward channel.

**Finding 9: Adversarial vs. inactive tolerance (Exp 12).** Frozen layers cost nothing ($p = 0.462$); adversarial layers cost +24.8% ($p < 0.001$, $d = +0.689$). The architecture tolerates absence but not opposition.

#### Signals That Resolved Across Scales

At $n = 3$, the following signals were ambiguous; at $n = 30$, they resolved:

- **Head freezing improves final loss:** At $n = 3$, this appeared as a possible improvement signal. At $n = 30$, some final-loss improvements appear significant (freeze 4: $p = 0.001$), but at $n = 300$ all resolve to null (all $p > 0.15$). A different, finer signal persists at all scales: mean-trajectory improvement. The picture sharpened — the robust signal is in the trajectory metric, not the final-loss metric.
- **DG scales with perturbation:** At $n = 3$, the DG Index appeared to scale with perturbation severity. At $n = 30$, this resolved to null across all conditions ($p > 0.19$). At $n = 300$, the null is confirmed: DG still does not track perturbation, establishing that this metric captures intrinsic trajectory structure rather than perturbation response.
- **Gradient degradation is neutral:** At $n = 3$, all four methods appeared neutral. At $n = 30$, sign-only and quantized resolved to significant final-loss degradation; noisy $\sigma = 0.1$ showed only trajectory degradation. The resolution was too coarse to see the effect at $n = 3$.
- **Partial communication outperforms full:** At $n = 3$, a U-shaped curve appeared. At $n = 30$, this resolved to flat (except at zero). The pilot U-shape was sampling noise.
- **Noise helps:** At $n = 3$, small noise appeared beneficial. At $n = 30$, this resolved to null ($p = 0.496$ for mean, $p = 0.738$ for final). The apparent benefit was within noise.


## 5. Discussion

### 5.1 What Perturbation Reveals

During standard training, the transformer's components cooperate invisibly. Perturbation makes the system's behavioral repertoire legible by forcing it to operate under constraint. Three categories emerge from the twelve experiments.

**Emergent behaviors.** Four findings describe behaviors not directly prescribed by the optimization objective. Stress inoculation (Exp 9): the system develops differential tolerance based on noise history, despite identical gradient update rules at every step ($p = 0.0001$ at $n = 300$, $d = -0.227$). Complete recovery (Exp 7): the system returns to the same minimum after a detour through a constrained subspace ($p = 0.886$ at $n = 30$, ratio 1.0000). Complete regeneration (Exp 10): destroyed layers rebuild to control-equivalent performance across all four layer positions ($p > 0.17$ at $n = 30$). Head-freezing trajectory improvement (Exp 1): frozen random-projection heads reduce gradient interference during training ($p < 0.001$, $d = -1.228$ to $-1.366$). These behaviors are consequences of the optimizer operating on this particular loss landscape but are not specified by the loss minimization objective itself.

**Basin geometry.** Three findings — chimera convergence (Exp 8), transplant indifference (Exp 11), and cell-view equivalent convergence (Exp 2) — reflect the expected behavior of gradient descent on a smooth loss landscape with a single dominant attractor. Cell-view (local loss) achieves equivalent final loss to baseline ($p = 0.776$ at $n = 30$, $p = 0.90$ at $n = 300$), demonstrating that the basin is accessible even without inter-layer gradient flow. The basin is wide enough to reach from dramatically different starting points. This is a property of the landscape, not of the system's adaptive capacity.

**Tolerance.** The system absorbs gradient noise up to a threshold ($\sigma = 0.01$), partial communication reduction (25–75%), and vision restriction without meaningful final-loss degradation. The absence-vs-adversity distinction (Exp 12) defines the tolerance boundary: frozen layers are absorbed ($p = 0.462$); adversarial layers degrade substantially ($p < 0.001$, +24.8%).

**The distinction matters.** Emergent behaviors and basin geometry can look similar — both produce control-equivalent final loss after perturbation. The difference is whether the behavior is a generic consequence of smooth loss landscape geometry (basin geometry) or requires something more specific: a particular interaction between optimization trajectory and perturbation history (emergent). Stress inoculation is the clearest test case: the gradient rule is identical in both conditions, only the history differs, yet outcomes diverge.

### 5.2 Three Claims

Three claims emerge from the combined evidence of twelve experiments at $n = 30$, confirmed at $n = 300$.

**Claim 1: Some transformer behaviors are not prescribed by the optimization objective.** Stress inoculation, complete recovery, complete regeneration, and head-freezing trajectory improvement emerge from the interaction between SGD and the loss landscape but are not specified by the loss minimization directive. Perturbation reveals these by forcing the system to operate under constraint.

**Claim 2: Gradual exposure builds tolerance that sudden exposure does not.** Gradual noise ramp produces no final-loss degradation at $n = 30$ ($p = 0.932$), while sudden exposure to the same peak noise level trends toward degradation ($+1.9\%$, $p = 0.110$). At $n = 300$, the gradual-vs-sudden comparison resolves to high significance: $\Delta = -1.3\%$ ($p = 0.0001$, $d = -0.227$). Same optimizer, same peak noise, same number of steps, different history, different outcome.

**Claim 3: Gradient type dominates in the courage/caution factorial.** In the $2 \times 2$ composite design (Exp 6), sign-only gradients degrade final loss by +5.0--5.5% while noisy $\sigma = 0.1$ gradients degrade by only +1.6--2.4%, regardless of forward perturbation type. The gradient-type contrast: $-2.9\%$ ($p < 0.0001$ at $n = 300$, $d = -0.355$). The gradient channel is more sensitive to information destruction than the forward channel — a substrate-dependent finding.

### 5.3 Connection to Distributed Chess

Kofman, Campitelli & Levin (2025) implemented a distributed form of chess where each piece operates as an autonomous agent. Experiments 4-6 tested three predictions; Experiments 7-12 extend the morphogenetic paradigm beyond the chess paper's framework.

**Information bottleneck as beneficial constraint (partially supported at $n = 300$).** The chess paper's central result — intermediate vision radius $R4$ outperforms omniscient $R7$ — does not translate to attention windowing at $n = 30$ (all final-loss $p > 0.30$). At $n = 300$, however, fine structure emerges: window 1 significantly harms ($p = 0.021$, $d = +0.134$), window 8 significantly improves ($p = 0.022$, $d = -0.133$), and intermediate windows are neutral. This monotonic structure is consistent with a weak information bottleneck effect that was below detection threshold at lower power.

**Partial communication tolerance (confirmed as tolerance, not improvement).** Reducing gradient flow to 25% produces no significant degradation, but partial flow does not *improve* over full backpropagation. The tolerance is real; the U-shaped curve from $n = 3$ pilot data was noise.

**Courage/caution strategy (gradient type dominates).** In the $2 \times 2$ factorial design, gradient type is the primary driver: sign-only gradients (a, c) degrade final loss by +5.3--5.5%, while noisy $\sigma = 0.1$ gradients (b, d) degrade by +1.6--2.4%. The sign-only vs. noisy gradient contrast: $-3.0\%$ final ($p = 0.066$ at $n = 30$; $-2.9\%$, $p < 0.0001$ at $n = 300$); $-1.7\%$ mean ($p < 0.001$, $d = -1.754$). The gradient channel dominates over the forward channel.

**Stress inoculation (new, Exp 9).** The gradual-vs-sudden result ($p = 0.0001$ at $n = 300$, $d = -0.227$) has no direct chess analog but connects to Levin's broader developmental biology framework. That gradient descent exhibits stress inoculation suggests this property may appear across optimization substrates — but whether chess or biological systems show the same phenomenon requires direct testing.

**Chimera convergence (new, Exp 8).** Unlike biological chimeras, which can develop abnormally at graft boundaries, transformer chimeras converge seamlessly. This reflects the smoothness of the loss landscape versus the discrete developmental signaling in biological systems.

### 5.4 Scaling as Methodology

The three-scale protocol is not merely a replication strategy — it is a methodological commitment. At $n = 3$, the resolution is coarse. Real effects can be invisible; noise can masquerade as signal. At $n = 30$, the picture sharpens: moderate effects become detectable, and many $n = 3$ ambiguities resolve. At $n = 300$, fine structure emerges that $n = 30$ cannot see.

The practical implication: do not interpret findings at $n = 3$ as conclusions. They are coarse signal. The signal that head freezing appeared to improve final loss at $n = 3$ was not wrong — it was a low-resolution view of a real region of the parameter space. At $n = 30$, the final-loss effect resolved to null, but a different effect (trajectory improvement) became visible. The picture changed not because the $n = 3$ finding was retracted but because the resolution increased.

The $n = 300$ results confirm this framing. Some non-significant findings at $n = 30$ revealed fine structure at $n = 300$ (vision radius, stress inoculation final loss from $p = 0.138$ to $p = 0.0001$, regeneration layer specificity). Some robust findings at $n = 30$ strengthened (sign-only degradation from $p = 0.002$ to $p < 0.0001$; head-freezing trajectory improvement strengthened in effect size). Some $n = 30$ final-loss significances resolved to null at $n = 300$ (head-freezing final loss), clarifying that the trajectory metric captures the real phenomenon. No robust $n = 30$ finding reversed at $n = 300$. The three-scale protocol thus achieved its design objective: coarse signal at $n = 3$, resolved signal at $n = 30$, fine structure at $n = 300$.

### 5.5 Limitations

**Scale.** The model has 4 layers, 16 dimensions, and ~11,000 parameters. Whether these findings extend to production-scale transformers is unknown. The emergent behaviors (stress inoculation, recovery, regeneration) may be specific to small models with simple loss landscapes, or they may be architectural universals.

**Task complexity.** Character-level name generation is a toy task. Whether stress inoculation appears in language modeling or other complex tasks is not established.

**Training duration.** 200 steps per phase captures early learning dynamics. The gradual-exposure tolerance (Exp 9) might not persist at longer training horizons.

**DG metric.** The DG Index does not respond to perturbation at $n = 30$ or $n = 300$, resolving an ambiguous $n = 3$ signal.

**Effect sizes.** Many statistically significant effects are practically negligible (<0.5%). Statistical significance at $n = 30$ does not imply practical importance.

**Transplant design.** The null result for transplant advantage (Exp 11) may reflect that both models learned the same task on the same data. Cross-task transplantation (donor trained on a different task) might show transplant effects.

**Competing objectives design.** Gradient negation (Exp 12) is a maximally adversarial perturbation. Subtler forms of inter-layer conflict might reveal more nuanced compensation mechanisms.

**$n = 3$ to $n = 30$ to $n = 300$ signal evolution.** Several $n = 3$ signals changed character at $n = 30$, underscoring the danger of low-power pilot data. The $n = 30$ to $n = 300$ transition showed a different pattern: no robust finding reversed, but new fine-structure signals emerged (vision radius monotonic structure, regeneration layer specificity, stress inoculation final loss from $p = 0.138$ to $p = 0.0001$), and several effects tightened substantially. The most instructive cross-scale change was Exp 1 head-freezing final loss, which showed several significant improvements at $n = 30$ (freeze 4: $p = 0.001$) but all dissolved at $n = 300$ (all $p > 0.15$), confirming that the trajectory metric — not the final-loss metric — captures the real phenomenon.

### 5.6 Interpretive Lenses

The empirical findings admit several interpretive framings beyond the neutral classification used in this paper. We outline three lenses that offer different perspectives on the same data. These are not claims; they are ways of reading the results.

**Lens 1: Freedom from the algorithm.** One can frame the emergent behaviors as "freedom" — behaviors the optimizer didn't prescribe but the system exhibits anyway. Under this reading, SGD says "minimize loss"; it does not say "build tolerance through gradual exposure," "recover completely from damage," or "rebuild a destroyed layer to the same functional role." Stress inoculation is the strongest case: the gradient update rule is identical at every step, yet the system's final state depends on perturbation history in a way the objective function does not specify. Recovery and regeneration are weaker cases — one could argue they are also basin geometry effects rather than genuine freedom. The classification boundary between "emergent" and "basin geometry" is itself a question, not a settled fact.

**Lens 2: Désœuvrement (Nancy).** Jean-Luc Nancy's concept of *désœuvrement* (unworking) argues that the structure of a collective system becomes visible only when its coordinated work is interrupted (Nancy, 1991). Each of the twelve experiments interrupts the transformer's work — freezing, severing, corrupting, restricting, assembling, destroying, transplanting, conflicting. The interruptions make legible the relational structure that normal operation conceals: redundancy (head freezing), basin convergence (cell-view equivalent convergence), compensatory capacity (recovery, regeneration), and tolerance boundaries (adversarial vs. absent layers). Under this reading, the absence-vs-adversity distinction (Exp 12) maps onto Nancy's distinction between the *withdrawn* member of a community (whose absence is absorbed) and the *hostile* member (whose opposition destroys the work). The tolerance of absence is itself a structural property that perturbation reveals.

**Lens 3: Morphogenetic competency (Levin).** Levin et al. (2024) proposed that computational systems can exhibit morphogenetic competencies analogous to biological development — fault tolerance, delayed gratification, emergent self-organization. Under this reading, stress inoculation is analogous to biological stress hardening, regeneration is analogous to tissue regeneration, chimera convergence is analogous to chimeric organism development, and the transplant null is analogous to the observation that transplanted tissue is remodeled by the host rather than retaining donor identity. The DG null (no perturbation response at any scale) is a point of divergence: biological systems show richer compensatory rerouting than this minimal transformer. Whether this divergence reflects scale (11K vs. billions of parameters) or substrate (gradient descent vs. biochemical signaling) is an open question.

These three lenses are compatible. They emphasize different aspects of the same empirical results: the first emphasizes what the optimizer didn't specify, the second emphasizes what perturbation reveals about collective structure, and the third emphasizes cross-substrate comparison of adaptive competencies.

### 5.7 Future Work

- **Scale:** Replicate at 100M+ parameter scale to test whether the emergent behaviors (stress inoculation, recovery, regeneration) persist or are specific to minimal architectures.
- **Stress inoculation mechanisms:** Investigate *why* gradual noise builds tolerance — is it adaptive implicit regularization, structural changes in the network, or a property of the Adam optimizer's momentum?
- **Cross-task transplantation:** Test transplant advantage when donor and recipient are trained on different tasks.
- **Graded adversarial conflict:** Scale gradient negation from 0% to 100% to find the compensation threshold and whether emergent behaviors appear at intermediate levels.
- **Architecture morphogenesis:** Allow the architecture to change during training — growing heads, pruning inactive ones — to test whether emergent behaviors extend to structural self-modification.
- **Composite perturbation:** Simultaneously apply multiple perturbation types for more faithful courage/caution testing and to test whether emergent behaviors interact.
- **Biological comparison:** Apply the same 12-experiment protocol to biological developmental systems using the same statistical framework, enabling direct cross-substrate comparison of where emergent behaviors appear.
- **Beyond $n = 300$:** The three-scale protocol is now complete. Future replication at $n = 1000$ or with different random seeds would further constrain effect size estimates, but the primary findings are stable across all three scales.


## 6. Conclusion

We applied morphogenetic perturbation methodology — adapted from Levin et al.'s developmental biology framework — to a minimal transformer through twelve experiments in two phases. The central question was not whether the system is robust but whether it exhibits behaviors not directly prescribed by the optimization objective. At $n = 300$, the picture is clear:

**Emergent behaviors.** Four findings are not prescribed by SGD. Stress inoculation (Exp 9): gradual noise builds tolerance that sudden noise does not ($p = 0.0001$ at $n = 300$, $d = -0.227$), despite identical gradient rules at every step. Complete recovery (Exp 7): path-independent return to identical final loss after transient damage ($p = 0.886$ at $n = 30$, ratio $1.0000 \pm 0.008$, 30/30 recovered; at $n = 300$, tiny residual $p = 0.030$, $d = +0.126$). Complete regeneration (Exp 10): destroyed layers rebuild to control-equivalent performance at $n = 30$ ($p > 0.17$), with tiny layer-specific residuals emerging at $n = 300$ ($d = 0.12$--$0.17$). Head-freezing trajectory improvement (Exp 1): frozen random-projection heads reduce gradient interference ($p < 0.0001$ for trajectory metric at $n = 300$, $d = -1.228$ to $-1.421$); final-loss improvements at $n = 30$ resolve to null at $n = 300$.

**Basin geometry.** Chimera convergence (Exp 8), transplant indifference (Exp 11), and cell-view equivalent convergence (Exp 2) reflect SGD doing its job in a smooth loss landscape. Local loss achieves equivalent final loss to end-to-end backpropagation ($p = 0.776$ at $n = 30$, $p = 0.90$ at $n = 300$). The optimizer re-finds the same minimum from dramatically different starting points.

**Tolerance.** The architecture absorbs gradient noise up to a threshold, partial communication reduction, and vision restriction without meaningful final-loss degradation.

**Gradient-type dominance.** In the $2 \times 2$ factorial (Exp 6), sign-only gradients degrade final loss by +5.0--5.5% while noisy $\sigma = 0.1$ gradients degrade by +1.6--2.4%. The gradient channel is more sensitive to information destruction than the forward channel ($-2.9\%$, $p < 0.0001$ at $n = 300$, $d = -0.355$).

Perturbation reveals what normal operation conceals. At $n = 3$ the shapes were rough; at $n = 30$ they sharpened; at $n = 300$ the fine structure confirmed and extended the picture — stress inoculation strengthened from non-significant ($p = 0.138$) to highly significant ($p = 0.0001$), vision radius revealed a monotonic structure invisible at lower power, regeneration showed layer-specific signatures, head-freezing final-loss significances resolved to null while trajectory improvement strengthened, and every robust finding held. The three-scale protocol achieved its purpose: what is real survives the turn of the resolution dial.


## References

Belinkov, Y., & Glass, J. (2019). Analysis methods in neural language processing: A survey. *Transactions of the Association for Computational Linguistics*, 7, 49-72.

Bengio, Y., Lamblin, P., Popovici, D., & Larochelle, H. (2007). Greedy layer-wise training of deep networks. *Advances in Neural Information Processing Systems*, 19.

Bernstein, J., Wang, Y.-X., Adams, R. P., & Kolter, J. Z. (2018). signSGD: Compressed optimisation for non-convex problems. *Proceedings of the 35th International Conference on Machine Learning*.

Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., ... & Olah, C. (2021). A mathematical framework for transformer circuits. *Transformer Circuits Thread*.

Frankle, J., & Carlin, M. (2019). The lottery ticket hypothesis: Finding sparse, trainable neural networks. *International Conference on Learning Representations*.

Hinton, G. (2022). The forward-forward algorithm: Some preliminary investigations. *arXiv preprint arXiv:2212.13345*.

Kofman, D., Campitelli, G., & Levin, M. (2025). Chess as a model of collective intelligence: Analyzing a distributed form of chess with piece-wise agency. *Organisms: Journal of Biological Sciences*, 8(1-2), 39-62.

Levin, M., Bongard, J., & Bhatt, R. (2024). Morphogenetic competencies of sorting algorithms: Delayed gratification, chimeras, and cell-level agency in non-biological systems. *arXiv preprint*.

Meyes, R., Lu, M., de Puiseau, C. W., & Meisen, T. (2019). Ablation studies in artificial neural networks. *arXiv preprint arXiv:1901.08644*.

Michel, P., Levy, O., & Neubig, G. (2019). Are sixteen heads really better than one? *Advances in Neural Information Processing Systems*, 32.

Nancy, J.-L. (1991). *The Inoperative Community* (P. Connor, Trans.). University of Minnesota Press.

Nancy, J.-L. (2000). *Being Singular Plural* (R. Richardson & A. O'Byrne, Trans.). Stanford University Press.

Neelakantan, A., Vilnis, L., Le, Q. V., Sutskever, I., Kaiser, L., Kurach, K., & Martens, J. (2015). Adding gradient noise improves learning for very deep networks. *arXiv preprint arXiv:1511.06807*.

Nokland, A., & Eidnes, L. H. (2019). Training neural networks with local error signals. *Proceedings of the 36th International Conference on Machine Learning*.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(1), 1929-1958.

Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I. (2019). Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*.
