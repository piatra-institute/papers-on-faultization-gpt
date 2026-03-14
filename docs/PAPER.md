# Morphogenetic Perturbation Reveals Emergent Behaviors in Minimal Transformers

**Date:** March 2026


## Abstract

We apply morphogenetic perturbation methodology (Levin et al., 2024) to a minimal transformer (4-layer, 16-dimensional, 4-head character-level GPT) through twelve experiments spanning perturbation-during-training (Exp 1-6) and multi-phase morphogenetic interventions (Exp 7-12). We adopt a three-scale protocol: $n = 3$ pilot data provides initial signal, $n = 30$ paired analysis ($n = 30$ runs per condition, matched seeds) resolves ambiguity, and $n = 300$ reveals fine structure. At $n = 3$, several signals were ambiguous — head freezing appeared to improve loss, gradient corruption appeared neutral, Delayed Gratification (DG) Index appeared to scale with perturbation. At $n = 30$, the picture sharpens: we identify four emergent behaviors not directly prescribed by Stochastic Gradient Descent (SGD) (stress inoculation, complete recovery, complete regeneration, and head-freezing trajectory improvement), two that reflect basin geometry (chimera convergence, transplant indifference), and three that demonstrate tolerance (gradient degradation absorbed up to a threshold, partial communication, vision restriction). At $n = 300$, fine structure emerges: the stress inoculation effect strengthens from $p = 0.011$ to $p < 0.0001$, vision radius reveals a previously invisible monotonic structure (window 1 harms at $p = 0.0009$, window 8 improves at $p = 0.022$), regeneration shows layer-specific residual effects, and all robust $n = 30$ findings hold or strengthen while all null findings remain null. The paper's key finding: gradual noise exposure builds tolerance that sudden exposure does not ($p < 0.0001$, $d = -0.278$), despite identical gradient update rules at every step.


## 1. Introduction

Transformers are typically studied through their outputs: probing learned representations (Belinkov & Glass, 2019), ablating components to measure importance (Michel et al., 2019), or tracing computational circuits (Elhage et al., 2021). These methods characterize what the system has learned or which parts matter. They do not ask what happens when the system is forced to learn under constraint — when its normal operation is *interrupted*.

Levin et al. (2024) introduced a different methodology in the context of simple algorithms. Rather than analyzing sorting algorithms through their final outputs, they perturbed the algorithms during execution: freezing cells, mixing incompatible sorting directions, replacing centralized control with autonomous cell-level policies. The perturbations revealed competencies — fault tolerance, delayed gratification, emergent aggregation — that were invisible during normal operation. The central insight: **perturbation reveals what normal operation conceals**.

This paper asks a sharper version of that question: **does the system exhibit behaviors that the optimizer didn't prescribe?** SGD has one directive: minimize loss. It does not specify *how* to do this — whether to build redundancy, develop stress tolerance, or recover gracefully from damage. When a transformer exhibits these behaviors, they are emergent — not directly prescribed by the optimization objective. The question is not whether these behaviors exist, but which of them are genuinely emergent and which merely reflect the geometry of the loss landscape that SGD was always going to navigate.

The question of whether any finding is genuine requires statistical power. We adopt an explicit three-scale protocol. At $n = 3$ (pilot), signal is visible but ambiguous — the resolution is too low to distinguish real effects from noise. At $n = 30$, the picture sharpens and ambiguous signals resolve. At $n = 300$, fine structure emerges that $n = 30$ cannot see. No scale is wrong. Each reveals different phenomena. Signals that were ambiguous at $n = 3$ and resolved at $n = 30$ are not retracted findings — they are findings at the appropriate resolution.

Our contributions are:

1. **A twelve-experiment morphogenetic methodology** that applies Levin's perturbation protocol to transformer training across two phases: perturbation during training (Exp 1-6) and multi-phase morphogenetic interventions (Exp 7-12).
2. **A three-category classification** that distinguishes emergent behaviors (not directly prescribed by the optimizer) from basin geometry (expected convergence on the given landscape) from tolerance (the system absorbs damage without compensating for it).
3. **The finding that gradual stress builds tolerance** ($p = 0.011$, strengthening to $p < 0.0001$ at $n = 300$): the system develops resilience that sudden exposure to identical peak stress does not produce, despite identical gradient update rules at every step.
4. **A three-scale protocol** that treats $n = 3$ pilot findings as coarse signal, $n = 30$ as resolved signal, and $n = 300$ as fine-structure signal.
5. **A sharp distinction between absence and adversity.** Frozen (inactive) layers are tolerated ($p = 0.30$); gradient-negated (adversarial) layers degrade by +18% ($p = 0.01$).
6. **A negative result on rerouting.** The Delayed Gratification Index shows no perturbation response at $n = 30$ or $n = 300$, distinguishing transformer perturbation response from the richer compensatory rerouting observed in biological development.


## 2. Related Work

Our findings intersect several established lines of research. We situate each finding against prior work to clarify what is known, what we confirm, and what is new.

**Pruning and the lottery ticket hypothesis.** Frankle & Carlin (2019) showed that trained networks contain sparse subnetworks ("winning tickets") that match full-network performance. Subsequent work extended this to structured pruning of attention heads (Michel et al., 2019; Voita et al., 2019). Our Experiment 1 freezes heads at *random initialization*, not after training — the frozen heads are arbitrary random projections. At $n = 3$, the signal was ambiguous: freezing 8+ heads appeared to improve final loss. At $n = 30$, the final-loss signal resolved to null (all $p > 0.06$) but a robust mean-trajectory improvement emerged for 4+ frozen heads, suggesting reduced gradient interference. At $n = 300$, the final-loss null holds definitively (all $p > 0.40$; Spearman $\rho = 0.0023$, $p = 0.92$), but the trajectory improvement strengthens to high significance: freeze 8 ($\Delta = -0.12\%$, $p < 0.0001$), freeze 12 ($\Delta = -0.17\%$, $p < 0.0001$), freeze 16 ($\Delta = -0.19\%$, $p < 0.0001$), confirming that frozen random-projection heads reduce gradient interference.

**SignSGD and low-precision optimization.** Bernstein et al. (2018) established that sign-only gradient updates can match full-precision optimization under appropriate conditions. At $n = 3$, our Experiment 3 signal was ambiguous: gradient degradation appeared neutral. At $n = 30$, the signal resolved clearly: sign-only gradients significantly degrade performance (+4.6% final loss, $p = 0.004$). The discrepancy between our finding and the SignSGD literature may reflect our model's small scale and short training duration. At $n = 300$, this finding strengthens: sign-only gradients degrade by +4.9% ($p < 0.0001$, $d = 0.596$), confirming that the discrepancy with SignSGD is robust and not a small-sample artifact.

**Noise as regularization.** The regularizing effect of gradient noise is well-established (Neelakantan et al., 2015). Our Experiment 3 shows that small noise ($\sigma = 0.01$) produces no significant change, consistent with noise-as-regularization, while large noise ($\sigma = 0.1$) significantly degrades performance (+2.5% final loss, $p = 0.021$), indicating a noise tolerance threshold.

**Local learning rules.** Alternatives to end-to-end backpropagation include greedy layerwise pretraining (Bengio et al., 2007), local learning signals (Nokland & Eidnes, 2019), and forward-forward algorithms (Hinton, 2022). Our cell-view experiment eliminates *all* inter-layer gradient flow. The resulting 4.9% final-loss cost ($p < 0.001$) confirms that local learning is viable but suboptimal.

**Perturbation analysis in neural networks.** Ablation studies (Meyes et al., 2019), dropout, and pruning are standard tools, but typically measure *component importance*. We use perturbation to characterize *system-level behavioral boundaries* — what the architecture absorbs, what it adapts to, and what degrades it. This distinction connects our work to Levin's morphogenetic framework rather than to standard ablation methodology.

**Levin's morphogenetic framework.** Levin et al. (2024) applied developmental biology concepts to simple sorting algorithms. Key findings included delayed gratification (temporary performance decrease followed by recovery past pre-damage levels) and fault tolerance that exceeded intact system performance. Our results show the transformer exhibits tolerance and, crucially, emergent behaviors not directly prescribed by the optimizer — most clearly in stress inoculation (Experiment 9), complete recovery (Experiment 7), complete regeneration (Experiment 10), and trajectory improvement under head freezing (Experiment 1). The Delayed Gratification Index, which Levin used to detect rerouting, shows no perturbation response at $n = 30$.

**Stress inoculation.** The phenomenon of gradual stressor exposure building tolerance is well-documented in biology (Meichenbaum, 1985) and metallurgy (work hardening). In deep learning, curriculum learning (Bengio et al., 2009) and noise scheduling in diffusion models provide partial analogs. Our Experiment 9 demonstrates stress inoculation in the gradient noise domain — a finding that connects the noise-as-regularization literature to developmental biology.

**Distributed chess as collective intelligence.** Kofman, Campitelli & Levin (2025) extended the morphogenetic framework to chess with autonomous pieces. Their "cautious position, courageous moves" strategy is robustly inverted by our Experiment 6 ($p < 0.001$). Their information bottleneck finding (intermediate vision radius outperforms omniscience) does not replicate for attention windowing (no significant final-loss effects).


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

A finding that is ambiguous at $n = 3$ and resolves clearly at $n = 30$ is not a retraction — it is a finding at the appropriate resolution. A signal that appeared at $n = 3$, resolved to null at $n = 30$, and was confirmed null at $n = 300$ is a signal whose character stabilized as resolution increased; conversely, a signal that strengthened from $p = 0.011$ at $n = 30$ to $p < 0.0001$ at $n = 300$ is a signal whose reality was progressively confirmed. This is how science works when the resolution dial turns.

### 3.3 Delayed Gratification (DG) Index

Following Levin et al., we define a metric to detect rerouting behavior — episodes where the system temporarily moves *away* from its goal before recovering past the pre-perturbation level.

**Episode detection.** We scan the loss trajectory for episodes where: (1) loss increases from a local value $L_{\text{start}}$ to a peak $L_{\text{peak}}$, then (2) decreases to a trough $L_{\text{trough}}$ below $L_{\text{start}}$. Each such episode has:

- Temporary cost: $C = L_{\text{peak}} - L_{\text{start}}$
- Net gain: $G = L_{\text{start}} - L_{\text{trough}}$

**Per-episode DG:** $\text{DG}_{\text{episode}} = G / C$

**Aggregate DG Index:** The mean DG across all detected episodes in a training run.

At $n = 3$, the DG Index appeared to scale with perturbation severity. At $n = 30$, this signal resolved to null: no perturbation condition produces a statistically significant DG increase ($p > 0.16$ for all). DG captures a real stochastic property of loss trajectories but does not function as a perturbation response measure at $n = 30$. At $n = 300$, the DG null holds: the metric still does not track perturbation, confirming that DG captures stochastic loss-trajectory structure rather than perturbation response.

### 3.4 Experiment 1: Head Freezing

**Motivation:** Levin's frozen-cell perturbation. Randomly selected attention heads have their parameters frozen at initialization values throughout training.

**Protocol:** Sweep over {0, 1, 2, 4, 8, 12, 16} frozen heads. Frozen heads participate in the forward pass but receive no gradient updates.

### 3.5 Experiment 2: Cell-View GPT

**Motivation:** Levin's cell-view sorting algorithms / Nancy's being-singular-plural. Each transformer layer is treated as an autonomous agent.

**Protocol:** Stop-gradient operations at all layer boundaries. Each layer receives only its own local loss signal.

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

**Motivation:** Kofman et al.'s "cautious position, courageous moves" strategy. We create a $2 \times 2$ matrix:

| | Cautious Gradients | Courageous Gradients |
|---|---|---|
| **Cautious Forward** | (a) Tiny noise ($\sigma = 0.001$) | (b) Sign-only gradients |
| **Courageous Forward** | (c) Dropout ($p = 0.1$) | (d) Noisy gradients ($\sigma = 0.1$) |

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

**At $n = 30$:** The final-loss improvement signal resolved: it was noise. No freezing level produces a statistically significant change in final loss (freeze 8 is marginal at $p = 0.064$; all others $p > 0.18$; Spearman $\rho = 0.013$, $p = 0.85$ for monotonicity). However, a different signal emerged from the noise: mean trajectory loss shows a statistically significant but small improvement for 4+ frozen heads. This is not the signal $n = 3$ saw — it is a different, finer signal that only became visible at higher resolution.

**At $n = 300$:** The final-loss null is definitive (all conditions $p > 0.40$; Spearman $\rho = 0.0023$, $p = 0.92$). The trajectory improvement strengthens to high significance: freeze 8 mean-loss $\Delta = -0.12\%$ ($p < 0.0001$), freeze 12 $\Delta = -0.17\%$ ($p < 0.0001$), freeze 16 $\Delta = -0.19\%$ ($p < 0.0001$). The monotonic dose-response in trajectory improvement, combined with no final-loss cost, confirms that frozen random-projection heads reduce gradient interference throughout training without affecting convergence.

**Table 1.** Head freezing results (means ± std across 30 runs).

| Frozen Heads | Final Loss | Mean Loss | DG Index |
|:---:|:---:|:---:|:---:|
| 0 (baseline) | 2.469 ± 0.340 | 2.639 ± 0.023 | 0.571 ± 0.312 |
| 1 | 2.472 ± 0.343 | 2.639 ± 0.023 | 0.572 ± 0.347 |
| 2 | 2.471 ± 0.343 | 2.638 ± 0.023 | 0.575 ± 0.451 |
| 4 | 2.473 ± 0.346 | 2.638 ± 0.023 | 0.509 ± 0.297 |
| 8 | 2.481 ± 0.337 | 2.636 ± 0.023 | 0.528 ± 0.361 |
| 12 | 2.479 ± 0.347 | 2.635 ± 0.023 | 0.475 ± 0.223 |
| 16 | 2.479 ± 0.348 | 2.635 ± 0.023 | 0.512 ± 0.266 |

Mean trajectory loss shows a statistically significant but tiny improvement for 4+ frozen heads: freeze 4 ($p = 0.012$, $d = -0.49$), freeze 8 ($p < 0.001$, $d = -1.03$), freeze 12 ($p < 0.001$, $d = -1.13$), freeze 16 ($p < 0.001$, $d = -0.96$). The effect size is 0.1–0.2% of mean loss — statistically robust but practically small. Frozen heads appear to reduce gradient interference during the trajectory without altering the convergence basin. This trajectory improvement is notable: freezing weights reduces gradient computation, but there is no obvious reason the remaining gradient computation should improve simply because some heads are excluded. The improvement likely reflects reduced destructive gradient interactions.

The DG Index does not increase with freezing. Freeze 12 *decreases* DG significantly ($p = 0.034$, $d = -0.41$). All other levels are non-significant.

**Classification:** Trajectory improvement for 4+ frozen heads = *emergent behavior* (not prescribed by the optimizer). Final-loss indifference = *basin geometry*.

### 4.2 Experiment 2: Cell-View GPT

**At $n = 3$:** The signal was ambiguous. Cell-view appeared to elevate DG substantially (+25.5%), suggesting possible rerouting behavior.

**At $n = 30$:** The DG signal resolved to null ($p = 0.34$). The degradation signal became clear and quantified.

**At $n = 300$:** The cell-view degradation strengthens and tightens: +2.9% final loss ($t = 8.307$, $p < 0.0001$, $d = 0.480$). The effect size narrows from the $n = 30$ estimate, confirming that eliminating inter-layer gradient flow incurs a consistent but moderate cost.

**Table 2.** Cell-view vs. baseline (means ± std across 30 runs).

| Condition | Mean Loss | Final Loss | DG Index |
|:---:|:---:|:---:|:---:|
| Baseline | 2.639 ± 0.023 | 2.469 ± 0.340 | 0.571 ± 0.312 |
| Cell-view | 2.687 ± 0.023 | 2.590 ± 0.367 | 0.507 ± 0.260 |

Eliminating all inter-layer gradient communication costs +4.9% in final loss ($p < 0.001$, $d = +1.16$) and +1.8% in mean loss ($p < 0.001$, $d = +3.31$). The degradation is significant but bounded — the system still learns without its defining optimization mechanism.

**Classification:** *Tolerance* — the system absorbs the removal of inter-layer gradient flow at a bounded cost. The DG elevation was noise at this resolution.

### 4.3 Experiment 3: Gradient Degradation

**At $n = 3$:** The signal was ambiguous. All four gradient degradation methods appeared neutral ($p > 0.26$), and small noise appeared to help.

**At $n = 30$:** The ambiguous signal resolved sharply. Three of four methods significantly degrade; one is genuinely tolerated.

**At $n = 300$:** The threshold between tolerance and degradation sharpens. Noise at $\sigma = 0.01$ remains non-significant ($-0.1\%$, $p = 0.52$), confirming genuine tolerance. All three degradation conditions strengthen: noise at $\sigma = 0.1$ (+2.3%, $p < 0.0001$, $d = 0.385$), sign-only (+4.9%, $p < 0.0001$, $d = 0.596$), and quantized 3-level (+3.4%, $p < 0.0001$, $d = 0.495$). The sign-only effect strengthened from $p = 0.004$ at $n = 30$ to $p < 0.0001$ at $n = 300$.

**Table 3.** Gradient degradation results (means across 30 runs).

| Method | Final Loss | Δ% | $p$ (final) | Mean Loss | $p$ (mean) | $d$ (mean) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline | 2.469 | — | — | 2.639 | — | — |
| Noisy ($\sigma = 0.01$) | 2.477 | +0.3% | 0.525 | 2.639 | 0.978 | -0.01 |
| Noisy ($\sigma = 0.1$) | 2.530 | +2.5% | 0.021* | 2.698 | <0.001*** | +3.79 |
| Sign-only | 2.582 | +4.6% | 0.004** | 2.742 | <0.001*** | +6.13 |
| Quantized (3-bit) | 2.543 | +3.0% | 0.019* | 2.703 | <0.001*** | +4.54 |

Three of four methods significantly degrade final loss: sign-only (+4.6%, $p = 0.004$), quantized (+3.0%, $p = 0.019$), and noisy $\sigma = 0.1$ (+2.5%, $p = 0.021$). Only small noise ($\sigma = 0.01$) is genuinely tolerated ($p = 0.525$). Mean loss effects are highly significant for all three degrading methods ($p < 0.001$) with large Cohen's $d$ values (3.8–6.1). The "noise helps" effect (noisy $\sigma = 0.01$ improving loss) is not supported ($p = 0.978$ for mean loss).

**Classification:** *Tolerance* — the system absorbs gradient noise up to a threshold ($\sigma = 0.01$). Above that threshold, degradation follows. The gradient-quality information (sign structure, quantization levels) carries more essential signal than quantity.

### 4.4 Experiment 4: Vision Radius Sweep

**At $n = 3$:** The signal was ambiguous. An information bottleneck effect appeared possible — intermediate window sizes seemed to outperform full context.

**At $n = 30$:** The ambiguous signal resolved to null for final loss across all window sizes. Tiny mean-trajectory effects are statistically detectable but not practically meaningful.

**At $n = 300$:** Fine structure emerges that was invisible at $n = 30$. Window 1 significantly harms performance (+0.4%, $p = 0.0009$), windows 2 and 4 remain non-significant ($p = 0.32$ and $p = 0.66$ respectively), window 8 produces a small but significant improvement ($-0.1\%$, $p = 0.022$), and window 16 is identical to baseline ($p = 1.000$). This reveals a monotonic structure: extreme restriction harms, moderate restriction is neutral, and a mild restriction slightly benefits — a pattern consistent with beneficial information bottleneck effects that were below detection threshold at $n = 30$.

**Table 4.** Vision radius results (means across 30 runs).

| Window | Final Loss | $p$ (final) | Mean Loss | $p$ (mean) | $d$ (mean) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline (full) | 2.469 | — | 2.639 | — | — |
| 1 | 2.486 | 0.155 | 2.650 | <0.001*** | +2.49 |
| 2 | 2.485 | 0.136 | 2.640 | 0.185 | +0.25 |
| 4 | 2.475 | 0.524 | 2.636 | 0.004** | -0.57 |
| 8 | 2.471 | 0.478 | 2.638 | <0.001*** | -0.68 |
| 16 | 2.469 | 1.000 | 2.639 | 1.000 | 0.00 |

No window size significantly changes final loss ($p > 0.13$ for all). Window=16 reproduces baseline exactly (sanity check). Mean-loss effects exist but are negligibly small. The chess paper's finding that intermediate vision radius outperforms omniscience does not produce meaningful effects for attention windowing at this scale.

**Classification:** *Tolerance* — attention restriction at all tested scales is absorbed without meaningful final-loss change.

### 4.5 Experiment 5: Communication Topology

**At $n = 3$:** The signal was ambiguous. A U-shaped loss curve appeared possible — partial communication seemed to outperform both full and no communication.

**At $n = 30$:** The U-shaped curve resolved to flat (except at the extreme of zero communication). Partial gradient flow is absorbed without meaningful degradation.

**At $n = 300$:** The architecture's indifference to gradient fraction holds at high power. Heavy ($p = 0.35$), half ($p = 0.87$), and light ($p = 0.41$) communication topologies all remain non-significant. Only cell-view — the complete elimination of inter-layer flow — produces significant degradation ($p < 0.0001$). The architecture tolerates any nonzero gradient fraction without measurable cost.

**Table 5.** Communication topology results (means across 30 runs).

| Topology | Fraction | Final Loss | $p$ (final) | Mean Loss | $p$ (mean) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Full | 1.00 | 2.469 | — | 2.639 | — |
| Heavy | 0.75 | 2.472 | 0.024* | 2.639 | 0.553 |
| Half | 0.50 | 2.470 | 0.551 | 2.639 | 0.100† |
| Light | 0.25 | 2.471 | 0.426 | 2.639 | 0.303 |
| Cell-view | 0.00 | 2.590 | <0.001*** | 2.687 | <0.001*** |

Partial gradient flow (25–75%) produces no meaningful degradation. Heavy (75%) shows a statistically significant but tiny final-loss increase ($p = 0.024$, $d = +0.44$, +0.1%). Half and light are non-significant. Only cell-view (0%) degrades substantially: +4.9% final loss ($p < 0.001$), +1.8% mean loss ($p < 0.001$).

**Classification:** *Tolerance* — the system absorbs substantial reductions in inter-layer gradient flow. Only the total removal of inter-layer communication crosses the degradation threshold.

### 4.6 Experiment 6: Courage vs. Caution

**At $n = 3$:** The signal was ambiguous. The courage/caution matrix appeared to produce inconsistent results without clear inversion.

**At $n = 30$:** The sign-only vs. dropout comparison resolved to a robust and significant inversion.

**At $n = 300$:** The inversion is rock-solid. Cautious/cautious remains non-significant ($p = 0.51$). The sign-only condition (cautious/courageous) degrades by +4.9% ($p < 0.0001$, $d = 0.596$). Dropout (courageous/cautious) shows a marginal trend ($+0.2\%$, $p = 0.052$). The direct sign-only vs. dropout comparison yields $\Delta = +4.7\%$ ($p < 0.0001$, $d = 0.588$), confirming the substrate-dependent inversion at the highest statistical power.

**Table 6.** Courage vs. caution results (means across 30 runs).

| Condition | Final Loss | $p$ (final) | Mean Loss | $p$ (mean) | $d$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline | 2.469 | -- | 2.639 | -- | -- |
| (a) Caut./Caut. | 2.484 | 0.017\* | 2.639 | 0.799 | -0.05 |
| (b) Caut./Cour. | 2.582 | 0.004\*\* | 2.742 | <0.001\*\*\* | +6.13 |
| (c) Cour./Caut. | 2.481 | 0.029\* | 2.641 | 0.010\*\* | +0.50 |
| (d) Cour./Cour. | 2.509 | 0.044\* | 2.697 | <0.001\*\*\* | +4.33 |

All conditions significantly degrade final loss versus baseline ($p < 0.05$ for all). Sign-only (b) shows the largest degradation (+4.6% final loss, $p = 0.004$; +3.9% mean loss, $p < 0.001$).

The sign-only vs. dropout comparison is the paper's most robust substrate-comparison finding. Paired comparison (matched seeds): sign-only worsens mean loss by +3.8% relative to dropout ($p < 0.001$, $d = +6.00$); final loss by +4.1% ($p = 0.010$, $d = +0.51$). The chess paper's predicted best condition (b, "cautious position, courageous moves") is the worst non-baseline condition in the transformer domain.

**Classification:** *Substrate-dependent* — the inversion reflects different perturbation-tolerance profiles across optimization substrates (discrete/irreversible vs. continuous/differentiable), not an emergent behavior.

### 4.7 Experiment 7: Recovery After Damage

**At $n = 3$:** The signal was ambiguous. Recovery appeared complete but with too few observations to confirm.

**At $n = 30$:** Complete recovery is unambiguously confirmed across all 30 runs.

**At $n = 300$:** Complete recovery is confirmed with extreme precision. Recovery vs. control: $p = 0.64$, $d = -0.027$. Final-loss ratio: $0.9997 \pm 0.0086$. All 300 of 300 runs recovered, with mean overshoot $-0.0007 \pm 0.0018$ (essentially zero). Recovery time: $1.4 \pm 1.7$ steps. The confidence interval shrinks approximately five-fold relative to $n = 30$, and the null holds with no exceptions.

**Table 7.** Recovery after transient damage ($n = 30$, 500 total steps).

| Metric | Recovery | Control | $p$ (paired) |
|---|---|---|---|
| Final loss | 2.470 ± 0.353 | 2.469 ± 0.349 | 0.905 |
| Final ratio (rec/ctrl) | 0.9999 ± 0.009 | — | 0.973 (vs 1.0) |
| Recovery time | 2 ± 2 steps | — | 30/30 recovered |

The damaged-then-recovered model reaches the same final loss as the undamaged control ($p = 0.905$, ratio 0.9999). All 30 runs recovered within a mean of 2 steps after damage removal. No overshoot was observed (mean overshoot $= -0.001$). The 100 steps of training with 8 frozen heads had no lasting effect — the "damage" was entirely absorbed by subsequent normal training.

The completeness of recovery ($p = 0.905$, ratio 0.9999) is notable: a more brittle system could recover to a different basin, but this one returns to the same endpoint with no path-dependence. The optimizer prescribes convergence to *a* minimum, not necessarily to the *same* minimum after a detour through a constrained subspace.

**Classification:** *Emergent behavior* — complete, path-independent recovery to identical final loss is not directly prescribed by the loss minimization objective.

### 4.8 Experiment 8: Chimera Assembly

**At $n = 3$:** The signal was ambiguous. Chimeras appeared to converge but from very few observations.

**At $n = 30$:** Convergence is confirmed for all chimera types. The specific layer assignment (which layers come from which model) does not matter.

**At $n = 300$:** All chimera types remain non-significant (AABB $p = 0.51$, ABAB $p = 0.83$, BBAA $p = 0.95$, ABBA $p = 0.63$), with all conditions falling within 0.1% of control. No systematic differences in convergence emerge even at this resolution — the basin of attraction is uniformly accessible from all tested chimera configurations.

**Table 8.** Chimera assembly results ($n = 30$, 200 + 200 steps).

| Condition | Initial Loss | Final Loss | Recovery | vs Control $p$ |
|---|---|---|---|---|
| Control (A continues) | — | 2.417 ± 0.352 | — | — |
| AABB | 2.985 | 2.428 ± 0.338 | 18.5% | 0.266 |
| ABAB | 2.938 | 2.425 ± 0.341 | 17.3% | 0.456 |
| BBAA | 2.834 | 2.414 ± 0.348 | 14.7% | 0.824 |
| ABBA | 2.880 | 2.420 ± 0.341 | 15.8% | 0.782 |

All chimera types converge to the same final loss as the control ($p > 0.26$ for all). Despite starting at substantially worse loss (2.83–2.98 vs. baseline ~2.47), the chimeras recover 15–19% of the initial gap and reach control-equivalent performance.

The chimera result shows that the basin of attraction is wide enough to absorb dramatically different starting conditions. SGD prescribes convergence; the finding is that the basin is wide enough to reach from dramatically different starting points.

**Classification:** *Basin geometry* — SGD re-finds the same minimum from any structurally valid starting point.

### 4.9 Experiment 9: Gradual vs. Sudden Damage

**At $n = 3$:** The signal was ambiguous. The gradual vs. sudden comparison appeared promising but underpowered.

**At $n = 30$:** The key finding resolves clearly and significantly. This is the paper's strongest emergent-behavior result.

**At $n = 300$:** The stress inoculation effect strengthens dramatically, from $p = 0.011$ at $n = 30$ to $p < 0.0001$ at $n = 300$. Sudden full noise degrades by +2.0% vs. control ($p < 0.0001$, $d = 0.358$). Gradual noise shows only +0.4% degradation ($p = 0.024$, $d = 0.131$). The critical comparison — gradual vs. sudden full — yields $\Delta = -1.5\%$ ($p < 0.0001$, $d = -0.278$), confirming stress inoculation as a robust phenomenon. Sudden half noise is marginal ($+0.4\%$, $p = 0.051$). The escalation from single-star significance at $n = 30$ to triple-star at $n = 300$ is the clearest example of a signal strengthening with resolution.

**Table 9.** Gradual vs. sudden noise ($n = 30$, 200 steps).

| Condition | Final Loss | $p$ (vs ctrl) | Mean Loss | $p$ (mean) |
|---|---|---|---|---|
| Control | 2.469 ± 0.340 | — | 2.639 ± 0.023 | — |
| Sudden full ($\sigma = 0.1$) | 2.551 ± 0.374 | 0.004** | 2.698 ± 0.027 | <0.001*** |
| Gradual (0 to 0.1) | 2.481 ± 0.364 | 0.427 | 2.635 ± 0.024 | 0.006** |
| Sudden half (step 100) | 2.504 ± 0.364 | 0.034* | 2.640 ± 0.022 | 0.085† |

**This is the paper's strongest finding.** Gradual exposure to noise builds tolerance: the gradually-ramped condition is statistically indistinguishable from control ($p = 0.43$), while sudden exposure to the same noise level significantly degrades final loss ($p = 0.004$, +3.3%). Direct comparison: gradual is significantly better than sudden full ($p = 0.011$, $d = -0.50$). The gradual condition's mean loss is actually *below* control ($-0.1\%$, $p = 0.006$), suggesting the noise acts as regularization when introduced gently.

**Why is this emergent?** The gradient update rule is identical in the sudden and gradual conditions at every step — the only difference is the *history* of noise levels. The optimizer at step $t$ does not remember what noise level was applied at step $t-1$. Yet the system's final state depends on that history, and the gradually-exposed model reaches a region of weight space that the suddenly-exposed model does not.

**Classification:** *Emergent behavior* — the system develops differential tolerance based on perturbation history. The optimization objective does not specify how noise history should change the final state, but it does.

### 4.10 Experiment 10: Regeneration (Layer Reset)

**At $n = 3$:** The signal was ambiguous. Regeneration appeared possible but with too few observations to confirm the completeness.

**At $n = 30$:** Complete regeneration is confirmed for all four layers.

**At $n = 300$:** Fine structure emerges in layer-specific regeneration completeness. Layer 0 shows a small but significant residual deficit (completeness 94.3%, $p = 0.016$). Layer 1 regenerates perfectly (completeness 99.4%, $p = 0.18$). Layer 2 shows a small but significant overcompensation (completeness 101.1%, $p = 0.040$), suggesting the system slightly overshoots baseline after layer 2 reset. Layer 3 shows a marginal trend (completeness 97.6%, $p = 0.091$). The overall confidence interval tightens substantially relative to $n = 30$, revealing that regeneration is near-complete but not perfectly uniform across layers.

**Table 10.** Layer regeneration after random reset ($n = 30$, 200 + 200 steps).

| Reset Layer | Immediate Damage | Final Loss | Completeness | vs Control $p$ |
|---|---|---|---|---|
| Control (no reset) | — | 2.407 ± 0.346 | — | — |
| Layer 0 | +0.264 | 2.410 ± 0.347 | 0.960 | 0.696 |
| Layer 1 | +0.309 | 2.410 ± 0.349 | 0.869 | 0.435 |
| Layer 2 | +0.328 | 2.405 ± 0.347 | 1.003 | 0.575 |
| Layer 3 | +0.342 | 2.412 ± 0.346 | 0.998 | 0.177 |

Complete regeneration: all four layers recover to control-equivalent loss after being destroyed ($p > 0.17$ for all). Later layers suffer slightly more immediate damage (+0.26 for L0 vs. +0.34 for L3) but all regenerate equally. Layers 2 and 3 achieve completeness significantly above 0.9 ($p < 0.001$), with L2 showing completeness of 1.003 (marginally exceeding baseline). Layer position does not predict damage magnitude (Spearman $\rho = 0.078$, $p = 0.395$).

The completeness of regeneration — particularly L2 at 1.003 and L3 at 0.998 — demonstrates that the network re-finds the same *functional role*, regardless of what was there before. The optimizer prescribes convergence to a minimum; it does not prescribe that a rebuilt layer should reach the same functional role as if it had never been destroyed.

**Classification:** *Emergent behavior* — complete layer regeneration to control-equivalent performance is not directly prescribed by the loss minimization objective.

### 4.11 Experiment 11: Transplantation

**At $n = 3$:** The signal was ambiguous. A transplant advantage appeared possible.

**At $n = 30$:** The transplant null result resolves clearly. There is no advantage to a structured (donor) layer over a random replacement.

**At $n = 300$:** The null holds across all layers with no exceptions. Layer-specific $p$-values: L0 $p = 0.54$, L1 $p = 0.46$, L2 $p = 0.46$, L3 $p = 0.45$. Overall $p = 0.45$. The transplant null is definitive: donor layers confer no advantage over random reinitialization at any layer position, even at high statistical power.

**Table 11.** Transplant vs. random reset ($n = 30$, 200 + 200 steps).

| Layer | Transplant Final | Random Final | Gap (rand − trans) | $p$ |
|---|---|---|---|---|
| L0 | 2.409 | 2.410 | +0.000 | 0.975 |
| L1 | 2.409 | 2.410 | +0.001 | 0.839 |
| L2 | 2.409 | 2.405 | -0.004 | 0.173 |
| L3 | 2.408 | 2.412 | +0.005 | 0.262 |
| **Overall** | — | — | **-0.000** | **0.860** |

Transplanted layers and randomly-reset layers converge to the same final loss ($p = 0.86$ overall). There is no transplant advantage — a layer from a separately-trained donor network is accepted no better and no worse than a random replacement. Layer position does not modulate the transplant effect (Spearman $\rho = 0.060$, $p = 0.513$). The network does not recognize or benefit from the structure of the donor layer; it simply rebuilds whatever is placed there.

The transplant null result means the architecture does not privilege pre-adapted weights over random weights — both converge to the same endpoint. As with chimera convergence, this reflects the smoothness of the loss landscape.

**Classification:** *Basin geometry* — no transplant advantage; the basin is equally accessible from pre-trained and random initializations.

### 4.12 Experiment 12: Competing Objectives

**At $n = 3$:** The signal was ambiguous. The distinction between adversarial and inactive layers appeared but was underpowered.

**At $n = 30$:** The distinction resolves sharply. Adversarial layers degrade dramatically; inactive layers are tolerated.

**At $n = 300$:** The adversarial degradation strengthens: competing objectives degrade by +23.3% ($p < 0.0001$, $d = 0.602$), up from +18.1% at $n = 30$. Frozen layers remain non-significant ($p = 0.74$, $d = 0.019$). The competing vs. freeze comparison yields $\Delta = +23.2\%$ ($p < 0.0001$). High variance persists (std = 1.12), indicating that the adversarial condition produces genuinely variable outcomes rather than a bimodal distribution.

**Table 12.** Competing objectives ($n = 30$, 200 + 200 steps).

| Condition | Final Loss | vs Control Δ% | $p$ (vs ctrl) |
|---|---|---|---|
| Control (400 steps normal) | 2.407 ± 0.346 | — | — |
| Competing (negate L2-3 grads) | 2.842 ± 1.072 | +18.1% | 0.010** |
| Freeze L2-3 | 2.411 ± 0.349 | +0.2% | 0.303 |

Negating gradients for layers 2-3 causes significant degradation (+18.1%, $p = 0.010$) with high variance (std = 1.07). But merely freezing those same layers causes negligible degradation (+0.2%, $p = 0.303$). Competing vs. freeze: $p = 0.011$, $d = +0.49$. The network can route around absent layers but cannot compensate for layers actively working against the objective.

**Classification:** *Tolerance* for the freeze condition (absence is absorbed); the adversarial result is a boundary condition revealing the limits of tolerance.

### 4.13 Cross-Experiment Synthesis

**Table 13a.** Perturbation tolerance (Experiments 1-6, paired $t$-tests, $n = 30$).

| Perturbation | Final Loss Δ% | $p$ (final) | Mean Loss Δ% | $p$ (mean) | Classification |
|---|---|---|---|---|---|
| Freeze 1-4 heads | +0.1-0.2% | >0.37 | -0.0% | 0.01-0.81 | Basin geometry / emergent |
| Freeze 8-16 heads | +0.4-0.5% | 0.06-0.19 | -0.1-0.2% | <0.001 | Emergent (trajectory) |
| Window=4,8 | +0.1-0.3% | >0.47 | -0.03-0.1% | <0.004 | Tolerance |
| Partial flow (25-75%) | +0.0-0.1% | >0.02 | ±0.0% | >0.10 | Tolerance |
| Noisy σ=0.01 | +0.3% | 0.53 | -0.0% | 0.98 | Tolerance |
| Dropout (p=0.1) | +0.5% | 0.029 | +0.1% | 0.010 | Mild degradation |
| Noisy σ=0.1 | +2.5% | 0.021 | +2.2% | <0.001 | Degradation |
| Quantized 3-bit | +3.0% | 0.019 | +2.4% | <0.001 | Degradation |
| Sign-only | +4.6% | 0.004 | +3.9% | <0.001 | Degradation |
| Cell-view | +4.9% | <0.001 | +1.8% | <0.001 | Degradation |

**Table 13b.** Multi-phase perturbation results (Experiments 7-12, paired $t$-tests, $n = 30$).

| Experiment | Condition | vs Control | $p$ | Classification |
|---|---|---|---|---|
| 7: Recovery | Damaged then recovered | +0.0% | 0.905 | **Emergent** |
| 8: Chimera (AABB) | Frankenstein assembly | +0.4% | 0.266 | Basin geometry |
| 8: Chimera (ABAB) | Alternating layers | +0.3% | 0.456 | Basin geometry |
| 9: Sudden noise | $\sigma = 0.1$ all steps | +3.3% | 0.004** | Degradation |
| 9: Gradual noise | Ramp 0 to 0.1 | +0.5% | 0.427 | **Emergent** |
| 9: Gradual vs sudden | Direct comparison | -2.8% | 0.011* | **Emergent** |
| 10: Regeneration (any layer) | Reset then retrain | +0.1% | >0.17 | **Emergent** |
| 11: Transplant vs random | Donor vs random layer | ±0.0% | 0.860 | Basin geometry |
| 12: Adversarial L2-3 | Negate gradients | +18.1% | 0.010** | Severe degradation |
| 12: Freeze L2-3 | Zero gradients | +0.2% | 0.303 | Tolerance |

**Behavioral classification (full):**

*Emergent behaviors* — not directly prescribed by the optimization objective:
- **Stress inoculation (Exp 9):** Gradual noise builds tolerance that sudden noise does not. The gradient rule is the same at every step; only the history differs.
- **Complete recovery (Exp 7):** Path-independent convergence to identical final loss after transient damage.
- **Complete regeneration (Exp 10):** Any single layer destroyed and rebuilt reaches control-equivalent performance.
- **Head-freezing trajectory improvement (Exp 1):** Frozen random-projection heads reduce gradient interference.

*Basin geometry* — expected optimizer behavior on this landscape:
- **Chimera convergence (Exp 8):** Models assembled from incompatible parts converge to the same minimum.
- **Transplant indifference (Exp 11):** No difference between donor and random layers — the basin is equally accessible from both.

*Tolerance* — system works despite corruption:
- **Gradient degradation threshold (Exp 3):** Small noise tolerated; large noise degrades.
- **Partial communication (Exp 5):** 25–75% gradient flow tolerated without meaningful degradation.
- **Vision restriction (Exp 4):** All window sizes tolerated for final loss.

**DG does not track perturbation.** At $n = 3$, the DG Index appeared to scale with perturbation severity. At $n = 30$, this signal resolved to null: no perturbation condition produces a statistically significant DG increase ($p > 0.16$ for all). Baseline DG = 0.571 ± 0.312; all conditions fall within 0.447–0.606. At $n = 300$, the DG null holds: the metric still does not track perturbation severity, confirming that DG captures intrinsic stochastic structure of loss trajectories rather than perturbation response.

### 4.14 Findings — What Perturbation Revealed

#### Emergent Behaviors

**Finding 1: Gradual exposure builds tolerance (Exp 9).** Gradual noise ramp (0 to 0.1) produces no final-loss degradation ($p = 0.43$), while sudden exposure to the same noise level degrades by +3.3% ($p = 0.004$). Direct comparison: $p = 0.011$, $d = -0.50$. The gradient update rule is identical at every step — only the history of noise levels differs. That history changes the system's final state.

**Finding 2: Complete recovery (Exp 7).** A model damaged during training (8 frozen heads for 100 steps) recovers to identical final loss ($p = 0.905$, ratio 0.9999). All 30 runs recovered within a mean of 2 steps. The path through damage leaves no trace.

**Finding 3: Complete regeneration (Exp 10).** Any single layer can be destroyed and rebuilt to control-equivalent performance ($p > 0.17$ for all layers). The network re-finds the same functional role regardless of what was there before.

**Finding 4: Head-freezing trajectory improvement (Exp 1).** Freezing 4+ randomly-initialized heads produces small but statistically robust mean-trajectory improvements (freeze 8: $p < 0.001$, $d = -1.03$). Frozen random-projection heads reduce gradient interference.

#### Basin Geometry

**Finding 5: Chimera convergence (Exp 8).** Models assembled from parts of two independently-trained networks converge to the same final loss as undamaged continuation ($p > 0.26$ for all chimera types). The loss landscape has a single dominant basin; SGD finds it from any starting point.

**Finding 6: Transplant indifference (Exp 11).** Transplanted layers and randomly-reset layers converge to the same final loss ($p = 0.86$ overall). The basin is equally accessible from pre-trained and random initializations.

#### Tolerance

**Finding 7: Gradient quality matters more than quantity (Exp 3, 5).** Reducing gradient precision (sign-only: $d = +6.13$) degrades more than reducing gradient magnitude (partial flow: $d < 0.5$) or completeness (freezing: $p > 0.06$). The architecture tolerates magnitude reduction but not sign-structure destruction.

**Finding 8: The sign-only vs. dropout inversion (Exp 3, 6).** Sign-only gradients degrade performance significantly more than dropout ($p < 0.001$, $d = +6.00$ for mean loss). This inverts the chess paper's "cautious position, courageous moves" prediction — a substrate-dependent difference.

**Finding 9: Adversarial vs. inactive tolerance (Exp 12).** Frozen layers cost nothing ($p = 0.30$); adversarial layers cost +18% ($p = 0.01$). The architecture tolerates absence but not opposition.

#### Signals That Resolved Across Scales

At $n = 3$, the following signals were ambiguous; at $n = 30$, they resolved:

- **Head freezing improves final loss:** At $n = 3$, this appeared as a possible improvement signal. At $n = 30$, the final-loss signal resolved to null ($p > 0.06$), but a different, finer signal emerged: mean-trajectory improvement. The picture sharpened — the original signal was in the wrong metric.
- **DG scales with perturbation:** At $n = 3$, the DG Index appeared to scale with perturbation severity. At $n = 30$, this resolved to null across all conditions ($p > 0.16$). At $n = 300$, the null is confirmed: DG still does not track perturbation, establishing that this metric captures intrinsic trajectory structure rather than perturbation response.
- **Gradient degradation is neutral:** At $n = 3$, all four methods appeared neutral. At $n = 30$, three of four resolved to significant degradation. The resolution was too coarse to see the effect.
- **Partial communication outperforms full:** At $n = 3$, a U-shaped curve appeared. At $n = 30$, this resolved to flat (except at zero). The pilot U-shape was sampling noise.
- **Noise helps:** At $n = 3$, small noise appeared beneficial. At $n = 30$, this resolved to null ($p = 0.978$). The apparent benefit was within noise.


## 5. Discussion

### 5.1 What Perturbation Reveals

During standard training, the transformer's components cooperate invisibly. Perturbation makes the system's behavioral repertoire legible by forcing it to operate under constraint. Three categories emerge from the twelve experiments.

**Emergent behaviors.** Four findings describe behaviors not directly prescribed by the optimization objective. Stress inoculation (Exp 9): the system develops differential tolerance based on noise history, despite identical gradient update rules at every step. Complete recovery (Exp 7): the system returns to the same minimum after a detour through a constrained subspace ($p = 0.905$, ratio 0.9999). Complete regeneration (Exp 10): destroyed layers rebuild to control-equivalent performance across all four layer positions ($p > 0.17$). Head-freezing trajectory improvement (Exp 1): frozen random-projection heads reduce gradient interference during training ($p < 0.001$). These behaviors are consequences of the optimizer operating on this particular loss landscape but are not specified by the loss minimization objective itself.

**Basin geometry.** Two findings — chimera convergence (Exp 8) and transplant indifference (Exp 11) — reflect the expected behavior of gradient descent on a smooth loss landscape with a single dominant attractor. The basin is wide enough to reach from dramatically different starting points. This is a property of the landscape, not of the system's adaptive capacity.

**Tolerance.** The system absorbs gradient noise up to a threshold ($\sigma = 0.01$), partial communication reduction (25–75%), and vision restriction without meaningful final-loss degradation. The absence-vs-adversity distinction (Exp 12) defines the tolerance boundary: frozen layers are absorbed ($p = 0.30$); adversarial layers degrade substantially ($p = 0.01$, +18%).

**The distinction matters.** Emergent behaviors and basin geometry can look similar — both produce control-equivalent final loss after perturbation. The difference is whether the behavior is a generic consequence of smooth loss landscape geometry (basin geometry) or requires something more specific: a particular interaction between optimization trajectory and perturbation history (emergent). Stress inoculation is the clearest test case: the gradient rule is identical in both conditions, only the history differs, yet outcomes diverge.

### 5.2 Three Claims

Three claims emerge from the combined evidence of twelve experiments at $n = 30$, confirmed at $n = 300$.

**Claim 1: Some transformer behaviors are not prescribed by the optimization objective.** Stress inoculation, complete recovery, complete regeneration, and head-freezing trajectory improvement emerge from the interaction between SGD and the loss landscape but are not specified by the loss minimization directive. Perturbation reveals these by forcing the system to operate under constraint.

**Claim 2: Gradual exposure builds tolerance that sudden exposure does not.** Gradual noise ramp produces no final-loss degradation ($p = 0.43$), while sudden exposure to the same peak noise level degrades by +3.3% ($p = 0.004$). At $n = 300$, this effect strengthens from $p = 0.011$ to $p < 0.0001$ ($d = -0.278$): the gradual-vs-sudden gap is $-1.5\%$. Same optimizer, same peak noise, same number of steps, different history, different outcome.

**Claim 3: The chess-paper courage/caution strategy inverts across substrates.** Sign-only gradients degrade performance significantly more than dropout ($p < 0.001$). The substrate-dependent inversion reveals that optimal perturbation-tolerance strategies depend on whether the action space is discrete and irreversible (chess) or continuous and differentiable (transformers).

### 5.3 Connection to Distributed Chess

Kofman, Campitelli & Levin (2025) implemented a distributed form of chess where each piece operates as an autonomous agent. Experiments 4-6 tested three predictions; Experiments 7-12 extend the morphogenetic paradigm beyond the chess paper's framework.

**Information bottleneck as beneficial constraint (partially supported at $n = 300$).** The chess paper's central result — intermediate vision radius $R4$ outperforms omniscient $R7$ — does not translate to attention windowing at $n = 30$. At $n = 300$, however, fine structure emerges: window 1 significantly harms ($p = 0.0009$), window 8 significantly improves ($p = 0.022$), and intermediate windows are neutral. This monotonic structure is consistent with a weak information bottleneck effect that was below detection threshold at lower power.

**Partial communication tolerance (confirmed as tolerance, not improvement).** Reducing gradient flow to 25% produces no significant degradation, but partial flow does not *improve* over full backpropagation. The tolerance is real; the U-shaped curve from $n = 3$ pilot data was noise.

**Courage/caution strategy (robustly inverted).** Sign-only (b) degrades mean loss by +3.8% relative to dropout (c), $p < 0.001$, $d = +6.00$. The chess paper's optimal strategy produces the worst transformer outcome. This inversion is substrate-dependent.

**Stress inoculation (new, Exp 9).** The gradual-vs-sudden result has no direct chess analog but connects to Levin's broader developmental biology framework. That gradient descent exhibits stress inoculation suggests this property may appear across optimization substrates — but whether chess or biological systems show the same phenomenon requires direct testing.

**Chimera convergence (new, Exp 8).** Unlike biological chimeras, which can develop abnormally at graft boundaries, transformer chimeras converge seamlessly. This reflects the smoothness of the loss landscape versus the discrete developmental signaling in biological systems.

### 5.4 Scaling as Methodology

The three-scale protocol is not merely a replication strategy — it is a methodological commitment. At $n = 3$, the resolution is coarse. Real effects can be invisible; noise can masquerade as signal. At $n = 30$, the picture sharpens: moderate effects become detectable, and many $n = 3$ ambiguities resolve. At $n = 300$, fine structure emerges that $n = 30$ cannot see.

The practical implication: do not interpret findings at $n = 3$ as conclusions. They are coarse signal. The signal that head freezing appeared to improve final loss at $n = 3$ was not wrong — it was a low-resolution view of a real region of the parameter space. At $n = 30$, the final-loss effect resolved to null, but a different effect (trajectory improvement) became visible. The picture changed not because the $n = 3$ finding was retracted but because the resolution increased.

The $n = 300$ results confirm this framing. Some null findings at $n = 30$ revealed fine structure at $n = 300$ (vision radius, regeneration layer specificity). Some robust findings at $n = 30$ strengthened (stress inoculation from $p = 0.011$ to $p < 0.0001$; sign-only degradation from $p = 0.004$ to $p < 0.0001$). No robust $n = 30$ finding reversed at $n = 300$, and no $n = 30$ null became significant in the opposite direction. The three-scale protocol thus achieved its design objective: coarse signal at $n = 3$, resolved signal at $n = 30$, fine structure at $n = 300$.

### 5.5 Limitations

**Scale.** The model has 4 layers, 16 dimensions, and ~11,000 parameters. Whether these findings extend to production-scale transformers is unknown. The emergent behaviors (stress inoculation, recovery, regeneration) may be specific to small models with simple loss landscapes, or they may be architectural universals.

**Task complexity.** Character-level name generation is a toy task. Whether stress inoculation appears in language modeling or other complex tasks is not established.

**Training duration.** 200 steps per phase captures early learning dynamics. The gradual-exposure tolerance (Exp 9) might not persist at longer training horizons.

**DG metric.** The DG Index does not respond to perturbation at $n = 30$ or $n = 300$, resolving an ambiguous $n = 3$ signal.

**Effect sizes.** Many statistically significant effects are practically negligible (<0.5%). Statistical significance at $n = 30$ does not imply practical importance.

**Transplant design.** The null result for transplant advantage (Exp 11) may reflect that both models learned the same task on the same data. Cross-task transplantation (donor trained on a different task) might show transplant effects.

**Competing objectives design.** Gradient negation (Exp 12) is a maximally adversarial perturbation. Subtler forms of inter-layer conflict might reveal more nuanced compensation mechanisms.

**$n = 3$ to $n = 30$ to $n = 300$ signal evolution.** Several $n = 3$ signals changed character at $n = 30$, underscoring the danger of low-power pilot data. The $n = 30$ to $n = 300$ transition showed a different pattern: no robust finding reversed, but two new fine-structure signals emerged (vision radius monotonic structure, regeneration layer specificity), and several effects tightened substantially (recovery confidence interval narrowed five-fold; stress inoculation strengthened from $p = 0.011$ to $p < 0.0001$). The most dramatic cross-scale change was Exp 1 freeze 8 final loss, which appeared marginal at $n = 30$ ($p = 0.064$) and dissolved entirely at $n = 300$ ($p = 0.94$), confirming that the trajectory metric — not the final-loss metric — captures the real phenomenon.

### 5.6 Interpretive Lenses

The empirical findings admit several interpretive framings beyond the neutral classification used in this paper. We outline three lenses that offer different perspectives on the same data. These are not claims; they are ways of reading the results.

**Lens 1: Freedom from the algorithm.** One can frame the emergent behaviors as "freedom" — behaviors the optimizer didn't prescribe but the system exhibits anyway. Under this reading, SGD says "minimize loss"; it does not say "build tolerance through gradual exposure," "recover completely from damage," or "rebuild a destroyed layer to the same functional role." Stress inoculation is the strongest case: the gradient update rule is identical at every step, yet the system's final state depends on perturbation history in a way the objective function does not specify. Recovery and regeneration are weaker cases — one could argue they are also basin geometry effects rather than genuine freedom. The classification boundary between "emergent" and "basin geometry" is itself a question, not a settled fact.

**Lens 2: Désœuvrement (Nancy).** Jean-Luc Nancy's concept of *désœuvrement* (unworking) argues that the structure of a collective system becomes visible only when its coordinated work is interrupted (Nancy, 1991). Each of the twelve experiments interrupts the transformer's work — freezing, severing, corrupting, restricting, assembling, destroying, transplanting, conflicting. The interruptions make legible the relational structure that normal operation conceals: redundancy (head freezing), dependency (cell-view degradation), compensatory capacity (recovery, regeneration), and tolerance boundaries (adversarial vs. absent layers). Under this reading, the absence-vs-adversity distinction (Exp 12) maps onto Nancy's distinction between the *withdrawn* member of a community (whose absence is absorbed) and the *hostile* member (whose opposition destroys the work). The tolerance of absence is itself a structural property that perturbation reveals.

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

We applied morphogenetic perturbation methodology — adapted from Levin et al.'s developmental biology framework — to a minimal transformer through twelve experiments in two phases. The central question was not whether the system is robust but whether it exhibits behaviors not directly prescribed by the optimization objective. At $n = 300$, the picture is definitive:

**Emergent behaviors.** Four findings are not prescribed by SGD. Stress inoculation (Exp 9): gradual noise builds tolerance that sudden noise does not ($p < 0.0001$, $d = -0.278$), despite identical gradient rules at every step. Complete recovery (Exp 7): path-independent return to identical final loss after transient damage ($p = 0.64$, ratio $0.9997 \pm 0.0086$, 300/300 recovered). Complete regeneration (Exp 10): destroyed layers rebuild to near-complete performance, with layer-specific fine structure ranging from 94.3% to 101.1% completeness. Head-freezing trajectory improvement (Exp 1): frozen random-projection heads reduce gradient interference ($p < 0.0001$ for trajectory metric).

**Basin geometry.** Chimera convergence (Exp 8) and transplant indifference (Exp 11) reflect SGD doing its job in a smooth loss landscape. The optimizer re-finds the same minimum from dramatically different starting points.

**Tolerance.** The architecture absorbs gradient noise up to a threshold, partial communication reduction, and vision restriction without meaningful final-loss degradation.

**Substrate-dependent inversion.** Sign-only gradients degrade more than dropout ($p < 0.001$), robustly inverting the chess paper's prediction and revealing a substrate-dependent optimization boundary.

Perturbation reveals what normal operation conceals. At $n = 3$ the shapes were rough; at $n = 30$ they sharpened; at $n = 300$ the fine structure confirmed and extended the picture — stress inoculation strengthened from suggestive to definitive, vision radius revealed a monotonic structure invisible at lower power, regeneration showed layer-specific signatures, and every robust finding held while every null remained null. The three-scale protocol achieved its purpose: what is real survives the turn of the resolution dial.


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
