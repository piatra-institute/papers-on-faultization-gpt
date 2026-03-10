# Unworking the Transformer: Morphogenetic Perturbation Reveals Emergent Robustness in Minimal GPTs

**Date:** March 2026


## Abstract

Levin et al. (2024) demonstrated that systematic perturbation of simple algorithms reveals emergent competencies invisible during normal operation — including fault tolerance, delayed gratification, and self-organization. We apply this morphogenetic methodology to a minimal transformer (4-layer, 16-dimensional, 4-head character-level GPT) through six experiments: freezing attention heads at random initialization, severing inter-layer gradient flow, corrupting gradient signals via noise, sign-only reduction, and quantization, restricting attention context windows, scaling inter-layer gradient flow across a communication spectrum, and crossing forward-pass stability with gradient boldness. Paired statistical analysis ($n = 30$ runs per condition, matched seeds) reveals that the architecture exhibits *bounded tolerance* rather than the compensatory rerouting observed in biological systems: mild perturbations (small noise, partial gradient flow, moderate freezing) produce no significant change in final loss, while severe gradient corruption (sign-only, quantization) degrades performance by 3–5%. The strongest individual finding is the sign-only vs. dropout comparison ($p < 0.001$), which inverts the chess-predicted "cautious position, courageous moves" strategy in a substrate-dependent manner. A designed metric — the Delayed Gratification (DG) Index — shows no systematic response to perturbation at $n = 30$, contradicting $n = 3$ pilot data and revealing that transformer perturbation responses differ qualitatively from the rerouting observed in biological and algorithmic systems. Experiments 4-6 provide an exploratory bridge to Kofman, Campitelli & Levin's (2025) distributed chess framework. The methodology is the primary contribution; the findings demonstrate both the power of the perturbation approach and the importance of adequate statistical power.


## 1. Introduction

Transformers are typically studied through their outputs: probing learned representations (Belinkov & Glass, 2019), ablating components to measure importance (Michel et al., 2019), or tracing computational circuits (Elhage et al., 2021). These methods characterize what the system has learned or which parts matter. They do not ask what happens when the system is forced to learn under constraint — when its normal operation is *interrupted*.

Levin et al. (2024) introduced a different methodology in the context of simple algorithms. Rather than analyzing sorting algorithms through their final outputs, they perturbed the algorithms during execution: freezing cells, mixing incompatible sorting directions, replacing centralized control with autonomous cell-level policies. The perturbations revealed competencies — fault tolerance, delayed gratification, emergent aggregation — that were invisible during normal operation. The central insight: **perturbation reveals what normal operation conceals**.

This insight has a philosophical precedent. Jean-Luc Nancy's concept of *désœuvrement* (unworking) argues that the structure of a collective system becomes visible only when its coordinated work is interrupted (Nancy, 1991). The components of a working system appear as interchangeable parts; when the work stops, their relational structure — redundancy, dependency, compensatory capacity — becomes legible.

We apply this methodology to a minimal GPT. Our contributions are:

1. **A methodology** that maps Levin's morphogenetic perturbation protocol to transformer training: freezing components, severing information flow, degrading optimization signals, restricting attention, scaling communication, and manipulating forward-pass stability — not to measure component importance, but to reveal emergent system-level properties.
2. **Empirical findings at $n = 30$** that delineate the architecture's tolerance boundary: mild perturbations are absorbed without significant final-loss change; severe gradient corruption (sign-only, quantization) degrades performance by 3–5%; the sign-only vs. dropout gap robustly inverts the chess paper's courage/caution prediction ($p < 0.001$).
3. **A negative result on rerouting.** A designed Delayed Gratification (DG) Index — intended to detect perturbation-triggered rerouting analogous to Levin's biological findings — shows no systematic response to perturbation at $n = 30$. This suggests that transformer perturbation responses differ qualitatively from those in biological and algorithmic systems: the response is degradation or tolerance, not compensatory rerouting.
4. **An exploratory bridge to distributed chess.** Experiments 4-6 test predictions from Kofman, Campitelli & Levin (2025), confirming the courage/caution inversion ($p < 0.001$) but finding no support for the information bottleneck or partial-communication predictions.

Nancy's *désœuvrement* provides the interpretive frame: each experiment interrupts the transformer's work (next-token prediction via backpropagation) to expose the relational structure that makes the work possible. What we find is bounded tolerance rather than the compensatory flexibility that characterizes biological development.


## 2. Related Work

Our findings intersect several established lines of research. We situate each finding against prior work to clarify what is known, what we confirm, and what is new.

**Pruning and the lottery ticket hypothesis.** Frankle & Carlin (2019) showed that trained networks contain sparse subnetworks ("winning tickets") that match full-network performance. Subsequent work extended this to structured pruning of attention heads (Michel et al., 2019; Voita et al., 2019). Our Experiment 1 freezes heads at *random initialization*, not after training — the frozen heads are arbitrary random projections. At $n = 30$, freezing does not significantly improve final loss (contrary to $n = 3$ pilot data), but produces tiny improvements in mean trajectory loss for 4+ frozen heads, suggesting reduced gradient interference.

**SignSGD and low-precision optimization.** Bernstein et al. (2018) established that sign-only gradient updates can match full-precision optimization under appropriate conditions. Our Experiment 3 shows that sign-only gradients significantly degrade performance at $n = 30$ (+4.6% final loss, $p = 0.004$), contradicting the $n = 3$ null result. The discrepancy between our finding and the SignSGD literature may reflect our model's small scale and short training duration.

**Noise as regularization.** The regularizing effect of gradient noise is well-established (Neelakantan et al., 2015). Our Experiment 3 shows that small noise ($\sigma = 0.01$) produces no significant change, consistent with noise-as-regularization, while large noise ($\sigma = 0.1$) significantly degrades performance (+2.5% final loss, $p = 0.021$), indicating a noise tolerance threshold.

**Local learning rules.** Alternatives to end-to-end backpropagation include greedy layerwise pretraining (Bengio et al., 2007), local learning signals (Nokland & Eidnes, 2019), and forward-forward algorithms (Hinton, 2022). Our cell-view experiment eliminates *all* inter-layer gradient flow. The resulting 4.9% final-loss cost ($p < 0.001$) confirms that local learning is viable but suboptimal.

**Perturbation analysis in neural networks.** Ablation studies (Meyes et al., 2019), dropout, and pruning are standard tools, but typically measure *component importance*. We use perturbation to characterize *system-level tolerance boundaries* — what the architecture absorbs versus what degrades it. This distinction connects our work to Levin's morphogenetic framework rather than to standard ablation methodology.

**Levin's morphogenetic framework.** Levin et al. (2024) applied developmental biology concepts to simple sorting algorithms. Key findings included delayed gratification (temporary performance decrease followed by recovery past pre-damage levels) and fault tolerance that exceeded intact system performance. Our results show the transformer exhibits tolerance but *not* the compensatory rerouting that characterizes Levin's systems — a substrate-dependent difference revealed by our replication at adequate statistical power.

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

### 3.2 Delayed Gratification (DG) Index

Following Levin et al., we define a metric to detect rerouting behavior — episodes where the system temporarily moves *away* from its goal before recovering past the pre-perturbation level.

**Episode detection.** We scan the loss trajectory for episodes where: (1) loss increases from a local value $L_{\text{start}}$ to a peak $L_{\text{peak}}$, then (2) decreases to a trough $L_{\text{trough}}$ below $L_{\text{start}}$. Each such episode has:

- Temporary cost: $C = L_{\text{peak}} - L_{\text{start}}$
- Net gain: $G = L_{\text{start}} - L_{\text{trough}}$

**Per-episode DG:** $\text{DG}_{\text{episode}} = G / C$

**Aggregate DG Index:** The mean DG across all detected episodes in a training run.

At $n = 30$, the DG Index shows no systematic response to perturbation (Section 4.7), contradicting $n = 3$ pilot data. DG captures a real stochastic property of loss trajectories but does not function as a perturbation response measure. This negative result is itself informative — it reveals a substrate-dependent difference between transformer training and biological development.

### 3.3 Experiment 1: Head Freezing

**Analog:** Levin's frozen-cell perturbation. Randomly selected attention heads have their parameters frozen at initialization values throughout training.

**Protocol:** Sweep over {0, 1, 2, 4, 8, 12, 16} frozen heads. Frozen heads participate in the forward pass but receive no gradient updates.

### 3.4 Experiment 2: Cell-View GPT

**Analog:** Levin's cell-view sorting algorithms / Nancy's being-singular-plural. Each transformer layer is treated as an autonomous agent.

**Protocol:** Stop-gradient operations at all layer boundaries. Each layer receives only its own local loss signal.

### 3.5 Experiment 3: Gradient Degradation

**Analog:** Levin's noisy signaling channels. Four corruption methods:

| Method | Description |
|---|---|
| Noisy ($\sigma = 0.01$) | Additive Gaussian noise, small scale |
| Sign-only | Gradient reduced to {-1, 0, +1}, magnitude discarded |
| Quantized (3-bit) | Gradient values rounded to 8 levels |
| Noisy ($\sigma = 0.1$) | Additive Gaussian noise, large scale |

### 3.6 Training Protocol and Statistical Methods

All experiments use: 200 training steps, 30 independent runs per condition (seeds 42–71), loss and per-head metrics recorded at every step. The 200-step horizon captures early learning dynamics.

**Statistical analysis.** All comparisons use two-sided paired $t$-tests, with runs matched by seed across conditions ($n = 30$, $df = 29$). Pairing by seed controls for initialization variance. We report effects as statistically significant at $p < 0.05$ and marginal at $0.05 < p < 0.10$. With 30 paired observations, statistical power is adequate to detect moderate effects (Cohen's $d \geq 0.4$ at 80% power). Effect sizes are reported as Cohen's $d$ for paired differences. We distinguish between *statistically supported* findings ($p < 0.05$) and *observational* patterns.

### 3.7 Experiment 4: Vision Radius Sweep

**Analog:** Kofman, Campitelli & Levin's (2025) vision radius experiment in distributed chess. We restrict each attention head's context window.

**Protocol:** Sweep over window sizes $\{1, 2, 4, 8, 16\}$ plus unmodified baseline. Window=16 equals block size (sanity check).

### 3.8 Experiment 5: Communication Topology

**Analog:** The chess paper's relay chains. We scale the fraction of gradient signal passed through layer boundaries.

**Protocol:** Five topologies parameterized by gradient pass fraction:

| Topology | Pass Fraction | Description |
|---|---|---|
| Full | 1.00 | Standard backpropagation (baseline) |
| Heavy | 0.75 | 75% of gradient signal passes through |
| Half | 0.50 | 50% of gradient signal passes through |
| Light | 0.25 | 25% of gradient signal passes through |
| Cell-view | 0.00 | No inter-layer gradient flow |

### 3.9 Experiment 6: Courage vs. Caution

**Analog:** Kofman et al.'s "cautious position, courageous moves" strategy. We create a $2 \times 2$ matrix:

| | Cautious Gradients | Courageous Gradients |
|---|---|---|
| **Cautious Forward** | (a) Tiny noise ($\sigma = 0.001$) | (b) Sign-only gradients |
| **Courageous Forward** | (c) Dropout ($p = 0.1$) | (d) Noisy gradients ($\sigma = 0.1$) |


## 4. Results

### 4.1 Experiment 1: Head Freezing

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

No freezing level produces a statistically significant change in final loss (freeze 8 is marginal at $p = 0.064$; all others $p > 0.18$). The $n = 3$ pilot finding that freezing 8+ heads significantly *improves* final loss ($p = 0.009$) does not replicate at $n = 30$ — the mean differences are well within the distribution's noise (Spearman $\rho = 0.013$, $p = 0.85$ for monotonicity).

Mean trajectory loss shows a statistically significant but tiny improvement for 4+ frozen heads: freeze 4 ($p = 0.012$, $d = -0.49$), freeze 8 ($p < 0.001$, $d = -1.03$), freeze 12 ($p < 0.001$, $d = -1.13$), freeze 16 ($p < 0.001$, $d = -0.96$). The effect size is 0.1–0.2% of mean loss — statistically robust but practically negligible. Frozen heads reduce gradient interference during the trajectory without altering the convergence basin.

The DG Index does not increase with freezing. Freeze 12 *decreases* DG significantly ($p = 0.034$, $d = -0.41$). All other levels are non-significant.

### 4.2 Experiment 2: Cell-View GPT

**Table 2.** Cell-view vs. baseline (means ± std across 30 runs).

| Condition | Mean Loss | Final Loss | DG Index |
|:---:|:---:|:---:|:---:|
| Baseline | 2.639 ± 0.023 | 2.469 ± 0.340 | 0.571 ± 0.312 |
| Cell-view | 2.687 ± 0.023 | 2.590 ± 0.367 | 0.507 ± 0.260 |

Eliminating all inter-layer gradient communication costs +4.9% in final loss ($p < 0.001$, $d = +1.16$) and +1.8% in mean loss ($p < 0.001$, $d = +3.31$). The degradation is significant but bounded — the system still learns effectively without its defining optimization mechanism.

The DG Index does not increase under cell-view ($p = 0.34$). The $n = 3$ claim of +25.5% DG elevation was a sampling artifact.

### 4.3 Experiment 3: Gradient Degradation

**Table 3.** Gradient degradation results (means across 30 runs).

| Method | Final Loss | Δ% | $p$ (final) | Mean Loss | $p$ (mean) | $d$ (mean) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline | 2.469 | — | — | 2.639 | — | — |
| Noisy ($\sigma = 0.01$) | 2.477 | +0.3% | 0.525 | 2.639 | 0.978 | -0.01 |
| Noisy ($\sigma = 0.1$) | 2.530 | +2.5% | 0.021* | 2.698 | <0.001*** | +3.79 |
| Sign-only | 2.582 | +4.6% | 0.004** | 2.742 | <0.001*** | +6.13 |
| Quantized (3-bit) | 2.543 | +3.0% | 0.019* | 2.703 | <0.001*** | +4.54 |

At $n = 30$, the $n = 3$ "null result" (all methods $p > 0.26$) does not replicate. Three of four methods significantly degrade final loss: sign-only (+4.6%, $p = 0.004$), quantized (+3.0%, $p = 0.019$), and noisy $\sigma = 0.1$ (+2.5%, $p = 0.021$). Only small noise ($\sigma = 0.01$) is genuinely tolerated ($p = 0.525$). Mean loss effects are highly significant for all three degrading methods ($p < 0.001$) with large Cohen's $d$ values (3.8–6.1).

The "noise helps" effect (noisy $\sigma = 0.01$ improving loss) is not supported ($p = 0.978$ for mean loss).

### 4.4 Experiment 4: Vision Radius Sweep

**Table 4.** Vision radius results (means across 30 runs).

| Window | Final Loss | $p$ (final) | Mean Loss | $p$ (mean) | $d$ (mean) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline (full) | 2.469 | — | 2.639 | — | — |
| 1 | 2.486 | 0.155 | 2.650 | <0.001*** | +2.49 |
| 2 | 2.485 | 0.136 | 2.640 | 0.185 | +0.25 |
| 4 | 2.475 | 0.524 | 2.636 | 0.004** | -0.57 |
| 8 | 2.471 | 0.478 | 2.638 | <0.001*** | -0.68 |
| 16 | 2.469 | 1.000 | 2.639 | 1.000 | 0.00 |

No window size significantly changes final loss ($p > 0.13$ for all). Window=16 reproduces baseline exactly (sanity check). Mean-loss effects exist but are negligibly small: window=1 worsens by +0.4% ($p < 0.001$), window=4 improves by -0.1% ($p = 0.004$), window=8 improves by -0.03% ($p < 0.001$). The chess paper's finding that intermediate vision radius outperforms omniscience does not produce meaningful effects for attention windowing.

### 4.5 Experiment 5: Communication Topology

**Table 5.** Communication topology results (means across 30 runs).

| Topology | Fraction | Final Loss | $p$ (final) | Mean Loss | $p$ (mean) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Full | 1.00 | 2.469 | — | 2.639 | — |
| Heavy | 0.75 | 2.472 | 0.024* | 2.639 | 0.553 |
| Half | 0.50 | 2.470 | 0.551 | 2.639 | 0.100† |
| Light | 0.25 | 2.471 | 0.426 | 2.639 | 0.303 |
| Cell-view | 0.00 | 2.590 | <0.001*** | 2.687 | <0.001*** |

Partial gradient flow (25–75%) produces no meaningful degradation. Heavy (75%) shows a statistically significant but tiny final-loss increase ($p = 0.024$, $d = +0.44$, +0.1%). Half and light are non-significant. Only cell-view (0%) degrades substantially: +4.9% final loss ($p < 0.001$), +1.8% mean loss ($p < 0.001$).

The U-shaped loss curve reported in $n = 3$ pilot data does not replicate. No partial-flow condition outperforms full backpropagation.

### 4.6 Experiment 6: Courage vs. Caution

**Table 6.** Courage vs. caution results (means across 30 runs).

| Condition | Final Loss | $p$ (final) | Mean Loss | $p$ (mean) | $d$ (mean) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline | 2.469 | — | 2.639 | — | — |
| Cautious/Cautious (a) | 2.484 | 0.017* | 2.639 | 0.799 | -0.05 |
| Cautious/Courageous (b) | 2.582 | 0.004** | 2.742 | <0.001*** | +6.13 |
| Courageous/Cautious (c) | 2.481 | 0.029* | 2.641 | 0.010** | +0.50 |
| Courageous/Courageous (d) | 2.509 | 0.044* | 2.697 | <0.001*** | +4.33 |

All conditions significantly degrade final loss versus baseline ($p < 0.05$ for all). Sign-only (b) shows the largest degradation (+4.6% final loss, $p = 0.004$; +3.9% mean loss, $p < 0.001$).

The sign-only vs. dropout comparison is the paper's most robust finding. Paired comparison (matched seeds): sign-only worsens mean loss by +3.8% relative to dropout ($p < 0.001$, $d = +6.00$); final loss by +4.1% ($p = 0.010$, $d = +0.51$). The chess paper's predicted best condition (b, "cautious position, courageous moves") is the worst non-baseline condition.

### 4.7 Cross-Experiment Synthesis

**Table 7.** Performance across all perturbation types (paired $t$-tests, $n = 30$).

| Perturbation | Final Loss Δ% | $p$ (final) | Mean Loss Δ% | $p$ (mean) | Status |
|---|---|---|---|---|---|
| Freeze 1-4 heads | +0.1-0.2% | >0.37 | -0.0% | 0.01-0.81 | Tolerated |
| Freeze 8-16 heads | +0.4-0.5% | 0.06-0.19 | -0.1-0.2% | <0.001 | Trajectory only |
| Window=4,8 | +0.1-0.3% | >0.47 | -0.03-0.1% | <0.004 | Tiny trajectory |
| Partial flow (25-75%) | +0.0-0.1% | >0.02 | ±0.0% | >0.10 | Tolerated |
| Noisy σ=0.01 | +0.3% | 0.53 | -0.0% | 0.98 | Tolerated |
| Dropout (p=0.1) | +0.5% | 0.029 | +0.1% | 0.010 | Mild degradation |
| Noisy σ=0.1 | +2.5% | 0.021 | +2.2% | <0.001 | Degradation |
| Quantized 3-bit | +3.0% | 0.019 | +2.4% | <0.001 | Degradation |
| Sign-only | +4.6% | 0.004 | +3.9% | <0.001 | Degradation |
| Cell-view | +4.9% | <0.001 | +1.8% | <0.001 | Degradation |

The pattern is *bounded tolerance*: perturbations preserving approximate gradient direction (small noise, partial flow, freezing) are absorbed; perturbations destroying gradient information (sign-only, quantization, full stop-gradient) produce 3-5% degradation. The worst-case degradation (cell-view, +4.9% final loss) is bounded — the architecture works even when backpropagation is eliminated entirely.

**DG does not track perturbation.** At $n = 3$, the DG Index appeared to scale with perturbation severity. At $n = 30$, this finding does not replicate: no perturbation condition produces a statistically significant DG increase ($p > 0.16$ for all). Two conditions significantly *decrease* DG: freeze 12 ($p = 0.034$) and quantized 3-bit ($p = 0.028$). Baseline DG = 0.571 ± 0.312; all conditions fall within 0.447–0.606 with overlapping distributions. The DG metric captures a stochastic property of SGD loss trajectories, not a perturbation response.

### 4.8 Findings — What Perturbation Revealed

#### Statistically Robust Findings

**Finding 1: The sign-only vs. dropout inversion.** Sign-only gradients degrade performance significantly more than dropout across multiple experiments (Exp 3: sign-only +4.6% final loss, $p = 0.004$; Exp 6 paired comparison: $p < 0.001$, $d = +6.00$ for mean loss). This inverts the chess paper's "cautious position, courageous moves" prediction and is the paper's most replicable result.

**Finding 2: Cell-view is viable but costly.** Eliminating all inter-layer gradient flow degrades final loss by +4.9% ($p < 0.001$, $d = +1.16$). The degradation is bounded and the system still learns — each layer learns independently.

**Finding 3: Gradient quality matters more than gradient quantity.** The sharpest degradations come from reducing gradient *precision* (sign-only: $d = +6.13$; quantized: $d = +4.54$) rather than gradient *magnitude* (partial flow: all $d < 0.5$) or *completeness* (freezing: all final-loss $p > 0.06$). The optimization signal's directional accuracy is more critical than its strength or coverage.

**Finding 4: Tolerance has a clear boundary.** Perturbations that preserve approximate gradient direction (small noise, partial flow) produce no significant final-loss change. Perturbations that corrupt gradient information (sign-only, quantization, large noise) produce significant 2.5-4.6% degradation. The boundary is sharp.

#### Not Supported or Retracted

- **"Damage improves" (final loss):** Does not replicate at $n = 30$ ($p > 0.06$ for all freezing levels). **Retracted.**
- **"DG scales with perturbation":** Does not replicate ($p > 0.16$ for all DG increases). **Retracted.**
- **"Noise helps":** Not supported ($p > 0.5$ for noisy $\sigma = 0.01$). **Retracted.**
- **"Partial communication outperforms full":** Not supported ($p > 0.43$ for light topology). **Retracted.**
- **"Information bottleneck is beneficial":** No significant final-loss improvements under any restriction. **Retracted.**


## 5. Discussion

### 5.1 The Nancy Reading

Nancy's concept of *désœuvrement* — the interruption of work that reveals the community constituted by work — provides the interpretive frame for these findings.

During standard backpropagation, the transformer's components cooperate invisibly. Each experiment interrupts this work — freezing components, severing gradient flow, corrupting signals, restricting vision, scaling communication, introducing forward noise. The interruptions reveal:

**Bounded tolerance, not compensatory rerouting.** The transformer absorbs mild perturbations and degrades modestly under severe ones. It does not exhibit the compensatory rerouting observed in Levin's biological systems — damaged organisms that outperform intact ones. The DG metric, designed to detect such rerouting, shows no perturbation response at $n = 30$. This is itself a finding about substrate differences: the continuous optimization landscape may not support the discrete reorganization events that characterize developmental biology.

**Structural redundancy.** The residual stream, multi-head attention, and MLP layers provide redundant computational pathways. When some pathways are disrupted (freezing, partial gradient flow), others absorb the load without measurable degradation. But this redundancy has limits — eliminate enough signal (cell-view, sign-only) and performance degrades.

**Gradient precision as the critical resource.** The sharpest degradations come from corrupting gradient *quality*, not reducing gradient *quantity*. This reveals what the optimization process actually needs: approximate direction, not magnitude. Partial flow (25%) is tolerated; sign-only (which preserves direction but discards magnitude scaling) is not — suggesting that the relative magnitudes between gradient components carry essential information for navigating the loss landscape.

**Decentralized viability.** Cell-view training shows that each layer can learn autonomously at bounded cost. The layers are always simultaneously individual and communal — but global backprop makes their individuality invisible.

### 5.2 Two Claims

Two claims emerge from the combined evidence. The $n = 30$ replication constrains these claims tightly.

**Claim 1: Robustness is structural and bounded.** The transformer architecture — residual stream, multi-head attention, MLP layers — provides inherent tolerance to mild perturbation. No perturbation produces a final-loss degradation exceeding ~5%. But this tolerance has clear limits: perturbations corrupting gradient precision (sign-only, quantization) produce significant degradation. The robustness is not a property of the training algorithm; it arises from architectural redundancy.

**Claim 2: The chess-paper courage/caution strategy inverts across substrates.** Sign-only gradients (cautious perception, courageous moves) degrade performance significantly more than dropout (courageous perception, cautious moves), $p < 0.001$. This substrate-dependent inversion reveals that the optimal perturbation-tolerance strategy depends on whether the action space is discrete and irreversible (chess) or continuous and differentiable (transformers).

*Note on removed claims.* Three claims from the $n = 3$ pilot analysis are retracted: "DG tracks perturbation response" (no significant DG changes), "perturbation can be beneficial" (freezing does not improve final loss), and "centralized control is optional" (cell-view produces significant degradation). Cell-view learning is *viable* (+4.9% cost) but the framing as "optional" overstated the case.

### 5.3 Connection to Distributed Chess

Kofman, Campitelli & Levin (2025) implemented a distributed form of chess where each piece operates as an autonomous agent. Experiments 4-6 tested three predictions.

**Information bottleneck as beneficial constraint (not supported).** The chess paper's central result — intermediate vision radius $R4$ outperforms omniscient $R7$ — does not translate to attention windowing. No window size significantly changes final loss at $n = 30$. The mean-loss effects (window=4: -0.1%, window=8: -0.03%) are too small to constitute a meaningful information bottleneck benefit.

**Partial communication tolerance (confirmed as tolerance, not improvement).** Reducing gradient flow to 25% produces no significant degradation — consistent with tolerance. But partial flow does not *improve* over full backpropagation. The chess paper's relay-chain benefits do not have an analog in gradient-based optimization at this scale.

**Courage/caution strategy (robustly inverted).** The paper's strongest finding. Sign-only (b) degrades mean loss by +3.8% relative to dropout (c), $p < 0.001$, $d = +6.00$. The chess paper's optimal strategy produces the worst transformer outcome. This inversion is substrate-dependent: chess requires stable perception because moves are irreversible; transformers benefit from forward-pass noise (regularization) because the optimization landscape is continuous.

### 5.4 Limitations

**Scale.** The model has 4 layers, 16 dimensions, and ~11,000 parameters. Whether these findings extend to production-scale transformers is unknown.

**Task complexity.** Character-level name generation is a toy task.

**Training duration.** 200 steps captures early learning dynamics but not long-horizon phenomena.

**DG metric.** The DG Index does not respond to perturbation at $n = 30$, contradicting $n = 3$ pilot data. The metric captures a stochastic property of loss trajectories but does not function as a perturbation response measure in transformers. Its relationship to biological delayed gratification is questionable.

**Effect sizes.** Many statistically significant effects (mean-loss improvements under freezing, window effects) are practically negligible (<0.5%). Statistical significance at $n = 30$ does not imply practical importance.

**$n = 3$ vs. $n = 30$ divergence.** Several $n = 3$ findings did not replicate: the "damage improves" finding ($n = 3$: $p = 0.009$; $n = 30$: $p > 0.06$), the DG-perturbation scaling, and the gradient degradation null result. This highlights the danger of interpreting toy-scale pilot data with low statistical power.

**Chess-paper translation fidelity.** The courage/caution inversion may reflect genuine substrate differences. Whether richer translations of the chess paper's 13-gene behavioral chromosome would produce different results remains open.

### 5.5 Future Work

- **Scale:** Replicate at 100M+ parameter scale to test whether tolerance boundaries shift with model size.
- **Task diversity:** Apply the perturbation protocol to arithmetic, reasoning, and multi-modal tasks.
- **Chimeric experiments:** Interleave layers from separately trained models.
- **Architecture morphogenesis:** Allow the architecture to change during training — growing heads, pruning inactive ones.
- **Composite perturbation:** Simultaneously apply multiple perturbation types (dropout + sign-only) for more faithful courage/caution testing.
- **DG alternatives:** Design perturbation-response metrics that might capture rerouting in continuous optimization landscapes, if such rerouting exists.
- **Gradient precision analysis:** Investigate *why* sign-only degrades more than partial flow — is it the loss of relative magnitude information, or something else?


## 6. Conclusion

We applied morphogenetic perturbation methodology — adapted from Levin et al.'s developmental biology framework — to a minimal transformer through six experiments. At $n = 30$, the results reveal bounded tolerance rather than compensatory rerouting:

The architecture absorbs mild perturbations (small noise, partial gradient flow, moderate freezing) without significant final-loss change, but degrades measurably (3-5%) under severe gradient corruption (sign-only, quantization, full stop-gradient). The sign-only vs. dropout comparison robustly inverts the chess paper's "cautious position, courageous moves" strategy ($p < 0.001$), revealing a substrate-dependent boundary between discrete and continuous optimization.

The DG Index — designed to detect perturbation-triggered rerouting analogous to biological delayed gratification — shows no systematic perturbation response at $n = 30$, contradicting $n = 3$ pilot data. This negative result is itself informative: the transformer's response to perturbation is degradation or tolerance, not the compensatory reorganization observed in biological systems. The morphogenetic analogy holds for the methodology (systematic perturbation reveals hidden structure) but not for the specific mechanism (rerouting).

The $n = 3$ to $n = 30$ replication also retracted several findings: "damage improves" final loss, "monotonic improvement" with freezing, "noise helps," "partial communication outperforms full," and the entire DG-scales-with-perturbation narrative. This underscores the importance of adequate statistical power for interpreting perturbation experiments, even at toy scale.

Transformers reveal their structure through interruption — but what they reveal is architectural redundancy, not developmental competency. The work conceals the community; the unworking shows its tolerance.


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
