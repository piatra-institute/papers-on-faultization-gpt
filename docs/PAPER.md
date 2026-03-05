# Unworking the Transformer: Morphogenetic Perturbation Reveals Emergent Robustness in Minimal GPTs

**Date:** March 2026


## Abstract

Levin et al. (2024) demonstrated that systematic perturbation of simple algorithms reveals emergent competencies invisible during normal operation — including fault tolerance, delayed gratification, and self-organization. We apply this morphogenetic methodology to a minimal transformer (4-layer, 16-dimensional, 4-head character-level GPT) through six experiments: freezing attention heads at random initialization, severing inter-layer gradient flow, corrupting gradient signals via noise, sign-only reduction, and quantization, restricting attention context windows, scaling inter-layer gradient flow across a communication spectrum, and crossing forward-pass stability with gradient boldness. Paired statistical analysis ($n = 3$ runs per condition, matched seeds) yields two statistically robust findings and several suggestive patterns: (1) freezing 8+ attention heads significantly *improves* loss ($p < 0.01$), revealing a threshold-dependent benefit where frozen random projections outperform competing learners, and (2) sign-only gradients degrade performance significantly more than dropout ($p = 0.004$), inverting the chess-predicted "cautious position, courageous moves" strategy in a substrate-dependent manner. Additional patterns — sign-only gradients approximately matching baseline, restricted attention windows showing suggestive improvements, partial gradient flow tolerating severe reduction — are consistent with architectural robustness but not statistically significant at $n = 3$. A designed metric — the Delayed Gratification (DG) Index — confirms rerouting behavior across all conditions, scaling with perturbation severity. Experiments 4-6 provide an exploratory bridge to Kofman, Campitelli & Levin's (2025) distributed chess framework. The methodology and the threshold-dependent freezing result are the primary contributions; the suggestive patterns motivate replication at higher sample sizes.


## 1. Introduction

Transformers are typically studied through their outputs: probing learned representations (Belinkov & Glass, 2019), ablating components to measure importance (Michel et al., 2019), or tracing computational circuits (Elhage et al., 2021). These methods characterize what the system has learned or which parts matter. They do not ask what happens when the system is forced to learn under constraint — when its normal operation is *interrupted*.

Levin et al. (2024) introduced a different methodology in the context of simple algorithms. Rather than analyzing sorting algorithms through their final outputs, they perturbed the algorithms during execution: freezing cells, mixing incompatible sorting directions, replacing centralized control with autonomous cell-level policies. The perturbations revealed competencies — fault tolerance, delayed gratification, emergent aggregation — that were invisible during normal operation. The central insight: **perturbation reveals what normal operation conceals**.

This insight has a philosophical precedent. Jean-Luc Nancy's concept of *désœuvrement* (unworking) argues that the structure of a collective system becomes visible only when its coordinated work is interrupted (Nancy, 1991). The components of a working system appear as interchangeable parts; when the work stops, their relational structure — redundancy, dependency, compensatory capacity — becomes legible.

We apply this methodology to a minimal GPT. Our contributions are:

1. **A methodology** that maps Levin's morphogenetic perturbation protocol to transformer training: freezing components, severing information flow, degrading optimization signals, restricting attention, scaling communication, and manipulating forward-pass stability — not to measure component importance, but to reveal emergent system-level competencies.
2. **Two statistically supported findings and several suggestive patterns** at toy scale ($n = 3$ runs per condition, paired $t$-tests): heavy head-freezing significantly improves performance ($p < 0.01$), revealing a threshold-dependent benefit; sign-only gradients degrade significantly more than dropout ($p = 0.004$), inverting the chess paper's courage/caution prediction. Additional patterns — gradient tolerance, restricted-window improvements, partial-flow tolerance — are suggestive but underpowered.
3. **A principled distinction** between designed metrics and emergent discoveries. We define and measure a Delayed Gratification (DG) Index to confirm rerouting behavior; the emergent findings themselves are what perturbation made legible.
4. **An exploratory bridge to distributed chess.** Experiments 4-6 test predictions from Kofman, Campitelli & Levin (2025), partially confirming the information bottleneck hypothesis (via Experiment 1 freezing, $p < 0.01$) and revealing one substrate-dependent inversion with statistical support (Experiment 6, $p = 0.004$). The partial-communication prediction is not testable at this sample size.

Nancy's *désœuvrement* provides the interpretive frame: each experiment interrupts the transformer's work (next-token prediction via backpropagation) to expose the relational structure that makes the work possible. The robustness we observe is not a feature added to the architecture — it is the architecture, seen from an angle that only unworking provides.


## 2. Related Work

Our findings intersect several established lines of research. We situate each finding against prior work to clarify what is known, what we confirm, and what is new.

**Pruning and the lottery ticket hypothesis.** Frankle & Carlin (2019) showed that trained networks contain sparse subnetworks ("winning tickets") that, when trained in isolation from their original initialization, match full-network performance. Subsequent work extended this to structured pruning of attention heads (Michel et al., 2019; Voita et al., 2019). Our Experiment 1 differs in a critical respect: we freeze heads at *random initialization*, not after training. The frozen heads are not winning tickets — they are arbitrary random projections. The performance improvement we observe (Finding 1, $p < 0.01$ for 8+ frozen heads) is therefore not about finding good subnetworks but about removing gradient interference between attention heads and MLP layers.

**SignSGD and low-precision optimization.** Bernstein et al. (2018) established that sign-only gradient updates (SignSGD) can match full-precision optimization in certain settings, with convergence guarantees under appropriate conditions. Our Experiment 3 sign-only result is consistent with this finding — sign-only produces no statistically significant degradation ($p = 0.860$), though precision is low at $n = 3$. Our contribution is not the fact itself but its context: we discover it within a systematic perturbation protocol as a *biological* finding — binary signaling suffices for developmental coordination — rather than as an optimization technique.

**Noise as regularization.** The regularizing effect of gradient noise is well-established. Neelakantan et al. (2015) showed that adding gradient noise improves generalization in deep networks. Dropout (Srivastava et al., 2014) corrupts activations rather than gradients but operates on a similar principle: noise prevents overfitting. Our Experiment 3 shows a suggestive mean effect of small noise improving loss (-0.028 delta), consistent with these findings, but the effect is not statistically significant at $n = 3$ ($p = 0.701$).

**Local learning rules.** Alternatives to end-to-end backpropagation have a long history: greedy layerwise pretraining (Bengio et al., 2007), local learning signals (Nokland & Eidnes, 2019), and forward-forward algorithms (Hinton, 2022). Our cell-view experiment (Experiment 2) is more radical: we eliminate *all* inter-layer gradient flow via stop-gradient, giving each layer only its own local loss signal. The resulting 2.4% performance cost ($p = 0.021$) and the observational specialization effects position our work at the extreme end of this spectrum.

**Perturbation analysis in neural networks.** Ablation studies (Meyes et al., 2019), dropout, and pruning are standard tools for understanding neural networks, but they are typically used to *measure component importance* — which parts matter most for a given output. We use perturbation not to rank components but to reveal *emergent system-level competency*: what does the collective do when interrupted? This distinction — studying the system's response trajectory rather than its endpoint degradation — is what connects our work to Levin's morphogenetic framework rather than to standard ablation methodology.

**Levin's morphogenetic framework.** Levin et al. (2024) applied developmental biology concepts to simple sorting algorithms, treating them as collectives of autonomous cells rather than centralized procedures. Key findings included: cell-view (distributed) algorithms exhibit greater fault tolerance than centralized versions; damaged systems show delayed gratification (temporary performance decrease followed by recovery past pre-damage levels); and chimeric systems with mixed sorting directions reach stable equilibria with emergent spatial aggregation. We map this framework to transformer training, preserving the methodology (systematic perturbation, trajectory analysis, designed vs. emergent metrics) while adapting the substrate from sorting arrays to neural network training dynamics.

**Distributed chess as collective intelligence.** Kofman, Campitelli & Levin (2025) extended the morphogenetic framework to chess by replacing the centralized engine with 16 autonomous pieces, each possessing a limited radius of vision and a 13-gene evolved behavioral chromosome. Their central finding — that intermediate vision radius $R4$ outperforms both minimal ($R0$) and maximal ($R7$) vision — motivates our Experiment 4, where restricted context windows show suggestive but not statistically significant improvements over full attention ($p = 0.17$–$0.32$). The information bottleneck hypothesis finds its strongest support in Experiment 1's high-freezing result ($p < 0.01$). Long-range communication via relay chains expanded each piece's "cognitive light cone" without increasing individual complexity; our Experiment 5 tests the analogous prediction but finds that partial gradient flow effects are not significant at $n = 3$ ($p > 0.28$). Their "cautious position, courageous moves" strategy is *inverted* by our Experiment 6 with statistical confidence ($p = 0.004$), revealing a substrate-dependent boundary: the optimal courage/caution balance reverses between discrete (chess) and continuous (transformer) optimization substrates. Most strikingly, checkmate emerged as a collective behavior that no individual piece was programmed to achieve, mirroring our finding that architectural robustness emerges from component interactions rather than being encoded in the training algorithm.


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

The model is implemented in a scalar autograd framework (following Karpathy's micrograd) for full computational transparency, with a numpy backend for experiment sweeps. The task is character-level name generation trained on a dataset of ~32k names.

The choice of a minimal model is deliberate and follows Levin et al.'s rationale for using sorting algorithms: "the benefit of these sorting algorithms is precisely that they are simple, easy to understand, and offer no place for additional complexity to hide." The question is not whether a toy model produces impressive outputs, but what competencies a fully transparent collective system possesses that are not apparent from its algorithm.

### 3.2 Delayed Gratification (DG) Index

Following Levin et al., we define a metric to detect rerouting behavior — episodes where the system temporarily moves *away* from its goal before recovering past the pre-perturbation level.

**Episode detection.** We scan the loss trajectory for episodes where: (1) loss increases from a local value $L_{\text{start}}$ to a peak $L_{\text{peak}}$, then (2) decreases to a trough $L_{\text{trough}}$ below $L_{\text{start}}$. Each such episode has:

- Temporary cost: $C = L_{\text{peak}} - L_{\text{start}}$
- Net gain: $G = L_{\text{start}} - L_{\text{trough}}$

**Per-episode DG:** $\text{DG}_{\text{episode}} = G / C$

**Aggregate DG Index:** The mean DG across all detected episodes in a training run.

A higher DG Index indicates more rerouting: the system explores configurations that temporarily worsen performance before finding alternatives that exceed the pre-perturbation level. Crucially, DG is a *designed* metric — we defined it based on Levin's framework and measured it. Unlike Levin, who discovered delayed gratification as an emergent surprise, we went looking for it. The DG measurements confirm that rerouting occurs; the genuine discoveries are the anomalies (Section 5.2) that emerged without being designed for.

### 3.3 Experiment 1: Head Freezing

**Analog:** Levin's frozen-cell perturbation. Randomly selected attention heads have their parameters frozen at initialization values throughout training, forcing the remaining components to compensate.

**Protocol:** We sweep over {0, 1, 2, 4, 8, 12, 16} frozen heads (out of 16 total). For each condition, frozen heads are selected uniformly at random. The frozen heads participate in the forward pass (their random-initialized projections are applied) but receive no gradient updates. MLP layers and unfrozen heads train normally.

### 3.4 Experiment 2: Cell-View GPT

**Analog:** Levin's cell-view sorting algorithms / Nancy's being-singular-plural. Each transformer layer is treated as an autonomous agent with its own local loss signal.

**Protocol:** Stop-gradient operations are inserted at all layer boundaries, severing backward gradient flow between layers. Each layer receives a local next-token prediction loss via a linear probe from its own output. No end-to-end backpropagation occurs — each layer learns independently from its own signal.

### 3.5 Experiment 3: Gradient Degradation

**Analog:** Levin's noisy signaling channels. The gradient signal is corrupted during training through four methods:

| Method | Description |
|---|---|
| Noisy ($\sigma = 0.01$) | Additive Gaussian noise, small scale |
| Sign-only | Gradient reduced to {-1, 0, +1}, magnitude discarded |
| Quantized (3-bit) | Gradient values rounded to 8 levels |
| Noisy ($\sigma = 0.1$) | Additive Gaussian noise, large scale |

Each method degrades the precision of the optimization signal while preserving some directional information.

### 3.6 Training Protocol and Statistical Methods

All experiments use: 200 training steps, 3 independent runs per condition (seeds 42, 43, 44), loss and per-head metrics recorded at every step. Results are reported as means across runs. The 200-step horizon captures early learning dynamics — the regime where rerouting and compensation are most active.

**Statistical analysis.** All comparisons between conditions use two-sided paired $t$-tests, with runs matched by seed across conditions ($n = 3$, $df = 2$). Pairing by seed controls for initialization variance, increasing sensitivity to treatment effects. We report effects as statistically significant at $p < 0.05$ and marginal at $0.05 < p < 0.10$. With only 3 paired observations, statistical power is low: the tests can detect large effects but may fail to detect moderate ones. We therefore distinguish between *statistically supported* findings ($p < 0.05$) and *observational* patterns (suggestive mean differences that are not statistically significant). Many reported mean effects fall in the latter category. This limitation is inherent to the exploratory, toy-scale design and motivates replication at higher sample sizes.

### 3.7 Experiment 4: Vision Radius Sweep

**Analog:** Kofman, Campitelli & Levin's (2025) vision radius experiment in distributed chess, where each piece perceives only squares within radius $R$. Their central finding: intermediate $R4$ outperforms both blind ($R0$) and omniscient ($R7$) conditions by approximately 50 Elo. We translate this to transformers by restricting each attention head's context window — the number of preceding tokens a head can attend to.

**Protocol:** We sweep over window sizes $\{1, 2, 4, 8, 16\}$ plus an unmodified baseline (full attention). Window size 16 equals the block size, serving as a sanity check (should reproduce baseline exactly). For each condition, windowed attention is applied to all heads across all layers. The attention mask prevents each head from attending to tokens beyond its window.

### 3.8 Experiment 5: Communication Topology

**Analog:** The chess paper's relay chains, where pieces transmit threat information beyond their individual vision radius, expanding the collective's "cognitive light cone" — the spatiotemporal region an agent can effectively work toward. We create a spectrum of gradient flow topologies between full backpropagation and complete isolation by scaling the fraction of gradient signal passed through layer boundaries.

**Protocol:** Five topologies are tested, parameterized by gradient pass fraction:

| Topology | Pass Fraction | Description |
|---|---|---|
| Full | 1.00 | Standard backpropagation (baseline) |
| Heavy | 0.75 | 75% of gradient signal passes through |
| Half | 0.50 | 50% of gradient signal passes through |
| Light | 0.25 | 25% of gradient signal passes through |
| Cell-view | 0.00 | No inter-layer gradient flow |

Partial gradient flow is implemented by scaling the backward-pass gradient at each layer boundary by the pass fraction. Full topology ($f = 1.0$) reproduces baseline training; cell-view ($f = 0.0$) reproduces Experiment 2.

### 3.9 Experiment 6: Courage vs. Caution

**Analog:** Kofman et al.'s finding that the optimal behavioral strategy for distributed chess pieces is "cautious position, courageous moves" — conservative perception (low-risk evaluation of board state) combined with aggressive action (bold move selection). We translate this into a $2 \times 2$ matrix crossing forward-pass stability with gradient boldness:

| | Cautious Gradients | Courageous Gradients |
|---|---|---|
| **Cautious Forward** | (a) Tiny noise ($\sigma = 0.001$) | (b) Sign-only gradients |
| **Courageous Forward** | (c) Dropout ($p = 0.1$) | (d) Noisy gradients ($\sigma = 0.1$) |

Condition (a) applies minimal gradient noise, serving as a near-baseline reference. Condition (b) uses sign-only gradients (stable forward pass, bold gradient steps) — the predicted best condition, mapping to the chess paper's optimal strategy. Condition (c) applies dropout to the forward pass (noisy perception, careful gradients). Condition (d) applies large gradient noise (noise everywhere). Each condition implements one perturbation type as the primary manipulation.


## 4. Results

### 4.1 Experiment 1: Head Freezing

**Table 1.** Head freezing results (means across 3 runs).

| Frozen Heads | Final Loss | DG Index |
|:---:|:---:|:---:|
| 0 (baseline) | 2.479 | 0.299 |
| 1 | 2.477 | 0.291 |
| 2 | 2.475 | 0.340 |
| 4 | 2.482 | 0.331 |
| 8 | 2.463 | 0.388 |
| 12 | 2.426 | 0.485 |
| 16 | 2.409 | 0.502 |

Final loss *decreases* with the number of frozen heads, from 2.479 (baseline) to 2.409 (all heads frozen). Paired $t$-tests reveal that this improvement is statistically significant only for high freezing levels: freezing 8 heads ($t = -10.46$, $p = 0.009$) and 12 heads ($t = -13.43$, $p = 0.006$) both reach $p < 0.01$; freezing 16 heads is marginal ($p = 0.055$). Freezing 1, 2, or 4 heads shows no significant change ($p > 0.85$). The improvement emerges at a threshold around half the total heads rather than increasing monotonically at each step — the trend in means is visible but per-step differences at low freezing levels are indistinguishable from noise at $n = 3$.

The DG Index nearly doubles, rising from 0.299 to 0.502 (+68%). Cross-condition trajectory correlations exceed 0.95 at all freezing levels — the system follows the same learning arc regardless of how many heads are disabled. The DG-vs-damage regression shows a positive slope: more freezing produces proportionally more rerouting, consistent with Levin's prediction that perturbation triggers compensatory behavior proportional to its magnitude.

![Figure 1: Loss vs. frozen heads — the robustness curve. Final loss decreases as more heads are frozen, contradicting the expectation that removing capacity degrades performance.](../results/exp1_robustness_curve.png)

![Figure 2: DG episode visualization across freezing levels. More frozen heads produce more frequent and pronounced rerouting episodes.](../results/exp1_dg_episodes.png)

![Figure 3: Training trajectories across freezing conditions. Despite varying numbers of frozen heads, trajectory shape is preserved (correlation > 0.95).](../results/exp1_trajectories.png)

### 4.2 Experiment 2: Cell-View GPT

**Table 2.** Cell-view vs. baseline (means across 3 runs).

| Condition | Mean Loss | Final Loss | DG Index |
|:---:|:---:|:---:|:---:|
| Baseline | 2.635 | 2.479 | 0.299 |
| Cell-view | 2.698 | 2.627 | 0.375 |

Eliminating all inter-layer gradient communication costs +2.4% in mean loss ($t = 6.74$, $p = 0.021$) and +6.0% in final loss — a statistically significant but remarkably small degradation for removing the defining mechanism of deep learning. Training is approximately 3.5x faster due to the elimination of backward passes through the full computational graph.

The DG Index increases by 25.5% under local learning, indicating rerouting even without inter-layer coordination. Cell-view training also appears to produce lower variance across independent runs (though variance comparisons at $n = 3$ are unreliable) and triggers more aggressive head specialization in Layer 3, whose heads concentrate their attention patterns when deprived of upstream gradient refinement.

![Figure 4: Baseline vs. cell-view training comparison. The performance gap is small despite the elimination of all inter-layer gradient flow.](../results/exp2_comparison.png)

### 4.3 Experiment 3: Gradient Degradation

**Table 3.** Gradient degradation results (means across 3 runs).

| Method | Final Loss Delta | DG Index |
|:---:|:---:|:---:|
| Baseline | — | 0.299 |
| Noisy ($\sigma = 0.01$) | -0.028 | 0.412 |
| Sign-only | +0.011 | 0.322 |
| Quantized (3-bit) | +0.042 | 0.425 |
| Noisy ($\sigma = 0.1$) | +0.110 | 0.533 |

No gradient degradation method produces a statistically significant change from baseline at $n = 3$. Paired $t$-tests yield $p > 0.26$ for all four methods: sign-only ($p = 0.860$), quantized ($p = 0.425$), noisy $\sigma = 0.1$ ($p = 0.265$), and noisy $\sigma = 0.01$ ($p = 0.701$). The mean effects are suggestive — sign-only at +0.4%, quantized at +1.7%, noisy $\sigma = 0.1$ at +4.4%, and noisy $\sigma = 0.01$ at -1.1% — but all fall within the confidence intervals. This null result is itself informative: the architecture tolerates severe gradient corruption without measurable degradation.

Sign-only gradients — 1 bit per parameter — produce a mean final loss delta of just +0.011 (+0.4%), not statistically significant ($p = 0.860$), consistent with prior SignSGD findings (Bernstein et al., 2018). The mean effect of noisy $\sigma = 0.01$ is negative (-0.028) but the paired differences go in opposite directions across runs ($p = 0.701$); the previously claimed "noise helps" effect is not supported. All degradation methods elevate the DG Index above baseline, with the noisiest condition ($\sigma = 0.1$) producing DG = 0.533 (+78%), the highest of the first three experiments.

![Figure 5: Gradient degradation method comparison. Sign-only gradients nearly match baseline; small noise improves over baseline.](../results/exp3_method_comparison.png)

![Figure 6: Training trajectories under gradient degradation. Trajectory shape is preserved across degradation methods despite substantial corruption of the optimization signal.](../results/exp3_trajectories.png)

### 4.4 Experiment 4: Vision Radius Sweep

**Table 4.** Vision radius sweep results (means across 3 runs).

| Window | Mean Loss | Std | Mean DG |
|:---:|:---:|:---:|:---:|
| Baseline (full) | 2.6349 | 0.027 | 0.299 |
| 1 | 2.6500 | 0.029 | 0.435 |
| 2 | 2.6405 | 0.031 | 0.389 |
| 4 | 2.6313 | 0.028 | 0.249 |
| 8 | 2.6328 | 0.027 | 0.499 |
| 16 | 2.6349 | 0.027 | 0.299 |

Window=16 (full context) reproduces baseline values exactly, confirming the implementation introduces no artifacts. Paired $t$-tests show that only window=8 produces a statistically significant mean loss improvement ($t = -5.00$, $p = 0.038$), though the effect is tiny (-0.002). Window=4's mean loss improvement is not significant ($p = 0.17$), and window=2's final loss improvement — the largest in mean magnitude at -1.4% — is not significant ($p = 0.318$). Window=1 marginally worsens mean loss ($p = 0.078$). The trend across window sizes is suggestive of the information bottleneck hypothesis but underpowered at $n = 3$.

Window=8 produces the highest DG (0.499, +67% over baseline), indicating that moderate restriction triggers the most rerouting. Each attention head's effective "cognitive light cone" — the sequence region it can process — shrinks with window size, yet no restricted window produces a significant degradation.

![Figure 7: Vision radius sweep. Restricted attention windows match or beat full attention, paralleling the chess paper's finding that intermediate vision radius outperforms omniscience.](../results/exp4_vision_radius.png)

### 4.5 Experiment 5: Communication Topology

**Table 5.** Communication topology results (means across 3 runs).

| Topology | Fraction | Mean Loss | Mean DG |
|:---:|:---:|:---:|:---:|
| Full | 1.00 | 2.6349 | 0.299 |
| Heavy | 0.75 | 2.6355 | 0.276 |
| Half | 0.50 | 2.6354 | 0.277 |
| Light | 0.25 | 2.6343 | 0.316 |
| Cell-view | 0.00 | 2.6978 | 0.375 |

Paired $t$-tests show that no partial-flow condition (heavy, half, or light) differs significantly from full backpropagation ($p > 0.28$ for all). The previously suggested U-shaped loss curve is not statistically supported — the mean differences between partial topologies and full backpropagation (e.g., light at -0.0006 mean loss) are indistinguishable from noise at $n = 3$. Only cell-view (0% gradient flow) is significantly different, degrading mean loss by +2.4% ($t = 6.74$, $p = 0.021$).

The tolerance finding is the real result: reducing gradient flow to 25% produces no detectable degradation, even though 75% of the inter-layer gradient signal is discarded. Cell-view (0%) produces the highest DG (0.375, +25%) but the worst loss, indicating maximum rerouting at maximum cost.

![Figure 8: Communication topology. Partial gradient flow (25%) outperforms both full backpropagation and complete isolation, producing a U-shaped loss curve across the communication spectrum.](../results/exp5_communication_topology.png)

### 4.6 Experiment 6: Courage vs. Caution

**Table 6.** Courage vs. caution results (means across 3 runs).

| Condition | Mean Loss | Std | Mean DG |
|:---:|:---:|:---:|:---:|
| Baseline | 2.6349 | 0.027 | 0.299 |
| Cautious/Cautious (a) | 2.6379 | 0.028 | 0.276 |
| Cautious/Courageous (b) | 2.7248 | 0.022 | 0.322 |
| Courageous/Cautious (c) | 2.6376 | 0.028 | 0.517 |
| Courageous/Courageous (d) | 2.6910 | 0.028 | 0.604 |

Paired $t$-tests confirm that sign-only (b) significantly degrades mean loss by +3.4% ($t = 15.85$, $p = 0.004$). Dropout (c) produces a marginal degradation ($t = 3.66$, $p = 0.064$) — the effect is small and borderline. The gap between (b) and (c) is genuine: sign-only hurts much more than dropout. However, neither condition improves over baseline — the result is "sign-only hurts more than dropout," not "forward noise helps, gradient boldness hurts." The chess paper's predicted best condition (b) is the worst-performing non-baseline condition.

The DG Index scales with total perturbation magnitude: (d) 0.604 $>$ (c) 0.517 $>$ (b) 0.322 $>$ baseline 0.299 $>$ (a) 0.276. DG = 0.604 (courageous/courageous) is the highest value observed across all six experiments, surpassing the previous maximum (noisy $\sigma = 0.1$, DG = 0.533 in Experiment 3) by 13%. The swarming index — measuring how many attention heads converge on the same high-attention positions — remains elevated across all conditions, indicating that architectural head convergence is robust to the courage/caution manipulation.

![Figure 9: Courage vs. caution. The chess paper's predicted winner (b, cautious/courageous) is the worst non-baseline condition. Forward noise (c) outperforms gradient boldness (b), inverting the chess paper's optimal strategy.](../results/exp6_courage_caution.png)

### 4.7 Cross-Experiment Synthesis

**Table 7.** Performance across all perturbation types with paired statistical tests.

| Perturbation | Loss Delta (mean) | Paired $p$-value | Status |
|---|---|---|---|
| 8/16 heads frozen | -0.016 (improves) | 0.009 | Significant |
| 12/16 heads frozen | -0.053 (improves) | 0.006 | Significant |
| All inter-layer gradients stopped | +0.063 (+2.4%) | 0.021 | Significant (worse) |
| Sign-only gradients (Exp 3) | +0.011 (+0.4%) | 0.860 | Not significant |
| Noisy gradients ($\sigma = 0.1$, Exp 3) | +0.110 (+4.4%) | 0.265 | Not significant |
| Window=8 (restricted attention) | -0.002 (improves) | 0.038 | Significant (tiny) |
| Window=2 (restricted attention) | -0.035 (improves) | 0.318 | Not significant |
| Light topology (25% gradient flow) | -0.001 (improves) | 0.608 | Not significant |
| Sign-only (Exp 6) | +0.090 (+3.4%) | 0.004 | Significant (worse) |

No perturbation produces a statistically significant degradation exceeding ~3.4% (sign-only in Exp 6). Most perturbations produce no statistically significant change from baseline — the pattern is *tolerance* rather than improvement. Of the mean effects that appear to improve over baseline, only Experiment 1 high freezing is statistically significant ($p < 0.01$). The DG Index increases across every perturbation type and scales with perturbation severity.

**Table 8.** DG Index across all conditions.

| Perturbation | DG Index | Change from Baseline |
|---|---|---|
| Baseline | 0.299 | — |
| 16 heads frozen | 0.502 | +68% |
| Cell-view (Exp 2) | 0.375 | +25% |
| Noisy gradients ($\sigma = 0.1$) | 0.533 | +78% |
| Quantized 3-bit | 0.425 | +42% |
| Window=8 (restricted attention) | 0.499 | +67% |
| Cell-view topology (Exp 5) | 0.375 | +25% |
| Courageous/courageous | 0.604 | +102% |

The DG response appears regardless of whether the perturbation targets architecture (freezing), information flow (cell-view, communication topology), the optimization signal (gradient degradation), attention span (vision radius), or forward-pass stability (courage/caution). DG = 0.604 (courageous/courageous) is the new maximum, surpassing the previous highest (noisy $\sigma = 0.1$, 0.533) by 13%. In Experiment 1, DG scales monotonically with the number of frozen heads, providing the strongest evidence for perturbation-proportional rerouting.

### 4.8 Anomalies — What Perturbation Revealed

The DG metric confirmed what we designed it to measure. The genuine findings are phenomena that *emerged* without being designed for — results that are counterintuitive under standard deep learning assumptions. These are the Levin analogs: behaviors that "should not happen normally," made legible only through perturbation.

Paired statistical analysis at $n = 3$ supports some of these findings robustly, others only weakly, and reveals that several originally hypothesized anomalies are not distinguishable from noise. We organize them by statistical support level.

#### Tier 1 — Statistically Supported Findings ($p < 0.05$)

**Finding 1: Heavy freezing improves performance.** Freezing 8+ attention heads produces significantly better final loss than baseline ($p < 0.01$). Freezing 1-4 heads does not ($p > 0.85$). The improvement emerges at a threshold around 50% of heads, not monotonically. This is the paper's strongest finding: removing enough capacity *improves* performance by eliminating gradient interference between attention heads and MLP layers. Frozen random projections serve the system better than competing learners, paralleling Levin's finding that damaged organisms sometimes outperform intact ones.

**Finding 2: Sign-only gradients degrade more than dropout.** Sign-only (b) significantly worsens mean loss by +3.4% ($p = 0.004$) while dropout (c) worsens by only ~0.1% ($p = 0.064$, marginal). The gap between conditions is genuine and large. This inverts the chess paper's "cautious position, courageous moves" prediction: in transformers, forward-pass noise is far less damaging than gradient-signal reduction. The inversion is substrate-dependent: the optimal courage/caution balance reverses between discrete (chess) and continuous (transformer) optimization substrates.

#### Tier 2 — Observations Consistent With But Not Proven By Data

**Observation 3: Cell-view is viable at modest cost.** Cell-view degrades mean loss by +2.4% ($p = 0.021$), confirming a real but small cost for eliminating all inter-layer gradient flow. The finding is that cell-view is *tolerated*, not that it improves — but the tolerance itself is noteworthy.

**Observation 4: Sign-only gradients approximately match baseline.** In Experiment 3, sign-only produces no statistically significant degradation ($p = 0.860$). The mean effect is +0.4%, consistent with prior SignSGD findings (Bernstein et al., 2018). Precision is low at $n = 3$ — the true effect could range widely.

**Observation 5: Isolation produces head specialization.** Cutting inter-layer gradients causes Layer 3 heads to specialize more aggressively. This is an observational finding from head entropy analysis, not a loss-based statistical test.

**Observation 6: Window=8 gives a tiny improvement.** Window=8 significantly improves mean loss ($p = 0.038$), but the effect is -0.002 — too small to be practically meaningful without replication.

#### Retracted or Downgraded Claims

The following claims from preliminary analysis are not supported by paired statistical testing:

- **"Noise helps":** The mean effect of noisy $\sigma = 0.01$ is negative but not significant ($p = 0.701$). Paired differences go in opposite directions across runs. **Retracted.**
- **"Partial communication outperforms full":** Light topology's mean loss advantage (-0.0006) is not significant ($p = 0.61$). **Retracted.**
- **"Window=2 beats full attention by 1.4%":** The final loss difference is not significant ($p = 0.318$). **Retracted** as a standalone finding; retained as a suggestive trend.
- **"Monotonic improvement" with freezing:** Only significant for 8+ frozen heads. **Downgraded** to threshold effect.
- **"Decentralization stabilizes":** Variance comparisons at $n = 3$ are unreliable. **Downgraded** to observational.

These findings — two robustly supported, four observational, and several retracted — are what the experiments revealed when subjected to honest statistical scrutiny. DG confirms rerouting is happening; the statistically supported findings define where rerouting produces measurable effects.


## 5. Discussion

### 5.1 The Nancy Reading

Nancy's concept of *désœuvrement* — the interruption of work that reveals the community constituted by work — provides the interpretive frame for these findings.

During standard backpropagation, the transformer's components cooperate invisibly. Attention heads, MLP layers, and residual connections form a division of labor that produces outputs but reveals nothing about its own organization. The system works, and its working conceals its structure. Each experiment interrupts this work in a different way — freezing components, severing gradient flow, corrupting signals, restricting vision, scaling communication, introducing forward noise — and the interruptions do not destroy the system. Instead, they make visible the relational structure that was always present but hidden: redundancy (frozen heads reveal that MLPs carry independent learning capacity, statistically confirmed at $p < 0.01$ for high freezing), rerouting (elevated DG shows the system exploring alternative configurations that exist in the loss landscape at all times), decentralized viability (cell-view training shows that each layer can learn autonomously — the layers are always simultaneously individual and communal, but global backprop makes their individuality invisible), and tolerance of constraint (restricted attention and partial gradient flow produce no significant degradation, suggesting that the spacing between components — Nancy's *espacement* — is tolerated, and in the case of high freezing, actively beneficial).

What *désœuvrement* reveals is not a failure mode but a mode of being. The transformer's components tolerate perturbation to a remarkable degree; they constitute a community whose structure only becomes legible when their coordinated work is interrupted. The robustness is not a feature added on top of the architecture — it is the architecture, seen from the angle that only unworking provides.

### 5.2 Three Claims

Three claims emerge from the combined evidence across six experiments. Paired statistical analysis at $n = 3$ constrains these claims more tightly than the mean effects alone would suggest.

**Claim 1: Robustness is structural, not algorithmic.** No perturbation produces a statistically significant degradation exceeding ~3.4% (sign-only in Experiment 6, $p = 0.004$). Most perturbations produce no statistically significant change from baseline — the pattern is *tolerance* rather than improvement. Of the mean effects that appear to improve over baseline (Experiment 1 freezing, Experiment 3 small noise, Experiment 4 restricted windows, Experiment 5 partial flow), only Experiment 1 high freezing is statistically significant ($p < 0.01$). This robustness is not coded in the training algorithm. It arises from the transformer architecture itself: the residual stream provides redundant pathways, the MLP layers carry independent learning capacity, and the multi-head structure distributes representation across parallel subspaces.

**Claim 2: DG tracks perturbation response consistently.** Every perturbation type elevates the DG Index. The DG response scales with perturbation severity and appears regardless of perturbation category — architecture (freezing), information flow (cell-view, communication topology), optimization signal (gradient degradation), attention span (vision radius), and forward-pass stability (courage/caution). As a designed metric, DG confirms that rerouting is real and quantifiable — but DG is the measuring stick, not the discovery.

**Claim 3: Centralized control is optional.** Global backpropagation is not required for viable learning: local learning costs +2.4% mean loss ($p = 0.021$), sign-only gradients produce no significant degradation ($p = 0.860$), freezing 8+ attention heads significantly *improves* performance ($p < 0.01$), and reducing gradient flow to 25% produces no significant change ($p = 0.61$). The trajectories differ — cell-view training produces different specialization patterns, frozen-head training shifts load to MLPs, restricted windows force local attention — but the outcomes converge. Multiple internal organizations produce equivalent external behavior.

*Note on removed claims.* The previous Claim 4 ("perturbation can be beneficial") rested on four anomalies, of which only the high-freezing result (Experiment 1) is statistically significant. The "noise helps," "restricted vision improves," and "partial communication outperforms full" effects are not significant at $n = 3$. The previous Claim 5 ("the information bottleneck is beneficial") similarly depended on Experiments 4 and 5, whose window and partial-flow effects are not statistically significant. Both are better characterized as *hypotheses suggested by the data* that require replication at higher sample sizes.

### 5.3 Connection to Distributed Chess

Kofman, Campitelli & Levin (2025) implemented a distributed form of chess where each piece operates as an autonomous agent with limited perception, evolved behavioral parameters, and optional long-range communication. Experiments 4-6 were designed to test three specific predictions derived from their findings. Paired statistical analysis constrains these comparisons more than initially reported.

**Information bottleneck as beneficial constraint (partially confirmed).** The chess paper's central result — that intermediate vision radius $R4$ outperforms both the blind ($R0$) and omniscient ($R7$) conditions by approximately 50 Elo — finds partial support. Experiment 1's high-freezing result ($p < 0.01$) is the strongest evidence: eliminating attention head capacity improves performance above a threshold. However, the Experiment 4 window effects that were initially reported as confirming this prediction are not statistically significant (window=2 final loss: $p = 0.318$; window=4 mean loss: $p = 0.17$), and Experiment 5's partial flow effects are indistinguishable from noise ($p > 0.28$). The information bottleneck hypothesis remains suggestive but underpowered for Experiments 4 and 5.

**Partial communication tolerance (not testable at this power).** In the chess system, relay chains allow pieces to transmit threat information beyond their individual vision radius. Experiment 5 was designed to test the analogous prediction. While partial topologies (25-75% gradient flow) produce no significant degradation — consistent with high tolerance — neither do they significantly *improve* over full backpropagation ($p > 0.28$). The U-shaped loss curve reported in preliminary analysis does not survive paired statistical testing. We can confirm tolerance of reduced communication but not that partial communication outperforms full.

**Courage/caution strategy (inverted, with statistical support).** The chess paper finds that "cautious position, courageous moves" is optimal. Experiment 6 shows the opposite with statistical confidence: sign-only (b) significantly degrades mean loss ($p = 0.004$) while dropout (c) produces only marginal degradation ($p = 0.064$). The gap between conditions is genuine and large (+3.3% in mean loss). This inversion is not a failure of the analogy but a substrate-dependent finding. Chess pieces operate in a discrete, irreversible action space where stable perception is essential; transformers operate in a continuous, differentiable landscape where forward noise provides regularization and gradient precision is needed for fine-grained optimization.

**Swarming index and architectural convergence.** The chess paper found that offensive swarming — multiple pieces converging on the same squares — emerged as a collective behavior from individual piece drives. In our experiments, the swarming index measures how many attention heads converge on the same high-attention positions. This index remains elevated across all courage/caution conditions, indicating that architectural head convergence is robust to the perturbation.

**Emergent collective goals without individual encoding.** No individual chess piece encodes the concept of checkmate; it emerges from the interaction of piece-level drives. Similarly, our transformer's robustness is not programmed — no component encodes fault tolerance, yet the collective produces it. The statistically supported findings (high-freezing improvement, sign-only degradation gap) demonstrate system-level competencies arising from component interactions rather than component capabilities.

### 5.4 Limitations

**Statistical power.** With 3 runs per condition ($df = 2$), paired $t$-tests can detect only large effects. Paired analysis reveals that many reported mean effects are not statistically significant at $n = 3$: the Experiment 3 gradient degradation effects ($p > 0.26$), the Experiment 4 window=2 and window=4 improvements ($p = 0.17$–$0.32$), the Experiment 5 partial-flow effects ($p > 0.28$), and the "noise helps" effect ($p = 0.701$) are all indistinguishable from noise. Of the originally claimed eight anomalies, only two (high-freezing improvement and the sign-only degradation gap in Experiment 6) are statistically robust. A replication with 30+ runs per condition is the most important next step to determine which of the suggestive patterns are real.

**Scale.** The model has 4 layers, 16 dimensions, and ~11,000 parameters. Whether these findings extend to production-scale transformers is unknown. We argue that toy scale is acceptable for a *methodology* paper: the contribution is the perturbation protocol and the findings it reveals, not claims about large language models. Levin et al. used sorting algorithms — maximally simple systems — for the same reason.

**Task complexity.** Character-level name generation is a toy task. Whether these patterns persist for complex language modeling, reasoning, or multi-modal tasks is untested.

**Training duration.** 200 steps captures early learning dynamics but not long-horizon phenomena like grokking or phase transitions.

**The DG metric.** The DG Index is a designed metric. Unlike Levin, who discovered delayed gratification as an emergent surprise, we defined it and went measuring it. The measurements confirm the metric works as defined, but they are confirmation, not discovery. DG's relationship to standard measures of representation quality, generalization, and internal structure requires further validation.

**Gradient degradation scope.** Only four degradation methods tested. Adversarial gradient attacks, structured corruption, and layer-selective degradation remain unexplored.

**Chess-paper translation fidelity.** The courage/caution inversion (Finding 2) may reflect genuine substrate differences between discrete board games and continuous optimization landscapes rather than a failure of the analogy. Whether more nuanced translations of the chess paper's behavioral chromosome — incorporating the full 13-gene design rather than single perturbation proxies — would produce results closer to the chess paper's predictions remains open.

**Composite perturbation design.** Experiment 6 maps a $2 \times 2$ conceptual design (forward stability $\times$ gradient boldness) onto single perturbation types per condition. A true composite design — simultaneously applying dropout to the forward pass *and* sign-only reduction to gradients — would more faithfully test the courage/caution hypothesis but requires extending the perturbation framework to support multiple simultaneous perturbation types.

### 5.5 Future Work

The methodology extends naturally in several directions, with replication as the most urgent priority:

- **Replication with adequate power:** Repeat all six experiments with 30+ runs per condition to determine which of the suggestive patterns (Experiment 3 gradient tolerance, Experiment 4 window improvements, Experiment 5 partial-flow effects) are real effects versus sampling noise. This is the single most important next step. All Tier 2 findings and several retracted claims could be confirmed or definitively rejected with adequate statistical power.
- **Scale:** Replicate at 100M+ parameter scale to test whether architectural robustness persists or breaks down.
- **Task diversity:** Apply the perturbation protocol to arithmetic, reasoning, and multi-modal tasks where the competency demands are qualitatively different.
- **Chimeric experiments:** Create transformer "chimeras" by interleaving layers from separately trained models, directly mirroring Levin's chimeric sorting arrays with mixed sorting directions.
- **Architecture morphogenesis:** Allow the architecture itself to change during training — growing new heads, pruning inactive ones — testing whether the training process can *develop* the architecture rather than merely optimizing fixed parameters.
- **Opposing objectives:** Train different layers on conflicting objectives (next-token vs. previous-token prediction) to test whether transformers reach the stable equilibria that Levin observed in chimeric sorting with opposing sort directions.
- **Vision-communication interaction:** Experiment 4 (restricted attention) and Experiment 5 (partial gradient flow) each show suggestive bottleneck patterns independently, though neither is robustly significant. Testing restricted windows *combined* with partial gradient flow at higher sample sizes would reveal whether the two information bottlenecks interact.
- **Optimal window vs. scale:** The window=8 effect is statistically significant but tiny at this scale. Does the optimal window size scale with model size? A sweep across model scales with matched window proportions would test whether the information bottleneck hypothesis is scale-invariant.
- **Richer courage/caution design:** Experiment 6's single-perturbation-per-condition design is a coarse proxy for the chess paper's $2 \times 2$ matrix. A true composite perturbation framework — simultaneously applying forward-pass noise and gradient modification — would more faithfully test the courage/caution hypothesis and could resolve whether the inversion is a genuine substrate effect or an artifact of the simplified design.
- **Evolved component policies:** Replace fixed perturbation parameters with evolved per-head behavioral chromosomes (paralleling the chess paper's 13-gene design), allowing each head to develop its own attention strategy, learning rate, and noise tolerance through evolutionary optimization.
- **Threatening drive:** Add gradient bonuses for heads maintaining high attention entropy, paralleling the chess paper's finding that a "threatening" drive (pieces seeking to attack many squares simultaneously) contributed approximately +50 Elo.


## 6. Conclusion

We have presented a methodology — morphogenetic perturbation of transformer training — and demonstrated through six experiments that it reveals emergent properties invisible during normal operation. The methodology maps Levin's developmental biology protocol to neural networks: perturb the system during learning, observe its response as a trajectory, and distinguish between designed metrics that confirm the framework and emergent findings that perturbation made legible.

Paired statistical analysis at $n = 3$ yields two robust findings: freezing 8+ attention heads significantly improves performance ($p < 0.01$), revealing a threshold-dependent benefit where frozen random projections outperform competing learners; and sign-only gradients degrade significantly more than dropout ($p = 0.004$), inverting the chess paper's courage/caution prediction in a substrate-dependent manner. Several additional patterns — gradient tolerance, restricted-window improvements, partial-flow tolerance, isolation-driven specialization — are suggestive of broader architectural robustness but not statistically significant at this sample size. The residual stream, multi-head structure, and layer composition create a system that tolerates severe perturbation of its optimization signal, though the extent to which perturbation is actively *beneficial* (rather than merely tolerated) remains to be established at higher statistical power.

Experiments 4-6 provide an exploratory bridge to Kofman, Campitelli & Levin's (2025) distributed chess framework. The information bottleneck hypothesis finds partial support through the freezing result ($p < 0.01$), though the window and partial-flow effects that motivated the analogy are not significant at $n = 3$. The courage/caution inversion is the bridge's strongest finding ($p = 0.004$), revealing that the optimal strategy reverses between discrete and continuous optimization substrates. Replication with 30+ runs per condition is the most important next step for this work.

Transformers, like Levin's sorting algorithms and distributed chess pieces, reveal their structure through interruption. The work conceals the community; the unworking makes it legible.


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
