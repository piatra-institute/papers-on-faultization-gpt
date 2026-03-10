# MorphoGPT: Experimental Findings

Morphogenetic perturbation analysis of a minimal GPT, applying Levin's developmental biology methodology and Nancy's concept of désœuvrement (unworking) to transformer learning dynamics.


## 1. Setup

**Model:** 4-layer, 16-dimensional, 4-head GPT (16 total attention heads).

**Task:** Character-level name generation, trained on a names dataset.

**Protocol:** 200 training steps, 30 independent runs per condition (seeds 42–71), loss and head-level metrics recorded at every step.

**Statistical analysis:** All comparisons use paired $t$-tests with runs matched by seed across conditions ($n = 30$, $df = 29$). We report $p < 0.05$ as statistically significant and $0.05 < p < 0.10$ as marginal. With 30 paired observations, the tests have adequate power to detect moderate effects. We distinguish between *statistically supported* findings and *observational* patterns throughout. Effect sizes are reported as Cohen's $d$.

**Key metric — Delayed Gratification (DG) Index:** Measures how much the loss trajectory dips below its final value during training. A high DG means the system explored better configurations early but "gave them up." At $n = 30$, DG shows no systematic response to perturbation (see Section 8), contrary to the $n = 3$ pilot data. The DG metric captures a real property of loss trajectories but does not scale with perturbation severity.


## 2. Experiment 1: Head Freezing

**Analog:** Levin's frozen-cell perturbation. Randomly selected attention heads have their parameters frozen at initialization, forcing the remaining heads to compensate.

### Results

| Frozen Heads | Final Loss | Std | Mean Loss | DG Index |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 2.469 | 0.340 | 2.639 | 0.571 |
| 1 | 2.472 | 0.343 | 2.639 | 0.572 |
| 2 | 2.471 | 0.343 | 2.638 | 0.575 |
| 4 | 2.473 | 0.346 | 2.638 | 0.509 |
| 8 | 2.481 | 0.337 | 2.636 | 0.528 |
| 12 | 2.479 | 0.347 | 2.635 | 0.475 |
| 16 | 2.479 | 0.348 | 2.635 | 0.512 |

### Findings

**Final loss does not improve with freezing.** No freezing level produces a statistically significant improvement in final loss. Freeze 8 is marginally worse ($p = 0.064$); all others are non-significant ($p > 0.18$). The monotonic improvement trend reported at $n = 3$ does not replicate (Spearman $\rho = 0.013$, $p = 0.85$). The $n = 3$ finding that "damage improves" was a sampling artifact.

**Mean trajectory loss shows a tiny improvement.** Freezing 4+ heads produces a statistically significant reduction in mean loss: freeze 4 ($p = 0.012$, $d = -0.49$), freeze 8 ($p < 0.001$, $d = -1.03$), freeze 12 ($p < 0.001$, $d = -1.13$), freeze 16 ($p < 0.001$, $d = -0.96$). However, the effects are tiny: -0.1% to -0.2% of mean loss. This suggests frozen heads reduce gradient noise during the training trajectory without affecting the final convergence point.

**DG does not increase with freezing.** Contrary to the $n = 3$ results, DG does not scale with the number of frozen heads. Freeze 12 actually *decreases* DG significantly ($p = 0.034$, $d = -0.41$). All other levels are non-significant. The claimed "rerouting proportional to damage" narrative is not supported.

**Trajectory shape is preserved.** Cross-condition trajectory correlations exceed 0.95 at all freezing levels. The system follows the same learning arc regardless of how many heads are disabled.


## 3. Experiment 2: Cell-View GPT

**Analog:** Nancy's being-singular-plural — each layer treated as an autonomous agent. Stop-gradient applied at all layer boundaries so each layer learns only from its own local loss signal, with no end-to-end backpropagation.

### Results

| Condition | Mean Loss | Final Loss | DG Index |
|:---:|:---:|:---:|:---:|
| baseline | 2.639 | 2.469 | 0.571 |
| cell_view | 2.687 | 2.590 | 0.507 |

### Findings

**Local-only learning degrades performance significantly.** Cell-view increases final loss by +4.9% ($p < 0.001$, $d = +1.16$) and mean loss by +1.8% ($p < 0.001$, $d = +3.31$). While the degradation is significant and non-trivial, the system still learns effectively — eliminating all inter-layer gradient communication does not break the architecture.

**DG does not increase under cell-view.** The $n = 3$ claim of +25.5% DG elevation is not replicated ($p = 0.34$). Cell-view DG (0.507) is not significantly different from baseline (0.571).

**Head specialization patterns shift.** Under cell-view training, final-layer heads tend to specialize more aggressively when deprived of upstream gradient refinement. This is an observational finding from head entropy analysis.


## 4. Experiment 3: Gradient Degradation

**Analog:** Levin's noisy signaling channels. Gradients are corrupted during training through four methods: additive Gaussian noise (two scales), sign-only reduction (discarding magnitude), and 3-bit quantization.

### Results

| Method | Final Loss | Δ% | $p$ (final) | Mean Loss | $p$ (mean) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline | 2.469 | — | — | 2.639 | — |
| noisy (σ=0.01) | 2.477 | +0.3% | 0.525 | 2.639 | 0.978 |
| noisy (σ=0.1) | 2.530 | +2.5% | 0.021* | 2.698 | <0.001*** |
| sign_only | 2.582 | +4.6% | 0.004** | 2.742 | <0.001*** |
| quantized (3-bit) | 2.543 | +3.0% | 0.019* | 2.703 | <0.001*** |

### Findings

**Small noise is tolerated; severe corruption degrades.** At $n = 30$, the "null result" reported at $n = 3$ was due to insufficient power. Sign-only gradients significantly degrade final loss by +4.6% ($p = 0.004$, $d = +0.56$), quantized gradients by +3.0% ($p = 0.019$, $d = +0.45$), and noisy $\sigma = 0.1$ by +2.5% ($p = 0.021$, $d = +0.44$). Only small noise ($\sigma = 0.01$) produces no significant change ($p = 0.525$).

**Mean loss effects are highly significant.** For the training trajectory, sign-only worsens mean loss by +3.9% ($p < 0.001$, $d = +6.13$), quantized by +2.4% ($p < 0.001$, $d = +4.54$), and noisy $\sigma = 0.1$ by +2.2% ($p < 0.001$, $d = +3.79$). The large Cohen's $d$ values (>3) indicate massive effects on the training trajectory.

**The "noise helps" effect is not supported.** Noisy $\sigma = 0.01$ shows no significant effect on either final loss ($p = 0.525$) or mean loss ($p = 0.978$). The $n = 3$ suggestion that small noise improves performance was sampling noise.

**Architecture constrains the solution space.** Despite the degradation, the worst-performing method (sign-only) still achieves loss within 5% of baseline. The residual stream, attention patterns, and MLP structure collectively define a narrow enough solution manifold that even crude gradient approximations navigate it, albeit with measurable cost.


## 5. Experiment 4: Vision Radius Sweep

**Analog:** Kofman, Campitelli & Levin's (2025) vision radius experiment in distributed chess, where each piece perceives only squares within radius R. We translate this by restricting each attention head's context window.

### Results

| Window | Final Loss | $p$ (final) | Mean Loss | $p$ (mean) |
|:---:|:---:|:---:|:---:|:---:|
| Baseline (full) | 2.469 | — | 2.639 | — |
| 1 | 2.486 | 0.155 | 2.650 | <0.001*** |
| 2 | 2.485 | 0.136 | 2.640 | 0.185 |
| 4 | 2.475 | 0.524 | 2.636 | 0.004** |
| 8 | 2.471 | 0.478 | 2.638 | <0.001*** |
| 16 | 2.469 | 1.000 | 2.639 | 1.000 |

### Findings

**No window size significantly changes final loss.** All paired $t$-tests on final loss are non-significant ($p > 0.13$). The previously reported -1.4% improvement for window=2 does not replicate.

**Tiny mean-loss effects exist for some windows.** Window=1 significantly worsens mean loss (+0.4%, $p < 0.001$, $d = +2.49$). Window=4 significantly improves mean loss (-0.1%, $p = 0.004$, $d = -0.57$) and window=8 improves it (-0.03%, $p < 0.001$, $d = -0.68$). However, these effects are so small (<0.5%) as to be practically negligible.

**Window=16 is identical to baseline.** Full-context window (16 = block size) reproduces baseline values exactly, confirming the implementation introduces no artifacts.

**The information bottleneck hypothesis is not supported.** The chess paper's finding that intermediate vision radius outperforms omniscience does not translate to meaningful attention windowing effects. The final-loss results are all non-significant, and the mean-loss effects are negligibly small.


## 6. Experiment 5: Communication Topology

**Analog:** The chess paper's relay chains, where pieces transmit threat information beyond their individual vision radius. We create a spectrum of gradient flow topologies between full backpropagation and complete isolation.

### Results

| Topology | Fraction | Final Loss | $p$ (final) | Mean Loss | $p$ (mean) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Full | 1.00 | 2.469 | — | 2.639 | — |
| Heavy | 0.75 | 2.472 | 0.024* | 2.639 | 0.553 |
| Half | 0.50 | 2.470 | 0.551 | 2.639 | 0.100† |
| Light | 0.25 | 2.471 | 0.426 | 2.639 | 0.303 |
| Cell-view | 0.00 | 2.590 | <0.001*** | 2.687 | <0.001*** |

### Findings

**Partial gradient flow is largely tolerated.** Half (50%) and light (25%) gradient flow produce no significant change in either final or mean loss. Heavy (75%) shows a small but significant final-loss increase ($p = 0.024$, $d = +0.44$), though the effect is tiny (+0.1%).

**Only complete isolation degrades meaningfully.** Cell-view (0% gradient flow) degrades final loss by +4.9% ($p < 0.001$, $d = +1.16$) and mean loss by +1.8% ($p < 0.001$, $d = +3.31$).

**The U-shape claim is not supported.** No partial-flow condition outperforms full backpropagation. The system tolerates reduced gradient flow but does not benefit from it.


## 7. Experiment 6: Courage vs. Caution

**Analog:** Kofman et al.'s finding that "cautious position, courageous moves" is the optimal chess strategy. We translate this into a 2×2 matrix:

| | Cautious Gradients | Courageous Gradients |
|:---:|:---:|:---:|
| **Cautious Forward** | (a) Tiny noise (σ=0.001) | (b) Sign-only gradients |
| **Courageous Forward** | (c) Dropout (p=0.1) | (d) Noisy gradients (σ=0.1) |

### Results

| Condition | Final Loss | $p$ (final) | Mean Loss | $p$ (mean) |
|:---:|:---:|:---:|:---:|:---:|
| Baseline | 2.469 | — | 2.639 | — |
| Cautious/Cautious (a) | 2.484 | 0.017* | 2.639 | 0.799 |
| Cautious/Courageous (b) | 2.582 | 0.004** | 2.742 | <0.001*** |
| Courageous/Cautious (c) | 2.481 | 0.029* | 2.641 | 0.010** |
| Courageous/Courageous (d) | 2.509 | 0.044* | 2.697 | <0.001*** |

### Findings

**All perturbation conditions significantly degrade final loss.** Every condition performs worse than baseline, with sign-only (b) showing the largest degradation (+4.6%, $p = 0.004$, $d = +0.56$).

**Sign-only degrades much more than dropout.** The paired comparison between sign-only (b) and dropout (c) is highly significant for mean loss ($p < 0.001$, $d = +6.00$) and significant for final loss ($p = 0.010$, $d = +0.51$). Sign-only worsens mean loss by +3.8% relative to dropout.

**The chess prediction is inverted.** Condition (c) courageous/cautious (dropout with careful gradients) significantly outperforms (b) cautious/courageous (sign-only with stable forward pass). The chess paper's predicted winner is the worst non-baseline condition for mean loss.

**The substrate-dependent interpretation.** Chess pieces operate in a discrete, irreversible action space where stable perception is essential. Transformers operate in a continuous, differentiable landscape where forward noise acts as regularization and gradient precision is needed for fine-grained optimization.


## 8. Cross-Experiment Synthesis

Three findings emerge from the combined evidence across six experiments at $n = 30$.

### Finding 1: Robustness is structural and bounded

The architecture tolerates mild perturbations but degrades under severe gradient corruption. The tolerance boundary is clearly delineated:

| Perturbation | Final Loss Δ% | $p$-value | Status |
|---|---|---|---|
| Freeze 1-4 heads | +0.1-0.2% | >0.37 | Tolerated (ns) |
| Freeze 8-16 heads | +0.4-0.5% | 0.06-0.19 | Marginal at best |
| Window=2,4,8 | +0.1-0.6% | >0.13 | Tolerated (ns) |
| Light gradient flow (25%) | +0.1% | 0.43 | Tolerated (ns) |
| Half gradient flow (50%) | +0.0% | 0.55 | Tolerated (ns) |
| Noisy gradients (σ=0.01) | +0.3% | 0.53 | Tolerated (ns) |
| Dropout (p=0.1) | +0.5% | 0.029 | Mild degradation |
| Noisy gradients (σ=0.1) | +2.5% | 0.021 | Significant degradation |
| Quantized 3-bit | +3.0% | 0.019 | Significant degradation |
| Sign-only gradients | +4.6% | 0.004 | Significant degradation |
| Cell-view (no backprop) | +4.9% | <0.001 | Significant degradation |

The pattern is *tolerance with limits*: perturbations that preserve approximate gradient direction (small noise, partial flow) are absorbed by the architecture's redundancy; perturbations that destroy gradient information (sign-only, quantization) produce measurable degradation. The worst degradation (cell-view, +4.9%) is still modest — the architecture works even when its defining optimization mechanism is eliminated.

### Finding 2: DG does not track perturbation

At $n = 3$, the DG Index appeared to scale with perturbation severity — every perturbation elevated DG, and the response correlated with damage level. At $n = 30$, this finding does not replicate:

- **No perturbation significantly increases DG.** Across all 23 perturbation conditions tested, not a single one produces a statistically significant DG increase over baseline ($p > 0.16$ for all).
- **Two perturbations significantly *decrease* DG:** freeze 12 ($p = 0.034$, $d = -0.41$) and quantized 3-bit ($p = 0.028$, $d = -0.46$).
- **DG is indistinguishable from noise across conditions.** Baseline DG = 0.571 ± 0.312. All conditions fall within 0.447–0.606, with high within-condition variance.

The $n = 3$ DG patterns were sampling artifacts. DG captures a real property of loss trajectories (temporary increases followed by improvements) but this property is a stochastic feature of SGD training, not a perturbation response. The Levin analogy — that damage triggers compensatory rerouting — does not hold for transformer training dynamics at this scale.

### Finding 3: The chess-paper inversion is robust

Sign-only gradients degrade performance significantly more than dropout. This is the paper's most robust cross-experiment finding:

- Experiment 3: sign-only worsens mean loss by +3.9% ($p < 0.001$, $d = +6.13$)
- Experiment 6: sign-only (b) vs dropout (c) gap is +3.8% mean loss ($p < 0.001$, $d = +6.00$)

The chess paper predicts "cautious position, courageous moves" as optimal. In transformers, the opposite holds: forward-pass noise (dropout) is far less damaging than gradient-signal reduction (sign-only). This inversion is substrate-dependent and highly statistically robust.


## 9. Anomalies — What Perturbation Revealed

### Statistically Supported ($p < 0.05$)

**1. Sign-only gradients degrade significantly.** Discarding gradient magnitude (+4.6% final loss, $p = 0.004$; +3.9% mean loss, $p < 0.001$) is the most damaging single perturbation tested. The effect is large ($d > 0.5$ for final, $d > 6$ for mean), replicable, and inverts the chess-paper prediction. This is the paper's strongest individual finding.

**2. Cell-view is viable but costly.** Eliminating all inter-layer gradient flow degrades final loss by +4.9% ($p < 0.001$, $d = +1.16$). The cost is significant but not catastrophic — each layer learns independently and the system still generates coherent sequences. This confirms that centralized backpropagation is helpful but not strictly necessary.

**3. Severe gradient corruption degrades systematically.** Quantized 3-bit ($p = 0.019$) and noisy $\sigma = 0.1$ ($p = 0.021$) both significantly worsen final loss. The degradation scales with corruption severity: sign-only (+4.6%) > quantized (+3.0%) > noisy $\sigma = 0.1$ (+2.5%) > noisy $\sigma = 0.01$ (+0.3%, ns). The ordering is consistent across final and mean loss metrics.

**4. Tiny mean-loss improvements under freezing.** Freezing 4+ heads improves mean trajectory loss by 0.1-0.2% ($p < 0.05$ for all, $p < 0.001$ for 8+). This does not affect final loss — the improvement is in the training trajectory, not the endpoint. The effect suggests frozen heads reduce gradient interference during training without altering the convergence basin.

### Not Supported or Retracted

- **"Damage improves" (final loss):** The $n = 3$ finding that freezing 8+ heads significantly improves final loss ($p = 0.009$) does not replicate at $n = 30$. No freezing level improves final loss ($p > 0.06$ for all). **Retracted.**
- **"DG scales with perturbation":** The $n = 3$ pattern of DG increasing with perturbation severity does not replicate. DG shows no significant increase under any condition. Two conditions (freeze 12, quantized) significantly *decrease* DG. **Retracted.**
- **"Monotonic improvement" with freezing:** Spearman $\rho = 0.013$, $p = 0.85$. **Retracted.**
- **"Noise helps":** Noisy $\sigma = 0.01$ produces no significant effect ($p > 0.5$). **Retracted.**
- **"Partial communication outperforms full":** No partial-flow condition improves over baseline. **Retracted.**
- **"Restricted vision improves final loss":** No window size significantly changes final loss. **Retracted.**
- **"Information bottleneck is beneficial":** Not supported by any final-loss comparison. **Retracted.**


## 10. The Nancy Reading

Nancy's concept of désœuvrement — the interruption of work that reveals the community constituted by work — provides the interpretive frame for this *methodology*. The philosophical reading applies to the act of perturbation and what it makes visible.

**Normal training is opaque.** During standard backpropagation, the transformer's components cooperate invisibly. The system works, and its working conceals its structure.

**Perturbation as unworking.** Each experiment interrupts the system's work — freezing components, severing gradient flow, corrupting signals, restricting vision, scaling communication, introducing forward noise. These interruptions reveal:

- **Bounded tolerance**: The architecture absorbs mild perturbations (small noise, partial gradient flow, moderate freezing) without significant degradation. Severe perturbations (sign-only gradients, full gradient elimination) produce measurable but bounded costs (≤5%). This tolerance is structural — the residual stream, multi-head attention, and MLP layers provide redundant pathways.
- **Gradient precision matters more than gradient flow**: The sharpest degradation comes not from reducing gradient *quantity* (partial flow, freezing) but from reducing gradient *quality* (sign-only, quantization). The optimization signal's directional precision is more critical than its magnitude or completeness.
- **Decentralized viability**: Cell-view training confirms each layer can learn autonomously at a ~5% final-loss cost. The layers are always singular-plural — simultaneously individual and communal — but global backprop makes their individuality invisible.
- **No rerouting signal**: The DG Index — designed to detect perturbation-triggered rerouting analogous to Levin's biological findings — shows no systematic response to perturbation at $n = 30$. The transformer's response to perturbation is *degradation or tolerance*, not the *compensatory rerouting* observed in biological systems. This is itself a finding about substrate differences: the continuous optimization landscape may not exhibit the discrete reorganization events that characterize developmental biology.


## 11. Limitations

- **Scale**: 4 layers, 16 dimensions, 16 heads. Scaling behavior to production-sized transformers is unknown.
- **Task complexity**: Character-level name generation is a toy task. Whether these findings hold for complex language modeling, reasoning, or multi-modal tasks is untested.
- **Training duration**: 200 steps captures early learning dynamics but not long-horizon phenomena like grokking or phase transitions.
- **DG metric**: The DG Index does not respond to perturbation at $n = 30$, contrary to $n = 3$ pilot data. The metric captures a real stochastic property of loss trajectories but does not function as a perturbation response measure. Its relationship to biological delayed gratification is questionable at this scale.
- **Gradient degradation scope**: Only four degradation methods tested. Adversarial gradient attacks, structured corruption, and layer-selective degradation remain unexplored.
- **Chess-paper translation fidelity**: The courage/caution inversion (Finding 1) reflects genuine substrate differences. Whether richer composite perturbation designs would produce different results remains untested.
- **Effect sizes**: Many statistically significant effects are practically small (mean-loss improvements under 0.5%). Statistical significance at $n = 30$ does not imply practical importance.
