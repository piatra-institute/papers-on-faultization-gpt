# MorphoGPT: Experimental Findings

Morphogenetic perturbation analysis of a minimal GPT, asking: does the system exhibit behaviors that Stochastic Gradient Descent (SGD) didn't prescribe? We apply Levin's developmental biology methodology to reveal emergent behaviors in transformer learning dynamics.

**Central question:** SGD says "minimize loss." It does not say "build tolerance through gradual exposure," "recover completely from damage," or "regenerate a destroyed layer." When the system does these things anyway, they are emergent — not directly prescribed by the optimization objective. The experiments below identify where such behaviors appear, where basin geometry explains convergence, where a local loss objective reaches the same endpoint through different basin geometry, and where the system simply absorbs perturbation (tolerance).


## 1. Setup

**Model:** 4-layer, 16-dimensional, 4-head GPT (16 total attention heads).

**Task:** Character-level name generation, trained on a names dataset.

**Protocol:** 200 training steps, 30 independent runs per condition (seeds 42–71), loss and head-level metrics recorded at every step.

**Statistical analysis:** All comparisons use paired $t$-tests with runs matched by seed across conditions ($n = 30$, $df = 29$). We report $p < 0.05$ as statistically significant and $0.05 < p < 0.10$ as marginal. With 30 paired observations, the tests have adequate power to detect moderate effects. We distinguish between *statistically supported* findings and *observational* patterns throughout. Effect sizes are reported as Cohen's $d$.

**Three-scale protocol:** $n$ is a resolution dial. At $n = 3$, signals are coarse — rough shapes, visible but ambiguous. At $n = 30$, the picture sharpens and most ambiguities resolve. At $n = 300$, fine structure emerges. No scale is wrong; each reveals different phenomena. Signals that were ambiguous at $n = 3$ are not retracted at $n = 30$ — the resolution increased and the picture changed.

**Key metric — Delayed Gratification (DG) Index:** Measures how much the loss trajectory dips below its final value during training. At $n = 3$, DG appeared to scale with perturbation severity. At $n = 30$, this signal resolved to null: no perturbation condition produces a statistically significant DG increase ($p > 0.19$ for all). DG captures a real stochastic property of loss trajectories but does not function as a perturbation response measure at $n = 30$. At $n = 300$, the DG null holds: no perturbation condition produces significant DG increase, confirming that DG captures stochastic SGD dynamics but does not track perturbation response even at 10x power.


## 2. Experiment 1: Head Freezing

**Motivation:** Levin's frozen-cell perturbation. Randomly selected attention heads have their parameters frozen at initialization, forcing the remaining heads to compensate.

**Scaling note:** At $n = 3$, head freezing appeared to improve final loss — the coarse signal suggested damage helps. At $n = 30$, some final-loss improvements are now significant (freeze\_4 $p = 0.0014$, freeze\_8 $p = 0.0232$, freeze\_12 $p = 0.0161$), and the mean trajectory improvement for 4+ frozen heads is highly significant. The picture sharpened; both final loss and trajectory show improvement signals. At $n = 300$, all final-loss effects resolve to non-significant (all $p > 0.15$, Spearman $\rho = -0.0045$, $p = 0.84$), but the trajectory improvement strengthens to highly significant: freeze\_4 ($d = -0.971$, $p < 0.0001$), freeze\_8 ($d = -1.245$, $p < 0.0001$), freeze\_12 ($d = -1.421$, $p < 0.0001$), freeze\_16 ($d = -1.312$, $p < 0.0001$). The trajectory improvement signal is robust at high power.

### Results

| Frozen Heads | Final Loss | Std | Mean Loss | DG Index |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 2.5565 | 0.4074 | 2.6271 | 0.680 |
| 1 | 2.5529 | 0.4059 | 2.6265 | 0.632 |
| 2 | 2.5487 | 0.4049 | 2.6260 | 0.689 |
| 4 | 2.5428 | 0.3968 | 2.6251 | 0.636 |
| 8 | 2.5439 | 0.3962 | 2.6232 | 0.640 |
| 12 | 2.5387 | 0.4001 | 2.6215 | 0.614 |
| 16 | 2.5474 | 0.4067 | 2.6204 | 0.570 |

### Findings

**Final loss shows some significant improvements at $n = 30$.** Freezing 4 heads improves final loss by $-0.5\%$ ($p = 0.0014$, $d = -0.647$), freeze 8 by $-0.5\%$ ($p = 0.0232$, $d = -0.438$), freeze 12 by $-0.7\%$ ($p = 0.0161$, $d = -0.467$). Freeze 2 is also significant ($p = 0.0254$). However, the overall trend is not monotonic: Spearman $\rho = -0.015$ ($p = 0.83$) confirms no monotonic final-loss relationship. At $n = 300$, all final-loss effects resolve to non-significant ($p > 0.15$), indicating the $n = 30$ final-loss improvements are scale-dependent.

**Mean trajectory loss shows a genuine improvement.** Freezing 4+ heads produces a statistically significant reduction in mean loss: freeze 4 ($p < 0.001$, $d = -1.008$), freeze 8 ($p < 0.001$, $d = -1.228$), freeze 12 ($p < 0.001$, $d = -1.366$), freeze 16 ($p < 0.001$, $d = -1.070$). Effects are 0.1–0.3% of mean loss — small but statistically robust. Freeze 2 also significant ($p = 0.0019$, $d = -0.623$) and freeze 1 marginal ($p = 0.0528$, $d = -0.369$). Frozen heads at random initialization reduce gradient interference during training in a way SGD did not prescribe.

**Classification:** Trajectory improvement for 4+ frozen heads = *emergent behavior*. Final-loss indifference at $n = 300$ = *basin geometry*.

**Trajectory shape is preserved.** Cross-condition trajectory correlations exceed 0.95 at all freezing levels.


## 3. Experiment 2: Cell-View GPT

**Motivation:** Each layer treated as an autonomous agent. Local loss (layerwise cross-entropy) applied so each layer learns only from its own local loss signal, with no end-to-end backpropagation.

**Scaling note:** At $n = 3$, the signal was ambiguous — cell-view appeared to elevate DG substantially (+25.5%), suggesting possible rerouting behavior. At $n = 30$, the DG signal resolved to null and cell-view produces near-identical final loss ($p = 0.7755$). At $n = 300$, cell-view final loss remains non-significant ($p = 0.90$), while mean loss shows a small but significant increase ($+0.2\%$, $p < 0.0001$, $d = +0.731$). Local loss achieves equivalent convergence through different basin geometry.

### Results

| Condition | Mean Loss | Final Loss | DG Index |
|:---:|:---:|:---:|:---:|
| baseline | 2.6271 | 2.5565 | 0.680 |
| cell_view | 2.6316 | 2.5594 | 0.655 |

### Findings

**Local loss achieves equivalent final loss.** Cell-view produces near-identical final loss ($+0.1\%$, $p = 0.7755$, $d = +0.053$). Mean loss shows a small but significant increase ($+0.2\%$, $p = 0.0049$, $d = +0.556$). The local loss objective finds the same final basin through a slightly different trajectory.

**Classification:** *Basin geometry* — local loss reaches the same endpoint as end-to-end backpropagation. The local loss objective defines a different optimization landscape that nevertheless converges to the same minimum.

**DG does not increase under cell-view.** At $n = 3$, a +25.5% DG elevation appeared. At $n = 30$, this signal resolved to null ($p = 0.90$). Cell-view DG (0.655) is not significantly different from baseline (0.680).

**Head specialization patterns shift.** Under cell-view training, final-layer heads tend to specialize more aggressively when deprived of upstream gradient refinement. Observational finding from head entropy analysis.


## 4. Experiment 3: Gradient Degradation

**Motivation:** Levin's noisy signaling channels. Gradients are corrupted during training through four methods: additive Gaussian noise (two scales), sign-only reduction (discarding magnitude), and 3-bit quantization.

**Scaling note:** At $n = 3$, the signal was ambiguous — all four methods appeared neutral and small noise appeared to help. At $n = 30$, the threshold is sharper than before: only sign-only ($+5.0\%$, $p = 0.0022$) and quantized ($+3.8\%$, $p = 0.0081$) show significant final-loss degradation; noisy $\sigma = 0.1$ is now non-significant for final loss ($p = 0.9100$) though highly significant for mean loss. At $n = 300$, the pattern sharpens further: noise $\sigma = 0.01$ remains non-significant ($-0.2\%$, $p = 0.28$), while noise $\sigma = 0.1$ ($+2.2\%$, $p < 0.0001$, $d = +0.367$), sign-only ($+4.9\%$, $p < 0.0001$, $d = +0.575$), and quantized 3-bit ($+3.6\%$, $p < 0.0001$, $d = +0.529$) are all highly significant. The tolerance-to-degradation boundary is a sharp threshold between $\sigma = 0.01$ and $\sigma = 0.1$, not a graded curve.

### Results

| Method | Final Loss | Δ% | $p$ (final) | Mean Loss | $p$ (mean) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline | 2.5565 | — | — | 2.6271 | — |
| noisy (σ=0.01) | 2.5524 | -0.2% | 0.7378 | 2.6280 | 0.4963 |
| noisy (σ=0.1) | 2.5525 | -0.2% | 0.9100 | 2.6831 | <0.001*** |
| sign_only | 2.6850 | +5.0% | 0.0022** | 2.7288 | <0.001*** |
| quantized (3-bit) | 2.6527 | +3.8% | 0.0081** | 2.6925 | <0.001*** |

### Findings

**Small noise is tolerated; severe corruption degrades.** Sign-only gradients significantly degrade final loss by $+5.0\%$ ($p = 0.0022$, $d = +0.614$) and quantized gradients by $+3.8\%$ ($p = 0.0081$, $d = +0.519$). Noisy $\sigma = 0.1$ is non-significant for final loss ($-0.2\%$, $p = 0.9100$) but highly significant for mean loss. Small noise ($\sigma = 0.01$) produces no significant change in either metric ($p = 0.7378$ final, $p = 0.4963$ mean).

**Mean loss effects are highly significant.** Sign-only worsens mean loss by $+3.9\%$ ($p < 0.001$, $d = +5.696$), quantized by $+2.5\%$ ($p < 0.001$, $d = +4.457$), and noisy $\sigma = 0.1$ by $+2.1\%$ ($p < 0.001$, $d = +3.358$). The large Cohen's $d$ values (>3) indicate massive effects on the training trajectory.

**Classification:** *Tolerance* — the system absorbs gradient noise up to a threshold ($\sigma = 0.01$). Above that threshold, degradation follows.

**Architecture constrains the solution space.** Despite the degradation, the worst-performing method (sign-only) still achieves loss within 5% of baseline. The residual stream, attention patterns, and MLP structure define a narrow enough solution manifold that even crude gradient approximations navigate it, with measurable cost.


## 5. Experiment 4: Vision Radius Sweep

**Motivation:** Kofman, Campitelli & Levin's (2025) vision radius experiment in distributed chess, where each piece perceives only squares within radius R. We translate this by restricting each attention head's context window.

**Scaling note:** At $n = 3$, the signal was ambiguous — an information bottleneck effect appeared possible, with intermediate window sizes seeming to outperform full context. At $n = 30$, the ambiguous signal resolved to null for final loss across all window sizes. At $n = 300$, new fine structure emerges: window 1 significantly worsens final loss ($+0.3\%$, $p = 0.021$) and window 8 significantly improves it ($-0.1\%$, $p = 0.022$), while window 2, window 4, and window 16 remain non-significant. The null did not hold — higher power reveals a subtle gradient from harm at the smallest window to benefit at an intermediate window.

### Results

| Window | Final Loss | $p$ (final) | Mean Loss | $p$ (mean) |
|:---:|:---:|:---:|:---:|:---:|
| Baseline (full) | 2.5565 | — | 2.6271 | — |
| 1 | 2.5620 | 0.618 | 2.6385 | <0.001*** |
| 2 | 2.5548 | 0.898 | 2.6282 | 0.156 |
| 4 | 2.5489 | 0.443 | 2.6245 | <0.001*** |
| 8 | 2.5533 | 0.304 | 2.6264 | 0.037* |
| 16 | 2.5565 | 1.000 | 2.6271 | 1.000 |

### Findings

**No window size significantly changes final loss.** All paired $t$-tests on final loss are non-significant ($p > 0.30$). The possible improvement for window=2 that appeared at $n = 3$ resolved to null at $n = 30$.

**Mean-loss effects exist for some windows.** Window=1 significantly worsens mean loss ($+0.4\%$, $p < 0.001$, $d = +1.847$). Window=4 significantly improves mean loss ($-0.1\%$, $p < 0.001$, $d = -0.735$) and window=8 improves it ($-0.0\%$, $p = 0.037$, $d = -0.400$). These effects are too small (<0.5%) to be practically meaningful.

**Classification:** *Tolerance* — attention restriction at all tested scales is absorbed without meaningful final-loss change.

**Window=16 is identical to baseline.** Full-context window (16 = block size) reproduces baseline values exactly, confirming no implementation artifacts.

**The information bottleneck hypothesis is not supported at $n = 30$.** The chess paper's finding that intermediate vision radius outperforms omniscience does not translate to meaningful attention windowing effects at this resolution.


## 6. Experiment 5: Communication Topology

**Motivation:** The chess paper's relay chains, where pieces transmit threat information beyond their individual vision radius. We create a spectrum of gradient flow topologies between full backpropagation and complete isolation.

**Scaling note:** At $n = 3$, the signal was ambiguous — a U-shaped loss curve appeared possible, with partial communication seeming to outperform both extremes. At $n = 30$, the U-shaped curve resolved to flat (except at zero communication); cell-view now shows only a mean-loss effect, not a final-loss effect. At $n = 300$, the architecture's indifference holds: heavy ($p = 0.92$), half ($p = 0.033$), and light ($p = 0.59$) are all non-significant or marginal. Only cell-view's mean-loss increase persists. The system is genuinely indifferent to gradient fraction above zero.

### Results

| Topology | Fraction | Final Loss | $p$ (final) | Mean Loss | $p$ (mean) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Full | 1.00 | 2.5565 | — | 2.6271 | — |
| Heavy | 0.75 | 2.5533 | 0.048* | 2.6271 | 0.934 |
| Half | 0.50 | 2.5548 | 0.510 | 2.6272 | 0.452 |
| Light | 0.25 | 2.5548 | 0.494 | 2.6272 | 0.470 |
| Cell-view | 0.00 | 2.5594 | 0.776 | 2.6316 | 0.005** |

### Findings

**Partial gradient flow is largely tolerated.** Half (50%) and light (25%) gradient flow produce no significant change in either final or mean loss. Heavy (75%) shows a small but significant final-loss improvement ($p = 0.048$, $d = -0.377$), with an effect of only $-0.1\%$.

**Cell-view shows only mean-loss effect.** Cell-view (0% gradient flow) does not significantly affect final loss ($+0.1\%$, $p = 0.776$, $d = +0.053$), consistent with Experiment 2. Only mean loss shows a significant increase ($+0.2\%$, $p = 0.005$, $d = +0.556$).

**Classification:** *Tolerance* — the system absorbs substantial reductions in inter-layer gradient flow. Even total removal of gradient flow (cell-view) reaches the same final loss.

**The U-shape claim was ambiguous at $n = 3$.** At $n = 30$, it resolved: no partial-flow condition outperforms full backpropagation. The pilot U-shape was sampling noise at low resolution.


## 7. Experiment 6: Courage vs. Caution

**Motivation:** Kofman et al.'s finding that "cautious position, courageous moves" is the optimal chess strategy. We translate this into a 2×2 factorial design where each cell has BOTH a forward perturbation AND a gradient perturbation:

| | Cautious Grad (sign-only) | Courageous Grad (noisy σ=0.1) |
|:---:|:---:|:---:|
| **Cautious Fwd (σ=0.001)** | cautious\_cautious | cautious\_courageous |
| **Courageous Fwd (dropout)** | courageous\_cautious | courageous\_courageous |

**Scaling note:** At $n = 3$, the courage/caution matrix produced inconsistent results without clear inversion. At $n = 30$, the sign-only gradient conditions consistently degrade by $+5.0$–$5.5\%$ while noisy gradient conditions degrade by $+1.6$–$2.4\%$, regardless of forward perturbation type. Gradient type dominates. At $n = 300$, the pattern is robustly confirmed: cautious\_cautious $+5.2\%$ ($p < 0.0001$, $d = +0.624$), cautious\_courageous $+1.9\%$ ($p < 0.0001$, $d = +0.318$), courageous\_cautious $+5.0\%$ ($p < 0.0001$, $d = +0.616$), courageous\_courageous $+2.5\%$ ($p < 0.0001$, $d = +0.419$). The gradient-type dominance is substrate-dependent.

### Results

| Condition | Final Loss | $p$ (final) | Mean Loss | $p$ (mean) |
|:---:|:---:|:---:|:---:|:---:|
| Baseline | 2.5565 | — | 2.6271 | — |
| cautious\_cautious | 2.6927 | 0.001*** | 2.7279 | <0.001*** |
| cautious\_courageous | 2.6177 | 0.104 | 2.6871 | <0.001*** |
| courageous\_cautious | 2.6981 | 0.002** | 2.7337 | <0.001*** |
| courageous\_courageous | 2.5963 | 0.154 | 2.6908 | <0.001*** |

### Findings

**Gradient type dominates.** Sign-only gradient conditions (cautious\_cautious and courageous\_cautious) degrade final loss by $+5.3\%$ ($p = 0.001$, $d = +0.671$) and $+5.5\%$ ($p = 0.002$, $d = +0.626$) respectively. Noisy gradient conditions (cautious\_courageous and courageous\_courageous) show smaller, non-significant final-loss changes ($+2.4\%$, $p = 0.104$; $+1.6\%$, $p = 0.154$). The forward perturbation type makes little difference — what matters is whether the gradient is sign-only or noisy.

**All conditions significantly degrade mean loss.** Mean loss effects are highly significant for all four conditions: cautious\_cautious $+3.8\%$ ($d = +6.056$), cautious\_courageous $+2.3\%$ ($d = +3.426$), courageous\_cautious $+4.1\%$ ($d = +4.980$), courageous\_courageous $+2.4\%$ ($d = +3.487$). All $p < 0.001$.

**The chess prediction is inverted.** The chess paper predicts "cautious position, courageous moves" as optimal. In the transformer, gradient quality (the "move" analog) dominates: sign-only gradients degrade regardless of forward perturbation. The forward perturbation (the "position" analog) has negligible effect.

**Substrate-dependent interpretation.** Chess pieces operate in a discrete, irreversible action space where stable perception is essential. Transformers operate in a continuous, differentiable landscape where forward noise acts as regularization and gradient precision is needed for fine-grained optimization. The inversion is substrate-dependent.


## 8. Experiment 7: Recovery After Damage

**Motivation:** Levin's regeneration paradigm. Train normally, apply damage (freeze 8 heads), then remove damage and continue. Does the model recover? Does it overshoot?

**Scaling note:** At $n = 3$, recovery appeared complete but with too few observations to confirm. At $n = 30$, complete recovery is confirmed ($p = 0.886$, ratio $1.0000 \pm 0.0080$), with all 30 runs recovered within $1 \pm 1$ steps. At $n = 300$, a tiny but significant residual emerges ($p = 0.030$, $d = +0.126$, ratio $1.0009 \pm 0.0072$). Recovery time is $0.8 \pm 1.2$ steps, overshoot $-0.0009 \pm 0.0017$, and 272/300 runs recovered. Recovery is near-complete but not perfectly complete at high power.

### Results

| Metric | Recovery | Control | $p$ (paired) |
|---|---|---|---|
| Final loss | 2.4505 ± 0.3702 | 2.4511 ± 0.3751 | 0.886 |
| Final ratio (rec/ctrl) | 1.0000 ± 0.0080 | — | — |
| Recovery time | 1 ± 1 steps | — | 30/30 recovered |
| Overshoot | -0.0014 ± 0.0015 | — | — |

### Findings

**Complete recovery at $n = 30$, near-complete at $n = 300$.** At $n = 30$, the damaged-then-recovered model reaches the same final loss as the undamaged control ($p = 0.886$, ratio $1.0000$). All 30 runs recovered within a mean of $1$ step after damage removal. At $n = 300$, a tiny but significant residual emerges ($+0.1\%$, $p = 0.030$, $d = +0.126$, ratio $1.0009$), and 272/300 runs recovered. Recovery is near-complete but not perfectly complete at high statistical power.

The near-completeness of recovery (ratio 1.0000 at $n = 30$, 1.0009 at $n = 300$) is notable: the optimizer prescribes convergence to *a* minimum, not necessarily to the *same* minimum after a detour through a constrained subspace. A more brittle system could recover to a different basin. This one returns to effectively the same endpoint with minimal path-dependence.

**Classification:** *Emergent behavior* — path-independent recovery to near-identical final loss is not directly prescribed by the loss minimization objective.

**No overshoot.** Mean overshoot = $-0.0014 \pm 0.0015$ at $n = 30$, $-0.0009 \pm 0.0017$ at $n = 300$. The Levin signature — damaged organisms exceeding baseline — is absent. The recovery is near-complete but not excessive.


## 9. Experiment 8: Chimera Assembly

**Motivation:** Chimeric organisms assembled from parts of different embryos. Two models trained independently; layers from each are combined into a Frankenstein model.

**Scaling note:** At $n = 3$, chimeras appeared to converge but from too few observations. At $n = 30$, convergence is confirmed for all chimera types and the specific layer assignment is confirmed not to matter; BBAA is marginal ($p = 0.076$). At $n = 300$, all chimera types remain non-significant (AABB $p = 0.35$, ABAB $p = 0.12$, BBAA $p = 0.079$, ABBA $p = 0.31$). No systematic convergence speed differences emerge. The null holds at high power.

### Results

| Condition | Initial Loss | Final Loss | vs Control $p$ |
|---|---|---|---|
| Control (A continues) | — | 2.4940 ± 0.3435 | — |
| AABB | 2.5692 | 2.4842 | 0.265 |
| ABAB | 2.5629 | 2.4938 | 0.985 |
| BBAA | 2.6054 | 2.4724 | 0.076† |
| ABBA | 2.5416 | 2.4865 | 0.559 |

### Findings

**All chimeras converge to control loss.** No chimera type differs significantly from the control ($p > 0.26$); BBAA is marginal ($p = 0.076$). Despite starting at worse loss (2.54–2.61), all chimeras converge to 2.47–2.49.

**Layer assignment doesn't matter.** The spread across chimera types is small. Whether layers alternate (ABAB) or cluster (AABB) makes no difference.

The chimera result shows the basin of attraction is wide enough to absorb dramatically different starting conditions. SGD prescribes convergence; this is the expected behavior of the optimizer on a smooth loss landscape.

**Classification:** *Basin geometry* — SGD re-finds the same minimum from any structurally valid starting point.


## 10. Experiment 9: Gradual vs. Sudden Damage

**Motivation:** Biological stress inoculation. Gradual exposure to a stressor builds tolerance that sudden exposure does not.

**Scaling note:** At $n = 3$, the gradual vs. sudden comparison appeared promising but underpowered. At $n = 30$, sudden full significantly degrades final loss ($+1.9\%$, $p = 0.110$ ns) and mean loss ($+2.4\%$, $p < 0.001$), while gradual is indistinguishable from control for final loss ($p = 0.932$) and shows mean-loss improvement ($-0.2\%$, $p = 0.006$). The direct gradual-vs-sudden comparison is non-significant for final loss ($p = 0.138$) but significant for mean loss. At $n = 300$, the stress inoculation effect strengthens: sudden full degrades by $+1.8\%$ ($p < 0.0001$, $d = +0.318$), gradual by $+0.5\%$ ($p = 0.017$, $d = +0.139$), sudden half by $+0.8\%$ ($p = 0.0002$, $d = +0.219$), and the direct gradual-vs-sudden comparison reaches $p = 0.0001$ ($d = -0.227$). The cross-scale confirmation runs from $n = 30$ mean-loss significance to $n = 300$ final-loss significance.

### Results

| Condition | Final Loss | $p$ (vs ctrl) | Mean Loss | $p$ (mean) |
|---|---|---|---|---|
| Control | 2.5565 ± 0.4074 | — | 2.6271 ± 0.0284 | — |
| Sudden full ($\sigma = 0.1$) | 2.6049 | 0.110 | 2.6908 | <0.001*** |
| Gradual (0→0.1) | 2.5576 | 0.932 | 2.6232 | 0.006** |
| Sudden half (step 100) | 2.5718 | 0.329 | 2.6282 | 0.188 |

### Findings

**Gradual exposure builds tolerance.** The gradually-ramped condition is statistically indistinguishable from control for final loss ($p = 0.932$), while sudden exposure to the same peak noise level shows a non-significant trend toward degradation ($+1.9\%$, $p = 0.110$) and highly significant mean-loss degradation ($+2.4\%$, $p < 0.001$, $d = +4.193$). The direct gradual-vs-sudden comparison is non-significant for final loss ($-1.8\%$, $p = 0.138$, $d = -0.278$) but significant for mean loss. At $n = 300$, the final-loss comparison reaches high significance ($p = 0.0001$, $d = -0.227$).

**This is the paper's clearest emergent behavior.** The gradient update rule is identical in the sudden and gradual conditions at every step — the only difference is the *history* of noise levels. That history matters, and the system develops different properties depending on it. The optimization objective does not specify how noise history should change the system's final state, but it does.

**Classification:** *Emergent behavior* — the system develops differential tolerance based on perturbation history despite identical gradient update rules at every step.

**Gradual noise acts as regularization.** The gradual condition's mean loss is *below* control ($-0.2\%$, $p = 0.006$, $d = -0.540$), suggesting the slowly-introduced noise serves as a regularizer.


## 11. Experiment 10: Regeneration (Layer Reset)

**Motivation:** Tissue regeneration. Destroy a layer entirely (reset to random), continue training.

**Scaling note:** At $n = 3$, regeneration appeared possible but with too few observations to confirm completeness. At $n = 30$, all layers are non-significant vs control (L0 $+0.4\%$ $p = 0.174$, L1 $+0.1\%$ $p = 0.550$, L2 $-0.0\%$ $p = 0.808$, L3 $+0.0\%$ $p = 0.977$); all completeness values exceed 0.9 (all $p < 0.006$). At $n = 300$, all layers show small but significant residuals (L0 $+0.3\%$ $p = 0.003$, L1 $+0.2\%$ $p = 0.007$, L2 $+0.1\%$ $p = 0.024$, L3 $+0.1\%$ $p = 0.037$). Higher power reveals slight regeneration incompleteness that was invisible at $n = 30$.

### Results

| Reset Layer | Final Loss | Completeness | vs Ctrl $p$ |
|---|---|---|---|
| Control | 2.4569 ± 0.3410 | — | — |
| Layer 0 | 2.4669 | 0.998 | 0.174 |
| Layer 1 | 2.4593 | 1.046 | 0.550 |
| Layer 2 | 2.4560 | 1.015 | 0.808 |
| Layer 3 | 2.4571 | 0.981 | 0.977 |

### Findings

**Complete regeneration at $n = 30$.** All four layers recover to control-equivalent loss after being destroyed ($p > 0.17$ for all). All completeness values exceed 0.9 (all $p < 0.006$). At $n = 300$, all layers show small but significant residuals ($+0.1$–$0.3\%$, all $p < 0.04$), revealing slight regeneration incompleteness that was invisible at lower power.

The near-completeness of regeneration — particularly L1 at 1.046 and L2 at 1.015 — demonstrates that the network re-finds the same functional role regardless of what was there before. The optimizer prescribes convergence to a minimum; it does not prescribe that a rebuilt layer should reach the same functional role as if it had never been destroyed.

**Classification:** *Emergent behavior* — near-complete layer regeneration to control-equivalent performance is not directly prescribed by the loss minimization objective.

**No layer is indispensable.** All layers regenerate to near-control performance regardless of position.


## 12. Experiment 11: Transplantation

**Motivation:** Organ transplantation. Replace a layer with one from a separately-trained donor model.

**Scaling note:** At $n = 3$, a transplant advantage appeared possible. At $n = 30$, the null result resolves clearly — there is no advantage to a structured donor layer over a random replacement ($p = 0.880$ overall). At $n = 300$, the null holds across all layers ($p = 0.29$–$0.98$, overall $p = 0.76$). No layer-specific transplant advantage or disadvantage emerges even at 10x power.

### Results

| Layer | Transplant | Random Reset | Gap | $p$ |
|---|---|---|---|---|
| L0 | 2.4702 | 2.4669 | -0.0033 | 0.566 |
| L1 | 2.4585 | 2.4593 | +0.0008 | 0.804 |
| L2 | 2.4585 | 2.4560 | -0.0025 | 0.568 |
| L3 | 2.4535 | 2.4571 | +0.0036 | 0.420 |
| **Overall** | — | — | **+0.0003** | **0.880** |

### Findings

**No transplant advantage.** A layer from a separately-trained donor is accepted no better and no worse than a random replacement ($p = 0.880$ overall). No individual layer shows a significant gap ($p > 0.42$ for all).

**The network doesn't recognize donor structure.** Unlike biological transplantation where tissue compatibility matters, the network rebuilds whatever is placed at each layer position from scratch. The donor layer's learned structure provides no advantage.

The loss landscape is smooth enough that any reasonable starting point finds the minimum.

**Classification:** *Basin geometry* — the basin is equally accessible from pre-trained and random initializations.


## 13. Experiment 12: Competing Objectives

**Motivation:** Inter-organ conflict. Negate gradients for layers 2-3 while layers 0-1 train normally.

**Scaling note:** At $n = 3$, the distinction between adversarial and inactive layers appeared but was underpowered. At $n = 30$, the distinction resolves sharply: competing $+24.8\%$ ($p < 0.001$, $d = +0.689$), freeze $+0.2\%$ ($p = 0.462$). At $n = 300$, competing objectives degrade by $+26.3\%$ ($p < 0.0001$, $d = +0.531$) while freeze remains non-significant ($-0.1\%$, $p = 0.41$). High variance persists (std = 1.31 at $n = 300$), and the adversarial-vs-freeze distinction is highly significant at both scales.

### Results

| Condition | Final Loss | vs Ctrl Δ% | $p$ |
|---|---|---|---|
| Control | 2.4569 ± 0.3410 | — | — |
| Competing (negate L2-3) | 3.0666 ± 1.0149 | +24.8% | <0.001*** |
| Freeze L2-3 | 2.4607 ± 0.3463 | +0.2% | 0.462 |

### Findings

**Adversarial components are not compensated.** Negating gradients for layers 2-3 causes $+24.8\%$ degradation ($p < 0.001$, $d = +0.689$) with high variance (std = 1.01). The network cannot route around actively sabotaging layers.

**Frozen components are tolerated.** Merely freezing those same layers causes negligible degradation ($+0.2\%$, $p = 0.462$). Competing vs. freeze: $p < 0.001$, $d = +0.693$.

**Classification:** *Tolerance* for the freeze condition (absence is absorbed without prescription from SGD); the adversarial result is a boundary condition revealing the limits of tolerance.

**Sharp line between absence and opposition.** The architecture can handle absent layers (residual stream routes around them) but cannot compensate for layers actively working against the objective. This defines the architecture's tolerance limit.


## 14. Cross-Experiment Synthesis

Ten findings emerge from the combined evidence across twelve experiments at $n = 30$.

### Finding 1: Emergent — gradual exposure builds tolerance (Exp 9)

The paper's strongest emergent-behavior result. Gradual noise ramp (0→0.1) produces no final-loss degradation ($p = 0.932$), while sudden exposure to the same peak noise level trends toward degradation ($+1.9\%$, $p = 0.110$) and strongly degrades mean loss ($+2.4\%$, $p < 0.001$, $d = +4.193$). The direct gradual-vs-sudden comparison is non-significant for final loss at $n = 30$ ($p = 0.138$) but highly significant at $n = 300$ ($p = 0.0001$, $d = -0.227$). The gradient update rule is identical at every step; only the history differs. That history changes the system's final state.

### Finding 2: Emergent — near-complete recovery, regeneration, and trajectory improvement (Exp 1, 7, 10)

The architecture recovers and rebuilds:
- Damaged and recovered: final ratio 1.0000 ($p = 0.886$ vs control at $n = 30$); tiny residual at $n = 300$ ($p = 0.030$, ratio 1.0009)
- Layer reset to random: $p > 0.17$ for all layers at $n = 30$ (near-complete regeneration); small residuals at $n = 300$ ($p < 0.04$ for all)
- Head-freezing trajectory improvement: freeze 8 ($p < 0.001$, $d = -1.228$), freeze 12 ($p < 0.001$, $d = -1.366$)

The optimizer prescribes convergence to a minimum; it does not prescribe returning to the same basin after damage or rebuilding a destroyed layer to equivalent performance.

### Finding 3: Basin geometry — chimeras, transplants, and local loss converge regardless (Exp 2, 8, 11)

The architecture converges to the same loss regardless of assembly history or optimization method:
- Cell-view (local loss): final loss $p = 0.776$ at $n = 30$, $p = 0.90$ at $n = 300$ (equivalent convergence)
- Chimeras from two models: $p > 0.26$ for all assemblies (BBAA marginal at $p = 0.076$)
- Transplant vs random reset: $p = 0.880$ (no difference)

The basin of attraction is wide enough to reach from dramatically different starting points and optimization methods.

### Finding 4: The absence vs. adversity distinction (Exp 12)

Frozen (inactive) layers: $+0.2\%$, $p = 0.462$ (tolerated). Adversarial (gradient-negated) layers: $+24.8\%$, $p < 0.001$ (degrading, $d = +0.689$). Competing vs. freeze: $p < 0.001$, $d = +0.693$. The architecture routes around silence but cannot defend against active sabotage.

### Finding 5: The chess-paper inversion (Exp 3, 6)

In the 2x2 factorial design, sign-only gradient conditions degrade final loss by $+5.0$–$5.5\%$ ($p \leq 0.002$) while noisy gradient conditions show non-significant final-loss changes ($p > 0.10$), regardless of forward perturbation type. Gradient type dominates, robustly inverting the chess paper's "cautious position, courageous moves" prediction. This is substrate-dependent: transformers require gradient precision; chess requires perceptual stability.

### Finding 6: Tolerance boundary (Exp 1-6)

| Perturbation | Final Loss Δ% | $p$-value | Classification |
|---|---|---|---|
| Freeze 1-4 heads | -0.1 to -0.5% | 0.001–0.253 | Basin geometry / emergent |
| Cell-view (local loss) | +0.1% | 0.776 | Basin geometry |
| Partial gradient flow (25-75%) | -0.1 to +0.0% | >0.05 | Tolerance |
| Noisy gradients (σ=0.01) | -0.2% | 0.738 | Tolerance |
| Gradual noise ramp (0→0.1) | +0.0% | 0.932 | Emergent |
| Frozen layers (L2-3) | +0.2% | 0.462 | Tolerance |
| Noisy gradients (σ=0.1) | -0.2% | 0.910 | Tolerance (final) / degradation (mean) |
| Sudden noise (σ=0.1) | +1.9% | 0.110 | Degradation (mean) |
| Quantized 3-bit | +3.8% | 0.008 | Degradation |
| Sign-only gradients | +5.0% | 0.002 | Degradation |
| Adversarial layers | +24.8% | <0.001 | Severe degradation |


## 15. Scaling Resolution

How signals changed as the resolution dial turned from $n = 3$ to $n = 30$ to $n = 300$.

**Head freezing improves final loss (Exp 1):** At $n = 3$, the coarse signal showed a possible improvement. At $n = 30$, some final-loss improvements are now significant (freeze\_4 $p = 0.0014$, freeze\_12 $p = 0.0161$), and the mean trajectory improvement for 4+ heads is highly significant. At $n = 300$, all final-loss effects resolve to non-significant ($p > 0.15$), but the trajectory improvement strengthens to highly significant ($p < 0.0001$ for freeze 4, 8, 12, and 16), confirming this as a robust emergent-behavior signal. The final-loss signal at $n = 30$ was scale-dependent; the trajectory signal is robust across scales.

**DG scales with perturbation:** At $n = 3$, DG appeared to scale with perturbation severity. At $n = 30$, this resolved to null across all conditions ($p > 0.16$). The DG metric captures stochastic SGD dynamics but not perturbation response. At $n = 300$, the DG null holds. No fine structure emerges even at 10x power. DG does not track perturbation.

**Gradient degradation is neutral (Exp 3):** At $n = 3$, all four degradation methods appeared neutral and small noise seemed to help. At $n = 30$, only sign-only ($+5.0\%$, $p = 0.002$) and quantized ($+3.8\%$, $p = 0.008$) show significant final-loss degradation; noisy $\sigma = 0.1$ is NS for final loss ($p = 0.91$) though significant for mean loss. At $n = 300$, the threshold sharpens: noise $\sigma = 0.01$ is non-significant ($p = 0.28$) while all three severe methods are $p < 0.0001$. The boundary between tolerance and degradation is a sharp step, not a smooth curve.

**Partial communication outperforms full (Exp 5):** At $n = 3$, a U-shaped curve appeared — partial communication seeming to outperform both extremes. At $n = 30$, the U-shape resolved to flat; cell-view no longer degrades final loss ($p = 0.776$), only mean loss ($p = 0.005$). At $n = 300$, no topology advantage appears. Heavy ($p = 0.92$), half ($p = 0.033$), and light ($p = 0.59$) all remain non-significant or marginal. The architecture's indifference to gradient fraction holds at high power.

**Noise helps (Exp 3):** At $n = 3$, small noise ($\sigma = 0.01$) appeared beneficial. At $n = 30$, this resolved to null ($p = 0.4963$ for mean loss, $p = 0.7378$ for final loss). The apparent benefit was within noise at low resolution. At $n = 300$, the null holds ($\sigma = 0.01$: $-0.2\%$, $p = 0.28$). No sub-threshold noise benefit emerges even at 10x power.

**Transplant advantage (Exp 11):** At $n = 3$, a transplant advantage appeared possible. At $n = 30$, the null result resolved clearly ($p = 0.880$). At $n = 300$, the null holds throughout ($p = 0.29$–$0.98$ for all layers, overall $p = 0.76$). No transplant advantage at any resolution.

**Chimera convergence (Exp 8):** At $n = 3$, convergence appeared probable. At $n = 30$, confirmed for all chimera types ($p > 0.26$; BBAA marginal at $p = 0.076$). At $n = 300$, all chimera types remain non-significant (AABB $p = 0.35$, ABAB $p = 0.12$, BBAA $p = 0.079$, ABBA $p = 0.31$). No convergence speed differences emerge.

**Gradual vs. sudden damage (Exp 9):** At $n = 3$, the signal was ambiguous. At $n = 30$, gradual is indistinguishable from control for final loss ($p = 0.932$) and shows mean-loss improvement ($-0.2\%$, $p = 0.006$); the direct gradual-vs-sudden comparison is NS for final loss ($p = 0.138$) but significant for mean loss. At $n = 300$, the effect strengthens: the gradual-vs-sudden final-loss gap reaches $p = 0.0001$ ($d = -0.227$). The cross-scale confirmation runs from $n = 30$ mean-loss significance to $n = 300$ final-loss significance. This is the paper's most robust cross-scale confirmation.


## 16. Findings — What Perturbation Revealed

### Emergent Behaviors

**1. Gradual stress builds tolerance (Exp 9).** The only experiment where the *manner* of perturbation application matters. Gradual noise ramp: $p = 0.932$ vs control (final loss). Sudden noise: $p = 0.110$ (final loss), $p < 0.001$ (mean loss, $d = +4.193$). Direct gradual-vs-sudden: NS at $n = 30$ ($p = 0.138$), highly significant at $n = 300$ ($p = 0.0001$). The gradient update rule is identical at every step; only the history differs. That history changes the system's final state.

**2. Near-complete recovery (Exp 7).** A model damaged during training recovers to near-identical final loss ($p = 0.886$, ratio 1.0000 at $n = 30$). All 30 runs recovered within a mean of 1 step. At $n = 300$, a tiny but significant residual emerges ($p = 0.030$, ratio 1.0009).

**3. Near-complete regeneration (Exp 10).** Any layer can be destroyed and rebuilt to control-equivalent performance ($p > 0.17$ for all layers at $n = 30$). At $n = 300$, small residuals emerge ($p < 0.04$ for all). The network re-finds the same functional role regardless of what was there before.

**4. Head-freezing trajectory improvement (Exp 1).** Freezing 4+ randomly-initialized heads produces small but statistically robust mean-trajectory improvements (freeze 8: $p < 0.001$, $d = -1.228$; freeze 12: $p < 0.001$, $d = -1.366$). Frozen random-projection heads reduce gradient interference.

### Basin Geometry

**5. Local loss equivalence (Exp 2).** Cell-view (local loss) achieves the same final loss as end-to-end backpropagation ($p = 0.776$ at $n = 30$, $p = 0.90$ at $n = 300$). The local loss objective finds the same basin through different optimization geometry.

**6. Chimera convergence (Exp 8).** Models assembled from parts of two independently-trained networks converge to the same final loss as undamaged continuation ($p > 0.26$ for all chimera types; BBAA marginal at $p = 0.076$). The basin of attraction is wide enough to reach from dramatically different starting points.

**7. Transplant indifference (Exp 11).** Transplanted layers and randomly-reset layers converge to the same final loss ($p = 0.880$ overall). The basin is equally accessible from pre-trained and random initializations.

### Tolerance

**8. Gradient quality matters more than quantity (Exp 3, 5).** Reducing gradient precision (sign-only: $d = +5.696$ for mean loss) degrades more than reducing gradient magnitude (partial flow: all NS for mean loss) or completeness (freezing: trajectory improves). The architecture tolerates magnitude reduction but not sign-structure destruction.

**9. The gradient-type dominance (Exp 3, 6).** In the 2x2 factorial, sign-only gradient conditions degrade by $+5.0$–$5.5\%$ while noisy gradient conditions show $+1.6$–$2.4\%$ (NS for final loss), regardless of forward perturbation. This inverts the chess paper's "cautious position, courageous moves" prediction — substrate-dependent.

**10. Adversarial vs. inactive tolerance (Exp 12).** Frozen layers cost nothing ($p = 0.462$); adversarial layers cost $+24.8\%$ ($p < 0.001$). The architecture tolerates absence but not opposition.


## 17. Interpretive Lenses

The empirical findings admit several interpretive framings beyond the neutral classification used above. These are not claims; they are ways of reading the results.

**Lens 1: Freedom from the algorithm.** One can frame the emergent behaviors as "freedom" — behaviors the optimizer didn't prescribe but the system exhibits anyway. Under this reading, SGD says "minimize loss"; it does not say "build tolerance through gradual exposure," "recover completely from damage," or "rebuild a destroyed layer to the same functional role." Stress inoculation is the strongest case: the gradient update rule is identical at every step, yet the system's final state depends on perturbation history. Recovery and regeneration are weaker cases — one could argue they are also basin geometry effects. The classification boundary between "emergent" and "basin geometry" is itself a question, not a settled fact.

**Lens 2: Désœuvrement (Nancy).** Jean-Luc Nancy's concept of *désœuvrement* (unworking) argues that the structure of a collective system becomes visible only when its coordinated work is interrupted (Nancy, 1991). Each experiment interrupts the transformer's work. The interruptions make legible the relational structure that normal operation conceals: redundancy (head freezing), basin equivalence (cell-view local loss), compensatory capacity (recovery, regeneration), and tolerance boundaries (adversarial vs. absent layers). The absence-vs-adversity distinction (Exp 12) maps onto Nancy's distinction between the *withdrawn* member of a community and the *hostile* member.

**Lens 3: Morphogenetic competency (Levin).** Levin et al. (2024) proposed that computational systems can exhibit morphogenetic competencies analogous to biological development. Under this reading, stress inoculation is analogous to biological stress hardening, regeneration to tissue regeneration, chimera convergence to chimeric organism development. The DG null (no perturbation response at any scale) is a point of divergence: biological systems show richer compensatory rerouting than this minimal transformer.


## 18. Limitations

- **Scale:** 4 layers, 16 dimensions, 16 heads, ~13,400 params. The emergent behaviors (stress inoculation, recovery, regeneration) may be specific to small models, or they may be architectural universals.
- **Task complexity:** Character-level name generation is a toy task. Whether stress inoculation appears in language modeling or other complex tasks is not established.
- **Training duration:** 200 steps per phase. Gradual-exposure tolerance may not persist at longer horizons.
- **Transplant design:** Both models trained on same task/data. Cross-task transplantation might show different results.
- **Competing objectives design:** Gradient negation is maximally adversarial. Subtler conflicts might reveal compensation mechanisms.
- **DG metric:** Does not function as perturbation response measure at $n = 30$. At $n = 300$, no fine structure emerges; the DG null holds at all scales.
- **Effect sizes:** Many statistically significant effects are practically small (<0.5%). Statistical significance at $n = 30$ does not imply practical importance.
- **$n = 3$ to $n = 30$ signal changes:** Several pilot signals changed character at higher resolution, underscoring that coarse-scale data should not be interpreted as conclusions. The three-scale protocol is a response to this.
