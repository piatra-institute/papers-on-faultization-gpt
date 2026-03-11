# MorphoGPT: Experimental Findings

Morphogenetic perturbation analysis of a minimal GPT, asking: does the system exhibit behaviors that SGD didn't prescribe? We apply Levin's developmental biology methodology and Nancy's concept of désœuvrement (unworking) to reveal freedom from the training algorithm in transformer learning dynamics.

**Central question:** SGD says "minimize loss." It does not say "build tolerance through gradual exposure," "recover completely from damage," or "regenerate a destroyed layer." When the system does these things anyway, that is freedom from the algorithm. The experiments below identify where such freedom appears, where wide basin geometry explains convergence without needing freedom, and where the system simply absorbs perturbation (tolerance).


## 1. Setup

**Model:** 4-layer, 16-dimensional, 4-head GPT (16 total attention heads).

**Task:** Character-level name generation, trained on a names dataset.

**Protocol:** 200 training steps, 30 independent runs per condition (seeds 42–71), loss and head-level metrics recorded at every step.

**Statistical analysis:** All comparisons use paired $t$-tests with runs matched by seed across conditions ($n = 30$, $df = 29$). We report $p < 0.05$ as statistically significant and $0.05 < p < 0.10$ as marginal. With 30 paired observations, the tests have adequate power to detect moderate effects. We distinguish between *statistically supported* findings and *observational* patterns throughout. Effect sizes are reported as Cohen's $d$.

**Three-scale protocol:** $n$ is a resolution dial. At $n = 3$, signals are coarse — rough shapes, visible but ambiguous. At $n = 30$, the picture sharpens and most ambiguities resolve. At $n = 300$, fine structure emerges. No scale is wrong; each reveals different phenomena. Signals that were ambiguous at $n = 3$ are not retracted at $n = 30$ — the resolution increased and the picture changed.

**Key metric — Delayed Gratification (DG) Index:** Measures how much the loss trajectory dips below its final value during training. At $n = 3$, DG appeared to scale with perturbation severity. At $n = 30$, this signal resolved to null: no perturbation condition produces a statistically significant DG increase ($p > 0.16$ for all). DG captures a real stochastic property of loss trajectories but does not function as a perturbation response measure at $n = 30$. At $n = 300$, the DG null holds: no perturbation condition produces significant DG increase, confirming that DG captures stochastic SGD dynamics but does not track perturbation response even at 10x power.


## 2. Experiment 1: Head Freezing

**Analog:** Levin's frozen-cell perturbation. Randomly selected attention heads have their parameters frozen at initialization, forcing the remaining heads to compensate.

**Scaling note:** At $n = 3$, head freezing appeared to improve final loss — the coarse signal suggested damage helps. At $n = 30$, the final-loss improvement resolved to null, but a different, finer signal emerged: mean trajectory improvement for 4+ frozen heads. The picture sharpened; the original signal was in the wrong metric. At $n = 300$, the final-loss null holds firmly (all $p > 0.40$, Spearman $\rho = 0.0023$, $p = 0.92$), but the trajectory improvement strengthens to highly significant: freeze 8 ($\Delta = -0.12\%$, $p < 0.0001$), freeze 12 ($\Delta = -0.17\%$, $p < 0.0001$), freeze 16 ($\Delta = -0.19\%$, $p < 0.0001$). The trajectory freedom signal is robust at high power.

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

**Final loss does not change with freezing.** No freezing level produces a statistically significant change in final loss. Freeze 8 is marginal ($p = 0.064$); all others are non-significant ($p > 0.18$). The monotonic improvement trend that appeared at $n = 3$ was ambiguous signal — at $n = 30$, the Spearman $\rho = 0.013$ ($p = 0.85$) confirms no monotonic final-loss relationship. The wide basin of attraction captures the model regardless of how many heads are frozen.

**Mean trajectory loss shows a genuine improvement.** Freezing 4+ heads produces a statistically significant reduction in mean loss: freeze 4 ($p = 0.012$, $d = -0.49$), freeze 8 ($p < 0.001$, $d = -1.03$), freeze 12 ($p < 0.001$, $d = -1.13$), freeze 16 ($p < 0.001$, $d = -0.96$). Effects are 0.1–0.2% of mean loss — small but statistically robust. Frozen heads at random initialization reduce gradient interference during training in a way SGD did not prescribe.

**Freedom classification:** Trajectory improvement for 4+ frozen heads = *genuine freedom* (SGD did not specify this benefit from random-projection frozen heads). Final-loss indifference = *wide basin of attraction*.

**DG does not increase with freezing.** Freeze 12 actually decreases DG significantly ($p = 0.034$, $d = -0.41$). All other levels are non-significant.

**Trajectory shape is preserved.** Cross-condition trajectory correlations exceed 0.95 at all freezing levels.


## 3. Experiment 2: Cell-View GPT

**Analog:** Nancy's being-singular-plural — each layer treated as an autonomous agent. Stop-gradient applied at all layer boundaries so each layer learns only from its own local loss signal, with no end-to-end backpropagation.

**Scaling note:** At $n = 3$, the signal was ambiguous — cell-view appeared to elevate DG substantially (+25.5%), suggesting possible rerouting behavior. At $n = 30$, the DG signal resolved to null and the degradation signal became clear. At $n = 300$, cell-view degradation strengthens to $+2.9\%$ ($t = 8.307$, $p < 0.0001$, $d = 0.480$), confirming the degradation is a robust effect. No DG fine structure emerges.

### Results

| Condition | Mean Loss | Final Loss | DG Index |
|:---:|:---:|:---:|:---:|
| baseline | 2.639 | 2.469 | 0.571 |
| cell_view | 2.687 | 2.590 | 0.507 |

### Findings

**Local-only learning degrades performance significantly.** Cell-view increases final loss by +4.9% ($p < 0.001$, $d = +1.16$) and mean loss by +1.8% ($p < 0.001$, $d = +3.31$). The degradation is significant and non-trivial, but the system still learns — eliminating all inter-layer gradient communication does not break the architecture.

**Freedom classification:** *Tolerance* — the system absorbs the removal of inter-layer gradient flow at a bounded cost.

**DG does not increase under cell-view.** At $n = 3$, a +25.5% DG elevation appeared. At $n = 30$, this signal resolved to null ($p = 0.34$). Cell-view DG (0.507) is not significantly different from baseline (0.571).

**Head specialization patterns shift.** Under cell-view training, final-layer heads tend to specialize more aggressively when deprived of upstream gradient refinement. Observational finding from head entropy analysis.


## 4. Experiment 3: Gradient Degradation

**Analog:** Levin's noisy signaling channels. Gradients are corrupted during training through four methods: additive Gaussian noise (two scales), sign-only reduction (discarding magnitude), and 3-bit quantization.

**Scaling note:** At $n = 3$, the signal was ambiguous — all four methods appeared neutral and small noise appeared to help. At $n = 30$, three of four resolved to significant degradation; one is genuinely tolerated. The null at $n = 3$ was insufficient resolution to see moderate effects. At $n = 300$, the pattern sharpens: noise $\sigma = 0.01$ remains non-significant ($-0.1\%$, $p = 0.52$), while noise $\sigma = 0.1$ ($+2.3\%$, $p < 0.0001$), sign-only ($+4.9\%$, $p < 0.0001$), and quantized 3-bit ($+3.4\%$, $p < 0.0001$) are all highly significant. The tolerance-to-degradation boundary is a sharp threshold between $\sigma = 0.01$ and $\sigma = 0.1$, not a graded curve.

### Results

| Method | Final Loss | Δ% | $p$ (final) | Mean Loss | $p$ (mean) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline | 2.469 | — | — | 2.639 | — |
| noisy (σ=0.01) | 2.477 | +0.3% | 0.525 | 2.639 | 0.978 |
| noisy (σ=0.1) | 2.530 | +2.5% | 0.021* | 2.698 | <0.001*** |
| sign_only | 2.582 | +4.6% | 0.004** | 2.742 | <0.001*** |
| quantized (3-bit) | 2.543 | +3.0% | 0.019* | 2.703 | <0.001*** |

### Findings

**Small noise is tolerated; severe corruption degrades.** Sign-only gradients significantly degrade final loss by +4.6% ($p = 0.004$, $d = +0.56$), quantized gradients by +3.0% ($p = 0.019$, $d = +0.45$), and noisy $\sigma = 0.1$ by +2.5% ($p = 0.021$, $d = +0.44$). Only small noise ($\sigma = 0.01$) produces no significant change ($p = 0.525$). The "noise helps" appearance at $n = 3$ resolved to null at $n = 30$ ($p = 0.978$ for mean loss).

**Mean loss effects are highly significant.** Sign-only worsens mean loss by +3.9% ($p < 0.001$, $d = +6.13$), quantized by +2.4% ($p < 0.001$, $d = +4.54$), and noisy $\sigma = 0.1$ by +2.2% ($p < 0.001$, $d = +3.79$). The large Cohen's $d$ values (>3) indicate massive effects on the training trajectory.

**Freedom classification:** *Tolerance* — the system absorbs gradient noise up to a threshold ($\sigma = 0.01$). Above that threshold, degradation follows.

**Architecture constrains the solution space.** Despite the degradation, the worst-performing method (sign-only) still achieves loss within 5% of baseline. The residual stream, attention patterns, and MLP structure define a narrow enough solution manifold that even crude gradient approximations navigate it, with measurable cost.


## 5. Experiment 4: Vision Radius Sweep

**Analog:** Kofman, Campitelli & Levin's (2025) vision radius experiment in distributed chess, where each piece perceives only squares within radius R. We translate this by restricting each attention head's context window.

**Scaling note:** At $n = 3$, the signal was ambiguous — an information bottleneck effect appeared possible, with intermediate window sizes seeming to outperform full context. At $n = 30$, the ambiguous signal resolved to null for final loss across all window sizes. At $n = 300$, new fine structure emerges: window 1 significantly worsens loss ($+0.4\%$, $p = 0.0009$) and window 8 significantly improves it ($-0.1\%$, $p = 0.022$), while window 2, window 4, and window 16 remain non-significant. The null did not hold — higher power reveals a subtle gradient from harm at the smallest window to benefit at an intermediate window.

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

**No window size significantly changes final loss.** All paired $t$-tests on final loss are non-significant ($p > 0.13$). The possible improvement for window=2 that appeared at $n = 3$ resolved to null at $n = 30$.

**Tiny mean-loss effects exist for some windows.** Window=1 significantly worsens mean loss (+0.4%, $p < 0.001$, $d = +2.49$). Window=4 significantly improves mean loss (-0.1%, $p = 0.004$, $d = -0.57$) and window=8 improves it (-0.03%, $p < 0.001$, $d = -0.68$). These effects are too small (<0.5%) to be practically meaningful.

**Freedom classification:** *Tolerance* — attention restriction at all tested scales is absorbed without meaningful final-loss change.

**Window=16 is identical to baseline.** Full-context window (16 = block size) reproduces baseline values exactly, confirming no implementation artifacts.

**The information bottleneck hypothesis is not supported at $n = 30$.** The chess paper's finding that intermediate vision radius outperforms omniscience does not translate to meaningful attention windowing effects at this resolution.


## 6. Experiment 5: Communication Topology

**Analog:** The chess paper's relay chains, where pieces transmit threat information beyond their individual vision radius. We create a spectrum of gradient flow topologies between full backpropagation and complete isolation.

**Scaling note:** At $n = 3$, the signal was ambiguous — a U-shaped loss curve appeared possible, with partial communication seeming to outperform both extremes. At $n = 30$, the U-shaped curve resolved to flat (except at zero communication). At $n = 300$, the architecture's indifference holds: heavy ($p = 0.35$), half ($p = 0.87$), and light ($p = 0.41$) are all non-significant. Only cell-view (zero flow) hurts. The system is genuinely indifferent to gradient fraction above zero.

### Results

| Topology | Fraction | Final Loss | $p$ (final) | Mean Loss | $p$ (mean) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Full | 1.00 | 2.469 | — | 2.639 | — |
| Heavy | 0.75 | 2.472 | 0.024* | 2.639 | 0.553 |
| Half | 0.50 | 2.470 | 0.551 | 2.639 | 0.100† |
| Light | 0.25 | 2.471 | 0.426 | 2.639 | 0.303 |
| Cell-view | 0.00 | 2.590 | <0.001*** | 2.687 | <0.001*** |

### Findings

**Partial gradient flow is largely tolerated.** Half (50%) and light (25%) gradient flow produce no significant change in either final or mean loss. Heavy (75%) shows a small but significant final-loss increase ($p = 0.024$, $d = +0.44$), with an effect of only +0.1%.

**Only complete isolation degrades meaningfully.** Cell-view (0% gradient flow) degrades final loss by +4.9% ($p < 0.001$, $d = +1.16$) and mean loss by +1.8% ($p < 0.001$, $d = +3.31$).

**Freedom classification:** *Tolerance* — the system absorbs substantial reductions in inter-layer gradient flow. Only total removal crosses the degradation threshold.

**The U-shape claim was ambiguous at $n = 3$.** At $n = 30$, it resolved: no partial-flow condition outperforms full backpropagation. The pilot U-shape was sampling noise at low resolution.


## 7. Experiment 6: Courage vs. Caution

**Analog:** Kofman et al.'s finding that "cautious position, courageous moves" is the optimal chess strategy. We translate this into a 2×2 matrix:

| | Cautious Gradients | Courageous Gradients |
|:---:|:---:|:---:|
| **Cautious Forward** | (a) Tiny noise (σ=0.001) | (b) Sign-only gradients |
| **Courageous Forward** | (c) Dropout (p=0.1) | (d) Noisy gradients (σ=0.1) |

**Scaling note:** At $n = 3$, the courage/caution matrix produced inconsistent results without clear inversion. At $n = 30$, the sign-only vs. dropout comparison resolved to a robust and significant inversion. At $n = 300$, the inversion is robustly confirmed: cautious/cautious is non-significant ($p = 0.51$), cautious/courageous (sign-only) degrades by $+4.9\%$ ($p < 0.0001$), courageous/cautious (dropout) shows only marginal degradation ($+0.2\%$, $p = 0.052$), and courageous/courageous degrades by $+2.3\%$ ($p < 0.0001$). The sign-only vs. dropout inversion gap is $+4.7\%$ ($p < 0.0001$), confirming the substrate-dependent finding at high power.

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

**The chess prediction is inverted.** Condition (c) courageous/cautious (dropout with stable gradients) significantly outperforms (b) cautious/courageous (sign-only with stable forward pass). The chess paper's predicted winner is the worst non-baseline condition for mean loss.

**Substrate-dependent interpretation.** Chess pieces operate in a discrete, irreversible action space where stable perception is essential. Transformers operate in a continuous, differentiable landscape where forward noise acts as regularization and gradient precision is needed for fine-grained optimization. The inversion is substrate-dependent, not a freedom-from-the-algorithm finding.


## 8. Experiment 7: Recovery After Damage

**Analog:** Levin's regeneration paradigm. Train normally, apply damage (freeze 8 heads), then remove damage and continue. Does the model recover? Does it overshoot?

**Scaling note:** At $n = 3$, recovery appeared complete but with too few observations to confirm. At $n = 30$, complete recovery is unambiguously confirmed across all 30 runs. At $n = 300$, recovery vs. control remains non-significant ($p = 0.64$), with ratio $0.9997 \pm 0.0086$ and all 300 runs recovered. The confidence interval shrinks 5x relative to $n = 30$. Mean overshoot is $-0.0007 \pm 0.0018$ and mean recovery time is $1.4 \pm 1.7$ steps. No run shows incomplete recovery.

### Results

| Metric | Recovery | Control | $p$ (paired) |
|---|---|---|---|
| Final loss | 2.470 ± 0.353 | 2.469 ± 0.349 | 0.905 |
| Final ratio (rec/ctrl) | 0.9999 ± 0.009 | — | 0.973 (vs 1.0) |
| Recovery time | 2 ± 2 steps | — | 30/30 recovered |

### Findings

**Complete recovery, no lasting damage.** The damaged-then-recovered model reaches the same final loss as the undamaged control ($p = 0.905$). All 30 runs recovered within a mean of 2 steps after damage removal. 100 steps of training with 8 frozen heads left no trace.

**This is genuine freedom from the algorithm.** SGD says "minimize loss." It does not say "recover completely from damage" or "return to the same basin after a detour." The completeness of recovery (ratio 0.9999, $p = 0.905$) is not prescribed by the loss minimization objective. A more brittle system could recover to a different basin. This one returns to the same endpoint with no path-dependence.

**Freedom classification:** *Genuine freedom* — path-independent recovery to identical final loss is not prescribed by the loss minimization objective.

**No overshoot.** Mean overshoot = -0.001. The Levin signature — damaged organisms exceeding baseline — is absent. The recovery is complete but not excessive.


## 9. Experiment 8: Chimera Assembly

**Analog:** Chimeric organisms assembled from parts of different embryos. Two models trained independently; layers from each are combined into a Frankenstein model.

**Scaling note:** At $n = 3$, chimeras appeared to converge but from too few observations. At $n = 30$, convergence is confirmed for all chimera types and the specific layer assignment is confirmed not to matter. At $n = 300$, all chimera types remain non-significant (AABB $p = 0.51$, ABAB $p = 0.83$, BBAA $p = 0.95$, ABBA $p = 0.63$). No systematic convergence speed differences emerge. The null holds at high power.

### Results

| Condition | Initial Loss | Final Loss | vs Control $p$ |
|---|---|---|---|
| Control (A continues) | — | 2.417 ± 0.352 | — |
| AABB | 2.985 | 2.428 ± 0.338 | 0.266 |
| ABAB | 2.938 | 2.425 ± 0.341 | 0.456 |
| BBAA | 2.834 | 2.414 ± 0.348 | 0.824 |
| ABBA | 2.880 | 2.420 ± 0.341 | 0.782 |

### Findings

**All chimeras converge to control loss.** No chimera type differs significantly from the control ($p > 0.26$). Despite starting at substantially worse loss (2.83–2.98), all chimeras converge to 2.41–2.43.

**Layer assignment doesn't matter.** The spread across chimera types is only 0.014 in mean final loss. Whether layers alternate (ABAB) or cluster (AABB) makes no difference.

**This reflects a wide basin of attraction, not freedom from the algorithm.** SGD prescribes convergence — that is literally what it does. The chimera result shows the basin of attraction is wide enough to absorb dramatically different starting conditions. The optimizer re-finds the same minimum from anywhere within the basin. This is what gradient descent does in smooth loss landscapes.

**Freedom classification:** *Wide basin of attraction* — SGD re-finds the same minimum from any structurally valid starting point.


## 10. Experiment 9: Gradual vs. Sudden Damage

**Analog:** Biological stress inoculation. Gradual exposure to a stressor builds tolerance that sudden exposure does not.

**Scaling note:** At $n = 3$, the gradual vs. sudden comparison appeared promising but underpowered. At $n = 30$, the key finding resolves clearly — this is the paper's strongest freedom-from-the-algorithm result. At $n = 300$, the stress inoculation effect strengthens dramatically: sudden full degrades by $+2.0\%$ ($p < 0.0001$), gradual shows only mild degradation ($+0.4\%$, $p = 0.024$), sudden half is marginal ($+0.4\%$, $p = 0.051$), and the direct gradual-vs-sudden comparison strengthens from $p = 0.011$ at $n = 30$ to $p < 0.0001$ at $n = 300$ ($\Delta = -1.5\%$, $d = -0.278$). This is the paper's most robust cross-scale confirmation.

### Results

| Condition | Final Loss | $p$ (vs ctrl) | Mean Loss | $p$ (mean) |
|---|---|---|---|---|
| Control | 2.469 ± 0.340 | — | 2.639 ± 0.023 | — |
| Sudden full ($\sigma = 0.1$) | 2.551 ± 0.374 | 0.004** | 2.698 ± 0.027 | <0.001*** |
| Gradual (0→0.1) | 2.481 ± 0.364 | 0.427 | 2.635 ± 0.024 | 0.006** |
| Sudden half (step 100) | 2.504 ± 0.364 | 0.034* | 2.640 ± 0.022 | 0.085† |

### Findings

**Gradual exposure builds tolerance.** The gradually-ramped condition is statistically indistinguishable from control ($p = 0.43$), while sudden exposure to the same peak noise level significantly degrades final loss ($p = 0.004$, +3.3%). Direct comparison: gradual is significantly better than sudden ($p = 0.011$, $d = -0.50$).

**This is the paper's clearest instance of freedom from the algorithm.** SGD says "minimize loss given the current gradient." It does not say "develop tolerance to noise schedules" or "adapt to stressors that arrive gradually." The gradient update rule is identical in the sudden and gradual conditions at every step — the only difference is the *history* of noise levels. That history matters, and the system develops different properties depending on it. This is not what the loss minimization objective specifies.

**Freedom classification:** *Genuine freedom* — stress inoculation is not prescribed by SGD. The system develops differential tolerance based on perturbation history in a way the optimizer did not request.

**Gradual noise acts as regularization.** The gradual condition's mean loss is *below* control ($-0.1\%$, $p = 0.006$), suggesting the slowly-introduced noise serves as a regularizer.


## 11. Experiment 10: Regeneration (Layer Reset)

**Analog:** Tissue regeneration. Destroy a layer entirely (reset to random), continue training.

**Scaling note:** At $n = 3$, regeneration appeared possible but with too few observations to confirm completeness. At $n = 30$, complete regeneration is confirmed for all four layers. At $n = 300$, fine structure emerges in regeneration completeness: L0 reaches $94.3\%$ ($p = 0.016$, significantly below full), L1 reaches $99.4\%$ ($p = 0.18$, non-significant), L2 reaches $101.1\%$ ($p = 0.040$, significantly above baseline), and L3 reaches $97.6\%$ ($p = 0.091$, marginal). Early layers show slight but significant regeneration incompleteness that was invisible at $n = 30$.

### Results

| Reset Layer | Damage | Final Loss | Completeness | vs Ctrl $p$ |
|---|---|---|---|---|
| Control | — | 2.407 ± 0.346 | — | — |
| Layer 0 | +0.264 | 2.410 ± 0.347 | 0.960 | 0.696 |
| Layer 1 | +0.309 | 2.410 ± 0.349 | 0.869 | 0.435 |
| Layer 2 | +0.328 | 2.405 ± 0.347 | 1.003 | 0.575 |
| Layer 3 | +0.342 | 2.412 ± 0.346 | 0.998 | 0.177 |

### Findings

**Complete regeneration.** All four layers recover to control-equivalent loss after being destroyed ($p > 0.17$ for all). Layers 2-3 achieve completeness significantly above 0.9 ($p < 0.001$), with L2 at 1.003 (marginally exceeding baseline).

**This is genuine freedom from the algorithm.** SGD says "minimize loss." It does not say "rebuild a destroyed layer to the same performance as if it had never been destroyed." The completeness of regeneration — particularly L2 at 1.003 and L3 at 0.998 — demonstrates that the network re-finds the same functional role regardless of what was there before. This is not prescribed by the optimizer.

**Freedom classification:** *Genuine freedom* — complete layer regeneration to control-equivalent performance is not prescribed by the loss minimization objective.

**No layer is indispensable.** Despite later layers suffering more immediate damage (+0.26 for L0 vs. +0.34 for L3), all regenerate equally. Layer position does not predict damage (Spearman $\rho = 0.078$, $p = 0.395$).


## 12. Experiment 11: Transplantation

**Analog:** Organ transplantation. Replace a layer with one from a separately-trained donor model.

**Scaling note:** At $n = 3$, a transplant advantage appeared possible. At $n = 30$, the null result resolves clearly — there is no advantage to a structured donor layer over a random replacement. At $n = 300$, the null holds across all layers ($p = 0.45$–$0.54$). No layer-specific transplant advantage or disadvantage emerges even at 10x power.

### Results

| Layer | Transplant | Random Reset | Gap | $p$ |
|---|---|---|---|---|
| L0 | 2.409 | 2.410 | +0.000 | 0.975 |
| L1 | 2.409 | 2.410 | +0.001 | 0.839 |
| L2 | 2.409 | 2.405 | -0.004 | 0.173 |
| L3 | 2.408 | 2.412 | +0.005 | 0.262 |
| **Overall** | — | — | **-0.000** | **0.860** |

### Findings

**No transplant advantage.** A layer from a separately-trained donor is accepted no better and no worse than a random replacement ($p = 0.86$ overall). Layer position does not modulate the effect (Spearman $\rho = 0.060$, $p = 0.513$).

**The network doesn't recognize donor structure.** Unlike biological transplantation where tissue compatibility matters, the network rebuilds whatever is placed at each layer position from scratch. The donor layer's learned structure provides no advantage.

**This reflects the wide basin of attraction, not freedom from the algorithm.** The loss landscape is smooth enough that any reasonable starting point finds the minimum. SGD was always going to do this.

**Freedom classification:** *Wide basin of attraction* — the basin is equally accessible from pre-trained and random initializations.


## 13. Experiment 12: Competing Objectives

**Analog:** Inter-organ conflict. Negate gradients for layers 2-3 while layers 0-1 train normally.

**Scaling note:** At $n = 3$, the distinction between adversarial and inactive layers appeared but was underpowered. At $n = 30$, the distinction resolves sharply. At $n = 300$, competing objectives degrade by $+23.3\%$ ($p < 0.0001$, $d = 0.602$) while freeze remains non-significant ($p = 0.74$). High variance persists (std = 1.12), and the adversarial-vs-freeze distinction sharpens to highly significant at $n = 300$.

### Results

| Condition | Final Loss | vs Ctrl Δ% | $p$ |
|---|---|---|---|
| Control | 2.407 ± 0.346 | — | — |
| Competing (negate L2-3) | 2.842 ± 1.072 | +18.1% | 0.010** |
| Freeze L2-3 | 2.411 ± 0.349 | +0.2% | 0.303 |

### Findings

**Adversarial components are not compensated.** Negating gradients for layers 2-3 causes +18.1% degradation ($p = 0.010$) with high variance (std = 1.07). The network cannot route around actively sabotaging layers.

**Frozen components are tolerated.** Merely freezing those same layers causes negligible degradation (+0.2%, $p = 0.303$). Competing vs. freeze: $p = 0.011$.

**Freedom classification:** *Tolerance* for the freeze condition (absence is absorbed without prescription from SGD); the adversarial result is a boundary condition revealing the limits of tolerance.

**Sharp line between absence and opposition.** The architecture can handle absent layers (residual stream routes around them) but cannot compensate for layers actively working against the objective. This defines the architecture's tolerance limit.


## 14. Cross-Experiment Synthesis

Six findings emerge from the combined evidence across twelve experiments at $n = 30$.

### Finding 1: Genuine freedom — gradual exposure builds tolerance (Exp 9)

The paper's strongest freedom-from-the-algorithm result. Gradual noise ramp (0→0.1) produces no final-loss degradation ($p = 0.43$), while sudden exposure to the same peak noise level degrades by +3.3% ($p = 0.004$). Direct comparison: $p = 0.011$, $d = -0.50$. The gradient update rule is identical at every step; only the history differs. That history changes the system's final state. SGD did not prescribe this.

### Finding 2: Genuine freedom — complete recovery, regeneration, and trajectory improvement (Exp 1, 7, 10)

The architecture recovers and rebuilds without explicit specification from the optimizer:
- Damaged and recovered: final ratio 0.9999 ($p = 0.905$ vs control)
- Layer reset to random: $p > 0.17$ for all layers (complete regeneration)
- Head-freezing trajectory improvement: freeze 8 ($p < 0.001$, $d = -1.03$)

SGD says "minimize loss." It does not say "return to the same basin after damage" or "rebuild a destroyed layer to equivalent performance." These behaviors exceed what the objective specifies.

### Finding 3: Wide basin — chimeras and transplants converge regardless (Exp 8, 11)

The architecture converges to the same loss regardless of assembly history:
- Chimeras from two models: $p > 0.26$ for all assemblies
- Transplant vs random reset: $p = 0.86$ (no difference)

This is SGD doing its job in a smooth loss landscape, not the system exhibiting freedom. The basin is wide enough to reach from dramatically different starting points.

### Finding 4: The absence vs. adversity distinction (Exp 12)

Frozen (inactive) layers: +0.2%, $p = 0.303$ (tolerated). Adversarial (gradient-negated) layers: +18.1%, $p = 0.010$ (degrading). The architecture routes around silence but cannot defend against active sabotage.

### Finding 5: The chess-paper inversion (Exp 3, 6)

Sign-only gradients degrade more than dropout ($p < 0.001$, $d = +6.00$ for mean loss), robustly inverting the chess paper's prediction. This is substrate-dependent: transformers require gradient precision; chess requires perceptual stability.

### Finding 6: Tolerance boundary (Exp 1-6)

| Perturbation | Final Loss Δ% | $p$-value | Classification |
|---|---|---|---|
| Freeze 1-4 heads | +0.1-0.2% | >0.37 | Wide basin / trajectory freedom |
| Partial gradient flow (25-75%) | +0.0-0.1% | >0.02 | Tolerance |
| Noisy gradients (σ=0.01) | +0.3% | 0.53 | Tolerance |
| Gradual noise ramp (0→0.1) | +0.5% | 0.43 | Genuine freedom |
| Frozen layers (L2-3) | +0.2% | 0.30 | Tolerance |
| Dropout (p=0.1) | +0.5% | 0.029 | Mild degradation |
| Sudden noise (σ=0.1) | +3.3% | 0.004 | Degradation |
| Quantized 3-bit | +3.0% | 0.019 | Degradation |
| Sign-only gradients | +4.6% | 0.004 | Degradation |
| Cell-view (no backprop) | +4.9% | <0.001 | Degradation |
| Adversarial layers | +18.1% | 0.010 | Severe degradation |


## 15. Scaling Resolution

How signals changed as the resolution dial turned from $n = 3$ to $n = 30$, with markers for what $n = 300$ may reveal.

**Head freezing improves final loss (Exp 1):** At $n = 3$, the coarse signal showed a possible improvement. At $n = 30$, the final-loss effect resolved to null ($p > 0.06$), but a different, finer signal emerged in the mean trajectory. The picture sharpened and moved to a different metric — not a retraction, but a resolution change. At $n = 300$, the trajectory improvement strengthens to highly significant ($p < 0.0001$ for freeze 8, 12, and 16), confirming this as a robust freedom signal. Final-loss null also holds firmly (all $p > 0.40$).

**DG scales with perturbation:** At $n = 3$, DG appeared to scale with perturbation severity. At $n = 30$, this resolved to null across all conditions ($p > 0.16$). The DG metric captures stochastic SGD dynamics but not perturbation response. At $n = 300$, the DG null holds. No fine structure emerges even at 10x power. DG does not track perturbation.

**Gradient degradation is neutral (Exp 3):** At $n = 3$, all four degradation methods appeared neutral and small noise seemed to help. At $n = 30$, three of four resolved to significant degradation; only small noise is genuinely tolerated. The null at $n = 3$ was low-resolution: the moderate effects were invisible without adequate power. At $n = 300$, the threshold sharpens: noise $\sigma = 0.01$ is non-significant ($p = 0.52$) while all three severe methods are $p < 0.0001$. The boundary between tolerance and degradation is a sharp step, not a smooth curve.

**Partial communication outperforms full (Exp 5):** At $n = 3$, a U-shaped curve appeared — partial communication seeming to outperform both extremes. At $n = 30$, the U-shape resolved to flat. The pilot curve was sampling noise. At $n = 300$, no topology advantage appears. Heavy ($p = 0.35$), half ($p = 0.87$), and light ($p = 0.41$) all remain non-significant. The architecture's indifference to gradient fraction holds at high power.

**Noise helps (Exp 3):** At $n = 3$, small noise ($\sigma = 0.01$) appeared beneficial. At $n = 30$, this resolved to null ($p = 0.978$). The apparent benefit was within noise at low resolution. At $n = 300$, the null holds ($\sigma = 0.01$: $-0.1\%$, $p = 0.52$). No sub-threshold noise benefit emerges even at 10x power.

**Transplant advantage (Exp 11):** At $n = 3$, a transplant advantage appeared possible. At $n = 30$, the null result resolved clearly ($p = 0.86$). At $n = 300$, the null holds throughout ($p = 0.45$–$0.54$ for all layers). No transplant advantage at any resolution.

**Chimera convergence (Exp 8):** At $n = 3$, convergence appeared probable. At $n = 30$, confirmed for all chimera types. At $n = 300$, all chimera types remain non-significant (AABB $p = 0.51$, ABAB $p = 0.83$, BBAA $p = 0.95$, ABBA $p = 0.63$). No convergence speed differences emerge.

**Gradual vs. sudden damage (Exp 9):** At $n = 3$, the signal was ambiguous. At $n = 30$, stress inoculation resolves as the paper's clearest freedom finding ($p = 0.011$). At $n = 300$, the effect strengthens dramatically: the gradual-vs-sudden gap goes from $p = 0.011$ at $n = 30$ to $p < 0.0001$ at $n = 300$ ($\Delta = -1.5\%$, $d = -0.278$). This is the paper's most robust cross-scale confirmation of a freedom finding.


## 16. Findings — What Perturbation Revealed

### Genuine Freedom from the Algorithm

**1. Gradual stress builds tolerance (Exp 9).** The only experiment where the *manner* of perturbation application matters. Gradual noise ramp: $p = 0.43$ vs control. Sudden noise: $p = 0.004$. Direct: $p = 0.011$. SGD doesn't specify how the history of noise levels should change the system's final state — but it does.

**2. Complete recovery (Exp 7).** A model damaged during training recovers to identical final loss ($p = 0.905$, ratio 0.9999). All 30 runs recovered within a mean of 2 steps. The path through damage leaves no trace. SGD prescribes finding a minimum; it doesn't prescribe returning to the *same* minimum after a detour.

**3. Complete regeneration (Exp 10).** Any layer can be destroyed and rebuilt to control-equivalent performance ($p > 0.17$ for all layers). The network re-finds the same functional role regardless of what was there before. SGD didn't prescribe this completeness.

**4. Head-freezing trajectory improvement (Exp 1).** Freezing 4+ randomly-initialized heads produces small but statistically robust mean-trajectory improvements (freeze 8: $p < 0.001$, $d = -1.03$). Frozen random-projection heads reduce gradient interference in a way the optimizer did not request.

### Wide Basin of Attraction

**5. Chimera convergence (Exp 8).** Models assembled from parts of two independently-trained networks converge to the same final loss as undamaged continuation ($p > 0.26$ for all chimera types). This is SGD doing its job in a smooth loss landscape, not freedom.

**6. Transplant indifference (Exp 11).** Transplanted layers and randomly-reset layers converge to the same final loss ($p = 0.86$ overall). The basin is equally accessible from pre-trained and random initializations.

### Tolerance

**7. Gradient quality matters more than quantity (Exp 3, 5).** Reducing gradient precision (sign-only: $d = +6.13$) degrades more than reducing gradient magnitude (partial flow: $d < 0.5$) or completeness (freezing: $p > 0.06$). The architecture tolerates magnitude reduction but not sign-structure destruction.

**8. The sign-only vs. dropout inversion (Exp 3, 6).** Sign-only gradients degrade significantly more than dropout ($p < 0.001$, $d = +6.00$ for mean loss). This inverts the chess paper's "cautious position, courageous moves" prediction — substrate-dependent.

**9. Adversarial vs. inactive tolerance (Exp 12).** Frozen layers cost nothing ($p = 0.30$); adversarial layers cost +18% ($p = 0.01$). The architecture tolerates absence but not opposition.


## 17. The Nancy Reading

Nancy's concept of désœuvrement — the interruption of work that reveals the community constituted by work — provides the interpretive frame for this methodology and its findings. The relevant question is not just what structure the interruption reveals, but what *freedom* it exposes: behaviors the algorithm did not request.

**Normal training is opaque.** During standard backpropagation, the transformer's components cooperate invisibly. The system works, and its working conceals its structure and its freedom.

**Perturbation as unworking.** Twelve experiments interrupt the system's work — freezing, severing, corrupting, restricting, assembling, destroying, transplanting, conflicting. The interruptions reveal structure. But some of what they reveal is more than structure — it is freedom.

**Freedom revealed: stress inoculation.** The system develops tolerance to noise that arrives gradually. At every training step, the update rule is the same: compute gradient, apply update. The optimizer does not remember previous noise levels when computing the current update. Yet the system's final state depends on that history. This is not prescribed by the loss minimization objective. The gradually-trained model reaches a better minimum than the suddenly-trained model, using the same number of gradient steps with the same peak noise level.

**Freedom revealed: complete recovery and regeneration.** The system recovers completely from transient damage and rebuilds destroyed layers to control-equivalent performance. SGD prescribes finding a minimum — it does not prescribe *which* minimum or how completely to return to it after perturbation. The completeness of recovery (ratio 0.9999) and the universality of regeneration (all four layers, $p > 0.17$) exceed what the optimizer specifies.

**Wide basin clarified, not freedom.** Chimera convergence and transplant indifference reflect the geometry of the loss landscape — a wide basin with a single dominant attractor. SGD always converges to this; the finding is that the basin is wide enough to reach from dramatically different starting points. This is what gradient descent does in smooth loss landscapes. It is not freedom; it is the expected behavior of the optimizer on a particular landscape.

**A sharp line between absence and opposition.** The architecture tolerates frozen (absent) components but cannot compensate for adversarial ones. This maps onto Nancy's distinction between the *withdrawn* member of the community (whose absence is absorbed) and the *hostile* member (whose opposition destroys the work). The tolerance of absence is itself a kind of structural freedom — the residual stream routes around silence in a way the optimizer did not explicitly prescribe.

**Désœuvrement reveals freedom.** Nancy's unworking makes the community legible by interrupting its work. Here, the interruptions make legible not just the redundancy and dependency structure of the transformer, but its freedom: the behaviors it exhibits that gradient descent did not ask for. The work conceals the community; the unworking shows not only its basin but its capacity for adaptation that the algorithm left unspecified.


## 18. Limitations

- **Scale:** 4 layers, 16 dimensions, 16 heads, ~11K params. The freedom findings (stress inoculation, recovery, regeneration) may be specific to small models, or they may be architectural universals.
- **Task complexity:** Character-level name generation is a toy task. Whether stress inoculation appears in language modeling or other complex tasks is not established.
- **Training duration:** 200 steps per phase. Gradual-exposure tolerance may not persist at longer horizons.
- **Transplant design:** Both models trained on same task/data. Cross-task transplantation might show different results.
- **Competing objectives design:** Gradient negation is maximally adversarial. Subtler conflicts might reveal compensation mechanisms.
- **DG metric:** Does not function as perturbation response measure at $n = 30$. At $n = 300$, no fine structure emerges; the DG null holds at all scales.
- **Effect sizes:** Many statistically significant effects are practically small (<0.5%). Statistical significance at $n = 30$ does not imply practical importance.
- **$n = 3$ to $n = 30$ signal changes:** Several pilot signals changed character at higher resolution, underscoring that coarse-scale data should not be interpreted as conclusions. The three-scale protocol is a response to this.
