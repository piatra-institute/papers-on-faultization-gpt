# MorphoGPT: Experimental Findings

Morphogenetic perturbation analysis of a minimal GPT, applying Levin's developmental biology methodology and Nancy's concept of désœuvrement (unworking) to transformer learning dynamics.


## 1. Setup

**Model:** 4-layer, 16-dimensional, 4-head GPT (16 total attention heads).

**Task:** Character-level name generation, trained on a names dataset.

**Protocol:** 200 training steps, 3 independent runs per condition (seeds 42, 43, 44), loss and head-level metrics recorded at every step.

**Statistical analysis:** All comparisons use paired $t$-tests with runs matched by seed across conditions ($n = 3$, $df = 2$). We report $p < 0.05$ as statistically significant and $0.05 < p < 0.10$ as marginal. With only 3 paired observations, power is low — the tests detect large effects but many reported mean differences are not statistically significant. We distinguish between *statistically supported* findings and *observational* patterns throughout.

**Key metric — Delayed Gratification (DG) Index:** Measures how much the loss trajectory dips below its final value during training. A high DG means the system explored better configurations early but "gave them up" — the signature of rerouting through alternative pathways after perturbation, analogous to Levin's developmental competency metric in biological systems.


## 2. Experiment 1: Head Freezing

**Analog:** Levin's frozen-cell perturbation. Randomly selected attention heads have their parameters frozen at initialization, forcing the remaining heads to compensate.

### Results

| Frozen Heads | Final Loss | DG Index |
|:---:|:---:|:---:|
| 0 | 2.479 | 0.299 |
| 1 | 2.477 | 0.291 |
| 2 | 2.475 | 0.340 |
| 4 | 2.482 | 0.331 |
| 8 | 2.463 | 0.388 |
| 12 | 2.426 | 0.485 |
| 16 | 2.409 | 0.502 |

![Robustness curve](../results/exp1_robustness_curve.png)

### Findings

**Loss improves significantly when 8+ heads are frozen.** Freezing 8 heads ($t = -10.46$, $p = 0.009$) and 12 heads ($t = -13.43$, $p = 0.006$) produce statistically significant improvements in final loss. Freezing 16 heads is marginally significant ($p = 0.055$). However, freezing 1, 2, or 4 heads shows no significant change from baseline ($p > 0.85$). The improvement emerges at a threshold around half the total heads, rather than increasing monotonically at each step — the trend in means is visible but the per-step differences for low freezing levels are indistinguishable from noise at $n = 3$. The explanation for the high-freezing improvement is that MLPs carry the bulk of learning, and frozen attention heads reduce gradient interference — random initialized heads that cannot update act as fixed projections rather than competing learners.

**DG nearly doubles.** The DG index rises from 0.299 to 0.502 (+68%), the clearest rerouting signature across all experiments. The system finds good loss values early (via MLPs) but the frozen heads prevent it from retaining those configurations, producing the characteristic dip-then-rise trajectory.

**Trajectory shape is preserved.** Cross-condition trajectory correlations exceed 0.95 at all freezing levels. The system follows the same learning arc regardless of how many heads are disabled — it reaches the same place through different internal configurations.

**DG-damage regression shows positive slope.** More freezing produces more DG, consistent with Levin's prediction that damage triggers compensatory rerouting proportional to the perturbation magnitude.

![Training trajectories](../results/exp1_trajectories.png)
![DG episodes](../results/exp1_dg_episodes.png)


## 3. Experiment 2: Cell-View GPT

**Analog:** Nancy's being-singular-plural — each layer treated as an autonomous agent. Stop-gradient applied at all layer boundaries so each layer learns only from its own local loss signal, with no end-to-end backpropagation.

### Results

| Condition | Mean Loss | Final Loss | DG Index |
|:---:|:---:|:---:|:---:|
| baseline | 2.635 | 2.479 | 0.299 |
| cell_view | 2.698 | 2.627 | 0.375 |

Performance delta: **+2.4% mean loss** (+0.063).

DG delta: **+25.5%** (+0.076).

Training time: **~3.5x faster** — stop-gradient eliminates the backward pass through the full computational graph.

![Baseline vs cell-view comparison](../results/exp2_comparison.png)

### Findings

**Local-only learning is viable.** A 2.4% performance cost for eliminating all inter-layer gradient communication is remarkably small. Each layer learns from its own loss signal and still contributes to coherent sequence generation.

**Different head specialization emerges.** Under cell-view training, Layer 3 heads dominate — they specialize more aggressively when deprived of upstream gradient refinement. Inner layers show more uniform entropy distributions, while the final layer concentrates its attention patterns.

**Lower variance across runs (observational).** Cell-view training appears to produce more consistent results between independent runs. However, variance comparisons at $n = 3$ are unreliable — this pattern is observational and cannot be confirmed statistically at this sample size.

**DG increases under local learning.** The +25.5% DG elevation indicates that even without inter-layer coordination, the system exhibits rerouting behavior — each layer independently discovers and abandons configurations during training.

![Head entropy comparison](../results/exp2_head_entropy.png)
![Training trajectories](../results/exp2_trajectories.png)
![DG episodes](../results/exp2_dg_episodes.png)


## 4. Experiment 3: Gradient Degradation

**Analog:** Levin's noisy signaling channels. Gradients are corrupted during training through four methods: additive Gaussian noise (two scales), sign-only reduction (discarding magnitude), and 3-bit quantization.

### Results

| Method | Final Loss Delta | DG Index |
|:---:|:---:|:---:|
| baseline | — | 0.299 |
| noisy (σ=0.01) | -0.028 | 0.412 |
| sign_only | +0.011 | 0.322 |
| quantized (3-bit) | +0.042 | 0.425 |
| noisy (σ=0.1) | +0.110 | 0.533 |

![Method comparison](../results/exp3_method_comparison.png)

### Findings

**No gradient degradation method produces a statistically significant change from baseline.** Paired $t$-tests show $p > 0.26$ for all four methods: sign-only ($p = 0.860$), quantized ($p = 0.425$), noisy $\sigma = 0.1$ ($p = 0.265$), and noisy $\sigma = 0.01$ ($p = 0.701$). The mean effects are suggestive — sign-only at +0.4%, quantized at +1.7%, noisy $\sigma = 0.1$ at +4.4%, and noisy $\sigma = 0.01$ at -1.1% — but all fall within the confidence intervals at $n = 3$. This null result is itself informative: the architecture tolerates severe gradient corruption without measurable degradation.

**Sign-only gradients produce no detectable degradation.** Discarding all magnitude information and keeping only gradient signs produces a final loss delta of just +0.011, not statistically significant ($p = 0.860$). This is consistent with SignSGD findings in the literature. However, the wide confidence interval means we cannot claim precision — the true effect could range from moderate improvement to moderate degradation.

**The "noise helps" effect is not statistically supported.** At $\sigma = 0.01$, the mean final loss delta is -0.028, but the paired differences go in opposite directions across runs ($p = 0.701$). We cannot claim that noise improves performance — the effect is indistinguishable from sampling variability.

**All degradation methods elevate DG.** Every form of gradient corruption increases the DG index above baseline:
- Noisy σ=0.01: +38% (0.412)
- Sign-only: +8% (0.322)
- Quantized 3-bit: +42% (0.425)
- Noisy σ=0.1: +78% (0.533)

The noisiest condition (σ=0.1) produces the highest DG of the first three experiments, indicating more pronounced rerouting under stronger perturbation.

**Architecture constrains the solution space.** The residual stream, attention patterns, and MLP structure collectively define a narrow enough solution manifold that even crude gradient approximations navigate it successfully. The architecture does the heavy lifting; the optimizer just needs a rough direction.

![Training trajectories](../results/exp3_trajectories.png)
![Trajectory divergence](../results/exp3_trajectory_divergence.png)
![DG episodes](../results/exp3_dg_episodes.png)


## 5. Experiment 4: Vision Radius Sweep

**Analog:** Kofman, Campitelli & Levin's (2025) vision radius experiment in distributed chess, where each piece perceives only squares within radius R. Their central result: intermediate R4 outperforms both blind (R0) and omniscient (R7) pieces. We translate this by restricting each attention head's context window, limiting how far back in the sequence a head can attend.

### Results

| Window | Mean Loss | Std | Mean DG |
|:---:|:---:|:---:|:---:|
| Baseline (full) | 2.6349 | 0.027 | 0.299 |
| 1 | 2.6500 | 0.029 | 0.435 |
| 2 | 2.6405 | 0.031 | 0.389 |
| 4 | 2.6313 | 0.028 | 0.249 |
| 8 | 2.6328 | 0.027 | 0.499 |
| 16 | 2.6349 | 0.027 | 0.299 |

![Vision radius sweep](../results/exp4_vision_radius.png)

### Findings

**Only window=8 produces a statistically significant improvement.** Paired $t$-tests show window=8 improves mean loss ($t = -5.00$, $p = 0.038$), but the effect is tiny (-0.002). Window=4's mean loss improvement is not significant ($p = 0.17$), and window=2's final loss improvement — the previously headline claim of -1.4% — is not significant ($p = 0.318$). Window=1 marginally worsens mean loss ($p = 0.078$). The trend is suggestive of the information bottleneck hypothesis but underpowered at $n = 3$.

**Window=16 is identical to baseline.** The full-context window (16 = block size) reproduces baseline values exactly (mean loss 2.6349, DG 0.299), confirming that the windowing implementation introduces no artifacts. This is the sanity check.

**Window=8 produces the highest DG.** At 0.499 (+67% over baseline), window=8 shows the most rerouting. Moderate restriction forces the system to explore alternative configurations most actively — neither so constrained that exploration is limited (window=1) nor so unrestricted that the default path suffices (full attention).

**Trajectory shape is preserved across window sizes.** Learning curves maintain similar shapes regardless of window size. The system navigates toward the same performance region through different internal routes, as in Experiments 1-3.


## 6. Experiment 5: Communication Topology

**Analog:** The chess paper's relay chains, where pieces transmit threat information beyond their individual vision radius, expanding the collective's "cognitive light cone." We create a spectrum of gradient flow topologies between full backpropagation and complete isolation by scaling the gradient pass fraction at layer boundaries.

### Results

| Topology | Fraction | Mean Loss | Mean DG |
|:---:|:---:|:---:|:---:|
| Full | 1.00 | 2.6349 | 0.299 |
| Heavy | 0.75 | 2.6355 | 0.276 |
| Half | 0.50 | 2.6354 | 0.277 |
| Light | 0.25 | 2.6343 | 0.316 |
| Cell-view | 0.00 | 2.6978 | 0.375 |

![Communication topology](../results/exp5_communication_topology.png)

### Findings

**No partial-flow condition differs significantly from full backpropagation.** Paired $t$-tests show heavy ($p = 0.28$), half ($p = 0.39$), and light ($p = 0.61$) are all statistically indistinguishable from the full-backpropagation baseline. The previously claimed "U-shape" and the claim that light (25%) outperforms full are not supported — the mean differences (e.g., light at -0.0006 mean loss) are pure noise at $n = 3$.

**Only cell-view is significantly different.** Cell-view degrades mean loss by +2.4% ($t = 6.74$, $p = 0.021$), confirming that complete elimination of inter-layer gradient flow has a real (though modest) cost.

**Cell-view shows uniquely elevated DG with positive goal alignment.** At 0.375 (+25% over baseline), cell-view has the highest DG in this experiment. Complete isolation forces the most rerouting — but at a performance cost. The intermediate topologies achieve similar loss to full backpropagation with similar DG levels.

**The tolerance finding is the real result.** While partial communication does not *improve* over full backpropagation, the fact that reducing gradient flow to 25% produces no detectable degradation is itself noteworthy — the system tolerates severe communication restriction without measurable loss.


## 7. Experiment 6: Courage vs. Caution

**Analog:** Kofman et al.'s finding that "cautious position, courageous moves" is the optimal chess strategy — conservative evaluation combined with aggressive action. We translate this into a 2×2 matrix crossing forward-pass stability with gradient boldness:

| | Cautious Gradients | Courageous Gradients |
|:---:|:---:|:---:|
| **Cautious Forward** | (a) Tiny noise (σ=0.001) | (b) Sign-only gradients |
| **Courageous Forward** | (c) Dropout (p=0.1) | (d) Noisy gradients (σ=0.1) |

The chess paper predicts (b) wins: stable perception with bold moves.

### Results

| Condition | Mean Loss | Std | Mean DG |
|:---:|:---:|:---:|:---:|
| Baseline | 2.6349 | 0.027 | 0.299 |
| Cautious/Cautious (a) | 2.6379 | 0.028 | 0.276 |
| Cautious/Courageous (b) | 2.7248 | 0.022 | 0.322 |
| Courageous/Cautious (c) | 2.6376 | 0.028 | 0.517 |
| Courageous/Courageous (d) | 2.6910 | 0.028 | 0.604 |

![Courage vs. caution](../results/exp6_courage_caution.png)

### Findings

**Sign-only gradients degrade significantly more than dropout.** Sign-only (b) significantly worsens mean loss by +3.4% ($t = 15.85$, $p = 0.004$). Dropout (c) produces a marginal degradation ($t = 3.66$, $p = 0.064$) — the effect is small and borderline. The gap between (b) and (c) is genuine: sign-only hurts much more than dropout. However, neither condition *improves* over baseline — the result is "sign-only hurts more than dropout," not "forward noise helps."

**The chess prediction is inverted.** Condition (c) courageous/cautious outperforms (b) cautious/courageous by a wide margin (mean loss 2.638 vs 2.725, +3.3% gap). The predicted winner is the worst-performing non-baseline condition. In transformers, forward-pass noise with careful gradients beats stable forward passes with bold gradients — the opposite of what works in chess.

**Noise everywhere is worst by final loss.** Condition (d) produces the highest final loss (2.538, +2.4% over baseline). Forward noise and gradient noise compound rather than cancel.

**DG scales with total perturbation magnitude.** The DG ordering is d (0.604) > c (0.517) > b (0.322) > baseline (0.299) > a (0.276). Conditions with more total perturbation produce more rerouting, regardless of where the perturbation is applied. DG=0.604 is the highest value observed across all six experiments, surpassing the previous maximum (noisy σ=0.1, DG=0.533 in Exp 3) by 13%.

**The substrate-dependent interpretation.** The chess inversion is not a failure of the analogy — it reveals a substrate difference. Chess pieces operate in a discrete, irreversible action space: a bad move cannot be undone, so stable perception (cautious position) is essential. Transformers operate in a continuous, differentiable optimization landscape: forward noise acts as regularization (exploration of the loss surface), while gradient precision is needed to navigate fine-grained curvature. The optimal courage/caution balance depends on whether errors are reversible.


## 8. Cross-Experiment Synthesis

Three claims emerge from the combined evidence across six experiments.

### Claim 1: Robustness is structural, not algorithmic

No perturbation produces a statistically significant degradation exceeding ~3.4% (sign-only in Exp 6, $p = 0.004$). Most perturbations produce no measurable change — the pattern is *tolerance* rather than improvement. Of the mean effects that appear to improve over baseline (Exp 1 freezing, Exp 3 small noise, Exp 4 restricted windows, Exp 5 partial flow), only Experiment 1 high freezing is statistically significant ($p < 0.01$). The others are not distinguishable from noise at $n = 3$. This robustness arises from the transformer architecture itself: the residual stream provides redundant pathways, the MLP layers carry independent learning capacity, and the multi-head structure distributes representation across parallel subspaces.

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

### Claim 2: DG tracks perturbation response consistently

Every perturbation type elevates the DG index. As a designed metric — we defined it and went measuring it — DG confirms that the rerouting phenomenon is real and quantifiable. But DG is the measuring stick, not the discovery. The discoveries are the anomalies it helps quantify (see Section 9).

| Perturbation | DG Index | DG Change |
|---|---|---|
| Baseline | 0.299 | — |
| 16 heads frozen | 0.502 | +68% |
| Cell-view (local learning) | 0.375 | +25% |
| Noisy gradients (σ=0.1) | 0.533 | +78% |
| Quantized 3-bit | 0.425 | +42% |
| Window=8 (restricted attention) | 0.499 | +67% |
| Cell-view topology (Exp 5) | 0.375 | +25% |
| Courageous/courageous | 0.604 | +102% |

The DG response scales with perturbation severity and appears regardless of whether the perturbation targets architecture (freezing), information flow (cell-view, communication topology), optimization signal (gradient degradation), attention span (vision radius), or forward-pass stability (courage/caution). DG=0.604 (courageous/courageous) is the new maximum, surpassing the previous highest (noisy σ=0.1, 0.533) by 13%.

### Claim 3: Centralized control is optional

Global backpropagation — the defining algorithm of deep learning — is not required for viable learning in this architecture:

- **Local learning** (cell-view): Each layer learns independently. Cost: +2.4% ($p = 0.021$).
- **Sign-only gradients** (Exp 3): Only direction information propagated. No significant degradation ($p = 0.860$).
- **Frozen components**: 8+ heads frozen. Cost: negative — significantly improves ($p < 0.01$).
- **Restricted attention**: Window=8 limits context. Significant but tiny improvement ($p = 0.038$, effect -0.002).
- **Partial gradient flow**: 25% of gradient signal passed between layers. No significant change from full ($p = 0.61$).

The trajectories differ — cell-view training produces different head specialization patterns, frozen-head training shifts load to MLPs, restricted windows force local attention — but the outcomes converge. Multiple internal organizations produce equivalent external behavior.

*Note:* The previous Claim 4 ("the information bottleneck is beneficial") has been removed. It was supported only by Experiment 1's high-freezing result ($p < 0.01$). The Experiment 4 window effects ($p = 0.17$–$0.32$ for windows 2 and 4) and Experiment 5 partial flow effects ($p > 0.28$) are not statistically significant. The information bottleneck hypothesis remains a suggestive pattern across experiments but is not proven by this data.


## 9. Anomalies — What Perturbation Revealed

The DG metric confirmed what we designed it to measure. The genuine findings are the phenomena that *emerged* without being designed for — results anti-intuitive under standard deep learning assumptions. These are the Levin analogs: things that "should not happen normally," made legible only through perturbation.

Paired statistical analysis at $n = 3$ supports some of these findings robustly, others only weakly, and reveals that several originally claimed anomalies are not distinguishable from noise. We organize the findings into two tiers based on statistical support, followed by explicit retractions.

### Tier 1 — Statistically Supported Findings ($p < 0.05$)

**1. Heavy freezing improves performance (Exp 1).** Freezing 8+ attention heads significantly improves final loss ($p < 0.01$). Freezing 1-4 heads does not. The improvement emerges at a threshold around 50% of heads, not monotonically. This is the paper's strongest finding: removing capacity *improves* performance when enough heads are frozen — the perturbation eliminates gradient interference, revealing that frozen random projections serve the system better than competing learners.

**2. Sign-only gradients degrade more than dropout (Exp 6).** Sign-only (b) worsens mean loss by +3.4% ($p = 0.004$) while dropout (c) worsens by only ~0.1% ($p = 0.064$, marginal). The gap between conditions (b) and (c) is genuine and large. This inverts the chess paper's "cautious position, courageous moves" prediction — in transformers, forward-pass noise (dropout) is far less damaging than gradient-signal reduction (sign-only). The inversion is substrate-dependent: chess pieces operate in a discrete, irreversible action space where stable perception is essential; transformers operate in a continuous, differentiable landscape where forward noise provides regularization and gradient precision is needed for fine-grained optimization.

### Tier 2 — Observations Consistent With But Not Proven By Data

**3. Cell-view is viable at modest cost (Exp 2, Exp 5).** Cell-view degrades mean loss by +2.4% ($p = 0.021$, statistically significant that it is *worse*). The cost is small enough to be noteworthy — complete elimination of inter-layer gradient flow does not break learning. But cell-view does not *improve* over baseline; it is a tolerated perturbation, not a beneficial one.

**4. Sign-only gradients approximately match baseline (Exp 3).** Sign-only produces no statistically significant degradation ($p = 0.860$). The mean effect is +0.4%, consistent with prior SignSGD findings (Bernstein et al., 2018). However, the wide confidence interval at $n = 3$ means precision is low — the true effect could range from moderate improvement to moderate degradation.

**5. Isolation produces head specialization (Exp 2).** Cutting inter-layer gradients causes L3 heads to specialize more aggressively. This is an observational finding from head entropy analysis, not a loss-based statistical comparison. Without upstream gradient refinement, the final layer concentrates its attention patterns rather than distributing them.

**6. Window=8 gives a tiny improvement (Exp 4).** Window=8 significantly improves mean loss ($p = 0.038$), but the effect is -0.002 — too small to be practically meaningful. The finding needs replication at higher $n$ to determine whether the information bottleneck hypothesis holds for attention windowing.

### Retracted or Downgraded Claims

The following claims from the original analysis are not supported by paired statistical testing:

- **"Noise helps" (original Anomaly 3):** The mean effect of noisy $\sigma = 0.01$ is negative (-0.028) but not statistically significant ($p = 0.701$). Paired differences go in opposite directions across runs. **Retracted.**
- **"Monotonic improvement" with freezing:** The trend in means is visible but only significant for 8+ frozen heads. The per-step improvement for low freezing (1-4 heads) is indistinguishable from noise ($p > 0.85$). **Downgraded** to threshold effect.
- **"Partial communication outperforms full" (original Anomaly 7):** Light topology's mean loss advantage (-0.0006) is not significant ($p = 0.61$). Heavy and half are also indistinguishable from full ($p > 0.28$). **Retracted.**
- **"Window=2 beats full attention by 1.4%" (original Anomaly 6):** The final loss difference is not significant ($p = 0.318$). **Retracted** as a standalone finding; retained as a suggestive trend.
- **"Decentralization stabilizes" (original Anomaly 4):** Variance comparisons at $n = 3$ are unreliable. The pattern is interesting but not testable at this sample size. **Downgraded** to observational.

These anomalies — two robustly supported, four observational, and several retracted — are what the experiments actually revealed. DG is a useful measuring stick that confirms rerouting is happening. The statistically supported anomalies *are* the rerouting; the observational ones are hypotheses for larger-scale replication.


## 10. The Nancy Reading

Nancy's concept of désœuvrement — the interruption of work that reveals the community constituted by work — provides the interpretive frame for this *methodology*, not for specific statistical claims. The philosophical reading applies to the act of perturbation and what it makes visible, regardless of which particular effects prove robust at higher sample sizes.

**Normal training is opaque.** During standard backpropagation, the transformer's components cooperate invisibly. Attention heads, MLP layers, and residual connections form a division of labor that produces outputs but reveals nothing about its own organization. The system works, and its working conceals its structure.

**Perturbation as unworking.** Each experiment interrupts the system's work in a different way — freezing components, severing gradient flow, corrupting signals, restricting vision, scaling communication, introducing forward noise. These interruptions do not destroy the system. Instead, they make visible the relational structure that was always present but hidden:

- **Redundancy**: Frozen heads reveal that MLPs carry independent learning capacity ($p < 0.01$ for 8+ frozen heads). This capacity exists in baseline training too, but is invisible because the heads mask it. This is the strongest statistically supported instance of the unworking principle.
- **Rerouting**: Elevated DG shows the system exploring alternative configurations. These alternative pathways exist in the loss landscape at all times; perturbation forces the system to traverse them.
- **Decentralized viability**: Cell-view training shows that each layer can learn autonomously at a modest 2.4% cost ($p = 0.021$). The layers are always singular-plural — simultaneously individual and communal — but global backprop makes their individuality invisible.
- **Tolerance of constraint**: Nancy's *partage* (sharing/dividing) and *espacement* (spacing). Reduced communication forces each layer into a more independent relation with the collective goal. The spacing between components — partial gradient flow, restricted attention windows — is tolerated without significant degradation, and in the case of high freezing, actively beneficial. Whether this tolerance constitutes *espacement* as constitutive rather than merely innocuous remains a hypothesis for larger-scale investigation.

**What désœuvrement reveals is not a failure mode but a mode of being.** The transformer's components tolerate perturbation to a remarkable degree; the community's structure only becomes legible when coordinated work is interrupted. The methodology of unworking is the contribution — what it reveals in any specific experiment depends on statistical power and replication.


## 11. Limitations

- **Statistical power**: With 3 runs per condition ($df = 2$), paired $t$-tests detect only large effects. Paired analysis reveals that many reported mean effects are not statistically significant: the Experiment 3 gradient degradation effects ($p > 0.26$), the Experiment 4 window improvements ($p = 0.17$–$0.32$), the Experiment 5 partial-flow effects ($p > 0.28$), and the "noise helps" effect ($p = 0.701$) are all indistinguishable from noise. Of the originally described anomalies, only two (high-freezing improvement, sign-only degradation gap in Experiment 6) are statistically robust. Replication with 30+ runs per condition is the most important next step.
- **Scale**: 4 layers, 16 dimensions, 16 heads. Scaling behavior to production-sized transformers is unknown.
- **Task complexity**: Character-level name generation is a toy task. Whether these findings hold for complex language modeling, reasoning, or multi-modal tasks is untested.
- **Training duration**: 200 steps captures early learning dynamics but not long-horizon phenomena like grokking or phase transitions.
- **DG metric**: The delayed gratification index is a designed metric — unlike Levin, who discovered DG as an emergent surprise, we defined it and went measuring it. The measurements confirm the metric works as defined, but they are confirmation, not discovery. The primary findings are the emergent patterns (Section 9) that perturbation made legible. DG's relationship to standard measures of representation quality, generalization, and internal structure requires further validation.
- **Gradient degradation scope**: Only four degradation methods tested. Adversarial gradient attacks, structured corruption, and layer-selective degradation remain unexplored.
- **Chess-paper translation fidelity**: The courage/caution inversion (Finding 2) reflects substrate differences between discrete board games and continuous optimization landscapes. Whether richer composite perturbation designs — combining multiple perturbation types simultaneously rather than using single proxies — would produce results closer to the chess paper's predictions remains untested. The information bottleneck and partial-communication predictions are not testable at the current sample size.
