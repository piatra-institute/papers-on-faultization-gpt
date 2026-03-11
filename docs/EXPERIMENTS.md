# MorphoGPT: Experiment Summary

Twelve morphogenetic perturbation experiments on a minimal GPT (4-layer, 16-dim, 4-head, ~11K params). All results at n=300 (300 independent runs per condition, paired by seed).


## Experiment 1: Head Freezing

**Description/Assumptions:** Randomly selected attention heads have their parameters frozen at initialization, forcing remaining heads to compensate. Analogous to Levin's frozen-cell perturbation. Tests whether the system depends on all heads being trainable, or whether frozen random-projection heads can serve as useful fixed features.

**Findings:**
- Final loss is unaffected by freezing any number of heads (all p > 0.40, Spearman ρ = 0.0023, p = 0.92) — wide basin of attraction absorbs the perturbation
- Mean trajectory loss improves significantly when 4+ heads are frozen: freeze 8 (Δ = −0.12%, p < 0.0001), freeze 12 (Δ = −0.17%, p < 0.0001), freeze 16 (Δ = −0.19%, p < 0.0001)
- Frozen random-projection heads reduce gradient interference during training — a benefit SGD did not prescribe
- Classification: **genuine freedom** (trajectory improvement) + **wide basin** (final-loss indifference)


## Experiment 2: Cell-View GPT (Stop-Gradient Isolation)

**Description/Assumptions:** Stop-gradient applied at all layer boundaries so each layer learns only from its own local loss signal, with no end-to-end backpropagation. Analogous to Nancy's being-singular-plural — each layer treated as an autonomous agent. Tests whether layers can learn independently without inter-layer gradient communication.

**Findings:**
- Cell-view degrades mean loss by +2.9% (t = 8.307, p < 0.0001, d = 0.480) and final loss by +4.9% (p < 0.001, d = +1.16)
- The system still learns despite zero inter-layer gradient communication — degradation is bounded, not catastrophic
- DG index shows no significant change (p = 0.34), confirming DG does not track perturbation response
- Classification: **tolerance** — the system absorbs removal of inter-layer gradient flow at a bounded cost


## Experiment 3: Gradient Degradation

**Description/Assumptions:** Gradients are corrupted during training through four methods: additive Gaussian noise (σ=0.01 and σ=0.1), sign-only reduction (discarding magnitude), and 3-bit quantization. Analogous to Levin's noisy signaling channels. Tests how much gradient information the optimizer actually needs.

**Findings:**
- Sharp tolerance threshold: noise σ=0.01 is non-significant (−0.1%, p = 0.52), while all three severe methods are p < 0.0001
- Noise σ=0.1 degrades by +2.3%, sign-only by +4.9%, quantized 3-bit by +3.4% (all p < 0.0001)
- The boundary between tolerance and degradation is a sharp step between σ=0.01 and σ=0.1, not a smooth curve
- Even worst-case (sign-only) remains within 5% of baseline — architecture constrains the solution space
- Classification: **tolerance** up to σ=0.01; **degradation** above


## Experiment 4: Vision Radius Sweep

**Description/Assumptions:** Each attention head's context window is restricted to a fixed radius, analogous to the distributed chess vision-radius experiment (Kofman, Campitelli & Levin, 2025). Tests whether limiting how far each head can attend affects learning, and whether an intermediate window outperforms full context.

**Findings:**
- Fine structure emerges at n=300: window 1 significantly worsens loss (+0.4%, p = 0.0009), window 8 significantly improves it (−0.1%, p = 0.022)
- Windows 2, 4, and 16 remain non-significant
- A subtle monotonic gradient from harm at smallest window to benefit at intermediate window
- Window 16 (= full context) reproduces baseline exactly, confirming no implementation artifacts
- The chess paper's information-bottleneck hypothesis is not supported at meaningful effect sizes
- Classification: **tolerance** — attention restriction at all tested scales is absorbed without meaningful final-loss change


## Experiment 5: Communication Topology

**Description/Assumptions:** A spectrum of gradient flow topologies between full backpropagation and complete isolation, created by applying partial stop-gradient at layer boundaries. Analogous to the chess paper's relay chains. Tests whether the system needs full gradient flow or can operate on partial signals.

**Findings:**
- The system is genuinely indifferent to gradient fraction above zero: heavy/75% (p = 0.35), half/50% (p = 0.87), light/25% (p = 0.41) — all non-significant
- Only complete isolation (cell-view, 0%) hurts: +4.9% degradation (p < 0.001)
- No partial-flow condition outperforms full backpropagation — the U-shape seen at n=3 was sampling noise
- Classification: **tolerance** — substantial gradient flow reduction is absorbed; only total removal crosses the degradation threshold


## Experiment 6: Courage vs. Caution

**Description/Assumptions:** A 2×2 matrix crossing forward-pass perturbation (cautious=tiny noise vs. courageous=dropout) with gradient perturbation (cautious=sign-only vs. courageous=noisy σ=0.1). Tests the chess paper's prediction that "cautious position, courageous moves" is optimal — translated as stable forward pass with aggressive gradients.

**Findings:**
- The chess prediction is inverted: courageous/cautious (dropout, p = 0.052) significantly outperforms cautious/courageous (sign-only, +4.9%, p < 0.0001)
- The sign-only vs. dropout inversion gap is +4.7% (p < 0.0001), robustly confirmed at n=300
- Cautious/cautious (tiny noise) is non-significant (p = 0.51); courageous/courageous degrades by +2.3% (p < 0.0001)
- The inversion is substrate-dependent: transformers need gradient precision; chess needs perceptual stability
- Classification: **substrate-dependent inversion** — not a freedom-from-the-algorithm finding


## Experiment 7: Recovery After Damage

**Description/Assumptions:** Train normally, freeze 8 heads (damage phase), then unfreeze and continue training (recovery phase). Analogous to Levin's regeneration paradigm. Tests whether transient damage leaves lasting traces or whether the model recovers completely.

**Findings:**
- Complete recovery: final loss ratio recovery/control = 0.9997 ± 0.0086 (p = 0.64 vs control)
- All 300 runs recovered; mean recovery time = 1.4 ± 1.7 steps after damage removal
- No overshoot: mean overshoot = −0.0007 ± 0.0018 — the Levin signature (damaged organisms exceeding baseline) is absent
- 100 steps of training with 8 frozen heads left no trace whatsoever
- Classification: **genuine freedom** — path-independent recovery to identical final loss is not prescribed by the loss minimization objective


## Experiment 8: Chimera Assembly

**Description/Assumptions:** Two models trained independently; layers from each are combined into a Frankenstein model (AABB, ABAB, BBAA, ABBA configurations), then training continues. Tests whether a model assembled from incompatible parts can converge normally.

**Findings:**
- All chimera types converge to control loss: AABB (p = 0.51), ABAB (p = 0.83), BBAA (p = 0.95), ABBA (p = 0.63)
- Despite starting at substantially worse loss (2.83–2.98), all chimeras converge to 2.41–2.43
- Layer assignment doesn't matter — whether layers alternate or cluster makes no difference
- No systematic convergence speed differences emerge even at n=300
- Classification: **wide basin of attraction** — SGD re-finds the same minimum from any structurally valid starting point


## Experiment 9: Gradual vs. Sudden Damage (Stress Inoculation)

**Description/Assumptions:** Compare gradual noise ramp (0→σ=0.1 over training) against sudden full noise (σ=0.1 from step 1) and sudden half (σ=0.1 from step 100). Both gradual and sudden reach the same peak noise level. Tests whether the history of perturbation exposure matters — whether gradual exposure builds tolerance that sudden exposure does not.

**Findings:**
- Gradual shows only mild degradation (+0.4%, p = 0.024); sudden full degrades by +2.0% (p < 0.0001)
- Direct gradual-vs-sudden comparison: Δ = −1.5%, p < 0.0001, d = −0.278
- The effect strengthened from p = 0.011 at n=30 to p < 0.0001 at n=300 — the paper's most robust cross-scale confirmation
- Gradual noise acts as regularization: mean loss is below control (−0.1%, p = 0.006)
- The gradient update rule is identical at every step; only the history of noise levels differs — yet the system's final state depends on that history
- Classification: **genuine freedom** — stress inoculation is not prescribed by SGD; this is the paper's strongest freedom-from-the-algorithm result


## Experiment 10: Regeneration (Layer Reset)

**Description/Assumptions:** Destroy a layer entirely (reset to random weights at mid-training), then continue training. Tests whether the network can rebuild a completely destroyed layer to its original functional role.

**Findings:**
- Fine structure emerges at n=300: L0 reaches 94.3% recovery (p = 0.016, significantly below full), L1 reaches 99.4% (p = 0.18, non-significant), L2 reaches 101.1% (p = 0.040, significantly above baseline), L3 reaches 97.6% (p = 0.091, marginal)
- Early layers show slight but significant regeneration incompleteness invisible at n=30
- No layer is indispensable despite later layers suffering more immediate damage (+0.26 for L0 vs. +0.34 for L3)
- Classification: **genuine freedom** — complete layer regeneration to control-equivalent performance is not prescribed by the loss minimization objective


## Experiment 11: Transplantation (Foreign Layer Integration)

**Description/Assumptions:** Replace a layer with one from a separately-trained donor model, then continue training. Compare against random reset (Experiment 10). Tests whether a pre-trained donor layer provides any advantage over a random replacement.

**Findings:**
- No transplant advantage: overall p = 0.860, with per-layer p values ranging from 0.45 to 0.54
- The network doesn't recognize donor structure — a donor layer's learned weights provide no advantage over random initialization
- Unlike biological transplantation where tissue compatibility matters, the network rebuilds whatever is placed at each position from scratch
- Classification: **wide basin of attraction** — the basin is equally accessible from pre-trained and random initializations


## Experiment 12: Competing Objectives (Conflicting Gradients)

**Description/Assumptions:** Negate gradients for layers 2-3 while layers 0-1 train normally, creating actively adversarial components. Compare against simply freezing those same layers. Tests whether the architecture can compensate for components actively working against the training objective.

**Findings:**
- Competing objectives degrade by +23.3% (p < 0.0001, d = 0.602) with high variance (std = 1.12)
- Freezing the same layers is non-significant (p = 0.74) — absence is tolerated, opposition is not
- The adversarial-vs-freeze distinction is highly significant at n=300
- Defines the architecture's tolerance limit: the residual stream routes around silence but cannot defend against active sabotage
- Classification: **tolerance** (freeze) / **severe degradation** (adversarial) — sharp line between absence and opposition
