# MorphoGPT: Experiment Summary

Twelve morphogenetic perturbation experiments on a minimal GPT (4-layer, 16-dim, 4-head, ~11K params). All results at n=300 (300 independent runs per condition, paired by seed).


## Experiment 1: Head Freezing

**Description/Assumptions:** Randomly selected attention heads have their parameters frozen at initialization (true parameter freezing — weights unchanged, but heads still participate in the forward pass). Motivated by Levin's frozen-cell perturbation. Tests whether the system depends on all heads being trainable, or whether frozen random-projection heads can serve as useful fixed features.

**Findings:**
- Final loss is unaffected by freezing any number of heads (all p > 0.15, Spearman ρ = −0.0045, p = 0.84) — wide basin of attraction absorbs the perturbation
- Mean trajectory loss improves significantly when heads are frozen: freeze 4 (Δ = −0.1%, p < 0.0001, d = −0.971), freeze 8 (Δ = −0.1%, p < 0.0001, d = −1.245), freeze 12 (Δ = −0.2%, p < 0.0001, d = −1.421), freeze 16 (Δ = −0.2%, p < 0.0001, d = −1.312)
- With true parameter freezing, frozen heads still compute in the forward pass — the trajectory improvement reflects reduced gradient interference during training, not removal of computation
- Classification: **emergent behavior** (trajectory improvement) + **basin geometry** (final-loss indifference)


## Experiment 2: Cell-View GPT (Local Loss)

**Description/Assumptions:** Each layer receives its own local loss signal (layerwise cross-entropy against the target), with no end-to-end backpropagation. Each layer treated as an autonomous agent. Tests whether layers can learn independently without inter-layer gradient communication.

**Findings:**
- Cell-view produces near-identical final loss to baseline (−0.0%, p = 0.90, ns) — local learning achieves equivalent convergence
- Mean trajectory loss shows a small but significant increase (+0.2%, p < 0.0001, d = +0.731) — the path is slightly less efficient, but the destination is the same
- DG index shows no significant change (p = 0.14), confirming DG does not track perturbation response
- Classification: **basin geometry** — local loss reaches the same final loss as end-to-end backpropagation, indicating the loss landscape has a single dominant basin accessible by either optimization route


## Experiment 3: Gradient Degradation

**Description/Assumptions:** Gradients are corrupted during training through four methods: additive Gaussian noise (σ=0.01 and σ=0.1), sign-only reduction (discarding magnitude), and 3-bit quantization. Motivated by Levin's noisy signaling channels. Tests how much gradient information the optimizer actually needs.

**Findings:**
- Sharp tolerance threshold: noise σ=0.01 is non-significant (−0.2%, p = 0.28), while all three severe methods are p < 0.0001
- Noise σ=0.1 degrades by +2.2% (d = +0.367), sign-only by +4.9% (d = +0.575), quantized 3-bit by +3.6% (d = +0.529) — all p < 0.0001
- The boundary between tolerance and degradation is a sharp step between σ=0.01 and σ=0.1, not a smooth curve
- Even worst-case (sign-only) remains within 5% of baseline — architecture constrains the solution space
- Classification: **tolerance** up to σ=0.01; **degradation** above


## Experiment 4: Vision Radius Sweep

**Description/Assumptions:** Each attention head's context window is restricted to a fixed radius, motivated by the distributed chess vision-radius experiment (Kofman, Campitelli & Levin, 2025). Tests whether limiting how far each head can attend affects learning, and whether an intermediate window outperforms full context.

**Findings:**
- Fine structure emerges at n=300: window 1 significantly worsens loss (+0.3%, p = 0.021), window 8 significantly improves it (−0.1%, p = 0.022)
- Windows 2, 4, and 16 remain non-significant (p = 0.93, 0.93, 1.00)
- A subtle monotonic gradient from harm at smallest window to benefit at intermediate window
- Window 16 (= full context) reproduces baseline exactly, confirming no implementation artifacts
- The chess paper's information-bottleneck hypothesis is not supported at meaningful effect sizes
- Classification: **tolerance** — attention restriction at all tested scales is absorbed without meaningful final-loss change


## Experiment 5: Communication Topology

**Description/Assumptions:** A spectrum of gradient flow topologies between full backpropagation and complete isolation, created by scaling the fraction of gradient signal passed through layer boundaries. At 0% (cell-view), each layer receives only its own local loss signal. Motivated by the chess paper's relay chains. Tests whether the system needs full gradient flow or can operate on partial signals.

**Findings:**
- The system is largely indifferent to gradient fraction: heavy/75% (p = 0.92), half/50% (p = 0.033, marginal +0.1%), light/25% (p = 0.59)
- Cell-view (0% gradient flow) produces near-identical final loss to baseline (−0.0%, p = 0.90, ns) — matching Experiment 2's local-loss finding
- No partial-flow condition outperforms full backpropagation — the U-shape seen at n=3 was sampling noise
- Classification: **tolerance** — substantial gradient flow reduction is absorbed without meaningful degradation


## Experiment 6: Courage vs. Caution

**Description/Assumptions:** A proper 2×2 factorial crossing forward-pass perturbation (cautious = tiny noise σ=0.001, courageous = dropout p=0.1) with gradient perturbation (cautious = sign-only, courageous = noisy σ=0.1). Each cell applies BOTH a forward and gradient perturbation simultaneously. Tests how forward-pass and gradient perturbation types interact.

**Findings:**
- Gradient perturbation type dominates: sign-only gradient conditions degrade by +5.0–5.2% (p < 0.0001), while noisy gradient conditions degrade by +1.9–2.5% (p < 0.0001)
- Forward perturbation type has minimal effect: cautious forward (tiny noise) and courageous forward (dropout) produce similar degradation within each gradient type
- Cautious/cautious (tiny noise + sign-only): +5.2% (p < 0.0001, d = +0.624); cautious/courageous (tiny noise + noisy grad): +1.9% (p < 0.0001, d = +0.318); courageous/cautious (dropout + sign-only): +5.0% (p < 0.0001, d = +0.616); courageous/courageous (dropout + noisy grad): +2.5% (p < 0.0001, d = +0.419)
- The finding is about gradient precision: sign-only gradients (discarding magnitude) harm optimization much more than noisy gradients (preserving magnitude with added noise) — regardless of forward perturbation
- Classification: **gradient precision dominance** — the optimizer's sensitivity is to gradient quality, not forward-pass stability


## Experiment 7: Recovery After Damage

**Description/Assumptions:** Train normally, freeze 8 heads (damage phase), then unfreeze and continue training (recovery phase). Motivated by Levin's regeneration paradigm. Tests whether transient damage leaves lasting traces or whether the model recovers completely.

**Findings:**
- Near-complete recovery: final loss ratio recovery/control = 1.0009 ± 0.0072, but a small residual deficit reaches significance (p = 0.030, d = +0.126)
- Mean recovery time = 0.8 ± 1.2 steps after damage removal; 272/300 runs achieved ratio ≤ 1.01
- No overshoot: mean overshoot = −0.0009 ± 0.0017 — the Levin signature (damaged organisms exceeding baseline) is absent
- Recovery is near-complete but not perfectly complete: at n=300, the tiny residual (+0.1%) reaches statistical significance despite being practically negligible
- Classification: **emergent behavior** — near-complete path-independent recovery is not prescribed by the loss minimization objective; the small residual is only detectable at high statistical power


## Experiment 8: Chimera Assembly

**Description/Assumptions:** Two models trained independently; layers from each are combined into a Frankenstein model (AABB, ABAB, BBAA, ABBA configurations), then training continues. Tests whether a model assembled from incompatible parts can converge normally.

**Findings:**
- All chimera types converge to control loss: AABB (p = 0.35), ABAB (p = 0.12), BBAA (p = 0.079, marginal), ABBA (p = 0.31)
- Despite starting at substantially worse loss (2.51–2.57), all chimeras converge to 2.44–2.46
- Layer assignment doesn't matter — whether layers alternate or cluster makes no difference
- BBAA shows a marginal trend at n=300, suggesting slight asymmetry, but no chimera type significantly differs from control
- Classification: **basin geometry** — SGD re-finds the same minimum from any structurally valid starting point


## Experiment 9: Gradual vs. Sudden Damage (Stress Inoculation)

**Description/Assumptions:** Compare gradual noise ramp (0→σ=0.1 over training) against sudden full noise (σ=0.1 from step 1) and sudden half (σ=0.1 from step 100). Both gradual and sudden reach the same peak noise level. Tests whether the history of perturbation exposure matters — whether gradual exposure builds tolerance that sudden exposure does not.

**Findings:**
- Gradual shows only mild degradation (+0.5%, p = 0.017); sudden full degrades by +1.8% (p < 0.0001, d = +0.318)
- Direct gradual-vs-sudden comparison: Δ = −1.3%, p = 0.0001, d = −0.227
- Sudden half also significant (+0.8%, p = 0.0002, d = +0.219)
- Gradual noise acts as regularization: mean loss is below control (−0.1%, p < 0.0001)
- The gradient update rule is identical at every step; only the history of noise levels differs — yet the system's final state depends on that history
- Classification: **emergent behavior** — stress inoculation is not prescribed by the optimizer; this is the paper's strongest result


## Experiment 10: Regeneration (Layer Reset)

**Description/Assumptions:** Destroy a layer entirely (reset to random weights at mid-training), then continue training. Tests whether the network can rebuild a completely destroyed layer to its original functional role.

**Findings:**
- All layers regenerate to near-control levels, but at n=300 all show small significant residual deficits: L0 (+0.3%, p = 0.003), L1 (+0.2%, p = 0.007), L2 (+0.1%, p = 0.024), L3 (+0.1%, p = 0.037)
- Completeness near 1.0 for L1–L3 (0.988, 0.994, 1.021) — the deficits are practically negligible despite statistical significance
- Layer position does not predict regeneration quality — all layers recover to within +0.3% of control
- Classification: **emergent behavior** — near-complete layer regeneration to control-equivalent performance is not prescribed by the loss minimization objective


## Experiment 11: Transplantation (Foreign Layer Integration)

**Description/Assumptions:** Replace a layer with one from a separately-trained donor model, then continue training. Compare against random reset (Experiment 10). Tests whether a pre-trained donor layer provides any advantage over a random replacement.

**Findings:**
- No transplant advantage: overall p = 0.76, with per-layer p values ranging from 0.29 to 0.98
- The network doesn't recognize donor structure — a donor layer's learned weights provide no advantage over random initialization
- Unlike biological transplantation where tissue compatibility matters, the network rebuilds whatever is placed at each position from scratch
- Classification: **basin geometry** — the basin is equally accessible from pre-trained and random initializations


## Experiment 12: Competing Objectives (Conflicting Gradients)

**Description/Assumptions:** Negate gradients for layers 2-3 while layers 0-1 train normally, creating actively adversarial components. Compare against simply freezing those same layers. Tests whether the architecture can compensate for components actively working against the training objective.

**Findings:**
- Competing objectives degrade by +26.3% (p < 0.0001, d = +0.531) with high variance (std = 1.31)
- Freezing the same layers is non-significant (−0.1%, p = 0.41) — absence is tolerated, opposition is not
- The adversarial-vs-freeze distinction is highly significant at n=300 (p < 0.0001, d = +0.535)
- Defines the architecture's tolerance limit: the residual stream routes around silence but cannot defend against active sabotage
- Classification: **tolerance** (freeze) / **severe degradation** (adversarial) — sharp line between absence and opposition
