# Claim Ledger

Every numeric claim in `PAPER.md` traced to committed simulation output. The
experiments are seeded (seeds 42–71 for n=30) and deterministic. Regenerate:

```
uv run run.py all          # experiments 1-6  (n=30) -> results/experiment{1..6}*.json
uv run run.py all2         # experiments 7-12 (n=30) -> results/experiment{7..12}*.json
uv run run.py n300         # all 12 at n=300 (~hours) -> results/*_n300.json
uv run analyze_stats.py all # paired t-tests, Cohen's d
```

Regenerated 2026-05-29 (n=30). All n=30 values below reproduce the paper exactly
(the results had simply never been committed). Model: 4-layer, 16-dim, 4-head
GPT; 200 steps; paired t-tests, runs matched by seed.

## n = 30 (committed)

| # | Claim | Source | Value | ✓ |
|---|---|---|---|---|
| 1 | Head freezing: trajectory improves with 4+ frozen heads (free lunch) | `experiment1_head_freezing.json` | freeze 8 d=−1.228, freeze 12 d=−1.366 (p<0.001) | [x] |
| 2 | Cell-view (local loss) reaches equivalent final loss | `experiment2_cell_view.json` | −0.9%, p=0.237 | [x] |
| 3 | Gradient degradation: sharp tolerance threshold | `experiment3_gradient_degradation.json` | σ=0.01 tolerated (p=0.843); sign-only +5.0% (p=0.0022); quantized +3.8% (p=0.0081); σ=0.1 +2.4% (p=0.032) | [x] |
| 4 | Vision radius: no final-loss effect at n=30 | `experiment4_vision_radius.json` | all p>0.30 | [x] |
| 5 | Communication topology: partial gradient flow tolerated | `experiment5_communication.json` | heavy/half/light p>0.37 (final) | [x] |
| 6 | Courage/caution: gradient type dominates forward type | `experiment6_courage_caution.json` | sign-only +5.2–6.5%; noisy +2.4–2.6% | [x] |
| 7 | Recovery after damage: near-complete, no overshoot | `experiment7_recovery.json` | ratio 1.0000, p=0.886 | [x] |
| 8 | Chimera assembly: all converge to control | `experiment8_chimera.json` | p>0.26 (BBAA marginal 0.076) | [x] |
| 9 | Gradual vs sudden (stress inoculation), n=30: gradual NS, sudden degrades | `experiment9_gradual_vs_sudden.json` | gradual p=0.641; sudden +2.4% p=0.032; direct p=0.032, d=−0.412 | [x] |
| 10 | Regeneration (layer reset): all layers recover | `experiment10_regeneration.json` | all p>0.17 | [x] |
| 11 | Transplantation: no donor advantage over random | `experiment11_transplantation.json` | overall p=0.880 | [x] |
| 12 | Competing objectives: adversarial +24.8% vs freeze +0.2% | `experiment12_competing_objectives.json` | competing p<0.001 d=+0.689; freeze p=0.462 | [x] |
| 13 | DG index null at n=30 (no perturbation response) | all experiments | p>0.19 all | [x] |

## n = 300 (regenerated 2026-05-29)

Regenerated all 12 experiments at n=300 with `uv run run.py n300`. The raw
trajectory JSONs are large (150–200 MB each) and exceed GitHub's file limit, so
they are gitignored; the committed trace is `results/n300_analysis.txt` (the full
paired-t-test output) and the verified values below. Regenerate any time with the
command above (seeded, deterministic).

Most n=300 figures reproduce the paper exactly; the gradual-vs-sudden effect size
drifted from an earlier run and the prose was updated to the reproduced value
(stronger). Two further drifts are flagged for a follow-up reconciliation.

| # | Claim (n=300) | Source | Value | ✓ |
|---|---|---|---|---|
| 14 | Gradual-vs-sudden, paired | `experiment9_gradual_vs_sudden_n300.json` | p<0.0001, **d=−0.374** (paper updated from the earlier −0.227) | [x] |
| 15 | Competing objectives | `experiment12_competing_objectives_n300.json` | +26.3%, p<0.0001, d=+0.531 — reproduced exactly | [x] |
| 16 | Final-loss freezing effects vanish | `experiment1_head_freezing_n300.json` | all p>0.15, Spearman ρ=−0.0045 — reproduced exactly | [x] |
| 17 | Recovery near-complete | `experiment7_recovery_n300.json` | ratio 1.0009 — reproduced exactly | [x] |

Follow-up reconciliation (prose still shows the earlier run's figures):
- Exp 6 courage/caution effect sizes drifted (e.g. cautious_cautious +5.2%/d=0.624
  → reproduced +5.0%/d=0.572; courageous_cautious d=0.616 → 0.549).
- Exp 1 freeze-*trajectory* (mean-loss) d's (−0.971/−1.245/−1.421/−1.312) are not
  emitted by the n=300 analysis script and were not re-verified this pass; the
  final-loss + Spearman results (row 16) are verified.
