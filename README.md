# MorphoGPT

Morphogenetic perturbation of a minimal transformer. Applies Levin et al.'s (2024) developmental biology methodology to a 4-layer, 16-dimensional, 4-head character-level GPT (~13,400 parameters) to reveal emergent robustness properties invisible during normal training.

Twelve experiments systematically perturb the training process in two phases: perturbation-during-training (Exp 1-6: freezing attention heads, severing inter-layer gradient flow, corrupting gradient signals, restricting attention windows, scaling gradient communication, and crossing forward-pass stability with gradient boldness) and multi-phase morphogenetic interventions (Exp 7-12: recovery after damage, chimera assembly, stress inoculation, regeneration, transplantation, and competing objectives).

Three-scale statistical protocol: $n = 3$ pilot provides initial signal, $n = 30$ paired analysis (matched seeds) resolves ambiguity, $n = 300$ reveals fine structure. All formal results reported at $n = 30$ and $n = 300$.


## Findings

**Emergent behaviors (statistically supported at n=300):**
- Stress inoculation: gradual noise exposure builds tolerance that sudden exposure does not (direct comparison p=0.0001, d=-0.227)
- Head-freezing trajectory improvement: frozen random-projection heads reduce gradient interference (freeze 12: p<0.0001, d=-1.421)
- Complete recovery: damaged-then-recovered model returns to control loss (ratio 1.0009, p=0.030)
- Complete regeneration: any destroyed layer rebuilds to near-control performance (all layers p<0.04)

**Basin geometry:**
- Chimera convergence: models assembled from parts of two independently-trained networks converge to control loss (all p>0.07)
- Transplant indifference: donor layers provide no advantage over random reset (p=0.76)
- Cell-view equivalence: local layerwise loss achieves same final loss as end-to-end backpropagation (p=0.90)

**Tolerance:**
- Gradient degradation absorbed up to a sharp threshold between sigma=0.01 and sigma=0.1
- Partial communication (25-75% gradient flow) tolerated without degradation
- Vision restriction at all tested window sizes absorbed without meaningful final-loss change

**Chess-paper inversion:** Gradient type dominates over forward perturbation type, inverting the distributed chess "cautious position, courageous moves" prediction. Sign-only gradient conditions degrade by +5.0-5.2% regardless of forward perturbation. Substrate-dependent.

See `docs/PAPER.md` for the full paper and `docs/FINDINGS.md` for detailed experiment-by-experiment results.


## Quick Start

```bash
# Quick smoke test (20 steps, n_layer=1)
uv run --script run.py test

# Full baseline (500 steps, n_layer=4)
uv run --script run.py baseline
```


## Running Experiments

```bash
# Individual experiments (1-6: perturbation during training)
uv run --script run.py experiment1    # Head freezing robustness curve
uv run --script run.py experiment2    # Cell-view GPT (stop-gradient)
uv run --script run.py experiment3    # Gradient degradation
uv run --script run.py experiment4    # Vision radius sweep
uv run --script run.py experiment5    # Communication topology
uv run --script run.py experiment6    # Courage vs. caution

# Individual experiments (7-12: multi-phase morphogenetic interventions)
uv run --script run.py experiment7    # Recovery after damage
uv run --script run.py experiment8    # Chimera assembly
uv run --script run.py experiment9    # Gradual vs. sudden damage (stress inoculation)
uv run --script run.py experiment10   # Regeneration (layer reset)
uv run --script run.py experiment11   # Transplantation (foreign layer integration)
uv run --script run.py experiment12   # Competing objectives (conflicting gradients)

# Batch runs
uv run --script run.py all           # Experiments 1-6
uv run --script run.py all2          # Experiments 7-12
uv run --script run.py n300          # All 12 experiments at n=300

# With custom repetitions and steps
uv run --script run.py all --num-reps 30 --num-steps 200
uv run --script run.py experiment1 --num-reps 30
```


## Running on RunPod

For larger runs (30+ reps per condition), deploy to a RunPod instance:

```bash
# 1. Configure
cp .env.example .env
# Edit .env: set RUNPOD_SSH_HOST, RUNPOD_SSH_PORT, MORPHOGPT_NUM_REPS=30

# 2. Deploy and run
./scripts/runpod_experiments.sh

# 3. Monitor
./scripts/runpod_experiments.sh --status
./scripts/runpod_experiments.sh --tail

# 4. Download results
./scripts/runpod_experiments.sh --download
```

Results land in `output/runpod_results/`. A CPU pod is sufficient — the model is only ~13,400 parameters.


## Analysis

```bash
# Analyze individual experiments (reads from results/*.json)
uv run --script run.py analyze1    # Head freezing analysis
uv run --script run.py analyze2    # Cell-view analysis
uv run --script run.py analyze3    # Gradient degradation analysis
uv run --script run.py analyze4    # Vision radius analysis
uv run --script run.py analyze5    # Communication topology analysis
uv run --script run.py analyze6    # Courage/caution analysis

# Statistical analysis (paired t-tests and summary tables)
uv run --script analyze_stats.py         # Analyze experiments 1-6
uv run --script analyze_stats.py new     # Analyze experiments 7-12
uv run --script analyze_stats.py all     # Analyze all 12 experiments
```


## Project Structure

```
on-faultization-gpt/
├── run.py                    # CLI entry point
├── model.py                  # Core GPT model (numpy)
├── experiments.py            # 12 experiment functions
├── perturbations.py          # Perturbation hooks
├── metrics.py                # Statistical metrics
├── visualize.py              # Plotting
├── analyze_stats.py          # Paired t-tests
├── test_perturbation_semantics.py
├── CLAUDE.md
├── README.md
├── .gitignore
├── docs/
│   ├── PAPER.md
│   ├── FINDINGS.md
│   ├── EXPERIMENTS.md
│   └── MODIFICATIONS.md
├── scripts/
│   └── build-paper.sh
├── results/
├── data/
└── archive/
```


## Documentation

- **`docs/PAPER.md`** — Full paper with three-scale statistical analysis and three-category classification (emergent, basin geometry, tolerance)
- **`docs/FINDINGS.md`** — Experiment-by-experiment results with paired t-test p-values at n=30, n=300 annotations
- **`docs/EXPERIMENTS.md`** — Concise experiment summary with n=300 results
- **`docs/MODIFICATIONS.md`** — Systematic inventory of 12 assumptions and their violations
