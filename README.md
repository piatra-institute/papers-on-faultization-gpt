# MorphoGPT

Morphogenetic perturbation of a minimal transformer. Applies Levin et al.'s (2024) developmental biology methodology to a 4-layer, 16-dimensional, 4-head character-level GPT (~11K parameters) to reveal emergent robustness properties invisible during normal training.

Six experiments systematically perturb the training process: freezing attention heads, severing inter-layer gradient flow, corrupting gradient signals, restricting attention windows, scaling gradient communication, and crossing forward-pass stability with gradient boldness.

Paired statistical analysis (n=3, seeds matched across conditions) yields two statistically robust findings and several suggestive patterns that motivate replication at higher sample sizes.


## Findings

**Statistically supported (p < 0.05):**
- Freezing 8+ attention heads significantly *improves* loss (p < 0.01). The benefit emerges at a threshold (~50% of heads), not monotonically.
- Sign-only gradients degrade performance significantly more than dropout (p = 0.004), inverting the chess paper's "cautious position, courageous moves" prediction.

**Suggestive but underpowered at n=3:**
- Cell-view (no inter-layer gradients) is viable at +2.4% cost (p = 0.021)
- Sign-only gradients approximately match baseline in Experiment 3 (p = 0.860, consistent with SignSGD literature)
- Restricted attention windows and partial gradient flow show tolerance patterns

See `docs/PAPER.md` for the full paper and `docs/FINDINGS.md` for detailed experiment-by-experiment results.


## Quick Start

```bash
# Quick smoke test (20 steps, n_layer=1)
uv run run.py test

# Full baseline (500 steps, n_layer=4)
uv run run.py baseline
```


## Running Experiments

```bash
# Individual experiments
uv run run.py experiment1    # Head freezing robustness curve
uv run run.py experiment2    # Cell-view GPT (stop-gradient)
uv run run.py experiment3    # Gradient degradation
uv run run.py experiment4    # Vision radius sweep
uv run run.py experiment5    # Communication topology
uv run run.py experiment6    # Courage vs. caution

# All experiments
uv run run.py all

# With custom repetitions and steps
uv run run.py all --num-reps 30 --num-steps 200
uv run run.py experiment1 --num-reps 30
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

Results land in `output/runpod_results/`. A CPU pod is sufficient — the model is only ~11K parameters.


## Analysis

```bash
# Analyze individual experiments (reads from results/*.json)
uv run run.py analyze1    # Head freezing analysis
uv run run.py analyze2    # Cell-view analysis
uv run run.py analyze3    # Gradient degradation analysis
uv run run.py analyze4    # Vision radius analysis
uv run run.py analyze5    # Communication topology analysis
uv run run.py analyze6    # Courage/caution analysis
```


## Project Structure

```
morphogpt/
├── run.py              # Entry point — experiments + analysis
├── experiments.py      # Experiment runner, sweep infrastructure
├── morphogpt.py        # Core model, training loop, hooks, probes
├── perturbations.py    # Perturbation functions (freeze, noise, etc.)
├── metrics.py          # DG index, phase detection, anomaly analysis
├── visualize.py        # Plotting
├── microgpt.py         # Base micrograd GPT implementation
├── scripts/
│   ├── runpod_experiments.sh  # Deploy + run on RunPod
│   ├── runpod_run.sh          # Pod-side runner
│   ├── run_local.sh           # Local convenience wrapper
│   └── build-paper.sh         # Generate PDF from PAPER.md
├── docs/
│   ├── PAPER.md               # Full paper
│   ├── FINDINGS.md            # Detailed findings
│   ├── ARCHITECTURE.md        # Design document
│   ├── IMPLEMENTATION_NOTES.md
│   └── MODIFICATIONS.md       # Assumption inventory
├── data/               # Dataset (auto-downloaded)
└── results/            # Experiment outputs (JSON)
```


## Documentation

- **`docs/PAPER.md`** — Full paper with statistical analysis, two-tier anomaly structure, and chess-paper bridge
- **`docs/FINDINGS.md`** — Experiment-by-experiment results with paired t-test p-values
- **`docs/ARCHITECTURE.md`** — Original design document (Levin-Nancy philosophical framework)
- **`docs/MODIFICATIONS.md`** — Systematic inventory of 12 assumptions and their violations
