# MorphoGPT

Morphogenetic perturbation experiments on a tiny GPT (numpy backend).

## Running commands

Always use `uv run --script` to execute scripts (picks up inline dependencies):

```
uv run --script run.py <command> [--num-reps N] [--num-steps N] [--result-suffix SUFFIX]
```

## Key commands

```
uv run --script run.py test              # Quick smoke test
uv run --script run.py experiment1       # Single experiment (1-12)
uv run --script run.py all               # Experiments 1-6
uv run --script run.py all2              # Experiments 7-12
uv run --script run.py n300              # All 12 experiments at n=300 (saves with _n300 suffix)
uv run --script run.py analyze1          # Analyze experiment 1 (plots + tables)
```

Statistical analysis:
```
uv run --script analyze_stats.py         # Analyze experiments 1-6
uv run --script analyze_stats.py new     # Analyze experiments 7-12
uv run --script analyze_stats.py all     # Analyze all 12
```

## Project structure

- `morphogpt_np.py` — Core GPT model (numpy backend, fast)
- `perturbations_np.py` — All perturbation hooks (freeze, noise, stop-gradient, etc.)
- `experiments_np.py` — 12 experiment functions with `result_suffix` parameter
- `metrics.py` — Statistical metrics (DG index, robustness curves, etc.)
- `run.py` — CLI dispatcher
- `analyze_stats.py` — Paired t-tests and summary tables for all experiments
- `visualize.py` — Plotting functions
- `results/` — JSON result files (n=30 default, `_n300` suffix for n=300)
- `docs/PAPER.md` — Main paper
- `docs/FINDINGS.md` — Detailed findings

## Result file naming

- `results/experiment1_head_freezing.json` — n=30 results
- `results/experiment1_head_freezing_n300.json` — n=300 results
- Use `--result-suffix _n300` to save with suffix when running individual experiments

## Experiments

1. Head freezing robustness curve
2. Cell-view GPT (stop-gradient isolation)
3. Gradient degradation (noise, sign-only, quantized)
4. Vision radius sweep (windowed attention)
5. Communication topology (partial stop-gradient)
6. Courage vs. caution (forward noise vs backward corruption)
7. Recovery after damage
8. Chimera assembly (Franken-models)
9. Gradual vs sudden damage (stress inoculation)
10. Regeneration (layer reset)
11. Transplantation (foreign layer integration)
12. Competing objectives (conflicting gradients)
