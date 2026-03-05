``` bash
uv run microgpt.py

# Quick smoke test (20 steps, n_layer=1)
uv run run.py test

# Full baseline (500 steps, n_layer=4)
uv run run.py baseline

# Head freezing robustness curve
uv run run.py experiment1

# Cell-view GPT (stop-gradient)
uv run run.py experiment2

# Gradient degradation
uv run run.py experiment3

# All experiments
uv run run.py all
```
