#!/usr/bin/env bash
# Run morphogpt experiments on RunPod. Launched by runpod_experiments.sh via tmux.
# Configuration is read from env vars set by the launcher.
set -euo pipefail

REMOTE_DIR="/workspace/morphogpt"
export PATH="$HOME/.local/bin:$PATH"
cd "$REMOTE_DIR"

# Read config from env (set by launcher)
NUM_REPS="${MORPHOGPT_NUM_REPS:-30}"
NUM_STEPS="${MORPHOGPT_NUM_STEPS:-200}"
EXPERIMENTS="${MORPHOGPT_EXPERIMENTS:-all}"

echo "=== morphogpt experiment run ==="
echo "Start time:   $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Experiments:  $EXPERIMENTS"
echo "Num reps:     $NUM_REPS"
echo "Num steps:    $NUM_STEPS"
echo ""

mkdir -p "$REMOTE_DIR/output"

# Run with uv
uv run run.py "$EXPERIMENTS" \
    --num-reps "$NUM_REPS" \
    --num-steps "$NUM_STEPS" \
    2>&1 | tee "$REMOTE_DIR/output/run.log"

echo ""
echo "=== Completed: $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

# Show results summary
if [ -d "$REMOTE_DIR/results" ]; then
    echo ""
    echo "--- Results ---"
    for f in "$REMOTE_DIR/results"/*.json; do
        if [ -f "$f" ]; then
            n=$(python3 -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null || echo "?")
            echo "  $(basename "$f"): $n runs"
        fi
    done
fi
