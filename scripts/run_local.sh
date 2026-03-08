#!/usr/bin/env bash
# Run morphogpt experiments locally.
#
# Usage:
#   ./scripts/run_local.sh                              # All experiments, 3 reps (default)
#   ./scripts/run_local.sh --num-reps 30                # All experiments, 30 reps
#   ./scripts/run_local.sh experiment1 --num-reps 30    # Just experiment 1, 30 reps
#   ./scripts/run_local.sh all --num-reps 30 --num-steps 200

set -euo pipefail

cd "$(dirname "$0")/.."

EXPERIMENT="${1:-all}"
shift 2>/dev/null || true

echo "=== morphogpt local run ==="
echo "Start time:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "Experiment:   $EXPERIMENT"
echo "Extra args:   $*"
echo ""

uv run run.py "$EXPERIMENT" "$@" 2>&1 | tee "output/run_local.log"

echo ""
echo "=== Completed: $(date '+%Y-%m-%d %H:%M:%S') ==="
