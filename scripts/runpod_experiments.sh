#!/usr/bin/env bash
# Deploy morphogpt to RunPod and run experiments with configurable repetitions.
#
# Reads SSH connection + options from .env in project root.
# Does everything in one shot: upload code, install deps, and launch
# the experiment pipeline in tmux.
#
# .env variables:
#   RUNPOD_SSH_HOST          (required) RunPod SSH host IP
#   RUNPOD_SSH_PORT          (required) RunPod SSH port
#   MORPHOGPT_NUM_REPS       (optional) Repetitions per condition (default: 30)
#   MORPHOGPT_NUM_STEPS      (optional) Training steps per run (default: 200)
#   MORPHOGPT_EXPERIMENTS    (optional) Which experiments: all|experiment1|...|experiment6 (default: all)
#   RUNPOD_TZ                (optional) Timezone (default: Etc/GMT-2)
#
# Usage:
#   ./scripts/runpod_experiments.sh              # Full setup + run
#   ./scripts/runpod_experiments.sh --status     # Check pipeline status
#   ./scripts/runpod_experiments.sh --tail       # Watch log live
#   ./scripts/runpod_experiments.sh --download   # Download results
#   ./scripts/runpod_experiments.sh --setup-only # Upload + install, don't run
#
# Prerequisites:
#   - A RunPod pod with SSH enabled (CPU pod is fine, ~11K params)
#   - Your SSH key added to RunPod account settings

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env file not found at $ENV_FILE"
    echo "Create it from .env.example:"
    echo "  cp .env.example .env"
    echo "Then set RUNPOD_SSH_HOST and RUNPOD_SSH_PORT"
    exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

if [ -z "${RUNPOD_SSH_HOST:-}" ] || [ -z "${RUNPOD_SSH_PORT:-}" ]; then
    echo "ERROR: RUNPOD_SSH_HOST and RUNPOD_SSH_PORT must be set in .env"
    exit 1
fi

SSH_HOST="$RUNPOD_SSH_HOST"
SSH_PORT="$RUNPOD_SSH_PORT"

# Configurable defaults (override in .env)
NUM_REPS="${MORPHOGPT_NUM_REPS:-30}"
NUM_STEPS="${MORPHOGPT_NUM_STEPS:-200}"
EXPERIMENTS="${MORPHOGPT_EXPERIMENTS:-all}"
REMOTE_TZ="${RUNPOD_TZ:-Etc/GMT-2}"

REMOTE_DIR="/workspace/morphogpt"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
SSH_CMD="ssh $SSH_OPTS -p $SSH_PORT root@$SSH_HOST"
SCP_CMD="scp $SSH_OPTS -P $SSH_PORT"
LOCAL_RESULTS="$REPO_ROOT/output/runpod_results"

# ---------------------------------------------------------------------------
# Parse mode
# ---------------------------------------------------------------------------

MODE="run"
if [ $# -gt 0 ]; then
    case "$1" in
        --status)     MODE="status" ;;
        --tail)       MODE="tail" ;;
        --download)   MODE="download" ;;
        --setup-only) MODE="setup" ;;
        *) echo "Unknown option: $1"; echo "Use: --status, --tail, --download, --setup-only"; exit 1 ;;
    esac
fi

# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

if [ "$MODE" = "status" ]; then
    echo "==> Checking pipeline status on $SSH_HOST:$SSH_PORT..."
    $SSH_CMD bash << 'STATUS_EOF'
REMOTE_DIR="/workspace/morphogpt"

if tmux has-session -t morphogpt_run 2>/dev/null; then
    echo "STATUS: Pipeline is RUNNING"
else
    echo "STATUS: Pipeline is NOT running"
fi

if [ -f "$REMOTE_DIR/output/run.log" ]; then
    echo ""
    echo "--- Last 30 lines of log ---"
    tail -30 "$REMOTE_DIR/output/run.log"
fi

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
STATUS_EOF
    exit 0
fi

# ---------------------------------------------------------------------------
# Tail
# ---------------------------------------------------------------------------

if [ "$MODE" = "tail" ]; then
    echo "==> Tailing log (Ctrl-C to detach, pipeline keeps running)..."
    $SSH_CMD "tail -f $REMOTE_DIR/output/run.log" || true
    exit 0
fi

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

if [ "$MODE" = "download" ]; then
    echo "==> Downloading results..."
    mkdir -p "$LOCAL_RESULTS"
    $SCP_CMD -r "root@$SSH_HOST:$REMOTE_DIR/results/" "$LOCAL_RESULTS/"
    $SCP_CMD "root@$SSH_HOST:$REMOTE_DIR/output/run.log" "$LOCAL_RESULTS/" 2>/dev/null || true
    echo ""
    echo "==> Results in $LOCAL_RESULTS/"
    ls -la "$LOCAL_RESULTS/" 2>/dev/null || true
    exit 0
fi

# ---------------------------------------------------------------------------
# Connectivity check
# ---------------------------------------------------------------------------

echo "==> Checking SSH connectivity to $SSH_HOST:$SSH_PORT..."
$SSH_CMD "echo 'Connected to RunPod'" || {
    echo "ERROR: Cannot connect to root@$SSH_HOST:$SSH_PORT"
    echo "Check that:"
    echo "  - The pod is running"
    echo "  - Your SSH key is added to RunPod settings"
    echo "  - RUNPOD_SSH_HOST and RUNPOD_SSH_PORT are correct in .env"
    exit 1
}

# ---------------------------------------------------------------------------
# Step 1: Bundle and upload
# ---------------------------------------------------------------------------

echo "==> Bundling morphogpt source..."
TARBALL="$(mktemp -d)/morphogpt_src.tar.gz"
tar -czf "$TARBALL" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='output' \
    --exclude='results' \
    --exclude='data' \
    --exclude='.env' \
    --exclude='.git' \
    -C "$REPO_ROOT" \
    run.py \
    experiments.py \
    experiments_np.py \
    morphogpt.py \
    morphogpt_np.py \
    perturbations.py \
    perturbations_np.py \
    metrics.py \
    visualize.py \
    microgpt.py \
    scripts/runpod_run.sh
echo "   Tarball: $(du -h "$TARBALL" | cut -f1)"

echo "==> Uploading to RunPod..."
$SSH_CMD "mkdir -p $REMOTE_DIR"
$SCP_CMD "$TARBALL" "root@$SSH_HOST:$REMOTE_DIR/morphogpt_src.tar.gz"
rm -f "$TARBALL"

# ---------------------------------------------------------------------------
# Step 2: Install dependencies on pod
# ---------------------------------------------------------------------------

echo "==> Setting up environment on RunPod..."
$SSH_CMD bash << 'SETUP_EOF'
set -euo pipefail

REMOTE_DIR="/workspace/morphogpt"
cd "$REMOTE_DIR"

echo "--- Extracting source ---"
tar -xzf morphogpt_src.tar.gz
chmod +x scripts/runpod_run.sh

# -- uv --
if ! command -v uv &> /dev/null; then
    echo "--- Installing uv ---"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env" 2>/dev/null || true
fi
export PATH="$HOME/.local/bin:$PATH"

# -- venv + install --
if [ ! -d ".venv" ]; then
    echo "--- Creating venv ---"
    uv venv --python 3.12
fi
echo "--- Installing dependencies ---"
uv pip install numpy matplotlib 2>/dev/null || uv pip install numpy

# -- tmux --
if ! command -v tmux &> /dev/null; then
    apt-get update -qq && apt-get install -y -qq tmux > /dev/null 2>&1
fi

# -- Verify --
echo ""
echo "--- Verifying installation ---"
source "$REMOTE_DIR/.venv/bin/activate"
python -c "import numpy; print(f'numpy {numpy.__version__} OK')"
mkdir -p "$REMOTE_DIR/data" "$REMOTE_DIR/results"
python -c "from morphogpt import load_dataset; docs, *_ = load_dataset(); print(f'Dataset: {len(docs)} docs')"

echo ""
echo "--- Setup complete ---"
SETUP_EOF

if [ "$MODE" = "setup" ]; then
    echo "==> Setup complete. Pod is ready."
    echo "    To run: $0"
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 3: Launch experiments in tmux
# ---------------------------------------------------------------------------

echo "==> Launching experiment pipeline..."
echo "    Experiments:  $EXPERIMENTS"
echo "    Num reps:     $NUM_REPS"
echo "    Num steps:    $NUM_STEPS"
echo ""

$SSH_CMD bash << LAUNCH_EOF
set -euo pipefail

REMOTE_DIR="/workspace/morphogpt"

# Kill any previous run session
tmux kill-session -t morphogpt_run 2>/dev/null || true

# Launch in tmux with env vars for configuration
tmux new-session -d -s morphogpt_run \
    "TZ='$REMOTE_TZ' \
     MORPHOGPT_NUM_REPS='$NUM_REPS' \
     MORPHOGPT_NUM_STEPS='$NUM_STEPS' \
     MORPHOGPT_EXPERIMENTS='$EXPERIMENTS' \
     bash \$REMOTE_DIR/scripts/runpod_run.sh; \
     echo ''; echo 'Press Enter to close...'; read"

echo "Pipeline launched in tmux session 'morphogpt_run'"
LAUNCH_EOF

echo ""
echo "==> Pipeline is running in a detached tmux session on RunPod."
echo "    It will continue even if your SSH disconnects."
echo ""
echo "    Total runs: ~$(echo "$NUM_REPS * 24" | bc 2>/dev/null || echo "N*24") (6 experiments x ~4 conditions x $NUM_REPS reps)"
echo ""
echo "Commands:"
echo "    Status:    $0 --status"
echo "    Tail log:  $0 --tail"
echo "    Download:  $0 --download"
echo "    SSH in:    ssh $SSH_OPTS -p $SSH_PORT root@$SSH_HOST"
echo "    Attach:    ssh $SSH_OPTS -p $SSH_PORT root@$SSH_HOST -t 'tmux attach -t morphogpt_run'"
