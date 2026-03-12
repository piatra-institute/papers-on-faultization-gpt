# autoresearch-morphogpt

Autonomous research loop for a tiny numpy-based GPT (~11K–2M params, CPU-only).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar12`). The branch `autoresearch-morphogpt/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch-morphogpt/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `program.md` — this file. The experiment protocol.
   - `prepare.py` — fixed constants, data prep, tokenizer, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Run `uv run --script prepare.py` to download `data/input.txt` if needed.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on CPU (numpy backend). The training script runs for a **fixed time budget of 30 seconds** (wall clock training time, excluding warmup). You launch it as: `uv run --script train.py`

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, model size, activation functions, init schemes, batching strategies, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, block size, etc).
- Install new packages or add dependencies. Only `numpy` is available.
- Modify the evaluation harness. The `evaluate_val_loss` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_loss.** Since the time budget is fixed, you don't need to worry about training time — it's always 30 seconds. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_loss:         2.123456
training_seconds: 30.1
total_seconds:    30.5
num_steps:        4523
num_params:       13408
n_layer:          4
n_embd:           16
```

You can extract the key metric from the log file:

```
grep "^val_loss:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 4 columns:

```
commit	val_loss	status	description
```

1. git commit hash (short, 7 chars)
2. val_loss achieved (e.g. 2.123456) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	val_loss	status	description
a1b2c3d	2.123456	keep	baseline
b2c3d4e	2.089012	keep	increase LR to 0.02
c3d4e5f	2.234567	discard	switch to GeLU activation
d4e5f6g	0.000000	crash	double model width caused shape error
```

## Search space hints

Here are some directions worth exploring for tiny numpy GPT models:

- **Model size**: layer count (2–16), embedding dim (8–128), head count. More params = fewer steps in 30s, but each step is richer.
- **Optimizer**: Adam variants, SGD with momentum, learning rate schedules (cosine, warmup+decay), beta values.
- **Activation functions**: ReLU, squared ReLU, GELU, SiLU/swish, tanh.
- **Init schemes**: Xavier, He, scaled init, zero-init for output projections.
- **Architecture tweaks**: pre-norm vs post-norm, skip connection scaling, removing MLP, wider MLP ratio, tied embeddings.
- **Batching**: process multiple documents per step (mini-batching), gradient accumulation.
- **Regularization**: dropout, weight decay, gradient clipping.
- **Training tricks**: learning rate warmup, curriculum learning (short names first), label smoothing.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch-morphogpt/mar12`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run --script train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_loss:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_loss improved (lower), you "advance" the branch, keeping the git commit
9. If val_loss is equal or worse, you git reset back to where you started

**Timeout**: Each experiment should take ~30 seconds total (+ a few seconds for startup and eval overhead). If a run exceeds 2 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (a bug, shape mismatch, etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical architectural changes, try unusual optimizers. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. Each experiment takes ~30 seconds so you can run approx 120/hour, for a total of about 1000 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
