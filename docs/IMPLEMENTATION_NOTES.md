# MorphoGPT — Implementation Analysis & Critical Notes

> **Note:** This is an early design document written before experiments were run.
> It uses philosophical framing (Nancy, Levin) that motivated the methodology.
> For scientific results with neutral terminology, see `PAPER.md` and `FINDINGS.md`.


---


## 1. Critical Problems with the Current Architecture


### Problem 1: n_layer=1 Cripples Half the Experiments

microgpt defaults to `n_layer=1`. With one layer, the following experiments are impossible or meaningless:

- **Cell-view GPT with stop-gradient boundaries**: no boundaries to cut (only 1 layer)
- **Block-local loss**: only one block, so "local" loss IS the global loss
- **Chimeric layer mixing**: nothing to mix
- **Inter-layer dynamics**: no inter-layer anything
- **Drop layer**: dropping the only layer = no model

**Decision**: MorphoGPT must use **n_layer=2 minimum**, probably **n_layer=4** for the full experiment suite. microgpt already supports this parametrically. The parameter count scales as:

| n_layer | n_embd=16, n_head=4 | n_embd=32, n_head=4 |
|---|---|---|
| 1 | 4,192 | ~14,000 |
| 2 | 6,464 | ~24,000 |
| 4 | 11,008 | ~44,000 |

Even at n_layer=4, n_embd=32 we're at ~44k params — still fully transparent, every scalar traceable, but enough structure for meaningful experiments.

**However**: more layers = more Value nodes in the computation graph = much slower training. This leads directly to Problem 2.


### Problem 2: Scalar Autograd Is Extremely Slow

microgpt's Value-based autograd operates on individual scalars. For one training step on a name of length ~7:
- ~7 forward passes through the transformer (one per position)
- Each pass creates thousands of Value nodes (projections = n_embd² per matrix, times many matrices)
- backward() traverses the entire graph

Rough estimates for 1000 training steps:

| Config | Time per run (estimate) |
|---|---|
| n_layer=1, n_embd=16 | 10–30 minutes |
| n_layer=2, n_embd=16 | 20–60 minutes |
| n_layer=4, n_embd=16 | 40–120 minutes |
| n_layer=4, n_embd=32 | hours |

For statistical significance (Levin used 100 repetitions), we need ~100 runs per configuration. With 20+ configurations, that's 2000+ runs. At 30 min each = 1000 hours. **Not feasible with scalar autograd.**

**Decision**: We need a **dual backend**:

1. **Scalar backend** (microgpt's Value autograd): for transparency, debugging, single-run analysis, visualization of individual computation graphs. This is the "reference implementation" that proves we understand every scalar.

2. **Numpy backend**: same architecture, same math, but using numpy arrays instead of scalar Values. Implements manual backprop (not hard for this architecture). **100–1000x faster**. Used for sweeps and statistical analysis.

We validate equivalence between backends on small cases (same seed, same data → same loss trajectory to floating-point precision).


### Problem 3: Levin's Two Frozen Cell Types Map to Different Things in GPT

Levin distinguishes:
- **Movable frozen cell**: won't initiate moves, but can be moved by others
- **Immovable frozen cell**: can neither move nor be moved

In GPT, this distinction maps to:

| Levin | GPT Analog | Implementation |
|---|---|---|
| Movable frozen | Component contributes to forward pass but doesn't learn (gradient frozen) | Zero gradients for that component's params after backward |
| Immovable frozen | Component doesn't contribute at all (forward output zeroed) | Replace output with zeros in forward pass |

These are genuinely different interventions:
- **Gradient-frozen head**: still computes attention, still influences the residual stream, but its weights never change. The rest of the network can learn to use (or route around) its fixed output.
- **Output-zeroed head**: contributes nothing. The rest of the network must compensate entirely.

**Both must be implemented and compared**, exactly as Levin compared both types.

A subtle point: in the scalar autograd, if we zero a head's output by replacing with `[Value(0) for _ in range(head_dim)]`, these new Values have no children. Gradients flowing backward through them reach a dead end — they don't affect the frozen head's parameters. This is correct for "immovable." But for "movable" (gradient-frozen), we let the forward pass compute normally and only zero the gradients in the optimizer step.


### Problem 4: "Aggregation" Needs a Sharper Definition for GPT

In Levin's sorting, aggregation is spatial: cells of the same Algotype cluster together in the array. This is measurable because cells have positions and types.

In a chimeric GPT, what is the analog? Components don't have "spatial positions" in the same way. We need to define aggregation carefully.

**Proposed definition**: In a chimeric GPT where layers have different "algotypes" (e.g., layer 0 = standard attention, layer 1 = linear attention, layer 2 = standard, layer 3 = linear), aggregation means:

1. **Representational aggregation**: Do layers of the same type produce more similar activation patterns than layers of different types? Measure: average pairwise cosine similarity of layer outputs, same-type vs. cross-type.

2. **Functional aggregation**: Do layers of the same type develop similar functional roles? Measure: when we ablate layers, does removing a same-type pair cause a different pattern of degradation than removing a cross-type pair?

3. **Gradient aggregation**: During training, do same-type layers have more correlated gradient updates than cross-type layers?

This is weaker than Levin's spatial aggregation (his cells literally move to be next to same-type cells). But it's the most faithful analog available.

**A stronger analog**: If we allow layers to be *reordered* during training (like cells move in the array), we could measure whether same-type layers end up adjacent. But this is architecturally weird — layer order matters in transformers.

**Best approach**: Start with representational and gradient aggregation. These are well-defined and measurable.


### Problem 5: Process-Focused vs. Endpoint-Focused Methodology

This is the most important conceptual issue: **does the morphogenetic framing actually change what we do, or is it decorative?**

The answer: it changes what we **measure** and how we **interpret**, even when the interventions look similar to standard ablation.

| Standard Ablation | MorphoGPT (process-focused) |
|---|---|
| Remove head, measure accuracy drop | Remove head, measure full **trajectory** of loss over training |
| Ask: "how important is this head?" | Ask: "what does the system do when this component is interrupted?" |
| Endpoint-focused | Process-focused |
| One-time measurement | Dynamic observation of rerouting, delayed gratification, recovery |
| Component importance ranking | Competence characterization of the whole system |
| The system is a machine with parts | The system is a collection of interacting components |

Concrete difference: A standard ablation study would report "removing head 2 increases loss by 0.3." MorphoGPT reports "removing head 2 during training causes loss to increase by 0.5 for 50 steps, then the system reroutes through heads 0 and 3 (evidenced by increased gradient flow to those heads), recovering to within 0.1 of baseline by step 200, with a delayed gratification index of 1.4 — significantly higher than the 0.8 DG index observed with no damage (p < 0.01)."

The second is a characterization of *system behavior*, not *component importance*. This is the morphogenetic perturbation contribution.


### Problem 6: Delayed Gratification — Separating Signal from Noise

In Levin's sorting, delayed gratification is well-defined: Sortedness temporarily decreases then increases past the previous level. The algorithms are deterministic, so any backtracking is structural.

In GPT training, loss fluctuates naturally (stochastic selection of documents, the optimization landscape). We need to distinguish:
- **Noise**: random loss fluctuation that happens even without damage
- **Genuine rerouting**: damage-induced loss increase followed by compensatory recovery

**Levin's solution (which we should adopt)**: Measure DG as a function of damage level. If DG increases with more damage, it's context-dependent rerouting, not random noise. Random noise would not correlate with damage level.

Specifically:
- DG(0 frozen heads) = baseline noise level
- DG(1 frozen head) should be > DG(0) if rerouting occurs
- DG(2 frozen heads) should be > DG(1)
- etc.

**Statistical test**: Regression of DG index on damage level. Significant positive slope = evidence for genuine delayed gratification.


---


## 2. Revised Experiment Parameters

### Default Model Configuration

```
n_layer = 4           # enough for inter-layer dynamics + chimeras
n_embd = 16           # keep small for transparency
n_head = 4            # 4 heads per layer = 16 total heads
head_dim = 4          # n_embd // n_head
block_size = 16       # matches microgpt
vocab_size = ~28      # 26 letters + BOS + maybe a few special tokens
total params ≈ 11,000 # still fully transparent
```

### Training Configuration

```
num_steps = 500       # reduced from 1000 for faster sweeps (increase for final runs)
learning_rate = 0.01
beta1, beta2 = 0.85, 0.99
lr_decay = linear
```

### Sweep Configuration

```
repetitions_per_config = 30   # for initial exploration (increase to 100 for final)
seeds = [42, 123, 456, ...]   # fixed seeds for reproducibility
```

### Estimated Compute (numpy backend)

```
time per run (numpy, n_layer=4): ~5-15 seconds
runs per configuration: 30
configurations in Phase 2: ~25 (damage type × damage level × schedule)
total Phase 2: ~25 × 30 × 10s ≈ 2 hours
total all phases: ~10-20 hours
```

This is feasible on a single machine.


---


## 3. Detailed Code Sketches

### 3.1 The Value Class (unchanged from microgpt)

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    # ... all operators as in microgpt ...

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad
```

### 3.2 The Hook System

Keep it simple. No over-engineering. A hook is a dict mapping names to callables.

```python
class Hooks:
    def __init__(self):
        self._hooks = {}    # name -> list[callable]
        self.step = 0       # current training step (for schedules)

    def register(self, name, fn):
        self._hooks.setdefault(name, []).append(fn)

    def clear(self, name=None):
        if name is None:
            self._hooks.clear()
        else:
            self._hooks.pop(name, None)

    def apply(self, name, value):
        """Apply all hooks for this name. Return (possibly modified) value."""
        for fn in self._hooks.get(name, []):
            result = fn(value, step=self.step)
            if result is not None:
                value = result
        return value

    def has(self, name):
        return name in self._hooks and len(self._hooks[name]) > 0
```

### 3.3 The Modified gpt() Function

```python
def gpt(token_id, pos_id, keys, values, state_dict, config, hooks=None):
    """
    One step of the GPT forward pass (processes one token).
    Returns logits over vocabulary.

    If hooks is None, behaves identically to original microgpt.
    """
    if hooks is None:
        hooks = Hooks()  # no-op hooks

    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    head_dim = n_embd // n_head

    # Embeddings
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = hooks.apply('emb', x)

    x = rmsnorm(x)

    for li in range(n_layer):

        # --- Attention block ---
        x_residual = x
        x = rmsnorm(x)

        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])

        # Hook: after QKV projection
        qkv = hooks.apply(f'qkv.{li}', (q, k, v))
        if qkv is not None:
            q, k, v = qkv

        keys[li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]

            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)

            # Hook: attention weights
            attn_weights = hooks.apply(f'attn_w.{li}.{h}', attn_weights)

            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]

            # Hook: head output (THIS IS THE KEY HOOK for freeze experiments)
            head_out = hooks.apply(f'head_out.{li}.{h}', head_out)

            x_attn.extend(head_out)

        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        # Hook: post-attention residual
        x = hooks.apply(f'post_attn.{li}', x)

        # --- MLP block ---
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]

        # Hook: MLP hidden
        x = hooks.apply(f'mlp_hidden.{li}', x)

        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

        # Hook: post-MLP residual (also used for stop-gradient in cell-view experiments)
        x = hooks.apply(f'post_mlp.{li}', x)

    logits = linear(x, state_dict['lm_head'])
    logits = hooks.apply('logits', logits)

    return logits
```

### 3.4 Perturbation Implementations

```python
# --- FROZEN COMPONENTS (Levin's frozen cells) ---

def make_zero_head(layer, head, head_dim):
    """Immovable frozen cell: head contributes nothing to forward pass."""
    target = f'head_out.{layer}.{head}'
    def hook(head_out, step=0):
        return [Value(0.0) for _ in range(head_dim)]
    return target, hook


def make_freeze_params(param_names):
    """
    Movable frozen cell: params contribute to forward pass but don't learn.
    Applied AFTER backward, BEFORE optimizer step.
    Returns a function that zeroes gradients for the specified params.
    """
    def freeze(params):
        for name in param_names:
            for row in params[name]:
                for p in row:
                    p.grad = 0
    return freeze


def make_noise_injection(hook_name, noise_std):
    """Damaged cell: add Gaussian noise at a hook point."""
    def hook(values, step=0):
        return [Value(v.data + random.gauss(0, noise_std),
                      children=(v,), local_grads=(1,))
                for v in values]
    return hook_name, hook


def make_drop_layer(layer):
    """
    Skip a layer entirely: post-attention residual = pre-attention input.
    Implemented by making both attention and MLP output zero.
    """
    # Zero the attention output
    def attn_hook(x, step=0):
        # Return input unchanged (skip attention contribution)
        # This is tricky — we need to intercept at the right point.
        # Better approach: hook post_attn and post_mlp to pass through x_residual.
        pass
    # ... (needs more thought — see note below)


# --- STOP-GRADIENT (for cell-view GPT) ---

def make_stop_gradient(layer):
    """
    Cut gradient flow at the residual stream after a layer.
    Create new Value nodes with same data but no parents.
    """
    target = f'post_mlp.{layer}'
    def hook(values, step=0):
        return [Value(v.data) for v in values]  # new leaf nodes
    return target, hook


# --- SCHEDULES ---

def schedule_chronic(hook_fn):
    """Always active."""
    return hook_fn

def schedule_acute(hook_fn, start_step, end_step):
    """Active only during [start_step, end_step]."""
    def scheduled_hook(values, step=0):
        if start_step <= step <= end_step:
            return hook_fn(values, step=step)
        return None  # pass through
    return scheduled_hook

def schedule_stochastic(hook_fn, prob):
    """Active with probability prob at each call."""
    def scheduled_hook(values, step=0):
        if random.random() < prob:
            return hook_fn(values, step=step)
        return None
    return scheduled_hook
```

### 3.5 The Probe (Instrumentation)

```python
class Probe:
    """
    Records the state of the sorting—er, training—process.
    Named after Levin's Probe object.
    """

    def __init__(self, record_interval=1):
        self.record_interval = record_interval
        self.losses = []               # loss at every step
        self.grad_norms = {}           # param_group -> list of norms
        self.head_outputs = {}         # (layer, head) -> list of output norms
        self.attention_entropies = {}  # (layer, head) -> list of entropies
        self.samples = []              # generated samples at checkpoints
        self.custom = {}               # arbitrary named metrics

    def record_loss(self, step, loss_value):
        self.losses.append((step, loss_value))

    def record_grad_norm(self, step, group_name, norm):
        self.grad_norms.setdefault(group_name, []).append((step, norm))

    def record_head_output(self, step, layer, head, values):
        """Record the L2 norm of a head's output (a summary statistic)."""
        norm = sum(v.data ** 2 for v in values) ** 0.5
        self.head_outputs.setdefault((layer, head), []).append((step, norm))

    def record_attn_entropy(self, step, layer, head, weights):
        """Shannon entropy of attention weight distribution."""
        import math
        entropy = -sum(
            w.data * math.log(w.data + 1e-10) for w in weights
        )
        self.attention_entropies.setdefault((layer, head), []).append((step, entropy))

    def compute_delayed_gratification(self, window=10):
        """
        Compute Levin's delayed gratification index from the loss trajectory.

        Scan for episodes where loss increases for `window` steps then decreases
        past the pre-increase level.

        Returns list of (step, DG_value) for each detected episode.
        """
        if len(self.losses) < 2 * window:
            return []

        loss_vals = [l for _, l in self.losses]
        episodes = []

        i = 0
        while i < len(loss_vals) - 1:
            # Find start of a loss increase
            if loss_vals[i + 1] > loss_vals[i]:
                start = i
                peak_val = loss_vals[i + 1]
                # Find the peak
                j = i + 1
                while j < len(loss_vals) - 1 and loss_vals[j + 1] >= loss_vals[j]:
                    peak_val = loss_vals[j + 1]
                    j += 1
                # j is now at the peak. Find recovery.
                decrease_start = j
                k = j
                while k < len(loss_vals) - 1 and loss_vals[k + 1] <= loss_vals[k]:
                    k += 1
                trough_val = loss_vals[k]

                # DG = net gain / temporary loss
                temporary_loss = peak_val - loss_vals[start]
                net_gain_after = peak_val - trough_val

                if temporary_loss > 0:
                    dg = (net_gain_after - temporary_loss) / temporary_loss
                    if dg > 0:  # genuinely recovered past the pre-increase level
                        episodes.append((start, dg))

                i = k
            else:
                i += 1

        return episodes
```

### 3.6 The Numpy Backend (Sketch)

```python
import numpy as np

def gpt_numpy(token_id, pos_id, kv_cache, state_dict, config):
    """
    Same architecture as microgpt, but using numpy arrays.
    Returns logits (numpy array) and updated kv_cache.
    """
    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    head_dim = n_embd // n_head

    # Embeddings: state_dict values are numpy arrays
    x = state_dict['wte'][token_id] + state_dict['wpe'][pos_id]  # (n_embd,)
    x = rmsnorm_np(x)

    for li in range(n_layer):
        x_res = x
        x = rmsnorm_np(x)

        q = state_dict[f'layer{li}.attn_wq'] @ x   # (n_embd,)
        k = state_dict[f'layer{li}.attn_wk'] @ x
        v = state_dict[f'layer{li}.attn_wv'] @ x

        kv_cache[li]['k'].append(k)
        kv_cache[li]['v'].append(v)

        x_attn = np.zeros(n_embd)
        for h in range(n_head):
            s = slice(h * head_dim, (h + 1) * head_dim)
            q_h = q[s]                                          # (head_dim,)
            K_h = np.array([ki[s] for ki in kv_cache[li]['k']]) # (T, head_dim)
            V_h = np.array([vi[s] for vi in kv_cache[li]['v']]) # (T, head_dim)

            scores = K_h @ q_h / head_dim**0.5                 # (T,)
            weights = softmax_np(scores)                         # (T,)
            x_attn[s] = V_h.T @ weights                         # (head_dim,)

        x = state_dict[f'layer{li}.attn_wo'] @ x_attn
        x = x + x_res

        x_res = x
        x = rmsnorm_np(x)
        x = state_dict[f'layer{li}.mlp_fc1'] @ x
        x = np.maximum(0, x)  # ReLU
        x = state_dict[f'layer{li}.mlp_fc2'] @ x
        x = x + x_res

    logits = state_dict['lm_head'] @ x
    return logits, kv_cache


def rmsnorm_np(x):
    ms = np.mean(x ** 2)
    return x / np.sqrt(ms + 1e-5)

def softmax_np(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()
```

For backprop in the numpy backend, we implement manual gradients for each operation. This is more work but straightforward for the small set of operations used (linear, rmsnorm, softmax, relu, cross-entropy). The reward is ~1000x speedup.


---


## 4. The Actual First Experiment (Most Important)

Before the grand sweep, we need ONE experiment that validates the entire framework and produces an interesting result. This is the "proof of concept" experiment.

### The Experiment: Head Freezing Robustness Curve

**Setup**:
- n_layer=4, n_embd=16, n_head=4 (16 heads total)
- Train on names, 500 steps
- Baseline: no damage
- Conditions: freeze 0, 1, 2, 4, 8, 12, 16 heads (randomly chosen per run)
- For each condition: 30 repetitions with different seeds
- Freeze type: immovable (zero output in forward pass)

**Measurements**:
- Loss trajectory for each run
- Final loss (average ± std over repetitions)
- Delayed gratification index (average over repetitions)
- Sample quality (subjective: are generated names plausible?)

**Expected results (hypotheses)**:
- Loss increases smoothly with number of frozen heads (graceful degradation, H1)
- With 1-4 frozen heads (of 16), loss barely increases (redundancy)
- With 12+ frozen heads, loss is significantly worse but the model still learns something
- DG index increases with number of frozen heads (H2)

**This single experiment** produces Figure 1 of the paper: a robustness curve (x = fraction of heads frozen, y = final loss), plus an overlay of DG index vs. damage level.

If this works, we know:
1. The hook system functions correctly
2. The model has measurable redundancy / fault tolerance
3. The DG metric detects something meaningful
4. The framework is validated for the rest of the experiments

### Why This Is the Best First Experiment

It directly mirrors Levin's first result (error tolerance comparison between traditional and cell-view sorting, Figure 5 in the paper). It's simple, produces a clear plot, and tells us immediately whether the approach works.


---


## 5. What to Deprioritize

Not everything in the architecture needs to happen. Prioritize by information value:

### HIGH PRIORITY (must do)
1. Hook system + baseline
2. Head freezing robustness curves (Levin's core experiment)
3. Acute damage + recovery dynamics (rerouting)
4. DG index computation and validation
5. Numpy backend (needed for statistical power)

### MEDIUM PRIORITY (do if Phase 2 results are interesting)
6. Cell-view GPT with block-local loss
7. Stop-gradient experiments
8. Simple chimeras (mixed pretrained blocks)

### LOW PRIORITY (speculative, do last)
9. Hebbian attention (poorly defined, may not work at all)
10. Predictive coding (interesting but orthogonal)
11. Opposing objectives chimera (fun but hard to interpret)
12. Structured tasks (brackets, copy — delayed gratification in inference)

### CUT ENTIRELY
- Mixed precision chimeras (not interesting enough, confounds with quantization damage)
- Per-parameter damage (too fine-grained to produce interpretable results)


---


## 6. Addressing the "Is This Just Ablation?" Objection

The most likely criticism: "This is just ablation studies with extra philosophy."

### Why it's not:

1. **Ablation removes and measures once. We perturb and observe trajectories.** Standard ablation: "head 3 has importance score 0.7." MorphoGPT: "when head 3 is frozen at step 200, the system takes 80 steps to reroute through heads 0 and 1 (evidenced by their gradient norms increasing 3x), temporarily increasing loss by 0.4 before recovering to 0.15 above baseline, with DG index 1.8."

2. **We damage during training, not just at inference.** Most ablation work freezes a trained model and measures degradation at test time. We damage the model during training and observe how the training dynamics respond. This is fundamentally different — it asks whether the learning process itself can navigate around obstacles (Levin's question).

3. **Chimeric experiments are novel.** No one (to our knowledge) has mixed pretrained transformer blocks from different training runs and measured emergent aggregation of same-type components. This is only meaningful in the Levin framework.

4. **The DG metric and its context-sensitivity are new.** Showing that temporary loss increase during training is not noise but a damage-dependent rerouting strategy is a new claim.

5. **The philosophical frame changes the interpretation.** We're not asking "which parts are important?" (engineering question). We're asking "what competencies does this collective system possess?" (diverse intelligence question). Same data, different science.


---


## 7. Numpy Backend: Manual Backprop Sketch

For the numpy backend, we need to implement backward passes for:
- Linear (matrix-vector multiply)
- RMSNorm
- ReLU
- Softmax
- Cross-entropy loss
- Residual connections (addition)

The architecture processes one token at a time (with KV cache), so we need to track gradients for each forward step and accumulate across positions.

### Approach: Store Forward Activations, Compute Gradients in Reverse

```python
class NumpyGPT:
    def __init__(self, config, state_dict):
        self.config = config
        self.sd = state_dict  # dict of numpy arrays
        self.grads = {}       # same structure, accumulates gradients

    def forward_and_backward_one_doc(self, tokens):
        """
        Process a full document (list of token ids).
        Returns loss value and populates self.grads.
        """
        n = min(self.config['block_size'], len(tokens) - 1)
        kv_cache = [[{'k': [], 'v': []}] for _ in range(self.config['n_layer'])]

        total_loss = 0.0
        # We need to store intermediate activations for backprop
        all_activations = []

        for pos in range(n):
            token_id = tokens[pos]
            target_id = tokens[pos + 1]

            # Forward pass (store all intermediates)
            acts = self._forward_one_token(token_id, pos, kv_cache)
            all_activations.append(acts)

            # Loss
            probs = softmax_np(acts['logits'])
            loss = -np.log(probs[target_id] + 1e-10)
            total_loss += loss

            # Backward from loss through this position
            # dL/dlogits
            dlogits = probs.copy()
            dlogits[target_id] -= 1  # softmax + cross-entropy gradient
            dlogits /= n  # average over positions

            self._backward_one_token(dlogits, acts)

        return total_loss / n

    def _forward_one_token(self, token_id, pos_id, kv_cache):
        """Forward pass storing all intermediates needed for backprop."""
        acts = {'token_id': token_id, 'pos_id': pos_id}
        # ... (store x at each stage, attention weights, pre-relu, etc.)
        return acts

    def _backward_one_token(self, dlogits, acts):
        """Backward pass accumulating parameter gradients."""
        # ... (manual chain rule through each operation)
        pass
```

This is verbose but mechanical. Each operation has a known gradient. The key insight: because microgpt processes one token at a time (not in parallel), the backward pass through KV cache is tricky — gradients flow back through cached keys and values from earlier positions.

**Simplification**: For sweep experiments, we may not need full backprop correctness. We could:
1. Use PyTorch with `torch.no_grad()` + manual gradient computation
2. Or just use PyTorch's autograd on tensor operations (fastest)

But using PyTorch somewhat defeats the "dependency-free" ethos. A middle ground: **numpy for forward + manual backprop for the operations we actually use.** It's ~200 lines of gradient code for this small architecture.


---


## 8. Open Questions for Implementation

### Q1: How to handle KV cache gradients in the numpy backend?

In microgpt's scalar autograd, gradients flow naturally through cached Value objects. In the numpy backend, we need to explicitly track that k and v from position 0 are used in attention at positions 1, 2, 3, etc. This means the gradient for k at position 0 accumulates contributions from all later positions where it was attended to.

**Approach**: After processing all positions, sweep backward through positions. At each position, the attention gradient w.r.t. cached k's and v's is distributed to the positions that produced them.

### Q2: Should hooks in the numpy backend be tensor-level or scalar-level?

Tensor-level (numpy arrays). The hook API should be backend-agnostic: hooks receive and return either lists-of-Values (scalar backend) or numpy arrays (numpy backend). We can abstract this with a simple wrapper.

### Q3: How many seeds for final results?

Levin used 100 repetitions. For initial exploration, 30 is sufficient. For the paper, we should aim for 100. With the numpy backend, this is feasible.

### Q4: Should we also test damage at inference time (not just training time)?

Yes, but as a secondary analysis. The primary contribution is damage during training (which tests the learning dynamics, not just the learned representation). Inference-time damage is closer to standard ablation.

### Q5: What's the right way to measure "rerouting"?

When head 3 is frozen and the system compensates, we want to show that compensation is happening through specific other heads. Measure:
- Gradient norm per head before and after damage
- If head 0's gradient norm increases significantly after head 3 is frozen, head 0 is "rerouting" to compensate
- This can be plotted as a matrix: damage target (x) vs. gradient change in each other component (y)
