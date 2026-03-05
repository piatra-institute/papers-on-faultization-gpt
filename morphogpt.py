"""
MorphoGPT — Désœuvrement of the GPT Loop

microgpt + hook system + probe instrumentation.
When no hooks are active, reproduces microgpt behavior exactly.

Applies Michael Levin's morphogenetic perturbation methodology
to Karpathy's microgpt, guided by Jean-Luc Nancy's désœuvrement.
"""

import os
import math
import random
from dataclasses import dataclass


# ============================================================================
# Value (autograd) — identical to microgpt
# ============================================================================

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

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

    def __repr__(self):
        return f"Value({self.data:.4f})"


# ============================================================================
# Hooks — the perturbation injection system
# ============================================================================

class Hooks:
    """
    Named hook points in the computation graph.
    Each hook is a callable: fn(value, step=step) -> modified_value or None.
    If a hook returns None, the value passes through unchanged.
    """

    def __init__(self):
        self._hooks = {}   # name -> list[callable]
        self.step = 0      # current training step (for schedules)

    def register(self, name, fn):
        """Register a hook at a named point."""
        self._hooks.setdefault(name, []).append(fn)

    def clear(self, name=None):
        """Clear hooks. If name given, clear only that hook point."""
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

    def list_hooks(self):
        """Return dict of hook_name -> count."""
        return {k: len(v) for k, v in self._hooks.items()}


# ============================================================================
# Probe — instrumentation (Levin's terminology)
# ============================================================================

class Probe:
    """
    Records the trajectory of the training process.
    Named after Levin's Probe object.

    Supports three detail levels:
        'full'     — store per-step snapshots (embeddings, per-head vectors)
        'summary'  — store per-head norms/entropies per step (default)
        'loss_only' — loss and gradient norms only (original behavior)
    """

    def __init__(self, record_interval=1, detail_level='summary'):
        self.record_interval = record_interval
        self.detail_level = detail_level

        # Backward-compatible storage
        self.losses = []                # (step, loss_value)
        self.grad_norms = {}            # group_name -> [(step, norm)]
        self.head_outputs = {}          # (layer, head) -> [(step, norm)]
        self.attention_entropies = {}   # (layer, head) -> [(step, entropy)]
        self.samples = []               # (step, list_of_strings)
        self.custom = {}                # arbitrary named metrics

        # Trajectory storage (new)
        self.step_data = []             # list of per-step dicts

    # ------------------------------------------------------------------
    # Primary trajectory method
    # ------------------------------------------------------------------

    def record_step(self, step, loss, per_position_losses, snapshots):
        """
        Record rich per-step data from the training loop.

        Args:
            step: training step number
            loss: scalar loss value (float)
            per_position_losses: list of per-position loss floats
            snapshots: list of snapshot dicts from gpt(..., capture_state=True),
                       one per position. None if capture was not requested.
        """
        # Always record loss (backward compat)
        self.losses.append((step, loss))

        entry = {
            'step': step,
            'loss': loss,
            'per_position_losses': per_position_losses,
        }

        if snapshots is not None and self.detail_level != 'loss_only':
            n_layers = len(snapshots[0]['layers']) if snapshots else 0
            n_heads = len(snapshots[0]['layers'][0]['heads']) if snapshots and n_layers > 0 else 0

            # Aggregate per-head metrics across positions
            head_norms = {}   # (layer, head) -> [norm_per_pos]
            head_entropies = {}  # (layer, head) -> [entropy_per_pos]

            for snap in snapshots:
                for li, layer_data in enumerate(snap['layers']):
                    for hi, head_data in enumerate(layer_data['heads']):
                        key = (li, hi)
                        head_norms.setdefault(key, []).append(head_data['head_out_norm'])
                        head_entropies.setdefault(key, []).append(head_data['attn_entropy'])

            # Store per-head summary (mean across positions)
            entry['head_norms'] = {}
            entry['head_entropies'] = {}
            for key in head_norms:
                norms = head_norms[key]
                mean_norm = sum(norms) / len(norms)
                entry['head_norms'][key] = mean_norm
                # Also populate backward-compat storage
                self.head_outputs.setdefault(key, []).append((step, mean_norm))

                entropies = head_entropies[key]
                mean_entropy = sum(entropies) / len(entropies)
                entry['head_entropies'][key] = mean_entropy
                self.attention_entropies.setdefault(key, []).append((step, mean_entropy))

            if self.detail_level == 'full':
                entry['snapshots'] = snapshots

        self.step_data.append(entry)

    # ------------------------------------------------------------------
    # Backward-compatible methods
    # ------------------------------------------------------------------

    def record_loss(self, step, loss_value):
        self.losses.append((step, loss_value))

    def record_grad_norm(self, step, group_name, norm):
        self.grad_norms.setdefault(group_name, []).append((step, norm))

    def record_head_output(self, step, layer, head, values):
        """Record the L2 norm of a head's output."""
        norm = sum(v.data ** 2 for v in values) ** 0.5
        self.head_outputs.setdefault((layer, head), []).append((step, norm))

    def record_attn_entropy(self, step, layer, head, weights):
        """Shannon entropy of attention weight distribution."""
        entropy = -sum(
            w.data * math.log(w.data + 1e-10) for w in weights
        )
        self.attention_entropies.setdefault((layer, head), []).append((step, entropy))

    def record_custom(self, name, step, value):
        self.custom.setdefault(name, []).append((step, value))

    def record_samples(self, step, samples):
        self.samples.append((step, samples))

    def get_loss_values(self):
        """Return just the loss values as a list."""
        return [l for _, l in self.losses]

    # ------------------------------------------------------------------
    # Trajectory extraction methods
    # ------------------------------------------------------------------

    def get_head_norm_trajectory(self, layer, head):
        """Return [(step, mean_norm), ...] for a specific head."""
        key = (layer, head)
        if key in self.head_outputs:
            return list(self.head_outputs[key])
        return []

    def get_attn_entropy_trajectory(self, layer, head):
        """Return [(step, mean_entropy), ...] for a specific head."""
        key = (layer, head)
        if key in self.attention_entropies:
            return list(self.attention_entropies[key])
        return []

    def get_per_position_loss_trajectory(self):
        """Return [(step, [loss_pos0, loss_pos1, ...]), ...] from step_data."""
        return [
            (entry['step'], entry['per_position_losses'])
            for entry in self.step_data
            if 'per_position_losses' in entry
        ]

    def get_head_contribution_fractions(self):
        """
        Per-head fraction of total head norm, averaged over recorded steps.
        Returns dict of (layer, head) -> fraction.
        """
        if not self.head_outputs:
            return {}

        # Compute mean norm per head across all recorded steps
        mean_norms = {}
        for key, entries in self.head_outputs.items():
            norms = [n for _, n in entries]
            mean_norms[key] = sum(norms) / len(norms) if norms else 0.0

        total = sum(mean_norms.values())
        if total < 1e-10:
            return {k: 0.0 for k in mean_norms}
        return {k: v / total for k, v in mean_norms.items()}


# ============================================================================
# Model primitives — identical to microgpt
# ============================================================================

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def _shannon_entropy(weights_data):
    """Shannon entropy of a probability distribution (list of floats)."""
    return -sum(w * math.log(w + 1e-10) for w in weights_data)


# ============================================================================
# GPT forward pass with hook points
# ============================================================================

def gpt(token_id, pos_id, keys, values, state_dict, config, hooks=None,
        capture_state=False):
    """
    One step of the GPT forward pass (processes one token).

    When capture_state=False: returns logits (backward compatible).
    When capture_state=True: returns (logits, snapshot) where snapshot
    contains .data floats only (no Value objects / computation graph refs).

    Hook points (names used in hooks.apply):
        emb               - after token + position embedding
        pre_norm           - after initial rmsnorm
        qkv.{li}          - after Q, K, V projection for layer li
        attn_w.{li}.{h}   - attention weights for layer li, head h
        head_out.{li}.{h} - head output for layer li, head h
        post_attn.{li}    - after attention residual add
        mlp_hidden.{li}   - after fc1 + relu
        post_mlp.{li}     - after MLP residual add
        logits             - final logits before return

    If hooks is None, behaves identically to original microgpt.
    """
    if hooks is None:
        hooks = Hooks()

    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    head_dim = n_embd // n_head

    if capture_state:
        snapshot = {'emb': None, 'layers': []}

    # Embeddings
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = hooks.apply('emb', x)

    if capture_state:
        snapshot['emb'] = [v.data for v in x]

    x = rmsnorm(x)
    x = hooks.apply('pre_norm', x)

    for li in range(n_layer):
        if capture_state:
            layer_snap = {'heads': []}

        # --- Attention block ---
        x_residual = x
        x = rmsnorm(x)

        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])

        # Hook: after QKV projection
        qkv = hooks.apply(f'qkv.{li}', (q, k, v))
        if isinstance(qkv, tuple) and len(qkv) == 3:
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

            # Hook: head output (KEY for freeze experiments)
            head_out = hooks.apply(f'head_out.{li}.{h}', head_out)

            if capture_state:
                w_data = [w.data for w in attn_weights]
                layer_snap['heads'].append({
                    'attn_weights': w_data,
                    'attn_entropy': _shannon_entropy(w_data),
                    'head_out_norm': sum(v.data**2 for v in head_out)**0.5,
                    'head_out_vec': [v.data for v in head_out],
                })

            x_attn.extend(head_out)

        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        # Hook: post-attention residual
        x = hooks.apply(f'post_attn.{li}', x)

        if capture_state:
            layer_snap['post_attn_residual'] = [v.data for v in x]

        # --- MLP block ---
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]

        # Hook: MLP hidden
        x = hooks.apply(f'mlp_hidden.{li}', x)

        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

        # Hook: post-MLP residual
        x = hooks.apply(f'post_mlp.{li}', x)

        if capture_state:
            layer_snap['post_mlp_residual'] = [v.data for v in x]
            snapshot['layers'].append(layer_snap)

    logits = linear(x, state_dict['lm_head'])
    logits = hooks.apply('logits', logits)

    if capture_state:
        return logits, snapshot
    return logits


# ============================================================================
# Model initialization
# ============================================================================

def make_config(n_layer=4, n_embd=16, n_head=4, block_size=16, vocab_size=27):
    """Create a model config dict."""
    return {
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'head_dim': n_embd // n_head,
        'block_size': block_size,
        'vocab_size': vocab_size,
    }


def init_state_dict(config, seed=42):
    """Initialize model parameters. Returns (state_dict, params_list)."""
    rng = random.Random(seed)

    def matrix(nout, nin, std=0.08):
        return [[Value(rng.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

    sd = {
        'wte': matrix(config['vocab_size'], config['n_embd']),
        'wpe': matrix(config['block_size'], config['n_embd']),
        'lm_head': matrix(config['vocab_size'], config['n_embd']),
    }
    for i in range(config['n_layer']):
        sd[f'layer{i}.attn_wq'] = matrix(config['n_embd'], config['n_embd'])
        sd[f'layer{i}.attn_wk'] = matrix(config['n_embd'], config['n_embd'])
        sd[f'layer{i}.attn_wv'] = matrix(config['n_embd'], config['n_embd'])
        sd[f'layer{i}.attn_wo'] = matrix(config['n_embd'], config['n_embd'])
        sd[f'layer{i}.mlp_fc1'] = matrix(4 * config['n_embd'], config['n_embd'])
        sd[f'layer{i}.mlp_fc2'] = matrix(config['n_embd'], 4 * config['n_embd'])

    params = [p for mat in sd.values() for row in mat for p in row]
    return sd, params


# ============================================================================
# Dataset / Tokenizer
# ============================================================================

def load_dataset(path=None):
    """Load and prepare dataset. Returns (docs, uchars, BOS, vocab_size)."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'data', 'input.txt')
    if not os.path.exists(path):
        import urllib.request
        names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
        urllib.request.urlretrieve(names_url, path)

    docs = [line.strip() for line in open(path) if line.strip()]
    uchars = sorted(set(''.join(docs)))
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    return docs, uchars, BOS, vocab_size


def tokenize(doc, uchars, BOS):
    """Convert a document string to token ids."""
    return [BOS] + [uchars.index(ch) for ch in doc] + [BOS]


# ============================================================================
# Training loop
# ============================================================================

@dataclass
class TrainConfig:
    num_steps: int = 500
    learning_rate: float = 0.01
    beta1: float = 0.85
    beta2: float = 0.99
    eps_adam: float = 1e-8
    print_every: int = 50
    sample_every: int = 0        # 0 = never during training
    num_samples: int = 5
    temperature: float = 0.5
    detail_level: str = 'summary'  # 'full', 'summary', or 'loss_only'


def train(state_dict, params, config, train_config, docs, uchars, BOS,
          hooks=None, probe=None, grad_hooks=None, seed=42):
    """
    Train the model.

    Args:
        state_dict: model parameters
        params: flat list of all Value params
        config: model config dict
        train_config: TrainConfig
        docs: list of document strings
        uchars: character vocabulary
        BOS: BOS token id
        hooks: Hooks object for forward pass perturbations
        probe: Probe object for recording metrics
        grad_hooks: list of callables (params, step) called after backward, before optimizer
        seed: random seed for document order

    Returns:
        probe: the Probe object with all recorded metrics
    """
    tc = train_config

    if hooks is None:
        hooks = Hooks()
    if probe is None:
        probe = Probe(detail_level=tc.detail_level)
    rng = random.Random(seed)

    # Shuffle docs with this seed
    doc_order = list(range(len(docs)))
    rng.shuffle(doc_order)

    # Adam buffers
    m_buf = [0.0] * len(params)
    v_buf = [0.0] * len(params)

    n_layer = config['n_layer']
    block_size = config['block_size']

    for step in range(tc.num_steps):
        hooks.step = step

        # Pick document
        doc_idx = doc_order[step % len(doc_order)]
        doc = docs[doc_idx]
        tokens = tokenize(doc, uchars, BOS)
        n = min(block_size, len(tokens) - 1)

        # Decide whether to capture state this step
        capture_this_step = (
            probe.detail_level != 'loss_only'
            and step % max(1, probe.record_interval) == 0
        )

        # Forward pass
        keys = [[] for _ in range(n_layer)]
        vals = [[] for _ in range(n_layer)]
        losses = []
        per_position_losses = []
        snapshots = [] if capture_this_step else None

        for pos_id in range(n):
            token_id = tokens[pos_id]
            target_id = tokens[pos_id + 1]

            if capture_this_step:
                result = gpt(token_id, pos_id, keys, vals, state_dict, config,
                             hooks, capture_state=True)
                logits, snap = result
                snapshots.append(snap)
            else:
                logits = gpt(token_id, pos_id, keys, vals, state_dict, config, hooks)

            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
            per_position_losses.append(loss_t.data)

        loss = (1 / n) * sum(losses)

        # Record via record_step (handles loss + snapshots + per-position losses)
        if capture_this_step:
            probe.record_step(step, loss.data, per_position_losses, snapshots)
        else:
            probe.record_loss(step, loss.data)

        # Backward
        loss.backward()

        # Gradient hooks (for freeze, noise injection on gradients, etc.)
        if grad_hooks:
            for gh in grad_hooks:
                gh(params, state_dict, step)

        # Record gradient norms per layer
        if step % max(1, probe.record_interval) == 0:
            _record_grad_norms(probe, state_dict, config, step)

        # Adam optimizer update
        lr_t = tc.learning_rate * (1 - step / tc.num_steps)
        for i, p in enumerate(params):
            m_buf[i] = tc.beta1 * m_buf[i] + (1 - tc.beta1) * p.grad
            v_buf[i] = tc.beta2 * v_buf[i] + (1 - tc.beta2) * p.grad ** 2
            m_hat = m_buf[i] / (1 - tc.beta1 ** (step + 1))
            v_hat = v_buf[i] / (1 - tc.beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + tc.eps_adam)
            p.grad = 0

        # Print progress
        if tc.print_every > 0 and (step + 1) % tc.print_every == 0:
            print(f"step {step+1:4d}/{tc.num_steps} | loss {loss.data:.4f}")

        # Sample during training
        if tc.sample_every > 0 and (step + 1) % tc.sample_every == 0:
            samples = generate(state_dict, config, uchars, BOS,
                               num_samples=tc.num_samples,
                               temperature=tc.temperature)
            probe.record_samples(step, samples)

    return probe


def _record_grad_norms(probe, state_dict, config, step):
    """Record per-layer gradient norms."""
    for li in range(config['n_layer']):
        for comp in ['attn_wq', 'attn_wk', 'attn_wv', 'attn_wo', 'mlp_fc1', 'mlp_fc2']:
            key = f'layer{li}.{comp}'
            if key in state_dict:
                norm = sum(p.grad ** 2 for row in state_dict[key] for p in row) ** 0.5
                probe.record_grad_norm(step, key, norm)

    # Embeddings and head
    for key in ['wte', 'wpe', 'lm_head']:
        if key in state_dict:
            norm = sum(p.grad ** 2 for row in state_dict[key] for p in row) ** 0.5
            probe.record_grad_norm(step, key, norm)


# ============================================================================
# Inference / Generation
# ============================================================================

def generate(state_dict, config, uchars, BOS, num_samples=10,
             temperature=0.5, max_len=None, hooks=None, seed=None):
    """Generate samples from the model."""
    if hooks is None:
        hooks = Hooks()
    if max_len is None:
        max_len = config['block_size']
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    n_layer = config['n_layer']
    vocab_size = config['vocab_size']
    samples = []

    for _ in range(num_samples):
        keys = [[] for _ in range(n_layer)]
        vals = [[] for _ in range(n_layer)]
        token_id = BOS
        sample = []

        for pos_id in range(max_len):
            logits = gpt(token_id, pos_id, keys, vals, state_dict, config, hooks)
            probs = softmax([l / temperature for l in logits])
            weights = [p.data for p in probs]
            token_id = rng.choices(range(vocab_size), weights=weights)[0]
            if token_id == BOS:
                break
            if token_id < len(uchars):
                sample.append(uchars[token_id])

        samples.append(''.join(sample))

    return samples


# ============================================================================
# Main — run baseline (no hooks)
# ============================================================================

if __name__ == '__main__':
    print("=== MorphoGPT ===")
    print("Loading dataset...")
    docs, uchars, BOS, vocab_size = load_dataset()
    print(f"  docs: {len(docs)}, vocab: {vocab_size}")

    # Default: n_layer=4 for meaningful experiments
    config = make_config(n_layer=4, n_embd=16, n_head=4, vocab_size=vocab_size)
    print(f"  config: n_layer={config['n_layer']}, n_embd={config['n_embd']}, "
          f"n_head={config['n_head']}, block_size={config['block_size']}")

    state_dict, params = init_state_dict(config, seed=42)
    print(f"  params: {len(params)}")

    tc = TrainConfig(num_steps=500, print_every=50)
    print(f"\nTraining for {tc.num_steps} steps (baseline, no hooks)...\n")

    probe = train(state_dict, params, config, tc, docs, uchars, BOS)

    print(f"\n--- Inference (temperature={tc.temperature}) ---")
    samples = generate(state_dict, config, uchars, BOS,
                       num_samples=20, temperature=tc.temperature, seed=123)
    for i, s in enumerate(samples):
        print(f"  {i+1:2d}: {s}")

    print(f"\nFinal loss: {probe.losses[-1][1]:.4f}")
    print(f"Loss trajectory: {len(probe.losses)} points recorded")
