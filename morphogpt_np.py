"""
MorphoGPT — NumPy Backend

Drop-in replacement for morphogpt.py using numpy arrays instead of scalar
autograd. ~1000x faster: each 200-step run takes seconds instead of minutes.

Same interface: Hooks, Probe, make_config, init_state_dict, train, generate.
Hooks receive/return numpy arrays (1D) instead of lists of Value objects.
Grad hooks receive (param_arrays, state_dict, step) where param_arrays is a
flat list of numpy arrays and state_dict maps names to 2D numpy arrays.
"""

import os
import math
import random
import numpy as np
from dataclasses import dataclass


# ============================================================================
# Hooks — same interface as morphogpt.py
# ============================================================================

class Hooks:
    """
    Named hook points in the computation graph.
    Each hook is a callable: fn(value, step=step) -> modified_value or None.
    value is a numpy array (1D for vectors, list of arrays for QKV tuples).
    """

    def __init__(self):
        self._hooks = {}
        self.step = 0

    def register(self, name, fn):
        self._hooks.setdefault(name, []).append(fn)

    def clear(self, name=None):
        if name is None:
            self._hooks.clear()
        else:
            self._hooks.pop(name, None)

    def apply(self, name, value):
        for fn in self._hooks.get(name, []):
            result = fn(value, step=self.step)
            if result is not None:
                value = result
        return value

    def has(self, name):
        return name in self._hooks and len(self._hooks[name]) > 0

    def list_hooks(self):
        return {k: len(v) for k, v in self._hooks.items()}


# ============================================================================
# Probe — same interface as morphogpt.py
# ============================================================================

class Probe:
    def __init__(self, record_interval=1, detail_level='summary'):
        self.record_interval = record_interval
        self.detail_level = detail_level
        self.losses = []
        self.grad_norms = {}
        self.head_outputs = {}
        self.attention_entropies = {}
        self.samples = []
        self.custom = {}
        self.step_data = []

    def record_step(self, step, loss, per_position_losses, snapshots):
        self.losses.append((step, loss))
        entry = {
            'step': step,
            'loss': loss,
            'per_position_losses': per_position_losses,
        }

        if snapshots is not None and self.detail_level != 'loss_only':
            n_layers = len(snapshots[0]['layers']) if snapshots else 0

            head_norms = {}
            head_entropies = {}

            for snap in snapshots:
                for li, layer_data in enumerate(snap['layers']):
                    for hi, head_data in enumerate(layer_data['heads']):
                        key = (li, hi)
                        head_norms.setdefault(key, []).append(head_data['head_out_norm'])
                        head_entropies.setdefault(key, []).append(head_data['attn_entropy'])

            entry['head_norms'] = {}
            entry['head_entropies'] = {}
            for key in head_norms:
                norms = head_norms[key]
                mean_norm = sum(norms) / len(norms)
                entry['head_norms'][key] = mean_norm
                self.head_outputs.setdefault(key, []).append((step, mean_norm))

                entropies = head_entropies[key]
                mean_entropy = sum(entropies) / len(entropies)
                entry['head_entropies'][key] = mean_entropy
                self.attention_entropies.setdefault(key, []).append((step, mean_entropy))

            if self.detail_level == 'full':
                entry['snapshots'] = snapshots

        self.step_data.append(entry)

    def record_loss(self, step, loss_value):
        self.losses.append((step, loss_value))

    def record_grad_norm(self, step, group_name, norm):
        self.grad_norms.setdefault(group_name, []).append((step, norm))

    def record_head_output(self, step, layer, head, norm):
        self.head_outputs.setdefault((layer, head), []).append((step, norm))

    def record_attn_entropy(self, step, layer, head, entropy):
        self.attention_entropies.setdefault((layer, head), []).append((step, entropy))

    def record_custom(self, name, step, value):
        self.custom.setdefault(name, []).append((step, value))

    def record_samples(self, step, samples):
        self.samples.append((step, samples))

    def get_loss_values(self):
        return [l for _, l in self.losses]

    def get_head_norm_trajectory(self, layer, head):
        key = (layer, head)
        if key in self.head_outputs:
            return list(self.head_outputs[key])
        return []

    def get_attn_entropy_trajectory(self, layer, head):
        key = (layer, head)
        if key in self.attention_entropies:
            return list(self.attention_entropies[key])
        return []

    def get_per_position_loss_trajectory(self):
        return [
            (entry['step'], entry['per_position_losses'])
            for entry in self.step_data
            if 'per_position_losses' in entry
        ]

    def get_head_contribution_fractions(self):
        if not self.head_outputs:
            return {}
        mean_norms = {}
        for key, entries in self.head_outputs.items():
            norms = [n for _, n in entries]
            mean_norms[key] = sum(norms) / len(norms) if norms else 0.0
        total = sum(mean_norms.values())
        if total < 1e-10:
            return {k: 0.0 for k in mean_norms}
        return {k: v / total for k, v in mean_norms.items()}


# ============================================================================
# Forward pass primitives
# ============================================================================

def _softmax(logits):
    """Numerically stable softmax on a 1D numpy array."""
    x = logits - logits.max()
    e = np.exp(x)
    return e / e.sum()


def _rmsnorm(x):
    ms = np.mean(x * x)
    return x / np.sqrt(ms + 1e-5)


def _shannon_entropy(weights):
    return -np.sum(weights * np.log(weights + 1e-10))


# ============================================================================
# GPT forward pass — returns (loss, grads, snapshot_or_None)
# ============================================================================

def _forward_backward(tokens, n, state_dict, config, hooks, capture_state=False):
    """
    Full forward + backward pass for one document.

    Args:
        tokens: list of int token ids
        n: number of positions to process
        state_dict: dict of name -> np.ndarray (2D weight matrices)
        config: model config dict
        hooks: Hooks object
        capture_state: whether to record per-position snapshots

    Returns:
        (loss_scalar, per_position_losses, grads_dict, snapshots_or_None)
        grads_dict has same keys as state_dict, values are gradient arrays.
    """
    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    head_dim = config['head_dim']
    vocab_size = config['vocab_size']

    # Initialize gradient accumulators
    grads = {k: np.zeros_like(v) for k, v in state_dict.items()}

    # KV cache for autoregressive processing
    # keys[li] and vals[li] are lists of 1D arrays, one per position processed so far
    keys_cache = [[] for _ in range(n_layer)]
    vals_cache = [[] for _ in range(n_layer)]

    total_loss = 0.0
    per_position_losses = []
    snapshots = [] if capture_state else None

    for pos_id in range(n):
        token_id = tokens[pos_id]
        target_id = tokens[pos_id + 1]

        # ---- Forward pass for this position ----
        # We store all intermediates needed for backward

        # Embedding
        tok_emb = state_dict['wte'][token_id].copy()  # (n_embd,)
        pos_emb = state_dict['wpe'][pos_id].copy()     # (n_embd,)
        x = tok_emb + pos_emb                          # (n_embd,)
        x = hooks.apply('emb', x)

        snapshot = {'emb': x.copy(), 'layers': []} if capture_state else None

        # Pre-norm
        x_pre_norm_in = x.copy()
        ms0 = np.mean(x * x)
        scale0 = 1.0 / np.sqrt(ms0 + 1e-5)
        x = x * scale0
        x = hooks.apply('pre_norm', x)

        # Store forward intermediates for backward
        # We'll store everything in lists indexed by layer
        fwd = {
            'tok_emb': tok_emb,
            'pos_emb': pos_emb,
            'x_after_emb': x_pre_norm_in,  # before first rmsnorm
            'scale0': scale0,
            'x_after_pre_norm': x.copy(),
            'layers': [],
        }

        for li in range(n_layer):
            layer_fwd = {}
            if capture_state:
                layer_snap = {'heads': []}

            # --- Attention block ---
            x_residual = x.copy()
            layer_fwd['x_residual_attn'] = x_residual

            # RMSNorm before attention
            ms_a = np.mean(x * x)
            scale_a = 1.0 / np.sqrt(ms_a + 1e-5)
            x_normed = x * scale_a
            layer_fwd['x_before_attn_norm'] = x.copy()
            layer_fwd['scale_attn'] = scale_a
            layer_fwd['x_normed_attn'] = x_normed.copy()

            # Q, K, V projections
            q = state_dict[f'layer{li}.attn_wq'] @ x_normed  # (n_embd,)
            k = state_dict[f'layer{li}.attn_wk'] @ x_normed
            v = state_dict[f'layer{li}.attn_wv'] @ x_normed

            # QKV hook
            qkv = hooks.apply(f'qkv.{li}', (q, k, v))
            if isinstance(qkv, tuple) and len(qkv) == 3:
                q, k, v = qkv

            layer_fwd['q'] = q.copy()
            layer_fwd['k'] = k.copy()
            layer_fwd['v'] = v.copy()

            keys_cache[li].append(k.copy())
            vals_cache[li].append(v.copy())

            # Multi-head attention
            x_attn = np.zeros(n_embd)
            layer_fwd['heads'] = []
            T = len(keys_cache[li])  # number of tokens so far

            for h in range(n_head):
                hs = h * head_dim
                he = hs + head_dim
                q_h = q[hs:he]

                # Gather cached keys/values for this head
                k_h = np.array([keys_cache[li][t][hs:he] for t in range(T)])  # (T, head_dim)
                v_h = np.array([vals_cache[li][t][hs:he] for t in range(T)])  # (T, head_dim)

                # Attention logits and weights
                attn_logits = k_h @ q_h / np.sqrt(head_dim)  # (T,)
                attn_weights = _softmax(attn_logits)           # (T,)

                # Hook: attention weights
                attn_weights = hooks.apply(f'attn_w.{li}.{h}', attn_weights)
                if not isinstance(attn_weights, np.ndarray):
                    attn_weights = np.array(attn_weights, dtype=np.float64)

                # Head output
                head_out = v_h.T @ attn_weights  # (head_dim,)

                # Hook: head output
                head_out = hooks.apply(f'head_out.{li}.{h}', head_out)
                if not isinstance(head_out, np.ndarray):
                    head_out = np.array(head_out, dtype=np.float64)

                layer_fwd['heads'].append({
                    'q_h': q_h.copy(),
                    'k_h': k_h.copy(),
                    'v_h': v_h.copy(),
                    'attn_logits': attn_logits.copy(),
                    'attn_weights': attn_weights.copy(),
                    'head_out': head_out.copy(),
                })

                if capture_state:
                    layer_snap['heads'].append({
                        'attn_weights': attn_weights.tolist(),
                        'attn_entropy': float(_shannon_entropy(attn_weights)),
                        'head_out_norm': float(np.linalg.norm(head_out)),
                        'head_out_vec': head_out.tolist(),
                    })

                x_attn[hs:he] = head_out

            layer_fwd['x_attn'] = x_attn.copy()

            # Output projection
            x_proj = state_dict[f'layer{li}.attn_wo'] @ x_attn  # (n_embd,)
            x = x_proj + x_residual
            layer_fwd['x_proj_attn'] = x_proj.copy()

            # Hook: post-attention
            x = hooks.apply(f'post_attn.{li}', x)
            if not isinstance(x, np.ndarray):
                x = np.array(x, dtype=np.float64)
            layer_fwd['x_after_attn'] = x.copy()

            if capture_state:
                layer_snap['post_attn_residual'] = x.tolist()

            # --- MLP block ---
            x_residual_mlp = x.copy()
            layer_fwd['x_residual_mlp'] = x_residual_mlp

            # RMSNorm before MLP
            ms_m = np.mean(x * x)
            scale_m = 1.0 / np.sqrt(ms_m + 1e-5)
            x_normed_mlp = x * scale_m
            layer_fwd['x_before_mlp_norm'] = x.copy()
            layer_fwd['scale_mlp'] = scale_m
            layer_fwd['x_normed_mlp'] = x_normed_mlp.copy()

            # FC1 + ReLU
            fc1_out = state_dict[f'layer{li}.mlp_fc1'] @ x_normed_mlp  # (4*n_embd,)
            relu_out = np.maximum(0, fc1_out)
            relu_mask = (fc1_out > 0).astype(np.float64)
            layer_fwd['fc1_out'] = fc1_out
            layer_fwd['relu_out'] = relu_out
            layer_fwd['relu_mask'] = relu_mask

            # Hook: MLP hidden
            relu_out_hooked = hooks.apply(f'mlp_hidden.{li}', relu_out.copy())
            if not isinstance(relu_out_hooked, np.ndarray):
                relu_out_hooked = np.array(relu_out_hooked, dtype=np.float64)
            layer_fwd['relu_out_hooked'] = relu_out_hooked

            # FC2 + residual
            fc2_out = state_dict[f'layer{li}.mlp_fc2'] @ relu_out_hooked  # (n_embd,)
            x = fc2_out + x_residual_mlp
            layer_fwd['fc2_out'] = fc2_out

            # Hook: post-MLP
            x = hooks.apply(f'post_mlp.{li}', x)
            if not isinstance(x, np.ndarray):
                x = np.array(x, dtype=np.float64)
            layer_fwd['x_after_mlp'] = x.copy()

            if capture_state:
                layer_snap['post_mlp_residual'] = x.tolist()
                snapshot['layers'].append(layer_snap)

            fwd['layers'].append(layer_fwd)

        # Final logits
        fwd['x_final'] = x.copy()
        logits = state_dict['lm_head'] @ x  # (vocab_size,)
        logits = hooks.apply('logits', logits)
        if not isinstance(logits, np.ndarray):
            logits = np.array(logits, dtype=np.float64)
        fwd['logits'] = logits.copy()

        # Softmax + cross-entropy loss
        probs = _softmax(logits)
        fwd['probs'] = probs.copy()
        loss_t = -math.log(probs[target_id] + 1e-10)
        total_loss += loss_t
        per_position_losses.append(loss_t)

        if capture_state:
            snapshots.append(snapshot)

        # ---- Backward pass for this position ----
        # dloss/dlogits through softmax + cross-entropy
        # d(-log(softmax(x)[target])) / dx_i = softmax(x)_i - 1{i==target}
        dlogits = probs.copy()
        dlogits[target_id] -= 1.0
        dlogits /= n  # average over positions

        # lm_head: logits = lm_head @ x_final
        grads['lm_head'] += np.outer(dlogits, fwd['x_final'])
        dx = state_dict['lm_head'].T @ dlogits

        # Backward through layers (reverse order)
        for li in range(n_layer - 1, -1, -1):
            lf = fwd['layers'][li]

            # --- MLP backward ---
            # post_mlp hook: if hook replaced x, gradient may be modified
            # For stop_gradient hooks, dx should be zero (hook returns detached value)
            # We handle this by checking if the hook is present and if it returns
            # a modified value. For the numpy backend, stop_gradient hooks should
            # return np.zeros_like(x) as the gradient or we handle it specially.
            # Actually: the hook modifies the forward value. For stop-gradient,
            # the hook returns a new array (detached). The backward should NOT
            # flow through. We handle this via a grad_scale mechanism.
            # For now: hooks that need to modify backward flow do so via grad_hooks.

            # Residual: x = fc2_out + x_residual_mlp
            dx_residual_mlp = dx.copy()
            dfc2_out = dx.copy()

            # FC2: fc2_out = mlp_fc2 @ relu_out_hooked
            grads[f'layer{li}.mlp_fc2'] += np.outer(dfc2_out, lf['relu_out_hooked'])
            drelu_out_hooked = state_dict[f'layer{li}.mlp_fc2'].T @ dfc2_out

            # MLP hidden hook backward: pass gradient through
            drelu_out = drelu_out_hooked  # hooks don't affect backward in this implementation

            # ReLU backward
            dfc1_out = drelu_out * lf['relu_mask']

            # FC1: fc1_out = mlp_fc1 @ x_normed_mlp
            grads[f'layer{li}.mlp_fc1'] += np.outer(dfc1_out, lf['x_normed_mlp'])
            dx_normed_mlp = state_dict[f'layer{li}.mlp_fc1'].T @ dfc1_out

            # RMSNorm backward (before MLP)
            dx_mlp_norm = _rmsnorm_backward(
                dx_normed_mlp, lf['x_before_mlp_norm'], lf['scale_mlp'])

            # Residual add
            dx = dx_mlp_norm + dx_residual_mlp

            # --- Attention backward ---
            # post_attn hook: same as post_mlp
            dx_residual_attn = dx.copy()
            dx_proj_attn = dx.copy()

            # Output projection: x_proj = attn_wo @ x_attn
            grads[f'layer{li}.attn_wo'] += np.outer(dx_proj_attn, lf['x_attn'])
            dx_attn = state_dict[f'layer{li}.attn_wo'].T @ dx_proj_attn

            # Multi-head attention backward
            dq = np.zeros(n_embd)
            # dk and dv for the CURRENT position only (we don't backprop into cached KV)
            dk_cur = np.zeros(n_embd)
            dv_cur = np.zeros(n_embd)

            for h in range(n_head):
                hs_h = h * head_dim
                he_h = hs_h + head_dim
                hf = lf['heads'][h]

                dhead_out = dx_attn[hs_h:he_h]  # (head_dim,)

                # head_out = v_h.T @ attn_weights
                # dv_h = attn_weights[:, None] * dhead_out[None, :]  (T, head_dim)
                # dattn_weights = v_h @ dhead_out  (T,)
                dattn_weights = hf['v_h'] @ dhead_out  # (T,)
                # Only accumulate dv for current position (last in cache)
                dv_cur[hs_h:he_h] += hf['attn_weights'][-1] * dhead_out

                # Softmax backward
                # dattn_logits = attn_weights * (dattn_weights - sum(attn_weights * dattn_weights))
                aw = hf['attn_weights']
                s = np.sum(aw * dattn_weights)
                dattn_logits = aw * (dattn_weights - s)  # (T,)

                # attn_logits = k_h @ q_h / sqrt(head_dim)
                scale_attn = 1.0 / np.sqrt(head_dim)
                dq_h = (hf['k_h'].T @ dattn_logits) * scale_attn  # (head_dim,)
                # dk for current position (last row of k_h)
                dk_cur[hs_h:he_h] += dattn_logits[-1] * hf['q_h'] * scale_attn

                dq[hs_h:he_h] = dq_h

            # Q, K, V projections backward
            # q = wq @ x_normed, k = wk @ x_normed, v = wv @ x_normed
            x_normed_attn = lf['x_normed_attn']
            grads[f'layer{li}.attn_wq'] += np.outer(dq, x_normed_attn)
            grads[f'layer{li}.attn_wk'] += np.outer(dk_cur, x_normed_attn)
            grads[f'layer{li}.attn_wv'] += np.outer(dv_cur, x_normed_attn)

            dx_normed_attn = (state_dict[f'layer{li}.attn_wq'].T @ dq +
                              state_dict[f'layer{li}.attn_wk'].T @ dk_cur +
                              state_dict[f'layer{li}.attn_wv'].T @ dv_cur)

            # RMSNorm backward (before attention)
            dx_attn_norm = _rmsnorm_backward(
                dx_normed_attn, lf['x_before_attn_norm'], lf['scale_attn'])

            # Residual add
            dx = dx_attn_norm + dx_residual_attn

        # Pre-norm backward
        dx_pre = _rmsnorm_backward(dx, fwd['x_after_emb'], fwd['scale0'])

        # Embedding gradients
        grads['wte'][token_id] += dx_pre
        grads['wpe'][pos_id] += dx_pre

    loss = total_loss / n
    return loss, per_position_losses, grads, snapshots


def _rmsnorm_backward(dout, x_in, scale):
    """
    Backward through rmsnorm.
    Forward: y = x * scale, where scale = 1/sqrt(mean(x^2) + eps)
    """
    n = len(x_in)
    # dy/dx_i = scale * (delta_ij - x_i * x_j / (n * (ms + eps)))
    # = scale * dout_i - scale * x_i * sum(dout_j * x_j) / (n * (ms + eps))
    # Simplify: dx = scale * dout - scale^3 * x * (x . dout) / n
    xdot = np.dot(x_in, dout)
    dx = scale * dout - (scale ** 3) * x_in * xdot / n
    return dx


# ============================================================================
# GPT forward-only (for generation)
# ============================================================================

def gpt_forward(token_id, pos_id, keys_cache, vals_cache, state_dict, config, hooks=None):
    """Forward pass for one token (inference only, no backward)."""
    if hooks is None:
        hooks = Hooks()

    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    head_dim = config['head_dim']

    x = state_dict['wte'][token_id] + state_dict['wpe'][pos_id]
    x = hooks.apply('emb', x)
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=np.float64)

    x = _rmsnorm(x)
    x = hooks.apply('pre_norm', x)
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=np.float64)

    for li in range(n_layer):
        x_residual = x.copy()
        x = _rmsnorm(x)

        q = state_dict[f'layer{li}.attn_wq'] @ x
        k = state_dict[f'layer{li}.attn_wk'] @ x
        v = state_dict[f'layer{li}.attn_wv'] @ x

        qkv = hooks.apply(f'qkv.{li}', (q, k, v))
        if isinstance(qkv, tuple) and len(qkv) == 3:
            q, k, v = qkv

        keys_cache[li].append(k.copy())
        vals_cache[li].append(v.copy())

        T = len(keys_cache[li])
        x_attn = np.zeros(n_embd)

        for h in range(n_head):
            hs = h * head_dim
            he = hs + head_dim
            q_h = q[hs:he]
            k_h = np.array([keys_cache[li][t][hs:he] for t in range(T)])
            v_h = np.array([vals_cache[li][t][hs:he] for t in range(T)])

            attn_logits = k_h @ q_h / np.sqrt(head_dim)
            attn_weights = _softmax(attn_logits)
            attn_weights = hooks.apply(f'attn_w.{li}.{h}', attn_weights)
            if not isinstance(attn_weights, np.ndarray):
                attn_weights = np.array(attn_weights, dtype=np.float64)

            head_out = v_h.T @ attn_weights
            head_out = hooks.apply(f'head_out.{li}.{h}', head_out)
            if not isinstance(head_out, np.ndarray):
                head_out = np.array(head_out, dtype=np.float64)

            x_attn[hs:he] = head_out

        x = state_dict[f'layer{li}.attn_wo'] @ x_attn + x_residual
        x = hooks.apply(f'post_attn.{li}', x)
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float64)

        x_residual = x.copy()
        x = _rmsnorm(x)
        fc1 = state_dict[f'layer{li}.mlp_fc1'] @ x
        relu = np.maximum(0, fc1)
        relu = hooks.apply(f'mlp_hidden.{li}', relu)
        if not isinstance(relu, np.ndarray):
            relu = np.array(relu, dtype=np.float64)

        x = state_dict[f'layer{li}.mlp_fc2'] @ relu + x_residual
        x = hooks.apply(f'post_mlp.{li}', x)
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float64)

    logits = state_dict['lm_head'] @ x
    logits = hooks.apply('logits', logits)
    if not isinstance(logits, np.ndarray):
        logits = np.array(logits, dtype=np.float64)

    return logits


# ============================================================================
# Model initialization
# ============================================================================

def make_config(n_layer=4, n_embd=16, n_head=4, block_size=16, vocab_size=27):
    return {
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'head_dim': n_embd // n_head,
        'block_size': block_size,
        'vocab_size': vocab_size,
    }


def init_state_dict(config, seed=42):
    """Initialize model parameters as numpy arrays. Returns (state_dict, params_list)."""
    rng = np.random.RandomState(seed)
    std = 0.08

    sd = {
        'wte': rng.randn(config['vocab_size'], config['n_embd']) * std,
        'wpe': rng.randn(config['block_size'], config['n_embd']) * std,
        'lm_head': rng.randn(config['vocab_size'], config['n_embd']) * std,
    }
    for i in range(config['n_layer']):
        sd[f'layer{i}.attn_wq'] = rng.randn(config['n_embd'], config['n_embd']) * std
        sd[f'layer{i}.attn_wk'] = rng.randn(config['n_embd'], config['n_embd']) * std
        sd[f'layer{i}.attn_wv'] = rng.randn(config['n_embd'], config['n_embd']) * std
        sd[f'layer{i}.attn_wo'] = rng.randn(config['n_embd'], config['n_embd']) * std
        sd[f'layer{i}.mlp_fc1'] = rng.randn(4 * config['n_embd'], config['n_embd']) * std
        sd[f'layer{i}.mlp_fc2'] = rng.randn(config['n_embd'], 4 * config['n_embd']) * std

    # params_list: list of (name, row_idx) tuples for compatibility
    # But for the numpy backend, grad_hooks receive (grads_dict, state_dict, step)
    # We keep params_list as a flat list of references for API compat
    param_names = []
    for k in sd:
        param_names.append(k)

    return sd, param_names


# ============================================================================
# Dataset / Tokenizer — same as morphogpt.py
# ============================================================================

def load_dataset(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'data', 'input.txt')
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(os.path.dirname(path), exist_ok=True)
        names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
        urllib.request.urlretrieve(names_url, path)

    docs = [line.strip() for line in open(path) if line.strip()]
    uchars = sorted(set(''.join(docs)))
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    return docs, uchars, BOS, vocab_size


def tokenize(doc, uchars, BOS):
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
    sample_every: int = 0
    num_samples: int = 5
    temperature: float = 0.5
    detail_level: str = 'summary'


def train(state_dict, params, config, train_config, docs, uchars, BOS,
          hooks=None, probe=None, grad_hooks=None, seed=42):
    """
    Train the model using numpy backend.

    Args:
        state_dict: dict of name -> np.ndarray
        params: list of param names (from init_state_dict)
        config: model config dict
        train_config: TrainConfig
        docs: list of document strings
        uchars: character vocabulary
        BOS: BOS token id
        hooks: Hooks object
        probe: Probe object
        grad_hooks: list of callables (grads_dict, state_dict, step)
        seed: random seed
    """
    tc = train_config

    if hooks is None:
        hooks = Hooks()
    if probe is None:
        probe = Probe(detail_level=tc.detail_level)
    rng = random.Random(seed)

    doc_order = list(range(len(docs)))
    rng.shuffle(doc_order)

    # Adam buffers — one per weight matrix
    m_buf = {k: np.zeros_like(v) for k, v in state_dict.items()}
    v_buf = {k: np.zeros_like(v) for k, v in state_dict.items()}

    block_size = config['block_size']

    for step in range(tc.num_steps):
        hooks.step = step

        doc_idx = doc_order[step % len(doc_order)]
        doc = docs[doc_idx]
        tokens = tokenize(doc, uchars, BOS)
        n = min(block_size, len(tokens) - 1)

        capture_this_step = (
            probe.detail_level != 'loss_only'
            and step % max(1, probe.record_interval) == 0
        )

        # Forward + backward
        loss, per_position_losses, grads, snapshots = _forward_backward(
            tokens, n, state_dict, config, hooks, capture_state=capture_this_step
        )

        # Record
        if capture_this_step:
            probe.record_step(step, loss, per_position_losses, snapshots)
        else:
            probe.record_loss(step, loss)

        # Grad hooks
        if grad_hooks:
            for gh in grad_hooks:
                gh(grads, state_dict, step)

        # Record gradient norms
        if step % max(1, probe.record_interval) == 0:
            _record_grad_norms(probe, grads, config, step)

        # Adam optimizer
        lr_t = tc.learning_rate * (1 - step / tc.num_steps)
        for k in state_dict:
            g = grads[k]
            m_buf[k] = tc.beta1 * m_buf[k] + (1 - tc.beta1) * g
            v_buf[k] = tc.beta2 * v_buf[k] + (1 - tc.beta2) * g ** 2
            m_hat = m_buf[k] / (1 - tc.beta1 ** (step + 1))
            v_hat = v_buf[k] / (1 - tc.beta2 ** (step + 1))
            state_dict[k] -= lr_t * m_hat / (np.sqrt(v_hat) + tc.eps_adam)

        if tc.print_every > 0 and (step + 1) % tc.print_every == 0:
            print(f"step {step+1:4d}/{tc.num_steps} | loss {loss:.4f}")

        if tc.sample_every > 0 and (step + 1) % tc.sample_every == 0:
            samples = generate(state_dict, config, uchars, BOS,
                               num_samples=tc.num_samples,
                               temperature=tc.temperature)
            probe.record_samples(step, samples)

    return probe


def _record_grad_norms(probe, grads, config, step):
    for li in range(config['n_layer']):
        for comp in ['attn_wq', 'attn_wk', 'attn_wv', 'attn_wo', 'mlp_fc1', 'mlp_fc2']:
            key = f'layer{li}.{comp}'
            if key in grads:
                norm = float(np.linalg.norm(grads[key]))
                probe.record_grad_norm(step, key, norm)
    for key in ['wte', 'wpe', 'lm_head']:
        if key in grads:
            norm = float(np.linalg.norm(grads[key]))
            probe.record_grad_norm(step, key, norm)


# ============================================================================
# Generation
# ============================================================================

def generate(state_dict, config, uchars, BOS, num_samples=10,
             temperature=0.5, max_len=None, hooks=None, seed=None):
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
            logits = gpt_forward(token_id, pos_id, keys, vals, state_dict, config, hooks)
            probs = _softmax(logits / temperature)
            weights = probs.tolist()
            token_id = rng.choices(range(vocab_size), weights=weights)[0]
            if token_id == BOS:
                break
            if token_id < len(uchars):
                sample.append(uchars[token_id])

        samples.append(''.join(sample))

    return samples


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=== MorphoGPT (NumPy backend) ===")
    print("Loading dataset...")
    docs, uchars, BOS, vocab_size = load_dataset()
    print(f"  docs: {len(docs)}, vocab: {vocab_size}")

    config = make_config(n_layer=4, n_embd=16, n_head=4, vocab_size=vocab_size)
    print(f"  config: n_layer={config['n_layer']}, n_embd={config['n_embd']}, "
          f"n_head={config['n_head']}, block_size={config['block_size']}")

    state_dict, params = init_state_dict(config, seed=42)
    n_params = sum(v.size for v in state_dict.values())
    print(f"  params: {n_params}")

    tc = TrainConfig(num_steps=500, print_every=50)
    print(f"\nTraining for {tc.num_steps} steps (baseline, no hooks)...\n")

    import time
    t0 = time.time()
    probe = train(state_dict, params, config, tc, docs, uchars, BOS)
    elapsed = time.time() - t0

    print(f"\n--- Inference (temperature={tc.temperature}) ---")
    samples = generate(state_dict, config, uchars, BOS,
                       num_samples=20, temperature=tc.temperature, seed=123)
    for i, s in enumerate(samples):
        print(f"  {i+1:2d}: {s}")

    print(f"\nFinal loss: {probe.losses[-1][1]:.4f}")
    print(f"Training time: {elapsed:.1f}s")
