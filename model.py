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
        self.val_losses = []
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
    """Numerically stable softmax. Works on 1D or last axis of ND."""
    if logits.ndim == 1:
        x = logits - logits.max()
        e = np.exp(x)
        return e / e.sum()
    # ND: softmax along last axis
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _rmsnorm(x):
    """RMSNorm. Works on 1D vector or (n, d) batch."""
    if x.ndim == 1:
        ms = np.mean(x * x)
        return x / np.sqrt(ms + 1e-5)
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(ms + 1e-5)


def _shannon_entropy(weights):
    return -np.sum(weights * np.log(weights + 1e-10))


# ============================================================================
# GPT forward pass — fully vectorized across positions
# ============================================================================

def _forward_backward(tokens, n, state_dict, config, hooks, capture_state=False,
                      local_loss=False):
    """
    Full forward + backward pass for one document.
    Vectorized across all n positions simultaneously.

    If local_loss=True, each layer receives its own local loss signal
    (projected through lm_head) instead of end-to-end backpropagation.

    Returns:
        (loss_scalar, per_position_losses, grads_dict, snapshots_or_None)
    """
    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    head_dim = config['head_dim']
    vocab_size = config['vocab_size']
    sd = state_dict

    grads = {k: np.zeros_like(v) for k, v in sd.items()}

    token_ids = tokens[:n]
    target_ids = tokens[1:n+1]

    # Check which hook groups are active (to skip per-position loops when possible)
    has_fwd_hooks = any(hooks.has(name) for name in
        ['emb', 'pre_norm', 'logits'] +
        [f'qkv.{li}' for li in range(n_layer)] +
        [f'attn_w.{li}.{h}' for li in range(n_layer) for h in range(n_head)] +
        [f'head_out.{li}.{h}' for li in range(n_layer) for h in range(n_head)] +
        [f'post_attn.{li}' for li in range(n_layer)] +
        [f'mlp_hidden.{li}' for li in range(n_layer)] +
        [f'post_mlp.{li}' for li in range(n_layer)])

    # ---- FORWARD PASS (all positions at once) ----

    # Embeddings: (n, d)
    X = sd['wte'][token_ids] + sd['wpe'][:n]

    if hooks.has('emb'):
        for i in range(n):
            X[i] = hooks.apply('emb', X[i])

    snapshots = [] if capture_state else None

    # Pre-norm: (n, d)
    X_pre = X.copy()
    X = _rmsnorm(X)

    if hooks.has('pre_norm'):
        for i in range(n):
            X[i] = hooks.apply('pre_norm', X[i])

    # Store intermediates per layer for backward
    fwd_layers = []

    # Causal mask: (n, n), True where we should attend
    causal = np.tri(n, dtype=bool)  # lower triangle including diagonal

    for li in range(n_layer):
        lf = {}

        # --- Attention ---
        X_res = X.copy()
        lf['X_res_attn'] = X_res

        X_normed = _rmsnorm(X)
        lf['X_pre_attn'] = X.copy()
        lf['X_normed_attn'] = X_normed

        # QKV: (n, d)
        Q = X_normed @ sd[f'layer{li}.attn_wq'].T
        K = X_normed @ sd[f'layer{li}.attn_wk'].T
        V = X_normed @ sd[f'layer{li}.attn_wv'].T

        if hooks.has(f'qkv.{li}'):
            for i in range(n):
                qkv = hooks.apply(f'qkv.{li}', (Q[i], K[i], V[i]))
                if isinstance(qkv, tuple) and len(qkv) == 3:
                    Q[i], K[i], V[i] = qkv

        lf['Q'] = Q
        lf['K'] = K
        lf['V'] = V

        # Reshape to multi-head: (n, nh, hd)
        Q_h = Q.reshape(n, n_head, head_dim)
        K_h = K.reshape(n, n_head, head_dim)
        V_h = V.reshape(n, n_head, head_dim)

        # Attention scores: (nh, n, n) = (nh, n, hd) @ (nh, hd, n)
        Q_t = Q_h.transpose(1, 0, 2)  # (nh, n, hd)
        K_t = K_h.transpose(1, 0, 2)  # (nh, n, hd)
        V_t = V_h.transpose(1, 0, 2)  # (nh, n, hd)

        scores = np.einsum('hid,hjd->hij', Q_t, K_t) / np.sqrt(head_dim)  # (nh, n, n)
        # Apply causal mask: set future positions to -inf
        scores[:, ~causal] = -1e9

        # Softmax: (nh, n, n)
        attn_w = _softmax(scores)

        # Apply attention weight hooks per-head
        has_attn_hooks = any(hooks.has(f'attn_w.{li}.{h}') for h in range(n_head))
        if has_attn_hooks:
            for h in range(n_head):
                if hooks.has(f'attn_w.{li}.{h}'):
                    for i in range(n):
                        w = attn_w[h, i, :i+1].copy()
                        w = hooks.apply(f'attn_w.{li}.{h}', w)
                        if not isinstance(w, np.ndarray):
                            w = np.array(w, dtype=np.float64)
                        attn_w[h, i, :] = 0
                        attn_w[h, i, :len(w)] = w

        lf['attn_w'] = attn_w

        # Attention output: (nh, n, hd) = (nh, n, n) @ (nh, n, hd)
        attn_out = np.einsum('hij,hjd->hid', attn_w, V_t)  # (nh, n, hd)

        # Apply head_out hooks
        has_head_hooks = any(hooks.has(f'head_out.{li}.{h}') for h in range(n_head))
        if has_head_hooks:
            for h in range(n_head):
                if hooks.has(f'head_out.{li}.{h}'):
                    for i in range(n):
                        ho = hooks.apply(f'head_out.{li}.{h}', attn_out[h, i].copy())
                        if not isinstance(ho, np.ndarray):
                            ho = np.array(ho, dtype=np.float64)
                        attn_out[h, i] = ho

        # Capture state for probe
        if capture_state:
            for i in range(n):
                if i >= len(snapshots):
                    snapshots.append({'emb': X_pre[i].tolist(), 'layers': []})
                layer_snap = {'heads': []}
                for h in range(n_head):
                    w = attn_w[h, i, :i+1]
                    ho = attn_out[h, i]
                    layer_snap['heads'].append({
                        'attn_weights': w.tolist(),
                        'attn_entropy': float(_shannon_entropy(w)),
                        'head_out_norm': float(np.linalg.norm(ho)),
                        'head_out_vec': ho.tolist(),
                    })
                snapshots[i]['layers'].append(layer_snap)

        # Reshape back: (n, d)
        X_attn = attn_out.transpose(1, 0, 2).reshape(n, n_embd)
        lf['X_attn'] = X_attn

        # Output projection + residual
        X_proj = X_attn @ sd[f'layer{li}.attn_wo'].T  # (n, d)
        X = X_proj + X_res
        lf['X_proj_attn'] = X_proj

        if hooks.has(f'post_attn.{li}'):
            for i in range(n):
                X[i] = hooks.apply(f'post_attn.{li}', X[i])

        lf['X_after_attn'] = X.copy()

        if capture_state:
            for i in range(n):
                snapshots[i]['layers'][-1]['post_attn_residual'] = X[i].tolist()

        # --- MLP ---
        X_res_mlp = X.copy()
        lf['X_res_mlp'] = X_res_mlp

        X_normed_mlp = _rmsnorm(X)
        lf['X_pre_mlp'] = X.copy()
        lf['X_normed_mlp'] = X_normed_mlp

        # FC1 + ReLU: (n, 4d)
        fc1 = X_normed_mlp @ sd[f'layer{li}.mlp_fc1'].T
        relu = np.maximum(0, fc1)
        relu_mask = (fc1 > 0).astype(np.float64)
        lf['relu_mask'] = relu_mask

        # MLP hidden hooks
        if hooks.has(f'mlp_hidden.{li}'):
            for i in range(n):
                relu[i] = hooks.apply(f'mlp_hidden.{li}', relu[i].copy())
                if not isinstance(relu[i], np.ndarray):
                    relu[i] = np.array(relu[i], dtype=np.float64)

        lf['relu_hooked'] = relu

        # FC2 + residual: (n, d)
        fc2 = relu @ sd[f'layer{li}.mlp_fc2'].T
        X = fc2 + X_res_mlp

        if hooks.has(f'post_mlp.{li}'):
            for i in range(n):
                X[i] = hooks.apply(f'post_mlp.{li}', X[i])

        lf['X_after_mlp'] = X.copy()

        if capture_state:
            for i in range(n):
                snapshots[i]['layers'][-1]['post_mlp_residual'] = X[i].tolist()

        fwd_layers.append(lf)

    # Final logits: (n, vocab)
    logits = X @ sd['lm_head'].T

    if hooks.has('logits'):
        for i in range(n):
            logits[i] = hooks.apply('logits', logits[i])

    # Softmax + cross-entropy loss
    probs = _softmax(logits)  # (n, vocab)
    per_position_losses = [-math.log(probs[i, target_ids[i]] + 1e-10) for i in range(n)]
    loss = sum(per_position_losses) / n

    # ---- BACKWARD PASS (all positions at once) ----

    if local_loss:
        # Local-loss mode: each layer receives its own local loss signal
        # Uses per-layer probe heads to avoid hidden inter-layer coupling
        total_loss = 0.0
        last_local_per_pos = per_position_losses  # fallback

        for li in range(n_layer):
            lf = fwd_layers[li]
            X_after = lf['X_after_mlp']  # (n, n_embd)

            # Local logits and loss (using per-layer probe head)
            probe_key = f'probe_head_{li}'
            local_logits = X_after @ sd[probe_key].T  # (n, vocab)
            local_probs = _softmax(local_logits)
            local_per_pos = [-math.log(local_probs[i, target_ids[i]] + 1e-10)
                             for i in range(n)]
            local_loss_val = sum(local_per_pos) / n
            total_loss += local_loss_val
            last_local_per_pos = local_per_pos

            # Backward from local loss
            local_dlogits = local_probs.copy()
            for i in range(n):
                local_dlogits[i, target_ids[i]] -= 1.0
            local_dlogits /= n

            # Probe head gradients (per-layer, no shared lm_head coupling)
            grads[probe_key] += local_dlogits.T @ X_after

            # dX w.r.t. layer output
            dX = local_dlogits @ sd[probe_key]

            # Backward through layer li
            dX = _backward_through_layer(
                dX, li, lf, sd, grads, n, n_head, head_dim, n_embd, causal)

            # For layer 0, continue backward to embeddings
            if li == 0:
                dX = _rmsnorm_backward_batch(dX, X_pre)
                for i in range(n):
                    grads['wte'][token_ids[i]] += dX[i]
                    grads['wpe'][i] += dX[i]

        loss = total_loss / n_layer
        per_position_losses = last_local_per_pos

    else:
        # Standard end-to-end backward pass

        # dlogits: (n, vocab)
        dlogits = probs.copy()
        for i in range(n):
            dlogits[i, target_ids[i]] -= 1.0
        dlogits /= n

        # lm_head
        grads['lm_head'] += dlogits.T @ X  # (vocab, d)
        dX = dlogits @ sd['lm_head']       # (n, d)

        # Backward through layers
        for li in range(n_layer - 1, -1, -1):
            lf = fwd_layers[li]
            dX = _backward_through_layer(
                dX, li, lf, sd, grads, n, n_head, head_dim, n_embd, causal)

        # Pre-norm backward
        dX = _rmsnorm_backward_batch(dX, X_pre)

        # Embedding gradients
        for i in range(n):
            grads['wte'][token_ids[i]] += dX[i]
            grads['wpe'][i] += dX[i]

    return loss, per_position_losses, grads, snapshots


def _rmsnorm_backward_batch(dout, x_in):
    """
    Backward through rmsnorm for batch (n, d) inputs.
    Forward: y = x / sqrt(mean(x^2) + eps)
    """
    d = x_in.shape[-1]
    ms = np.mean(x_in * x_in, axis=-1, keepdims=True)  # (n, 1)
    scale = 1.0 / np.sqrt(ms + 1e-5)
    xdot = np.sum(x_in * dout, axis=-1, keepdims=True)  # (n, 1)
    return scale * dout - (scale ** 3) * x_in * xdot / d


def _backward_through_layer(dX, li, lf, sd, grads, n, n_head, head_dim, n_embd, causal):
    """Backward pass through a single transformer layer.
    Accumulates gradients for layer li's parameters into grads.
    Returns dX w.r.t. the input of this layer."""

    # --- MLP backward ---
    dX_res_mlp = dX.copy()

    grads[f'layer{li}.mlp_fc2'] += dX.T @ lf['relu_hooked']
    drelu = dX @ sd[f'layer{li}.mlp_fc2']
    dfc1 = drelu * lf['relu_mask']
    grads[f'layer{li}.mlp_fc1'] += dfc1.T @ lf['X_normed_mlp']
    dX_normed_mlp = dfc1 @ sd[f'layer{li}.mlp_fc1']
    dX = _rmsnorm_backward_batch(dX_normed_mlp, lf['X_pre_mlp']) + dX_res_mlp

    # --- Attention backward ---
    dX_res_attn = dX.copy()

    grads[f'layer{li}.attn_wo'] += dX.T @ lf['X_attn']
    dX_attn = dX @ sd[f'layer{li}.attn_wo']

    dX_attn_h = dX_attn.reshape(n, n_head, head_dim).transpose(1, 0, 2)

    attn_w = lf['attn_w']
    V_t = lf['V'].reshape(n, n_head, head_dim).transpose(1, 0, 2)
    Q_t = lf['Q'].reshape(n, n_head, head_dim).transpose(1, 0, 2)
    K_t = lf['K'].reshape(n, n_head, head_dim).transpose(1, 0, 2)

    dattn_w = np.einsum('hid,hjd->hij', dX_attn_h, V_t)
    dV_t = np.einsum('hij,hid->hjd', attn_w, dX_attn_h)

    s = np.sum(attn_w * dattn_w, axis=-1, keepdims=True)
    dscores = attn_w * (dattn_w - s)
    dscores[:, ~causal] = 0

    scale_attn = 1.0 / np.sqrt(head_dim)
    dQ_t = np.einsum('hij,hjd->hid', dscores, K_t) * scale_attn
    dK_t = np.einsum('hji,hjd->hid', dscores, Q_t) * scale_attn

    dQ = dQ_t.transpose(1, 0, 2).reshape(n, n_embd)
    dK = dK_t.transpose(1, 0, 2).reshape(n, n_embd)
    dV = dV_t.transpose(1, 0, 2).reshape(n, n_embd)

    X_na = lf['X_normed_attn']
    grads[f'layer{li}.attn_wq'] += dQ.T @ X_na
    grads[f'layer{li}.attn_wk'] += dK.T @ X_na
    grads[f'layer{li}.attn_wv'] += dV.T @ X_na

    dX_normed_attn = (dQ @ sd[f'layer{li}.attn_wq'] +
                      dK @ sd[f'layer{li}.attn_wk'] +
                      dV @ sd[f'layer{li}.attn_wv'])

    dX = _rmsnorm_backward_batch(dX_normed_attn, lf['X_pre_attn']) + dX_res_attn
    return dX


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
    assert n_embd % n_head == 0, f'n_embd ({n_embd}) must be divisible by n_head ({n_head})'
    assert n_layer > 0, f'n_layer must be positive, got {n_layer}'
    assert block_size > 0, f'block_size must be positive, got {block_size}'
    assert vocab_size > 0, f'vocab_size must be positive, got {vocab_size}'
    assert n_head > 0, f'n_head must be positive, got {n_head}'
    assert n_embd > 0, f'n_embd must be positive, got {n_embd}'
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

    # Per-layer probe heads for local-loss mode (independent per layer)
    for i in range(config['n_layer']):
        sd[f'probe_head_{i}'] = rng.randn(config['vocab_size'], config['n_embd']) * std

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

def load_dataset(path=None, val_fraction=0.1, val_seed=12345):
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

    # Deterministic train/val split
    rng = random.Random(val_seed)
    indices = list(range(len(docs)))
    rng.shuffle(indices)
    split = int(len(docs) * (1 - val_fraction))
    train_docs = [docs[i] for i in indices[:split]]
    val_docs = [docs[i] for i in indices[split:]]

    return train_docs, val_docs, uchars, BOS, vocab_size


def tokenize(doc, uchars, BOS):
    return [BOS] + [uchars.index(ch) for ch in doc] + [BOS]


# ============================================================================
# Training loop
# ============================================================================

def _evaluate(docs, state_dict, config, uchars, BOS, n_eval=50, seed=99999):
    """Forward-only pass on sampled docs. Returns mean loss."""
    rng = random.Random(seed)
    block_size = config['block_size']
    hooks = Hooks()
    total_loss = 0.0
    count = 0
    sample_indices = list(range(len(docs)))
    rng.shuffle(sample_indices)
    for idx in sample_indices[:n_eval]:
        doc = docs[idx]
        tokens = tokenize(doc, uchars, BOS)
        n = min(block_size, len(tokens) - 1)
        if n < 1:
            continue
        loss, _, _, _ = _forward_backward(
            tokens, n, state_dict, config, hooks, capture_state=False)
        total_loss += loss
        count += 1
    return total_loss / max(count, 1)


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

    def __post_init__(self):
        assert self.num_steps > 0, f'num_steps must be positive, got {self.num_steps}'
        assert self.learning_rate > 0, f'learning_rate must be positive, got {self.learning_rate}'


def train(state_dict, params, config, train_config, docs, uchars, BOS,
          hooks=None, probe=None, grad_hooks=None, seed=42,
          local_loss=False, val_docs=None):
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
        local_loss: if True, each layer gets its own local loss signal
        val_docs: optional validation docs for periodic evaluation
    """
    tc = train_config

    if hooks is None:
        hooks = Hooks()
    if probe is None:
        probe = Probe(detail_level=tc.detail_level)
    np.random.seed(seed)
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
            tokens, n, state_dict, config, hooks, capture_state=capture_this_step,
            local_loss=local_loss
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

        # Periodic validation evaluation
        if val_docs is not None and step % 20 == 0:
            val_loss = _evaluate(val_docs, state_dict, config, uchars, BOS)
            probe.val_losses.append((step, val_loss))

    # Final validation evaluation
    if val_docs is not None:
        val_loss = _evaluate(val_docs, state_dict, config, uchars, BOS)
        probe.val_losses.append((tc.num_steps - 1, val_loss))

    return probe


def train_with_state(state_dict, params, config, train_config, docs, uchars, BOS,
                     hooks=None, probe=None, grad_hooks=None, seed=42,
                     m_buf=None, v_buf=None, start_step=0, total_steps=None,
                     local_loss=False, val_docs=None):
    """
    Like train() but supports multi-phase training.
    Accepts/returns Adam optimizer state (m_buf, v_buf).
    Returns (probe, m_buf, v_buf).
    """
    tc = train_config

    if hooks is None:
        hooks = Hooks()
    if probe is None:
        probe = Probe(detail_level=tc.detail_level)
    np.random.seed(seed)
    rng = random.Random(seed)

    doc_order = list(range(len(docs)))
    rng.shuffle(doc_order)
    # Advance RNG to match start_step position
    for _ in range(start_step):
        rng.random()

    if m_buf is None:
        m_buf = {k: np.zeros_like(v) for k, v in state_dict.items()}
    if v_buf is None:
        v_buf = {k: np.zeros_like(v) for k, v in state_dict.items()}

    if total_steps is None:
        total_steps = start_step + tc.num_steps

    block_size = config['block_size']

    for local_step in range(tc.num_steps):
        global_step = start_step + local_step
        hooks.step = global_step

        doc_idx = doc_order[(global_step) % len(doc_order)]
        doc = docs[doc_idx]
        tokens = tokenize(doc, uchars, BOS)
        n = min(block_size, len(tokens) - 1)

        capture_this_step = (
            probe.detail_level != 'loss_only'
            and global_step % max(1, probe.record_interval) == 0
        )

        loss, per_position_losses, grads, snapshots = _forward_backward(
            tokens, n, state_dict, config, hooks, capture_state=capture_this_step,
            local_loss=local_loss
        )

        if capture_this_step:
            probe.record_step(global_step, loss, per_position_losses, snapshots)
        else:
            probe.record_loss(global_step, loss)

        if grad_hooks:
            for gh in grad_hooks:
                gh(grads, state_dict, global_step)

        if global_step % max(1, probe.record_interval) == 0:
            _record_grad_norms(probe, grads, config, global_step)

        lr_t = tc.learning_rate * (1 - global_step / total_steps)
        for k in state_dict:
            g = grads[k]
            m_buf[k] = tc.beta1 * m_buf[k] + (1 - tc.beta1) * g
            v_buf[k] = tc.beta2 * v_buf[k] + (1 - tc.beta2) * g ** 2
            m_hat = m_buf[k] / (1 - tc.beta1 ** (global_step + 1))
            v_hat = v_buf[k] / (1 - tc.beta2 ** (global_step + 1))
            state_dict[k] -= lr_t * m_hat / (np.sqrt(v_hat) + tc.eps_adam)

        if tc.print_every > 0 and (global_step + 1) % tc.print_every == 0:
            print(f"step {global_step+1:4d}/{total_steps} | loss {loss:.4f}")

        if tc.sample_every > 0 and (global_step + 1) % tc.sample_every == 0:
            samples = generate(state_dict, config, uchars, BOS,
                               num_samples=tc.num_samples,
                               temperature=tc.temperature)
            probe.record_samples(global_step, samples)

        # Periodic validation evaluation
        if val_docs is not None and global_step % 20 == 0:
            val_loss = _evaluate(val_docs, state_dict, config, uchars, BOS)
            probe.val_losses.append((global_step, val_loss))

    return probe, m_buf, v_buf


# ============================================================================
# Multi-phase utilities
# ============================================================================

LAYER_COMPONENTS = ['attn_wq', 'attn_wk', 'attn_wv', 'attn_wo', 'mlp_fc1', 'mlp_fc2']


def get_layer_keys(layer_idx):
    """Return the state_dict keys for a given layer."""
    return [f'layer{layer_idx}.{comp}' for comp in LAYER_COMPONENTS]


def reset_layer(state_dict, layer_idx, config, seed=None, m_buf=None, v_buf=None):
    """Re-initialize one layer's weights to random. Optionally reset Adam buffers."""
    rng = np.random.RandomState(seed)
    std = 0.08
    n_embd = config['n_embd']
    keys = get_layer_keys(layer_idx)
    shapes = {
        'attn_wq': (n_embd, n_embd), 'attn_wk': (n_embd, n_embd),
        'attn_wv': (n_embd, n_embd), 'attn_wo': (n_embd, n_embd),
        'mlp_fc1': (4 * n_embd, n_embd), 'mlp_fc2': (n_embd, 4 * n_embd),
    }
    for key in keys:
        comp = key.split('.')[1]
        state_dict[key] = rng.randn(*shapes[comp]) * std
        if m_buf is not None and key in m_buf:
            m_buf[key] = np.zeros_like(state_dict[key])
        if v_buf is not None and key in v_buf:
            v_buf[key] = np.zeros_like(state_dict[key])


def transplant_layer(sd_recipient, sd_donor, layer_idx, m_buf=None, v_buf=None):
    """Copy one layer's weights from donor into recipient. Reset Adam buffers."""
    for key in get_layer_keys(layer_idx):
        sd_recipient[key] = sd_donor[key].copy()
        if m_buf is not None and key in m_buf:
            m_buf[key] = np.zeros_like(sd_recipient[key])
        if v_buf is not None and key in v_buf:
            v_buf[key] = np.zeros_like(sd_recipient[key])


def assemble_chimera(sd_a, sd_b, layer_assignment, config):
    """
    Build chimera state_dict from two models.
    layer_assignment: dict {layer_idx: 'A' or 'B'}
    Shared params (wte, wpe, lm_head) come from model A.
    """
    sd = {}
    # Shared params from A
    for key in ['wte', 'wpe', 'lm_head']:
        sd[key] = sd_a[key].copy()
    # Per-layer params from assignment
    for li in range(config['n_layer']):
        source = sd_a if layer_assignment.get(li, 'A') == 'A' else sd_b
        for key in get_layer_keys(li):
            sd[key] = source[key].copy()
    # Per-layer probe heads from A
    for li in range(config['n_layer']):
        probe_key = f'probe_head_{li}'
        if probe_key in sd_a:
            sd[probe_key] = sd_a[probe_key].copy()
    return sd


def copy_state_dict(state_dict):
    """Deep copy a state_dict."""
    return {k: v.copy() for k, v in state_dict.items()}


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
    docs, val_docs, uchars, BOS, vocab_size = load_dataset()
    print(f"  docs: {len(docs)} train + {len(val_docs)} val, vocab: {vocab_size}")

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
