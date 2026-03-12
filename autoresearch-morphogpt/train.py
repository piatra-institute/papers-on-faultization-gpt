# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
Autoresearch-MorphoGPT training script. CPU-only, single-file, numpy backend.
Stripped-down version of morphogpt_np.py — no hooks, no probe, no generation.

Usage: uv run --script train.py
"""

import math
import time
import random
import numpy as np

from prepare import (
    BLOCK_SIZE, TIME_BUDGET, VOCAB_SIZE,
    load_dataset, split_dataset, tokenize, evaluate_val_loss,
)

# ---------------------------------------------------------------------------
# Model primitives
# ---------------------------------------------------------------------------

def _softmax(logits):
    """Numerically stable softmax. Works on 1D or last axis of ND."""
    if logits.ndim == 1:
        x = logits - logits.max()
        e = np.exp(x)
        return e / e.sum()
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


def _rmsnorm_backward_batch(dout, x_in):
    """
    Backward through rmsnorm for batch (n, d) inputs.
    Forward: y = x / sqrt(mean(x^2) + eps)
    """
    d = x_in.shape[-1]
    ms = np.mean(x_in * x_in, axis=-1, keepdims=True)
    scale = 1.0 / np.sqrt(ms + 1e-5)
    xdot = np.sum(x_in * dout, axis=-1, keepdims=True)
    return scale * dout - (scale ** 3) * x_in * xdot / d


# ---------------------------------------------------------------------------
# Forward + backward pass
# ---------------------------------------------------------------------------

def _forward_backward(tokens, n, state_dict, config):
    """
    Full forward + backward pass for one document.
    Vectorized across all n positions simultaneously.

    Returns:
        (loss_scalar, grads_dict)
    """
    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    head_dim = config['head_dim']
    sd = state_dict

    grads = {k: np.zeros_like(v) for k, v in sd.items()}

    token_ids = tokens[:n]
    target_ids = tokens[1:n+1]

    # ---- FORWARD PASS (all positions at once) ----

    # Embeddings: (n, d)
    X = sd['wte'][token_ids] + sd['wpe'][:n]

    # Pre-norm: (n, d)
    X_pre = X.copy()
    X = _rmsnorm(X)

    # Store intermediates per layer for backward
    fwd_layers = []

    # Causal mask: (n, n), True where we should attend
    causal = np.tri(n, dtype=bool)

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

        lf['Q'] = Q
        lf['K'] = K
        lf['V'] = V

        # Reshape to multi-head: (n, nh, hd)
        Q_h = Q.reshape(n, n_head, head_dim)
        K_h = K.reshape(n, n_head, head_dim)
        V_h = V.reshape(n, n_head, head_dim)

        # Attention scores: (nh, n, n) = (nh, n, hd) @ (nh, hd, n)
        Q_t = Q_h.transpose(1, 0, 2)
        K_t = K_h.transpose(1, 0, 2)
        V_t = V_h.transpose(1, 0, 2)

        scores = np.einsum('hid,hjd->hij', Q_t, K_t) / np.sqrt(head_dim)
        scores[:, ~causal] = -1e9

        # Softmax: (nh, n, n)
        attn_w = _softmax(scores)
        lf['attn_w'] = attn_w

        # Attention output: (nh, n, hd) = (nh, n, n) @ (nh, n, hd)
        attn_out = np.einsum('hij,hjd->hid', attn_w, V_t)

        # Reshape back: (n, d)
        X_attn = attn_out.transpose(1, 0, 2).reshape(n, n_embd)
        lf['X_attn'] = X_attn

        # Output projection + residual
        X_proj = X_attn @ sd[f'layer{li}.attn_wo'].T
        X = X_proj + X_res
        lf['X_proj_attn'] = X_proj
        lf['X_after_attn'] = X.copy()

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
        lf['relu_hooked'] = relu

        # FC2 + residual: (n, d)
        fc2 = relu @ sd[f'layer{li}.mlp_fc2'].T
        X = fc2 + X_res_mlp
        lf['X_after_mlp'] = X.copy()

        fwd_layers.append(lf)

    # Final logits: (n, vocab)
    logits = X @ sd['lm_head'].T

    # Softmax + cross-entropy loss
    probs = _softmax(logits)
    per_position_losses = [-math.log(probs[i, target_ids[i]] + 1e-10) for i in range(n)]
    loss = sum(per_position_losses) / n

    # ---- BACKWARD PASS (all positions at once) ----

    # dlogits: (n, vocab)
    dlogits = probs.copy()
    for i in range(n):
        dlogits[i, target_ids[i]] -= 1.0
    dlogits /= n

    # lm_head
    grads['lm_head'] += dlogits.T @ X
    dX = dlogits @ sd['lm_head']

    # Backward through layers
    for li in range(n_layer - 1, -1, -1):
        lf = fwd_layers[li]

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

    # Pre-norm backward
    dX = _rmsnorm_backward_batch(dX, X_pre)

    # Embedding gradients
    for i in range(n):
        grads['wte'][token_ids[i]] += dX[i]
        grads['wpe'][i] += dX[i]

    return loss, grads


# ---------------------------------------------------------------------------
# Model config and initialization
# ---------------------------------------------------------------------------

def make_config(n_layer=4, n_embd=16, n_head=4, block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE):
    return {
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'head_dim': n_embd // n_head,
        'block_size': block_size,
        'vocab_size': vocab_size,
    }


def init_state_dict(config, seed=42):
    """Initialize model parameters as numpy arrays."""
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

    return sd


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
N_LAYER = 4
N_EMBD = 16
N_HEAD = 4

# Optimization
LR = 0.01
BETA1 = 0.85
BETA2 = 0.99
EPS_ADAM = 1e-8
SEED = 42

# ---------------------------------------------------------------------------
# Setup: data, model, optimizer
# ---------------------------------------------------------------------------

t_start = time.time()
np.random.seed(SEED)
random.seed(SEED)

# Load data
docs, uchars, BOS, vocab_size = load_dataset()
train_docs, val_docs = split_dataset(docs)
print(f"Dataset: {len(train_docs)} train, {len(val_docs)} val, vocab={vocab_size}")

# Build model
config = make_config(n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD, vocab_size=vocab_size)
state_dict = init_state_dict(config, seed=SEED)
num_params = sum(v.size for v in state_dict.values())
print(f"Model: {N_LAYER}L/{N_EMBD}d/{N_HEAD}h, {num_params} params")

# Adam optimizer buffers
m_buf = {k: np.zeros_like(v) for k, v in state_dict.items()}
v_buf = {k: np.zeros_like(v) for k, v in state_dict.items()}

# Data order
doc_order = list(range(len(train_docs)))
random.shuffle(doc_order)

print(f"Time budget: {TIME_BUDGET}s")

# ---------------------------------------------------------------------------
# Training loop (time-budgeted)
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0.0
step = 0
warmup_steps = 5

while True:
    t0 = time.time()

    # Pick a training document
    doc_idx = doc_order[step % len(doc_order)]
    doc = train_docs[doc_idx]
    tokens = tokenize(doc, uchars, BOS)
    n = min(config['block_size'], len(tokens) - 1)

    # Forward + backward
    loss, grads = _forward_backward(tokens, n, state_dict, config)

    # Adam optimizer with linear LR decay
    lr_t = LR * max(0.0, 1.0 - total_training_time / TIME_BUDGET)
    for k in state_dict:
        g = grads[k]
        m_buf[k] = BETA1 * m_buf[k] + (1 - BETA1) * g
        v_buf[k] = BETA2 * v_buf[k] + (1 - BETA2) * g ** 2
        m_hat = m_buf[k] / (1 - BETA1 ** (step + 1))
        v_hat = v_buf[k] / (1 - BETA2 ** (step + 1))
        state_dict[k] -= lr_t * m_hat / (np.sqrt(v_hat) + EPS_ADAM)

    t1 = time.time()
    dt = t1 - t0

    # Exclude warmup steps from time budget
    if step >= warmup_steps:
        total_training_time += dt

    # Logging
    if (step + 1) % 500 == 0:
        pct = 100 * min(total_training_time / TIME_BUDGET, 1.0)
        print(f"step {step+1:6d} | loss {loss:.4f} | lr {lr_t:.6f} | {pct:.1f}% time used")

    step += 1

    # Time's up
    if step > warmup_steps and total_training_time >= TIME_BUDGET:
        break

t_end_training = time.time()

print(f"\nTraining done: {step} steps in {total_training_time:.1f}s")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_forward_fn(tokens, n, sd, cfg):
    """Closure for evaluate_val_loss: returns (loss, grads) but grads are discarded."""
    return _forward_backward(tokens, n, sd, cfg)

val_loss = evaluate_val_loss(eval_forward_fn, state_dict, config, val_docs, uchars, BOS)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

t_end = time.time()

print("---")
print(f"val_loss:         {val_loss:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"num_steps:        {step}")
print(f"num_params:       {num_params}")
print(f"n_layer:          {N_LAYER}")
print(f"n_embd:           {N_EMBD}")
