"""
MorphoGPT — NumPy Perturbations

Drop-in replacement for perturbations.py using numpy arrays.
All hooks receive/return numpy arrays instead of Value objects.
Grad hooks receive (grads_dict, state_dict, step) where grads_dict
maps weight names to gradient numpy arrays.
"""

import math
import random
import numpy as np
from morphogpt_np import Hooks


# ============================================================================
# SCHEDULES
# ============================================================================

def schedule_chronic(hook_fn):
    return hook_fn


def schedule_acute(hook_fn, start_step, end_step):
    def scheduled(value, step=0):
        if start_step <= step <= end_step:
            return hook_fn(value, step=step)
        return None
    return scheduled


def schedule_stochastic(hook_fn, prob, rng=None):
    _rng = rng or random.Random()
    def scheduled(value, step=0):
        if _rng.random() < prob:
            return hook_fn(value, step=step)
        return None
    return scheduled


def schedule_progressive(hook_fn, intensity_fn):
    def scheduled(value, step=0):
        intensity = intensity_fn(step)
        if intensity > 0:
            return hook_fn(value, step=step, intensity=intensity)
        return None
    return scheduled


# ============================================================================
# A4 + A10: FROZEN / DAMAGED COMPONENTS
# ============================================================================

def make_zero_head(layer, head, head_dim):
    """Zero a head's output. Returns (hook_name, hook_fn)."""
    target = f'head_out.{layer}.{head}'
    def hook(head_out, step=0, **kw):
        return np.zeros(head_dim)
    return target, hook


def make_noise_head(layer, head, head_dim, noise_std=0.1, rng=None):
    """Add Gaussian noise to a head's output. Returns (hook_name, hook_fn)."""
    _rng = rng or random.Random()
    target = f'head_out.{layer}.{head}'
    def hook(head_out, step=0, **kw):
        noise = np.array([_rng.gauss(0, noise_std) for _ in range(head_dim)])
        return head_out + noise
    return target, hook


def make_freeze_params(param_names):
    """Zero gradients for named parameters. Returns a grad_hook."""
    def grad_hook(grads, state_dict, step):
        for name in param_names:
            if name in grads:
                grads[name][:] = 0
    return grad_hook


def make_freeze_head_params(layer, head, n_embd, head_dim):
    """Freeze specific head's Q, K, V gradients."""
    hs = head * head_dim
    he = hs + head_dim
    def grad_hook(grads, state_dict, step):
        for comp in ['attn_wq', 'attn_wk', 'attn_wv']:
            key = f'layer{layer}.{comp}'
            if key in grads:
                grads[key][hs:he, :] = 0
    return grad_hook


def make_noise_injection(hook_name, noise_std, rng=None):
    """Add Gaussian noise at any hook point. Returns (hook_name, hook_fn)."""
    _rng = rng or random.Random()
    def hook(values, step=0, **kw):
        if isinstance(values, np.ndarray):
            noise = np.array([_rng.gauss(0, noise_std) for _ in range(len(values))])
            return values + noise
        return values
    return hook_name, hook


def make_quantize_activations(hook_name, bits=4):
    levels = 2 ** bits
    def hook(values, step=0, **kw):
        if isinstance(values, np.ndarray):
            vmin, vmax = values.min(), values.max()
            if vmax - vmin < 1e-10:
                return values
            scale = (vmax - vmin) / levels
            return np.round((values - vmin) / scale) * scale + vmin
        return values
    return hook_name, hook


# ============================================================================
# A2: ATTENTION MODIFICATIONS
# ============================================================================

def make_windowed_attention(layer, head, window_size):
    """Limit attention to K most recent tokens. Returns (hook_name, hook_fn)."""
    target = f'attn_w.{layer}.{head}'
    def hook(attn_weights, step=0, **kw):
        n = len(attn_weights)
        if n <= window_size:
            return None
        masked = attn_weights.copy()
        masked[:n - window_size] = 0.0
        total = masked.sum()
        if total > 1e-10:
            return masked / total
        return masked
    return target, hook


def make_sparse_attention(layer, head, keep_prob=0.5, rng=None):
    _rng = rng or random.Random()
    target = f'attn_w.{layer}.{head}'
    def hook(attn_weights, step=0, **kw):
        n = len(attn_weights)
        if n <= 1:
            return None
        mask = np.array([1.0 if _rng.random() < keep_prob else 0.0 for _ in range(n)])
        masked = attn_weights * mask
        total = masked.sum()
        if total > 1e-10:
            return masked / total
        return masked
    return target, hook


# ============================================================================
# A4: STOCHASTIC FORWARD PASS
# ============================================================================

def make_stochastic_relu(hook_name, flip_prob=0.05, rng=None):
    _rng = rng or random.Random()
    def hook(values, step=0, **kw):
        result = values.copy()
        for i in range(len(result)):
            if _rng.random() < flip_prob:
                if result[i] > 0:
                    result[i] = 0.0
                else:
                    result[i] = abs(result[i])
        return result
    return hook_name, hook


def make_dropout(hook_name, drop_prob=0.1, rng=None):
    """Dropout: randomly zero activations. Returns (hook_name, hook_fn)."""
    _rng = rng or random.Random()
    scale = 1.0 / (1.0 - drop_prob) if drop_prob < 1.0 else 1.0
    def hook(values, step=0, **kw):
        mask = np.array([scale if _rng.random() > drop_prob else 0.0
                         for _ in range(len(values))])
        return values * mask
    return hook_name, hook


def make_attention_temperature(layer, head, temp_fn):
    target = f'attn_w.{layer}.{head}'
    def hook(attn_weights, step=0, **kw):
        temp = temp_fn(step)
        if abs(temp - 1.0) < 1e-6:
            return None
        powered = np.maximum(attn_weights, 1e-10) ** (1.0 / temp)
        total = powered.sum()
        if total > 1e-10:
            return powered / total
        return attn_weights
    return target, hook


# ============================================================================
# A5: BREAKING GLOBAL BACKPROPAGATION
# ============================================================================

def make_stop_gradient(layer):
    """
    Cut gradient flow at layer boundary.
    In the numpy backend, this is handled by zeroing gradients for
    all layers below this one in the grad_hook.
    Returns (hook_name, hook_fn) — the forward hook is identity,
    plus a grad_hook that zeros earlier-layer gradients.
    """
    target = f'post_mlp.{layer}'
    # The forward hook returns a copy (detached semantically)
    def hook(values, step=0, **kw):
        return values.copy()
    return target, hook


def make_stop_gradient_grad_hook(layer, config):
    """
    Grad hook that zeros gradients for layers 0..layer.
    This implements the backward effect of stop_gradient.
    """
    def grad_hook(grads, state_dict, step):
        for li in range(layer + 1):
            for comp in ['attn_wq', 'attn_wk', 'attn_wv', 'attn_wo', 'mlp_fc1', 'mlp_fc2']:
                key = f'layer{li}.{comp}'
                if key in grads:
                    grads[key][:] = 0
        # Also zero embedding gradients since no gradient flows past layer 0
        if layer == 0:
            grads['wte'][:] = 0
            grads['wpe'][:] = 0
    return grad_hook


def make_sign_only_gradients():
    """Replace each gradient with its sign. Returns a grad_hook."""
    def grad_hook(grads, state_dict, step):
        for k in grads:
            grads[k] = np.sign(grads[k])
    return grad_hook


def make_delayed_gradients(delay_steps=5):
    from collections import deque
    buffer = {}
    def grad_hook(grads, state_dict, step):
        for k in grads:
            if k not in buffer:
                buffer[k] = deque(maxlen=delay_steps + 1)
            buffer[k].append(grads[k].copy())
            if len(buffer[k]) > delay_steps:
                grads[k] = buffer[k][0]
    return grad_hook


# ============================================================================
# A10: GRADIENT PERTURBATIONS
# ============================================================================

def make_noisy_gradients(noise_std=0.01, rng=None):
    """Add Gaussian noise to every gradient. Returns a grad_hook."""
    def grad_hook(grads, state_dict, step):
        for k in grads:
            grads[k] = grads[k] + np.random.randn(*grads[k].shape) * noise_std
    return grad_hook


def make_quantized_gradients(levels=3):
    """Quantize gradients to discrete levels. Returns a grad_hook."""
    def grad_hook(grads, state_dict, step):
        # Compute mean absolute gradient across all params
        all_abs = []
        for k in grads:
            g = grads[k]
            nonzero = np.abs(g[g != 0])
            if len(nonzero) > 0:
                all_abs.append(nonzero)
        if not all_abs:
            return
        threshold = float(np.mean(np.concatenate(all_abs)))

        for k in grads:
            g = grads[k]
            if levels == 3:
                grads[k] = np.where(g > threshold, 1.0,
                           np.where(g < -threshold, -1.0, 0.0))
            else:
                if threshold > 0:
                    bucket = np.round(g / threshold * (levels // 2))
                    bucket = np.clip(bucket, -levels // 2, levels // 2)
                    grads[k] = bucket * threshold / (levels // 2)
    return grad_hook


def make_shuffled_gradients(shuffle_within='layer', rng=None):
    _rng = rng or random.Random()
    np_rng = np.random.RandomState(_rng.randint(0, 2**31))
    def grad_hook(grads, state_dict, step):
        for k in grads:
            flat = grads[k].ravel()
            np_rng.shuffle(flat)
            grads[k] = flat.reshape(grads[k].shape)
    return grad_hook


# ============================================================================
# A6: ASYNCHRONOUS UPDATES
# ============================================================================

def make_async_updates(layer_frequencies):
    def grad_hook(grads, state_dict, step):
        for li, freq in layer_frequencies.items():
            if step % freq != 0:
                for comp in ['attn_wq', 'attn_wk', 'attn_wv', 'attn_wo', 'mlp_fc1', 'mlp_fc2']:
                    key = f'layer{li}.{comp}'
                    if key in grads:
                        grads[key][:] = 0
    return grad_hook


def make_update_budget(budget_fraction=0.5, rng=None):
    def grad_hook(grads, state_dict, step):
        all_grads = np.concatenate([g.ravel() for g in grads.values()])
        threshold_idx = int(len(all_grads) * budget_fraction)
        if threshold_idx <= 0:
            return
        sorted_abs = np.sort(np.abs(all_grads))[::-1]
        threshold = sorted_abs[min(threshold_idx, len(sorted_abs) - 1)]
        for k in grads:
            grads[k] = np.where(np.abs(grads[k]) >= threshold, grads[k], 0.0)
    return grad_hook


# ============================================================================
# A8: MULTIPLE OBJECTIVES
# ============================================================================

def make_adversarial_head(layer, head, head_dim):
    hs = head * head_dim
    he = hs + head_dim
    def grad_hook(grads, state_dict, step):
        for comp in ['attn_wq', 'attn_wk', 'attn_wv']:
            key = f'layer{layer}.{comp}'
            if key in grads:
                grads[key][hs:he, :] *= -1
    return grad_hook


# ============================================================================
# CHESS-PAPER INSPIRED PERTURBATIONS
# ============================================================================

def make_layered_vision(config, radius_per_layer):
    hooks_list = []
    n_head = config['n_head']
    for layer_idx, window_size in radius_per_layer.items():
        for head in range(n_head):
            name, fn = make_windowed_attention(layer_idx, head, window_size)
            hooks_list.append((name, fn))
    return hooks_list


def make_partial_stop_gradient(layer, pass_fraction):
    """
    Fractional gradient flow. In numpy backend, this is implemented
    as a grad_hook that scales gradients for layers 0..layer by pass_fraction.
    The forward hook is identity.
    """
    target = f'post_mlp.{layer}'
    def hook(values, step=0, **kw):
        if pass_fraction <= 0.0:
            return values.copy()  # semantically detached
        elif pass_fraction >= 1.0:
            return None
        else:
            return values  # forward pass unchanged; gradient scaling in grad_hook
    return target, hook


def make_partial_stop_gradient_grad_hook(layer, pass_fraction, config):
    """
    Scale gradients for layers 0..layer by pass_fraction.
    At pass_fraction=0.0: full stop-gradient (same as make_stop_gradient).
    At pass_fraction=1.0: no effect.
    """
    def grad_hook(grads, state_dict, step):
        if pass_fraction >= 1.0:
            return
        for li in range(layer + 1):
            for comp in ['attn_wq', 'attn_wk', 'attn_wv', 'attn_wo', 'mlp_fc1', 'mlp_fc2']:
                key = f'layer{li}.{comp}'
                if key in grads:
                    grads[key] *= pass_fraction
        if layer == 0:
            grads['wte'] *= pass_fraction
            grads['wpe'] *= pass_fraction
    return grad_hook


def make_threatening_drive(layer, head, head_dim, strength=0.1):
    def grad_hook(grads, state_dict, step):
        hs = head * head_dim
        he = hs + head_dim
        for comp in ['attn_wq', 'attn_wk']:
            key = f'layer{layer}.{comp}'
            if key in grads:
                grads[key][hs:he, :] += strength * (-state_dict[key][hs:he, :])
    return grad_hook


def make_round_robin_updates(config, period=None):
    n_layer = config['n_layer']
    if period is None:
        period = 1
    def grad_hook(grads, state_dict, step):
        active_layer = (step // period) % n_layer
        for li in range(n_layer):
            if li == active_layer:
                continue
            for comp in ['attn_wq', 'attn_wk', 'attn_wv', 'attn_wo', 'mlp_fc1', 'mlp_fc2']:
                key = f'layer{li}.{comp}'
                if key in grads:
                    grads[key][:] = 0
    return grad_hook


# ============================================================================
# Convenience functions
# ============================================================================

def register_perturbations(hooks, perturbation_list):
    for name, fn in perturbation_list:
        hooks.register(name, fn)


def freeze_random_heads(hooks, config, num_heads, rng=None):
    _rng = rng or random.Random()
    n_layer = config['n_layer']
    n_head = config['n_head']
    head_dim = config['head_dim']

    all_heads = [(li, h) for li in range(n_layer) for h in range(n_head)]
    _rng.shuffle(all_heads)
    frozen = all_heads[:num_heads]

    for li, h in frozen:
        name, fn = make_zero_head(li, h, head_dim)
        hooks.register(name, fn)

    return frozen


def apply_stop_gradient_all(hooks, config, grad_hooks_list):
    """Apply stop-gradient at every layer boundary.
    Registers forward hooks AND adds grad_hooks to zero earlier-layer grads.
    """
    for li in range(config['n_layer'] - 1):
        name, fn = make_stop_gradient(li)
        hooks.register(name, fn)
        grad_hooks_list.append(make_stop_gradient_grad_hook(li, config))
