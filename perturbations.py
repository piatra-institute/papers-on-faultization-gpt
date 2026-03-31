"""
Faultization GPT — NumPy Perturbations

Drop-in replacement for perturbations.py using numpy arrays.
All hooks receive/return numpy arrays instead of Value objects.
Grad hooks receive (grads_dict, state_dict, step) where grads_dict
maps weight names to gradient numpy arrays.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any

import numpy as np
from model import Hooks


# ============================================================================
# SCHEDULES
# ============================================================================

def schedule_chronic(hook_fn: Callable) -> Callable:
    return hook_fn


def schedule_acute(hook_fn: Callable, start_step: int, end_step: int) -> Callable:
    def scheduled(value: Any, step: int = 0) -> Any:
        if start_step <= step <= end_step:
            return hook_fn(value, step=step)
        return None
    return scheduled


def schedule_stochastic(hook_fn: Callable, prob: float, rng: random.Random | None = None) -> Callable:
    _rng = rng or random.Random()
    def scheduled(value: Any, step: int = 0) -> Any:
        if _rng.random() < prob:
            return hook_fn(value, step=step)
        return None
    return scheduled


def schedule_progressive(hook_fn: Callable, intensity_fn: Callable[[int], float]) -> Callable:
    def scheduled(value: Any, step: int = 0) -> Any:
        intensity = intensity_fn(step)
        if intensity > 0:
            return hook_fn(value, step=step, intensity=intensity)
        return None
    return scheduled


# ============================================================================
# A4 + A10: FROZEN / DAMAGED COMPONENTS
# ============================================================================

def make_ablate_head(layer: int, head: int, head_dim: int) -> tuple[str, Callable]:
    """Ablate (zero) a head's output. Returns (hook_name, hook_fn)."""
    target = f'head_out.{layer}.{head}'
    def hook(head_out: np.ndarray, step: int = 0, **kw: Any) -> np.ndarray:
        return np.zeros(head_dim)
    return target, hook


# Backward-compatible alias
make_zero_head = make_ablate_head


def make_noise_head(layer: int, head: int, head_dim: int, noise_std: float = 0.1, rng: random.Random | None = None) -> tuple[str, Callable]:
    """Add Gaussian noise to a head's output. Returns (hook_name, hook_fn)."""
    _rng = rng or random.Random()
    target = f'head_out.{layer}.{head}'
    def hook(head_out: np.ndarray, step: int = 0, **kw: Any) -> np.ndarray:
        noise = np.array([_rng.gauss(0, noise_std) for _ in range(head_dim)])
        return head_out + noise
    return target, hook


def make_freeze_params(param_names: list[str]) -> Callable:
    """Zero gradients for named parameters. Returns a grad_hook."""
    def grad_hook(grads: dict[str, np.ndarray], state_dict: dict[str, np.ndarray], step: int) -> None:
        for name in param_names:
            if name in grads:
                grads[name][:] = 0
    return grad_hook


def make_freeze_head_params(layer: int, head: int, n_embd: int, head_dim: int) -> Callable:
    """Freeze specific head's Q, K, V, Wo gradients (true parameter freezing)."""
    hs = head * head_dim
    he = hs + head_dim
    def grad_hook(grads: dict[str, np.ndarray], state_dict: dict[str, np.ndarray], step: int) -> None:
        for comp in ['attn_wq', 'attn_wk', 'attn_wv']:
            key = f'layer{layer}.{comp}'
            if key in grads:
                grads[key][hs:he, :] = 0
        # Wo: columns correspond to head slices (input dim)
        wo_key = f'layer{layer}.attn_wo'
        if wo_key in grads:
            grads[wo_key][:, hs:he] = 0
    return grad_hook


def make_noise_injection(hook_name: str, noise_std: float, rng: random.Random | None = None) -> tuple[str, Callable]:
    """Add Gaussian noise at any hook point. Returns (hook_name, hook_fn)."""
    _rng = rng or random.Random()
    def hook(values: Any, step: int = 0, **kw: Any) -> Any:
        if isinstance(values, np.ndarray):
            noise = np.array([_rng.gauss(0, noise_std) for _ in range(len(values))])
            return values + noise
        return values
    return hook_name, hook


def make_quantize_activations(hook_name: str, bits: int = 4) -> tuple[str, Callable]:
    levels = 2 ** bits
    def hook(values: Any, step: int = 0, **kw: Any) -> Any:
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

def make_windowed_attention(layer: int, head: int, window_size: int) -> tuple[str, Callable]:
    """Limit attention to K most recent tokens. Returns (hook_name, hook_fn)."""
    target = f'attn_w.{layer}.{head}'
    def hook(attn_weights: np.ndarray, step: int = 0, **kw: Any) -> np.ndarray | None:
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


def make_sparse_attention(layer: int, head: int, keep_prob: float = 0.5, rng: random.Random | None = None) -> tuple[str, Callable]:
    _rng = rng or random.Random()
    target = f'attn_w.{layer}.{head}'
    def hook(attn_weights: np.ndarray, step: int = 0, **kw: Any) -> np.ndarray | None:
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

def make_stochastic_relu(hook_name: str, flip_prob: float = 0.05, rng: random.Random | None = None) -> tuple[str, Callable]:
    _rng = rng or random.Random()
    def hook(values: np.ndarray, step: int = 0, **kw: Any) -> np.ndarray:
        result = values.copy()
        for i in range(len(result)):
            if _rng.random() < flip_prob:
                if result[i] > 0:
                    result[i] = 0.0
                else:
                    result[i] = abs(result[i])
        return result
    return hook_name, hook


def make_dropout(hook_name: str, drop_prob: float = 0.1, rng: random.Random | None = None) -> tuple[str, Callable]:
    """Dropout: randomly zero activations. Returns (hook_name, hook_fn)."""
    _rng = rng or random.Random()
    scale = 1.0 / (1.0 - drop_prob) if drop_prob < 1.0 else 1.0
    def hook(values: np.ndarray, step: int = 0, **kw: Any) -> np.ndarray:
        mask = np.array([scale if _rng.random() > drop_prob else 0.0
                         for _ in range(len(values))])
        return values * mask
    return hook_name, hook


def make_attention_temperature(layer: int, head: int, temp_fn: Callable[[int], float]) -> tuple[str, Callable]:
    target = f'attn_w.{layer}.{head}'
    def hook(attn_weights: np.ndarray, step: int = 0, **kw: Any) -> np.ndarray | None:
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

def make_stop_gradient(layer: int) -> tuple[str, Callable]:
    """
    Cut gradient flow at layer boundary.
    In the numpy backend, this is handled by zeroing gradients for
    all layers below this one in the grad_hook.
    Returns (hook_name, hook_fn) -- the forward hook is identity,
    plus a grad_hook that zeros earlier-layer gradients.
    """
    target = f'post_mlp.{layer}'
    # The forward hook returns a copy (detached semantically)
    def hook(values: np.ndarray, step: int = 0, **kw: Any) -> np.ndarray:
        return values.copy()
    return target, hook


def make_stop_gradient_grad_hook(layer: int, config: dict[str, Any]) -> Callable:
    """
    Grad hook that zeros gradients for layers 0..layer.
    This implements the backward effect of stop_gradient.
    """
    def grad_hook(grads: dict[str, np.ndarray], state_dict: dict[str, np.ndarray], step: int) -> None:
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


def make_sign_only_gradients() -> Callable:
    """Replace each gradient with its sign. Returns a grad_hook."""
    def grad_hook(grads: dict[str, np.ndarray], state_dict: dict[str, np.ndarray], step: int) -> None:
        for k in grads:
            grads[k] = np.sign(grads[k])
    return grad_hook


def make_delayed_gradients(delay_steps: int = 5) -> Callable:
    from collections import deque
    buffer: dict[str, deque] = {}
    def grad_hook(grads: dict[str, np.ndarray], state_dict: dict[str, np.ndarray], step: int) -> None:
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

def make_noisy_gradients(noise_std: float = 0.01, rng: np.random.RandomState | None = None) -> Callable:
    """Add Gaussian noise to every gradient. Returns a grad_hook."""
    _rng = rng if rng is not None else np.random.RandomState()
    def grad_hook(grads: dict[str, np.ndarray], state_dict: dict[str, np.ndarray], step: int) -> None:
        for k in grads:
            grads[k] = grads[k] + _rng.randn(*grads[k].shape) * noise_std
    return grad_hook


def make_quantized_gradients(levels: int = 3) -> Callable:
    """Quantize gradients to discrete levels. Returns a grad_hook."""
    def grad_hook(grads: dict[str, np.ndarray], state_dict: dict[str, np.ndarray], step: int) -> None:
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


def make_shuffled_gradients(shuffle_within: str = 'layer', rng: random.Random | None = None) -> Callable:
    _rng = rng or random.Random()
    np_rng = np.random.RandomState(_rng.randint(0, 2**31))
    def grad_hook(grads: dict[str, np.ndarray], state_dict: dict[str, np.ndarray], step: int) -> None:
        for k in grads:
            flat = grads[k].ravel()
            np_rng.shuffle(flat)
            grads[k] = flat.reshape(grads[k].shape)
    return grad_hook


# ============================================================================
# A6: ASYNCHRONOUS UPDATES
# ============================================================================

def make_async_updates(layer_frequencies: dict[int, int]) -> Callable:
    def grad_hook(grads: dict[str, np.ndarray], state_dict: dict[str, np.ndarray], step: int) -> None:
        for li, freq in layer_frequencies.items():
            if step % freq != 0:
                for comp in ['attn_wq', 'attn_wk', 'attn_wv', 'attn_wo', 'mlp_fc1', 'mlp_fc2']:
                    key = f'layer{li}.{comp}'
                    if key in grads:
                        grads[key][:] = 0
    return grad_hook


def make_update_budget(budget_fraction: float = 0.5, rng: random.Random | None = None) -> Callable:
    def grad_hook(grads: dict[str, np.ndarray], state_dict: dict[str, np.ndarray], step: int) -> None:
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

def make_adversarial_head(layer: int, head: int, head_dim: int) -> Callable:
    hs = head * head_dim
    he = hs + head_dim
    def grad_hook(grads: dict[str, np.ndarray], state_dict: dict[str, np.ndarray], step: int) -> None:
        for comp in ['attn_wq', 'attn_wk', 'attn_wv']:
            key = f'layer{layer}.{comp}'
            if key in grads:
                grads[key][hs:he, :] *= -1
    return grad_hook


# ============================================================================
# CHESS-PAPER INSPIRED PERTURBATIONS
# ============================================================================

def make_layered_vision(config: dict[str, Any], radius_per_layer: dict[int, int]) -> list[tuple[str, Callable]]:
    hooks_list: list[tuple[str, Callable]] = []
    n_head = config['n_head']
    for layer_idx, window_size in radius_per_layer.items():
        for head in range(n_head):
            name, fn = make_windowed_attention(layer_idx, head, window_size)
            hooks_list.append((name, fn))
    return hooks_list


def make_partial_stop_gradient(layer: int, pass_fraction: float) -> tuple[str, Callable]:
    """
    Fractional gradient flow. In numpy backend, this is implemented
    as a grad_hook that scales gradients for layers 0..layer by pass_fraction.
    The forward hook is identity.
    """
    target = f'post_mlp.{layer}'
    def hook(values: np.ndarray, step: int = 0, **kw: Any) -> np.ndarray | None:
        if pass_fraction <= 0.0:
            return values.copy()  # semantically detached
        elif pass_fraction >= 1.0:
            return None
        else:
            return values  # forward pass unchanged; gradient scaling in grad_hook
    return target, hook


def make_partial_stop_gradient_grad_hook(layer: int, pass_fraction: float, config: dict[str, Any]) -> Callable:
    """
    Scale gradients for the boundary layer only by pass_fraction.
    At pass_fraction=0.0: full stop-gradient for this layer.
    At pass_fraction=1.0: no effect.

    With multiple hooks registered (one per boundary), each hook scales
    only its own layer -- no compounding. Layer 0 also scales embeddings.
    """
    def grad_hook(grads: dict[str, np.ndarray], state_dict: dict[str, np.ndarray], step: int) -> None:
        if pass_fraction >= 1.0:
            return
        for comp in ['attn_wq', 'attn_wk', 'attn_wv', 'attn_wo', 'mlp_fc1', 'mlp_fc2']:
            key = f'layer{layer}.{comp}'
            if key in grads:
                grads[key] *= pass_fraction
        if layer == 0:
            grads['wte'] *= pass_fraction
            grads['wpe'] *= pass_fraction
    return grad_hook


def make_threatening_drive(layer: int, head: int, head_dim: int, strength: float = 0.1) -> Callable:
    def grad_hook(grads: dict[str, np.ndarray], state_dict: dict[str, np.ndarray], step: int) -> None:
        hs = head * head_dim
        he = hs + head_dim
        for comp in ['attn_wq', 'attn_wk']:
            key = f'layer{layer}.{comp}'
            if key in grads:
                grads[key][hs:he, :] += strength * (-state_dict[key][hs:he, :])
    return grad_hook


def make_round_robin_updates(config: dict[str, Any], period: int | None = None) -> Callable:
    n_layer = config['n_layer']
    if period is None:
        period = 1
    def grad_hook(grads: dict[str, np.ndarray], state_dict: dict[str, np.ndarray], step: int) -> None:
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

def register_perturbations(hooks: Hooks, perturbation_list: list[tuple[str, Callable]]) -> None:
    for name, fn in perturbation_list:
        hooks.register(name, fn)


def freeze_random_heads(config: dict[str, Any], num_heads: int, rng: random.Random | None = None) -> tuple[list[tuple[int, int]], list[Callable]]:
    """Freeze random heads via gradient zeroing (true parameter freezing).
    Returns (frozen_head_list, grad_hooks_list)."""
    _rng = rng or random.Random()
    n_layer = config['n_layer']
    n_head = config['n_head']
    n_embd = config['n_embd']
    head_dim = config['head_dim']

    all_heads = [(li, h) for li in range(n_layer) for h in range(n_head)]
    _rng.shuffle(all_heads)
    frozen = all_heads[:num_heads]

    freeze_grad_hooks: list[Callable] = []
    for li, h in frozen:
        freeze_grad_hooks.append(
            make_freeze_head_params(li, h, n_embd, head_dim))

    return frozen, freeze_grad_hooks


def freeze_specific_heads(head_list: list[tuple[int, int]], config: dict[str, Any]) -> tuple[list[tuple[int, int]], list[Callable]]:
    """Freeze a specific list of (layer, head) tuples via gradient zeroing.
    Returns (head_list, grad_hooks_list)."""
    n_embd = config['n_embd']
    head_dim = config['head_dim']
    freeze_grad_hooks: list[Callable] = []
    for li, h in head_list:
        freeze_grad_hooks.append(
            make_freeze_head_params(li, h, n_embd, head_dim))
    return head_list, freeze_grad_hooks


def unfreeze_heads(grad_hooks_list: list[Callable], freeze_grad_hooks: list[Callable]) -> None:
    """Remove freeze grad hooks from the active grad_hooks list."""
    for gh in freeze_grad_hooks:
        if gh in grad_hooks_list:
            grad_hooks_list.remove(gh)


def make_gradual_noisy_gradients(max_noise_std: float, ramp_steps: int, rng: np.random.RandomState | None = None) -> Callable:
    """Gradient noise that linearly ramps from 0 to max_noise_std."""
    _rng = rng if rng is not None else np.random.RandomState()
    def grad_hook(grads: dict[str, np.ndarray], state_dict: dict[str, np.ndarray], step: int) -> None:
        intensity = min(1.0, step / max(1, ramp_steps)) * max_noise_std
        if intensity > 0:
            for k in grads:
                grads[k] = grads[k] + _rng.randn(*grads[k].shape) * intensity
    return grad_hook


def make_noisy_gradients_scheduled(noise_std: float, start_step: int = 0, end_step: float = float('inf'), rng: np.random.RandomState | None = None) -> Callable:
    """Gradient noise applied only within [start_step, end_step]."""
    _rng = rng if rng is not None else np.random.RandomState()
    def grad_hook(grads: dict[str, np.ndarray], state_dict: dict[str, np.ndarray], step: int) -> None:
        if start_step <= step <= end_step:
            for k in grads:
                grads[k] = grads[k] + _rng.randn(*grads[k].shape) * noise_std
    return grad_hook


def make_layer_selective_gradients(forward_layers: list[int], backward_layers: list[int], config: dict[str, Any]) -> Callable:
    """
    Zero gradients for layers NOT in the active set.
    forward_layers: list of layer indices that receive forward-objective gradients.
    backward_layers: list of layer indices that receive backward-objective gradients.
    This is used for the competing objectives experiment.
    """
    def grad_hook(grads: dict[str, np.ndarray], state_dict: dict[str, np.ndarray], step: int) -> None:
        active = set(forward_layers) | set(backward_layers)
        for li in range(config['n_layer']):
            if li not in active:
                for comp in ['attn_wq', 'attn_wk', 'attn_wv', 'attn_wo', 'mlp_fc1', 'mlp_fc2']:
                    key = f'layer{li}.{comp}'
                    if key in grads:
                        grads[key][:] = 0
    return grad_hook


def apply_stop_gradient_all(hooks: Hooks, config: dict[str, Any], grad_hooks_list: list[Callable]) -> None:
    """Apply stop-gradient at every layer boundary.
    Registers forward hooks AND adds grad_hooks to zero earlier-layer grads.
    """
    for li in range(config['n_layer'] - 1):
        name, fn = make_stop_gradient(li)
        hooks.register(name, fn)
        grad_hooks_list.append(make_stop_gradient_grad_hook(li, config))
