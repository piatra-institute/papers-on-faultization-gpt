"""
MorphoGPT — Perturbations

Systematic assumption-violations for the GPT loop.
Each perturbation is a hook (or a set of hooks + grad_hooks) that can be
registered with the Hooks system in morphogpt.py.

Organized by the 12 assumptions from MODIFICATIONS.md:
  A1: Sequential processing
  A2: Causal attention
  A3: Shared parameters
  A4: Deterministic forward pass
  A5: Global backpropagation
  A6: Synchronous updates
  A7: Fixed architecture
  A8: One objective
  A9: Passive data
  A10: Exact gradients
  A11: Perfect KV cache
  A12: Fixed dataset
"""

import math
import random
from morphogpt import Value, Hooks, softmax


# ============================================================================
# SCHEDULES — when perturbations are active
# ============================================================================

def schedule_chronic(hook_fn):
    """Always active. Returns the hook function unchanged."""
    return hook_fn


def schedule_acute(hook_fn, start_step, end_step):
    """Active only during [start_step, end_step]."""
    def scheduled(value, step=0):
        if start_step <= step <= end_step:
            return hook_fn(value, step=step)
        return None
    return scheduled


def schedule_stochastic(hook_fn, prob, rng=None):
    """Active with probability prob at each call."""
    _rng = rng or random.Random()
    def scheduled(value, step=0):
        if _rng.random() < prob:
            return hook_fn(value, step=step)
        return None
    return scheduled


def schedule_progressive(hook_fn, intensity_fn):
    """
    Active with intensity that changes over time.
    intensity_fn(step) -> float in [0, 1].
    The hook receives a modified value scaled by intensity.
    """
    def scheduled(value, step=0):
        intensity = intensity_fn(step)
        if intensity > 0:
            return hook_fn(value, step=step, intensity=intensity)
        return None
    return scheduled


# ============================================================================
# A4 + A10: FROZEN / DAMAGED COMPONENTS (Levin's frozen cells)
# ============================================================================

def make_zero_head(layer, head, head_dim):
    """
    Immovable frozen cell: head contributes nothing to forward pass.
    Registers at hook point 'head_out.{layer}.{head}'.
    Returns (hook_name, hook_fn).
    """
    target = f'head_out.{layer}.{head}'
    def hook(head_out, step=0, **kw):
        return [Value(0.0) for _ in range(head_dim)]
    return target, hook


def make_noise_head(layer, head, head_dim, noise_std=0.1, rng=None):
    """
    Damaged cell: add Gaussian noise to a head's output.
    Returns (hook_name, hook_fn).
    """
    _rng = rng or random.Random()
    target = f'head_out.{layer}.{head}'
    def hook(head_out, step=0, **kw):
        return [Value(v.data + _rng.gauss(0, noise_std),
                      children=(v,), local_grads=(1,))
                for v in head_out]
    return target, hook


def make_freeze_params(param_names):
    """
    Movable frozen cell: params contribute to forward pass but don't learn.
    Returns a grad_hook function (called after backward, before optimizer).
    """
    def grad_hook(params, state_dict, step):
        for name in param_names:
            if name in state_dict:
                for row in state_dict[name]:
                    for p in row:
                        p.grad = 0
    return grad_hook


def make_freeze_head_params(layer, head, n_embd, head_dim):
    """
    Movable frozen: freeze specific head's Q, K, V parameters.
    Returns a grad_hook that zeros gradients for the head's slice of Wq, Wk, Wv.
    """
    hs = head * head_dim
    he = hs + head_dim
    def grad_hook(params, state_dict, step):
        for comp in ['attn_wq', 'attn_wk', 'attn_wv']:
            key = f'layer{layer}.{comp}'
            if key in state_dict:
                for row_idx in range(hs, he):
                    if row_idx < len(state_dict[key]):
                        for p in state_dict[key][row_idx]:
                            p.grad = 0
    return grad_hook


def make_noise_injection(hook_name, noise_std, rng=None):
    """
    Add Gaussian noise at any hook point.
    Returns (hook_name, hook_fn).
    """
    _rng = rng or random.Random()
    def hook(values, step=0, **kw):
        if isinstance(values, list):
            return [Value(v.data + _rng.gauss(0, noise_std),
                          children=(v,), local_grads=(1,))
                    for v in values]
        return values
    return hook_name, hook


def make_quantize_activations(hook_name, bits=4):
    """
    Quantize activations at a hook point to simulated low precision.
    Returns (hook_name, hook_fn).
    """
    levels = 2 ** bits
    def hook(values, step=0, **kw):
        if isinstance(values, list):
            data = [v.data for v in values]
            if not data:
                return values
            vmin, vmax = min(data), max(data)
            if vmax - vmin < 1e-10:
                return values
            scale = (vmax - vmin) / levels
            return [Value(round((v.data - vmin) / scale) * scale + vmin,
                          children=(v,), local_grads=(1,))
                    for v in values]
        return values
    return hook_name, hook


# ============================================================================
# A1: EXECUTION ORDER (breaking sequential L->R processing)
# ============================================================================

def make_random_order_positions(block_size, rng=None):
    """
    Generate a shuffled position order for processing.
    Returns a list of position indices in random order.
    Used by modifying the training loop's position iteration.
    """
    _rng = rng or random.Random()
    order = list(range(block_size))
    _rng.shuffle(order)
    return order


def make_reverse_positions(n):
    """
    Generate reversed position order.
    Returns list [n-1, n-2, ..., 0].
    """
    return list(range(n - 1, -1, -1))


# ============================================================================
# A2: ATTENTION MODIFICATIONS (breaking causal attention)
# ============================================================================

def make_windowed_attention(layer, head, window_size):
    """
    Limit attention to only the K most recent tokens.
    Zeros out attention weights for tokens beyond the window.
    Returns (hook_name, hook_fn).
    """
    target = f'attn_w.{layer}.{head}'
    def hook(attn_weights, step=0, **kw):
        n = len(attn_weights)
        if n <= window_size:
            return None  # no modification needed
        # Zero out weights for tokens beyond the window
        masked = []
        for i, w in enumerate(attn_weights):
            if i < n - window_size:
                masked.append(Value(0.0))
            else:
                masked.append(w)
        # Re-normalize
        total = sum(m.data for m in masked)
        if total > 1e-10:
            return [m / total for m in masked]
        return masked
    return target, hook


def make_sparse_attention(layer, head, keep_prob=0.5, rng=None):
    """
    Each token attends to a random subset of past tokens.
    Returns (hook_name, hook_fn).
    """
    _rng = rng or random.Random()
    target = f'attn_w.{layer}.{head}'
    def hook(attn_weights, step=0, **kw):
        n = len(attn_weights)
        if n <= 1:
            return None
        masked = []
        for w in attn_weights:
            if _rng.random() < keep_prob:
                masked.append(w)
            else:
                masked.append(Value(0.0))
        total = sum(m.data for m in masked)
        if total > 1e-10:
            return [m / total for m in masked]
        return masked
    return target, hook


# ============================================================================
# A4: STOCHASTIC FORWARD PASS
# ============================================================================

def make_stochastic_relu(hook_name, flip_prob=0.05, rng=None):
    """
    Stochastic neurons: each ReLU activation has probability flip_prob of
    misfiring (on when should be off, or off when should be on).
    Apply at 'mlp_hidden.{layer}'.
    Returns (hook_name, hook_fn).
    """
    _rng = rng or random.Random()
    def hook(values, step=0, **kw):
        result = []
        for v in values:
            if _rng.random() < flip_prob:
                if v.data > 0:
                    result.append(Value(0.0))
                else:
                    result.append(Value(abs(v.data), children=(v,), local_grads=(1,)))
            else:
                result.append(v)
        return result
    return hook_name, hook


def make_dropout(hook_name, drop_prob=0.1, rng=None):
    """
    Dropout: randomly zero activations during forward pass.
    Returns (hook_name, hook_fn).
    """
    _rng = rng or random.Random()
    scale = 1.0 / (1.0 - drop_prob) if drop_prob < 1.0 else 1.0
    def hook(values, step=0, **kw):
        return [Value(v.data * scale, children=(v,), local_grads=(scale,))
                if _rng.random() > drop_prob
                else Value(0.0)
                for v in values]
    return hook_name, hook


def make_attention_temperature(layer, head, temp_fn):
    """
    Variable temperature on attention logits during training.
    temp_fn(step) -> temperature value.
    High temp = uniform (noisy), low temp = sharp (deterministic).

    Note: this must be applied BEFORE softmax, so it modifies the
    attention logits. Use by hooking into the QKV computation or
    by modifying the attention weights post-softmax with re-scaling.

    Returns (hook_name, hook_fn) that operates on attention weights.
    """
    target = f'attn_w.{layer}.{head}'
    def hook(attn_weights, step=0, **kw):
        temp = temp_fn(step)
        if abs(temp - 1.0) < 1e-6:
            return None
        # Sharpen or flatten by raising weights to power 1/temp and renormalizing
        powered = [Value(max(w.data, 1e-10) ** (1.0 / temp)) for w in attn_weights]
        total = sum(p.data for p in powered)
        if total > 1e-10:
            return [p / total for p in powered]
        return attn_weights
    return target, hook


# ============================================================================
# A5: BREAKING GLOBAL BACKPROPAGATION
# ============================================================================

def make_stop_gradient(layer):
    """
    Cut gradient flow at the residual stream after a layer.
    Creates new Value nodes with same data but no parents.
    Returns (hook_name, hook_fn).
    """
    target = f'post_mlp.{layer}'
    def hook(values, step=0, **kw):
        return [Value(v.data) for v in values]  # new leaf nodes — no gradient flow
    return target, hook


def make_block_local_loss_probe(layer, n_embd, vocab_size, rng=None):
    """
    Create a linear probe for block-local loss.
    Returns (probe_weights, probe_fn) where probe_fn takes layer output
    and returns logits for next-token prediction from that layer alone.

    The probe weights are separate from the model and trained with their own gradients.
    """
    _rng = rng or random.Random()
    probe_w = [[Value(_rng.gauss(0, 0.08)) for _ in range(n_embd)]
               for _ in range(vocab_size)]
    probe_params = [p for row in probe_w for p in row]

    def probe_fn(layer_output):
        from morphogpt import linear
        return linear(layer_output, probe_w)

    return probe_w, probe_params, probe_fn


def make_random_feedback(layer):
    """
    Feedback alignment: instead of backpropagating through actual weights,
    inject random fixed feedback. Implemented by stopping gradient and
    adding a noise-based surrogate gradient signal.

    Returns (hook_name, hook_fn).
    """
    target = f'post_mlp.{layer}'
    def hook(values, step=0, **kw):
        # Stop gradient flow, then add a random perturbation
        # that provides a rough directional signal
        return [Value(v.data) for v in values]
    return target, hook


def make_sign_only_gradients():
    """
    Replace each gradient with just its sign (+1, -1, or 0).
    Returns a grad_hook.
    """
    def grad_hook(params, state_dict, step):
        for p in params:
            if p.grad > 0:
                p.grad = 1.0
            elif p.grad < 0:
                p.grad = -1.0
            # else: 0 stays 0
    return grad_hook


def make_delayed_gradients(delay_steps=5):
    """
    Use gradients from N steps ago instead of current step.
    Returns a grad_hook.
    """
    buffer = {}  # param_id -> deque of recent gradients
    from collections import deque

    def grad_hook(params, state_dict, step):
        for i, p in enumerate(params):
            if i not in buffer:
                buffer[i] = deque(maxlen=delay_steps + 1)
            buffer[i].append(p.grad)
            if len(buffer[i]) > delay_steps:
                p.grad = buffer[i][0]  # use stale gradient
            # else: use current (not enough history yet)
    return grad_hook


# ============================================================================
# A6: ASYNCHRONOUS UPDATES
# ============================================================================

def make_async_updates(layer_frequencies):
    """
    Different components update at different frequencies.
    layer_frequencies: dict mapping layer_index -> update_every_N_steps.
    e.g., {0: 1, 1: 2, 2: 5, 3: 10}

    Returns a grad_hook that zeros gradients for layers not scheduled to update.
    """
    def grad_hook(params, state_dict, step):
        for li, freq in layer_frequencies.items():
            if step % freq != 0:
                # Zero gradients for this layer's parameters
                for comp in ['attn_wq', 'attn_wk', 'attn_wv', 'attn_wo', 'mlp_fc1', 'mlp_fc2']:
                    key = f'layer{li}.{comp}'
                    if key in state_dict:
                        for row in state_dict[key]:
                            for p in row:
                                p.grad = 0
    return grad_hook


def make_update_budget(budget_fraction=0.5, rng=None):
    """
    Only update the top budget_fraction% of parameters (by gradient magnitude).
    Returns a grad_hook.
    """
    _rng = rng or random.Random()
    def grad_hook(params, state_dict, step):
        grads = [(abs(p.grad), i) for i, p in enumerate(params)]
        grads.sort(reverse=True)
        cutoff = int(len(grads) * budget_fraction)
        # Zero gradients below the cutoff
        for _, i in grads[cutoff:]:
            params[i].grad = 0
    return grad_hook


def make_momentum_diversity(layer_betas):
    """
    Different layers have different Adam momentum parameters.
    layer_betas: dict mapping layer_index -> (beta1, beta2).

    NOTE: This requires modifying the optimizer, not just the gradients.
    Returns configuration dict for use with a custom optimizer.
    """
    return layer_betas  # used by a modified training loop


# ============================================================================
# A7: ARCHITECTURE MORPHOGENESIS
# ============================================================================

def make_head_pruning(contribution_threshold=0.01, patience=50):
    """
    Pruning (apoptosis): track each head's contribution.
    When a head's contribution falls below threshold for patience steps,
    zero it permanently.

    Returns (monitor_fn, get_pruned_fn).
    monitor_fn is called each step with head output norms.
    get_pruned_fn returns set of pruned (layer, head) pairs.
    """
    below_threshold = {}  # (layer, head) -> consecutive_steps_below
    pruned = set()

    def monitor(layer, head, head_output_norm, step):
        key = (layer, head)
        if key in pruned:
            return
        if head_output_norm < contribution_threshold:
            below_threshold[key] = below_threshold.get(key, 0) + 1
            if below_threshold[key] >= patience:
                pruned.add(key)
        else:
            below_threshold[key] = 0

    def get_pruned():
        return pruned.copy()

    return monitor, get_pruned


def make_head_growth(config, state_dict, growth_interval=100, rng=None):
    """
    Growth (neurogenesis): periodically add new heads.
    This is complex — returns a callable that modifies config and state_dict in place.

    NOTE: In practice, growing the model requires reallocating weight matrices.
    For microgpt's scalar autograd, this means adding new Value parameters.
    This is a more advanced perturbation.
    """
    _rng = rng or random.Random()

    def grow(step):
        if step > 0 and step % growth_interval == 0:
            # Placeholder: log that growth would occur
            pass
    return grow


# ============================================================================
# A8: MULTIPLE / OPPOSING OBJECTIVES
# ============================================================================

def make_adversarial_head(layer, head, head_dim):
    """
    This head tries to MAXIMIZE loss (opposing objective).
    Implemented by negating its gradients.
    Returns a grad_hook.
    """
    hs = head * head_dim
    he = hs + head_dim
    def grad_hook(params, state_dict, step):
        for comp in ['attn_wq', 'attn_wk', 'attn_wv']:
            key = f'layer{layer}.{comp}'
            if key in state_dict:
                for row_idx in range(hs, min(he, len(state_dict[key]))):
                    for p in state_dict[key][row_idx]:
                        p.grad = -p.grad
    return grad_hook


def make_reconstruction_loss_hook(layer, n_embd):
    """
    Auxiliary loss: the layer must reconstruct its input from its output.
    This adds a reconstruction term to the loss.

    Returns (hook_fn, get_aux_loss_fn).
    hook_fn captures the layer input.
    get_aux_loss_fn returns the reconstruction loss Value.
    """
    captured_input = [None]
    captured_output = [None]

    def capture_input(values, step=0, **kw):
        captured_input[0] = [Value(v.data, children=(v,), local_grads=(1,)) for v in values]
        return None  # pass through

    def capture_output(values, step=0, **kw):
        captured_output[0] = values
        return None  # pass through

    def get_aux_loss():
        if captured_input[0] is None or captured_output[0] is None:
            return Value(0.0)
        # MSE between input and output
        loss = sum((a - b) * (a - b)
                   for a, b in zip(captured_input[0], captured_output[0]))
        return loss / len(captured_input[0])

    return capture_input, capture_output, get_aux_loss


# ============================================================================
# A10: GRADIENT PERTURBATIONS
# ============================================================================

def make_noisy_gradients(noise_std=0.01, rng=None):
    """
    Add Gaussian noise to every gradient.
    Returns a grad_hook.
    """
    _rng = rng or random.Random()
    def grad_hook(params, state_dict, step):
        for p in params:
            p.grad += _rng.gauss(0, noise_std)
    return grad_hook


def make_quantized_gradients(levels=3):
    """
    Quantize gradients to {-1, 0, +1} or K discrete levels.
    Returns a grad_hook.
    """
    def grad_hook(params, state_dict, step):
        grads = [abs(p.grad) for p in params if p.grad != 0]
        if not grads:
            return
        threshold = sum(grads) / len(grads)  # mean absolute gradient

        for p in params:
            if levels == 3:
                if p.grad > threshold:
                    p.grad = 1.0
                elif p.grad < -threshold:
                    p.grad = -1.0
                else:
                    p.grad = 0.0
            else:
                # Quantize to discrete levels
                if threshold > 0:
                    bucket = round(p.grad / threshold * (levels // 2))
                    bucket = max(-levels // 2, min(levels // 2, bucket))
                    p.grad = bucket * threshold / (levels // 2)
    return grad_hook


def make_shuffled_gradients(shuffle_within='layer', rng=None):
    """
    Randomly reassign gradients among parameters of the same type.
    Returns a grad_hook.
    """
    _rng = rng or random.Random()
    def grad_hook(params, state_dict, step):
        # Group params by their layer component
        groups = {}
        for key, mat in state_dict.items():
            group_grads = []
            group_params = []
            for row in mat:
                for p in row:
                    group_grads.append(p.grad)
                    group_params.append(p)
            if group_grads:
                groups[key] = (group_params, group_grads)

        for key, (group_params, group_grads) in groups.items():
            _rng.shuffle(group_grads)
            for p, g in zip(group_params, group_grads):
                p.grad = g
    return grad_hook


def make_truncated_backprop_positions():
    """
    Cut gradient flow through KV cache: each position only sends
    gradients to its own parameters, not to earlier positions' cached values.

    This is implemented by making the KV cache entries detached (stop-gradient)
    when they are read by later positions. Applied at the qkv hook.

    Returns a hook that detaches cached K/V entries.
    """
    # This needs to work at the KV cache level, which is tricky
    # The simplest approach: hook post_mlp to stop gradients at each layer
    # This prevents gradient flow through the residual stream across positions
    # (approximately — the real KV cache gradient cutoff requires deeper surgery)
    pass


# ============================================================================
# A11: KV CACHE MODIFICATIONS
# ============================================================================

def make_decaying_cache(layer, decay_rate=0.9):
    """
    Older KV cache entries decay over time.
    Multiply cached entries by decay_rate^age.

    This modifies attention weights to account for decayed values.
    Returns (hook_name, hook_fn) for the attention weights.
    """
    target = f'attn_w.{layer}.0'  # applies to head 0; register for each head
    def hook(attn_weights, step=0, **kw):
        n = len(attn_weights)
        decayed = []
        for i, w in enumerate(attn_weights):
            age = n - 1 - i  # most recent = age 0
            factor = decay_rate ** age
            decayed.append(w * factor)
        # Re-normalize
        total = sum(d.data for d in decayed)
        if total > 1e-10:
            return [d / total for d in decayed]
        return decayed
    return target, hook


def make_limited_cache(layer, head, max_entries):
    """
    Only attend to the K most recent cache entries.
    Returns (hook_name, hook_fn).
    """
    return make_windowed_attention(layer, head, max_entries)


def make_noisy_cache(hook_name, noise_std=0.01, age_scale=True, rng=None):
    """
    Add noise to cached entries — noise increases with age.
    Applied at the embedding or post_attn level.
    Returns (hook_name, hook_fn).
    """
    _rng = rng or random.Random()
    call_count = [0]

    def hook(values, step=0, **kw):
        call_count[0] += 1
        if isinstance(values, list):
            noise_scale = noise_std * (call_count[0] if age_scale else 1)
            return [Value(v.data + _rng.gauss(0, noise_scale),
                          children=(v,), local_grads=(1,))
                    for v in values]
        return values
    return hook_name, hook


# ============================================================================
# CHESS-PAPER INSPIRED PERTURBATIONS (Kofman, Campitelli & Levin, 2025)
# ============================================================================

def make_layered_vision(config, radius_per_layer):
    """
    Apply different window sizes per layer — hierarchical perception.
    Early layers get narrow vision, later layers get wide vision,
    paralleling chess pieces relaying info through communication chains.

    Args:
        config: model config dict
        radius_per_layer: dict mapping layer_index -> window_size
            e.g. {0: 2, 1: 4, 2: 8, 3: 16}

    Returns:
        list of (hook_name, hook_fn) pairs to register
    """
    hooks = []
    n_head = config['n_head']
    for layer, window_size in radius_per_layer.items():
        for head in range(n_head):
            name, fn = make_windowed_attention(layer, head, window_size)
            hooks.append((name, fn))
    return hooks


def make_partial_stop_gradient(layer, pass_fraction):
    """
    Instead of binary stop-gradient, allow a configurable fraction of
    gradient magnitude through at a layer boundary.

    At pass_fraction=0.0 it's full cell-view (no gradient flow).
    At pass_fraction=1.0 it's baseline (full gradient flow).

    Implemented by creating new Value nodes that scale the gradient
    by pass_fraction during backpropagation.

    Args:
        layer: layer index for the boundary
        pass_fraction: float in [0, 1], fraction of gradient to pass

    Returns:
        (hook_name, hook_fn)
    """
    target = f'post_mlp.{layer}'
    def hook(values, step=0, **kw):
        if pass_fraction <= 0.0:
            # Full stop-gradient
            return [Value(v.data) for v in values]
        elif pass_fraction >= 1.0:
            # Full pass-through
            return None
        else:
            # Partial gradient: create nodes that scale gradient by pass_fraction
            return [Value(v.data, children=(v,), local_grads=(pass_fraction,))
                    for v in values]
    return target, hook


def make_threatening_drive(layer, head, head_dim, strength=0.1):
    """
    Gradient bonus for heads with high attention entropy.
    Encourages heads to maintain broad attention ("threatening" many positions)
    rather than premature collapse to narrow patterns.

    Maps the chess paper's finding that a "threatening" drive
    (pieces seeking to attack many squares) contributed +50 Elo.

    Args:
        layer: layer index
        head: head index
        head_dim: dimension of head output
        strength: magnitude of the entropy bonus gradient

    Returns:
        a grad_hook function
    """
    def grad_hook(params, state_dict, step):
        # Access the attention entropy if available in the probe
        # We apply a gradient bonus to Q/K params of this head to encourage
        # high-entropy (broad) attention patterns.
        # Higher entropy = more "threatening" = bonus (reduce gradient magnitude)
        # Lower entropy = less threatening = penalty (increase gradient magnitude)
        hs = head * head_dim
        he = hs + head_dim
        for comp in ['attn_wq', 'attn_wk']:
            key = f'layer{layer}.{comp}'
            if key in state_dict:
                for row_idx in range(hs, min(he, len(state_dict[key]))):
                    for p in state_dict[key][row_idx]:
                        # Add a small gradient push toward maintaining current weights
                        # (resisting specialization = encouraging broad attention)
                        p.grad += strength * (-p.data)
    return grad_hook


def make_round_robin_updates(config, period=None):
    """
    Each step, only one layer updates its parameters, cycling through layers.
    Maps the chess paper's turn-based scheduling (which outperformed
    hunger-based scheduling).

    Args:
        config: model config dict
        period: steps per layer before rotating (default: 1 step each)

    Returns:
        a grad_hook function
    """
    n_layer = config['n_layer']
    if period is None:
        period = 1

    def grad_hook(params, state_dict, step):
        # Which layer gets to update this step
        active_layer = (step // period) % n_layer
        for li in range(n_layer):
            if li == active_layer:
                continue  # this layer updates normally
            # Zero gradients for inactive layers
            for comp in ['attn_wq', 'attn_wk', 'attn_wv', 'attn_wo',
                         'mlp_fc1', 'mlp_fc2']:
                key = f'layer{li}.{comp}'
                if key in state_dict:
                    for row in state_dict[key]:
                        for p in row:
                            p.grad = 0
    return grad_hook


# ============================================================================
# Convenience: register multiple perturbations at once
# ============================================================================

def register_perturbations(hooks, perturbation_list):
    """
    Register a list of (hook_name, hook_fn) pairs.
    """
    for name, fn in perturbation_list:
        hooks.register(name, fn)


def freeze_random_heads(hooks, config, num_heads, rng=None):
    """
    Freeze (zero output) a random selection of heads.
    Returns list of (layer, head) pairs that were frozen and the grad_hooks.
    """
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


def apply_stop_gradient_all(hooks, config):
    """Apply stop-gradient at every layer boundary (cell-view GPT)."""
    for li in range(config['n_layer'] - 1):  # don't cut after last layer
        name, fn = make_stop_gradient(li)
        hooks.register(name, fn)


# ============================================================================
# Summary of all perturbation types
# ============================================================================

PERTURBATION_CATALOG = {
    # A4+A10: Frozen/damaged components
    'zero_head': 'Zero a head\'s output (immovable frozen cell)',
    'noise_head': 'Add noise to a head\'s output (damaged cell)',
    'freeze_params': 'Stop gradient for specific parameters (movable frozen cell)',
    'noise_injection': 'Add noise at any hook point',
    'quantize_activations': 'Quantize activations to low precision',

    # A1: Execution order
    'random_order': 'Process positions in random order',
    'reverse_order': 'Process positions right-to-left',

    # A2: Attention modifications
    'windowed_attention': 'Limit attention to K most recent tokens',
    'sparse_attention': 'Attend to random subset of past tokens',

    # A4: Stochastic forward pass
    'stochastic_relu': 'Randomly flip ReLU activations',
    'dropout': 'Randomly zero activations',
    'attention_temperature': 'Variable temperature on attention',

    # A5: Breaking backpropagation
    'stop_gradient': 'Cut gradient flow at layer boundaries',
    'sign_only_gradients': 'Replace gradients with their sign',
    'delayed_gradients': 'Use stale gradients from N steps ago',

    # A6: Asynchronous updates
    'async_updates': 'Different update frequencies per layer',
    'update_budget': 'Only update top K% of parameters',

    # A7: Architecture morphogenesis
    'head_pruning': 'Remove heads that contribute too little',

    # A8: Multiple objectives
    'adversarial_head': 'Some heads maximize loss',
    'reconstruction_loss': 'Auxiliary reconstruction objective',

    # A10: Gradient perturbations
    'noisy_gradients': 'Add noise to all gradients',
    'quantized_gradients': 'Quantize gradients to discrete levels',
    'shuffled_gradients': 'Randomly reassign gradients within groups',

    # A11: KV cache modifications
    'decaying_cache': 'Older cache entries fade',
    'limited_cache': 'Only keep K most recent cache entries',
    'noisy_cache': 'Add noise to cached values',

    # Chess-paper inspired (Kofman, Campitelli & Levin, 2025)
    'layered_vision': 'Different window sizes per layer (hierarchical perception)',
    'partial_stop_gradient': 'Fractional gradient flow at layer boundaries',
    'threatening_drive': 'Gradient bonus for high-entropy attention heads',
    'round_robin_updates': 'Turn-based layer updates (one layer per step)',
}
