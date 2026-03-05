"""
MorphoGPT — Metrics

Competence metrics, trajectory analysis, and Levin-style measurements.
Operates on Probe data collected during training.
"""

import math


# ============================================================================
# Delayed Gratification (Levin's core metric)
# ============================================================================

def compute_delayed_gratification(loss_values, min_increase=0.01, min_recovery=0.01):
    """
    Compute delayed gratification episodes from a loss trajectory.

    DG = the system temporarily moves AWAY from its goal (loss increases)
    then recovers past the pre-increase level. This is evidence of rerouting.

    Args:
        loss_values: list of float loss values (one per step)
        min_increase: minimum loss increase to count as a DG episode start
        min_recovery: minimum net recovery past pre-increase level

    Returns:
        list of dicts, each with:
            start: step where loss started increasing
            peak: step where loss peaked
            end: step where loss recovered past pre-increase level
            temporary_loss: peak - start_value (how much it got worse)
            net_gain: start_value - end_value (how much better than before)
            dg_index: net_gain / temporary_loss
    """
    if len(loss_values) < 3:
        return []

    episodes = []
    i = 0
    while i < len(loss_values) - 2:
        # Look for a loss increase
        if loss_values[i + 1] > loss_values[i] + min_increase:
            start = i
            start_val = loss_values[i]

            # Find the peak
            j = i + 1
            while j < len(loss_values) - 1 and loss_values[j + 1] >= loss_values[j]:
                j += 1
            peak = j
            peak_val = loss_values[j]

            # Find recovery (if any)
            k = j
            while k < len(loss_values) - 1 and loss_values[k] > start_val - min_recovery:
                k += 1

            if loss_values[k] < start_val - min_recovery:
                # Genuine DG episode: recovered past the pre-increase level
                temporary_loss = peak_val - start_val
                net_gain = start_val - loss_values[k]

                if temporary_loss > 0:
                    episodes.append({
                        'start': start,
                        'peak': peak,
                        'end': k,
                        'temporary_loss': temporary_loss,
                        'net_gain': net_gain,
                        'dg_index': net_gain / temporary_loss,
                    })
                i = k
            else:
                i = peak
        else:
            i += 1

    return episodes


def dg_index(loss_values, **kwargs):
    """
    Compute the aggregate DG index for a loss trajectory.
    Returns the mean DG index across all episodes, or 0 if none found.
    """
    episodes = compute_delayed_gratification(loss_values, **kwargs)
    if not episodes:
        return 0.0
    return sum(ep['dg_index'] for ep in episodes) / len(episodes)


def dg_count(loss_values, **kwargs):
    """Count of DG episodes in a loss trajectory."""
    return len(compute_delayed_gratification(loss_values, **kwargs))


# ============================================================================
# Rerouting Score
# ============================================================================

def rerouting_score(grad_norms_before, grad_norms_after):
    """
    Measure how gradient distribution shifts after damage.

    When a component is frozen and the system compensates, other components
    should show increased gradient flow (they're "taking over").

    Args:
        grad_norms_before: dict of component_name -> list of gradient norms (pre-damage)
        grad_norms_after: dict of component_name -> list of gradient norms (post-damage)

    Returns:
        dict of component_name -> relative_change (positive = increased gradient flow)
    """
    scores = {}
    for name in grad_norms_after:
        if name in grad_norms_before:
            before_avg = sum(grad_norms_before[name]) / max(1, len(grad_norms_before[name]))
            after_avg = sum(grad_norms_after[name]) / max(1, len(grad_norms_after[name]))
            if before_avg > 1e-10:
                scores[name] = (after_avg - before_avg) / before_avg
            else:
                scores[name] = float('inf') if after_avg > 1e-10 else 0.0
    return scores


# ============================================================================
# Recovery Dynamics
# ============================================================================

def recovery_time(loss_values, damage_step, baseline_loss, tolerance=0.05):
    """
    How many steps after damage until loss returns to within tolerance of baseline.

    Args:
        loss_values: full loss trajectory
        damage_step: step at which damage was introduced
        baseline_loss: the loss level to recover to
        tolerance: acceptable deviation from baseline

    Returns:
        Number of steps to recover, or -1 if never recovered.
    """
    if damage_step >= len(loss_values):
        return -1

    target = baseline_loss * (1 + tolerance)
    for i in range(damage_step, len(loss_values)):
        if loss_values[i] <= target:
            return i - damage_step
    return -1


def loss_slope(loss_values, window=20):
    """
    Compute the slope of loss over a sliding window.
    Returns list of (step, slope) pairs.
    Negative slope = improving. Positive = degrading.
    """
    if len(loss_values) < window:
        return []
    slopes = []
    for i in range(len(loss_values) - window):
        segment = loss_values[i:i+window]
        # Simple linear regression slope
        n = len(segment)
        x_mean = (n - 1) / 2.0
        y_mean = sum(segment) / n
        numer = sum((j - x_mean) * (segment[j] - y_mean) for j in range(n))
        denom = sum((j - x_mean) ** 2 for j in range(n))
        slope = numer / denom if denom > 0 else 0
        slopes.append((i + window // 2, slope))
    return slopes


def loss_variance(loss_values, window=20):
    """
    Compute variance of loss over a sliding window.
    Returns list of (step, variance) pairs.
    """
    if len(loss_values) < window:
        return []
    variances = []
    for i in range(len(loss_values) - window):
        segment = loss_values[i:i+window]
        mean = sum(segment) / len(segment)
        var = sum((x - mean) ** 2 for x in segment) / len(segment)
        variances.append((i + window // 2, var))
    return variances


# ============================================================================
# Robustness Curve
# ============================================================================

def robustness_curve(results_by_damage_level):
    """
    Compute a robustness curve: final loss as a function of damage level.

    Args:
        results_by_damage_level: dict of damage_level -> list of final_loss values
            (multiple runs per damage level for statistics)

    Returns:
        list of (damage_level, mean_loss, std_loss, mean_dg, std_dg)
    """
    curve = []
    for level in sorted(results_by_damage_level.keys()):
        losses = results_by_damage_level[level]
        n = len(losses)
        mean_l = sum(losses) / n
        std_l = (sum((x - mean_l) ** 2 for x in losses) / max(1, n - 1)) ** 0.5
        curve.append({
            'damage_level': level,
            'mean_loss': mean_l,
            'std_loss': std_l,
            'n': n,
        })
    return curve


def robustness_curve_with_dg(results_by_damage_level):
    """
    Compute robustness curve including DG index.

    Args:
        results_by_damage_level: dict of damage_level -> list of
            (final_loss, loss_trajectory) tuples

    Returns:
        list of dicts with damage_level, mean_loss, std_loss, mean_dg, std_dg
    """
    curve = []
    for level in sorted(results_by_damage_level.keys()):
        entries = results_by_damage_level[level]
        losses = [e[0] for e in entries]
        dg_indices = [dg_index(e[1]) for e in entries]

        n = len(losses)
        mean_l = sum(losses) / n
        std_l = (sum((x - mean_l) ** 2 for x in losses) / max(1, n - 1)) ** 0.5
        mean_dg = sum(dg_indices) / n
        std_dg = (sum((x - mean_dg) ** 2 for x in dg_indices) / max(1, n - 1)) ** 0.5

        curve.append({
            'damage_level': level,
            'mean_loss': mean_l,
            'std_loss': std_l,
            'mean_dg': mean_dg,
            'std_dg': std_dg,
            'n': n,
        })
    return curve


# ============================================================================
# Aggregation (for chimeric experiments)
# ============================================================================

def aggregation_index(activations_by_type):
    """
    Measure whether components of the same "algotype" produce more similar
    activation patterns than components of different types.

    Args:
        activations_by_type: dict of type_label -> list of activation vectors
            Each activation vector is a list of floats.

    Returns:
        aggregation_index: ratio of mean same-type similarity to mean cross-type similarity.
        > 1 means same-type components are more similar (aggregation).
    """
    def cosine_sim(a, b):
        dot = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = sum(ai ** 2 for ai in a) ** 0.5
        norm_b = sum(bi ** 2 for bi in b) ** 0.5
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return dot / (norm_a * norm_b)

    types = list(activations_by_type.keys())
    if len(types) < 2:
        return 1.0

    # Same-type similarities
    same_sims = []
    for t in types:
        acts = activations_by_type[t]
        for i in range(len(acts)):
            for j in range(i + 1, len(acts)):
                same_sims.append(cosine_sim(acts[i], acts[j]))

    # Cross-type similarities
    cross_sims = []
    for i, t1 in enumerate(types):
        for j, t2 in enumerate(types):
            if i >= j:
                continue
            for a1 in activations_by_type[t1]:
                for a2 in activations_by_type[t2]:
                    cross_sims.append(cosine_sim(a1, a2))

    mean_same = sum(same_sims) / max(1, len(same_sims))
    mean_cross = sum(cross_sims) / max(1, len(cross_sims))

    if abs(mean_cross) < 1e-10:
        return float('inf') if mean_same > 0 else 1.0
    return mean_same / mean_cross


# ============================================================================
# Head Contribution Analysis
# ============================================================================

def head_contribution_scores(probe):
    """
    Compute each head's average output norm from probe data.
    Returns dict of (layer, head) -> mean_norm.
    """
    scores = {}
    for (layer, head), entries in probe.head_outputs.items():
        norms = [n for _, n in entries]
        scores[(layer, head)] = sum(norms) / max(1, len(norms))
    return scores


def attention_entropy_analysis(probe):
    """
    Compute each head's average attention entropy from probe data.
    Returns dict of (layer, head) -> mean_entropy.
    """
    results = {}
    for (layer, head), entries in probe.attention_entropies.items():
        entropies = [e for _, e in entries]
        results[(layer, head)] = sum(entropies) / max(1, len(entropies))
    return results


# ============================================================================
# Statistical tests
# ============================================================================

def dg_damage_regression(damage_levels, dg_indices):
    """
    Test whether DG index increases with damage level.
    Simple linear regression of DG on damage level.

    Returns (slope, intercept, r_squared).
    Positive slope with high R² = evidence for genuine delayed gratification.
    """
    n = len(damage_levels)
    if n < 2:
        return 0, 0, 0

    x_mean = sum(damage_levels) / n
    y_mean = sum(dg_indices) / n

    numer = sum((x - x_mean) * (y - y_mean) for x, y in zip(damage_levels, dg_indices))
    denom_x = sum((x - x_mean) ** 2 for x in damage_levels)
    denom_y = sum((y - y_mean) ** 2 for y in dg_indices)

    if denom_x < 1e-10:
        return 0, y_mean, 0

    slope = numer / denom_x
    intercept = y_mean - slope * x_mean

    ss_res = sum((y - (slope * x + intercept)) ** 2
                 for x, y in zip(damage_levels, dg_indices))
    ss_tot = denom_y
    r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0

    return slope, intercept, r_squared


# ============================================================================
# Summary statistics
# ============================================================================

def summarize_probe(probe):
    """
    Create a summary dict from a Probe object.
    """
    loss_vals = probe.get_loss_values()

    summary = {
        'num_steps': len(loss_vals),
        'final_loss': loss_vals[-1] if loss_vals else None,
        'min_loss': min(loss_vals) if loss_vals else None,
        'mean_loss': sum(loss_vals) / len(loss_vals) if loss_vals else None,
    }

    # DG analysis
    dg_eps = compute_delayed_gratification(loss_vals)
    summary['dg_count'] = len(dg_eps)
    summary['dg_index'] = (sum(ep['dg_index'] for ep in dg_eps) / len(dg_eps)
                           if dg_eps else 0.0)

    # Loss dynamics
    slopes = loss_slope(loss_vals)
    if slopes:
        final_slopes = [s for _, s in slopes[-20:]]
        summary['final_slope'] = sum(final_slopes) / len(final_slopes)

    variances = loss_variance(loss_vals)
    if variances:
        summary['loss_variance'] = variances[-1][1]

    return summary


# ============================================================================
# Trajectory Analysis — Levin-style trajectory-focused metrics
# ============================================================================

def trajectory_envelope(trajectories):
    """
    Compute min/max/mean/std at each step across multiple runs.
    Levin's Figure 3 analog: the shape of how multiple runs explore problem space.

    Args:
        trajectories: list of lists, each inner list is a loss trajectory

    Returns:
        list of dicts, one per step:
            {'step': i, 'min': ..., 'max': ..., 'mean': ..., 'std': ...}
    """
    if not trajectories:
        return []

    max_len = max(len(t) for t in trajectories)
    envelope = []

    for i in range(max_len):
        vals = [t[i] for t in trajectories if i < len(t)]
        if not vals:
            continue
        n = len(vals)
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / max(1, n - 1)) ** 0.5 if n > 1 else 0.0
        envelope.append({
            'step': i,
            'min': min(vals),
            'max': max(vals),
            'mean': mean,
            'std': std,
            'n': n,
        })

    return envelope


def compare_trajectory_envelopes(env_a, env_b):
    """
    Compare two sets of trajectory envelopes.
    Are the trajectories shaped differently?

    Returns dict with:
        mean_divergence: average absolute difference in means
        max_divergence: maximum absolute difference in means
        divergence_step: step at which max divergence occurs
        shape_correlation: correlation of mean curves
        overlap_fraction: fraction of steps where confidence bands overlap
    """
    n = min(len(env_a), len(env_b))
    if n == 0:
        return {'mean_divergence': 0, 'max_divergence': 0,
                'divergence_step': 0, 'shape_correlation': 0,
                'overlap_fraction': 1.0}

    diffs = []
    overlaps = 0
    means_a = []
    means_b = []

    for i in range(n):
        a, b = env_a[i], env_b[i]
        diff = abs(a['mean'] - b['mean'])
        diffs.append(diff)
        means_a.append(a['mean'])
        means_b.append(b['mean'])

        # Check if mean +/- 1 std bands overlap
        a_lo, a_hi = a['mean'] - a['std'], a['mean'] + a['std']
        b_lo, b_hi = b['mean'] - b['std'], b['mean'] + b['std']
        if a_lo <= b_hi and b_lo <= a_hi:
            overlaps += 1

    mean_div = sum(diffs) / n
    max_div = max(diffs)
    max_step = diffs.index(max_div)

    # Pearson correlation of mean curves
    ma = sum(means_a) / n
    mb = sum(means_b) / n
    numer = sum((a - ma) * (b - mb) for a, b in zip(means_a, means_b))
    denom_a = sum((a - ma) ** 2 for a in means_a) ** 0.5
    denom_b = sum((b - mb) ** 2 for b in means_b) ** 0.5
    corr = numer / (denom_a * denom_b) if denom_a > 1e-10 and denom_b > 1e-10 else 0.0

    return {
        'mean_divergence': mean_div,
        'max_divergence': max_div,
        'divergence_step': max_step,
        'shape_correlation': corr,
        'overlap_fraction': overlaps / n,
    }


# ============================================================================
# Per-step Rerouting Analysis
# ============================================================================

def per_step_rerouting(probe):
    """
    Which heads show increased contribution step-over-step after damage?

    Analyzes head norm trajectories from probe to find heads whose
    contribution increases over time (evidence of compensation/rerouting).

    Returns:
        list of dicts per head:
            {'layer': li, 'head': hi, 'slope': ..., 'mean_norm': ..., 'increase_ratio': ...}
        sorted by slope (most increasing first)
    """
    results = []
    for (li, hi), entries in probe.head_outputs.items():
        if len(entries) < 2:
            continue
        norms = [n for _, n in entries]
        steps = [s for s, _ in entries]
        n = len(norms)
        mean_norm = sum(norms) / n

        # Linear regression of norm on step
        x_mean = sum(steps) / n
        y_mean = mean_norm
        numer = sum((s - x_mean) * (norm - y_mean) for s, norm in zip(steps, norms))
        denom = sum((s - x_mean) ** 2 for s in steps)
        slope = numer / denom if denom > 1e-10 else 0.0

        first_norm = sum(norms[:max(1, n // 5)]) / max(1, n // 5)
        last_norm = sum(norms[-max(1, n // 5):]) / max(1, n // 5)
        increase_ratio = last_norm / first_norm if first_norm > 1e-10 else 0.0

        results.append({
            'layer': li, 'head': hi,
            'slope': slope, 'mean_norm': mean_norm,
            'increase_ratio': increase_ratio,
        })

    results.sort(key=lambda x: x['slope'], reverse=True)
    return results


def rerouting_matrix(baseline_probe, damaged_probe, frozen_heads):
    """
    For each frozen head, identify which heads compensate (increase norm).

    Args:
        baseline_probe: Probe from undamaged training
        damaged_probe: Probe from training with frozen heads
        frozen_heads: list of (layer, head) that were frozen

    Returns:
        dict: frozen (layer, head) -> list of compensating heads with increase
    """
    frozen_set = set(frozen_heads)

    # Get mean norms per head from each probe
    def mean_norms(probe):
        result = {}
        for key, entries in probe.head_outputs.items():
            norms = [n for _, n in entries]
            result[key] = sum(norms) / len(norms) if norms else 0.0
        return result

    baseline_norms = mean_norms(baseline_probe)
    damaged_norms = mean_norms(damaged_probe)

    # Find which non-frozen heads increased
    compensators = []
    for key in damaged_norms:
        if key in frozen_set:
            continue
        baseline_n = baseline_norms.get(key, 0.0)
        damaged_n = damaged_norms.get(key, 0.0)
        if baseline_n > 1e-10:
            increase = (damaged_n - baseline_n) / baseline_n
            if increase > 0.05:  # >5% increase
                compensators.append({
                    'layer': key[0], 'head': key[1],
                    'baseline_norm': baseline_n,
                    'damaged_norm': damaged_n,
                    'increase': increase,
                })

    compensators.sort(key=lambda x: x['increase'], reverse=True)

    # Group by frozen head (attribute to nearest frozen head by layer)
    matrix = {}
    for fh in frozen_heads:
        matrix[fh] = compensators  # simplified: all compensators listed for each frozen head

    return matrix


# ============================================================================
# Phase Detection
# ============================================================================

def detect_phases(loss_values, window=20):
    """
    Identify regime changes in a loss trajectory:
    rapid descent, plateau, DG episode, convergence.

    Returns list of phases:
        {'type': str, 'start': int, 'end': int, 'mean_slope': float, 'mean_loss': float}
    """
    if len(loss_values) < window * 2:
        return [{'type': 'short_run', 'start': 0, 'end': len(loss_values) - 1,
                 'mean_slope': 0, 'mean_loss': sum(loss_values) / len(loss_values)}]

    slopes = loss_slope(loss_values, window)
    if not slopes:
        return []

    phases = []
    i = 0
    slope_vals = [s for _, s in slopes]
    slope_steps = [st for st, _ in slopes]

    while i < len(slope_vals):
        # Classify current region
        j = i
        slope = slope_vals[i]

        if slope < -0.005:
            # Rapid descent
            phase_type = 'rapid_descent'
            while j < len(slope_vals) and slope_vals[j] < -0.002:
                j += 1
        elif slope < -0.002:
            # Gradual descent (mild negative slopes)
            phase_type = 'gradual_descent'
            while j < len(slope_vals) and -0.005 <= slope_vals[j] < -0.001:
                j += 1
        elif slope > 0.002:
            # Loss increasing (possible DG episode start)
            phase_type = 'loss_increase'
            while j < len(slope_vals) and slope_vals[j] > 0.001:
                j += 1
        else:
            # Plateau / convergence
            phase_type = 'plateau'
            while j < len(slope_vals) and abs(slope_vals[j]) <= 0.003:
                j += 1
            if j == i:
                j += 1

        start_step = slope_steps[i]
        end_step = slope_steps[min(j - 1, len(slope_steps) - 1)]
        segment = loss_values[start_step:end_step + 1]
        mean_slope = sum(slope_vals[i:j]) / max(1, j - i)
        mean_loss = sum(segment) / len(segment) if segment else 0

        phases.append({
            'type': phase_type,
            'start': start_step,
            'end': end_step,
            'mean_slope': mean_slope,
            'mean_loss': mean_loss,
        })
        i = j

    return phases


def detect_regime_changes(trajectory, window=20, threshold=2.0):
    """
    Generic change-point detection using sliding window variance ratio.

    Args:
        trajectory: list of float values
        window: window size for comparison
        threshold: how many std devs to count as a regime change

    Returns:
        list of step indices where regime changes are detected
    """
    if len(trajectory) < window * 3:
        return []

    changes = []
    for i in range(window, len(trajectory) - window):
        before = trajectory[i - window:i]
        after = trajectory[i:i + window]

        mean_before = sum(before) / window
        mean_after = sum(after) / window
        var_before = sum((x - mean_before) ** 2 for x in before) / window
        std_before = var_before ** 0.5

        if std_before > 1e-10:
            z = abs(mean_after - mean_before) / std_before
            if z > threshold:
                # Avoid detecting the same change multiple times
                if not changes or i - changes[-1] > window:
                    changes.append(i)

    return changes


# ============================================================================
# Mid-Trajectory Phenomena
# ============================================================================

def mid_trajectory_peak_detection(metric_trajectory, progress_range=(0.15, 0.50)):
    """
    Search for peaks at 15-50% through training (Levin's aggregation peak analog).

    Args:
        metric_trajectory: list of (step, value) pairs
        progress_range: (start_frac, end_frac) — where to look

    Returns:
        list of peaks found: {'step': ..., 'value': ..., 'progress': ...}
    """
    if not metric_trajectory:
        return []

    steps = [s for s, _ in metric_trajectory]
    values = [v for _, v in metric_trajectory]
    total_steps = max(steps) if steps else 1

    peaks = []
    for i in range(1, len(values) - 1):
        progress = steps[i] / total_steps if total_steps > 0 else 0
        if progress_range[0] <= progress <= progress_range[1]:
            if values[i] > values[i - 1] and values[i] > values[i + 1]:
                peaks.append({
                    'step': steps[i],
                    'value': values[i],
                    'progress': progress,
                })

    return peaks


def head_contribution_evolution(probe):
    """
    Per-head contribution fraction over time.
    Shows how the relative importance of heads shifts during training.

    Returns:
        dict of (layer, head) -> [(step, fraction), ...]
    """
    # Collect all steps that have head norm data
    step_to_norms = {}
    for (li, hi), entries in probe.head_outputs.items():
        for step, norm in entries:
            step_to_norms.setdefault(step, {})[( li, hi)] = norm

    evolution = {}
    for step in sorted(step_to_norms.keys()):
        norms = step_to_norms[step]
        total = sum(norms.values())
        if total < 1e-10:
            continue
        for key, norm in norms.items():
            evolution.setdefault(key, []).append((step, norm / total))

    return evolution


def attention_pattern_evolution(probe, layer, head):
    """
    Entropy, max attention position, and stability over time for one head.

    Returns:
        list of dicts per recorded step:
            {'step': ..., 'entropy': ..., 'max_weight': ..., 'max_position': ...}
    """
    results = []
    for entry in probe.step_data:
        if 'head_entropies' not in entry:
            continue
        key = (layer, head)
        if key not in entry.get('head_entropies', {}):
            continue

        step_result = {
            'step': entry['step'],
            'entropy': entry['head_entropies'][key],
            'norm': entry.get('head_norms', {}).get(key, 0.0),
        }

        # If full snapshots available, extract max attention position
        if 'snapshots' in entry and entry['snapshots']:
            snap = entry['snapshots'][0]  # first position
            if layer < len(snap['layers']):
                head_data = snap['layers'][layer]['heads'][head]
                weights = head_data['attn_weights']
                if weights:
                    max_w = max(weights)
                    step_result['max_weight'] = max_w
                    step_result['max_position'] = weights.index(max_w)

        results.append(step_result)

    return results


# ============================================================================
# Anomaly Detection — emergent phenomena not specified by the algorithm
# ============================================================================

def detect_sync_events(probe, threshold=2.0):
    """
    Find steps where multiple heads spike or drop norm simultaneously.
    Each head learns independently through backprop, so correlated changes
    suggest emergent coordination through the shared loss landscape.

    Returns list of {'step', 'num_heads', 'direction', 'heads', 'type'}.
    """
    head_keys = sorted(probe.head_outputs.keys())
    if len(head_keys) < 2:
        return []

    # Collect per-head deltas at each step
    head_deltas = {}
    for key in head_keys:
        entries = probe.head_outputs[key]
        norms = [n for _, n in entries]
        steps = [s for s, _ in entries]
        for i in range(1, len(norms)):
            head_deltas.setdefault(key, []).append((steps[i], norms[i] - norms[i - 1]))

    # For each step, check how many heads have large deltas
    step_to_deltas = {}
    for key, deltas in head_deltas.items():
        for step, d in deltas:
            step_to_deltas.setdefault(step, {})[key] = d

    events = []
    for step in sorted(step_to_deltas.keys()):
        deltas = step_to_deltas[step]
        if len(deltas) < 2:
            continue
        vals = list(deltas.values())
        mean_d = sum(vals) / len(vals)
        std_d = (sum((v - mean_d) ** 2 for v in vals) / len(vals)) ** 0.5
        if std_d < 1e-10:
            continue

        z_scores = {k: (v - mean_d) / std_d for k, v in deltas.items()}
        spiking = [k for k, z in z_scores.items() if abs(z) > threshold]
        if len(spiking) >= max(2, len(head_keys) // 2):
            direction = 'up' if sum(deltas[k] for k in spiking) > 0 else 'down'
            events.append({
                'step': step,
                'num_heads': len(spiking),
                'direction': direction,
                'heads': spiking,
                'type': 'sync_event',
            })
    return events


def detect_ghost_spikes(probe):
    """
    Head norm spikes at steps where loss is flat or improving.
    The head is reorganizing internally without it showing in the loss.

    Returns list of {'step', 'head', 'norm_change', 'loss_change', 'type'}.
    """
    loss_vals = probe.get_loss_values()
    if len(loss_vals) < 3:
        return []

    loss_by_step = {s: l for s, l in probe.losses}

    events = []
    for key in sorted(probe.head_outputs.keys()):
        entries = probe.head_outputs[key]
        norms = [n for _, n in entries]
        steps = [s for s, _ in entries]
        if len(norms) < 3:
            continue

        mean_n = sum(norms) / len(norms)
        std_n = (sum((n - mean_n) ** 2 for n in norms) / len(norms)) ** 0.5
        if std_n < 1e-10:
            continue

        for i in range(1, len(norms) - 1):
            norm_delta = norms[i] - norms[i - 1]
            z = norm_delta / std_n
            if z > 2.0:
                step = steps[i]
                if step in loss_by_step and steps[i - 1] in loss_by_step:
                    loss_delta = loss_by_step[step] - loss_by_step[steps[i - 1]]
                    if loss_delta <= 0:
                        events.append({
                            'step': step,
                            'head': key,
                            'norm_change': norm_delta,
                            'loss_change': loss_delta,
                            'type': 'ghost_spike',
                        })
    return events


def detect_sudden_specialization(probe, entropy_drop_threshold=0.3):
    """
    A head with stable high entropy suddenly drops (sharpens attention).
    Like a latent capability activating unprompted.

    Returns list of {'step', 'head', 'entropy_before', 'entropy_after', 'drop', 'type'}.
    """
    events = []
    for key in sorted(probe.attention_entropies.keys()):
        entries = probe.attention_entropies[key]
        entropies = [e for _, e in entries]
        steps = [s for s, _ in entries]
        if len(entropies) < 5:
            continue

        window = max(3, len(entropies) // 10)
        for i in range(window, len(entropies)):
            before = entropies[i - window:i]
            mean_before = sum(before) / len(before)
            std_before = (sum((e - mean_before) ** 2 for e in before) / len(before)) ** 0.5

            if mean_before > 0.5 and std_before < 0.2:
                drop = mean_before - entropies[i]
                if drop > entropy_drop_threshold:
                    events.append({
                        'step': steps[i],
                        'head': key,
                        'entropy_before': mean_before,
                        'entropy_after': entropies[i],
                        'drop': drop,
                        'type': 'sudden_specialization',
                    })
    return events


def detect_role_reversals(probe):
    """
    Heads swapping contribution rankings mid-training.
    Suggests spontaneous reorganization of computational roles.

    Returns list of {'step', 'head_rising', 'head_falling', 'type'}.
    """
    evolution = head_contribution_evolution(probe)
    if len(evolution) < 2:
        return []

    heads = sorted(evolution.keys())
    if len(heads) < 2:
        return []

    all_steps = set()
    for fracs in evolution.values():
        for s, _ in fracs:
            all_steps.add(s)
    all_steps = sorted(all_steps)

    frac_at = {}
    for head, fracs in evolution.items():
        for s, f in fracs:
            frac_at[(head, s)] = f

    events = []
    for i in range(1, len(all_steps)):
        step = all_steps[i]
        prev_step = all_steps[i - 1]

        rank_now = sorted(heads, key=lambda h: frac_at.get((h, step), 0), reverse=True)
        rank_prev = sorted(heads, key=lambda h: frac_at.get((h, prev_step), 0), reverse=True)

        if rank_now[0] != rank_prev[0]:
            rising = rank_now[0]
            falling = rank_prev[0]
            frac_rising_now = frac_at.get((rising, step), 0)
            frac_falling_now = frac_at.get((falling, step), 0)
            if frac_rising_now - frac_falling_now > 0.02:
                events.append({
                    'step': step,
                    'head_rising': rising,
                    'head_falling': falling,
                    'type': 'role_reversal',
                })
    return events


def detect_gradient_divergence(probe):
    """
    Gradient norms spiking without corresponding loss changes.
    The model is restructuring internally without affecting output yet.

    Returns list of {'step', 'group', 'grad_spike', 'loss_change', 'type'}.
    """
    loss_by_step = dict(probe.losses)
    events = []

    for group, entries in probe.grad_norms.items():
        norms = [n for _, n in entries]
        steps = [s for s, _ in entries]
        if len(norms) < 5:
            continue

        mean_n = sum(norms) / len(norms)
        std_n = (sum((n - mean_n) ** 2 for n in norms) / len(norms)) ** 0.5
        if std_n < 1e-10:
            continue

        for i in range(1, len(norms)):
            z = (norms[i] - mean_n) / std_n
            if z > 2.0:
                step = steps[i]
                prev_step = steps[i - 1]
                if step in loss_by_step and prev_step in loss_by_step:
                    loss_change = abs(loss_by_step[step] - loss_by_step[prev_step])
                    loss_std = _loss_local_std(probe, step)
                    if loss_std > 1e-10 and loss_change / loss_std < 1.0:
                        events.append({
                            'step': step,
                            'group': group,
                            'grad_spike': norms[i],
                            'loss_change': loss_change,
                            'type': 'gradient_divergence',
                        })
    return events


def _loss_local_std(probe, step, window=10):
    """Helper: local std of loss around a step."""
    loss_vals = probe.get_loss_values()
    loss_steps = [s for s, _ in probe.losses]
    try:
        idx = loss_steps.index(step)
    except ValueError:
        return 1.0
    start = max(0, idx - window)
    end = min(len(loss_vals), idx + window)
    segment = loss_vals[start:end]
    if len(segment) < 2:
        return 1.0
    mean = sum(segment) / len(segment)
    return (sum((x - mean) ** 2 for x in segment) / len(segment)) ** 0.5


def detect_periodicity(trajectory, min_period=10, max_period=None):
    """
    Detect periodic patterns via autocorrelation.
    SGD on random batches has no built-in periodicity, so any regularity is emergent.

    Returns list of {'period', 'strength'} for significant periodicities.
    """
    n = len(trajectory)
    if n < min_period * 3:
        return []

    if max_period is None:
        max_period = n // 3

    mean = sum(trajectory) / n
    var = sum((x - mean) ** 2 for x in trajectory) / n
    if var < 1e-10:
        return []

    results = []
    for lag in range(min_period, min(max_period + 1, n // 2)):
        autocorr = sum(
            (trajectory[i] - mean) * (trajectory[i + lag] - mean)
            for i in range(n - lag)
        ) / ((n - lag) * var)

        if autocorr > 0.3:
            is_peak = True
            if lag > min_period:
                prev_ac = sum(
                    (trajectory[i] - mean) * (trajectory[i + lag - 1] - mean)
                    for i in range(n - lag + 1)
                ) / ((n - lag + 1) * var)
                if prev_ac > autocorr:
                    is_peak = False
            if lag < max_period and lag + 1 < n // 2:
                next_ac = sum(
                    (trajectory[i] - mean) * (trajectory[i + lag + 1] - mean)
                    for i in range(n - lag - 1)
                ) / ((n - lag - 1) * var)
                if next_ac > autocorr:
                    is_peak = False

            if is_peak:
                results.append({'period': lag, 'strength': autocorr})

    return results


# ============================================================================
# Chess-Paper Inspired Metrics (Kofman, Campitelli & Levin, 2025)
# ============================================================================

def cognitive_light_cone(probe, layer, head):
    """
    The spatiotemporal radius an agent (head) can effectively work toward.

    Spatial component: Mean attention entropy — how broadly the head
    distributes attention across context positions. Higher entropy means
    the head "sees" more of the context (larger spatial radius).

    Temporal component: Autocorrelation length of head norms — how many
    steps a perturbation's effect persists. Longer autocorrelation means
    the head's state has longer temporal reach.

    Returns:
        dict with 'spatial', 'temporal', 'combined' (product)
    """
    # Spatial: mean attention entropy for this head
    spatial = 0.0
    key = (layer, head)
    if key in probe.attention_entropies:
        entries = probe.attention_entropies[key]
        entropies = [e for _, e in entries]
        if entropies:
            spatial = sum(entropies) / len(entropies)

    # Temporal: autocorrelation length of head norms
    temporal = 0.0
    if key in probe.head_outputs:
        entries = probe.head_outputs[key]
        norms = [n for _, n in entries]
        if len(norms) >= 10:
            # Compute autocorrelation at increasing lags until it drops below 0.5
            n = len(norms)
            mean_n = sum(norms) / n
            var_n = sum((x - mean_n) ** 2 for x in norms) / n
            if var_n > 1e-10:
                for lag in range(1, n // 2):
                    ac = sum(
                        (norms[i] - mean_n) * (norms[i + lag] - mean_n)
                        for i in range(n - lag)
                    ) / ((n - lag) * var_n)
                    if ac < 0.5:
                        # Interpolate the exact crossing point
                        temporal = lag - 1 + (0.5 - ac) if lag > 1 else lag
                        break
                else:
                    temporal = n // 2  # never dropped below threshold

    return {
        'spatial': spatial,
        'temporal': temporal,
        'combined': spatial * temporal,
    }


def collective_light_cone(probe):
    """
    Aggregate cognitive light cone across all heads.

    Returns:
        dict with 'mean_spatial', 'mean_temporal', 'mean_combined',
        'max_spatial', 'max_temporal', 'per_head' details
    """
    all_heads = sorted(set(probe.head_outputs.keys()) | set(probe.attention_entropies.keys()))
    if not all_heads:
        return {
            'mean_spatial': 0.0, 'mean_temporal': 0.0, 'mean_combined': 0.0,
            'max_spatial': 0.0, 'max_temporal': 0.0, 'per_head': {},
        }

    per_head = {}
    for (layer, head) in all_heads:
        per_head[(layer, head)] = cognitive_light_cone(probe, layer, head)

    spatials = [v['spatial'] for v in per_head.values()]
    temporals = [v['temporal'] for v in per_head.values()]
    combineds = [v['combined'] for v in per_head.values()]

    return {
        'mean_spatial': sum(spatials) / len(spatials),
        'mean_temporal': sum(temporals) / len(temporals),
        'mean_combined': sum(combineds) / len(combineds),
        'max_spatial': max(spatials),
        'max_temporal': max(temporals),
        'per_head': per_head,
    }


def goal_alignment_score(probe):
    """
    Emergent checkmate analog: do individual head drives align with
    the collective goal (loss reduction)?

    For each head, correlate step-over-step norm change with step-over-step
    loss change. Negative correlation = head growth aligned with loss
    reduction (the head is contributing to the collective goal).

    Returns:
        dict with 'mean_alignment', 'per_head' details
        per_head: dict of (layer, head) -> correlation value
        Negative = aligned, positive = misaligned
    """
    loss_vals = probe.get_loss_values()
    loss_steps = [s for s, _ in probe.losses]
    if len(loss_vals) < 3:
        return {'mean_alignment': 0.0, 'per_head': {}}

    # Build loss delta by step
    loss_delta = {}
    for i in range(1, len(loss_steps)):
        loss_delta[loss_steps[i]] = loss_vals[i] - loss_vals[i - 1]

    per_head = {}
    for (li, hi), entries in probe.head_outputs.items():
        if len(entries) < 3:
            continue
        norms = [n for _, n in entries]
        steps = [s for s, _ in entries]

        # Compute norm deltas aligned with loss deltas
        norm_deltas = []
        loss_deltas = []
        for i in range(1, len(norms)):
            step = steps[i]
            if step in loss_delta:
                norm_deltas.append(norms[i] - norms[i - 1])
                loss_deltas.append(loss_delta[step])

        if len(norm_deltas) < 3:
            continue

        # Pearson correlation
        n = len(norm_deltas)
        mx = sum(norm_deltas) / n
        my = sum(loss_deltas) / n
        numer = sum((a - mx) * (b - my) for a, b in zip(norm_deltas, loss_deltas))
        dx = sum((a - mx) ** 2 for a in norm_deltas) ** 0.5
        dy = sum((b - my) ** 2 for b in loss_deltas) ** 0.5
        corr = numer / (dx * dy) if dx > 1e-10 and dy > 1e-10 else 0.0
        per_head[(li, hi)] = corr

    mean_alignment = (sum(per_head.values()) / len(per_head)) if per_head else 0.0

    return {
        'mean_alignment': mean_alignment,
        'per_head': per_head,
    }


def swarming_index(probe, loss_threshold_percentile=75):
    """
    Offensive swarming analog: do multiple heads converge on the same
    patterns during difficult steps?

    During high-loss steps (above percentile threshold), measure pairwise
    overlap of head norm patterns across heads. Compare to overlap during
    low-loss steps. Ratio > 1 means heads converge during difficulty.

    Returns:
        dict with 'swarming_ratio', 'high_loss_overlap', 'low_loss_overlap'
    """
    loss_vals = probe.get_loss_values()
    loss_steps = [s for s, _ in probe.losses]
    if len(loss_vals) < 5:
        return {'swarming_ratio': 1.0, 'high_loss_overlap': 0.0, 'low_loss_overlap': 0.0}

    # Determine threshold
    sorted_losses = sorted(loss_vals)
    threshold_idx = int(len(sorted_losses) * loss_threshold_percentile / 100)
    threshold = sorted_losses[min(threshold_idx, len(sorted_losses) - 1)]

    loss_by_step = dict(probe.losses)
    high_loss_steps = {s for s, l in probe.losses if l >= threshold}
    low_loss_steps = {s for s, l in probe.losses if l < threshold}

    # Collect head norms at each step
    head_keys = sorted(probe.head_outputs.keys())
    if len(head_keys) < 2:
        return {'swarming_ratio': 1.0, 'high_loss_overlap': 0.0, 'low_loss_overlap': 0.0}

    step_to_norms = {}
    for key in head_keys:
        for step, norm in probe.head_outputs[key]:
            step_to_norms.setdefault(step, {})[key] = norm

    def mean_pairwise_similarity(step_set):
        """Mean cosine similarity of head norm vectors across steps in set."""
        sims = []
        for step in step_set:
            if step not in step_to_norms:
                continue
            norms = step_to_norms[step]
            keys = sorted(norms.keys())
            if len(keys) < 2:
                continue
            # Pairwise similarity of head norms (normalized)
            vals = [norms[k] for k in keys]
            total = sum(v ** 2 for v in vals) ** 0.5
            if total < 1e-10:
                continue
            normed = [v / total for v in vals]
            # Measure concentration: how similar are the head contributions?
            mean_v = sum(normed) / len(normed)
            # Low variance = heads converging (similar contributions)
            variance = sum((v - mean_v) ** 2 for v in normed) / len(normed)
            # Invert: high overlap = low variance
            sims.append(1.0 / (1.0 + variance * len(normed)))
        return sum(sims) / len(sims) if sims else 0.0

    high_overlap = mean_pairwise_similarity(high_loss_steps)
    low_overlap = mean_pairwise_similarity(low_loss_steps)

    ratio = high_overlap / low_overlap if low_overlap > 1e-10 else (
        float('inf') if high_overlap > 0 else 1.0)

    return {
        'swarming_ratio': ratio,
        'high_loss_overlap': high_overlap,
        'low_loss_overlap': low_overlap,
    }


def detect_anomalies(probe):
    """
    Run all anomaly detectors and return a unified list.
    Each anomaly dict has a 'type' field identifying the detector.
    """
    anomalies = []
    anomalies.extend(detect_sync_events(probe))
    anomalies.extend(detect_ghost_spikes(probe))
    anomalies.extend(detect_sudden_specialization(probe))
    anomalies.extend(detect_role_reversals(probe))
    anomalies.extend(detect_gradient_divergence(probe))

    for key in sorted(probe.head_outputs.keys()):
        norms = [n for _, n in probe.head_outputs[key]]
        periods = detect_periodicity(norms)
        for p in periods:
            anomalies.append({
                'type': 'periodicity',
                'head': key,
                'period': p['period'],
                'strength': p['strength'],
            })

    anomalies.sort(key=lambda a: a.get('step', 0))
    return anomalies
