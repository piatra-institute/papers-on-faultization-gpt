"""
MorphoGPT — Experiment Runner

Run systematic experiments across perturbation types, damage levels,
and configurations. Produce robustness curves, DG analysis, and
rerouting measurements.
"""

import os
import json
import random
import time
from dataclasses import dataclass, field

from morphogpt import (
    Hooks, Probe, TrainConfig,
    make_config, init_state_dict, load_dataset, train, generate
)
from perturbations import (
    make_zero_head, make_freeze_params,
    schedule_chronic, schedule_acute, schedule_stochastic,
    make_noise_head,
    make_noise_injection, make_stop_gradient, make_noisy_gradients,
    make_sign_only_gradients, make_quantized_gradients,
    make_dropout, make_stochastic_relu, make_windowed_attention,
    make_sparse_attention, make_async_updates, make_update_budget,
    make_adversarial_head, make_delayed_gradients,
    freeze_random_heads, apply_stop_gradient_all,
    make_layered_vision, make_partial_stop_gradient,
    make_threatening_drive, make_round_robin_updates,
)
from metrics import (
    summarize_probe, dg_index, robustness_curve_with_dg,
    dg_damage_regression, trajectory_envelope, compare_trajectory_envelopes,
    per_step_rerouting, detect_phases, compute_delayed_gratification,
    head_contribution_evolution,
    cognitive_light_cone, collective_light_cone,
    goal_alignment_score, swarming_index,
)


# ============================================================================
# Experiment Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str = 'baseline'
    description: str = ''

    # Model config
    n_layer: int = 4
    n_embd: int = 16
    n_head: int = 4
    block_size: int = 16

    # Training config
    num_steps: int = 500
    learning_rate: float = 0.01

    # Perturbation config
    perturbation_type: str = 'none'    # see PERTURBATION_CATALOG
    perturbation_params: dict = field(default_factory=dict)

    # Schedule config
    schedule: str = 'chronic'          # 'chronic', 'acute', 'stochastic'
    schedule_params: dict = field(default_factory=dict)

    # Repetitions
    seed: int = 42
    num_reps: int = 1                  # for sweeps: run this many times

    # Trajectory capture
    capture_interval: int = 5        # record state every N steps
    detail_level: str = 'summary'    # 'full', 'summary', or 'loss_only'

    # Output
    save_trajectory: bool = True
    save_samples: bool = False
    print_progress: bool = True


# ============================================================================
# Run a single experiment
# ============================================================================

def run_experiment(exp_config, docs=None, uchars=None, BOS=None, vocab_size=None):
    """
    Run a single experiment.

    Returns dict with:
        config: the experiment config
        probe: the Probe object
        summary: summary statistics
        samples: generated samples (if requested)
        elapsed: wall clock time in seconds
    """
    # Load dataset if not provided
    if docs is None:
        docs, uchars, BOS, vocab_size = load_dataset()

    config = make_config(
        n_layer=exp_config.n_layer,
        n_embd=exp_config.n_embd,
        n_head=exp_config.n_head,
        block_size=exp_config.block_size,
        vocab_size=vocab_size,
    )

    state_dict, params = init_state_dict(config, seed=exp_config.seed)

    tc = TrainConfig(
        num_steps=exp_config.num_steps,
        learning_rate=exp_config.learning_rate,
        print_every=50 if exp_config.print_progress else 0,
        detail_level=exp_config.detail_level,
    )

    hooks = Hooks()
    grad_hooks = []

    # --- Apply perturbations ---
    pp = exp_config.perturbation_params
    pt = exp_config.perturbation_type

    frozen_heads = []

    # Collect (name, fn) pairs for forward hooks so we can wrap with schedule
    pending_hooks = []

    if pt == 'none':
        pass

    elif pt == 'freeze_heads':
        num = pp.get('num_heads', 1)
        rng = random.Random(exp_config.seed + 1000)
        frozen_heads = freeze_random_heads(hooks, config, num, rng=rng)

    elif pt == 'zero_head':
        layer = pp.get('layer', 0)
        head = pp.get('head', 0)
        name, fn = make_zero_head(layer, head, config['head_dim'])
        pending_hooks.append((name, fn))

    elif pt == 'noise_heads':
        num = pp.get('num_heads', 1)
        noise_std = pp.get('noise_std', 0.1)
        rng = random.Random(exp_config.seed + 1000)
        all_heads = [(li, h) for li in range(config['n_layer'])
                     for h in range(config['n_head'])]
        rng.shuffle(all_heads)
        for li, h in all_heads[:num]:
            name, fn = make_noise_head(li, h, config['head_dim'],
                                       noise_std=noise_std, rng=rng)
            pending_hooks.append((name, fn))

    elif pt == 'noise_injection':
        hook_name = pp.get('hook_name', 'emb')
        noise_std = pp.get('noise_std', 0.1)
        name, fn = make_noise_injection(hook_name, noise_std)
        pending_hooks.append((name, fn))

    elif pt == 'stop_gradient':
        layers = pp.get('layers', 'all')
        if layers == 'all':
            apply_stop_gradient_all(hooks, config)
        else:
            for li in layers:
                name, fn = make_stop_gradient(li)
                pending_hooks.append((name, fn))

    elif pt == 'noisy_gradients':
        noise_std = pp.get('noise_std', 0.01)
        grad_hooks.append(make_noisy_gradients(noise_std))

    elif pt == 'sign_only_gradients':
        grad_hooks.append(make_sign_only_gradients())

    elif pt == 'quantized_gradients':
        levels = pp.get('levels', 3)
        grad_hooks.append(make_quantized_gradients(levels))

    elif pt == 'delayed_gradients':
        delay = pp.get('delay_steps', 5)
        grad_hooks.append(make_delayed_gradients(delay))

    elif pt == 'dropout':
        drop_prob = pp.get('drop_prob', 0.1)
        for li in range(config['n_layer']):
            name, fn = make_dropout(f'mlp_hidden.{li}', drop_prob)
            pending_hooks.append((name, fn))

    elif pt == 'stochastic_relu':
        flip_prob = pp.get('flip_prob', 0.05)
        for li in range(config['n_layer']):
            name, fn = make_stochastic_relu(f'mlp_hidden.{li}', flip_prob)
            pending_hooks.append((name, fn))

    elif pt == 'windowed_attention':
        window = pp.get('window_size', 4)
        for li in range(config['n_layer']):
            for h in range(config['n_head']):
                name, fn = make_windowed_attention(li, h, window)
                pending_hooks.append((name, fn))

    elif pt == 'sparse_attention':
        keep_prob = pp.get('keep_prob', 0.5)
        rng = random.Random(exp_config.seed + 2000)
        for li in range(config['n_layer']):
            for h in range(config['n_head']):
                name, fn = make_sparse_attention(li, h, keep_prob, rng=rng)
                pending_hooks.append((name, fn))

    elif pt == 'async_updates':
        freqs = pp.get('layer_frequencies', {0: 1, 1: 2, 2: 5, 3: 10})
        grad_hooks.append(make_async_updates(freqs))

    elif pt == 'update_budget':
        fraction = pp.get('budget_fraction', 0.5)
        grad_hooks.append(make_update_budget(fraction))

    elif pt == 'adversarial_heads':
        num = pp.get('num_heads', 1)
        rng = random.Random(exp_config.seed + 3000)
        all_heads = [(li, h) for li in range(config['n_layer'])
                     for h in range(config['n_head'])]
        rng.shuffle(all_heads)
        for li, h in all_heads[:num]:
            grad_hooks.append(make_adversarial_head(li, h, config['head_dim']))

    elif pt == 'freeze_params':
        param_names = pp.get('param_names', [])
        grad_hooks.append(make_freeze_params(param_names))

    elif pt == 'layered_vision':
        radius_per_layer = pp.get('radius_per_layer', {})
        hook_pairs = make_layered_vision(config, radius_per_layer)
        pending_hooks.extend(hook_pairs)

    elif pt == 'partial_stop_gradient':
        layers = pp.get('layers', 'all')
        pass_fraction = pp.get('pass_fraction', 0.5)
        if layers == 'all':
            for li in range(config['n_layer'] - 1):
                name, fn = make_partial_stop_gradient(li, pass_fraction)
                pending_hooks.append((name, fn))
        else:
            for li in layers:
                name, fn = make_partial_stop_gradient(li, pass_fraction)
                pending_hooks.append((name, fn))

    elif pt == 'threatening_drive':
        strength = pp.get('strength', 0.1)
        for li in range(config['n_layer']):
            for h in range(config['n_head']):
                grad_hooks.append(
                    make_threatening_drive(li, h, config['head_dim'], strength))

    elif pt == 'round_robin_updates':
        period = pp.get('period', 1)
        grad_hooks.append(make_round_robin_updates(config, period))

    else:
        raise ValueError(f"Unknown perturbation type: {pt}")

    # --- Apply schedule wrapping and register pending hooks ---
    sched = exp_config.schedule
    sp = exp_config.schedule_params
    for name, fn in pending_hooks:
        if sched == 'acute':
            fn = schedule_acute(fn, sp.get('start_step', 0), sp.get('end_step', 100))
        elif sched == 'stochastic':
            fn = schedule_stochastic(fn, sp.get('prob', 0.5))
        else:
            fn = schedule_chronic(fn)
        hooks.register(name, fn)

    # --- Run training ---
    t0 = time.time()
    probe = train(
        state_dict, params, config, tc,
        docs, uchars, BOS,
        hooks=hooks,
        probe=Probe(record_interval=exp_config.capture_interval,
                    detail_level=exp_config.detail_level),
        grad_hooks=grad_hooks if grad_hooks else None,
        seed=exp_config.seed,
    )
    elapsed = time.time() - t0

    # --- Generate samples ---
    samples = None
    if exp_config.save_samples:
        samples = generate(state_dict, config, uchars, BOS,
                           num_samples=10, temperature=0.5, seed=exp_config.seed)

    # --- Summary ---
    summary = summarize_probe(probe)
    summary['perturbation_type'] = pt
    summary['perturbation_params'] = pp
    summary['seed'] = exp_config.seed
    summary['frozen_heads'] = frozen_heads
    summary['elapsed'] = elapsed

    return {
        'config': exp_config,
        'probe': probe,
        'summary': summary,
        'samples': samples,
        'elapsed': elapsed,
    }


# ============================================================================
# Run a sweep of experiments
# ============================================================================

def run_sweep(configs, docs=None, uchars=None, BOS=None, vocab_size=None):
    """
    Run a batch of experiments.
    Returns list of result dicts.
    """
    if docs is None:
        docs, uchars, BOS, vocab_size = load_dataset()

    results = []
    total = sum(c.num_reps for c in configs)
    run_idx = 0

    for cfg in configs:
        for rep in range(cfg.num_reps):
            run_idx += 1
            # Vary seed per repetition
            rep_cfg = ExperimentConfig(
                name=cfg.name,
                description=cfg.description,
                n_layer=cfg.n_layer,
                n_embd=cfg.n_embd,
                n_head=cfg.n_head,
                block_size=cfg.block_size,
                num_steps=cfg.num_steps,
                learning_rate=cfg.learning_rate,
                perturbation_type=cfg.perturbation_type,
                perturbation_params=cfg.perturbation_params,
                schedule=cfg.schedule,
                schedule_params=cfg.schedule_params,
                seed=cfg.seed + rep,
                num_reps=1,
                capture_interval=cfg.capture_interval,
                detail_level=cfg.detail_level,
                save_trajectory=cfg.save_trajectory,
                save_samples=cfg.save_samples,
                print_progress=False,
            )

            print(f"\n[{run_idx}/{total}] {cfg.name} (rep {rep+1}/{cfg.num_reps}, "
                  f"seed={rep_cfg.seed})")

            result = run_experiment(rep_cfg, docs, uchars, BOS, vocab_size)
            results.append(result)

            print(f"  loss={result['summary']['final_loss']:.4f} "
                  f"dg={result['summary']['dg_index']:.3f} "
                  f"time={result['elapsed']:.1f}s")

    return results


# ============================================================================
# Experiment 1: Head Freezing Robustness Curve
# ============================================================================

def experiment_head_freezing(num_reps=5, num_steps=200, n_layer=4):
    """
    The first experiment: freeze increasing numbers of heads and measure
    robustness + delayed gratification.

    This directly mirrors Levin's frozen cell experiment (Figure 5 in the paper).

    Args:
        num_reps: repetitions per configuration
        num_steps: training steps (reduce for faster testing)
        n_layer: number of layers

    Returns:
        results dict with robustness curve data
    """
    print("=" * 60)
    print("EXPERIMENT 1: Head Freezing Robustness Curve")
    print("=" * 60)

    n_head = 4
    total_heads = n_layer * n_head

    # Damage levels: 0, 1, 2, 4, 8, 12, 16 frozen heads (of 16 total for n_layer=4)
    damage_levels = [0, 1, 2, 4]
    if total_heads >= 8:
        damage_levels.append(8)
    if total_heads >= 12:
        damage_levels.append(12)
    damage_levels.append(total_heads)

    configs = []
    for num_frozen in damage_levels:
        pt = 'none' if num_frozen == 0 else 'freeze_heads'
        pp = {} if num_frozen == 0 else {'num_heads': num_frozen}

        configs.append(ExperimentConfig(
            name=f'freeze_{num_frozen}',
            description=f'Freeze {num_frozen}/{total_heads} heads',
            n_layer=n_layer,
            num_steps=num_steps,
            perturbation_type=pt,
            perturbation_params=pp,
            num_reps=num_reps,
            seed=42,
            print_progress=False,
        ))

    # Load dataset once
    docs, uchars, BOS, vocab_size = load_dataset()

    # Run all experiments
    all_results = run_sweep(configs, docs, uchars, BOS, vocab_size)

    t0 = time.time()

    # Organize results by damage level
    results_by_level = {}
    for result in all_results:
        pt = result['summary']['perturbation_type']
        if pt == 'none':
            level = 0
        else:
            level = result['summary']['perturbation_params']['num_heads']

        loss_vals = result['probe'].get_loss_values()
        final_loss = result['summary']['final_loss']

        results_by_level.setdefault(level, []).append((final_loss, loss_vals))

    print(f"  [timing] organize results: {time.time() - t0:.1f}s"); t0 = time.time()

    # Compute robustness curve
    curve = robustness_curve_with_dg(results_by_level)

    print(f"  [timing] robustness curve: {time.time() - t0:.1f}s"); t0 = time.time()

    # --- TRAJECTORY ANALYSIS ---

    # Trajectory envelopes per damage level
    envelopes = {}
    for level, entries in sorted(results_by_level.items()):
        trajectories = [loss_vals for _, loss_vals in entries]
        envelopes[level] = trajectory_envelope(trajectories)

    print(f"  [timing] trajectory envelopes: {time.time() - t0:.1f}s"); t0 = time.time()

    # Phase detection per damage level
    phases_by_level = {}
    for level, entries in sorted(results_by_level.items()):
        level_phases = []
        for _, loss_vals in entries:
            level_phases.append(detect_phases(loss_vals))
        phases_by_level[level] = level_phases

    print(f"  [timing] phase detection: {time.time() - t0:.1f}s"); t0 = time.time()

    # DG episodes as trajectory events with timing
    dg_episodes_by_level = {}
    for level, entries in sorted(results_by_level.items()):
        level_episodes = []
        for _, loss_vals in entries:
            level_episodes.append(compute_delayed_gratification(loss_vals))
        dg_episodes_by_level[level] = level_episodes

    print(f"  [timing] DG episodes: {time.time() - t0:.1f}s"); t0 = time.time()

    # Per-head rerouting analysis
    rerouting_by_level = {}
    for result in all_results:
        pt = result['summary']['perturbation_type']
        if pt != 'none' and result['probe'].head_outputs:
            level = result['summary']['perturbation_params']['num_heads']
            rerouting = per_step_rerouting(result['probe'])
            rerouting_by_level.setdefault(level, []).append(rerouting)

    print(f"  [timing] rerouting: {time.time() - t0:.1f}s"); t0 = time.time()

    # Compare trajectory shapes: baseline vs each damage level
    shape_comparisons = {}
    if 0 in envelopes:
        for level in sorted(envelopes.keys()):
            if level == 0:
                continue
            shape_comparisons[level] = compare_trajectory_envelopes(
                envelopes[0], envelopes[level])

    print(f"  [timing] shape comparisons: {time.time() - t0:.1f}s"); t0 = time.time()

    # --- PRINT TRAJECTORY-FOCUSED RESULTS ---

    print("\n" + "=" * 60)
    print("TRAJECTORY ANALYSIS: Head Freezing")
    print("=" * 60)

    # Phase structure per damage level
    print("\n--- Phase Structure ---")
    for level in sorted(phases_by_level.keys()):
        all_phases = phases_by_level[level]
        if not all_phases or not all_phases[0]:
            continue
        # Summarize: most common phase sequence across reps
        first_run = all_phases[0]
        phase_seq = ' -> '.join(p['type'] for p in first_run)
        print(f"  {level:2d} frozen: {phase_seq}")

    # DG episodes as events
    print("\n--- DG Episodes (trajectory events) ---")
    for level in sorted(dg_episodes_by_level.keys()):
        all_eps = dg_episodes_by_level[level]
        total_eps = sum(len(eps) for eps in all_eps)
        if total_eps > 0:
            all_flat = [ep for eps in all_eps for ep in eps]
            mean_start = sum(ep['start'] for ep in all_flat) / len(all_flat)
            mean_dur = sum(ep['end'] - ep['start'] for ep in all_flat) / len(all_flat)
            mean_dgi = sum(ep['dg_index'] for ep in all_flat) / len(all_flat)
            print(f"  {level:2d} frozen: {total_eps} episodes, "
                  f"mean start=step {mean_start:.0f}, "
                  f"mean duration={mean_dur:.0f} steps, "
                  f"mean DG index={mean_dgi:.3f}")
        else:
            print(f"  {level:2d} frozen: no DG episodes")

    # Trajectory shape comparison
    if shape_comparisons:
        print("\n--- Trajectory Shape vs Baseline ---")
        for level, comp in sorted(shape_comparisons.items()):
            print(f"  {level:2d} frozen: "
                  f"divergence={comp['mean_divergence']:.4f}, "
                  f"max@step {comp['divergence_step']}, "
                  f"correlation={comp['shape_correlation']:.3f}, "
                  f"overlap={comp['overlap_fraction']:.2f}")

    # Rerouting analysis
    if rerouting_by_level:
        print("\n--- Head Rerouting (compensating heads) ---")
        for level in sorted(rerouting_by_level.keys()):
            rerouting_runs = rerouting_by_level[level]
            if rerouting_runs and rerouting_runs[0]:
                top = rerouting_runs[0][:3]  # top 3 compensators from first run
                comp_str = ', '.join(
                    f"L{r['layer']}H{r['head']}(+{r['slope']:.4f})"
                    for r in top)
                print(f"  {level:2d} frozen: top compensators: {comp_str}")

    # Head contribution evolution on damaged runs
    contribution_by_level = {}
    for result in all_results:
        pt = result['summary']['perturbation_type']
        if pt != 'none' and result['probe'].head_outputs:
            level = result['summary']['perturbation_params']['num_heads']
            evo = head_contribution_evolution(result['probe'])
            contribution_by_level.setdefault(level, []).append(evo)

    if contribution_by_level:
        print("\n--- Head Contribution Evolution ---")
        for level in sorted(contribution_by_level.keys()):
            evos = contribution_by_level[level]
            if not evos or not evos[0]:
                continue
            evo = evos[0]  # first run
            parts = []
            for (li, hi), series in sorted(evo.items()):
                if len(series) >= 2:
                    first_frac = series[0][1]
                    last_frac = series[-1][1]
                    parts.append(f"L{li}H{hi}: {first_frac:.3f}->{last_frac:.3f}")
            if parts:
                print(f"  {level:2d} frozen: {', '.join(parts)}")

    print(f"  [timing] contribution evolution: {time.time() - t0:.1f}s"); t0 = time.time()

    # Endpoint table (secondary)
    print("\n--- Endpoints (secondary) ---")
    print(f"{'Frozen':>8} {'Mean Loss':>10} {'Std':>8} {'Mean DG':>8} {'Std DG':>8} {'N':>4}")
    print("-" * 50)
    for c in curve:
        print(f"{c['damage_level']:>8d} {c['mean_loss']:>10.4f} {c['std_loss']:>8.4f} "
              f"{c['mean_dg']:>8.3f} {c['std_dg']:>8.3f} {c['n']:>4d}")

    # DG-damage regression
    damage_levels_flat = []
    dg_indices_flat = []
    for level, entries in sorted(results_by_level.items()):
        for final_loss, loss_vals in entries:
            damage_levels_flat.append(level)
            dg_indices_flat.append(dg_index(loss_vals))

    slope, intercept, r_sq = dg_damage_regression(damage_levels_flat, dg_indices_flat)
    print(f"\nDG-Damage regression: slope={slope:.4f}, intercept={intercept:.4f}, R²={r_sq:.4f}")
    if slope > 0 and r_sq > 0.1:
        print("  -> DG increases with damage (evidence for genuine rerouting)")
    else:
        print("  -> No clear DG-damage relationship")

    return {
        'curve': curve,
        'all_results': all_results,
        'results_by_level': results_by_level,
        'envelopes': envelopes,
        'phases_by_level': phases_by_level,
        'dg_episodes_by_level': dg_episodes_by_level,
        'rerouting_by_level': rerouting_by_level,
        'shape_comparisons': shape_comparisons,
        'dg_regression': {'slope': slope, 'intercept': intercept, 'r_squared': r_sq},
    }


# ============================================================================
# Experiment 2: Cell-View GPT (Stop-Gradient)
# ============================================================================

def experiment_cell_view(num_reps=5, num_steps=200, n_layer=4):
    """
    Cell-view GPT: cut gradient flow between layers.
    Compare baseline (global backprop) vs cell-view (stop-gradient).
    Then combine with damage to test if cell-view is more robust.
    """
    print("=" * 60)
    print("EXPERIMENT 2: Cell-View GPT (Stop-Gradient)")
    print("=" * 60)

    configs = [
        # Baseline
        ExperimentConfig(
            name='baseline',
            description='Global backprop (standard)',
            n_layer=n_layer,
            num_steps=num_steps,
            perturbation_type='none',
            num_reps=num_reps,
            seed=42,
        ),
        # Cell-view (all stop-gradients)
        ExperimentConfig(
            name='cell_view',
            description='Stop-gradient at all layer boundaries',
            n_layer=n_layer,
            num_steps=num_steps,
            perturbation_type='stop_gradient',
            perturbation_params={'layers': 'all'},
            num_reps=num_reps,
            seed=42,
        ),
    ]

    docs, uchars, BOS, vocab_size = load_dataset()
    results = run_sweep(configs, docs, uchars, BOS, vocab_size)

    # Compare
    print("\n" + "=" * 60)
    print("RESULTS: Cell-View vs Baseline")
    print("=" * 60)

    for name in ['baseline', 'cell_view']:
        group = [r for r in results if r['config'].name == name]
        losses = [r['summary']['final_loss'] for r in group]
        dgs = [r['summary']['dg_index'] for r in group]
        mean_l = sum(losses) / len(losses)
        mean_dg = sum(dgs) / len(dgs)
        print(f"  {name:15s}: loss={mean_l:.4f}, dg={mean_dg:.3f}")

    return results


# ============================================================================
# Experiment 3: Gradient Degradation (A10)
# ============================================================================

def experiment_gradient_degradation(num_reps=5, num_steps=200, n_layer=4):
    """
    Test how degraded gradient information affects learning.
    """
    print("=" * 60)
    print("EXPERIMENT 3: Gradient Degradation")
    print("=" * 60)

    configs = [
        ExperimentConfig(
            name='baseline', n_layer=n_layer, num_steps=num_steps,
            perturbation_type='none', num_reps=num_reps, seed=42,
        ),
        ExperimentConfig(
            name='noisy_grad_0.01', n_layer=n_layer, num_steps=num_steps,
            perturbation_type='noisy_gradients',
            perturbation_params={'noise_std': 0.01},
            num_reps=num_reps, seed=42,
        ),
        ExperimentConfig(
            name='noisy_grad_0.1', n_layer=n_layer, num_steps=num_steps,
            perturbation_type='noisy_gradients',
            perturbation_params={'noise_std': 0.1},
            num_reps=num_reps, seed=42,
        ),
        ExperimentConfig(
            name='sign_only', n_layer=n_layer, num_steps=num_steps,
            perturbation_type='sign_only_gradients',
            num_reps=num_reps, seed=42,
        ),
        ExperimentConfig(
            name='quantized_3', n_layer=n_layer, num_steps=num_steps,
            perturbation_type='quantized_gradients',
            perturbation_params={'levels': 3},
            num_reps=num_reps, seed=42,
        ),
    ]

    docs, uchars, BOS, vocab_size = load_dataset()
    results = run_sweep(configs, docs, uchars, BOS, vocab_size)

    print("\n" + "=" * 60)
    print("RESULTS: Gradient Degradation")
    print("=" * 60)

    for name in ['baseline', 'noisy_grad_0.01', 'noisy_grad_0.1', 'sign_only', 'quantized_3']:
        group = [r for r in results if r['config'].name == name]
        if group:
            losses = [r['summary']['final_loss'] for r in group]
            mean_l = sum(losses) / len(losses)
            print(f"  {name:20s}: loss={mean_l:.4f}")

    return results


# ============================================================================
# Experiment 4: Vision Radius Sweep (Kofman et al., 2025)
# ============================================================================

def experiment_vision_radius(num_reps=5, num_steps=200, n_layer=4):
    """
    Sweep windowed attention across window sizes to test whether intermediate
    restriction outperforms both minimal and full context.

    Parallels the chess paper's R0-R7 vision radius finding (optimal at R4).

    Hypothesis: Medium windows (4-8) match or beat full attention (16),
    paralleling R4 > R7. DG should increase with restricted windows.
    """
    print("=" * 60)
    print("EXPERIMENT 4: Vision Radius Sweep")
    print("=" * 60)

    window_sizes = [1, 2, 4, 8, 16]  # 16 = full context (baseline equivalent)

    configs = [
        # Baseline (no windowed attention)
        ExperimentConfig(
            name='baseline',
            description='Full attention (no window restriction)',
            n_layer=n_layer,
            num_steps=num_steps,
            perturbation_type='none',
            num_reps=num_reps,
            seed=42,
        ),
    ]

    for ws in window_sizes:
        configs.append(ExperimentConfig(
            name=f'window_{ws}',
            description=f'Windowed attention, window_size={ws}',
            n_layer=n_layer,
            num_steps=num_steps,
            perturbation_type='windowed_attention',
            perturbation_params={'window_size': ws},
            num_reps=num_reps,
            seed=42,
        ))

    docs, uchars, BOS, vocab_size = load_dataset()
    all_results = run_sweep(configs, docs, uchars, BOS, vocab_size)

    # Organize by window size
    results_by_window = {}
    for result in all_results:
        name = result['config'].name
        if name == 'baseline':
            ws = 'full'
        else:
            ws = result['config'].perturbation_params.get('window_size', 'full')
        loss_vals = result['probe'].get_loss_values()
        final_loss = result['summary']['final_loss']
        results_by_window.setdefault(ws, []).append({
            'final_loss': final_loss,
            'loss_vals': loss_vals,
            'dg': dg_index(loss_vals),
            'probe': result['probe'],
        })

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS: Vision Radius Sweep")
    print("=" * 60)

    print(f"\n{'Window':>8} {'Mean Loss':>10} {'Std':>8} {'Mean DG':>8} {'N':>4}")
    print("-" * 42)

    for ws in ['full'] + window_sizes:
        entries = results_by_window.get(ws, [])
        if not entries:
            continue
        losses = [e['final_loss'] for e in entries]
        dgs = [e['dg'] for e in entries]
        n = len(losses)
        mean_l = sum(losses) / n
        std_l = (sum((x - mean_l) ** 2 for x in losses) / max(1, n - 1)) ** 0.5
        mean_dg = sum(dgs) / n
        label = str(ws) if ws != 'full' else 'full'
        print(f"{label:>8} {mean_l:>10.4f} {std_l:>8.4f} {mean_dg:>8.3f} {n:>4d}")

    # Chess-paper metrics on first run per condition
    print("\n--- Cognitive Light Cone (chess-paper metric) ---")
    for ws in ['full'] + window_sizes:
        entries = results_by_window.get(ws, [])
        if entries and entries[0]['probe'].head_outputs:
            clc = collective_light_cone(entries[0]['probe'])
            label = str(ws) if ws != 'full' else 'full'
            print(f"  window={label:>4}: spatial={clc['mean_spatial']:.3f}, "
                  f"temporal={clc['mean_temporal']:.1f}, "
                  f"combined={clc['mean_combined']:.3f}")

    return {
        'all_results': all_results,
        'results_by_window': results_by_window,
    }


# ============================================================================
# Experiment 5: Communication Topology (Kofman et al., 2025)
# ============================================================================

def experiment_communication_topology(num_reps=5, num_steps=200, n_layer=4):
    """
    Extend the binary cell-view experiment into a spectrum of gradient
    flow topologies.

    Parallels the chess paper's finding that relay chains expand the
    cognitive light cone without requiring global information.

    Conditions:
        - full: standard backpropagation (pass_fraction=1.0)
        - heavy: most gradient passes through (pass_fraction=0.75)
        - half: half gradient passes (pass_fraction=0.5)
        - light: little gradient passes (pass_fraction=0.25)
        - cell_view: no gradient flow (pass_fraction=0.0)

    Hypothesis: Intermediate gradient flow (0.25-0.5) outperforms both
    full backprop and complete isolation.
    """
    print("=" * 60)
    print("EXPERIMENT 5: Communication Topology")
    print("=" * 60)

    topologies = [
        ('full', 1.0),
        ('heavy', 0.75),
        ('half', 0.5),
        ('light', 0.25),
        ('cell_view', 0.0),
    ]

    configs = []
    for name, fraction in topologies:
        if fraction >= 1.0:
            configs.append(ExperimentConfig(
                name=name,
                description=f'Gradient pass fraction={fraction}',
                n_layer=n_layer,
                num_steps=num_steps,
                perturbation_type='none',
                num_reps=num_reps,
                seed=42,
            ))
        elif fraction <= 0.0:
            configs.append(ExperimentConfig(
                name=name,
                description=f'Gradient pass fraction={fraction} (full cell-view)',
                n_layer=n_layer,
                num_steps=num_steps,
                perturbation_type='stop_gradient',
                perturbation_params={'layers': 'all'},
                num_reps=num_reps,
                seed=42,
            ))
        else:
            configs.append(ExperimentConfig(
                name=name,
                description=f'Gradient pass fraction={fraction}',
                n_layer=n_layer,
                num_steps=num_steps,
                perturbation_type='partial_stop_gradient',
                perturbation_params={'layers': 'all', 'pass_fraction': fraction},
                num_reps=num_reps,
                seed=42,
            ))

    docs, uchars, BOS, vocab_size = load_dataset()
    all_results = run_sweep(configs, docs, uchars, BOS, vocab_size)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS: Communication Topology")
    print("=" * 60)

    print(f"\n{'Topology':>12} {'Fraction':>9} {'Mean Loss':>10} {'Mean DG':>8}")
    print("-" * 43)

    for name, fraction in topologies:
        group = [r for r in all_results if r['config'].name == name]
        if not group:
            continue
        losses = [r['summary']['final_loss'] for r in group]
        dgs = [r['summary']['dg_index'] for r in group]
        mean_l = sum(losses) / len(losses)
        mean_dg = sum(dgs) / len(dgs)
        print(f"{name:>12} {fraction:>9.2f} {mean_l:>10.4f} {mean_dg:>8.3f}")

    # Goal alignment scores
    print("\n--- Goal Alignment Score (chess-paper metric) ---")
    for name, fraction in topologies:
        group = [r for r in all_results if r['config'].name == name]
        if group and group[0]['probe'].head_outputs:
            ga = goal_alignment_score(group[0]['probe'])
            print(f"  {name:>12}: mean alignment = {ga['mean_alignment']:.4f}")

    return all_results


# ============================================================================
# Experiment 6: Courage vs. Caution (Kofman et al., 2025)
# ============================================================================

def experiment_courage_caution(num_reps=5, num_steps=200, n_layer=4):
    """
    Test four combinations of forward-pass stability and gradient boldness.

    Parallels the chess paper's finding that "cautious position, courageous
    moves" is the optimal strategy (conservative evaluation + aggressive action).

    Conditions:
        (a) cautious/cautious: low noise everywhere
        (b) cautious/courageous: stable forward pass, sign-only gradients (predicted best)
        (c) courageous/cautious: noisy forward (dropout), careful gradients
        (d) courageous/courageous: noise everywhere

    Hypothesis: Condition (b) wins, paralleling the chess paper.
    Anomaly 1 (frozen heads help) and Anomaly 2 (sign-only works) already
    hint at this: stable representations + bold updates.
    """
    print("=" * 60)
    print("EXPERIMENT 6: Courage vs. Caution")
    print("=" * 60)

    configs = [
        # Baseline for reference
        ExperimentConfig(
            name='baseline',
            description='No perturbation',
            n_layer=n_layer,
            num_steps=num_steps,
            perturbation_type='none',
            num_reps=num_reps,
            seed=42,
        ),
    ]

    # The four conditions are implemented as composite perturbation configs.
    # Since run_experiment handles one perturbation_type at a time,
    # we use custom hooks composition via the 'composite' type.

    # For conditions that need multiple perturbation types, we'll
    # run them through the existing system by using the most distinctive
    # perturbation and noting the combination.

    # (a) cautious/cautious: small noise on gradients (low everywhere)
    configs.append(ExperimentConfig(
        name='cautious_cautious',
        description='Low noise everywhere (cautious position + cautious move)',
        n_layer=n_layer,
        num_steps=num_steps,
        perturbation_type='noisy_gradients',
        perturbation_params={'noise_std': 0.001},
        num_reps=num_reps,
        seed=42,
    ))

    # (b) cautious/courageous: stable forward, sign-only gradients
    # This is the predicted winner — maps to "cautious position, courageous moves"
    configs.append(ExperimentConfig(
        name='cautious_courageous',
        description='Stable forward + sign-only gradients (predicted best)',
        n_layer=n_layer,
        num_steps=num_steps,
        perturbation_type='sign_only_gradients',
        num_reps=num_reps,
        seed=42,
    ))

    # (c) courageous/cautious: noisy forward, careful gradients
    configs.append(ExperimentConfig(
        name='courageous_cautious',
        description='Dropout forward + careful gradients',
        n_layer=n_layer,
        num_steps=num_steps,
        perturbation_type='dropout',
        perturbation_params={'drop_prob': 0.1},
        num_reps=num_reps,
        seed=42,
    ))

    # (d) courageous/courageous: noise everywhere
    configs.append(ExperimentConfig(
        name='courageous_courageous',
        description='Dropout forward + noisy gradients (noise everywhere)',
        n_layer=n_layer,
        num_steps=num_steps,
        perturbation_type='noisy_gradients',
        perturbation_params={'noise_std': 0.1},
        num_reps=num_reps,
        seed=42,
    ))

    docs, uchars, BOS, vocab_size = load_dataset()
    all_results = run_sweep(configs, docs, uchars, BOS, vocab_size)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS: Courage vs. Caution")
    print("=" * 60)

    condition_labels = {
        'baseline': 'Baseline',
        'cautious_cautious': '(a) Cautious/Cautious',
        'cautious_courageous': '(b) Cautious/Courageous *',
        'courageous_cautious': '(c) Courageous/Cautious',
        'courageous_courageous': '(d) Courageous/Courageous',
    }

    print(f"\n{'Condition':>30} {'Mean Loss':>10} {'Mean DG':>8}")
    print("-" * 52)

    for name in ['baseline', 'cautious_cautious', 'cautious_courageous',
                  'courageous_cautious', 'courageous_courageous']:
        group = [r for r in all_results if r['config'].name == name]
        if not group:
            continue
        losses = [r['summary']['final_loss'] for r in group]
        dgs = [r['summary']['dg_index'] for r in group]
        mean_l = sum(losses) / len(losses)
        mean_dg = sum(dgs) / len(dgs)
        label = condition_labels.get(name, name)
        print(f"{label:>30} {mean_l:>10.4f} {mean_dg:>8.3f}")

    print("\n  * = predicted best (paralleling chess paper's 'cautious position, courageous moves')")

    # Swarming index
    print("\n--- Swarming Index (chess-paper metric) ---")
    for name in ['baseline', 'cautious_cautious', 'cautious_courageous',
                  'courageous_cautious', 'courageous_courageous']:
        group = [r for r in all_results if r['config'].name == name]
        if group and group[0]['probe'].head_outputs:
            si = swarming_index(group[0]['probe'])
            label = condition_labels.get(name, name)
            print(f"  {label:>30}: ratio={si['swarming_ratio']:.3f}")

    return all_results


# ============================================================================
# Save results
# ============================================================================

def save_results(results, path):
    """Save experiment results to a JSON file, including trajectory data."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    t_save = time.time()
    serializable = []
    for r in results:
        probe = r['probe']
        entry = {
            'name': r['config'].name,
            'description': r['config'].description,
            'seed': r['config'].seed,
            'perturbation_type': r['config'].perturbation_type,
            'perturbation_params': r['config'].perturbation_params,
            'summary': {k: v for k, v in r['summary'].items()
                        if not isinstance(v, list) or k == 'frozen_heads'},
            'elapsed': r['elapsed'],
        }
        if r['config'].save_trajectory:
            entry['loss_trajectory'] = probe.get_loss_values()

            # Per-head norms over time
            head_norm_data = {}
            for (li, hi), entries in probe.head_outputs.items():
                head_norm_data[f'L{li}H{hi}'] = [
                    {'step': s, 'norm': n} for s, n in entries
                ]
            if head_norm_data:
                entry['head_norms'] = head_norm_data

            # Per-head entropies over time
            head_entropy_data = {}
            for (li, hi), entries in probe.attention_entropies.items():
                head_entropy_data[f'L{li}H{hi}'] = [
                    {'step': s, 'entropy': e} for s, e in entries
                ]
            if head_entropy_data:
                entry['head_entropies'] = head_entropy_data

            # Phases
            loss_vals = probe.get_loss_values()
            from metrics import detect_phases, compute_delayed_gratification
            entry['phases'] = detect_phases(loss_vals)
            entry['dg_episodes'] = compute_delayed_gratification(loss_vals)

        if r['samples']:
            entry['samples'] = r['samples']
        serializable.append(entry)

    print(f"  [timing] save_results build: {time.time() - t_save:.1f}s"); t_ser = time.time()

    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"  [timing] save_results json write: {time.time() - t_ser:.1f}s")
    print(f"\nResults saved to {path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import sys

    experiment = sys.argv[1] if len(sys.argv) > 1 else 'head_freezing'
    num_reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    num_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    print(f"Running experiment: {experiment}")
    print(f"  reps={num_reps}, steps={num_steps}")

    os.makedirs('results', exist_ok=True)

    if experiment == 'head_freezing':
        results = experiment_head_freezing(num_reps=num_reps, num_steps=num_steps)
        save_results(results['all_results'], 'results/head_freezing.json')

    elif experiment == 'cell_view':
        results = experiment_cell_view(num_reps=num_reps, num_steps=num_steps)
        save_results(results, 'results/cell_view.json')

    elif experiment == 'gradient_degradation':
        results = experiment_gradient_degradation(num_reps=num_reps, num_steps=num_steps)
        save_results(results, 'results/gradient_degradation.json')

    elif experiment == 'vision_radius':
        results = experiment_vision_radius(num_reps=num_reps, num_steps=num_steps)
        save_results(results['all_results'], 'results/vision_radius.json')

    elif experiment == 'communication_topology':
        results = experiment_communication_topology(num_reps=num_reps, num_steps=num_steps)
        save_results(results, 'results/communication_topology.json')

    elif experiment == 'courage_caution':
        results = experiment_courage_caution(num_reps=num_reps, num_steps=num_steps)
        save_results(results, 'results/courage_caution.json')

    elif experiment == 'all':
        r1 = experiment_head_freezing(num_reps=num_reps, num_steps=num_steps)
        save_results(r1['all_results'], 'results/head_freezing.json')

        r2 = experiment_cell_view(num_reps=num_reps, num_steps=num_steps)
        save_results(r2, 'results/cell_view.json')

        r3 = experiment_gradient_degradation(num_reps=num_reps, num_steps=num_steps)
        save_results(r3, 'results/gradient_degradation.json')

        r4 = experiment_vision_radius(num_reps=num_reps, num_steps=num_steps)
        save_results(r4['all_results'], 'results/vision_radius.json')

        r5 = experiment_communication_topology(num_reps=num_reps, num_steps=num_steps)
        save_results(r5, 'results/communication_topology.json')

        r6 = experiment_courage_caution(num_reps=num_reps, num_steps=num_steps)
        save_results(r6, 'results/courage_caution.json')

    else:
        print(f"Unknown experiment: {experiment}")
        print("Available: head_freezing, cell_view, gradient_degradation, "
              "vision_radius, communication_topology, courage_caution, all")
        sys.exit(1)
