"""
MorphoGPT — Experiment Runner (NumPy Backend)

Drop-in replacement for experiments.py using numpy arrays.
Same interface: ExperimentConfig, run_experiment, run_sweep, experiment_*.
"""

import os
import json
import random
import time
from dataclasses import dataclass, field

from morphogpt_np import (
    Hooks, Probe, TrainConfig,
    make_config, init_state_dict, load_dataset, train, generate
)
from perturbations_np import (
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
    make_stop_gradient_grad_hook,
    make_partial_stop_gradient_grad_hook,
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
    perturbation_type: str = 'none'
    perturbation_params: dict = field(default_factory=dict)

    # Schedule config
    schedule: str = 'chronic'
    schedule_params: dict = field(default_factory=dict)

    # Repetitions
    seed: int = 42
    num_reps: int = 1

    # Trajectory capture
    capture_interval: int = 5
    detail_level: str = 'summary'

    # Output
    save_trajectory: bool = True
    save_samples: bool = False
    print_progress: bool = True


# ============================================================================
# Run a single experiment
# ============================================================================

def run_experiment(exp_config, docs=None, uchars=None, BOS=None, vocab_size=None):
    """Run a single experiment. Returns result dict."""
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
            apply_stop_gradient_all(hooks, config, grad_hooks)
        else:
            for li in layers:
                name, fn = make_stop_gradient(li)
                pending_hooks.append((name, fn))
                grad_hooks.append(make_stop_gradient_grad_hook(li, config))

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
                grad_hooks.append(
                    make_partial_stop_gradient_grad_hook(li, pass_fraction, config))
        else:
            for li in layers:
                name, fn = make_partial_stop_gradient(li, pass_fraction)
                pending_hooks.append((name, fn))
                grad_hooks.append(
                    make_partial_stop_gradient_grad_hook(li, pass_fraction, config))

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
    if docs is None:
        docs, uchars, BOS, vocab_size = load_dataset()

    results = []
    total = sum(c.num_reps for c in configs)
    run_idx = 0

    for cfg in configs:
        for rep in range(cfg.num_reps):
            run_idx += 1
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
# Save/load results (same format as original)
# ============================================================================

def save_results(results, filepath):
    """Save experiment results to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = []
    for r in results:
        entry = {
            'name': r['config'].name,
            'description': r['config'].description,
            'seed': r['config'].seed,
            'perturbation_type': r['config'].perturbation_type,
            'perturbation_params': r['config'].perturbation_params,
            'num_steps': r['config'].num_steps,
            'summary': r['summary'],
            'loss_trajectory': r['probe'].get_loss_values(),
        }
        # Head norms and entropies
        if r['probe'].head_outputs:
            entry['head_norms'] = {
                f"{k[0]},{k[1]}": [(s, n) for s, n in v]
                for k, v in r['probe'].head_outputs.items()
            }
        if r['probe'].attention_entropies:
            entry['head_entropies'] = {
                f"{k[0]},{k[1]}": [(s, e) for s, e in v]
                for k, v in r['probe'].attention_entropies.items()
            }
        # Phases and DG
        loss_vals = r['probe'].get_loss_values()
        entry['phases'] = detect_phases(loss_vals)
        entry['dg_episodes'] = compute_delayed_gratification(loss_vals)
        if r['samples']:
            entry['samples'] = r['samples']
        data.append(entry)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved {len(data)} results to {filepath}")


# ============================================================================
# Experiment 1: Head Freezing Robustness Curve
# ============================================================================

def experiment_head_freezing(num_reps=5, num_steps=200, n_layer=4):
    print("=" * 60)
    print("EXPERIMENT 1: Head Freezing Robustness Curve")
    print("=" * 60)

    n_head = 4
    total_heads = n_layer * n_head
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
            n_layer=n_layer, num_steps=num_steps,
            perturbation_type=pt, perturbation_params=pp,
            num_reps=num_reps, seed=42, print_progress=False,
        ))

    docs, uchars, BOS, vocab_size = load_dataset()
    all_results = run_sweep(configs, docs, uchars, BOS, vocab_size)

    # Save results
    save_results(all_results, os.path.join(os.path.dirname(__file__),
                 'results', 'experiment1_head_freezing.json'))

    # Organize and print
    results_by_level = {}
    for result in all_results:
        pt = result['summary']['perturbation_type']
        level = 0 if pt == 'none' else result['summary']['perturbation_params']['num_heads']
        loss_vals = result['probe'].get_loss_values()
        final_loss = result['summary']['final_loss']
        results_by_level.setdefault(level, []).append((final_loss, loss_vals))

    curve = robustness_curve_with_dg(results_by_level)

    # Trajectory analysis
    envelopes = {}
    for level, entries in sorted(results_by_level.items()):
        trajectories = [loss_vals for _, loss_vals in entries]
        envelopes[level] = trajectory_envelope(trajectories)

    phases_by_level = {}
    for level, entries in sorted(results_by_level.items()):
        level_phases = []
        for _, loss_vals in entries:
            level_phases.append(detect_phases(loss_vals))
        phases_by_level[level] = level_phases

    dg_episodes_by_level = {}
    for level, entries in sorted(results_by_level.items()):
        level_episodes = []
        for _, loss_vals in entries:
            level_episodes.append(compute_delayed_gratification(loss_vals))
        dg_episodes_by_level[level] = level_episodes

    rerouting_by_level = {}
    for result in all_results:
        pt = result['summary']['perturbation_type']
        if pt != 'none' and result['probe'].head_outputs:
            level = result['summary']['perturbation_params']['num_heads']
            rerouting = per_step_rerouting(result['probe'])
            rerouting_by_level.setdefault(level, []).append(rerouting)

    shape_comparisons = {}
    if 0 in envelopes:
        for level in sorted(envelopes.keys()):
            if level == 0:
                continue
            shape_comparisons[level] = compare_trajectory_envelopes(
                envelopes[0], envelopes[level])

    # Print results
    print("\n" + "=" * 60)
    print("TRAJECTORY ANALYSIS: Head Freezing")
    print("=" * 60)

    print("\n--- Phase Structure ---")
    for level in sorted(phases_by_level.keys()):
        all_phases = phases_by_level[level]
        if not all_phases or not all_phases[0]:
            continue
        first_run = all_phases[0]
        phase_seq = ' -> '.join(p['type'] for p in first_run)
        print(f"  {level:2d} frozen: {phase_seq}")

    print("\n--- DG Episodes ---")
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

    if shape_comparisons:
        print("\n--- Trajectory Shape vs Baseline ---")
        for level, comp in sorted(shape_comparisons.items()):
            print(f"  {level:2d} frozen: "
                  f"divergence={comp['mean_divergence']:.4f}, "
                  f"max@step {comp['divergence_step']}, "
                  f"correlation={comp['shape_correlation']:.3f}, "
                  f"overlap={comp['overlap_fraction']:.2f}")

    print("\n--- Endpoints ---")
    print(f"{'Frozen':>8} {'Mean Loss':>10} {'Std':>8} {'Mean DG':>8} {'Std DG':>8} {'N':>4}")
    print("-" * 50)
    for c in curve:
        print(f"{c['damage_level']:>8d} {c['mean_loss']:>10.4f} {c['std_loss']:>8.4f} "
              f"{c['mean_dg']:>8.3f} {c['std_dg']:>8.3f} {c['n']:>4d}")

    damage_levels_flat = []
    dg_indices_flat = []
    for level, entries in sorted(results_by_level.items()):
        for final_loss, loss_vals in entries:
            damage_levels_flat.append(level)
            dg_indices_flat.append(dg_index(loss_vals))

    slope, intercept, r_sq = dg_damage_regression(damage_levels_flat, dg_indices_flat)
    print(f"\nDG-Damage regression: slope={slope:.4f}, intercept={intercept:.4f}, R²={r_sq:.4f}")

    return {
        'curve': curve,
        'all_results': all_results,
        'results_by_level': results_by_level,
        'envelopes': envelopes,
    }


# ============================================================================
# Experiment 2: Cell-View GPT
# ============================================================================

def experiment_cell_view(num_reps=5, num_steps=200, n_layer=4):
    print("=" * 60)
    print("EXPERIMENT 2: Cell-View GPT (Stop-Gradient)")
    print("=" * 60)

    configs = [
        ExperimentConfig(
            name='baseline', n_layer=n_layer, num_steps=num_steps,
            perturbation_type='none', num_reps=num_reps, seed=42,
        ),
        ExperimentConfig(
            name='cell_view', n_layer=n_layer, num_steps=num_steps,
            perturbation_type='stop_gradient',
            perturbation_params={'layers': 'all'},
            num_reps=num_reps, seed=42,
        ),
    ]

    docs, uchars, BOS, vocab_size = load_dataset()
    results = run_sweep(configs, docs, uchars, BOS, vocab_size)

    save_results(results, os.path.join(os.path.dirname(__file__),
                 'results', 'experiment2_cell_view.json'))

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
# Experiment 3: Gradient Degradation
# ============================================================================

def experiment_gradient_degradation(num_reps=5, num_steps=200, n_layer=4):
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

    save_results(results, os.path.join(os.path.dirname(__file__),
                 'results', 'experiment3_gradient_degradation.json'))

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
# Experiment 4: Vision Radius Sweep
# ============================================================================

def experiment_vision_radius(num_reps=5, num_steps=200, n_layer=4):
    print("=" * 60)
    print("EXPERIMENT 4: Vision Radius Sweep")
    print("=" * 60)

    window_sizes = [1, 2, 4, 8, 16]

    configs = [
        ExperimentConfig(
            name='baseline', n_layer=n_layer, num_steps=num_steps,
            perturbation_type='none', num_reps=num_reps, seed=42,
        ),
    ]

    for ws in window_sizes:
        configs.append(ExperimentConfig(
            name=f'window_{ws}', n_layer=n_layer, num_steps=num_steps,
            perturbation_type='windowed_attention',
            perturbation_params={'window_size': ws},
            num_reps=num_reps, seed=42,
        ))

    docs, uchars, BOS, vocab_size = load_dataset()
    all_results = run_sweep(configs, docs, uchars, BOS, vocab_size)

    save_results(all_results, os.path.join(os.path.dirname(__file__),
                 'results', 'experiment4_vision_radius.json'))

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
        })

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

    return {'all_results': all_results, 'results_by_window': results_by_window}


# ============================================================================
# Experiment 5: Communication Topology
# ============================================================================

def experiment_communication_topology(num_reps=5, num_steps=200, n_layer=4):
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
                name=name, n_layer=n_layer, num_steps=num_steps,
                perturbation_type='none', num_reps=num_reps, seed=42,
            ))
        elif fraction <= 0.0:
            configs.append(ExperimentConfig(
                name=name, n_layer=n_layer, num_steps=num_steps,
                perturbation_type='stop_gradient',
                perturbation_params={'layers': 'all'},
                num_reps=num_reps, seed=42,
            ))
        else:
            configs.append(ExperimentConfig(
                name=name, n_layer=n_layer, num_steps=num_steps,
                perturbation_type='partial_stop_gradient',
                perturbation_params={'layers': 'all', 'pass_fraction': fraction},
                num_reps=num_reps, seed=42,
            ))

    docs, uchars, BOS, vocab_size = load_dataset()
    all_results = run_sweep(configs, docs, uchars, BOS, vocab_size)

    save_results(all_results, os.path.join(os.path.dirname(__file__),
                 'results', 'experiment5_communication.json'))

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

    return all_results


# ============================================================================
# Experiment 6: Courage vs. Caution
# ============================================================================

def experiment_courage_caution(num_reps=5, num_steps=200, n_layer=4):
    print("=" * 60)
    print("EXPERIMENT 6: Courage vs. Caution")
    print("=" * 60)

    configs = [
        ExperimentConfig(
            name='baseline', n_layer=n_layer, num_steps=num_steps,
            perturbation_type='none', num_reps=num_reps, seed=42,
        ),
        ExperimentConfig(
            name='cautious_cautious', n_layer=n_layer, num_steps=num_steps,
            perturbation_type='noisy_gradients',
            perturbation_params={'noise_std': 0.001},
            num_reps=num_reps, seed=42,
        ),
        ExperimentConfig(
            name='cautious_courageous', n_layer=n_layer, num_steps=num_steps,
            perturbation_type='sign_only_gradients',
            num_reps=num_reps, seed=42,
        ),
        ExperimentConfig(
            name='courageous_cautious', n_layer=n_layer, num_steps=num_steps,
            perturbation_type='dropout',
            perturbation_params={'drop_prob': 0.1},
            num_reps=num_reps, seed=42,
        ),
        ExperimentConfig(
            name='courageous_courageous', n_layer=n_layer, num_steps=num_steps,
            perturbation_type='noisy_gradients',
            perturbation_params={'noise_std': 0.1},
            num_reps=num_reps, seed=42,
        ),
    ]

    docs, uchars, BOS, vocab_size = load_dataset()
    all_results = run_sweep(configs, docs, uchars, BOS, vocab_size)

    save_results(all_results, os.path.join(os.path.dirname(__file__),
                 'results', 'experiment6_courage_caution.json'))

    print("\n" + "=" * 60)
    print("RESULTS: Courage vs. Caution")
    print("=" * 60)

    condition_names = ['baseline', 'cautious_cautious', 'cautious_courageous',
                       'courageous_cautious', 'courageous_courageous']

    print(f"\n{'Condition':>25} {'Mean Loss':>10} {'Mean DG':>8}")
    print("-" * 47)

    for name in condition_names:
        group = [r for r in all_results if r['config'].name == name]
        if not group:
            continue
        losses = [r['summary']['final_loss'] for r in group]
        dgs = [r['summary']['dg_index'] for r in group]
        mean_l = sum(losses) / len(losses)
        mean_dg = sum(dgs) / len(dgs)
        print(f"{name:>25} {mean_l:>10.4f} {mean_dg:>8.3f}")

    # Swarming index for composite analysis
    print("\n--- Swarming Index (chess-paper metric) ---")
    for name in condition_names:
        group = [r for r in all_results if r['config'].name == name]
        if group and group[0]['probe'].head_outputs:
            si = swarming_index(group[0]['probe'])
            print(f"  {name:>25}: swarm={si['swarming_ratio']:.4f}")

    return all_results
