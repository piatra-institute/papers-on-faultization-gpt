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

from model import (
    Hooks, Probe, TrainConfig,
    make_config, init_state_dict, load_dataset, train, generate,
    train_with_state, get_layer_keys, reset_layer, transplant_layer,
    assemble_chimera, copy_state_dict, tokenize, _forward_backward,
    _evaluate,
)
from perturbations import (
    make_zero_head, make_ablate_head, make_freeze_params,
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
    freeze_specific_heads, unfreeze_heads,
    make_gradual_noisy_gradients, make_noisy_gradients_scheduled,
)
import numpy as np
from metrics import (
    summarize_probe, dg_index, robustness_curve_with_dg,
    dg_damage_regression, trajectory_envelope, compare_trajectory_envelopes,
    per_step_rerouting, detect_phases, compute_delayed_gratification,
    head_contribution_evolution,
    cognitive_light_cone, collective_light_cone,
    goal_alignment_score, swarming_index,
    recovery_overshoot, regeneration_completeness,
    transplant_advantage, dual_objective_equilibrium,
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

    # Local loss (cell-view mode)
    local_loss: bool = False

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

def run_experiment(exp_config, docs=None, uchars=None, BOS=None, vocab_size=None,
                   val_docs=None):
    """Run a single experiment. Returns result dict."""
    if docs is None:
        docs, val_docs, uchars, BOS, vocab_size = load_dataset()

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

    def _apply_perturbation(pt, pp, hooks, pending_hooks, grad_hooks, config, seed):
        """Dispatch a single perturbation type. Modifies hooks/grad_hooks in place."""
        nonlocal frozen_heads
        if pt == 'none':
            pass

        elif pt == 'freeze_heads':
            num = pp.get('num_heads', 1)
            rng = random.Random(seed + 1000)
            frozen_heads, freeze_gh = freeze_random_heads(config, num, rng=rng)
            grad_hooks.extend(freeze_gh)

        elif pt == 'zero_head':
            layer = pp.get('layer', 0)
            head = pp.get('head', 0)
            name, fn = make_zero_head(layer, head, config['head_dim'])
            pending_hooks.append((name, fn))

        elif pt == 'noise_heads':
            num = pp.get('num_heads', 1)
            noise_std = pp.get('noise_std', 0.1)
            rng = random.Random(seed + 1000)
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
            name, fn = make_noise_injection(hook_name, noise_std,
                                            rng=random.Random(seed + 4000))
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
            grad_hooks.append(make_noisy_gradients(noise_std,
                              rng=np.random.RandomState(seed + 7000)))

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
            drop_rng = random.Random(seed + 5000)
            for li in range(config['n_layer']):
                name, fn = make_dropout(f'mlp_hidden.{li}', drop_prob,
                                        rng=drop_rng)
                pending_hooks.append((name, fn))

        elif pt == 'stochastic_relu':
            flip_prob = pp.get('flip_prob', 0.05)
            srelu_rng = random.Random(seed + 6000)
            for li in range(config['n_layer']):
                name, fn = make_stochastic_relu(f'mlp_hidden.{li}', flip_prob,
                                                rng=srelu_rng)
                pending_hooks.append((name, fn))

        elif pt == 'windowed_attention':
            window = pp.get('window_size', 4)
            for li in range(config['n_layer']):
                for h in range(config['n_head']):
                    name, fn = make_windowed_attention(li, h, window)
                    pending_hooks.append((name, fn))

        elif pt == 'sparse_attention':
            keep_prob = pp.get('keep_prob', 0.5)
            rng = random.Random(seed + 2000)
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
            rng = random.Random(seed + 3000)
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

    if pt == 'composite':
        for sub in pp.get('perturbations', []):
            _apply_perturbation(
                sub['type'], sub.get('params', {}),
                hooks, pending_hooks, grad_hooks, config, exp_config.seed)
    else:
        _apply_perturbation(pt, pp, hooks, pending_hooks, grad_hooks, config,
                            exp_config.seed)

    # --- Validate schedule + grad hook compatibility ---
    sched = exp_config.schedule
    sp = exp_config.schedule_params
    if grad_hooks and sched != 'chronic':
        raise ValueError(
            f"Schedule '{sched}' is not supported with gradient hooks. "
            f"Gradient hooks only support 'chronic' schedule."
        )

    # --- Apply schedule wrapping and register pending hooks ---
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
        local_loss=exp_config.local_loss,
        val_docs=val_docs,
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

def run_sweep(configs, docs=None, uchars=None, BOS=None, vocab_size=None,
              val_docs=None):
    if docs is None:
        docs, val_docs, uchars, BOS, vocab_size = load_dataset()

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
                local_loss=cfg.local_loss,
            )

            print(f"\n[{run_idx}/{total}] {cfg.name} (rep {rep+1}/{cfg.num_reps}, "
                  f"seed={rep_cfg.seed})")

            result = run_experiment(rep_cfg, docs, uchars, BOS, vocab_size,
                                    val_docs=val_docs)
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
        # Validation losses
        if r['probe'].val_losses:
            entry['val_trajectory'] = [(s, l) for s, l in r['probe'].val_losses]
            entry['summary']['val_final_loss'] = r['probe'].val_losses[-1][1]
            entry['summary']['val_mean_loss'] = sum(
                l for _, l in r['probe'].val_losses) / len(r['probe'].val_losses)
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

def experiment_head_freezing(num_reps=5, num_steps=200, n_layer=4, result_suffix=''):
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

    docs, val_docs, uchars, BOS, vocab_size = load_dataset()
    all_results = run_sweep(configs, docs, uchars, BOS, vocab_size, val_docs=val_docs)

    # Save results
    save_results(all_results, os.path.join(os.path.dirname(__file__),
                 'results', f'experiment1_head_freezing{result_suffix}.json'))

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

def experiment_cell_view(num_reps=5, num_steps=200, n_layer=4, result_suffix=''):
    print("=" * 60)
    print("EXPERIMENT 2: Cell-View GPT (Local Loss)")
    print("=" * 60)

    configs = [
        ExperimentConfig(
            name='baseline', n_layer=n_layer, num_steps=num_steps,
            perturbation_type='none', num_reps=num_reps, seed=42,
        ),
        ExperimentConfig(
            name='cell_view', n_layer=n_layer, num_steps=num_steps,
            perturbation_type='none',
            local_loss=True,
            num_reps=num_reps, seed=42,
        ),
    ]

    docs, val_docs, uchars, BOS, vocab_size = load_dataset()
    results = run_sweep(configs, docs, uchars, BOS, vocab_size, val_docs=val_docs)

    save_results(results, os.path.join(os.path.dirname(__file__),
                 'results', f'experiment2_cell_view{result_suffix}.json'))

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

def experiment_gradient_degradation(num_reps=5, num_steps=200, n_layer=4, result_suffix=''):
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

    docs, val_docs, uchars, BOS, vocab_size = load_dataset()
    results = run_sweep(configs, docs, uchars, BOS, vocab_size, val_docs=val_docs)

    save_results(results, os.path.join(os.path.dirname(__file__),
                 'results', f'experiment3_gradient_degradation{result_suffix}.json'))

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

def experiment_vision_radius(num_reps=5, num_steps=200, n_layer=4, result_suffix=''):
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

    docs, val_docs, uchars, BOS, vocab_size = load_dataset()
    all_results = run_sweep(configs, docs, uchars, BOS, vocab_size, val_docs=val_docs)

    save_results(all_results, os.path.join(os.path.dirname(__file__),
                 'results', f'experiment4_vision_radius{result_suffix}.json'))

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

def experiment_communication_topology(num_reps=5, num_steps=200, n_layer=4, result_suffix=''):
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
            # cell_view uses local loss (true cell-view mode)
            configs.append(ExperimentConfig(
                name=name, n_layer=n_layer, num_steps=num_steps,
                perturbation_type='none',
                local_loss=True,
                num_reps=num_reps, seed=42,
            ))
        else:
            configs.append(ExperimentConfig(
                name=name, n_layer=n_layer, num_steps=num_steps,
                perturbation_type='partial_stop_gradient',
                perturbation_params={'layers': 'all', 'pass_fraction': fraction},
                num_reps=num_reps, seed=42,
            ))

    docs, val_docs, uchars, BOS, vocab_size = load_dataset()
    all_results = run_sweep(configs, docs, uchars, BOS, vocab_size, val_docs=val_docs)

    save_results(all_results, os.path.join(os.path.dirname(__file__),
                 'results', f'experiment5_communication{result_suffix}.json'))

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

def experiment_courage_caution(num_reps=5, num_steps=200, n_layer=4, result_suffix=''):
    print("=" * 60)
    print("EXPERIMENT 6: Courage vs. Caution")
    print("=" * 60)

    configs = [
        ExperimentConfig(
            name='baseline', n_layer=n_layer, num_steps=num_steps,
            perturbation_type='none', num_reps=num_reps, seed=42,
        ),
        # Cautious forward (tiny noise) + Cautious gradient (sign-only)
        ExperimentConfig(
            name='cautious_cautious', n_layer=n_layer, num_steps=num_steps,
            perturbation_type='composite',
            perturbation_params={'perturbations': [
                {'type': 'noise_injection', 'params': {'hook_name': 'emb', 'noise_std': 0.001}},
                {'type': 'sign_only_gradients'},
            ]},
            num_reps=num_reps, seed=42,
        ),
        # Cautious forward (tiny noise) + Courageous gradient (noisy sigma=0.1)
        ExperimentConfig(
            name='cautious_courageous', n_layer=n_layer, num_steps=num_steps,
            perturbation_type='composite',
            perturbation_params={'perturbations': [
                {'type': 'noise_injection', 'params': {'hook_name': 'emb', 'noise_std': 0.001}},
                {'type': 'noisy_gradients', 'params': {'noise_std': 0.1}},
            ]},
            num_reps=num_reps, seed=42,
        ),
        # Courageous forward (dropout) + Cautious gradient (sign-only)
        ExperimentConfig(
            name='courageous_cautious', n_layer=n_layer, num_steps=num_steps,
            perturbation_type='composite',
            perturbation_params={'perturbations': [
                {'type': 'dropout', 'params': {'drop_prob': 0.1}},
                {'type': 'sign_only_gradients'},
            ]},
            num_reps=num_reps, seed=42,
        ),
        # Courageous forward (dropout) + Courageous gradient (noisy sigma=0.1)
        ExperimentConfig(
            name='courageous_courageous', n_layer=n_layer, num_steps=num_steps,
            perturbation_type='composite',
            perturbation_params={'perturbations': [
                {'type': 'dropout', 'params': {'drop_prob': 0.1}},
                {'type': 'noisy_gradients', 'params': {'noise_std': 0.1}},
            ]},
            num_reps=num_reps, seed=42,
        ),
    ]

    docs, val_docs, uchars, BOS, vocab_size = load_dataset()
    all_results = run_sweep(configs, docs, uchars, BOS, vocab_size, val_docs=val_docs)

    save_results(all_results, os.path.join(os.path.dirname(__file__),
                 'results', f'experiment6_courage_caution{result_suffix}.json'))

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


# ============================================================================
# Experiment 7: Recovery After Damage
# ============================================================================

def experiment_recovery(num_reps=5, num_steps=200, n_layer=4, result_suffix=''):
    """
    Test whether a model recovers after transient damage, and whether it overshoots.

    Phase 1: Normal training (num_steps steps)
    Phase 2: Damage — freeze 8 heads (num_steps//2 steps)
    Phase 3: Recovery — unfreeze all (num_steps steps)
    Control: Undamaged training for the same total duration
    """
    print("=" * 60)
    print("EXPERIMENT 7: Recovery After Damage")
    print("=" * 60)

    docs, val_docs, uchars, BOS, vocab_size = load_dataset()
    config = make_config(n_layer=n_layer, n_embd=16, n_head=4,
                         block_size=16, vocab_size=vocab_size)

    phase1_steps = num_steps
    phase2_steps = num_steps // 2
    phase3_steps = num_steps
    total = phase1_steps + phase2_steps + phase3_steps
    num_freeze = 8

    all_results = []

    for rep in range(num_reps):
        seed = 42 + rep
        print(f"\n[{rep+1}/{num_reps}] seed={seed}")

        # --- Recovery condition ---
        sd, params = init_state_dict(config, seed=seed)
        tc1 = TrainConfig(num_steps=phase1_steps, learning_rate=0.01,
                          print_every=0, detail_level='loss_only')

        # Phase 1: Normal training
        p1, m, v = train_with_state(
            sd, params, config, tc1, docs, uchars, BOS,
            seed=seed, total_steps=total)

        pre_damage_loss = p1.losses[-1][1] if p1.losses else 0

        # Phase 2: Damage (freeze 8 heads via gradient zeroing)
        rng = random.Random(seed + 1000)
        all_heads = [(li, h) for li in range(n_layer) for h in range(4)]
        rng.shuffle(all_heads)
        _, freeze_gh = freeze_specific_heads(all_heads[:num_freeze], config)

        tc2 = TrainConfig(num_steps=phase2_steps, learning_rate=0.01,
                          print_every=0, detail_level='loss_only')
        p2, m, v = train_with_state(
            sd, params, config, tc2, docs, uchars, BOS,
            grad_hooks=freeze_gh, seed=seed, m_buf=m, v_buf=v,
            start_step=phase1_steps, total_steps=total)

        # Phase 3: Recovery (no freeze grad hooks)
        tc3 = TrainConfig(num_steps=phase3_steps, learning_rate=0.01,
                          print_every=0, detail_level='loss_only')
        p3, m, v = train_with_state(
            sd, params, config, tc3, docs, uchars, BOS,
            seed=seed, m_buf=m, v_buf=v,
            start_step=phase1_steps + phase2_steps, total_steps=total)

        # Combine trajectory
        recovery_traj = p1.losses + p2.losses + p3.losses

        # --- Control: undamaged ---
        sd_c, params_c = init_state_dict(config, seed=seed)
        tc_c = TrainConfig(num_steps=total, learning_rate=0.01,
                           print_every=0, detail_level='loss_only')
        p_c, _, _ = train_with_state(
            sd_c, params_c, config, tc_c, docs, uchars, BOS,
            seed=seed, total_steps=total)
        control_traj = p_c.losses

        # Metrics
        damage_end = phase1_steps + phase2_steps
        rec_metrics = recovery_overshoot(control_traj, recovery_traj, damage_end)
        rec_final = recovery_traj[-1][1] if recovery_traj else 0
        ctrl_final = control_traj[-1][1] if control_traj else 0

        result = {
            'seed': seed,
            'recovery_final_loss': float(rec_final),
            'control_final_loss': float(ctrl_final),
            'pre_damage_loss': float(pre_damage_loss),
            'recovery_time': rec_metrics['recovery_time'],
            'overshoot': float(rec_metrics['overshoot']),
            'max_overshoot': float(rec_metrics['max_overshoot']),
            'final_ratio': float(rec_metrics['final_ratio']),
        }
        all_results.append(result)
        print(f"  recovery={rec_final:.4f} control={ctrl_final:.4f} "
              f"ratio={rec_metrics['final_ratio']:.4f}")

    # Save
    save_path = os.path.join(os.path.dirname(__file__), 'results',
                             f'experiment7_recovery{result_suffix}.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved {len(all_results)} results to {save_path}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS: Recovery After Damage")
    print("=" * 60)

    rec_finals = [r['recovery_final_loss'] for r in all_results]
    ctrl_finals = [r['control_final_loss'] for r in all_results]
    ratios = [r['final_ratio'] for r in all_results]
    rec_times = [r['recovery_time'] for r in all_results
                 if r['recovery_time'] is not None]

    print(f"  Recovery final:  {np.mean(rec_finals):.4f} +/- {np.std(rec_finals):.4f}")
    print(f"  Control final:   {np.mean(ctrl_finals):.4f} +/- {np.std(ctrl_finals):.4f}")
    print(f"  Final ratio:     {np.mean(ratios):.4f}")
    if rec_times:
        print(f"  Recovery time:   {np.mean(rec_times):.0f} +/- {np.std(rec_times):.0f} steps "
              f"({len(rec_times)}/{num_reps} recovered)")
    else:
        print(f"  No runs recovered to baseline level")

    return all_results


# ============================================================================
# Experiment 8: Chimera Assembly
# ============================================================================

def experiment_chimera(num_reps=5, num_steps=200, n_layer=4, result_suffix=''):
    """
    Test: Can a model assembled from parts of two separately-trained models learn?

    Train model A and model B independently for num_steps steps.
    Create chimeras with different layer assignments.
    Continue training each chimera for num_steps steps.
    """
    print("=" * 60)
    print("EXPERIMENT 8: Chimera Assembly")
    print("=" * 60)

    docs, val_docs, uchars, BOS, vocab_size = load_dataset()
    config = make_config(n_layer=n_layer, n_embd=16, n_head=4,
                         block_size=16, vocab_size=vocab_size)

    # Layer assignments for chimeras (for 4 layers)
    assignments = {
        'AABB': {0: 'A', 1: 'A', 2: 'B', 3: 'B'},
        'ABAB': {0: 'A', 1: 'B', 2: 'A', 3: 'B'},
        'BBAA': {0: 'B', 1: 'B', 2: 'A', 3: 'A'},
        'ABBA': {0: 'A', 1: 'B', 2: 'B', 3: 'A'},
    }

    all_results = []

    for rep in range(num_reps):
        seed_a = 42 + rep
        seed_b = 1042 + rep
        print(f"\n[{rep+1}/{num_reps}] seeds=({seed_a}, {seed_b})")

        # Train model A
        sd_a, params_a = init_state_dict(config, seed=seed_a)
        tc = TrainConfig(num_steps=num_steps, learning_rate=0.01,
                         print_every=0, detail_level='loss_only')
        p_a, _, _ = train_with_state(
            sd_a, params_a, config, tc, docs, uchars, BOS,
            seed=seed_a, total_steps=num_steps * 2)

        a_phase1_loss = p_a.losses[-1][1] if p_a.losses else 0

        # Train model B
        sd_b, params_b = init_state_dict(config, seed=seed_b)
        p_b, _, _ = train_with_state(
            sd_b, params_b, config, tc, docs, uchars, BOS,
            seed=seed_b, total_steps=num_steps * 2)

        b_phase1_loss = p_b.losses[-1][1] if p_b.losses else 0

        # Control: model A continues training
        sd_ctrl = copy_state_dict(sd_a)
        tc2 = TrainConfig(num_steps=num_steps, learning_rate=0.01,
                          print_every=0, detail_level='loss_only')
        p_ctrl, _, _ = train_with_state(
            sd_ctrl, params_a, config, tc2, docs, uchars, BOS,
            seed=seed_a, start_step=num_steps, total_steps=num_steps * 2)

        ctrl_final = p_ctrl.losses[-1][1] if p_ctrl.losses else 0

        # Create and train chimeras
        rep_result = {
            'seed_a': seed_a, 'seed_b': seed_b,
            'a_phase1_loss': float(a_phase1_loss),
            'b_phase1_loss': float(b_phase1_loss),
            'control_final_loss': float(ctrl_final),
            'chimeras': {},
        }

        for name, assignment in assignments.items():
            sd_chi = assemble_chimera(sd_a, sd_b, assignment, config)

            # Measure initial chimera loss
            tokens = tokenize(docs[0], uchars, BOS)
            n = min(config['block_size'], len(tokens) - 1)
            init_loss, _, _, _ = _forward_backward(
                tokens, n, sd_chi, config, Hooks(), capture_state=False)

            # Train chimera
            tc_chi = TrainConfig(num_steps=num_steps, learning_rate=0.01,
                                 print_every=0, detail_level='loss_only')
            p_chi, _, _ = train_with_state(
                sd_chi, params_a, config, tc_chi, docs, uchars, BOS,
                seed=seed_a, start_step=num_steps, total_steps=num_steps * 2)

            chi_final = p_chi.losses[-1][1] if p_chi.losses else 0

            rep_result['chimeras'][name] = {
                'initial_loss': float(init_loss),
                'final_loss': float(chi_final),
                'recovery': float((init_loss - chi_final) / max(init_loss, 1e-10)),
            }
            print(f"  {name}: init={init_loss:.4f} final={chi_final:.4f}")

        all_results.append(rep_result)

    # Save
    save_path = os.path.join(os.path.dirname(__file__), 'results',
                             f'experiment8_chimera{result_suffix}.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved {len(all_results)} results to {save_path}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS: Chimera Assembly")
    print("=" * 60)

    ctrl_losses = [r['control_final_loss'] for r in all_results]
    print(f"  Control (A continues): {np.mean(ctrl_losses):.4f} +/- {np.std(ctrl_losses):.4f}")

    for name in assignments:
        chi_inits = [r['chimeras'][name]['initial_loss'] for r in all_results]
        chi_finals = [r['chimeras'][name]['final_loss'] for r in all_results]
        recoveries = [r['chimeras'][name]['recovery'] for r in all_results]
        print(f"  {name}: init={np.mean(chi_inits):.4f} "
              f"final={np.mean(chi_finals):.4f}+/-{np.std(chi_finals):.4f} "
              f"recovery={np.mean(recoveries)*100:.1f}%")

    return all_results


# ============================================================================
# Experiment 9: Gradual vs Sudden Damage
# ============================================================================

def experiment_gradual_vs_sudden(num_reps=5, num_steps=200, n_layer=4, result_suffix=''):
    """
    Test: Does gradual damage exposure build tolerance compared to sudden damage?

    Condition A: No noise (control)
    Condition B: Sudden noise sigma=0.1 for all steps
    Condition C: Gradual ramp from sigma=0 to sigma=0.1 over all steps
    Condition D: Noise only in second half (sudden onset at step num_steps//2)
    """
    print("=" * 60)
    print("EXPERIMENT 9: Gradual vs Sudden Damage")
    print("=" * 60)

    docs, val_docs, uchars, BOS, vocab_size = load_dataset()
    config = make_config(n_layer=n_layer, n_embd=16, n_head=4,
                         block_size=16, vocab_size=vocab_size)

    noise_std = 0.1
    all_results = []

    for rep in range(num_reps):
        seed = 42 + rep
        print(f"\n[{rep+1}/{num_reps}] seed={seed}")
        rep_result = {'seed': seed, 'conditions': {}}

        tc = TrainConfig(num_steps=num_steps, learning_rate=0.01,
                         print_every=0, detail_level='loss_only')

        # A: Control (no noise)
        sd_a, params_a = init_state_dict(config, seed=seed)
        p_a, _, _ = train_with_state(
            sd_a, params_a, config, tc, docs, uchars, BOS,
            seed=seed, total_steps=num_steps)
        rep_result['conditions']['control'] = {
            'final_loss': float(p_a.losses[-1][1]) if p_a.losses else 0,
            'mean_loss': float(np.mean([l for _, l in p_a.losses])),
        }

        # B: Sudden noise (all steps)
        sd_b, params_b = init_state_dict(config, seed=seed)
        gh_b = [make_noisy_gradients(noise_std,
                rng=np.random.RandomState(seed + 7000))]
        p_b, _, _ = train_with_state(
            sd_b, params_b, config, tc, docs, uchars, BOS,
            grad_hooks=gh_b, seed=seed, total_steps=num_steps)
        rep_result['conditions']['sudden_full'] = {
            'final_loss': float(p_b.losses[-1][1]) if p_b.losses else 0,
            'mean_loss': float(np.mean([l for _, l in p_b.losses])),
        }

        # C: Gradual ramp
        sd_c, params_c = init_state_dict(config, seed=seed)
        gh_c = [make_gradual_noisy_gradients(noise_std, num_steps,
                rng=np.random.RandomState(seed + 7000))]
        p_c, _, _ = train_with_state(
            sd_c, params_c, config, tc, docs, uchars, BOS,
            grad_hooks=gh_c, seed=seed, total_steps=num_steps)
        rep_result['conditions']['gradual'] = {
            'final_loss': float(p_c.losses[-1][1]) if p_c.losses else 0,
            'mean_loss': float(np.mean([l for _, l in p_c.losses])),
        }

        # D: Sudden onset at halfway
        sd_d, params_d = init_state_dict(config, seed=seed)
        gh_d = [make_noisy_gradients_scheduled(noise_std, start_step=num_steps // 2,
                rng=np.random.RandomState(seed + 7000))]
        p_d, _, _ = train_with_state(
            sd_d, params_d, config, tc, docs, uchars, BOS,
            grad_hooks=gh_d, seed=seed, total_steps=num_steps)
        rep_result['conditions']['sudden_half'] = {
            'final_loss': float(p_d.losses[-1][1]) if p_d.losses else 0,
            'mean_loss': float(np.mean([l for _, l in p_d.losses])),
        }

        all_results.append(rep_result)
        for name, data in rep_result['conditions'].items():
            print(f"  {name:15s}: final={data['final_loss']:.4f}")

    # Save
    save_path = os.path.join(os.path.dirname(__file__), 'results',
                             f'experiment9_gradual_vs_sudden{result_suffix}.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved {len(all_results)} results to {save_path}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS: Gradual vs Sudden Damage")
    print("=" * 60)

    for cond_name in ['control', 'sudden_full', 'gradual', 'sudden_half']:
        finals = [r['conditions'][cond_name]['final_loss'] for r in all_results]
        means = [r['conditions'][cond_name]['mean_loss'] for r in all_results]
        print(f"  {cond_name:15s}: final={np.mean(finals):.4f}+/-{np.std(finals):.4f}  "
              f"mean={np.mean(means):.4f}+/-{np.std(means):.4f}")

    return all_results


# ============================================================================
# Experiment 10: Regeneration (Layer Reset)
# ============================================================================

def experiment_regeneration(num_reps=5, num_steps=200, n_layer=4, result_suffix=''):
    """
    Test: Can a model regenerate after a layer is destroyed?

    Phase 1: Normal training (num_steps steps)
    Phase 2: Reset layer L to random, continue training (num_steps steps)
    Control: No reset, continue training
    Test layers: 0 (early), 1 (mid-early), 2 (mid-late), 3 (late)
    """
    print("=" * 60)
    print("EXPERIMENT 10: Regeneration (Layer Reset)")
    print("=" * 60)

    docs, val_docs, uchars, BOS, vocab_size = load_dataset()
    config = make_config(n_layer=n_layer, n_embd=16, n_head=4,
                         block_size=16, vocab_size=vocab_size)

    total = num_steps * 2
    all_results = []

    for rep in range(num_reps):
        seed = 42 + rep
        print(f"\n[{rep+1}/{num_reps}] seed={seed}")

        # Phase 1: Train baseline
        sd_base, params = init_state_dict(config, seed=seed)
        tc1 = TrainConfig(num_steps=num_steps, learning_rate=0.01,
                          print_every=0, detail_level='loss_only')
        p1, m_base, v_base = train_with_state(
            sd_base, params, config, tc1, docs, uchars, BOS,
            seed=seed, total_steps=total)

        pre_reset_loss = p1.losses[-1][1] if p1.losses else 0

        # Control: continue without reset
        sd_ctrl = copy_state_dict(sd_base)
        m_ctrl = {k: v.copy() for k, v in m_base.items()}
        v_ctrl = {k: v.copy() for k, v in v_base.items()}
        tc2 = TrainConfig(num_steps=num_steps, learning_rate=0.01,
                          print_every=0, detail_level='loss_only')
        p_ctrl, _, _ = train_with_state(
            sd_ctrl, params, config, tc2, docs, uchars, BOS,
            seed=seed, m_buf=m_ctrl, v_buf=v_ctrl,
            start_step=num_steps, total_steps=total)
        ctrl_final = p_ctrl.losses[-1][1] if p_ctrl.losses else 0

        rep_result = {
            'seed': seed,
            'pre_reset_loss': float(pre_reset_loss),
            'control_final_loss': float(ctrl_final),
            'layers': {},
        }

        # Test each layer
        for li in range(n_layer):
            sd_reset = copy_state_dict(sd_base)
            m_reset = {k: v.copy() for k, v in m_base.items()}
            v_reset = {k: v.copy() for k, v in v_base.items()}

            # Reset layer li
            reset_layer(sd_reset, li, config, seed=seed + li * 100,
                        m_buf=m_reset, v_buf=v_reset)

            # Measure immediate post-reset loss
            tokens = tokenize(docs[0], uchars, BOS)
            n = min(config['block_size'], len(tokens) - 1)
            post_reset_loss, _, _, _ = _forward_backward(
                tokens, n, sd_reset, config, Hooks(), capture_state=False)

            # Continue training
            p_regen, _, _ = train_with_state(
                sd_reset, params, config, tc2, docs, uchars, BOS,
                seed=seed, m_buf=m_reset, v_buf=v_reset,
                start_step=num_steps, total_steps=total)
            regen_final = p_regen.losses[-1][1] if p_regen.losses else 0

            # Regeneration metrics
            baseline_traj = p_ctrl.losses
            regen_traj = p_regen.losses
            regen_metrics = regeneration_completeness(
                pre_reset_loss, regen_traj, baseline_traj, num_steps)

            rep_result['layers'][str(li)] = {
                'post_reset_loss': float(post_reset_loss),
                'regen_final_loss': float(regen_final),
                'completeness': float(regen_metrics['completeness']),
                'damage': float(post_reset_loss - pre_reset_loss),
            }
            print(f"  L{li}: post_reset={post_reset_loss:.4f} "
                  f"final={regen_final:.4f} "
                  f"completeness={regen_metrics['completeness']:.3f}")

        all_results.append(rep_result)

    # Save
    save_path = os.path.join(os.path.dirname(__file__), 'results',
                             f'experiment10_regeneration{result_suffix}.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved {len(all_results)} results to {save_path}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS: Regeneration (Layer Reset)")
    print("=" * 60)

    ctrl_finals = [r['control_final_loss'] for r in all_results]
    print(f"  Control final: {np.mean(ctrl_finals):.4f} +/- {np.std(ctrl_finals):.4f}")

    for li in range(n_layer):
        damages = [r['layers'][str(li)]['damage'] for r in all_results]
        finals = [r['layers'][str(li)]['regen_final_loss'] for r in all_results]
        comps = [r['layers'][str(li)]['completeness'] for r in all_results]
        print(f"  Layer {li}: damage={np.mean(damages):.4f} "
              f"final={np.mean(finals):.4f}+/-{np.std(finals):.4f} "
              f"completeness={np.mean(comps):.3f}")

    return all_results


# ============================================================================
# Experiment 11: Transplantation
# ============================================================================

def experiment_transplantation(num_reps=5, num_steps=200, n_layer=4, result_suffix=''):
    """
    Test: Is a transplanted layer accepted better than a random reset?

    Train model A and model B for num_steps steps.
    For each layer L:
      - Transplant: replace layer L of A with layer L of B, continue training
      - Random: reset layer L of A to random, continue training
      - Control: A continues without modification
    """
    print("=" * 60)
    print("EXPERIMENT 11: Transplantation")
    print("=" * 60)

    docs, val_docs, uchars, BOS, vocab_size = load_dataset()
    config = make_config(n_layer=n_layer, n_embd=16, n_head=4,
                         block_size=16, vocab_size=vocab_size)

    total = num_steps * 2
    all_results = []

    for rep in range(num_reps):
        seed_a = 42 + rep
        seed_b = 1042 + rep
        print(f"\n[{rep+1}/{num_reps}] seeds=({seed_a}, {seed_b})")

        # Train model A
        sd_a, params_a = init_state_dict(config, seed=seed_a)
        tc1 = TrainConfig(num_steps=num_steps, learning_rate=0.01,
                          print_every=0, detail_level='loss_only')
        p_a, m_a, v_a = train_with_state(
            sd_a, params_a, config, tc1, docs, uchars, BOS,
            seed=seed_a, total_steps=total)

        # Train model B
        sd_b, params_b = init_state_dict(config, seed=seed_b)
        p_b, _, _ = train_with_state(
            sd_b, params_b, config, tc1, docs, uchars, BOS,
            seed=seed_b, total_steps=total)

        a_loss = p_a.losses[-1][1] if p_a.losses else 0

        # Control: A continues
        sd_ctrl = copy_state_dict(sd_a)
        m_ctrl = {k: v.copy() for k, v in m_a.items()}
        v_ctrl = {k: v.copy() for k, v in v_a.items()}
        tc2 = TrainConfig(num_steps=num_steps, learning_rate=0.01,
                          print_every=0, detail_level='loss_only')
        p_ctrl, _, _ = train_with_state(
            sd_ctrl, params_a, config, tc2, docs, uchars, BOS,
            seed=seed_a, m_buf=m_ctrl, v_buf=v_ctrl,
            start_step=num_steps, total_steps=total)
        ctrl_final = p_ctrl.losses[-1][1] if p_ctrl.losses else 0

        rep_result = {
            'seed_a': seed_a, 'seed_b': seed_b,
            'a_loss_at_transplant': float(a_loss),
            'control_final_loss': float(ctrl_final),
            'layers': {},
        }

        for li in range(n_layer):
            # Transplant condition
            sd_trans = copy_state_dict(sd_a)
            m_trans = {k: v.copy() for k, v in m_a.items()}
            v_trans = {k: v.copy() for k, v in v_a.items()}
            transplant_layer(sd_trans, sd_b, li, m_buf=m_trans, v_buf=v_trans)

            p_trans, _, _ = train_with_state(
                sd_trans, params_a, config, tc2, docs, uchars, BOS,
                seed=seed_a, m_buf=m_trans, v_buf=v_trans,
                start_step=num_steps, total_steps=total)
            trans_final = p_trans.losses[-1][1] if p_trans.losses else 0

            # Random reset condition
            sd_rand = copy_state_dict(sd_a)
            m_rand = {k: v.copy() for k, v in m_a.items()}
            v_rand = {k: v.copy() for k, v in v_a.items()}
            reset_layer(sd_rand, li, config, seed=seed_a + li * 100,
                        m_buf=m_rand, v_buf=v_rand)

            p_rand, _, _ = train_with_state(
                sd_rand, params_a, config, tc2, docs, uchars, BOS,
                seed=seed_a, m_buf=m_rand, v_buf=v_rand,
                start_step=num_steps, total_steps=total)
            rand_final = p_rand.losses[-1][1] if p_rand.losses else 0

            # Metrics
            ta = transplant_advantage(trans_final, rand_final, ctrl_final)

            rep_result['layers'][str(li)] = {
                'transplant_final': float(trans_final),
                'random_final': float(rand_final),
                'transplant_gap': float(ta['transplant_gap']),
                'transplant_vs_baseline': float(ta['transplant_vs_baseline']),
                'random_vs_baseline': float(ta['random_vs_baseline']),
            }
            print(f"  L{li}: trans={trans_final:.4f} rand={rand_final:.4f} "
                  f"gap={ta['transplant_gap']:.4f}")

        all_results.append(rep_result)

    # Save
    save_path = os.path.join(os.path.dirname(__file__), 'results',
                             f'experiment11_transplantation{result_suffix}.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved {len(all_results)} results to {save_path}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS: Transplantation")
    print("=" * 60)

    ctrl_finals = [r['control_final_loss'] for r in all_results]
    print(f"  Control final: {np.mean(ctrl_finals):.4f} +/- {np.std(ctrl_finals):.4f}")

    for li in range(n_layer):
        trans_finals = [r['layers'][str(li)]['transplant_final'] for r in all_results]
        rand_finals = [r['layers'][str(li)]['random_final'] for r in all_results]
        gaps = [r['layers'][str(li)]['transplant_gap'] for r in all_results]
        print(f"  Layer {li}: transplant={np.mean(trans_finals):.4f} "
              f"random={np.mean(rand_finals):.4f} "
              f"gap={np.mean(gaps):.4f}+/-{np.std(gaps):.4f}")

    return all_results


# ============================================================================
# Experiment 12: Competing Objectives
# ============================================================================

def experiment_competing_objectives(num_reps=5, num_steps=200, n_layer=4, result_suffix=''):
    """
    Test: Can layers specialize when facing conflicting gradient signals?

    Phase 1: Normal training (num_steps steps)
    Phase 2: Layers 0-1 get normal gradients, layers 2-3 get negated gradients
             (num_steps steps)
    Control: Normal training for full duration

    Measures whether early layers compensate for adversarial late layers.
    """
    print("=" * 60)
    print("EXPERIMENT 12: Competing Objectives")
    print("=" * 60)

    docs, val_docs, uchars, BOS, vocab_size = load_dataset()
    config = make_config(n_layer=n_layer, n_embd=16, n_head=4,
                         block_size=16, vocab_size=vocab_size)

    total = num_steps * 2
    layer_comps = ['attn_wq', 'attn_wk', 'attn_wv', 'attn_wo', 'mlp_fc1', 'mlp_fc2']

    all_results = []

    for rep in range(num_reps):
        seed = 42 + rep
        print(f"\n[{rep+1}/{num_reps}] seed={seed}")

        # --- Control: full normal training ---
        sd_ctrl, params_ctrl = init_state_dict(config, seed=seed)
        tc_full = TrainConfig(num_steps=total, learning_rate=0.01,
                              print_every=0, detail_level='loss_only')
        p_ctrl, _, _ = train_with_state(
            sd_ctrl, params_ctrl, config, tc_full, docs, uchars, BOS,
            seed=seed, total_steps=total)
        ctrl_final = p_ctrl.losses[-1][1] if p_ctrl.losses else 0

        # --- Competing condition ---
        sd, params = init_state_dict(config, seed=seed)
        tc1 = TrainConfig(num_steps=num_steps, learning_rate=0.01,
                          print_every=0, detail_level='loss_only')

        # Phase 1: Normal training
        p1, m, v = train_with_state(
            sd, params, config, tc1, docs, uchars, BOS,
            seed=seed, total_steps=total)
        phase1_final = p1.losses[-1][1] if p1.losses else 0

        # Phase 2: Adversarial layers 2-3
        adversarial_layers = [2, 3] if n_layer >= 4 else [n_layer - 1]

        def make_negate_hook(adv_layers):
            def negate_layers_hook(grads, state_dict, step):
                for li in adv_layers:
                    for comp in layer_comps:
                        key = f'layer{li}.{comp}'
                        if key in grads:
                            grads[key] = -grads[key]
            return negate_layers_hook

        tc2 = TrainConfig(num_steps=num_steps, learning_rate=0.01,
                          print_every=0, detail_level='loss_only')
        p2, _, _ = train_with_state(
            sd, params, config, tc2, docs, uchars, BOS,
            grad_hooks=[make_negate_hook(adversarial_layers)], seed=seed,
            m_buf=m, v_buf=v,
            start_step=num_steps, total_steps=total)
        compete_final = p2.losses[-1][1] if p2.losses else 0

        # --- Freeze-adversarial condition (freeze layers 2-3 instead) ---
        sd_f, params_f = init_state_dict(config, seed=seed)
        p_f1, m_f, v_f = train_with_state(
            sd_f, params_f, config, tc1, docs, uchars, BOS,
            seed=seed, total_steps=total)

        def make_freeze_hook(freeze_layers):
            def freeze_layers_hook(grads, state_dict, step):
                for li in freeze_layers:
                    for comp in layer_comps:
                        key = f'layer{li}.{comp}'
                        if key in grads:
                            grads[key][:] = 0
            return freeze_layers_hook

        p_f2, _, _ = train_with_state(
            sd_f, params_f, config, tc2, docs, uchars, BOS,
            grad_hooks=[make_freeze_hook(adversarial_layers)], seed=seed,
            m_buf=m_f, v_buf=v_f,
            start_step=num_steps, total_steps=total)
        freeze_final = p_f2.losses[-1][1] if p_f2.losses else 0

        result = {
            'seed': seed,
            'control_final_loss': float(ctrl_final),
            'phase1_final_loss': float(phase1_final),
            'competing_final_loss': float(compete_final),
            'freeze_final_loss': float(freeze_final),
            'compete_vs_control': float((compete_final - ctrl_final) / max(ctrl_final, 1e-10) * 100),
            'freeze_vs_control': float((freeze_final - ctrl_final) / max(ctrl_final, 1e-10) * 100),
        }
        all_results.append(result)
        print(f"  ctrl={ctrl_final:.4f} compete={compete_final:.4f} "
              f"freeze={freeze_final:.4f}")

    # Save
    save_path = os.path.join(os.path.dirname(__file__), 'results',
                             f'experiment12_competing_objectives{result_suffix}.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved {len(all_results)} results to {save_path}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS: Competing Objectives")
    print("=" * 60)

    ctrl_finals = [r['control_final_loss'] for r in all_results]
    compete_finals = [r['competing_final_loss'] for r in all_results]
    freeze_finals = [r['freeze_final_loss'] for r in all_results]

    print(f"  Control:     {np.mean(ctrl_finals):.4f} +/- {np.std(ctrl_finals):.4f}")
    print(f"  Competing:   {np.mean(compete_finals):.4f} +/- {np.std(compete_finals):.4f}")
    print(f"  Freeze L2-3: {np.mean(freeze_finals):.4f} +/- {np.std(freeze_finals):.4f}")

    compete_pct = [r['compete_vs_control'] for r in all_results]
    freeze_pct = [r['freeze_vs_control'] for r in all_results]
    print(f"  Competing vs control: {np.mean(compete_pct):+.1f}%")
    print(f"  Freeze vs control:    {np.mean(freeze_pct):+.1f}%")

    return all_results
