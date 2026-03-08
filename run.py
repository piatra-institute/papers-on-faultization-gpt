# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
"""
MorphoGPT — Quick Run Script

Uses the numpy backend (morphogpt_np) for ~1000x speedup over scalar autograd.

Usage:
    uv run run.py test           # Quick smoke test (20 steps, n_layer=1)
    uv run run.py baseline       # Train baseline — trajectory report
    uv run run.py trajectory     # Baseline vs damaged trajectory comparison
    uv run run.py experiment1    # Head freezing robustness curve
    uv run run.py experiment2    # Cell-view GPT (stop-gradient)
    uv run run.py experiment3    # Gradient degradation
    uv run run.py experiment4    # Vision radius sweep (chess paper)
    uv run run.py experiment5    # Communication topology (chess paper)
    uv run run.py experiment6    # Courage vs. caution (chess paper)
    uv run run.py analyze1       # Analyze experiment 1 results (plots + tables)
    uv run run.py analyze2       # Analyze experiment 2 results (plots + tables)
    uv run run.py analyze3       # Analyze experiment 3 results (plots + tables)
    uv run run.py analyze4       # Analyze experiment 4 results (plots + tables)
    uv run run.py analyze5       # Analyze experiment 5 results (plots + tables)
    uv run run.py analyze6       # Analyze experiment 6 results (plots + tables)
    uv run run.py all            # Run all experiments
"""

import sys
import os

# Add current dir to path
sys.path.insert(0, os.path.dirname(__file__))


def test():
    """Quick smoke test: 20 steps, n_layer=1."""
    from morphogpt_np import (
        make_config, init_state_dict, load_dataset, train, generate,
        TrainConfig, Hooks, Probe
    )
    import time
    import numpy as np

    print("=== SMOKE TEST (numpy backend) ===")
    docs, uchars, BOS, vocab_size = load_dataset()
    print(f"docs: {len(docs)}, vocab: {vocab_size}")

    config = make_config(n_layer=1, n_embd=16, n_head=4, vocab_size=vocab_size)
    state_dict, params = init_state_dict(config, seed=42)
    n_params = sum(v.size for v in state_dict.values())
    print(f"params: {n_params}")

    tc = TrainConfig(num_steps=20, print_every=5, detail_level='summary')
    t0 = time.time()
    probe = train(state_dict, params, config, tc, docs, uchars, BOS)
    elapsed = time.time() - t0
    print(f"Final loss: {probe.losses[-1][1]:.4f}")
    print(f"Steps with trajectory data: {len(probe.step_data)}")
    print(f"Training time: {elapsed:.2f}s")

    samples = generate(state_dict, config, uchars, BOS,
                       num_samples=5, temperature=0.5, seed=123)
    print("\nSamples:")
    for i, s in enumerate(samples):
        print(f"  {i+1}: {s}")

    # Test hooks
    print("\nHook system...")
    hooks = Hooks()
    hooks.register('emb', lambda v, step=0: None)
    print(f"  registered: {hooks.list_hooks()}")

    # Test with a freeze hook
    from perturbations_np import make_zero_head, freeze_random_heads
    hooks2 = Hooks()
    frozen = freeze_random_heads(hooks2, config, num_heads=2)
    print(f"  frozen heads: {frozen}")
    print(f"  active hooks: {hooks2.list_hooks()}")

    # Train with frozen heads
    state_dict2, params2 = init_state_dict(config, seed=42)
    tc2 = TrainConfig(num_steps=20, print_every=10, detail_level='summary')
    probe2 = train(state_dict2, params2, config, tc2, docs, uchars, BOS, hooks=hooks2)
    print(f"  loss with {len(frozen)} frozen heads: {probe2.losses[-1][1]:.4f}")
    print(f"  head norm entries: {len(probe2.head_outputs)}")

    # Test metrics
    from metrics import summarize_probe, compute_delayed_gratification, detect_phases
    summary = summarize_probe(probe)
    print(f"\nMetrics summary: {summary}")

    phases = detect_phases(probe.get_loss_values())
    print(f"Phases detected: {len(phases)}")

    print("\n=== ALL TESTS PASSED ===")


def baseline():
    """Train baseline model — trajectory-focused report."""
    from morphogpt_np import (
        make_config, init_state_dict, load_dataset, train, generate,
        TrainConfig, Probe
    )
    from metrics import (
        summarize_probe, detect_phases, compute_delayed_gratification,
        head_contribution_evolution, mid_trajectory_peak_detection,
        detect_anomalies,
    )
    from visualize import (
        plot_loss_with_phases, plot_head_norm_heatmap,
        plot_entropy_evolution, plot_anomaly_timeline,
        plot_head_contribution_evolution,
    )

    print("=== BASELINE TRAJECTORY REPORT (n_layer=4, 500 steps) ===")
    docs, uchars, BOS, vocab_size = load_dataset()
    config = make_config(n_layer=4, n_embd=16, n_head=4, vocab_size=vocab_size)
    state_dict, params = init_state_dict(config, seed=42)
    print(f"params: {len(params)}")

    tc = TrainConfig(num_steps=500, print_every=50, detail_level='summary')
    probe = train(state_dict, params, config, tc, docs, uchars, BOS,
                  probe=Probe(record_interval=5, detail_level='summary'))

    loss_vals = probe.get_loss_values()

    # --- Phase Detection ---
    print("\n--- Training Phases ---")
    phases = detect_phases(loss_vals)
    for p in phases:
        print(f"  steps {p['start']:3d}-{p['end']:3d}: {p['type']:15s} "
              f"(slope={p['mean_slope']:+.4f}, loss={p['mean_loss']:.3f})")

    # --- DG Episodes as Events ---
    print("\n--- Delayed Gratification Episodes ---")
    dg_eps = compute_delayed_gratification(loss_vals)
    if dg_eps:
        for i, ep in enumerate(dg_eps):
            print(f"  episode {i+1}: steps {ep['start']}-{ep['end']} "
                  f"(peak at {ep['peak']}, "
                  f"temp loss={ep['temporary_loss']:.4f}, "
                  f"net gain={ep['net_gain']:.4f}, "
                  f"DG index={ep['dg_index']:.3f})")
    else:
        print("  no DG episodes detected")

    # --- Per-Head Dynamics ---
    print("\n--- Per-Head Contribution Fractions ---")
    fractions = probe.get_head_contribution_fractions()
    if fractions:
        for (li, hi), frac in sorted(fractions.items()):
            bar = '#' * int(frac * 40)
            print(f"  L{li}H{hi}: {frac:.3f} {bar}")

    # Mid-trajectory peaks in head norms
    print("\n--- Mid-Trajectory Phenomena (15-50% progress) ---")
    found_peaks = False
    for (li, hi) in sorted(probe.head_outputs.keys()):
        traj = probe.get_head_norm_trajectory(li, hi)
        peaks = mid_trajectory_peak_detection(traj)
        if peaks:
            found_peaks = True
            for pk in peaks:
                print(f"  L{li}H{hi}: norm peak at step {pk['step']} "
                      f"({pk['progress']:.0%} through training), value={pk['value']:.4f}")
    if not found_peaks:
        print("  no mid-trajectory peaks detected")

    # --- Attention Entropy Trajectories ---
    print("\n--- Attention Entropy (first/last recorded step) ---")
    for (li, hi) in sorted(probe.attention_entropies.keys()):
        entries = probe.attention_entropies[(li, hi)]
        if len(entries) >= 2:
            first_e = entries[0][1]
            last_e = entries[-1][1]
            delta = last_e - first_e
            print(f"  L{li}H{hi}: {first_e:.3f} -> {last_e:.3f} (delta={delta:+.3f})")

    # --- Anomaly Detection ---
    print("\n--- Anomalies (emergent phenomena) ---")
    anomalies = detect_anomalies(probe)
    if anomalies:
        # Group by type
        by_type = {}
        for a in anomalies:
            by_type.setdefault(a['type'], []).append(a)
        for atype, items in sorted(by_type.items()):
            print(f"  {atype} ({len(items)} events):")
            for a in items[:5]:  # show up to 5 per type
                if atype == 'sync_event':
                    print(f"    step {a['step']}: {a['num_heads']} heads {a['direction']}")
                elif atype == 'ghost_spike':
                    print(f"    step {a['step']}: L{a['head'][0]}H{a['head'][1]} "
                          f"norm +{a['norm_change']:.4f}, loss {a['loss_change']:+.4f}")
                elif atype == 'sudden_specialization':
                    print(f"    step {a['step']}: L{a['head'][0]}H{a['head'][1]} "
                          f"entropy {a['entropy_before']:.3f} -> {a['entropy_after']:.3f}")
                elif atype == 'role_reversal':
                    r = a['head_rising']
                    f = a['head_falling']
                    print(f"    step {a['step']}: L{r[0]}H{r[1]} overtakes L{f[0]}H{f[1]}")
                elif atype == 'gradient_divergence':
                    print(f"    step {a['step']}: {a['group']} "
                          f"grad={a['grad_spike']:.4f}, loss_change={a['loss_change']:.4f}")
                elif atype == 'periodicity':
                    print(f"    L{a['head'][0]}H{a['head'][1]}: "
                          f"period={a['period']}, strength={a['strength']:.3f}")
            if len(items) > 5:
                print(f"    ... and {len(items) - 5} more")
    else:
        print("  no anomalies detected")

    # --- Summary ---
    summary = summarize_probe(probe)
    print(f"\n--- Summary ---")
    print(f"  final loss: {summary['final_loss']:.4f}")
    print(f"  min loss: {summary['min_loss']:.4f}")
    print(f"  DG episodes: {summary['dg_count']}, DG index: {summary['dg_index']:.3f}")
    print(f"  steps with trajectory data: {len(probe.step_data)}")

    # --- Visualizations ---
    os.makedirs('results', exist_ok=True)
    print("\n--- Generating Visualizations ---")

    plot_loss_with_phases(loss_vals, phases, 'results/loss_phases.png')
    print("  saved results/loss_phases.png")

    plot_head_norm_heatmap(probe, anomalies, 'results/head_norms.png')
    print("  saved results/head_norms.png")

    plot_entropy_evolution(probe, anomalies, 'results/entropy.png')
    print("  saved results/entropy.png")

    plot_anomaly_timeline(anomalies, len(loss_vals), 'results/anomalies.png')
    print("  saved results/anomalies.png")

    plot_head_contribution_evolution(probe, 'results/contributions.png')
    print("  saved results/contributions.png")

    # --- Generated Names (secondary) ---
    print("\n--- Generated Names (secondary) ---")
    samples = generate(state_dict, config, uchars, BOS,
                       num_samples=20, temperature=0.5, seed=123)
    for i, s in enumerate(samples):
        print(f"  {i+1:2d}: {s}")


def trajectory():
    """
    Run baseline vs damaged variant, compare trajectory shapes,
    report divergence points and rerouting.
    """
    from morphogpt_np import (
        make_config, init_state_dict, load_dataset, train,
        TrainConfig, Hooks, Probe
    )
    from perturbations_np import freeze_random_heads
    from metrics import (
        trajectory_envelope, compare_trajectory_envelopes,
        detect_phases, compute_delayed_gratification,
        per_step_rerouting, rerouting_matrix,
    )
    import random

    print("=== TRAJECTORY COMPARISON: Baseline vs 4 Frozen Heads ===")
    docs, uchars, BOS, vocab_size = load_dataset()
    config = make_config(n_layer=4, n_embd=16, n_head=4, vocab_size=vocab_size)

    num_reps = 3
    num_steps = 300
    num_frozen = 4

    # --- Baseline runs ---
    print(f"\nRunning {num_reps} baseline runs ({num_steps} steps each)...")
    baseline_trajectories = []
    baseline_probes = []
    for rep in range(num_reps):
        sd, params = init_state_dict(config, seed=42 + rep)
        tc = TrainConfig(num_steps=num_steps, print_every=0, detail_level='summary')
        probe = train(sd, params, config, tc, docs, uchars, BOS,
                      probe=Probe(record_interval=5, detail_level='summary'),
                      seed=42 + rep)
        baseline_trajectories.append(probe.get_loss_values())
        baseline_probes.append(probe)
        print(f"  rep {rep+1}: final loss={probe.losses[-1][1]:.4f}")

    # --- Damaged runs ---
    print(f"\nRunning {num_reps} damaged runs ({num_frozen} frozen heads)...")
    damaged_trajectories = []
    damaged_probes = []
    frozen_heads_all = []
    for rep in range(num_reps):
        sd, params = init_state_dict(config, seed=42 + rep)
        hooks = Hooks()
        rng = random.Random(1042 + rep)
        frozen = freeze_random_heads(hooks, config, num_heads=num_frozen, rng=rng)
        frozen_heads_all.append(frozen)
        tc = TrainConfig(num_steps=num_steps, print_every=0, detail_level='summary')
        probe = train(sd, params, config, tc, docs, uchars, BOS,
                      hooks=hooks,
                      probe=Probe(record_interval=5, detail_level='summary'),
                      seed=42 + rep)
        damaged_trajectories.append(probe.get_loss_values())
        damaged_probes.append(probe)
        print(f"  rep {rep+1}: final loss={probe.losses[-1][1]:.4f}, "
              f"frozen={frozen}")

    # --- Trajectory Envelopes ---
    env_base = trajectory_envelope(baseline_trajectories)
    env_dmg = trajectory_envelope(damaged_trajectories)
    comparison = compare_trajectory_envelopes(env_base, env_dmg)

    print("\n--- Trajectory Shape Comparison ---")
    print(f"  mean divergence: {comparison['mean_divergence']:.4f}")
    print(f"  max divergence:  {comparison['max_divergence']:.4f} at step {comparison['divergence_step']}")
    print(f"  shape correlation: {comparison['shape_correlation']:.3f}")
    print(f"  overlap fraction: {comparison['overlap_fraction']:.2f}")

    # --- Phase Comparison ---
    print("\n--- Phase Structure ---")
    print("  Baseline:")
    base_phases = detect_phases(baseline_trajectories[0])
    for p in base_phases:
        print(f"    steps {p['start']:3d}-{p['end']:3d}: {p['type']}")
    print("  Damaged:")
    dmg_phases = detect_phases(damaged_trajectories[0])
    for p in dmg_phases:
        print(f"    steps {p['start']:3d}-{p['end']:3d}: {p['type']}")

    # --- DG Events ---
    print("\n--- DG Episodes ---")
    base_dg = compute_delayed_gratification(baseline_trajectories[0])
    dmg_dg = compute_delayed_gratification(damaged_trajectories[0])
    print(f"  Baseline: {len(base_dg)} episodes")
    for ep in base_dg:
        print(f"    steps {ep['start']}-{ep['end']}, DG={ep['dg_index']:.3f}")
    print(f"  Damaged:  {len(dmg_dg)} episodes")
    for ep in dmg_dg:
        print(f"    steps {ep['start']}-{ep['end']}, DG={ep['dg_index']:.3f}")

    # --- Rerouting ---
    print("\n--- Rerouting Analysis ---")
    for rep in range(num_reps):
        rerouting = per_step_rerouting(damaged_probes[rep])
        if rerouting:
            top3 = rerouting[:3]
            comp_str = ', '.join(
                f"L{r['layer']}H{r['head']}(slope={r['slope']:.4f})"
                for r in top3)
            print(f"  rep {rep+1}: top compensators: {comp_str}")

    # --- Rerouting Matrix (first rep) ---
    if baseline_probes[0].head_outputs and damaged_probes[0].head_outputs:
        matrix = rerouting_matrix(baseline_probes[0], damaged_probes[0],
                                  frozen_heads_all[0])
        print("\n--- Rerouting Matrix (rep 1) ---")
        for fh, comps in matrix.items():
            if comps:
                comp_str = ', '.join(
                    f"L{c['layer']}H{c['head']}(+{c['increase']:.0%})"
                    for c in comps[:3])
                print(f"  frozen L{fh[0]}H{fh[1]} -> compensated by: {comp_str}")


def experiment1(num_reps=3, num_steps=200):
    """Head freezing robustness curve."""
    from experiments_np import experiment_head_freezing, save_results
    os.makedirs('results', exist_ok=True)
    results = experiment_head_freezing(num_reps=num_reps, num_steps=num_steps)
    save_results(results['all_results'], 'results/head_freezing.json')


def experiment2(num_reps=3, num_steps=200):
    """Cell-view GPT."""
    from experiments_np import experiment_cell_view, save_results
    os.makedirs('results', exist_ok=True)
    results = experiment_cell_view(num_reps=num_reps, num_steps=num_steps)
    save_results(results, 'results/cell_view.json')


def experiment3(num_reps=3, num_steps=200):
    """Gradient degradation."""
    from experiments_np import experiment_gradient_degradation, save_results
    os.makedirs('results', exist_ok=True)
    results = experiment_gradient_degradation(num_reps=num_reps, num_steps=num_steps)
    save_results(results, 'results/gradient_degradation.json')


def experiment4(num_reps=3, num_steps=200):
    """Vision radius sweep."""
    from experiments_np import experiment_vision_radius, save_results
    os.makedirs('results', exist_ok=True)
    results = experiment_vision_radius(num_reps=num_reps, num_steps=num_steps)
    save_results(results['all_results'], 'results/vision_radius.json')


def experiment5(num_reps=3, num_steps=200):
    """Communication topology."""
    from experiments_np import experiment_communication_topology, save_results
    os.makedirs('results', exist_ok=True)
    results = experiment_communication_topology(num_reps=num_reps, num_steps=num_steps)
    save_results(results, 'results/communication_topology.json')


def experiment6(num_reps=3, num_steps=200):
    """Courage vs. caution."""
    from experiments_np import experiment_courage_caution, save_results
    os.makedirs('results', exist_ok=True)
    results = experiment_courage_caution(num_reps=num_reps, num_steps=num_steps)
    save_results(results, 'results/courage_caution.json')


def analyze1():
    """Analyze experiment 1 results from results/head_freezing.json."""
    import json
    from metrics import (
        dg_index, dg_damage_regression, trajectory_envelope,
        compare_trajectory_envelopes, compute_delayed_gratification,
    )
    from visualize import (
        plot_robustness_curve, plot_trajectory_overlay,
        plot_head_compensation, plot_dg_episode_structure,
        plot_trajectory_divergence,
    )

    path = 'results/head_freezing.json'
    print(f"Loading {path}...")
    with open(path) as f:
        runs = json.load(f)
    print(f"Loaded {len(runs)} runs")

    # ── Organize by damage level ──────────────────────────────────────
    runs_by_level = {}
    for r in runs:
        if r['perturbation_type'] == 'none':
            level = 0
        else:
            level = r['perturbation_params']['num_heads']
        runs_by_level.setdefault(level, []).append(r)

    levels = sorted(runs_by_level.keys())

    # ── Table 1: Loss robustness ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("TABLE 1: Loss Robustness")
    print("=" * 60)
    print(f"{'Frozen':>8} {'Mean Loss':>10} {'Std':>8} {'N':>4}")
    print("-" * 34)

    curve = []
    for level in levels:
        group = runs_by_level[level]
        losses = [r['summary']['mean_loss'] for r in group]
        n = len(losses)
        mean_l = sum(losses) / n
        std_l = (sum((x - mean_l) ** 2 for x in losses) / max(1, n - 1)) ** 0.5

        # DG indices from trajectories
        dg_vals = [dg_index(r['loss_trajectory']) for r in group]
        mean_dg = sum(dg_vals) / n
        std_dg = (sum((x - mean_dg) ** 2 for x in dg_vals) / max(1, n - 1)) ** 0.5

        curve.append({
            'damage_level': level,
            'mean_loss': mean_l,
            'std_loss': std_l,
            'mean_dg': mean_dg,
            'std_dg': std_dg,
            'n': n,
        })
        print(f"{level:>8d} {mean_l:>10.4f} {std_l:>8.4f} {n:>4d}")

    # ── Table 2: DG index ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TABLE 2: DG Index vs Damage")
    print("=" * 60)
    print(f"{'Frozen':>8} {'Mean DG':>8} {'Std DG':>8}")
    print("-" * 28)
    for c in curve:
        print(f"{c['damage_level']:>8d} {c['mean_dg']:>8.3f} {c['std_dg']:>8.3f}")

    # DG-damage regression
    damage_flat = []
    dg_flat = []
    for level in levels:
        for r in runs_by_level[level]:
            damage_flat.append(level)
            dg_flat.append(dg_index(r['loss_trajectory']))

    slope, intercept, r_sq = dg_damage_regression(damage_flat, dg_flat)
    dg_reg = {'slope': slope, 'intercept': intercept, 'r_squared': r_sq}
    print(f"\nRegression: slope={slope:.4f}, intercept={intercept:.4f}, "
          f"R\u00b2={r_sq:.3f}")
    if slope > 0 and r_sq > 0.1:
        print("  -> DG increases with damage (evidence for rerouting)")

    # ── Trajectory envelopes + shape comparison ───────────────────────
    trajectories_by_level = {}
    envelopes = {}
    for level in levels:
        trajs = [r['loss_trajectory'] for r in runs_by_level[level]]
        trajectories_by_level[level] = trajs
        envelopes[level] = trajectory_envelope(trajs)

    shape_comparisons = {}
    if 0 in envelopes:
        for level in levels:
            if level == 0:
                continue
            shape_comparisons[level] = compare_trajectory_envelopes(
                envelopes[0], envelopes[level])

    # ── Table 3: Trajectory shape ─────────────────────────────────────
    if shape_comparisons:
        print("\n" + "=" * 60)
        print("TABLE 3: Trajectory Shape vs Baseline")
        print("=" * 60)
        print(f"{'Frozen':>8} {'Divergence':>11} {'Correlation':>12} {'Overlap':>9}")
        print("-" * 44)
        for level in sorted(shape_comparisons.keys()):
            sc = shape_comparisons[level]
            print(f"{level:>8d} {sc['mean_divergence']:>11.4f} "
                  f"{sc['shape_correlation']:>12.3f} {sc['overlap_fraction']:>9.2f}")

    # ── Table 4: Head compensation ────────────────────────────────────
    print("\n" + "=" * 60)
    print("TABLE 4: Head Contribution Evolution")
    print("=" * 60)

    head_comp_data = {}  # level -> {head_name: (start, end)}
    for level in levels:
        if level == 0:
            continue
        group = runs_by_level[level]
        # Use first run's head_norms for contribution analysis
        r = group[0]
        if 'head_norms' not in r:
            continue

        head_norms = r['head_norms']
        # Compute total norm at each step, then per-head fraction
        step_totals = {}
        for hname, entries in head_norms.items():
            for e in entries:
                s = e['step']
                step_totals[s] = step_totals.get(s, 0.0) + e['norm']

        head_fracs = {}
        for hname, entries in head_norms.items():
            fracs = []
            for e in entries:
                s = e['step']
                total = step_totals.get(s, 1.0)
                fracs.append(e['norm'] / total if total > 1e-10 else 0.0)
            if len(fracs) >= 2:
                # Average first 20% and last 20%
                n = len(fracs)
                chunk = max(1, n // 5)
                start_frac = sum(fracs[:chunk]) / chunk
                end_frac = sum(fracs[-chunk:]) / chunk
                head_fracs[hname] = (start_frac, end_frac)

        if head_fracs:
            head_comp_data[level] = head_fracs

            # Print top compensators (biggest increase)
            ranked = sorted(head_fracs.items(),
                            key=lambda x: x[1][1] - x[1][0], reverse=True)
            top = ranked[:4]
            parts = [f"{h}: {s:.3f}->{e:.3f}" for h, (s, e) in top]
            print(f"  {level:2d} frozen: {', '.join(parts)}")

    # ── Table 5: DG episode structure ─────────────────────────────────
    print("\n" + "=" * 60)
    print("TABLE 5: DG Episode Structure")
    print("=" * 60)
    print(f"{'Frozen':>8} {'Episodes':>9} {'Mean Dur':>9} {'Mean DG':>8}")
    print("-" * 38)

    episode_stats = []
    for level in levels:
        group = runs_by_level[level]
        all_eps = []
        for r in group:
            eps = r.get('dg_episodes', [])
            if not eps:
                eps = compute_delayed_gratification(r['loss_trajectory'])
            all_eps.extend(eps)

        total = len(all_eps)
        if total > 0:
            mean_dur = sum(ep['end'] - ep['start'] for ep in all_eps) / total
            mean_dgi = sum(ep['dg_index'] for ep in all_eps) / total
        else:
            mean_dur = 0.0
            mean_dgi = 0.0

        episode_stats.append({
            'level': level,
            'num_episodes': total,
            'mean_duration': mean_dur,
            'mean_dg_index': mean_dgi,
        })
        print(f"{level:>8d} {total:>9d} {mean_dur:>9.1f} {mean_dgi:>8.3f}")

    # ── Table 6: Training speed ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("TABLE 6: Training Speed")
    print("=" * 60)
    for level in levels:
        group = runs_by_level[level]
        times = [r['elapsed'] for r in group]
        mean_t = sum(times) / len(times)
        print(f"  {level:2d} frozen: ~{mean_t:.0f}s/run")

    # ── Generate plots ────────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    print("\n--- Generating Plots ---")

    plot_robustness_curve(curve, dg_reg, 'results/exp1_robustness_curve.png')
    print("  saved results/exp1_robustness_curve.png")

    plot_trajectory_overlay(trajectories_by_level, 'results/exp1_trajectories.png')
    print("  saved results/exp1_trajectories.png")

    if head_comp_data:
        # Pick representative levels for the compensation plot
        comp_levels = [l for l in [4, 8, 12] if l in head_comp_data]
        if comp_levels:
            plot_head_compensation(
                {l: head_comp_data[l] for l in comp_levels},
                'results/exp1_head_compensation.png')
            print("  saved results/exp1_head_compensation.png")

    plot_dg_episode_structure(episode_stats, 'results/exp1_dg_episodes.png')
    print("  saved results/exp1_dg_episodes.png")

    if shape_comparisons:
        plot_trajectory_divergence(shape_comparisons,
                                   'results/exp1_trajectory_divergence.png')
        print("  saved results/exp1_trajectory_divergence.png")

    # ── Interpretation ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    loss_range = max(c['mean_loss'] for c in curve) - min(c['mean_loss'] for c in curve)
    baseline_loss = curve[0]['mean_loss']
    max_damage_loss = curve[-1]['mean_loss']

    print(f"\n1. Loss robustness: range {loss_range:.4f} across all damage levels")
    if max_damage_loss <= baseline_loss:
        print("   Loss DECREASES with more freezing — MLPs carry the learning")
    elif loss_range < 0.1:
        print("   Loss essentially flat — strong morphogenetic robustness")
    else:
        print(f"   Loss degrades by {loss_range:.4f} — moderate robustness")

    print(f"\n2. DG-damage trend: slope={slope:.4f}, R\u00b2={r_sq:.3f}")
    if slope > 0 and r_sq > 0.1:
        print("   Positive slope — more damage produces more delayed gratification")
        print("   This is the core Levin signature for rerouting")
    else:
        print("   No clear DG-damage relationship")

    if shape_comparisons:
        min_corr = min(sc['shape_correlation']
                       for sc in shape_comparisons.values())
        print(f"\n3. Trajectory shape: min correlation = {min_corr:.3f}")
        if min_corr > 0.98:
            print("   Shape preserved >0.98 — strong target attractor")
        elif min_corr > 0.95:
            print("   Shape preserved >0.95 — moderate attractor")
        else:
            print("   Shape degrades significantly under damage")

    print()


def analyze2():
    """Analyze experiment 2 results from results/cell_view.json."""
    import json
    from metrics import (
        dg_index, trajectory_envelope, compare_trajectory_envelopes,
        compute_delayed_gratification,
    )
    from visualize import (
        plot_group_comparison, plot_trajectory_overlay,
        plot_entropy_comparison, plot_dg_episode_structure,
    )

    path = 'results/cell_view.json'
    print(f"Loading {path}...")
    with open(path) as f:
        runs = json.load(f)
    print(f"Loaded {len(runs)} runs")

    # ── Organize by group name ───────────────────────────────────────
    runs_by_group = {}
    for r in runs:
        group = r['name']
        runs_by_group.setdefault(group, []).append(r)

    groups = sorted(runs_by_group.keys())
    print(f"Groups: {groups}")

    # ── Table 1: Loss comparison ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("TABLE 1: Loss Comparison")
    print("=" * 60)
    print(f"{'Group':>15} {'Mean Loss':>10} {'Std':>8} {'DG Index':>9} {'N':>4}")
    print("-" * 50)

    group_stats = []
    for group in groups:
        grp = runs_by_group[group]
        losses = [r['summary']['mean_loss'] for r in grp]
        n = len(losses)
        mean_l = sum(losses) / n
        std_l = (sum((x - mean_l) ** 2 for x in losses) / max(1, n - 1)) ** 0.5

        dg_vals = [dg_index(r['loss_trajectory']) for r in grp]
        mean_dg = sum(dg_vals) / n
        std_dg = (sum((x - mean_dg) ** 2 for x in dg_vals) / max(1, n - 1)) ** 0.5

        group_stats.append({
            'name': group,
            'mean_loss': mean_l,
            'std_loss': std_l,
            'mean_dg': mean_dg,
            'std_dg': std_dg,
            'n': n,
        })
        print(f"{group:>15} {mean_l:>10.4f} {std_l:>8.4f} {mean_dg:>9.3f} {n:>4d}")

    # ── Trajectory envelopes ─────────────────────────────────────────
    trajectories_by_group = {}
    envelopes = {}
    for group in groups:
        trajs = [r['loss_trajectory'] for r in runs_by_group[group]]
        trajectories_by_group[group] = trajs
        envelopes[group] = trajectory_envelope(trajs)

    # ── Table 2: Trajectory shape ────────────────────────────────────
    shape_comparisons = {}
    baseline_key = 'baseline' if 'baseline' in envelopes else groups[0]
    for group in groups:
        if group == baseline_key:
            continue
        shape_comparisons[group] = compare_trajectory_envelopes(
            envelopes[baseline_key], envelopes[group])

    if shape_comparisons:
        print("\n" + "=" * 60)
        print(f"TABLE 2: Trajectory Shape vs {baseline_key}")
        print("=" * 60)
        print(f"{'Group':>15} {'Divergence':>11} {'Correlation':>12} {'Overlap':>9}")
        print("-" * 51)
        for group in sorted(shape_comparisons.keys()):
            sc = shape_comparisons[group]
            print(f"{group:>15} {sc['mean_divergence']:>11.4f} "
                  f"{sc['shape_correlation']:>12.3f} {sc['overlap_fraction']:>9.2f}")

    # ── Table 3: Head contribution evolution ─────────────────────────
    print("\n" + "=" * 60)
    print("TABLE 3: Head Contribution Evolution")
    print("=" * 60)

    for group in groups:
        grp = runs_by_group[group]
        r = grp[0]  # use first run
        head_norms = r.get('head_norms', {})
        if not head_norms:
            continue

        step_totals = {}
        for hname, entries in head_norms.items():
            for e in entries:
                s = e['step']
                step_totals[s] = step_totals.get(s, 0.0) + e['norm']

        head_fracs = {}
        for hname, entries in head_norms.items():
            fracs = []
            for e in entries:
                s = e['step']
                total = step_totals.get(s, 1.0)
                fracs.append(e['norm'] / total if total > 1e-10 else 0.0)
            if len(fracs) >= 2:
                n = len(fracs)
                chunk = max(1, n // 5)
                start_frac = sum(fracs[:chunk]) / chunk
                end_frac = sum(fracs[-chunk:]) / chunk
                head_fracs[hname] = (start_frac, end_frac)

        if head_fracs:
            ranked = sorted(head_fracs.items(),
                            key=lambda x: abs(x[1][1] - x[1][0]), reverse=True)
            print(f"  {group}:")
            for h, (s, e) in ranked[:6]:
                delta = e - s
                print(f"    {h}: {s:.3f} -> {e:.3f} (delta={delta:+.3f})")

    # ── Table 4: DG episode structure ────────────────────────────────
    print("\n" + "=" * 60)
    print("TABLE 4: DG Episode Structure")
    print("=" * 60)
    print(f"{'Group':>15} {'Episodes':>9} {'Mean Dur':>9} {'Mean DG':>8}")
    print("-" * 45)

    episode_stats = []
    for group in groups:
        grp = runs_by_group[group]
        all_eps = []
        for r in grp:
            eps = r.get('dg_episodes', [])
            if not eps:
                eps = compute_delayed_gratification(r['loss_trajectory'])
            all_eps.extend(eps)

        total = len(all_eps)
        if total > 0:
            mean_dur = sum(ep['end'] - ep['start'] for ep in all_eps) / total
            mean_dgi = sum(ep['dg_index'] for ep in all_eps) / total
        else:
            mean_dur = 0.0
            mean_dgi = 0.0

        episode_stats.append({
            'level': group,
            'num_episodes': total,
            'mean_duration': mean_dur,
            'mean_dg_index': mean_dgi,
        })
        print(f"{group:>15} {total:>9d} {mean_dur:>9.1f} {mean_dgi:>8.3f}")

    # ── Table 5: Training speed ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("TABLE 5: Training Speed")
    print("=" * 60)
    for group in groups:
        grp = runs_by_group[group]
        times = [r['elapsed'] for r in grp]
        mean_t = sum(times) / len(times)
        print(f"  {group}: ~{mean_t:.1f}s/run")

    # ── Table 6: Attention entropy ───────────────────────────────────
    print("\n" + "=" * 60)
    print("TABLE 6: Attention Entropy by Head")
    print("=" * 60)

    entropy_by_group = {}
    for group in groups:
        grp = runs_by_group[group]
        head_entropies_agg = {}
        for r in grp:
            he = r.get('head_entropies', {})
            for hname, entries in he.items():
                vals = [e['entropy'] for e in entries]
                mean_e = sum(vals) / len(vals) if vals else 0.0
                head_entropies_agg.setdefault(hname, []).append(mean_e)

        group_means = {}
        for hname, vals in head_entropies_agg.items():
            group_means[hname] = sum(vals) / len(vals)
        entropy_by_group[group] = group_means

        heads_sorted = sorted(group_means.keys())
        parts = [f"{h}={group_means[h]:.3f}" for h in heads_sorted]
        print(f"  {group}: {', '.join(parts)}")

    # ── Generate plots ───────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    print("\n--- Generating Plots ---")

    plot_group_comparison(group_stats, 'results/exp2_comparison.png')
    print("  saved results/exp2_comparison.png")

    plot_trajectory_overlay(trajectories_by_group, 'results/exp2_trajectories.png')
    print("  saved results/exp2_trajectories.png")

    plot_entropy_comparison(entropy_by_group, 'results/exp2_head_entropy.png')
    print("  saved results/exp2_head_entropy.png")

    plot_dg_episode_structure(episode_stats, 'results/exp2_dg_episodes.png')
    print("  saved results/exp2_dg_episodes.png")

    # ── Interpretation ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if len(group_stats) >= 2:
        baseline_s = next((g for g in group_stats if g['name'] == 'baseline'), group_stats[0])
        cell_s = next((g for g in group_stats if g['name'] == 'cell_view'), group_stats[-1])

        loss_diff = cell_s['mean_loss'] - baseline_s['mean_loss']
        print(f"\n1. Performance: cell_view loss {loss_diff:+.4f} vs baseline")
        if abs(loss_diff) < 0.05:
            print("   Cell-view preserves performance — local learning suffices")
        elif loss_diff > 0:
            print("   Cell-view degrades performance — global backprop matters")
        else:
            print("   Cell-view improves performance — local learning helps")

        dg_diff = cell_s['mean_dg'] - baseline_s['mean_dg']
        print(f"\n2. DG pattern: cell_view DG {dg_diff:+.3f} vs baseline")
        if abs(dg_diff) < 0.01:
            print("   Similar DG patterns — rerouting behavior preserved")
        elif dg_diff > 0:
            print("   More DG under cell-view — local learning creates more rerouting")
        else:
            print("   Less DG under cell-view — local learning is smoother")

    if shape_comparisons:
        for group, sc in shape_comparisons.items():
            print(f"\n3. Trajectory shape ({group} vs baseline):")
            print(f"   Correlation: {sc['shape_correlation']:.3f}, "
                  f"Overlap: {sc['overlap_fraction']:.2f}")
            if sc['shape_correlation'] > 0.95:
                print("   Trajectories highly similar — same attractor")
            else:
                print("   Trajectories diverge — different learning dynamics")

    # Entropy comparison
    if len(entropy_by_group) >= 2:
        baseline_ent = entropy_by_group.get('baseline', {})
        cell_ent = entropy_by_group.get('cell_view', {})
        if baseline_ent and cell_ent:
            common_heads = set(baseline_ent.keys()) & set(cell_ent.keys())
            if common_heads:
                diffs = [cell_ent[h] - baseline_ent[h] for h in common_heads]
                mean_diff = sum(diffs) / len(diffs)
                print(f"\n4. Entropy: cell_view mean entropy delta {mean_diff:+.3f}")
                if mean_diff > 0.05:
                    print("   Cell-view heads have higher entropy — less specialized")
                elif mean_diff < -0.05:
                    print("   Cell-view heads have lower entropy — more specialized locally")
                else:
                    print("   Similar entropy patterns")

    print()


def analyze3():
    """Analyze experiment 3 results from results/gradient_degradation.json."""
    import json
    from metrics import (
        dg_index, trajectory_envelope, compare_trajectory_envelopes,
        compute_delayed_gratification,
    )
    from visualize import (
        plot_group_comparison, plot_trajectory_overlay,
        plot_dg_episode_structure, plot_trajectory_divergence,
    )

    path = 'results/gradient_degradation.json'
    print(f"Loading {path}...")
    with open(path) as f:
        runs = json.load(f)
    print(f"Loaded {len(runs)} runs")

    # ── Organize by method name ──────────────────────────────────────
    runs_by_method = {}
    for r in runs:
        method = r['name']
        runs_by_method.setdefault(method, []).append(r)

    methods = sorted(runs_by_method.keys())
    print(f"Methods: {methods}")

    # ── Table 1: Loss by method ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("TABLE 1: Loss by Method")
    print("=" * 60)
    print(f"{'Method':>20} {'Mean Loss':>10} {'Std':>8} {'N':>4}")
    print("-" * 46)

    method_stats = []
    for method in methods:
        grp = runs_by_method[method]
        losses = [r['summary']['mean_loss'] for r in grp]
        n = len(losses)
        mean_l = sum(losses) / n
        std_l = (sum((x - mean_l) ** 2 for x in losses) / max(1, n - 1)) ** 0.5

        dg_vals = [dg_index(r['loss_trajectory']) for r in grp]
        mean_dg = sum(dg_vals) / n
        std_dg = (sum((x - mean_dg) ** 2 for x in dg_vals) / max(1, n - 1)) ** 0.5

        method_stats.append({
            'name': method,
            'mean_loss': mean_l,
            'std_loss': std_l,
            'mean_dg': mean_dg,
            'std_dg': std_dg,
            'n': n,
        })
        print(f"{method:>20} {mean_l:>10.4f} {std_l:>8.4f} {n:>4d}")

    # ── Table 2: DG by method ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TABLE 2: DG Index by Method")
    print("=" * 60)
    print(f"{'Method':>20} {'Mean DG':>8} {'Std DG':>8}")
    print("-" * 40)
    for ms in method_stats:
        print(f"{ms['name']:>20} {ms['mean_dg']:>8.3f} {ms['std_dg']:>8.3f}")

    # ── Trajectory envelopes ─────────────────────────────────────────
    trajectories_by_method = {}
    envelopes = {}
    for method in methods:
        trajs = [r['loss_trajectory'] for r in runs_by_method[method]]
        trajectories_by_method[method] = trajs
        envelopes[method] = trajectory_envelope(trajs)

    # ── Table 3: Trajectory shape vs baseline ────────────────────────
    shape_comparisons = {}
    baseline_key = 'baseline' if 'baseline' in envelopes else methods[0]
    for method in methods:
        if method == baseline_key:
            continue
        shape_comparisons[method] = compare_trajectory_envelopes(
            envelopes[baseline_key], envelopes[method])

    if shape_comparisons:
        print("\n" + "=" * 60)
        print(f"TABLE 3: Trajectory Shape vs {baseline_key}")
        print("=" * 60)
        print(f"{'Method':>20} {'Divergence':>11} {'Correlation':>12} {'Overlap':>9}")
        print("-" * 56)
        for method in sorted(shape_comparisons.keys()):
            sc = shape_comparisons[method]
            print(f"{method:>20} {sc['mean_divergence']:>11.4f} "
                  f"{sc['shape_correlation']:>12.3f} {sc['overlap_fraction']:>9.2f}")

    # ── Table 4: DG episode structure ────────────────────────────────
    print("\n" + "=" * 60)
    print("TABLE 4: DG Episode Structure")
    print("=" * 60)
    print(f"{'Method':>20} {'Episodes':>9} {'Mean Dur':>9} {'Mean DG':>8}")
    print("-" * 50)

    episode_stats = []
    for method in methods:
        grp = runs_by_method[method]
        all_eps = []
        for r in grp:
            eps = r.get('dg_episodes', [])
            if not eps:
                eps = compute_delayed_gratification(r['loss_trajectory'])
            all_eps.extend(eps)

        total = len(all_eps)
        if total > 0:
            mean_dur = sum(ep['end'] - ep['start'] for ep in all_eps) / total
            mean_dgi = sum(ep['dg_index'] for ep in all_eps) / total
        else:
            mean_dur = 0.0
            mean_dgi = 0.0

        episode_stats.append({
            'level': method,
            'num_episodes': total,
            'mean_duration': mean_dur,
            'mean_dg_index': mean_dgi,
        })
        print(f"{method:>20} {total:>9d} {mean_dur:>9.1f} {mean_dgi:>8.3f}")

    # ── Table 5: Training speed ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("TABLE 5: Training Speed")
    print("=" * 60)
    for method in methods:
        grp = runs_by_method[method]
        times = [r['elapsed'] for r in grp]
        mean_t = sum(times) / len(times)
        print(f"  {method}: ~{mean_t:.1f}s/run")

    # ── Generate plots ───────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    print("\n--- Generating Plots ---")

    plot_group_comparison(method_stats, 'results/exp3_method_comparison.png')
    print("  saved results/exp3_method_comparison.png")

    plot_trajectory_overlay(trajectories_by_method, 'results/exp3_trajectories.png')
    print("  saved results/exp3_trajectories.png")

    plot_dg_episode_structure(episode_stats, 'results/exp3_dg_episodes.png')
    print("  saved results/exp3_dg_episodes.png")

    if shape_comparisons:
        plot_trajectory_divergence(shape_comparisons,
                                   'results/exp3_trajectory_divergence.png')
        print("  saved results/exp3_trajectory_divergence.png")

    # ── Interpretation ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    baseline_s = next((m for m in method_stats if m['name'] == 'baseline'), method_stats[0])
    print(f"\nBaseline loss: {baseline_s['mean_loss']:.4f}")

    # Rank methods by loss
    ranked = sorted(method_stats, key=lambda m: m['mean_loss'])
    print("\n1. Loss ranking (best to worst):")
    for i, ms in enumerate(ranked):
        delta = ms['mean_loss'] - baseline_s['mean_loss']
        print(f"   {i+1}. {ms['name']}: {ms['mean_loss']:.4f} ({delta:+.4f} vs baseline)")

    # Convergence check
    print("\n2. Convergence analysis:")
    for ms in method_stats:
        if ms['name'] == 'baseline':
            continue
        delta = ms['mean_loss'] - baseline_s['mean_loss']
        if abs(delta) < 0.05:
            print(f"   {ms['name']}: converges normally (delta={delta:+.4f})")
        elif delta > 0.2:
            print(f"   {ms['name']}: significantly degraded (delta={delta:+.4f})")
        elif delta > 0:
            print(f"   {ms['name']}: mildly degraded (delta={delta:+.4f})")
        else:
            print(f"   {ms['name']}: improves on baseline (delta={delta:+.4f})")

    # DG-quality relationship
    print("\n3. Gradient quality vs DG:")
    for ms in method_stats:
        if ms['name'] == 'baseline':
            continue
        dg_diff = ms['mean_dg'] - baseline_s['mean_dg']
        print(f"   {ms['name']}: DG {dg_diff:+.3f} vs baseline")

    # Sign-only special case
    sign_ms = next((m for m in method_stats if m['name'] == 'sign_only'), None)
    if sign_ms:
        delta = sign_ms['mean_loss'] - baseline_s['mean_loss']
        print(f"\n4. Sign-only gradient (extreme information loss):")
        print(f"   Loss delta: {delta:+.4f}")
        if abs(delta) < 0.1:
            print("   Still converges — direction matters more than magnitude")
        else:
            print("   Convergence affected — gradient magnitude carries information")

    if shape_comparisons:
        min_corr = min(sc['shape_correlation'] for sc in shape_comparisons.values())
        max_div = max(sc['mean_divergence'] for sc in shape_comparisons.values())
        print(f"\n5. Trajectory preservation: min correlation={min_corr:.3f}, "
              f"max divergence={max_div:.4f}")

    print()


def analyze4():
    """Analyze experiment 4 results from results/vision_radius.json."""
    import json
    from metrics import (
        dg_index, trajectory_envelope, compare_trajectory_envelopes,
        compute_delayed_gratification, collective_light_cone,
        goal_alignment_score, swarming_index,
    )

    path = 'results/vision_radius.json'
    if not os.path.exists(path):
        print(f"No results found at {path}. Run experiment4 first.")
        return

    with open(path) as f:
        data = json.load(f)

    print("=" * 60)
    print("ANALYSIS: Experiment 4 — Vision Radius Sweep")
    print("=" * 60)

    # Group by condition
    by_name = {}
    for entry in data:
        by_name.setdefault(entry['name'], []).append(entry)

    # Summary table
    print(f"\n{'Condition':>12} {'Mean Loss':>10} {'Std':>8} {'Mean DG':>8} {'N':>4}")
    print("-" * 46)
    for name in sorted(by_name.keys()):
        entries = by_name[name]
        losses = [e['summary']['final_loss'] for e in entries]
        trajectories = [e.get('loss_trajectory', []) for e in entries]
        dgs = [dg_index(t) for t in trajectories if t]
        n = len(losses)
        mean_l = sum(losses) / n
        std_l = (sum((x - mean_l) ** 2 for x in losses) / max(1, n - 1)) ** 0.5
        mean_dg = sum(dgs) / len(dgs) if dgs else 0.0
        print(f"{name:>12} {mean_l:>10.4f} {std_l:>8.4f} {mean_dg:>8.3f} {n:>4d}")

    # Trajectory comparison
    envelopes = {}
    for name, entries in by_name.items():
        trajectories = [e.get('loss_trajectory', []) for e in entries]
        trajectories = [t for t in trajectories if t]
        if trajectories:
            envelopes[name] = trajectory_envelope(trajectories)

    if 'baseline' in envelopes:
        print("\n--- Trajectory Shape vs Baseline ---")
        for name in sorted(envelopes.keys()):
            if name == 'baseline':
                continue
            comp = compare_trajectory_envelopes(envelopes['baseline'], envelopes[name])
            print(f"  {name:>12}: corr={comp['shape_correlation']:.3f}, "
                  f"divergence={comp['mean_divergence']:.4f}")

    # DG episodes
    print("\n--- DG Episode Summary ---")
    for name in sorted(by_name.keys()):
        trajectories = [e.get('loss_trajectory', []) for e in by_name[name]]
        all_eps = [compute_delayed_gratification(t) for t in trajectories if t]
        total = sum(len(eps) for eps in all_eps)
        print(f"  {name:>12}: {total} episodes across {len(all_eps)} runs")

    # Interpretation
    print("\n--- Chess Paper Parallel ---")
    print("  R0-R7 vision radius in chess (optimal at R4) ↔ window size sweep")
    print("  Hypothesis: intermediate windows outperform both extremes")

    # Check hypothesis
    losses_by_name = {}
    for name, entries in by_name.items():
        losses_by_name[name] = sum(e['summary']['final_loss'] for e in entries) / len(entries)

    baseline_loss = losses_by_name.get('baseline', float('inf'))
    best_name = min(losses_by_name, key=losses_by_name.get)
    best_loss = losses_by_name[best_name]

    if best_name != 'baseline' and best_name not in ('window_16',):
        print(f"\n  CONFIRMED: {best_name} (loss={best_loss:.4f}) beats "
              f"baseline (loss={baseline_loss:.4f})")
        print("  -> Intermediate vision radius is optimal, paralleling R4 > R7")
    else:
        print(f"\n  Best condition: {best_name} (loss={best_loss:.4f})")
        if best_name in ('baseline', 'window_16'):
            print("  -> Full context is still best (hypothesis not confirmed at this scale)")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss vs window size
        ax = axes[0]
        ws_labels = []
        ws_losses = []
        ws_dgs = []
        for name in ['baseline'] + [f'window_{ws}' for ws in [1, 2, 4, 8, 16]]:
            if name not in by_name:
                continue
            entries = by_name[name]
            label = name.replace('window_', 'W') if name != 'baseline' else 'Full'
            ws_labels.append(label)
            ws_losses.append(sum(e['summary']['final_loss'] for e in entries) / len(entries))
            trajs = [e.get('loss_trajectory', []) for e in entries]
            ws_dgs.append(sum(dg_index(t) for t in trajs if t) / max(1, len(trajs)))

        ax.bar(range(len(ws_labels)), ws_losses, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(ws_labels)))
        ax.set_xticklabels(ws_labels)
        ax.set_ylabel('Final Loss')
        ax.set_title('Vision Radius: Final Loss by Window Size')

        # DG by window size
        ax2 = axes[1]
        ax2.bar(range(len(ws_labels)), ws_dgs, color='coral', alpha=0.7)
        ax2.set_xticks(range(len(ws_labels)))
        ax2.set_xticklabels(ws_labels)
        ax2.set_ylabel('DG Index')
        ax2.set_title('Vision Radius: DG Index by Window Size')

        plt.tight_layout()
        plt.savefig('results/exp4_vision_radius.png', dpi=150)
        print(f"\nPlot saved to results/exp4_vision_radius.png")
        plt.close()
    except ImportError:
        print("\nmatplotlib not available, skipping plots")

    print()


def analyze5():
    """Analyze experiment 5 results from results/communication_topology.json."""
    import json
    from metrics import (
        dg_index, trajectory_envelope, compare_trajectory_envelopes,
        compute_delayed_gratification, goal_alignment_score,
    )

    path = 'results/communication_topology.json'
    if not os.path.exists(path):
        print(f"No results found at {path}. Run experiment5 first.")
        return

    with open(path) as f:
        data = json.load(f)

    print("=" * 60)
    print("ANALYSIS: Experiment 5 — Communication Topology")
    print("=" * 60)

    by_name = {}
    for entry in data:
        by_name.setdefault(entry['name'], []).append(entry)

    topology_order = ['full', 'heavy', 'half', 'light', 'cell_view']

    print(f"\n{'Topology':>12} {'Mean Loss':>10} {'Std':>8} {'Mean DG':>8} {'N':>4}")
    print("-" * 46)
    for name in topology_order:
        entries = by_name.get(name, [])
        if not entries:
            continue
        losses = [e['summary']['final_loss'] for e in entries]
        trajectories = [e.get('loss_trajectory', []) for e in entries]
        dgs = [dg_index(t) for t in trajectories if t]
        n = len(losses)
        mean_l = sum(losses) / n
        std_l = (sum((x - mean_l) ** 2 for x in losses) / max(1, n - 1)) ** 0.5
        mean_dg = sum(dgs) / len(dgs) if dgs else 0.0
        print(f"{name:>12} {mean_l:>10.4f} {std_l:>8.4f} {mean_dg:>8.3f} {n:>4d}")

    # Interpretation
    print("\n--- Chess Paper Parallel ---")
    print("  Relay chains expand cognitive light cone ↔ partial gradient flow")
    print("  Hypothesis: intermediate flow outperforms both full and zero")

    losses_by_name = {}
    for name, entries in by_name.items():
        losses_by_name[name] = sum(e['summary']['final_loss'] for e in entries) / len(entries)

    best_name = min(losses_by_name, key=losses_by_name.get)
    best_loss = losses_by_name[best_name]
    full_loss = losses_by_name.get('full', float('inf'))

    if best_name not in ('full', 'cell_view'):
        print(f"\n  CONFIRMED: {best_name} (loss={best_loss:.4f}) beats "
              f"both full ({full_loss:.4f}) and cell_view")
        print("  -> Intermediate gradient flow is optimal, paralleling R4 > R0 and R4 > R7")
    else:
        print(f"\n  Best condition: {best_name} (loss={best_loss:.4f})")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        labels = []
        losses = []
        dgs = []
        fractions = []
        for name in topology_order:
            entries = by_name.get(name, [])
            if not entries:
                continue
            labels.append(name)
            losses.append(sum(e['summary']['final_loss'] for e in entries) / len(entries))
            trajs = [e.get('loss_trajectory', []) for e in entries]
            dgs.append(sum(dg_index(t) for t in trajs if t) / max(1, len(trajs)))
            frac_map = {'full': 1.0, 'heavy': 0.75, 'half': 0.5, 'light': 0.25, 'cell_view': 0.0}
            fractions.append(frac_map.get(name, 0.5))

        ax = axes[0]
        ax.plot(fractions, losses, 'o-', color='steelblue', markersize=8)
        for i, name in enumerate(labels):
            ax.annotate(name, (fractions[i], losses[i]), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=8)
        ax.set_xlabel('Gradient Pass Fraction')
        ax.set_ylabel('Final Loss')
        ax.set_title('Communication Topology: Loss vs Gradient Flow')
        ax.invert_xaxis()

        ax2 = axes[1]
        ax2.plot(fractions, dgs, 'o-', color='coral', markersize=8)
        for i, name in enumerate(labels):
            ax2.annotate(name, (fractions[i], dgs[i]), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=8)
        ax2.set_xlabel('Gradient Pass Fraction')
        ax2.set_ylabel('DG Index')
        ax2.set_title('Communication Topology: DG vs Gradient Flow')
        ax2.invert_xaxis()

        plt.tight_layout()
        plt.savefig('results/exp5_communication_topology.png', dpi=150)
        print(f"\nPlot saved to results/exp5_communication_topology.png")
        plt.close()
    except ImportError:
        print("\nmatplotlib not available, skipping plots")

    print()


def analyze6():
    """Analyze experiment 6 results from results/courage_caution.json."""
    import json
    from metrics import (
        dg_index, trajectory_envelope, compare_trajectory_envelopes,
        compute_delayed_gratification, swarming_index,
    )

    path = 'results/courage_caution.json'
    if not os.path.exists(path):
        print(f"No results found at {path}. Run experiment6 first.")
        return

    with open(path) as f:
        data = json.load(f)

    print("=" * 60)
    print("ANALYSIS: Experiment 6 — Courage vs. Caution")
    print("=" * 60)

    by_name = {}
    for entry in data:
        by_name.setdefault(entry['name'], []).append(entry)

    condition_labels = {
        'baseline': 'Baseline',
        'cautious_cautious': '(a) Cautious/Cautious',
        'cautious_courageous': '(b) Cautious/Courageous *',
        'courageous_cautious': '(c) Courageous/Cautious',
        'courageous_courageous': '(d) Courageous/Courageous',
    }
    condition_order = ['baseline', 'cautious_cautious', 'cautious_courageous',
                       'courageous_cautious', 'courageous_courageous']

    print(f"\n{'Condition':>30} {'Mean Loss':>10} {'Std':>8} {'Mean DG':>8} {'N':>4}")
    print("-" * 64)
    for name in condition_order:
        entries = by_name.get(name, [])
        if not entries:
            continue
        losses = [e['summary']['final_loss'] for e in entries]
        trajectories = [e.get('loss_trajectory', []) for e in entries]
        dgs = [dg_index(t) for t in trajectories if t]
        n = len(losses)
        mean_l = sum(losses) / n
        std_l = (sum((x - mean_l) ** 2 for x in losses) / max(1, n - 1)) ** 0.5
        mean_dg = sum(dgs) / len(dgs) if dgs else 0.0
        label = condition_labels.get(name, name)
        print(f"{label:>30} {mean_l:>10.4f} {std_l:>8.4f} {mean_dg:>8.3f} {n:>4d}")

    print("\n  * = predicted best (paralleling chess paper's 'cautious position, courageous moves')")

    # Interpretation
    print("\n--- Chess Paper Parallel ---")
    print("  Cautious position + courageous moves ↔ stable forward + bold gradients")
    print("  Anomaly 1 (frozen heads help) = stable representations")
    print("  Anomaly 2 (sign-only works) = bold gradient updates")

    losses_by_name = {}
    for name, entries in by_name.items():
        losses_by_name[name] = sum(e['summary']['final_loss'] for e in entries) / len(entries)

    best_name = min((n for n in losses_by_name if n != 'baseline'),
                     key=losses_by_name.get, default=None)
    if best_name:
        best_loss = losses_by_name[best_name]
        label = condition_labels.get(best_name, best_name)
        if best_name == 'cautious_courageous':
            print(f"\n  CONFIRMED: {label} wins (loss={best_loss:.4f})")
            print("  -> Stable representations + bold updates is optimal")
        else:
            print(f"\n  Best non-baseline: {label} (loss={best_loss:.4f})")
            if best_name == 'cautious_cautious':
                print("  -> Low noise everywhere wins — system prefers stability")
            elif best_name == 'courageous_cautious':
                print("  -> Noisy forward + careful gradients — exploration helps")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        labels = []
        losses = []
        dgs = []
        colors = []
        color_map = {
            'baseline': 'gray',
            'cautious_cautious': 'lightblue',
            'cautious_courageous': 'steelblue',
            'courageous_cautious': 'lightsalmon',
            'courageous_courageous': 'coral',
        }
        for name in condition_order:
            entries = by_name.get(name, [])
            if not entries:
                continue
            labels.append(condition_labels.get(name, name).replace(' *', ''))
            losses.append(sum(e['summary']['final_loss'] for e in entries) / len(entries))
            trajs = [e.get('loss_trajectory', []) for e in entries]
            dgs.append(sum(dg_index(t) for t in trajs if t) / max(1, len(trajs)))
            colors.append(color_map.get(name, 'steelblue'))

        ax = axes[0]
        ax.barh(range(len(labels)), losses, color=colors, alpha=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Final Loss')
        ax.set_title('Courage vs Caution: Final Loss')
        ax.invert_yaxis()

        ax2 = axes[1]
        ax2.barh(range(len(labels)), dgs, color=colors, alpha=0.8)
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.set_xlabel('DG Index')
        ax2.set_title('Courage vs Caution: DG Index')
        ax2.invert_yaxis()

        plt.tight_layout()
        plt.savefig('results/exp6_courage_caution.png', dpi=150)
        print(f"\nPlot saved to results/exp6_courage_caution.png")
        plt.close()
    except ImportError:
        print("\nmatplotlib not available, skipping plots")

    print()


def run_all(num_reps=3, num_steps=200):
    experiment1(num_reps=num_reps, num_steps=num_steps)
    experiment2(num_reps=num_reps, num_steps=num_steps)
    experiment3(num_reps=num_reps, num_steps=num_steps)
    experiment4(num_reps=num_reps, num_steps=num_steps)
    experiment5(num_reps=num_reps, num_steps=num_steps)
    experiment6(num_reps=num_reps, num_steps=num_steps)


def _parse_args():
    """Parse --num-reps and --num-steps from CLI args."""
    num_reps = 3
    num_steps = 200
    for i, arg in enumerate(sys.argv):
        if arg == '--num-reps' and i + 1 < len(sys.argv):
            num_reps = int(sys.argv[i + 1])
        elif arg == '--num-steps' and i + 1 < len(sys.argv):
            num_steps = int(sys.argv[i + 1])
    return num_reps, num_steps


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'test'

    # Commands that accept --num-reps / --num-steps
    experiment_commands = {
        'experiment1': experiment1,
        'experiment2': experiment2,
        'experiment3': experiment3,
        'experiment4': experiment4,
        'experiment5': experiment5,
        'experiment6': experiment6,
        'all': run_all,
    }

    # Commands that don't
    other_commands = {
        'test': test,
        'baseline': baseline,
        'trajectory': trajectory,
        'analyze1': analyze1,
        'analyze2': analyze2,
        'analyze3': analyze3,
        'analyze4': analyze4,
        'analyze5': analyze5,
        'analyze6': analyze6,
    }

    if cmd in experiment_commands:
        num_reps, num_steps = _parse_args()
        print(f"Running {cmd} with num_reps={num_reps}, num_steps={num_steps}")
        experiment_commands[cmd](num_reps=num_reps, num_steps=num_steps)
    elif cmd in other_commands:
        other_commands[cmd]()
    else:
        all_commands = {**experiment_commands, **other_commands}
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(all_commands.keys())}")
        print(f"\nExperiment commands accept: --num-reps N --num-steps N")
        sys.exit(1)
