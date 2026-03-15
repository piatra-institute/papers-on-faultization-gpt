# /// script
# dependencies = ["numpy", "scipy"]
# ///
"""
Paired statistical analysis of all experiments (n=30).
Uses paired t-tests (matching seeds across conditions) and
independent t-tests where appropriate.
"""

import json
import os
import numpy as np
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def load_results(filename):
    with open(os.path.join(RESULTS_DIR, filename)) as f:
        return json.load(f)


def group_by_name(data):
    groups = {}
    for entry in data:
        groups.setdefault(entry['name'], []).append(entry)
    return groups


def paired_ttest(baseline_vals, condition_vals):
    """Paired t-test. Returns (t_stat, p_value, mean_diff, se_diff, cohen_d)."""
    diffs = np.array(condition_vals) - np.array(baseline_vals)
    n = len(diffs)
    mean_diff = np.mean(diffs)
    se_diff = np.std(diffs, ddof=1) / np.sqrt(n)
    if se_diff < 1e-15:
        return 0.0, 1.0, mean_diff, se_diff, 0.0
    t_stat = mean_diff / se_diff
    p_value = stats.t.sf(np.abs(t_stat), df=n-1) * 2  # two-tailed
    cohen_d = mean_diff / np.std(diffs, ddof=1)
    return t_stat, p_value, mean_diff, se_diff, cohen_d


def independent_ttest(a, b):
    """Welch's t-test. Returns (t_stat, p_value, mean_diff, cohen_d)."""
    a, b = np.array(a), np.array(b)
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    cohen_d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 1e-15 else 0.0
    return t_stat, p_value, np.mean(a) - np.mean(b), cohen_d


def get_metric(entries, metric='final_loss'):
    """Extract metric from list of result entries."""
    if metric == 'final_loss':
        return [e['summary']['final_loss'] for e in entries]
    elif metric == 'mean_loss':
        return [np.mean(e['loss_trajectory']) for e in entries]
    elif metric == 'dg_index':
        return [e['summary']['dg_index'] for e in entries]
    elif metric == 'min_loss':
        return [e['summary']['min_loss'] for e in entries]
    elif metric == 'val_final_loss':
        return [e['summary'].get('val_final_loss', e['summary']['final_loss'])
                for e in entries]
    elif metric == 'val_mean_loss':
        return [e['summary'].get('val_mean_loss', e['summary'].get('mean_loss', 0))
                for e in entries]
    raise ValueError(f"Unknown metric: {metric}")


def sig_marker(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.10:
        return '†'
    return 'ns'


def print_val_summary(groups, baseline_name='baseline', condition_names=None):
    """Print validation loss summary if available."""
    if condition_names is None:
        condition_names = [n for n in groups if n != baseline_name]
    bl = groups.get(baseline_name, [])
    if not bl or 'val_final_loss' not in bl[0].get('summary', {}):
        return  # no val data
    print("\n--- Validation Loss ---")
    bl_val = get_metric(bl, 'val_final_loss')
    print(f"  {baseline_name:30s}: val_final={np.mean(bl_val):.4f}±{np.std(bl_val, ddof=1):.4f}")
    for name in condition_names:
        cond = groups.get(name, [])
        if not cond:
            continue
        cond_val = get_metric(cond, 'val_final_loss')
        if len(cond_val) == len(bl_val):
            print_comparison(f"{name} (val)", bl_val, cond_val)
        else:
            print(f"  {name:30s}: val_final={np.mean(cond_val):.4f}±{np.std(cond_val, ddof=1):.4f}")


def print_comparison(label, baseline_vals, condition_vals, paired=True):
    """Print a statistical comparison line."""
    bl_mean = np.mean(baseline_vals)
    cond_mean = np.mean(condition_vals)
    pct_change = (cond_mean - bl_mean) / bl_mean * 100

    if paired:
        t, p, md, se, d = paired_ttest(baseline_vals, condition_vals)
        test_type = 'paired'
    else:
        t, p, md, d = independent_ttest(condition_vals, baseline_vals)
        test_type = 'indep'

    print(f"  {label:30s}: mean={cond_mean:.4f} ({pct_change:+.1f}%) "
          f"t={t:+.3f} p={p:.4f}{sig_marker(p):>3s} d={d:+.3f}")
    return {'label': label, 'mean': cond_mean, 'bl_mean': bl_mean,
            'pct_change': pct_change, 't': t, 'p': p, 'd': d}


# ============================================================================
# Experiment 1: Head Freezing
# ============================================================================

def analyze_exp1():
    print("=" * 70)
    print("EXPERIMENT 1: Head Freezing — Paired t-tests (n=30)")
    print("=" * 70)

    data = load_results('experiment1_head_freezing.json')
    groups = group_by_name(data)

    # Sort entries by seed within each group
    for name in groups:
        groups[name].sort(key=lambda e: e['seed'])

    baseline = groups['freeze_0']
    bl_final = get_metric(baseline, 'final_loss')
    bl_mean = get_metric(baseline, 'mean_loss')

    print(f"\nBaseline: final_loss={np.mean(bl_final):.4f}±{np.std(bl_final, ddof=1):.4f}, "
          f"mean_loss={np.mean(bl_mean):.4f}±{np.std(bl_mean, ddof=1):.4f} (n={len(bl_final)})")

    print("\n--- Final Loss vs Baseline (paired) ---")
    results = []
    for name in ['freeze_1', 'freeze_2', 'freeze_4', 'freeze_8', 'freeze_12', 'freeze_16']:
        cond = groups[name]
        cond.sort(key=lambda e: e['seed'])
        r = print_comparison(name, bl_final, get_metric(cond, 'final_loss'))
        results.append(r)

    print("\n--- Mean Loss vs Baseline (paired) ---")
    for name in ['freeze_1', 'freeze_2', 'freeze_4', 'freeze_8', 'freeze_12', 'freeze_16']:
        cond = groups[name]
        cond.sort(key=lambda e: e['seed'])
        print_comparison(name, bl_mean, get_metric(cond, 'mean_loss'))

    print("\n--- DG Index vs Baseline (paired) ---")
    bl_dg = get_metric(baseline, 'dg_index')
    for name in ['freeze_1', 'freeze_2', 'freeze_4', 'freeze_8', 'freeze_12', 'freeze_16']:
        cond = groups[name]
        cond.sort(key=lambda e: e['seed'])
        print_comparison(name, bl_dg, get_metric(cond, 'dg_index'))

    # Monotonicity test: Spearman correlation of freeze level vs loss
    print("\n--- Monotonicity (Spearman rank correlation) ---")
    levels = []
    losses = []
    for name in ['freeze_0', 'freeze_1', 'freeze_2', 'freeze_4', 'freeze_8', 'freeze_12', 'freeze_16']:
        for e in groups[name]:
            level = int(name.split('_')[1])
            levels.append(level)
            losses.append(e['summary']['final_loss'])
    rho, p = stats.spearmanr(levels, losses)
    print(f"  Spearman ρ={rho:.4f}, p={p:.4f} {sig_marker(p)}")

    return results


# ============================================================================
# Experiment 2: Cell-View
# ============================================================================

def analyze_exp2():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Cell-View GPT — Paired t-tests (n=30)")
    print("=" * 70)

    data = load_results('experiment2_cell_view.json')
    groups = group_by_name(data)
    for name in groups:
        groups[name].sort(key=lambda e: e['seed'])

    baseline = groups['baseline']
    cell_view = groups['cell_view']

    bl_final = get_metric(baseline, 'final_loss')
    cv_final = get_metric(cell_view, 'final_loss')
    bl_mean = get_metric(baseline, 'mean_loss')
    cv_mean = get_metric(cell_view, 'mean_loss')

    print(f"\nBaseline: final={np.mean(bl_final):.4f}±{np.std(bl_final, ddof=1):.4f}")
    print(f"Cell-view: final={np.mean(cv_final):.4f}±{np.std(cv_final, ddof=1):.4f}")

    print("\n--- Final Loss ---")
    print_comparison('cell_view', bl_final, cv_final)

    print("\n--- Mean Loss ---")
    print_comparison('cell_view', bl_mean, cv_mean)

    print("\n--- DG Index ---")
    bl_dg = get_metric(baseline, 'dg_index')
    cv_dg = get_metric(cell_view, 'dg_index')
    print_comparison('cell_view', bl_dg, cv_dg)

    # Effect size on mean loss
    diff = np.array(cv_mean) - np.array(bl_mean)
    print(f"\n  Mean loss increase: {np.mean(diff):.4f} ({np.mean(diff)/np.mean(bl_mean)*100:+.1f}%)")
    print(f"  95% CI of diff: [{np.mean(diff) - 1.96*np.std(diff, ddof=1)/np.sqrt(len(diff)):.4f}, "
          f"{np.mean(diff) + 1.96*np.std(diff, ddof=1)/np.sqrt(len(diff)):.4f}]")


# ============================================================================
# Experiment 3: Gradient Degradation
# ============================================================================

def analyze_exp3():
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Gradient Degradation — Paired t-tests (n=30)")
    print("=" * 70)

    data = load_results('experiment3_gradient_degradation.json')
    groups = group_by_name(data)
    for name in groups:
        groups[name].sort(key=lambda e: e['seed'])

    baseline = groups['baseline']
    bl_final = get_metric(baseline, 'final_loss')
    bl_mean = get_metric(baseline, 'mean_loss')

    print(f"\nBaseline: final={np.mean(bl_final):.4f}±{np.std(bl_final, ddof=1):.4f}, "
          f"mean={np.mean(bl_mean):.4f}±{np.std(bl_mean, ddof=1):.4f}")

    print("\n--- Final Loss vs Baseline ---")
    for name in ['noisy_grad_0.01', 'noisy_grad_0.1', 'sign_only', 'quantized_3']:
        cond = groups[name]
        cond.sort(key=lambda e: e['seed'])
        print_comparison(name, bl_final, get_metric(cond, 'final_loss'))

    print("\n--- Mean Loss vs Baseline ---")
    for name in ['noisy_grad_0.01', 'noisy_grad_0.1', 'sign_only', 'quantized_3']:
        cond = groups[name]
        cond.sort(key=lambda e: e['seed'])
        print_comparison(name, bl_mean, get_metric(cond, 'mean_loss'))


# ============================================================================
# Experiment 4: Vision Radius
# ============================================================================

def analyze_exp4():
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Vision Radius — Paired t-tests (n=30)")
    print("=" * 70)

    data = load_results('experiment4_vision_radius.json')
    groups = group_by_name(data)
    for name in groups:
        groups[name].sort(key=lambda e: e['seed'])

    baseline = groups['baseline']
    bl_final = get_metric(baseline, 'final_loss')
    bl_mean = get_metric(baseline, 'mean_loss')

    print(f"\nBaseline: final={np.mean(bl_final):.4f}±{np.std(bl_final, ddof=1):.4f}, "
          f"mean={np.mean(bl_mean):.4f}±{np.std(bl_mean, ddof=1):.4f}")

    print("\n--- Final Loss vs Baseline ---")
    for name in ['window_1', 'window_2', 'window_4', 'window_8', 'window_16']:
        if name in groups:
            cond = groups[name]
            cond.sort(key=lambda e: e['seed'])
            print_comparison(name, bl_final, get_metric(cond, 'final_loss'))

    print("\n--- Mean Loss vs Baseline ---")
    for name in ['window_1', 'window_2', 'window_4', 'window_8', 'window_16']:
        if name in groups:
            cond = groups[name]
            cond.sort(key=lambda e: e['seed'])
            print_comparison(name, bl_mean, get_metric(cond, 'mean_loss'))

    # Note: window_16 == baseline (block_size=16), so expect p≈1
    print("\n  Note: window_16 = full context (block_size=16), should match baseline")


# ============================================================================
# Experiment 5: Communication Topology
# ============================================================================

def analyze_exp5():
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Communication Topology — Paired t-tests (n=30)")
    print("=" * 70)

    data = load_results('experiment5_communication.json')
    groups = group_by_name(data)
    for name in groups:
        groups[name].sort(key=lambda e: e['seed'])

    baseline = groups['full']
    bl_final = get_metric(baseline, 'final_loss')
    bl_mean = get_metric(baseline, 'mean_loss')

    print(f"\nBaseline (full): final={np.mean(bl_final):.4f}±{np.std(bl_final, ddof=1):.4f}, "
          f"mean={np.mean(bl_mean):.4f}±{np.std(bl_mean, ddof=1):.4f}")

    print("\n--- Final Loss vs Full ---")
    for name in ['heavy', 'half', 'light', 'cell_view']:
        cond = groups[name]
        cond.sort(key=lambda e: e['seed'])
        print_comparison(name, bl_final, get_metric(cond, 'final_loss'))

    print("\n--- Mean Loss vs Full ---")
    for name in ['heavy', 'half', 'light', 'cell_view']:
        cond = groups[name]
        cond.sort(key=lambda e: e['seed'])
        print_comparison(name, bl_mean, get_metric(cond, 'mean_loss'))


# ============================================================================
# Experiment 6: Courage vs. Caution
# ============================================================================

def analyze_exp6():
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Courage vs. Caution — Paired t-tests (n=30)")
    print("=" * 70)

    data = load_results('experiment6_courage_caution.json')
    groups = group_by_name(data)
    for name in groups:
        groups[name].sort(key=lambda e: e['seed'])

    baseline = groups['baseline']
    bl_final = get_metric(baseline, 'final_loss')
    bl_mean = get_metric(baseline, 'mean_loss')

    print(f"\nBaseline: final={np.mean(bl_final):.4f}±{np.std(bl_final, ddof=1):.4f}, "
          f"mean={np.mean(bl_mean):.4f}±{np.std(bl_mean, ddof=1):.4f}")

    conditions = ['cautious_cautious', 'cautious_courageous',
                  'courageous_cautious', 'courageous_courageous']

    print("\n--- Final Loss vs Baseline ---")
    for name in conditions:
        cond = groups[name]
        cond.sort(key=lambda e: e['seed'])
        print_comparison(name, bl_final, get_metric(cond, 'final_loss'))

    print("\n--- Mean Loss vs Baseline ---")
    for name in conditions:
        cond = groups[name]
        cond.sort(key=lambda e: e['seed'])
        print_comparison(name, bl_mean, get_metric(cond, 'mean_loss'))

    # Cross-comparison: sign-only vs dropout (the chess-paper prediction)
    print("\n--- Chess-Paper Comparison: Sign-Only vs Dropout ---")
    sign_only = groups['cautious_courageous']
    dropout = groups['courageous_cautious']
    sign_only.sort(key=lambda e: e['seed'])
    dropout.sort(key=lambda e: e['seed'])

    so_final = get_metric(sign_only, 'final_loss')
    do_final = get_metric(dropout, 'final_loss')
    so_mean = get_metric(sign_only, 'mean_loss')
    do_mean = get_metric(dropout, 'mean_loss')

    print("  Sign-only vs Dropout (paired):")
    print_comparison('  sign_only - dropout (final)', do_final, so_final)
    print_comparison('  sign_only - dropout (mean)', do_mean, so_mean)


# ============================================================================
# Summary Table
# ============================================================================

def summary_table():
    print("\n" + "=" * 70)
    print("CROSS-EXPERIMENT SUMMARY (n=30, 200 steps, paired t-tests)")
    print("=" * 70)
    print(f"\n{'Experiment':35s} {'Condition':25s} {'Mean':>7s} {'Δ%':>6s} {'p':>8s} {'Sig':>4s}")
    print("-" * 87)

    # Exp 1
    data = load_results('experiment1_head_freezing.json')
    groups = group_by_name(data)
    for name in groups:
        groups[name].sort(key=lambda e: e['seed'])
    bl = get_metric(groups['freeze_0'], 'final_loss')
    for name in ['freeze_1', 'freeze_2', 'freeze_4', 'freeze_8', 'freeze_12', 'freeze_16']:
        cond = groups[name]
        cond.sort(key=lambda e: e['seed'])
        vals = get_metric(cond, 'final_loss')
        t, p, md, se, d = paired_ttest(bl, vals)
        pct = md / np.mean(bl) * 100
        print(f"{'1: Head Freezing':35s} {name:25s} {np.mean(vals):7.4f} {pct:+5.1f}% {p:8.4f} {sig_marker(p):>4s}")

    # Exp 2
    data = load_results('experiment2_cell_view.json')
    groups = group_by_name(data)
    for name in groups:
        groups[name].sort(key=lambda e: e['seed'])
    bl = get_metric(groups['baseline'], 'final_loss')
    vals = get_metric(groups['cell_view'], 'final_loss')
    t, p, md, se, d = paired_ttest(bl, vals)
    pct = md / np.mean(bl) * 100
    print(f"{'2: Cell-View':35s} {'cell_view':25s} {np.mean(vals):7.4f} {pct:+5.1f}% {p:8.4f} {sig_marker(p):>4s}")

    # Exp 3
    data = load_results('experiment3_gradient_degradation.json')
    groups = group_by_name(data)
    for name in groups:
        groups[name].sort(key=lambda e: e['seed'])
    bl = get_metric(groups['baseline'], 'final_loss')
    for name in ['noisy_grad_0.01', 'noisy_grad_0.1', 'sign_only', 'quantized_3']:
        cond = groups[name]
        cond.sort(key=lambda e: e['seed'])
        vals = get_metric(cond, 'final_loss')
        t, p, md, se, d = paired_ttest(bl, vals)
        pct = md / np.mean(bl) * 100
        print(f"{'3: Gradient Degradation':35s} {name:25s} {np.mean(vals):7.4f} {pct:+5.1f}% {p:8.4f} {sig_marker(p):>4s}")

    # Exp 4
    data = load_results('experiment4_vision_radius.json')
    groups = group_by_name(data)
    for name in groups:
        groups[name].sort(key=lambda e: e['seed'])
    bl = get_metric(groups['baseline'], 'final_loss')
    for name in ['window_1', 'window_2', 'window_4', 'window_8', 'window_16']:
        if name not in groups:
            continue
        cond = groups[name]
        cond.sort(key=lambda e: e['seed'])
        vals = get_metric(cond, 'final_loss')
        t, p, md, se, d = paired_ttest(bl, vals)
        pct = md / np.mean(bl) * 100
        print(f"{'4: Vision Radius':35s} {name:25s} {np.mean(vals):7.4f} {pct:+5.1f}% {p:8.4f} {sig_marker(p):>4s}")

    # Exp 5
    data = load_results('experiment5_communication.json')
    groups = group_by_name(data)
    for name in groups:
        groups[name].sort(key=lambda e: e['seed'])
    bl = get_metric(groups['full'], 'final_loss')
    for name in ['heavy', 'half', 'light', 'cell_view']:
        cond = groups[name]
        cond.sort(key=lambda e: e['seed'])
        vals = get_metric(cond, 'final_loss')
        t, p, md, se, d = paired_ttest(bl, vals)
        pct = md / np.mean(bl) * 100
        print(f"{'5: Communication Topology':35s} {name:25s} {np.mean(vals):7.4f} {pct:+5.1f}% {p:8.4f} {sig_marker(p):>4s}")

    # Exp 6
    data = load_results('experiment6_courage_caution.json')
    groups = group_by_name(data)
    for name in groups:
        groups[name].sort(key=lambda e: e['seed'])
    bl = get_metric(groups['baseline'], 'final_loss')
    for name in ['cautious_cautious', 'cautious_courageous', 'courageous_cautious', 'courageous_courageous']:
        cond = groups[name]
        cond.sort(key=lambda e: e['seed'])
        vals = get_metric(cond, 'final_loss')
        t, p, md, se, d = paired_ttest(bl, vals)
        pct = md / np.mean(bl) * 100
        print(f"{'6: Courage/Caution':35s} {name:25s} {np.mean(vals):7.4f} {pct:+5.1f}% {p:8.4f} {sig_marker(p):>4s}")

    print(f"\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, † p<0.10, ns p≥0.10")
    print(f"All tests: two-tailed paired t-test, seeds matched across conditions")


# ============================================================================
# Experiment 7: Recovery After Damage
# ============================================================================

def analyze_exp7():
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Recovery After Damage — Paired t-tests (n=30)")
    print("=" * 70)

    data = load_results('experiment7_recovery.json')

    rec_finals = np.array([r['recovery_final_loss'] for r in data])
    ctrl_finals = np.array([r['control_final_loss'] for r in data])
    ratios = np.array([r['final_ratio'] for r in data])
    rec_times = [r['recovery_time'] for r in data if r['recovery_time'] is not None]

    print(f"\nn={len(data)}")
    print(f"  Recovery final:  {np.mean(rec_finals):.4f} +/- {np.std(rec_finals, ddof=1):.4f}")
    print(f"  Control final:   {np.mean(ctrl_finals):.4f} +/- {np.std(ctrl_finals, ddof=1):.4f}")
    print(f"  Final ratio:     {np.mean(ratios):.4f} +/- {np.std(ratios, ddof=1):.4f}")

    # Paired t-test: recovery vs control
    t, p, md, se, d = paired_ttest(ctrl_finals.tolist(), rec_finals.tolist())
    pct = md / np.mean(ctrl_finals) * 100
    print(f"\n  Recovery vs Control: diff={md:+.4f} ({pct:+.1f}%) "
          f"t={t:+.3f} p={p:.4f}{sig_marker(p):>3s} d={d:+.3f}")

    if rec_times:
        print(f"  Recovery time: {np.mean(rec_times):.0f} +/- {np.std(rec_times, ddof=1):.0f} steps "
              f"({len(rec_times)}/{len(data)} recovered)")
    else:
        print(f"  No runs recovered to baseline level")

    overshoots = np.array([r['overshoot'] for r in data])
    print(f"  Overshoot: {np.mean(overshoots):.4f} +/- {np.std(overshoots, ddof=1):.4f}")

    # Test: is the ratio significantly different from 1.0?
    diffs = ratios - 1.0
    t_ratio = np.mean(diffs) / (np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
    p_ratio = stats.t.sf(np.abs(t_ratio), df=len(diffs)-1) * 2
    print(f"  Ratio vs 1.0: t={t_ratio:+.3f} p={p_ratio:.4f}{sig_marker(p_ratio):>3s}")


# ============================================================================
# Experiment 8: Chimera Assembly
# ============================================================================

def analyze_exp8():
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: Chimera Assembly — Paired t-tests (n=30)")
    print("=" * 70)

    data = load_results('experiment8_chimera.json')

    ctrl_finals = np.array([r['control_final_loss'] for r in data])
    print(f"\nn={len(data)}")
    print(f"  Control (A continues): {np.mean(ctrl_finals):.4f} +/- {np.std(ctrl_finals, ddof=1):.4f}")

    chimera_names = ['AABB', 'ABAB', 'BBAA', 'ABBA']

    print(f"\n  {'Chimera':8s} {'Init':>8s} {'Final':>8s} {'Recov%':>7s} {'vs Ctrl':>8s} {'t':>7s} {'p':>8s} {'Sig':>4s}")
    print("  " + "-" * 60)

    for name in chimera_names:
        inits = np.array([r['chimeras'][name]['initial_loss'] for r in data])
        finals = np.array([r['chimeras'][name]['final_loss'] for r in data])
        recovs = np.array([r['chimeras'][name]['recovery'] for r in data])

        # Paired t-test: chimera final vs control final
        t, p, md, se, d = paired_ttest(ctrl_finals.tolist(), finals.tolist())
        pct = md / np.mean(ctrl_finals) * 100

        print(f"  {name:8s} {np.mean(inits):8.4f} {np.mean(finals):8.4f} "
              f"{np.mean(recovs)*100:6.1f}% {pct:+7.1f}% {t:+7.3f} {p:8.4f} {sig_marker(p):>4s}")

    # Test: do chimeras converge to each other? (variance across chimera types)
    all_chi_finals = []
    for name in chimera_names:
        all_chi_finals.append([r['chimeras'][name]['final_loss'] for r in data])
    # Repeated measures ANOVA approximation: just report means
    print(f"\n  Chimera convergence (do different assemblies reach same loss?):")
    chi_means = [np.mean(f) for f in all_chi_finals]
    print(f"    Range of chimera means: {min(chi_means):.4f} to {max(chi_means):.4f} "
          f"(spread={max(chi_means)-min(chi_means):.4f})")


# ============================================================================
# Experiment 9: Gradual vs Sudden Damage
# ============================================================================

def analyze_exp9():
    print("\n" + "=" * 70)
    print("EXPERIMENT 9: Gradual vs Sudden Damage — Paired t-tests (n=30)")
    print("=" * 70)

    data = load_results('experiment9_gradual_vs_sudden.json')

    cond_names = ['control', 'sudden_full', 'gradual', 'sudden_half']
    print(f"\nn={len(data)}")

    # Extract data
    cond_data = {}
    for name in cond_names:
        finals = [r['conditions'][name]['final_loss'] for r in data]
        means = [r['conditions'][name]['mean_loss'] for r in data]
        cond_data[name] = {'finals': finals, 'means': means}
        print(f"  {name:15s}: final={np.mean(finals):.4f}+/-{np.std(finals, ddof=1):.4f}  "
              f"mean={np.mean(means):.4f}+/-{np.std(means, ddof=1):.4f}")

    # Paired comparisons vs control
    bl_final = cond_data['control']['finals']
    bl_mean = cond_data['control']['means']

    print(f"\n--- Final Loss vs Control (paired) ---")
    for name in ['sudden_full', 'gradual', 'sudden_half']:
        print_comparison(name, bl_final, cond_data[name]['finals'])

    print(f"\n--- Mean Loss vs Control (paired) ---")
    for name in ['sudden_full', 'gradual', 'sudden_half']:
        print_comparison(name, bl_mean, cond_data[name]['means'])

    # Key comparison: gradual vs sudden_full
    print(f"\n--- Gradual vs Sudden Full (paired) ---")
    print_comparison('gradual vs sudden', cond_data['sudden_full']['finals'],
                     cond_data['gradual']['finals'])
    print_comparison('sudden_half vs sudden', cond_data['sudden_full']['finals'],
                     cond_data['sudden_half']['finals'])


# ============================================================================
# Experiment 10: Regeneration (Layer Reset)
# ============================================================================

def analyze_exp10():
    print("\n" + "=" * 70)
    print("EXPERIMENT 10: Regeneration — Paired t-tests (n=30)")
    print("=" * 70)

    data = load_results('experiment10_regeneration.json')

    ctrl_finals = [r['control_final_loss'] for r in data]
    print(f"\nn={len(data)}")
    print(f"  Control final: {np.mean(ctrl_finals):.4f} +/- {np.std(ctrl_finals, ddof=1):.4f}")

    print(f"\n  {'Layer':6s} {'Damage':>8s} {'Final':>8s} {'Compl':>6s} {'vs Ctrl':>8s} {'t':>7s} {'p':>8s} {'Sig':>4s}")
    print("  " + "-" * 60)

    n_layer = 4
    for li in range(n_layer):
        damages = [r['layers'][str(li)]['damage'] for r in data]
        finals = [r['layers'][str(li)]['regen_final_loss'] for r in data]
        comps = [r['layers'][str(li)]['completeness'] for r in data]

        # Paired t-test: regen final vs control final
        t, p, md, se, d = paired_ttest(ctrl_finals, finals)
        pct = md / np.mean(ctrl_finals) * 100

        print(f"  L{li:1d}     {np.mean(damages):8.4f} {np.mean(finals):8.4f} "
              f"{np.mean(comps):5.3f} {pct:+7.1f}% {t:+7.3f} {p:8.4f} {sig_marker(p):>4s}")

    # Test: is completeness significantly > 0.9 (near-full recovery)?
    print(f"\n--- Completeness vs 0.9 (one-sample t-test) ---")
    for li in range(n_layer):
        comps = np.array([r['layers'][str(li)]['completeness'] for r in data])
        t_comp = (np.mean(comps) - 0.9) / (np.std(comps, ddof=1) / np.sqrt(len(comps)))
        p_comp = stats.t.sf(np.abs(t_comp), df=len(comps)-1) * 2
        print(f"  L{li}: completeness={np.mean(comps):.3f} vs 0.9: "
              f"t={t_comp:+.3f} p={p_comp:.4f}{sig_marker(p_comp):>3s}")

    # Test: does damage vary by layer position?
    print(f"\n--- Immediate damage by layer (Spearman: layer position vs damage) ---")
    all_li = []
    all_damage = []
    for li in range(n_layer):
        for r in data:
            all_li.append(li)
            all_damage.append(r['layers'][str(li)]['damage'])
    rho, p_rho = stats.spearmanr(all_li, all_damage)
    print(f"  Spearman rho={rho:.3f}, p={p_rho:.4f}{sig_marker(p_rho):>3s}")


# ============================================================================
# Experiment 11: Transplantation
# ============================================================================

def analyze_exp11():
    print("\n" + "=" * 70)
    print("EXPERIMENT 11: Transplantation — Paired t-tests (n=30)")
    print("=" * 70)

    data = load_results('experiment11_transplantation.json')

    ctrl_finals = [r['control_final_loss'] for r in data]
    print(f"\nn={len(data)}")
    print(f"  Control final: {np.mean(ctrl_finals):.4f} +/- {np.std(ctrl_finals, ddof=1):.4f}")

    print(f"\n  {'Layer':6s} {'Trans':>8s} {'Random':>8s} {'Gap':>8s} {'t(gap)':>7s} {'p':>8s} {'Sig':>4s}")
    print("  " + "-" * 55)

    n_layer = 4
    for li in range(n_layer):
        trans = [r['layers'][str(li)]['transplant_final'] for r in data]
        rand = [r['layers'][str(li)]['random_final'] for r in data]
        gaps = [r['layers'][str(li)]['transplant_gap'] for r in data]

        # Paired t-test: transplant vs random reset
        t, p, md, se, d = paired_ttest(rand, trans)

        print(f"  L{li:1d}     {np.mean(trans):8.4f} {np.mean(rand):8.4f} "
              f"{np.mean(gaps):+7.4f} {t:+7.3f} {p:8.4f} {sig_marker(p):>4s}")

    # Overall: is transplant better than random across all layers?
    print(f"\n--- Overall transplant vs random (all layers pooled) ---")
    all_trans = []
    all_rand = []
    for li in range(n_layer):
        for r in data:
            all_trans.append(r['layers'][str(li)]['transplant_final'])
            all_rand.append(r['layers'][str(li)]['random_final'])
    t_all, p_all, md_all, se_all, d_all = paired_ttest(all_rand, all_trans)
    print(f"  Transplant vs Random: diff={md_all:+.4f} t={t_all:+.3f} "
          f"p={p_all:.4f}{sig_marker(p_all):>3s} d={d_all:+.3f}")

    # Does transplant advantage vary by layer?
    print(f"\n--- Transplant advantage by layer position ---")
    all_li = []
    all_gaps = []
    for li in range(n_layer):
        for r in data:
            all_li.append(li)
            all_gaps.append(r['layers'][str(li)]['transplant_gap'])
    rho, p_rho = stats.spearmanr(all_li, all_gaps)
    print(f"  Spearman rho={rho:.3f}, p={p_rho:.4f}{sig_marker(p_rho):>3s}")


# ============================================================================
# Experiment 12: Competing Objectives
# ============================================================================

def analyze_exp12():
    print("\n" + "=" * 70)
    print("EXPERIMENT 12: Competing Objectives — Paired t-tests (n=30)")
    print("=" * 70)

    data = load_results('experiment12_competing_objectives.json')

    ctrl = np.array([r['control_final_loss'] for r in data])
    compete = np.array([r['competing_final_loss'] for r in data])
    freeze = np.array([r['freeze_final_loss'] for r in data])

    print(f"\nn={len(data)}")
    print(f"  Control:     {np.mean(ctrl):.4f} +/- {np.std(ctrl, ddof=1):.4f}")
    print(f"  Competing:   {np.mean(compete):.4f} +/- {np.std(compete, ddof=1):.4f}")
    print(f"  Freeze L2-3: {np.mean(freeze):.4f} +/- {np.std(freeze, ddof=1):.4f}")

    print(f"\n--- vs Control (paired) ---")
    print_comparison('competing', ctrl.tolist(), compete.tolist())
    print_comparison('freeze L2-3', ctrl.tolist(), freeze.tolist())

    print(f"\n--- Competing vs Freeze (paired) ---")
    print_comparison('competing vs freeze', freeze.tolist(), compete.tolist())

    # How much worse is competing vs control as a percentage?
    pct_diffs = [(c - ct) / ct * 100 for c, ct in zip(compete, ctrl)]
    print(f"\n  Competing vs control: {np.mean(pct_diffs):+.1f}% "
          f"+/- {np.std(pct_diffs, ddof=1):.1f}%")


# ============================================================================
# Summary table for experiments 7-12
# ============================================================================

def summary_table_new():
    print("\n" + "=" * 70)
    print("EXPERIMENTS 7-12 SUMMARY (n=30, 200 steps)")
    print("=" * 70)

    # Exp 7: Recovery
    try:
        data = load_results('experiment7_recovery.json')
        rec = [r['recovery_final_loss'] for r in data]
        ctrl = [r['control_final_loss'] for r in data]
        t, p, md, se, d = paired_ttest(ctrl, rec)
        pct = md / np.mean(ctrl) * 100
        print(f"{'7: Recovery':35s} {'rec vs ctrl':25s} {np.mean(rec):7.4f} {pct:+5.1f}% {p:8.4f} {sig_marker(p):>4s}")
        ratios = [r['final_ratio'] for r in data]
        print(f"{'':35s} {'final ratio':25s} {np.mean(ratios):7.4f}")
    except FileNotFoundError:
        print(f"{'7: Recovery':35s} — no results file —")

    # Exp 8: Chimera
    try:
        data = load_results('experiment8_chimera.json')
        ctrl = [r['control_final_loss'] for r in data]
        for name in ['AABB', 'ABAB', 'BBAA', 'ABBA']:
            finals = [r['chimeras'][name]['final_loss'] for r in data]
            t, p, md, se, d = paired_ttest(ctrl, finals)
            pct = md / np.mean(ctrl) * 100
            print(f"{'8: Chimera':35s} {name:25s} {np.mean(finals):7.4f} {pct:+5.1f}% {p:8.4f} {sig_marker(p):>4s}")
    except FileNotFoundError:
        print(f"{'8: Chimera':35s} — no results file —")

    # Exp 9: Gradual vs Sudden
    try:
        data = load_results('experiment9_gradual_vs_sudden.json')
        ctrl = [r['conditions']['control']['final_loss'] for r in data]
        for name in ['sudden_full', 'gradual', 'sudden_half']:
            finals = [r['conditions'][name]['final_loss'] for r in data]
            t, p, md, se, d = paired_ttest(ctrl, finals)
            pct = md / np.mean(ctrl) * 100
            print(f"{'9: Gradual vs Sudden':35s} {name:25s} {np.mean(finals):7.4f} {pct:+5.1f}% {p:8.4f} {sig_marker(p):>4s}")
    except FileNotFoundError:
        print(f"{'9: Gradual vs Sudden':35s} — no results file —")

    # Exp 10: Regeneration
    try:
        data = load_results('experiment10_regeneration.json')
        ctrl = [r['control_final_loss'] for r in data]
        for li in range(4):
            finals = [r['layers'][str(li)]['regen_final_loss'] for r in data]
            t, p, md, se, d = paired_ttest(ctrl, finals)
            pct = md / np.mean(ctrl) * 100
            print(f"{'10: Regeneration':35s} {'reset L'+str(li):25s} {np.mean(finals):7.4f} {pct:+5.1f}% {p:8.4f} {sig_marker(p):>4s}")
    except FileNotFoundError:
        print(f"{'10: Regeneration':35s} — no results file —")

    # Exp 11: Transplantation
    try:
        data = load_results('experiment11_transplantation.json')
        for li in range(4):
            trans = [r['layers'][str(li)]['transplant_final'] for r in data]
            rand = [r['layers'][str(li)]['random_final'] for r in data]
            t, p, md, se, d = paired_ttest(rand, trans)
            print(f"{'11: Transplantation':35s} {'L'+str(li)+' trans vs rand':25s} {np.mean(trans):7.4f} {md:+5.4f} {p:8.4f} {sig_marker(p):>4s}")
    except FileNotFoundError:
        print(f"{'11: Transplantation':35s} — no results file —")

    # Exp 12: Competing Objectives
    try:
        data = load_results('experiment12_competing_objectives.json')
        ctrl = [r['control_final_loss'] for r in data]
        compete = [r['competing_final_loss'] for r in data]
        freeze = [r['freeze_final_loss'] for r in data]
        t, p, md, se, d = paired_ttest(ctrl, compete)
        pct = md / np.mean(ctrl) * 100
        print(f"{'12: Competing Objectives':35s} {'competing':25s} {np.mean(compete):7.4f} {pct:+5.1f}% {p:8.4f} {sig_marker(p):>4s}")
        t, p, md, se, d = paired_ttest(ctrl, freeze)
        pct = md / np.mean(ctrl) * 100
        print(f"{'':35s} {'freeze L2-3':25s} {np.mean(freeze):7.4f} {pct:+5.1f}% {p:8.4f} {sig_marker(p):>4s}")
    except FileNotFoundError:
        print(f"{'12: Competing Objectives':35s} — no results file —")

    print(f"\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, † p<0.10, ns p>=0.10")


# ============================================================================
# N=300 Analysis (parameterized suffix)
# ============================================================================

def load_results_suffix(filename, suffix=''):
    """Load results with optional suffix (e.g. '_n300')."""
    base, ext = os.path.splitext(filename)
    return load_results(f'{base}{suffix}{ext}')


def analyze_exp1_n300():
    print("=" * 70)
    print("EXPERIMENT 1: Head Freezing — Paired t-tests (n=300)")
    print("=" * 70)
    data = load_results('experiment1_head_freezing_n300.json')
    groups = group_by_name(data)
    for name in groups:
        groups[name].sort(key=lambda e: e['seed'])
    baseline = groups['freeze_0']
    bl_final = get_metric(baseline, 'final_loss')
    print(f"\nBaseline: final_loss={np.mean(bl_final):.4f}±{np.std(bl_final, ddof=1):.4f} (n={len(bl_final)})")
    print("\n--- Final Loss vs Baseline (paired) ---")
    for name in ['freeze_1', 'freeze_2', 'freeze_4', 'freeze_8', 'freeze_12', 'freeze_16']:
        cond = groups[name]
        cond.sort(key=lambda e: e['seed'])
        print_comparison(name, bl_final, get_metric(cond, 'final_loss'))
    print("\n--- Monotonicity (Spearman rank correlation) ---")
    levels, losses = [], []
    for name in ['freeze_0', 'freeze_1', 'freeze_2', 'freeze_4', 'freeze_8', 'freeze_12', 'freeze_16']:
        for e in groups[name]:
            levels.append(int(name.split('_')[1]))
            losses.append(e['summary']['final_loss'])
    rho, p = stats.spearmanr(levels, losses)
    print(f"  Spearman ρ={rho:.4f}, p={p:.4f} {sig_marker(p)}")


def analyze_exp2_n300():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Cell-View GPT — Paired t-tests (n=300)")
    print("=" * 70)
    data = load_results('experiment2_cell_view_n300.json')
    groups = group_by_name(data)
    for name in groups:
        groups[name].sort(key=lambda e: e['seed'])
    bl_final = get_metric(groups['baseline'], 'final_loss')
    cv_final = get_metric(groups['cell_view'], 'final_loss')
    print(f"\nBaseline: final={np.mean(bl_final):.4f}±{np.std(bl_final, ddof=1):.4f}")
    print(f"Cell-view: final={np.mean(cv_final):.4f}±{np.std(cv_final, ddof=1):.4f}")
    print("\n--- Final Loss ---")
    print_comparison('cell_view', bl_final, cv_final)


def analyze_exp3_n300():
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Gradient Degradation — Paired t-tests (n=300)")
    print("=" * 70)
    data = load_results('experiment3_gradient_degradation_n300.json')
    groups = group_by_name(data)
    for name in groups:
        groups[name].sort(key=lambda e: e['seed'])
    bl_final = get_metric(groups['baseline'], 'final_loss')
    print(f"\nBaseline: final={np.mean(bl_final):.4f}±{np.std(bl_final, ddof=1):.4f}")
    print("\n--- Final Loss vs Baseline ---")
    for name in ['noisy_grad_0.01', 'noisy_grad_0.1', 'sign_only', 'quantized_3']:
        cond = groups[name]
        cond.sort(key=lambda e: e['seed'])
        print_comparison(name, bl_final, get_metric(cond, 'final_loss'))


def analyze_exp4_n300():
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Vision Radius — Paired t-tests (n=300)")
    print("=" * 70)
    data = load_results('experiment4_vision_radius_n300.json')
    groups = group_by_name(data)
    for name in groups:
        groups[name].sort(key=lambda e: e['seed'])
    bl_final = get_metric(groups['baseline'], 'final_loss')
    print(f"\nBaseline: final={np.mean(bl_final):.4f}±{np.std(bl_final, ddof=1):.4f}")
    print("\n--- Final Loss vs Baseline ---")
    for name in ['window_1', 'window_2', 'window_4', 'window_8', 'window_16']:
        if name in groups:
            cond = groups[name]
            cond.sort(key=lambda e: e['seed'])
            print_comparison(name, bl_final, get_metric(cond, 'final_loss'))


def analyze_exp5_n300():
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Communication Topology — Paired t-tests (n=300)")
    print("=" * 70)
    data = load_results('experiment5_communication_n300.json')
    groups = group_by_name(data)
    for name in groups:
        groups[name].sort(key=lambda e: e['seed'])
    bl_final = get_metric(groups['full'], 'final_loss')
    print(f"\nBaseline (full): final={np.mean(bl_final):.4f}±{np.std(bl_final, ddof=1):.4f}")
    print("\n--- Final Loss vs Full ---")
    for name in ['heavy', 'half', 'light', 'cell_view']:
        cond = groups[name]
        cond.sort(key=lambda e: e['seed'])
        print_comparison(name, bl_final, get_metric(cond, 'final_loss'))


def analyze_exp6_n300():
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Courage vs. Caution — Paired t-tests (n=300)")
    print("=" * 70)
    data = load_results('experiment6_courage_caution_n300.json')
    groups = group_by_name(data)
    for name in groups:
        groups[name].sort(key=lambda e: e['seed'])
    bl_final = get_metric(groups['baseline'], 'final_loss')
    print(f"\nBaseline: final={np.mean(bl_final):.4f}±{np.std(bl_final, ddof=1):.4f}")
    conditions = ['cautious_cautious', 'cautious_courageous', 'courageous_cautious', 'courageous_courageous']
    print("\n--- Final Loss vs Baseline ---")
    for name in conditions:
        cond = groups[name]
        cond.sort(key=lambda e: e['seed'])
        print_comparison(name, bl_final, get_metric(cond, 'final_loss'))
    print("\n--- Sign-Only vs Dropout (paired) ---")
    sign_only = groups['cautious_courageous']
    dropout = groups['courageous_cautious']
    sign_only.sort(key=lambda e: e['seed'])
    dropout.sort(key=lambda e: e['seed'])
    print_comparison('sign_only - dropout (final)',
                     get_metric(dropout, 'final_loss'),
                     get_metric(sign_only, 'final_loss'))


def analyze_exp7_n300():
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Recovery After Damage — Paired t-tests (n=300)")
    print("=" * 70)
    data = load_results('experiment7_recovery_n300.json')
    rec_finals = np.array([r['recovery_final_loss'] for r in data])
    ctrl_finals = np.array([r['control_final_loss'] for r in data])
    ratios = np.array([r['final_ratio'] for r in data])
    print(f"\nn={len(data)}")
    print(f"  Recovery final:  {np.mean(rec_finals):.4f} +/- {np.std(rec_finals, ddof=1):.4f}")
    print(f"  Control final:   {np.mean(ctrl_finals):.4f} +/- {np.std(ctrl_finals, ddof=1):.4f}")
    print(f"  Final ratio:     {np.mean(ratios):.4f} +/- {np.std(ratios, ddof=1):.4f}")
    t, p, md, se, d = paired_ttest(ctrl_finals.tolist(), rec_finals.tolist())
    pct = md / np.mean(ctrl_finals) * 100
    print(f"\n  Recovery vs Control: diff={md:+.4f} ({pct:+.1f}%) t={t:+.3f} p={p:.4f}{sig_marker(p):>3s} d={d:+.3f}")
    overshoots = np.array([r['overshoot'] for r in data])
    print(f"  Overshoot: {np.mean(overshoots):.4f} +/- {np.std(overshoots, ddof=1):.4f}")


def analyze_exp8_n300():
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: Chimera Assembly — Paired t-tests (n=300)")
    print("=" * 70)
    data = load_results('experiment8_chimera_n300.json')
    ctrl_finals = np.array([r['control_final_loss'] for r in data])
    print(f"\nn={len(data)}")
    print(f"  Control: {np.mean(ctrl_finals):.4f} +/- {np.std(ctrl_finals, ddof=1):.4f}")
    for name in ['AABB', 'ABAB', 'BBAA', 'ABBA']:
        finals = np.array([r['chimeras'][name]['final_loss'] for r in data])
        t, p, md, se, d = paired_ttest(ctrl_finals.tolist(), finals.tolist())
        pct = md / np.mean(ctrl_finals) * 100
        print(f"  {name}: final={np.mean(finals):.4f} ({pct:+.1f}%) p={p:.4f}{sig_marker(p):>3s}")


def analyze_exp9_n300():
    print("\n" + "=" * 70)
    print("EXPERIMENT 9: Gradual vs Sudden — Paired t-tests (n=300)")
    print("=" * 70)
    data = load_results('experiment9_gradual_vs_sudden_n300.json')
    cond_names = ['control', 'sudden_full', 'gradual', 'sudden_half']
    print(f"\nn={len(data)}")
    cond_data = {}
    for name in cond_names:
        finals = [r['conditions'][name]['final_loss'] for r in data]
        means = [r['conditions'][name]['mean_loss'] for r in data]
        cond_data[name] = {'finals': finals, 'means': means}
        print(f"  {name:15s}: final={np.mean(finals):.4f}+/-{np.std(finals, ddof=1):.4f}")
    bl_final = cond_data['control']['finals']
    print(f"\n--- Final Loss vs Control (paired) ---")
    for name in ['sudden_full', 'gradual', 'sudden_half']:
        print_comparison(name, bl_final, cond_data[name]['finals'])
    print(f"\n--- KEY: Gradual vs Sudden Full (paired) ---")
    print_comparison('gradual vs sudden', cond_data['sudden_full']['finals'],
                     cond_data['gradual']['finals'])


def analyze_exp10_n300():
    print("\n" + "=" * 70)
    print("EXPERIMENT 10: Regeneration — Paired t-tests (n=300)")
    print("=" * 70)
    data = load_results('experiment10_regeneration_n300.json')
    ctrl_finals = [r['control_final_loss'] for r in data]
    print(f"\nn={len(data)}")
    print(f"  Control final: {np.mean(ctrl_finals):.4f} +/- {np.std(ctrl_finals, ddof=1):.4f}")
    for li in range(4):
        finals = [r['layers'][str(li)]['regen_final_loss'] for r in data]
        comps = [r['layers'][str(li)]['completeness'] for r in data]
        t, p, md, se, d = paired_ttest(ctrl_finals, finals)
        pct = md / np.mean(ctrl_finals) * 100
        print(f"  L{li}: final={np.mean(finals):.4f} compl={np.mean(comps):.3f} ({pct:+.1f}%) p={p:.4f}{sig_marker(p):>3s}")


def analyze_exp11_n300():
    print("\n" + "=" * 70)
    print("EXPERIMENT 11: Transplantation — Paired t-tests (n=300)")
    print("=" * 70)
    data = load_results('experiment11_transplantation_n300.json')
    print(f"\nn={len(data)}")
    for li in range(4):
        trans = [r['layers'][str(li)]['transplant_final'] for r in data]
        rand = [r['layers'][str(li)]['random_final'] for r in data]
        t, p, md, se, d = paired_ttest(rand, trans)
        print(f"  L{li}: trans={np.mean(trans):.4f} rand={np.mean(rand):.4f} gap={md:+.4f} p={p:.4f}{sig_marker(p):>3s}")
    all_trans, all_rand = [], []
    for li in range(4):
        for r in data:
            all_trans.append(r['layers'][str(li)]['transplant_final'])
            all_rand.append(r['layers'][str(li)]['random_final'])
    t_all, p_all, md_all, se_all, d_all = paired_ttest(all_rand, all_trans)
    print(f"  Overall: diff={md_all:+.4f} p={p_all:.4f}{sig_marker(p_all):>3s}")


def analyze_exp12_n300():
    print("\n" + "=" * 70)
    print("EXPERIMENT 12: Competing Objectives — Paired t-tests (n=300)")
    print("=" * 70)
    data = load_results('experiment12_competing_objectives_n300.json')
    ctrl = np.array([r['control_final_loss'] for r in data])
    compete = np.array([r['competing_final_loss'] for r in data])
    freeze = np.array([r['freeze_final_loss'] for r in data])
    print(f"\nn={len(data)}")
    print(f"  Control:     {np.mean(ctrl):.4f} +/- {np.std(ctrl, ddof=1):.4f}")
    print(f"  Competing:   {np.mean(compete):.4f} +/- {np.std(compete, ddof=1):.4f}")
    print(f"  Freeze L2-3: {np.mean(freeze):.4f} +/- {np.std(freeze, ddof=1):.4f}")
    print(f"\n--- vs Control (paired) ---")
    print_comparison('competing', ctrl.tolist(), compete.tolist())
    print_comparison('freeze L2-3', ctrl.tolist(), freeze.tolist())
    print(f"\n--- Competing vs Freeze (paired) ---")
    print_comparison('competing vs freeze', freeze.tolist(), compete.tolist())


# ============================================================================
# Cross-Scale Comparison (n=30 vs n=300)
# ============================================================================

def ci_95(vals):
    """Return (mean, lower, upper) 95% CI."""
    m = np.mean(vals)
    se = np.std(vals, ddof=1) / np.sqrt(len(vals))
    return m, m - 1.96 * se, m + 1.96 * se


def cross_scale_summary():
    """Compare key findings across n=30 and n=300."""
    print("\n" + "=" * 70)
    print("CROSS-SCALE COMPARISON: n=30 vs n=300")
    print("=" * 70)

    scales = [
        ('n=30', ''),
        ('n=300', '_n300'),
    ]

    print(f"\n{'Experiment':35s} {'Scale':>6s} {'Mean':>8s} {'95% CI':>18s} {'p':>8s} {'Sig':>4s}")
    print("-" * 80)

    # Exp 1: Head freezing threshold
    for label, suffix in scales:
        try:
            data = load_results(f'experiment1_head_freezing{suffix}.json')
            groups = group_by_name(data)
            for name in groups:
                groups[name].sort(key=lambda e: e['seed'])
            bl = get_metric(groups['freeze_0'], 'final_loss')
            for name in ['freeze_8']:
                cond = groups[name]
                cond.sort(key=lambda e: e['seed'])
                vals = get_metric(cond, 'final_loss')
                t, p, md, se, d = paired_ttest(bl, vals)
                m, lo, hi = ci_95(vals)
                print(f"{'1: Head freeze 8':35s} {label:>6s} {m:8.4f} [{lo:.4f}, {hi:.4f}] {p:8.4f} {sig_marker(p):>4s}")
        except FileNotFoundError:
            print(f"{'1: Head freeze 8':35s} {label:>6s} — no data —")

    # Exp 2: Cell-view
    for label, suffix in scales:
        try:
            data = load_results(f'experiment2_cell_view{suffix}.json')
            groups = group_by_name(data)
            for name in groups:
                groups[name].sort(key=lambda e: e['seed'])
            bl = get_metric(groups['baseline'], 'final_loss')
            vals = get_metric(groups['cell_view'], 'final_loss')
            t, p, md, se, d = paired_ttest(bl, vals)
            m, lo, hi = ci_95(vals)
            print(f"{'2: Cell-view':35s} {label:>6s} {m:8.4f} [{lo:.4f}, {hi:.4f}] {p:8.4f} {sig_marker(p):>4s}")
        except FileNotFoundError:
            print(f"{'2: Cell-view':35s} {label:>6s} — no data —")

    # Exp 3: Sign-only
    for label, suffix in scales:
        try:
            data = load_results(f'experiment3_gradient_degradation{suffix}.json')
            groups = group_by_name(data)
            for name in groups:
                groups[name].sort(key=lambda e: e['seed'])
            bl = get_metric(groups['baseline'], 'final_loss')
            vals = get_metric(groups['sign_only'], 'final_loss')
            t, p, md, se, d = paired_ttest(bl, vals)
            m, lo, hi = ci_95(vals)
            print(f"{'3: Sign-only':35s} {label:>6s} {m:8.4f} [{lo:.4f}, {hi:.4f}] {p:8.4f} {sig_marker(p):>4s}")
        except FileNotFoundError:
            print(f"{'3: Sign-only':35s} {label:>6s} — no data —")

    # Exp 7: Recovery ratio
    for label, suffix in scales:
        try:
            data = load_results(f'experiment7_recovery{suffix}.json')
            ratios = [r['final_ratio'] for r in data]
            m, lo, hi = ci_95(ratios)
            diffs = np.array(ratios) - 1.0
            t_r = np.mean(diffs) / (np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
            p_r = stats.t.sf(np.abs(t_r), df=len(diffs)-1) * 2
            print(f"{'7: Recovery ratio':35s} {label:>6s} {m:8.4f} [{lo:.4f}, {hi:.4f}] {p_r:8.4f} {sig_marker(p_r):>4s}")
        except FileNotFoundError:
            print(f"{'7: Recovery ratio':35s} {label:>6s} — no data —")

    # Exp 9: Gradual vs sudden (KEY)
    for label, suffix in scales:
        try:
            data = load_results(f'experiment9_gradual_vs_sudden{suffix}.json')
            gradual = [r['conditions']['gradual']['final_loss'] for r in data]
            sudden = [r['conditions']['sudden_full']['final_loss'] for r in data]
            t, p, md, se, d = paired_ttest(sudden, gradual)
            m, lo, hi = ci_95(np.array(gradual) - np.array(sudden))
            print(f"{'9: Gradual-Sudden gap':35s} {label:>6s} {np.mean(np.array(gradual)-np.array(sudden)):+8.4f} [{lo:.4f}, {hi:.4f}] {p:8.4f} {sig_marker(p):>4s}")
        except FileNotFoundError:
            print(f"{'9: Gradual-Sudden gap':35s} {label:>6s} — no data —")

    # Exp 10: Regeneration completeness
    for label, suffix in scales:
        try:
            data = load_results(f'experiment10_regeneration{suffix}.json')
            all_comps = []
            for li in range(4):
                all_comps.extend([r['layers'][str(li)]['completeness'] for r in data])
            m, lo, hi = ci_95(all_comps)
            print(f"{'10: Regen completeness (all L)':35s} {label:>6s} {m:8.3f} [{lo:.3f}, {hi:.3f}]")
        except FileNotFoundError:
            print(f"{'10: Regen completeness (all L)':35s} {label:>6s} — no data —")

    # Exp 12: Adversarial
    for label, suffix in scales:
        try:
            data = load_results(f'experiment12_competing_objectives{suffix}.json')
            ctrl = [r['control_final_loss'] for r in data]
            compete = [r['competing_final_loss'] for r in data]
            t, p, md, se, d = paired_ttest(ctrl, compete)
            pct = md / np.mean(ctrl) * 100
            print(f"{'12: Adversarial Δ%':35s} {label:>6s} {pct:+7.1f}%                    {p:8.4f} {sig_marker(p):>4s}")
        except FileNotFoundError:
            print(f"{'12: Adversarial Δ%':35s} {label:>6s} — no data —")

    print(f"\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, † p<0.10, ns p≥0.10")


def behavioral_classification():
    """Classify findings by behavioral category (emergent / basin geometry / tolerance)."""
    print("\n" + "=" * 70)
    print("BEHAVIORAL CLASSIFICATION at n=300")
    print("=" * 70)

    suffix = '_n300'

    print("\n=== EMERGENT BEHAVIORS (not directly prescribed by the optimizer) ===\n")

    # Exp 9: Stress inoculation
    try:
        data = load_results(f'experiment9_gradual_vs_sudden{suffix}.json')
        gradual = [r['conditions']['gradual']['final_loss'] for r in data]
        sudden = [r['conditions']['sudden_full']['final_loss'] for r in data]
        ctrl = [r['conditions']['control']['final_loss'] for r in data]
        t, p, md, se, d = paired_ttest(sudden, gradual)
        print(f"  Exp 9 — Stress Inoculation")
        print(f"    Gradual vs control: p={paired_ttest(ctrl, gradual)[1]:.4f}")
        print(f"    Sudden vs control:  p={paired_ttest(ctrl, sudden)[1]:.4f}")
        print(f"    Gradual vs sudden:  p={p:.4f} d={d:+.3f} {sig_marker(p)}")
        print(f"    The optimizer prescribes loss minimization — not tolerance to noise schedules")
    except FileNotFoundError:
        print(f"  Exp 9 — no n=300 data")

    # Exp 7: Recovery
    try:
        data = load_results(f'experiment7_recovery{suffix}.json')
        rec = [r['recovery_final_loss'] for r in data]
        ctrl = [r['control_final_loss'] for r in data]
        t, p, md, se, d = paired_ttest(ctrl, rec)
        rec_times = [r['recovery_time'] for r in data if r['recovery_time'] is not None]
        print(f"\n  Exp 7 — Complete Recovery")
        print(f"    Recovery vs control: p={p:.4f} {sig_marker(p)}")
        if rec_times:
            print(f"    Recovery time: {np.mean(rec_times):.1f}±{np.std(rec_times, ddof=1):.1f} steps ({len(rec_times)}/{len(data)} recovered)")
        print(f"    Complete recovery: the optimizer finds the same minimum after transient damage")
    except FileNotFoundError:
        print(f"  Exp 7 — no n=300 data")

    # Exp 10: Regeneration
    try:
        data = load_results(f'experiment10_regeneration{suffix}.json')
        ctrl = [r['control_final_loss'] for r in data]
        print(f"\n  Exp 10 — Regeneration")
        for li in range(4):
            finals = [r['layers'][str(li)]['regen_final_loss'] for r in data]
            comps = [r['layers'][str(li)]['completeness'] for r in data]
            t, p, md, se, d = paired_ttest(ctrl, finals)
            print(f"    L{li}: completeness={np.mean(comps):.3f} p={p:.4f} {sig_marker(p)}")
        print(f"    Destroyed layers rebuild to control-equivalent performance")
    except FileNotFoundError:
        print(f"  Exp 10 — no n=300 data")

    # Exp 1: Head freezing trajectory improvement
    try:
        data = load_results(f'experiment1_head_freezing{suffix}.json')
        groups = group_by_name(data)
        for name in groups:
            groups[name].sort(key=lambda e: e['seed'])
        bl_mean = get_metric(groups['freeze_0'], 'mean_loss')
        print(f"\n  Exp 1 — Removing Components Improves Trajectory")
        for name in ['freeze_8', 'freeze_12', 'freeze_16']:
            cond = groups[name]
            cond.sort(key=lambda e: e['seed'])
            vals = get_metric(cond, 'mean_loss')
            t, p, md, se, d = paired_ttest(bl_mean, vals)
            pct = md / np.mean(bl_mean) * 100
            print(f"    {name}: mean_loss Δ={pct:+.2f}% p={p:.4f} {sig_marker(p)}")
        print(f"    Frozen random-projection heads reduce gradient interference")
    except FileNotFoundError:
        print(f"  Exp 1 — no n=300 data")

    print("\n=== BASIN GEOMETRY (expected optimizer behavior on this landscape) ===\n")

    # Exp 8: Chimera
    try:
        data = load_results(f'experiment8_chimera{suffix}.json')
        ctrl = [r['control_final_loss'] for r in data]
        print(f"  Exp 8 — Chimera Convergence")
        for name in ['AABB', 'ABAB', 'BBAA', 'ABBA']:
            finals = [r['chimeras'][name]['final_loss'] for r in data]
            t, p, md, se, d = paired_ttest(ctrl, finals)
            print(f"    {name}: p={p:.4f} {sig_marker(p)}")
        print(f"    Assembled parts converge — SGD re-finds the same basin")
    except FileNotFoundError:
        print(f"  Exp 8 — no n=300 data")

    # Exp 11: Transplant
    try:
        data = load_results(f'experiment11_transplantation{suffix}.json')
        all_trans, all_rand = [], []
        for li in range(4):
            for r in data:
                all_trans.append(r['layers'][str(li)]['transplant_final'])
                all_rand.append(r['layers'][str(li)]['random_final'])
        t, p, md, se, d = paired_ttest(all_rand, all_trans)
        print(f"\n  Exp 11 — Transplant = Random Reset")
        print(f"    Overall: p={p:.4f} {sig_marker(p)}")
        print(f"    Foreign parts work no better than random — same basin explanation")
    except FileNotFoundError:
        print(f"  Exp 11 — no n=300 data")

    print("\n=== TOLERANCE (system absorbs perturbation without meaningful degradation) ===\n")

    # Exp 3: Gradient degradation
    try:
        data = load_results(f'experiment3_gradient_degradation{suffix}.json')
        groups = group_by_name(data)
        for name in groups:
            groups[name].sort(key=lambda e: e['seed'])
        bl = get_metric(groups['baseline'], 'final_loss')
        print(f"  Exp 3 — Gradient Degradation Tolerance")
        for name in ['noisy_grad_0.01', 'noisy_grad_0.1', 'sign_only', 'quantized_3']:
            vals = get_metric(groups[name], 'final_loss')
            t, p, md, se, d = paired_ttest(bl, vals)
            pct = md / np.mean(bl) * 100
            print(f"    {name:20s}: Δ={pct:+.1f}% p={p:.4f} {sig_marker(p)}")
    except FileNotFoundError:
        print(f"  Exp 3 — no n=300 data")

    # Exp 5: Communication
    try:
        data = load_results(f'experiment5_communication{suffix}.json')
        groups = group_by_name(data)
        for name in groups:
            groups[name].sort(key=lambda e: e['seed'])
        bl = get_metric(groups['full'], 'final_loss')
        print(f"\n  Exp 5 — Partial Communication Tolerance")
        for name in ['heavy', 'half', 'light', 'cell_view']:
            vals = get_metric(groups[name], 'final_loss')
            t, p, md, se, d = paired_ttest(bl, vals)
            pct = md / np.mean(bl) * 100
            print(f"    {name:12s}: Δ={pct:+.1f}% p={p:.4f} {sig_marker(p)}")
    except FileNotFoundError:
        print(f"  Exp 5 — no n=300 data")

    # Exp 12: Absence vs adversity
    try:
        data = load_results(f'experiment12_competing_objectives{suffix}.json')
        ctrl = [r['control_final_loss'] for r in data]
        compete = [r['competing_final_loss'] for r in data]
        freeze = [r['freeze_final_loss'] for r in data]
        t_c, p_c, md_c, se_c, d_c = paired_ttest(ctrl, compete)
        t_f, p_f, md_f, se_f, d_f = paired_ttest(ctrl, freeze)
        pct_c = md_c / np.mean(ctrl) * 100
        pct_f = md_f / np.mean(ctrl) * 100
        print(f"\n  Exp 12 — Absence vs Adversity Boundary")
        print(f"    Adversarial: Δ={pct_c:+.1f}% p={p_c:.4f} {sig_marker(p_c)}")
        print(f"    Frozen:      Δ={pct_f:+.1f}% p={p_f:.4f} {sig_marker(p_f)}")
        print(f"    Routes around silence; cannot defend against sabotage")
    except FileNotFoundError:
        print(f"  Exp 12 — no n=300 data")


if __name__ == '__main__':
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''

    if cmd == 'new':
        analyze_exp7()
        analyze_exp8()
        analyze_exp9()
        analyze_exp10()
        analyze_exp11()
        analyze_exp12()
        summary_table_new()
    elif cmd == 'all':
        analyze_exp1()
        analyze_exp2()
        analyze_exp3()
        analyze_exp4()
        analyze_exp5()
        analyze_exp6()
        summary_table()
        analyze_exp7()
        analyze_exp8()
        analyze_exp9()
        analyze_exp10()
        analyze_exp11()
        analyze_exp12()
        summary_table_new()
    elif cmd == 'n300':
        analyze_exp1_n300()
        analyze_exp2_n300()
        analyze_exp3_n300()
        analyze_exp4_n300()
        analyze_exp5_n300()
        analyze_exp6_n300()
        analyze_exp7_n300()
        analyze_exp8_n300()
        analyze_exp9_n300()
        analyze_exp10_n300()
        analyze_exp11_n300()
        analyze_exp12_n300()
    elif cmd == 'cross':
        cross_scale_summary()
    elif cmd in ('classify', 'freedom'):
        behavioral_classification()
    elif cmd == 'full':
        # All scales + cross-scale + behavioral classification
        print("=" * 70)
        print("N=30 ANALYSIS")
        print("=" * 70)
        analyze_exp1()
        analyze_exp2()
        analyze_exp3()
        analyze_exp4()
        analyze_exp5()
        analyze_exp6()
        summary_table()
        analyze_exp7()
        analyze_exp8()
        analyze_exp9()
        analyze_exp10()
        analyze_exp11()
        analyze_exp12()
        summary_table_new()
        print("\n\n" + "=" * 70)
        print("N=300 ANALYSIS")
        print("=" * 70)
        analyze_exp1_n300()
        analyze_exp2_n300()
        analyze_exp3_n300()
        analyze_exp4_n300()
        analyze_exp5_n300()
        analyze_exp6_n300()
        analyze_exp7_n300()
        analyze_exp8_n300()
        analyze_exp9_n300()
        analyze_exp10_n300()
        analyze_exp11_n300()
        analyze_exp12_n300()
        cross_scale_summary()
        behavioral_classification()
    else:
        analyze_exp1()
        analyze_exp2()
        analyze_exp3()
        analyze_exp4()
        analyze_exp5()
        analyze_exp6()
        summary_table()
