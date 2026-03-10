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


if __name__ == '__main__':
    analyze_exp1()
    analyze_exp2()
    analyze_exp3()
    analyze_exp4()
    analyze_exp5()
    analyze_exp6()
    summary_table()
