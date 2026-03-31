"""
Faultization GPT — Visualization

Plotting functions that take Probe data and produce graphs.
Requires matplotlib (installed via run.py inline script metadata).
"""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Phase type -> color mapping
PHASE_COLORS = {
    'rapid_descent': '#2196F3',    # blue
    'gradual_descent': '#64B5F6',  # light blue
    'plateau': '#FFC107',          # amber
    'loss_increase': '#F44336',    # red
    'short_run': '#9E9E9E',        # grey
}

# Anomaly type -> color mapping
ANOMALY_COLORS = {
    'sync_event': '#E91E63',           # pink
    'ghost_spike': '#9C27B0',          # purple
    'sudden_specialization': '#FF9800', # orange
    'role_reversal': '#4CAF50',        # green
    'gradient_divergence': '#00BCD4',  # cyan
    'periodicity': '#795548',          # brown
}


def plot_loss_with_phases(loss_values, phases, path):
    """Loss curve with background bands colored by phase type."""
    fig, ax = plt.subplots(figsize=(12, 4))
    steps = list(range(len(loss_values)))

    # Draw phase bands
    for phase in phases:
        color = PHASE_COLORS.get(phase['type'], '#EEEEEE')
        ax.axvspan(phase['start'], phase['end'], alpha=0.2, color=color)

    # Loss curve
    ax.plot(steps, loss_values, color='#212121', linewidth=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Trajectory with Training Phases')

    # Legend for phases
    seen = set()
    handles = []
    for phase in phases:
        t = phase['type']
        if t not in seen:
            seen.add(t)
            handles.append(mpatches.Patch(
                color=PHASE_COLORS.get(t, '#EEEEEE'), alpha=0.4, label=t))
    if handles:
        ax.legend(handles=handles, loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_head_norm_heatmap(probe, anomalies, path):
    """All heads x steps heatmap showing norm intensity, with anomaly markers."""
    head_keys = sorted(probe.head_outputs.keys())
    if not head_keys:
        return

    # Build the matrix
    all_steps = set()
    for key in head_keys:
        for s, _ in probe.head_outputs[key]:
            all_steps.add(s)
    all_steps = sorted(all_steps)
    step_idx = {s: i for i, s in enumerate(all_steps)}

    matrix = []
    labels = []
    for key in head_keys:
        row = [0.0] * len(all_steps)
        for s, n in probe.head_outputs[key]:
            if s in step_idx:
                row[step_idx[s]] = n
        matrix.append(row)
        labels.append(f'L{key[0]}H{key[1]}')

    fig, ax = plt.subplots(figsize=(12, max(3, len(head_keys) * 0.4)))
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    fig.colorbar(im, ax=ax, label='Head Norm')

    # Set tick labels
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    # Show a subset of step labels
    n_ticks = min(10, len(all_steps))
    tick_positions = [int(i * (len(all_steps) - 1) / max(1, n_ticks - 1)) for i in range(n_ticks)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(all_steps[p]) for p in tick_positions], fontsize=8)

    ax.set_xlabel('Step')
    ax.set_title('Head Norm Heatmap')

    # Mark anomalies that have a step and head
    head_idx = {k: i for i, k in enumerate(head_keys)}
    for a in anomalies:
        if 'step' in a and 'head' in a and a['head'] in head_idx:
            s = a['step']
            if s in step_idx:
                color = ANOMALY_COLORS.get(a['type'], 'white')
                ax.plot(step_idx[s], head_idx[a['head']], 'o',
                        color=color, markersize=4, markeredgecolor='white',
                        markeredgewidth=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_entropy_evolution(probe, anomalies, path):
    """Per-head entropy trajectories highlighting sudden specialization events."""
    head_keys = sorted(probe.attention_entropies.keys())
    if not head_keys:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    for key in head_keys:
        entries = probe.attention_entropies[key]
        steps = [s for s, _ in entries]
        entropies = [e for _, e in entries]
        ax.plot(steps, entropies, linewidth=0.8, alpha=0.7,
                label=f'L{key[0]}H{key[1]}')

    # Mark sudden specialization events
    for a in anomalies:
        if a['type'] == 'sudden_specialization' and 'step' in a:
            ax.axvline(x=a['step'], color=ANOMALY_COLORS['sudden_specialization'],
                       linestyle='--', alpha=0.6, linewidth=0.8)
            ax.annotate(f'L{a["head"][0]}H{a["head"][1]}',
                        xy=(a['step'], a['entropy_after']),
                        fontsize=7, color=ANOMALY_COLORS['sudden_specialization'])

    ax.set_xlabel('Step')
    ax.set_ylabel('Attention Entropy')
    ax.set_title('Attention Entropy Evolution')
    ax.legend(fontsize=7, ncol=4, loc='upper right')

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_anomaly_timeline(anomalies, total_steps, path):
    """All detected anomalies on a single timeline, color-coded by type."""
    if not anomalies:
        # Still produce an empty plot
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.set_xlim(0, total_steps)
        ax.set_title('Anomaly Timeline (none detected)')
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return

    # Group by type for y-axis lanes
    types = list(ANOMALY_COLORS.keys())
    type_idx = {t: i for i, t in enumerate(types)}

    fig, ax = plt.subplots(figsize=(12, max(2, len(types) * 0.5 + 1)))

    for a in anomalies:
        t = a['type']
        step = a.get('step', 0)
        y = type_idx.get(t, 0)
        color = ANOMALY_COLORS.get(t, '#999999')
        ax.scatter(step, y, color=color, s=30, zorder=3, edgecolors='white',
                   linewidth=0.5)

    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([t.replace('_', ' ') for t in types], fontsize=8)
    ax.set_xlim(0, total_steps)
    ax.set_xlabel('Step')
    ax.set_title('Anomaly Timeline')
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_head_contribution_evolution(probe, path):
    """Stacked area plot showing how each head's relative contribution changes."""
    from metrics import head_contribution_evolution

    evolution = head_contribution_evolution(probe)
    if not evolution:
        return

    heads = sorted(evolution.keys())
    all_steps = set()
    for fracs in evolution.values():
        for s, _ in fracs:
            all_steps.add(s)
    all_steps = sorted(all_steps)

    # Build fraction matrix
    frac_at = {}
    for head, fracs in evolution.items():
        for s, f in fracs:
            frac_at[(head, s)] = f

    # Prepare data for stackplot
    step_list = all_steps
    head_data = []
    for head in heads:
        head_data.append([frac_at.get((head, s), 0.0) for s in step_list])

    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [f'L{h[0]}H{h[1]}' for h in heads]

    ax.stackplot(step_list, *head_data, labels=labels, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Contribution Fraction')
    ax.set_title('Head Contribution Evolution')
    ax.legend(fontsize=7, ncol=4, loc='upper right')
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================================
# Experiment 1: Head Freezing Analysis Plots
# ============================================================================

def plot_robustness_curve(curve, dg_regression, path):
    """
    Dual-panel robustness curve: loss vs damage (left) and DG vs damage (right).
    The core visualization for Levin-style morphogenetic robustness.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    levels = [c['damage_level'] for c in curve]
    mean_loss = [c['mean_loss'] for c in curve]
    std_loss = [c['std_loss'] for c in curve]
    mean_dg = [c['mean_dg'] for c in curve]
    std_dg = [c['std_dg'] for c in curve]

    # --- Left panel: Loss vs Damage ---
    ax1.errorbar(levels, mean_loss, yerr=std_loss,
                 fmt='o-', color='#1976D2', capsize=5, capthick=1.5,
                 linewidth=2, markersize=8, markerfacecolor='white',
                 markeredgewidth=2)
    ax1.fill_between(levels,
                     [m - s for m, s in zip(mean_loss, std_loss)],
                     [m + s for m, s in zip(mean_loss, std_loss)],
                     alpha=0.15, color='#1976D2')
    ax1.set_xlabel('Frozen Heads', fontsize=12)
    ax1.set_ylabel('Mean Final Loss', fontsize=12)
    ax1.set_title('Loss Robustness Curve', fontsize=13, fontweight='bold')
    ax1.set_xticks(levels)
    ax1.grid(True, alpha=0.3)

    # Annotate flatness
    loss_range = max(mean_loss) - min(mean_loss)
    ax1.annotate(f'Range: {loss_range:.4f}',
                 xy=(0.95, 0.95), xycoords='axes fraction',
                 ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', alpha=0.8))

    # --- Right panel: DG vs Damage ---
    ax2.errorbar(levels, mean_dg, yerr=std_dg,
                 fmt='s-', color='#E65100', capsize=5, capthick=1.5,
                 linewidth=2, markersize=8, markerfacecolor='white',
                 markeredgewidth=2)
    ax2.fill_between(levels,
                     [m - s for m, s in zip(mean_dg, std_dg)],
                     [m + s for m, s in zip(mean_dg, std_dg)],
                     alpha=0.15, color='#E65100')

    # Regression line
    slope = dg_regression['slope']
    intercept = dg_regression['intercept']
    r_sq = dg_regression['r_squared']
    x_fit = [min(levels), max(levels)]
    y_fit = [slope * x + intercept for x in x_fit]
    ax2.plot(x_fit, y_fit, '--', color='#BF360C', linewidth=1.5, alpha=0.7)
    ax2.annotate(f'slope={slope:.4f}, R\u00b2={r_sq:.3f}',
                 xy=(0.95, 0.05), xycoords='axes fraction',
                 ha='right', va='bottom', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', alpha=0.8))

    ax2.set_xlabel('Frozen Heads', fontsize=12)
    ax2.set_ylabel('Mean DG Index', fontsize=12)
    ax2.set_title('Delayed Gratification vs Damage', fontsize=13, fontweight='bold')
    ax2.set_xticks(levels)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Experiment 1: Head Freezing Robustness',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_trajectory_overlay(trajectories_by_level, path):
    """
    Overlay mean loss trajectories for each damage level with std shading.
    Shows whether learning curves maintain the same shape under damage.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    cmap = plt.cm.viridis
    levels = sorted(trajectories_by_level.keys())
    n_levels = len(levels)

    for idx, level in enumerate(levels):
        trajs = trajectories_by_level[level]
        min_len = min(len(t) for t in trajs)
        trimmed = [t[:min_len] for t in trajs]
        steps = list(range(min_len))

        means = []
        stds = []
        for i in range(min_len):
            vals = [t[i] for t in trimmed]
            m = sum(vals) / len(vals)
            s = (sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
            means.append(m)
            stds.append(s)

        color = cmap(idx / max(1, n_levels - 1))
        ax.plot(steps, means, linewidth=1.5, color=color,
                label=f'{level} frozen', alpha=0.9)
        ax.fill_between(steps,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.1, color=color)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Trajectories by Damage Level',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_head_compensation(head_data_by_level, path):
    """
    Bar chart showing head contribution fraction shifts (start -> end) for
    select damage levels. Visualizes compensatory rerouting.

    head_data_by_level: dict of level -> dict of head_name -> (start_frac, end_frac)
    """
    levels = sorted(head_data_by_level.keys())
    if not levels:
        return

    n_panels = len(levels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, level in zip(axes, levels):
        data = head_data_by_level[level]
        heads = sorted(data.keys())
        starts = [data[h][0] for h in heads]
        ends = [data[h][1] for h in heads]

        x = list(range(len(heads)))
        width = 0.35

        ax.bar([xi - width / 2 for xi in x], starts, width,
               label='Start', color='#90CAF9', edgecolor='#1565C0', linewidth=0.5)
        ax.bar([xi + width / 2 for xi in x], ends, width,
               label='End', color='#EF9A9A', edgecolor='#C62828', linewidth=0.5)

        # Highlight heads that increased significantly
        for i, h in enumerate(heads):
            delta = ends[i] - starts[i]
            if delta > 0.03:
                ax.annotate(f'+{delta:.3f}', xy=(x[i] + width / 2, ends[i]),
                            ha='center', va='bottom', fontsize=7, color='#C62828',
                            fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(heads, fontsize=8, rotation=45, ha='right')
        ax.set_title(f'{level} frozen', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('Contribution Fraction', fontsize=11)
    axes[0].legend(fontsize=9)
    fig.suptitle('Head Contribution Shifts Under Damage',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_dg_episode_structure(episode_stats, path):
    """
    Three-panel figure showing DG episode structure vs damage:
    - Episode count
    - Mean duration
    - Mean DG index

    episode_stats: list of dicts with keys:
        level, num_episodes, mean_duration, mean_dg_index
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    levels = [s['level'] for s in episode_stats]
    counts = [s['num_episodes'] for s in episode_stats]
    durations = [s['mean_duration'] for s in episode_stats]
    dg_indices = [s['mean_dg_index'] for s in episode_stats]

    # Episode count
    ax1.bar(levels, counts, color='#7986CB', edgecolor='#283593', width=0.8)
    ax1.set_xlabel('Frozen Heads', fontsize=11)
    ax1.set_ylabel('Total DG Episodes', fontsize=11)
    ax1.set_title('Episode Count', fontsize=12, fontweight='bold')
    ax1.set_xticks(levels)
    ax1.grid(axis='y', alpha=0.3)

    # Mean duration
    ax2.bar(levels, durations, color='#A5D6A7', edgecolor='#2E7D32', width=0.8)
    ax2.set_xlabel('Frozen Heads', fontsize=11)
    ax2.set_ylabel('Mean Duration (steps)', fontsize=11)
    ax2.set_title('Episode Duration', fontsize=12, fontweight='bold')
    ax2.set_xticks(levels)
    ax2.grid(axis='y', alpha=0.3)

    # Mean DG index
    ax3.bar(levels, dg_indices, color='#FFAB91', edgecolor='#BF360C', width=0.8)
    ax3.set_xlabel('Frozen Heads', fontsize=11)
    ax3.set_ylabel('Mean DG Index', fontsize=11)
    ax3.set_title('Episode Depth', fontsize=12, fontweight='bold')
    ax3.set_xticks(levels)
    ax3.grid(axis='y', alpha=0.3)

    fig.suptitle('DG Episode Structure vs Damage',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_group_comparison(groups, path):
    """
    Grouped bar chart with loss (left axis) and DG index (right axis) for named groups.
    Used by analyze2 (cell_view) and analyze3 (gradient_degradation).

    groups: list of dicts with keys:
        name, mean_loss, std_loss, mean_dg, std_dg
    """
    fig, ax1 = plt.subplots(figsize=(max(8, len(groups) * 2), 5))

    names = [g['name'] for g in groups]
    mean_loss = [g['mean_loss'] for g in groups]
    std_loss = [g['std_loss'] for g in groups]
    mean_dg = [g['mean_dg'] for g in groups]
    std_dg = [g['std_dg'] for g in groups]

    x = list(range(len(names)))
    width = 0.35

    # Loss bars on left axis
    bars1 = ax1.bar([xi - width / 2 for xi in x], mean_loss, width,
                    yerr=std_loss, capsize=4,
                    label='Mean Loss', color='#42A5F5', edgecolor='#1565C0',
                    linewidth=0.5, alpha=0.85)
    ax1.set_xlabel('Group', fontsize=12)
    ax1.set_ylabel('Mean Loss', fontsize=12, color='#1565C0')
    ax1.tick_params(axis='y', labelcolor='#1565C0')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10, rotation=20, ha='right')

    # DG bars on right axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar([xi + width / 2 for xi in x], mean_dg, width,
                    yerr=std_dg, capsize=4,
                    label='Mean DG', color='#FF7043', edgecolor='#BF360C',
                    linewidth=0.5, alpha=0.85)
    ax2.set_ylabel('Mean DG Index', fontsize=12, color='#BF360C')
    ax2.tick_params(axis='y', labelcolor='#BF360C')

    # Combined legend
    lines = [bars1, bars2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10)

    ax1.set_title('Loss and DG Comparison by Group',
                  fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_entropy_comparison(entropy_by_group, path):
    """
    Per-head entropy bar chart comparing groups side-by-side.
    Specific to exp2 where cell-view may change entropy patterns.

    entropy_by_group: dict of group_name -> dict of head_name -> mean_entropy
    """
    group_names = sorted(entropy_by_group.keys())
    if not group_names:
        return

    # Collect all head names across groups
    all_heads = set()
    for g in group_names:
        all_heads.update(entropy_by_group[g].keys())
    heads = sorted(all_heads)

    n_groups = len(group_names)
    width = 0.8 / n_groups
    colors = ['#42A5F5', '#FF7043', '#66BB6A', '#AB47BC', '#FFA726']

    fig, ax = plt.subplots(figsize=(max(10, len(heads) * 1.2), 5))

    for gi, gname in enumerate(group_names):
        vals = [entropy_by_group[gname].get(h, 0.0) for h in heads]
        offset = (gi - (n_groups - 1) / 2) * width
        x = [i + offset for i in range(len(heads))]
        ax.bar(x, vals, width * 0.9,
               label=gname, color=colors[gi % len(colors)],
               edgecolor='white', linewidth=0.5, alpha=0.85)

    ax.set_xticks(range(len(heads)))
    ax.set_xticklabels(heads, fontsize=9, rotation=45, ha='right')
    ax.set_xlabel('Head', fontsize=12)
    ax.set_ylabel('Mean Entropy', fontsize=12)
    ax.set_title('Attention Entropy by Head and Group',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_trajectory_divergence(shape_comparisons, path):
    """
    Dual-panel: divergence and correlation vs damage level.
    Shows how trajectory shapes deviate from baseline as damage increases.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    levels = sorted(shape_comparisons.keys())
    divergences = [shape_comparisons[l]['mean_divergence'] for l in levels]
    correlations = [shape_comparisons[l]['shape_correlation'] for l in levels]
    overlaps = [shape_comparisons[l]['overlap_fraction'] for l in levels]

    # Divergence
    ax1.plot(levels, divergences, 'o-', color='#D32F2F', linewidth=2,
             markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax1.set_xlabel('Frozen Heads', fontsize=11)
    ax1.set_ylabel('Mean Divergence', fontsize=11)
    ax1.set_title('Trajectory Divergence from Baseline',
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(levels)
    ax1.grid(True, alpha=0.3)

    # Correlation + overlap
    ax2.plot(levels, correlations, 's-', color='#1976D2', linewidth=2,
             markersize=8, markerfacecolor='white', markeredgewidth=2,
             label='Shape Correlation')
    ax2.plot(levels, overlaps, '^-', color='#388E3C', linewidth=2,
             markersize=8, markerfacecolor='white', markeredgewidth=2,
             label='Overlap Fraction')
    ax2.set_xlabel('Frozen Heads', fontsize=11)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('Trajectory Shape Preservation',
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(levels)
    ax2.set_ylim(0.9, 1.01)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Trajectory Shape Analysis',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
