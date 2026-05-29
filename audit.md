# Audit

Dated log of editorial passes and verification runs. Newest first.

## 2026-05-29 — upgrade pass (Group C): closed the results gap (n=30)

The paper carried ~90 numeric claims across 12 experiments with no committed
result artifacts (`claims_target: none`, empty `output/`). Voice was already
clean. This pass is reproducibility, not prose.

Changes:
- Regenerated all 12 experiments at n=30 (seeded 42–71, deterministic; ~15 min):
  `run.py all` + `run.py all2` → 12 `experiment*.json`; `analyze_stats.py`
  reproduced every headline number **exactly** against FINDINGS — competing
  objectives +24.8% (p=0.0007, d=+0.689), freeze L2-3 +0.2% (p=0.462),
  gradual-vs-sudden p=0.032 (d=−0.412), sign-only +5.0% (p=0.0022), recovery
  p=0.886, transplant p=0.880, head-freezing trajectory freeze 12 d=−1.366.
  No drift. (The `run.py` console "+24.3%" for exp 12 is a raw control-diff, not
  the paired-test metric; `analyze_stats` gives the paper's +24.8%.)
- Added `docs/CLAIM_LEDGER.md` (n=30 claims verified; n=300 confirmations marked
  pending) and set `claims_target: claim-ledger`. `.gitignore` now tracks
  `results/*.json`.
- References checked: no placeholder locators (unlike the pigeonhole paper);
  Kofman, Campitelli & Levin (2025) has a full journal locator; Levin (2026) is
  honestly labelled a blog post. No citation fix needed.

Pending: the n=300 artifacts (`*_n300.json`) regenerate via `uv run run.py n300`
(~hours of CPU). Launched separately; ledger rows 14–17 flip to verified once
committed.

Verification: voice 0 errors; refs OK; claims => claim-ledger present (n=30 rows
verified); build clean; check => PASS.

## 2026-05-29 (later) — n=300 regenerated + reconciled

Ran `run.py n300` (all 12, ~2.5h). Findings:
- Reproduces the paper exactly: competing +26.3%/d=0.531, freeze final-loss all
  p>0.15 + Spearman ρ=−0.0045, recovery ratio 1.0009.
- **Drift on the abstract headline**: gradual-vs-sudden d reproduced as −0.374,
  not the reported −0.227 (same direction/significance, stronger). Per the
  user's call (regeneration is canonical), updated all 8 occurrences in the prose
  −0.227 → −0.374, including the abstract.
- Raw n=300 trajectory JSONs are 150–200 MB each (exceed GitHub's limit) — they
  are gitignored; committed `results/n300_analysis.txt` (8 KB) as the trace.
- Ledger n=300 rows 14–17 flipped to verified.

Outstanding (flagged in CLAIM_LEDGER.md): exp 6 courage/caution effect sizes also
drifted (have the reproduced values); exp 1 freeze-trajectory d's aren't emitted
by the n=300 analysis script and weren't re-verified. A focused per-figure pass
would close these.

Verification: build clean (34pp); voice 0; refs OK; check => PASS.
