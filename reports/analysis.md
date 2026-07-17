# Screening analysis — running notes

*(Auto-maintained during the autonomous weekend run; newest at top. All claims
cite wandb group `mps`; single seed s0, 100-epoch fp32/MPS local recipe —
paired deltas are the signal, absolute numbers are regime-specific.)*

## P1 — decomposition, first contrast (a1 done; a2 running)

**Baseline `a0`: 61.40%** top-1 best (100 ep). Sanity checks out: mCA 49.9%,
shift±2 agreement 0.769, flip agreement 0.929, q participation ratio ≈ 9/64.

**`a1` (content-gated low-rank, ingredient 1): 61.61% → +0.21 pp paired.**
Below the +0.4 pp promote bar, positive but within single-seed noise. The
mechanism diagnostics are the real story this early:

- **The arm opted in**: α grew 0 → |0.22| at L11 (α starts at exactly 0 by the
  equivalence contract, so any growth is the optimizer choosing the delta).
  Delta ratio settled at ~2.8% of the base query — a genuine but small
  correction, not a takeover.
- **Gates behave as designed**: variation across content (std 0.234) is 4.3×
  the variation across positions (std 0.054). A1 has no positional term, so
  this is exactly the signature a content-only mechanism should show; the
  residual positional spread is image statistics (different content at
  different grid positions), not mechanism leakage.
- Gate mean drifted 0.5 → 0.27: the arm learned to *suppress* most basis
  channels and open selectively — not a degenerate always-on gate. Kill-rule
  check (near-constant gates): **passes** (content std is healthy).
- Small robustness lean: mCA +0.7 pp, shift agreement +0.9 pp, shift4 delta
  less negative; flip agreement −1.2 pp. Nothing decisive at one seed.

**What decides P1**: `a2` (position-only, running now) vs this. If a2 lands
near a1 with high gates_std_positions, position alone carries the signal and
A3's combination becomes the interesting cell; if a2 is flat, the fixed-N
positional hook loses its cheapest argument and B1's positional *prior* has
to carry it.

## Status
- Queue: 2/30 done (`a0` 2h44, `a1` 3h42 — slower than the 2.5h estimate;
  breadth through P5 still lands this weekend, tail cells may not).
- Next up after a2: a3 → b1 → b1_prior → a4_res → a5 → b2 → b2_replace →
  a0_depth13 → b3 → a0_rope.
