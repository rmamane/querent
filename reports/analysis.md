# Screening analysis — running notes

*(Auto-maintained during the autonomous weekend run; newest at top. All claims
cite wandb group `mps`; single seed s0, 100-epoch fp32/MPS local recipe —
paired deltas are the signal, absolute numbers are regime-specific.)*

## P1 wrap — decomposition trio done: content carries it; position is a minor modifier

**`a3` (content+position): 61.86% (+0.46 pp)** — statistically indistinguishable
from a1 (+0.21) and a2's dead-arm +0.52 at one seed; all three sit inside the
noise band. The mechanism evidence, which is noise-immune, is consistent:

- **Depth-selective adoption**: α opened at L0 (|α| 0.23, delta ratio **8.1%**
  — the strongest engagement measured anywhere) and L11 (0.23, 2.6%), but the
  middle of the network opted out (L5: α 0.016, ratio ≈ 0). The first layer
  wants content-adapted queries most — consistent with early layers doing
  texture/local binding where per-token metrics plausibly help.
- **Content still dominates the gates where adopted**: content std 0.33 vs
  positional std 0.09 at L0 (≈3.6:1). The b-term contributes *some* positional
  structure beyond a1's incidental 0.054, but it's a modifier, not a driver.

**P1 provisional verdict** (pending seed-1 pairs): query adaptivity, where the
optimizer wants it at all, comes from *content*; static position-indexed maps
are refused alone (a2) and secondary in combination (a3). The fixed-N
positional hook now rides on B1's routing *prior* (b1_prior vs b1) — exactly
the contrast the headline arm was designed to isolate. Accuracy separation, if
any, needs the 2-seed pairs queued after breadth.

## P1 — a2's paradox: best score, dead mechanism → the noise floor is ~±0.5pp

**`a2` (position-only gating, ingredient 2): 61.92% → +0.52 pp — nominally the
best cell and above the promote bar. Do not believe it.** The diagnostics show
the mechanism never turned on:

- α stayed pinned: |α|max 0.005–0.02 across layers (vs a1's 0.22) — the
  optimizer *refused* the delta.
- delta ratio ≈ 0.0003 (0.03% of the base query) — functionally OFF.
- The positional table barely moved: gates_mean ≈ 0.494 ≈ σ(0), positional
  spread 0.002–0.007 (a1's content spread was 0.234 for comparison).

A functionally-dead arm is accuracy-equivalent to the baseline, so its
+0.52 pp IS a clean estimate of run-to-run noise in this regime (arms consume
different init RNG, so paired seeds align data order but not weights).
**Consequences, applied to the program immediately:**

1. **Single-seed deltas below ~±0.5–0.7 pp are uninterpretable**; the +0.4 pp
   promote gate only means something with ≥2 seeds (as the original plan
   specified). The auto-table's GATE column at seeds=1 is decorative — noted
   here so nobody reads a2's row as a promotion.
2. **Queue reshuffled**: seed-1 repeats of {a0, a1, a2, a3, b1, b1_prior}
   pulled forward, ahead of the depth grid; deep-tail cells postponed.
3. **The α-adoption asymmetry is the real P1 finding so far**, and it's
   noise-immune (α starts at exactly 0 by construction — any growth is an
   optimizer choice): content-gating was *adopted* (α→0.22, gates
   content-differentiated), static positional gating was *refused* (α≈0,
   table undifferentiated). Ingredient 2 alone gets no uptake at this scale;
   the fixed-N hook's remaining hopes are A3's combination (running) and B1's
   positional routing prior.

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
