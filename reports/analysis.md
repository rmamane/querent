# Screening analysis — running notes

*(Auto-maintained during the autonomous weekend run; newest at top. All claims
cite wandb group `mps`; single seed s0, 100-epoch fp32/MPS local recipe —
paired deltas are the signal, absolute numbers are regime-specific.)*

## P2 — a5 head-remix: first clear-signal cell (+1.17 pp) from the cheapest arm

**`a5_remix` (content-dependent query-space head mixing): 62.57% — +1.17 pp,
the first delta to clear the ±0.5 pp noise floor decisively (2.3×).** mCA
50.87% (ties a4_res for best). Mechanism: adopted uniformly at every depth
(α 0.24–0.33, delta ratio 0.16–0.20 — including the middle layers that
refused A3), at near-zero cost: h² = 9 mixing logits per token, ~1.7 K
params/layer, +0.5% attention MACs.

Law check: A5's init delta is the softmax-uniform head-mean of the per-head
queries (init ratio ≈ 0.5) — the adoption law predicted uptake between A1 (3%)
and TPA (100%), and α landed exactly there. Three interpolation points plus
two dead clusters now on the curve; `b1_gain4` is the interventional test.

Standing caveat: single seed. `a5 seed=1` and `a4_res seed=1` promoted into
the seed-pair block (replacing pairs for the two dead B1 cells — deadness has
already replicated within the family). If a5's pair holds anywhere near
+1 pp, the P6 promotion story starts with the humblest mechanism in the grid.

*Ops note*: the first `b1_gain4` attempt crashed instantly — Hydra struct mode
requires `+arm.attention.kwargs.init_gain=4.0` for a key not in the arm
config; queue line fixed (runner correctly skipped past the failure, cost: 6
seconds).

## P2 — TPA-residual: the strongest adoption yet, and it completes the init-scale curve

**`a4_res` (TPA query factorization, residual): 62.02% (+0.62 pp, best nominal
score; edge of the noise band) and mCA 50.87% (+1.0 pp, best robustness).**
The mechanism was embraced wholesale: α 0.27/0.42/0.75 (L0/L5/L11 — even the
middle layers that refused A3), delta ratio 0.87/0.77/**1.28** — at the last
layer the generated query is *larger* than the static one. TPA didn't
modulate the projection; it substantially replaced it.

**The emerging law**: TPA's generator is the one arm whose init delta matches
the base query's magnitude (its init was scale-derived on purpose), and
adoption now correlates cleanly with init delta scale across every cell run so
far — ~100% init ratio → α 0.75 (a4_res); ~3% → α 0.22 (a1/a3); ~0.03% → dead
(b1 family, a2). Same optimizer, same recipe, same α=0 start. `b1_gain4`
(queued) is the *interventional* test of exactly this curve.

Caveat for the eventual writeup: adoption ≠ accuracy (a4's +0.62 still needs
its seed pair), and TPA pays +22% attention MACs — the param-matched
comparison is mandatory before any claim.

## P3 arbiter — competition exonerated; it's the init scale. Fix implemented, prediction registered.

**`b1_sigmoid`: 61.74% (noise band), mechanism DEAD** — α 0.010–0.017, delta
ratio 0.0, gates uniform. Non-competitive routing at the same /M init scale
was refused exactly like softmax routing. **Verdict: the bank's refusal is an
init-scale bootstrap failure, not a rejection of competition** — the /M
normalization makes the starting delta ~√M× weaker than A1's, and the
α-uptake loop (α-grad ∝ ⟨dL/dq, Δq⟩; payload-grad ∝ α) never ignites from a
doubly-cold start.

**Follow-up implemented on the spot**: `init_gain` knob in `metric_bank`
(residual mode only; α still starts at exactly 0, so the equivalence contract
is untouched — 56 tests re-run green). Queued `b1_gain4`
(init_gain = 4 = √M), which equalizes the init delta with A1's.
**Registered prediction**: if the scale story is right, b1_gain4's α opens to
O(0.1+) and routing entropy drops below uniform; if it stays dead, the routed
bank is structurally disfavored at this scale and the honest headline becomes
the negative result + the A1-style composition that IS adopted.

## P3 opener — the headline metric bank was refused; competition-vs-composition sharpens

**`b1_bank` (softmax routing over 16 rank-1 metric bases, residual): 61.70%
(+0.30 pp — inside noise) and the mechanism is DEAD**: α pinned at
0.013–0.032, delta ratio ≈ 10⁻⁴, routing entropy 2.770–2.772 vs log 16 = 2.773
(uniform), load_max 0.063 ≈ 1/16, position↔latent MI ≈ 0. Perfect load balance
because nothing is being routed at all.

The contrast that makes this interesting rather than merely disappointing:
**A1 — sigmoid gates over shared low-rank bases, nearly the same expressive
family — was adopted (α→0.22); B1's softmax-routed version was refused.**
Two candidate explanations, and the queued grid already separates them:

1. **Competition hypothesis**: softmax couples the bases (raising one gate
   lowers the rest), slowing symmetry-breaking; independent sigmoid gates
   break symmetry channel-by-channel. This is exactly the SwitchHead-imported
   axis the plan flagged.
2. **Init-scale hypothesis**: near-uniform softmax averages 16 random rank-1
   responses (∝1/M each), so the init delta is ~4× weaker than A1's, and the
   α-gradient with it.

`b1_sigmoid` (queued) divides its sigmoid gates by M — SAME small init scale,
but non-competitive. If it's adopted → competition story; if it also stalls →
scale story (and a one-line init fix becomes the follow-up).

**Prediction registered before the fact**: `b1_prior` (running now) adds the
positional prior inside a mechanism whose output is unused — expect another
dead cell (α pinned, prior irrelevant). If it somehow gets adopted, that's a
positional-bootstrap effect worth chasing.

> **Outcome — prediction confirmed** (`mps_b1_prior_s0`): α 0.014, routing
> entropy 2.772 (uniform), MI 1.4e-5, delta ratio 1e-5; top-1 61.91% — a third
> independent draw inside the ±0.5 pp noise band around baseline. The prior
> question stays open only in whatever form survives `b1_sigmoid`
> (now running — the competition-vs-scale arbiter).

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
