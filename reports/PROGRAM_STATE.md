# PROGRAM STATE — SCREENING COMPLETE (2026-07-30)

**29 cells run; the queue drained itself; no processes or paid instances
remain anywhere.** Full narrative (newest-first) in [analysis.md](analysis.md);
auto table in [mps_screening.md](mps_screening.md); all runs + diagnostics in
wandb `raphaelma/querent-attn`, group `mps`.

## What this project is

Testing whether a Vision Transformer's **queries** (each patch's "question")
should be **dynamic** instead of one fixed learned matrix. Nine mechanisms on
one shared skeleton; each adds its adjustment through a **zero-init learnable
gate α** so training starts exactly at the vanilla baseline (bitwise
equivalence test enforced).

**Codename → idea map**: a0 = baseline · a1/a2/a3 = query adjusted by content
/ position / both · a4 = TPA (published; query built on the fly) · a5 =
re-mix existing heads' queries by content · b1 = routed menu of query-styles
· b2 = pooled scene context steers queries · b3 = query from an inner
attention pass. `_replace` = forced, no residual · `_forced` = α frozen 0.3 ·
`_rope` = on the 2D-RoPE backbone.

## Final results (100-ep fp32/MPS screening; baseline a0 = 61.40/61.79 across seeds)

| Cell | Outcome |
|---|---|
| **a0_rope (2D-RoPE control)** | **+4.2–4.4 pp, replicated — the largest effect in the program** |
| a5_rope (winner stacked on RoPE) | no gain over a0_rope (−0.37, in noise) — **geometry subsumes dynamics** |
| a5 remix | +1.17 / −0.85 across seeds → paired mean +0.16 — headline did not replicate |
| a4_res TPA-residual | +0.62 / −0.56 → +0.03; strongest α-adoption (0.75) yet no mean effect |
| a2 position-only | mechanically DEAD both seeds; +0.44 paired — the noise-floor / gate-reductio arm |
| a1 / a3 content gating | adopted (α≈0.22) yet paired means +0.05 / +0.27 — in noise |
| b1 family (softmax/sigmoid/prior/gain4/m4/top2/vector) | all refused or degenerate; **b1_forced: functions when forced, still no gain** |
| a2_forced | useless even when forced (shift behavior identical to baseline) |
| b2 regional | residual ≈ noise; **b2_replace −0.49 (harmful)** |
| a4_replace (faithful TPA) | **−2.69, worst robustness — replacement is harmful; the residual gate made TPA viable** |
| b3 nested | +0.37 vs depth-matched control (in noise); unique mid-depth adoption profile |
| a0_depth13 | +1 plain layer: +0.26 for +445K params |

## Conclusions (final)

1. **Geometry subsumes dynamics at this scale**: fixed relative rotations
   (RoPE) gain +4.3 pp replicated; no dynamic-query mechanism adds anything on
   top of it, and none replicates on the weak baseline either.
2. **Position belongs in score geometry, not the query projection**: learned
   per-position maps were refused when offered and useless when forced, while
   RoPE dominates. The founding fixed-input-size hook is closed on the query
   side.
3. **Adoption ≠ merit, demonstrated three ways**: refused-but-worthless (fair
   trials), embraced-but-redundant (a5 on RoPE), embraced-but-degenerate
   (b1_vector's uniform routing = constant offset). The α readout measures
   trainability, not value.
4. **The routed-menu negative result is airtight**: refused under five
   variations; every excuse (competition, init scale, gradient dilution)
   experimentally eliminated; functions when forced and still adds nothing.
5. **Replacement is harmful, residual is safe**: every `_replace` cell is at
   or below baseline (TPA −2.7, context −0.5); the zero-init residual is what
   made these mechanisms trainable at all.
6. **Screening protocol lesson**: single-seed deltas < ~2σ (≈1 pp here) are
   noise; the formal promotion gate was passed only by a dead arm at n=2.
   Any future phase needs ≥3 seeds or bigger effects.
7. Softer observations worth follow-up: adopted dynamics add seed *variance*
   without mean; depth division of labor (content mechanisms engage L0,
   nested attention peaks mid-network) → untested depth-heterogeneous hybrid.

## The one open question

**Does any of this flip at ImageNet scale**, where the relative-geometry prior
is weak relative to data? Sharpened to a cheap concrete comparison:
`a0_rope vs a5_rope (± a4_res_rope) at IN-100, 3 seeds, frozen 200-ep bf16
recipe` on the vast.ai fleet (~$20–40 at current 3090 prices; all launch
tooling hardened and ready: `scripts/launch_vast.py --dry-run`). USER GO
REQUIRED — costs money. Cloud spent so far: $0.18; credit $24.50.

## Analysis backlog (checkpoints on disk cover all of it)

- Sign of α per head in a5 (collaboration vs differentiation between heads).
- Cross-head Q/K subspace principal angles, a5-trained vs baseline
  (registered prediction, 2026-07-19).
- Routing/prior heatmap figures for the negative-result section.
- NOVELTY.md: re-run the literature sweep immediately before any writeup.
