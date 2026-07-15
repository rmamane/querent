# NOVELTY.md — claims ledger and prior-art map

**Project:** Dynamic query/key projections in fixed-resolution ViTs
**Last verified:** 2026-07-16 (arXiv web sweep; re-run this pass immediately before any submission — see Risks §6)
**Paper framing:** *"Where should query adaptivity come from — content, position, competition, or context?"*

## 1. Claims table

| Claim | Nearest prior art (arXiv) | One-line distinction | Verdict |
|---|---|---|---|
| (a) Static per-position low-rank Q/K projections in fixed-resolution ViTs (arm A2) | RoPE for ViT (2403.13298); Synthesizer-Random (2005.00743); Static Key Attention in Vision (2412.07049); Involution (2103.06255) | Ours *learns* low-rank per-position offsets to W_q/W_k (position-varying metric); RoPE is a fixed non-learned rotation, Synthesizer replaces logits entirely, Static-Key uses one shared static key, Involution generates kernels from content and has no Q/K metric | **ADJACENT** |
| (b) Sub-head-granularity routed metric bases for queries + learned per-position routing prior (arm B1) | Grouped Query Experts (2606.20945); SwitchHead (2312.07987); MoH (2410.11842); TPA (2501.06425) | Prior work routes whole query heads/projections per token by content (GQE, MoH, SwitchHead) or mixes low-rank Q/K bases densely by content (TPA); none route *below* head granularity over metric bases, and none has a per-position routing prior; MoH already covers ViTs at head level | **ADJACENT — highest collision risk** |
| (c) Residual scene-context query modulation with raw K/V (arm B2) | GC-ViT (2206.09959); FocalNet (2203.11926); Focal self-attention (2107.00641); Nexus (2512.03377) | GC-ViT *replaces* local queries with pooled global ones; FocalNet injects context into the query but *deletes* attention; Focal attention pools on the K/V side; Nexus refines Q/K with nested full attention in LLMs — none residually modulate ViT queries while keeping K/V and softmax attention untouched | **ADJACENT** |
| (d) Controlled decomposition: query adaptivity from content vs position vs competition vs context (the study itself) | Empirical Study of Spatial Attention (1904.05873); Static Key Attention (2412.07049); Synthesizer (2005.00743); Learning Spatial Decay (2508.09525) | Factor ablations exist piecemeal (attention-logit terms in 2019 detection models; static-vs-dynamic keys; static-vs-content decay) but no single matched-budget ViT study spans all four sources of *query-side* adaptivity; competition (routing) and context axes are absent from 1904.05873 | **CLEAR** (as a unified study) |

## 2. Verified citations

1. **Tensor Product Attention Is All You Need** (TPA/T6) — arXiv:2501.06425 (Jan 2025). Contextual low-rank tensor-product factorization of Q/K/V; ~10x KV-cache reduction. → our arm A4.
2. **Nexus: Higher-Order Attention Mechanisms in Transformers** — arXiv:2512.03377 (Dec 2025). Q/K are outputs of nested inner attention loops (LLMs, weight-shared). → our arm B3 is the vision-transfer study; its honest depth-matched control is ours, not theirs.
3. **Mixture of Attention Heads** (MoA) — arXiv:2210.05144 (EMNLP 2022). Per-token routing over attention-head experts.
4. **SwitchHead** — arXiv:2312.07987 (NeurIPS 2024). MoE over attention projections (value/output in the final method); non-competitive sigmoid routing worked without regularizers — imported as our sigmoid-vs-softmax axis.
5. **MoH: Multi-Head Attention as Mixture-of-Head Attention** — arXiv:2410.11842 (ICML 2025). Heads as experts, top-k per token, validated on ViTs.
6. **Global Context Vision Transformers** — arXiv:2206.09959 (ICML 2023). Pooled global query tokens replace per-token queries. → our B2 replace-mode control.
7. **Multi-Token Attention** — arXiv:2504.00927 (Meta FAIR, Apr 2025). Convolutions over attention logits across queries/keys/heads (triadic influence in score space; ours is in Q-space — complementary).
8. **Synthesizer** — arXiv:2005.00743 (ICML 2021). Learned synthetic attention logits, incl. static/random variants.
9. **Involution** — arXiv:2103.06255 (CVPR 2021). Position-specific content-generated kernels (CV relative of dynamic projections).
10. **Talking-Heads Attention** — arXiv:2003.02436 (2020). Static learned head-mixing of logits. → our A0 control + contrast for A5.

### Additional prior art surfaced in the 2026-07 sweep
- **Grouped Query Experts** — arXiv:2606.20945 (Jun 2026): per-token router picks k query-head experts inside each GQA group, KV dense (250M LLMs).
- **Static Key Attention in Vision** — arXiv:2412.07049 (Dec 2024).
- **Focal Modulation Networks** — arXiv:2203.11926 (NeurIPS 2022).
- **Empirical Study of Spatial Attention Mechanisms** — arXiv:1904.05873 (ICCV 2019): the canonical factor ablation; cite in paragraph 1 of the study.
- **RoPE for ViT** — arXiv:2403.13298 (ECCV 2024): mandatory direct baseline (our a0_rope arm).
- Contemporaneous 2026 preprints on the Q/K pathway: 2603.13381 (nonlinear/identity query projections), 2604.10791 (pre-projection nonlinear features), 2607.00580 (positional-mechanism ablation in ViTs), 2512.22212 (dynamic value attention).

## 3. Risks and mitigations (mapped to the experiment grid)

1. **Claim (b) is the exposed flank.** GQE (contemporaneous) does query-side-only MoE per token; MoH routes heads in ViTs. Novelty rests on (i) sub-head/basis granularity and (ii) the per-position routing prior. → *Mitigation already in the grid:* P3's `b1_headlr` cell IS the head-granularity (MoH/GQE-style) control; the granularity spectrum vector→rank1→head is the decisive ablation. Cite GQE + MoH prominently.
2. **TPA overlaps (b) conceptually** (dense contextual low-rank mixing vs our competitive routing). → P2 runs TPA as a first-class arm; frame B1 as "routed, not densely mixed."
3. **Nexus threatens (c)/(B3) framing.** → B3 is explicitly a transfer/replication arm with the depth-matched control; B2 is positioned as the cheap residual instance (Q-side, raw K/V).
4. **Claim (a) may be compressed to "learned RoPE generalization."** → a0_rope is a mandatory paired baseline; A2's story must lean on the decomposition (position-only vs content-only vs both), not on standalone novelty.
5. **Claim (d) has the 1904.05873 ancestor** — scope explicitly to the query projection pathway + the two new axes (competition, context).
6. **Field velocity:** four relevant hits are 2026 preprints. Re-run the sweep before submission; keep a "concurrent work" paragraph ready.
7. **No outright COLLISION found** for any claim as stated — but (b)/(c) survive on conjunctions of qualifiers; the unified study (d) is the robust contribution. Lead with it.
