# querent — project instructions

Research codebase testing dynamic Q/K attention variants in fixed-resolution
ViTs. The full experimental plan (arms, phases, budgets, contracts) lives in
the approved plan; the operational summary is here.

## Commands
- `uv sync` — env setup (python 3.11; mac gets CPU/MPS wheels, linux cu128; ONE lockfile)
- `make test` — CPU pytest suite; the α=0 equivalence test is the linchpin
- `make smoke` — 1-epoch MPS/CPU run, wandb offline
- `make overfit ARM=a3` — single-batch trainability check for an arm
- `python train.py experiment=p1/a3 seed=0` — canonical run invocation

## Hard rules
- **PUBLIC REPO + PUBLIC IMAGE — ZERO SECRETS.** Never write keys/tokens into any
  file. `scripts/check_no_secrets.sh` must pass before any push.
- **Never break the α=0 equivalence contract**: `DynamicQKBase.forward` is the
  only attention forward; subclasses implement `prepare`/`delta_q`/`delta_k`;
  the residual is the single expression `base + alpha * delta`; α is the ONLY
  zero-init parameter (generators use trunc_normal 0.02).
- Weight decay must exclude `no_weight_decay()` names (α, QK-norm γ,
  per-position tables, w_temp, latent keys) — WD on α silently kills variants.
- `querent/models/` imports torch only (no timm/lightning/hydra).
- Attention modules own their init; the ViT blanket init never touches `.attn`.
- Every vast.ai launch costs money → per-phase go-ahead from the user, always.
- Token order is row-major (`i = row*gw + col`); fixed input size is asserted.

## Layout
- `src/querent/models/` — ViT + attention variants (registry-selected)
- `src/querent/training/`, `callbacks/`, `diagnostics/` — Lightning harness
- `configs/` — Hydra groups: data / model / recipe / arm / experiment (+ manifests)
- `scripts/` — argparse-only tools (launcher, bench, aggregation, downloads)
- `docker/` — deps-only image + idempotent vast.ai onstart template

## State (update as phases progress)
- Phase: pre-P0. Code complete + all local gates green (145 tests; smoke +
  overfit verified; diagnostics confirmed live on wandb project `querent`,
  entity `raphaelma`). Published to github.com/rmamane/querent (public).
  Image: docker.io/rmamane/querent-deps:cu128-t271-v1.
- Next: cloud bring-up + preemption drill on vast.ai (USER GO REQUIRED — costs
  money; ~$2–4), then P0 pilot → fleet (~$15, gated again).
