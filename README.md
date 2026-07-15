# querent

*The querent is the one who asks the question.*

A controlled experimental program on **dynamic query/key projections in Vision
Transformers**: instead of one fixed linear map shared by every token, the
query (and optionally key) projection varies — by token content, by absolute
position (exploiting fixed input size), by competitive routing over a bank of
learned metrics, or by pooled scene context. The scientific question:
**where should query adaptivity come from — content, position, competition, or
context?**

Nine arms, one shared attention code path, zero-init residual gates so every
variant starts exactly at the vanilla baseline, and an α=0 equivalence test
that keeps them honest. See `NOVELTY.md` for the claims ledger and prior art
(TPA, Nexus, MoA/SwitchHead/MoH, GC-ViT).

## Quickstart

```bash
uv sync                 # env (mac: CPU/MPS wheels; linux: cu128 — same lockfile)
make test               # CPU test suite incl. the α=0 ≡ vanilla equivalence gate
make smoke              # 1-epoch local sanity run (wandb offline)
python train.py experiment=p1/a3 seed=0     # a real training cell
```

Training runs on rented GPUs (vast.ai) via a deps-only Docker image; code is
cloned at a pinned SHA in `docker/onstart.sh.template`. Results and diagnostics
land in the `querent` wandb project.

**This repo is public and contains no secrets.** Keys (`WANDB_API_KEY`) exist
only in the local shell and as per-instance vast.ai env vars.
