#!/usr/bin/env python
"""Paired throughput bench: all arms back-to-back on the SAME GPU.

vast.ai hosts vary wildly in clocks/PCIe — only same-instance ratios are
admissible for the promotion gate; absolute imgs/s is decorative.

    python scripts/bench_throughput.py --arms a0,a1,a3,b1_bank --train \
        [--batch-size 512] [--log-wandb-phase p1]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="comma-separated arm config names; first = baseline")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--train", action="store_true", help="bench full train step (fwd+bwd+AdamW)")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--log-wandb-phase", default=None)
    args = ap.parse_args()

    import torch

    from querent.efficiency.bench import bench_model
    from querent.efficiency.flops import count_flops, param_breakdown
    from querent.models.vit import create_vit

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    arms = args.arms.split(",")
    results: dict[str, dict] = {}
    for arm in arms:
        with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
            cfg = compose(config_name="config", overrides=[f"arm={arm}"])
        net = create_vit(dict(cfg.model), cfg.arm.attention.name, dict(cfg.arm.attention.kwargs))
        r = param_breakdown(net)
        r.update(count_flops(net, int(cfg.model.img_size)))
        r.update(bench_model(net, batch_size=args.batch_size, img_size=int(cfg.model.img_size),
                             num_classes=int(cfg.model.num_classes),
                             train_step=args.train, iters=args.iters))
        results[arm] = r
        del net
        torch.cuda.empty_cache()
        print(f"{arm:16s} {r['imgs_per_sec']:10.0f} img/s   {r['peak_mem_gb']:.2f} GB   "
              f"{r['eff/gflops_fwd']:.3f} GF   {int(r['eff/params'])/1e6:.2f}M")

    base = results[arms[0]]["imgs_per_sec"]
    for arm in arms:
        results[arm]["thpt_ratio_vs_first"] = results[arm]["imgs_per_sec"] / base
    print("\nratios vs", arms[0])
    for arm in arms:
        print(f"  {arm:16s} {results[arm]['thpt_ratio_vs_first']:.3f}")

    print(json.dumps(results, indent=2))
    if args.log_wandb_phase:
        import wandb

        run = wandb.init(project="querent", group=args.log_wandb_phase,
                         job_type="bench", name=f"bench_{args.log_wandb_phase}")
        run.log({f"{arm}/{k}": v for arm, r in results.items() for k, v in r.items()})
        run.finish()


if __name__ == "__main__":
    main()
