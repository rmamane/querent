#!/usr/bin/env python
"""Standalone CIFAR-100-C eval for any checkpoint; can attach results to the
original wandb run (used to re-score old checkpoints).

    uv run python scripts/eval_corruptions.py --ckpt ckpts/p1_a3_s0/best.ckpt \
        [--data-dir ./data] [--attach-wandb]
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--attach-wandb", action="store_true",
                    help="log results into the checkpoint's original wandb run")
    args = ap.parse_args()

    from querent.callbacks.corruption import evaluate_corruptions
    from querent.data.cifar100c import is_available
    from querent.training.lit_module import ClassificationLitModule

    if not is_available(args.data_dir):
        raise SystemExit("CIFAR-100-C not found; run scripts/download_data.py --with-c100c")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    module = ClassificationLitModule.load_from_checkpoint(args.ckpt, map_location=device)
    module.to(device)

    results = evaluate_corruptions(module.net, args.data_dir, device, batch_size=args.batch_size)
    for k in sorted(results):
        print(f"{k:40s} {results[k]:.4f}")

    if args.attach_wandb:
        import wandb

        cfg = module.cfg
        run = wandb.init(project=str(cfg.wandb.project), entity=cfg.wandb.entity,
                         id=str(cfg.run_id), resume="allow")
        run.log({f"final/c100c/{k}": v for k, v in results.items()})
        run.summary["final/c100c/mCA"] = results["mCA"]
        run.finish()


if __name__ == "__main__":
    main()
