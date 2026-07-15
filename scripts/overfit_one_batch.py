#!/usr/bin/env python
"""Single-batch trainability check: the arm must memorize one batch, fast.

The single highest-value "is this arm even trainable" check — a dead delta
path (e.g. zero-init generator killing dL/dalpha) fails here immediately.

    uv run python scripts/overfit_one_batch.py --arm a3 [--steps 300] [--acc 0.99]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="a0", help="configs/arm/<arm>.yaml")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--acc", type=float, default=0.99)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"arm={args.arm}", "recipe=overfit", "data=cifar100_smoke",
                f"recipe.batch_size={args.batch_size}",
            ],
        )

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    from querent.data.cifar100 import CIFAR100DataModule
    from querent.diagnostics.stats_api import collect_aux_losses
    from querent.training.lit_module import ClassificationLitModule

    dm = CIFAR100DataModule(cfg)
    dm.prepare_data()
    dm.setup()
    x, y = next(iter(dm.train_dataloader()))
    x, y = x.to(device), y.to(device)

    net = ClassificationLitModule(cfg).net.to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.0)
    loss_fn = torch.nn.CrossEntropyLoss()

    acc = 0.0
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        logits = net(x)
        loss = loss_fn(logits, y)
        aux = collect_aux_losses(net)
        if aux is not None:
            loss = loss + aux
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == args.steps:
            acc = (logits.argmax(-1) == y).float().mean().item()
            print(f"step {step:4d}  loss {loss.item():.4f}  acc {acc:.3f}")
            if acc >= args.acc:
                print(f"PASS: arm {args.arm!r} memorized one batch (acc {acc:.3f} >= {args.acc})")
                return 0

    print(f"FAIL: arm {args.arm!r} stuck at acc {acc:.3f} < {args.acc} after {args.steps} steps")
    return 1


if __name__ == "__main__":
    sys.exit(main())
