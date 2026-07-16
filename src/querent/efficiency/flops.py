"""FLOPs + parameter accounting.

``torch.utils.flop_counter.FlopCounterMode`` is dispatch-based: it sees the
actual aten ops, including SDPA kernels and whatever einsum soup a routing
variant lowers to. (fvcore was rejected — it silently reports 0 for SDPA
without hand-written handles.) Convention: torch counts one MAC as 2 FLOPs;
we log GMACs = GFLOPs/2 alongside for paper comparability. For
conditional-compute arms the count is input-dependent — we report the count on
a fixed dummy input; treat cross-arm deltas as indicative, params as exact.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode


def count_flops(net: nn.Module, img_size: int) -> dict[str, float]:
    was_training = net.training
    net.eval()
    x = torch.randn(1, 3, img_size, img_size)
    with FlopCounterMode(display=False) as fcm:
        with torch.no_grad():
            net(x)
    net.train(was_training)  # never leave the net in eval mode behind our back
    total = float(fcm.get_total_flops())
    return {"eff/gflops_fwd": total / 1e9, "eff/gmacs_fwd": total / 2e9}


def param_breakdown(net: nn.Module) -> dict[str, float]:
    groups = {"attn": 0, "mlp": 0, "embed_head": 0}
    total = trainable = 0
    for name, p in net.named_parameters():
        n = p.numel()
        total += n
        trainable += n if p.requires_grad else 0
        if ".attn." in name:
            groups["attn"] += n
        elif ".mlp." in name:
            groups["mlp"] += n
        else:
            groups["embed_head"] += n
    out = {"eff/params": float(total), "eff/params_trainable": float(trainable)}
    out.update({f"eff/params_{k}": float(v) for k, v in groups.items()})
    return out
