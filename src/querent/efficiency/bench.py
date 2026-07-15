"""CUDA-event throughput benchmarking core (used by scripts/bench_throughput.py).

Protocol: 30 warmup iters (covers torch.compile when enabled), then 100 timed
iters bracketed by CUDA events, median of 5 trials; peak memory via
``max_memory_allocated``. CUDA-only by design — throughput on MPS/CPU is not a
quantity this program reports.
"""

from __future__ import annotations

import statistics

import torch


def bench_model(net, *, batch_size: int, img_size: int, num_classes: int,
                train_step: bool, iters: int = 100, warmup: int = 30,
                trials: int = 5) -> dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("bench is CUDA-only")
    device = "cuda"
    net = net.to(device)
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4) if train_step else None

    def one_iter():
        if train_step:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
        else:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                net(x)

    net.train(mode=train_step)
    for _ in range(warmup):
        one_iter()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(trials):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            one_iter()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)  # seconds

    sec = statistics.median(times)
    return {
        "imgs_per_sec": batch_size * iters / sec,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
