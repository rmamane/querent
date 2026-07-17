# Local MPS screening program (weekend autonomous run)

While the vast.ai marketplace stabilizes, the P0–P5 screening ladder runs
serially on the local Mac (MPS, fp32, 100-epoch recipe — chosen so the whole
breadth of P0→P5 fits the time budget at ~2.5 h/cell; 90 s/epoch measured).

- **Live results:** wandb project [`querent-attn`](https://wandb.ai/raphaelma/querent-attn),
  group `mps`, run ids `mps_{arm}_s0`.
- **[mps_screening.md](mps_screening.md)** — auto-regenerated promotion table
  after every finished cell (paired deltas vs `a0`, consistency, corruption,
  efficiency columns).
- **[analysis.md](analysis.md)** — running research notes: per-phase readings
  of the diagnostics (routing entropy/MI, gate splits, alpha trajectories),
  kill-rule flags, and queue adaptations.
- **[queue.txt](queue.txt)** / **[queue_done.txt](queue_done.txt)** — the cell
  queue (breadth-first over phases) and completion ledger.

Caveats for interpreting these numbers: single seed (s0), fp32-on-MPS
(the GPU fleet will use bf16-CUDA — absolute numbers will shift; paired
deltas are the signal), 100 epochs not 200. The fleet re-run under the frozen
200-epoch recipe remains the confirmatory pass.
