#!/usr/bin/env python
"""Promotion-gate table for a phase: paired-by-seed deltas vs the baseline arm.

Gate (printed with every table so the artifact is self-describing):
  PROMOTE if  mean paired Δtop-1 >= +0.4 pp  AND  worst-seed Δ > 0
          AND throughput ratio >= 0.85 (from the paired bench run, if present).
Incomplete/crashed cells surface as WARN rows — never silently dropped.

    python scripts/aggregate_results.py --phase p1 [--baseline a0] [--out results/p1.md]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

GATE_D_MEAN = 0.4   # pp, paired mean delta
GATE_D_MIN = 0.0    # pp, worst seed must beat baseline
GATE_THPT = 0.85    # paired same-instance ratio

SUMMARY_COLS = {
    "top1": "val/acc1_best",
    "agree2": "final/consist/shift2/agreement",
    "d4": "final/consist/shift4/top1_delta",
    "flip_agree": "final/consist/flip/agreement",
    "mca": "final/c100c/mCA",
    "params": "eff/params",
    "gflops": "eff/gflops_fwd",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--baseline", default="a0")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--project", default="querent")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import wandb

    api = wandb.Api()
    path = f"{args.entity}/{args.project}" if args.entity else args.project
    runs = [r for r in api.runs(path, filters={"group": args.phase}) if r.job_type != "bench"]
    if not runs:
        raise SystemExit(f"no runs in group {args.phase!r}")

    rows = []
    for r in runs:
        row = {
            "arm": r.config.get("cfg", {}).get("arm", {}).get("name", r.job_type),
            "seed": r.config.get("cfg", {}).get("seed", -1),
            "state": r.state,
            "done": bool(r.summary.get("fit_completed", False)),
        }
        for col, key in SUMMARY_COLS.items():
            row[col] = r.summary.get(key)
        rows.append(row)
    df = pd.DataFrame(rows)

    bench: dict[str, float] = {}
    for r in api.runs(path, filters={"group": args.phase, "jobType": "bench"}):
        for key, val in r.summary.items():
            if key.endswith("/thpt_ratio_vs_first"):
                bench[key.split("/")[0]] = float(val)

    base = df[(df.arm == args.baseline) & df.done].set_index("seed")["top1"]
    if base.empty:
        print(f"WARNING: no completed baseline ({args.baseline}) runs — deltas unavailable")

    def per_arm(g: pd.DataFrame) -> pd.Series:
        done = g[g.done]
        paired = done.set_index("seed")["top1"].sub(base, fill_value=None).dropna() * 100 \
            if not base.empty else pd.Series(dtype=float)
        thpt = bench.get(g.name)
        gate = (
            not paired.empty
            and paired.mean() >= GATE_D_MEAN
            and paired.min() > GATE_D_MIN
            and (thpt is None or thpt >= GATE_THPT)
        )
        return pd.Series({
            "seeds_done": int(done.shape[0]),
            "seeds_total": int(g.shape[0]),
            "top1_mean": done.top1.mean(),
            "top1_std": done.top1.std(),
            "d_mean_pp": paired.mean() if not paired.empty else None,
            "d_min_pp": paired.min() if not paired.empty else None,
            "agree2": done.agree2.mean(),
            "d4_pp": done.d4.mean() * 100 if done.d4.notna().any() else None,
            "flip_agree": done.flip_agree.mean(),
            "mCA": done.mca.mean(),
            "params_M": done.params.mean() / 1e6 if done.params.notna().any() else None,
            "gflops": done.gflops.mean(),
            "thpt_ratio": thpt,
            "WARN": "" if done.shape[0] == g.shape[0] else f"{g.shape[0] - done.shape[0]} incomplete",
            "GATE": "PROMOTE" if gate else "",
        })

    table = df.groupby("arm").apply(per_arm, include_groups=False).sort_values(
        "d_mean_pp", ascending=False, na_position="last"
    )

    header = (
        f"# {args.phase} promotion table (baseline={args.baseline})\n\n"
        f"Gate: dmean>={GATE_D_MEAN}pp AND dmin>{GATE_D_MIN}pp AND thpt>={GATE_THPT}\n\n"
    )
    md = header + table.to_markdown(floatfmt=".3f")
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)
        table.to_csv(Path(args.out).with_suffix(".csv"))


if __name__ == "__main__":
    main()
