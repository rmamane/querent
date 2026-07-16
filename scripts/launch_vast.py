#!/usr/bin/env python
"""Thin fleet launcher: phase manifest -> `vastai` CLI (search offers, create
interruptible instances with rendered onstart).

    python scripts/launch_vast.py --manifest configs/manifests/p0.yaml --dry-run
    python scripts/launch_vast.py --manifest ... --skip-existing [--only 'b1*'] [--limit 4]

- ``--skip-existing`` dedupes against wandb run states (finished/running) — the
  safe way to re-fire a whole manifest is exactly this flag; it doubles as the
  resurrection mechanism for preempted-then-destroyed cells.
- Secrets come from the local environment (WANDB_API_KEY), never the manifest.
- Thin by design: no daemon, no state files; state of record = wandb +
  ``vastai show instances``. CLI flags verified against vastai 1.4.

Run IDs are resolved through Hydra composition per cell, so the launcher and
train.py can never disagree about identity.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def _redact(s: str) -> str:
    return re.sub(r"(WANDB_API_KEY|VAST_API_KEY)=\S+", r"\1=***", s or "")


def sh(cmd: list[str]) -> str:
    # Never check=True with raw argv: CalledProcessError reprs the full command,
    # which leaks --env secrets into logs. Redact everything on the error path.
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"command failed ({r.returncode}): {_redact(shlex.join(cmd))}\n"
            f"stderr: {_redact(r.stderr.strip())}\nstdout: {_redact(r.stdout.strip())}"
        )
    return r.stdout


def resolve_run_id(experiment: str, seed: int) -> str:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment}", f"seed={seed}"])
        return str(cfg.run_id)


def wandb_states(project: str, entity: str | None) -> dict[str, str]:
    try:
        import wandb

        api = wandb.Api(timeout=20)
        path = f"{entity or api.default_entity}/{project}"
        return {r.id: r.state for r in api.runs(path)}
    except Exception as e:
        print(f"WARN: could not query wandb ({e}); --skip-existing degraded", file=sys.stderr)
        return {}


def search_offers(v: dict, n: int) -> list[dict]:
    q = (
        f"num_gpus={v['num_gpus']} gpu_name={v['gpu_name']} "
        f"reliability>{v['reliability_min']} inet_down>{v['inet_down_min']} "
        f"disk_space>{v['disk']} dph_total<{v['price_max']} cuda_vers>={v['cuda_min']} "
        f"rentable=true"
    )
    if v.get("verified"):
        q += " verified=true"
    offer_type = v.get("type", "bid")
    raw = sh(["vastai", "search", "offers", q, "--type", offer_type, "-o", "dph_total", "--raw"])
    offers = json.loads(raw)
    return offers[: n * 2]  # headroom for failed creates


def render_onstart(manifest: dict, experiment: str, seed: int, extra: str) -> str:
    tpl = (REPO / "docker" / "onstart.sh.template").read_text()
    for key, val in {
        "REPO_URL": manifest["repo_url"],
        "GIT_SHA": manifest["git_sha"],
        "EXPERIMENT": experiment,
        "SEED": str(seed),
        "EXTRA": extra,
    }.items():
        tpl = tpl.replace("{{" + key + "}}", val)
    return tpl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--only", default=None, help="glob on run_id, e.g. 'p3_b1*'")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--on-demand", action="store_true",
                    help="rent at ask price (no bid preemption) instead of interruptible")
    ap.add_argument("--disk", type=int, default=None,
                    help="override manifest disk GB (e.g. 32 for a lean drill box)")
    ap.add_argument("--price-max", type=float, default=None,
                    help="override manifest price ceiling (on-demand asks run higher)")
    ap.add_argument("--ls", action="store_true", help="list current vast instances and exit")
    args = ap.parse_args()

    if args.ls:
        print(sh(["vastai", "show", "instances"]))
        return

    m = yaml.safe_load(Path(args.manifest).read_text())
    if not m.get("git_sha"):
        m["git_sha"] = sh(["git", "-C", str(REPO), "rev-parse", "HEAD"]).strip()
    if args.disk:
        m["vast"]["disk"] = args.disk
    if args.price_max:
        m["vast"]["price_max"] = args.price_max
    m["vast"]["type"] = "on-demand" if args.on_demand else "bid"

    if "REPLACE_ME" in (m.get("image", "") + m.get("repo_url", "")):
        sys.exit("manifest still has REPLACE_ME placeholders (image/repo_url) — fill them in first")
    if not args.dry_run and "WANDB_API_KEY" not in os.environ:
        sys.exit("WANDB_API_KEY not in environment")

    cells: list[tuple[str, int, str, str]] = []  # (experiment, seed, extra, run_id)
    for cell in m["cells"]:
        for seed in cell.get("seeds", [0]):
            rid = resolve_run_id(cell["experiment"], seed)
            cells.append((cell["experiment"], seed, cell.get("extra", ""), rid))

    if args.only:
        cells = [c for c in cells if fnmatch.fnmatch(c[3], args.only)]
    if args.skip_existing:
        states = wandb_states(m["project"], m.get("entity"))
        before = len(cells)
        cells = [c for c in cells if states.get(c[3]) not in ("finished", "running")]
        print(f"--skip-existing: {before - len(cells)} cells already finished/running")
    if args.limit:
        cells = cells[: args.limit]
    if not cells:
        print("nothing to launch")
        return

    offers = search_offers(m["vast"], len(cells))
    if len(offers) < len(cells):
        print(f"WARN: only {len(offers)} offers for {len(cells)} cells", file=sys.stderr)

    env_parts = [f"-e {k}={os.environ.get(k, '')}" for k in m.get("env_passthrough", ["WANDB_API_KEY"])]
    if args.dry_run:
        env_parts = [f"-e {k}=***" for k in m.get("env_passthrough", ["WANDB_API_KEY"])]
    env_str = " ".join(env_parts)

    on_demand = m["vast"].get("type") == "on-demand"
    for (experiment, seed, extra, rid), offer in zip(cells, offers):
        onstart = render_onstart(m, experiment, seed, extra)
        min_bid = float(offer.get("min_bid", m["vast"]["price_max"]))
        bid = round(min(m["vast"]["price_max"], min_bid * 1.15), 3)
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
            f.write(onstart)
            onstart_path = f.name
        cmd = [
            "vastai", "create", "instance", str(offer["id"]),
            "--image", m["image"],
            "--disk", str(m["vast"]["disk"]),
            "--onstart", onstart_path,
            "--env", env_str,
            "--label", rid,
        ]
        if not on_demand:
            cmd += ["--bid_price", str(bid)]  # interruptible; omit -> on-demand at ask
        price_note = f"ask ${offer['dph_total']:.3f} on-demand" if on_demand \
            else f"${offer['dph_total']:.3f}/h ask, bid ${bid}"
        if args.dry_run:
            print(f"\n# {rid}  (offer {offer['id']}, {price_note})")
            print(shlex.join(cmd))
        else:
            print(f"launching {rid} on offer {offer['id']} ({price_note}) …")
            print(sh(cmd).strip())

    if args.dry_run and cells:
        print("\n# rendered onstart for first cell:\n")
        print(render_onstart(m, cells[0][0], cells[0][1], cells[0][2]))


if __name__ == "__main__":
    main()
