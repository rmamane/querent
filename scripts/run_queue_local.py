#!/usr/bin/env python
"""Serial local-MPS screening queue: runs cells from reports/queue.txt one at a
time, aggregates after each into reports/mps_screening.md, commits + pushes.

Designed for unattended multi-day operation:
- launch detached under caffeinate:  nohup caffeinate -is uv run python
  scripts/run_queue_local.py >> reports/runner.log 2>&1 &
- state on disk (queue.txt / queue_done.txt); edit queue.txt anytime — the
  runner re-reads it between cells, so the queue is adaptively steerable.
- a failed cell retries once, then is marked FAILED and skipped — the queue
  never stalls. Re-running a finished cell is a no-op (fit_completed guard).
- every line of stdout is the event stream for the session monitor.
"""

from __future__ import annotations

import datetime
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QUEUE = REPO / "reports" / "queue.txt"
DONE = REPO / "reports" / "queue_done.txt"
FIXED = ["phase=mps", "seed=0", "data.num_workers=4", "recipe.epochs=100"]


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def sh(cmd: list[str], **kw) -> int:
    return subprocess.run(cmd, cwd=REPO, **kw).returncode


def read_queue() -> list[str]:
    if not QUEUE.exists():
        return []
    return [ln.strip() for ln in QUEUE.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def read_done() -> set[str]:
    if not DONE.exists():
        return set()
    return {ln.split(" :: ")[0] for ln in DONE.read_text().splitlines() if " :: " in ln}


def mark_done(line: str, rc: int) -> None:
    with open(DONE, "a") as f:
        f.write(f"{line} :: rc={rc} at {datetime.datetime.now().isoformat(timespec='seconds')}\n")


def run_cell(line: str) -> int:
    parts = line.split()
    exp, extra = parts[0], parts[1:]
    safe = exp.replace("/", "_")
    cmd = ["uv", "run", "python", "train.py", f"experiment={exp}", *FIXED, *extra]
    log(f"START {line}")
    with open(f"/tmp/mps_{safe}.log", "w") as lf:
        rc = sh(cmd, stdout=lf, stderr=subprocess.STDOUT)
    return rc


def aggregate_and_push() -> None:
    rc = sh(["uv", "run", "python", "scripts/aggregate_results.py", "--phase", "mps",
             "--baseline", "a0", "--out", "reports/mps_screening.md"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if rc != 0:
        log("WARN aggregate failed (ok before first finished run)")
    sh(["git", "add", "reports"], stdout=subprocess.DEVNULL)
    if sh(["git", "diff", "--cached", "--quiet"]) != 0:
        sh(["git", "commit", "-q", "-m", "reports: auto-update mps screening table"],
           stdout=subprocess.DEVNULL)
        if sh(["git", "push", "-q"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
            log("WARN git push failed (will retry next cell)")


def main() -> None:
    log(f"runner up (fixed overrides: {' '.join(FIXED)})")
    while subprocess.run(["pgrep", "-f", "train.py experiment"],
                         capture_output=True).returncode == 0:
        log("waiting for a pre-existing training process to exit")
        time.sleep(300)

    while True:
        pending = [ln for ln in read_queue() if ln not in read_done()]
        if not pending:
            log("QUEUE_EMPTY — final aggregate")
            aggregate_and_push()
            break
        line = pending[0]
        rc = run_cell(line)
        if rc != 0:
            log(f"RETRY {line} (rc={rc})")
            rc = run_cell(line)
        mark_done(line, rc)
        log(f"{'DONE' if rc == 0 else 'FAILED'} rc={rc} {line} ({len(pending) - 1} left)")
        aggregate_and_push()
    log("runner exiting")


if __name__ == "__main__":
    main()
