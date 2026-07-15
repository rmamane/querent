#!/usr/bin/env python
"""Idempotent data fetcher: CIFAR-100 (torchvision) + CIFAR-100-C (Zenodo).

CIFAR-100-C checksums are fetched from the Zenodo record API and verified —
nothing hardcoded to rot. A ``.download_ok`` marker makes re-runs (vast.ai
onstart replays!) free.

    uv run python scripts/download_data.py --data-dir ./data [--with-c100c]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path

import requests

ZENODO_RECORD = "https://zenodo.org/api/records/3555552"
TAR_NAME = "CIFAR-100-C.tar"


def fetch_cifar100(data_dir: Path) -> None:
    from torchvision.datasets import CIFAR100

    CIFAR100(str(data_dir), train=True, download=True)
    CIFAR100(str(data_dir), train=False, download=True)
    print(f"CIFAR-100 ready in {data_dir}")


def fetch_c100c(data_dir: Path) -> None:
    target = data_dir / "CIFAR-100-C"
    marker = target / ".download_ok"
    if marker.exists():
        print(f"CIFAR-100-C already present ({target}) — skipping")
        return

    meta = requests.get(ZENODO_RECORD, timeout=30).json()
    entry = next(f for f in meta["files"] if f["key"] == TAR_NAME)
    url = entry["links"]["self"]
    algo, _, expected = entry["checksum"].partition(":")
    size = int(entry["size"])
    print(f"downloading {TAR_NAME} ({size / 1e9:.2f} GB) from Zenodo…")

    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = data_dir / TAR_NAME
    h = hashlib.new(algo)
    done = 0
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tar_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
                h.update(chunk)
                done += len(chunk)
                print(f"\r  {done / 1e9:.2f} / {size / 1e9:.2f} GB", end="", flush=True)
    print()
    if h.hexdigest() != expected:
        tar_path.unlink()
        sys.exit(f"checksum mismatch for {TAR_NAME}: got {h.hexdigest()}, want {expected}")

    print("extracting…")
    with tarfile.open(tar_path) as tf:
        tf.extractall(data_dir, filter="data")
    tar_path.unlink()
    marker.touch()
    print(f"CIFAR-100-C ready in {target}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--with-c100c", action="store_true")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    fetch_cifar100(data_dir)
    if args.with_c100c:
        fetch_c100c(data_dir)


if __name__ == "__main__":
    main()
