.PHONY: setup test lint smoke overfit bench secrets-check docker-build

setup:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts

# 1-epoch local sanity run (MPS/CPU, fp32, tiny subset, wandb offline)
smoke:
	uv run python train.py experiment=smoke_local

# Single-batch trainability check; ARM=a3 etc. (default a0)
overfit:
	uv run python scripts/overfit_one_batch.py --arm $(or $(ARM),a0)

secrets-check:
	bash scripts/check_no_secrets.sh

docker-build:
	bash docker/build_push.sh --build-only
