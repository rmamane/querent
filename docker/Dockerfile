# Deps-only image: NO code, NO secrets, NO CMD (vast.ai supplies onstart).
# Code is git-cloned at a pinned SHA by docker/onstart.sh.template.
# The layer cache busts only when pyproject.toml/uv.lock change — bump the
# image tag (querent-deps:cu128-t271-vN) on every lock change, never mutate tags.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates build-essential tmux htop rsync openssh-client \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv python install 3.11

WORKDIR /opt/deps
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# vastai CLI for optional AUTO_DESTROY self-destruct on completion
RUN uv pip install --python /opt/venv vastai

WORKDIR /workspace
