#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"

uv python install 3.10.12 --install-dir .tools/python --no-bin --cache-dir .cache/uv
uv venv --python .tools/python/cpython-3.10.12-linux-x86_64-gnu/bin/python3.10 .venv
uv pip install --python .venv/bin/python 'torch==2.2.1' 'torchvision==0.17.1' \
  --index-url https://download.pytorch.org/whl/cpu --cache-dir .cache/uv
uv pip install --python .venv/bin/python -e '.[test]' --cache-dir .cache/uv
.venv/bin/python -m pytest -q
