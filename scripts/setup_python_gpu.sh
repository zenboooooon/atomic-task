#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
scripts/setup_gpu_driver_libs.sh
uv venv --python .tools/python/cpython-3.10.12-linux-x86_64-gnu/bin/python3.10 .venv-gpu
uv pip install --python .venv-gpu/bin/python 'torch==2.10.0' 'torchvision==0.25.0' --index-url https://download.pytorch.org/whl/cu130 --cache-dir .cache/uv
uv pip install --python .venv-gpu/bin/python 'mlagents-envs==1.1.0' 'numpy==1.23.5' 'opencv-python-headless==4.8.1.78' 'pytest>=8,<9' --cache-dir .cache/uv
uv pip install --python .venv-gpu/bin/python -e . --no-deps --cache-dir .cache/uv
scripts/with_gpu.sh .venv-gpu/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print(torch.ones(4, device="cuda").sum().item())'
