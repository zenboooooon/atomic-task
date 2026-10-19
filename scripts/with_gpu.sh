#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
driver_libs="$repo_dir/.tools/nvidia-59584/root/usr/lib/x86_64-linux-gnu"
if [[ ! -f "$driver_libs/libcuda.so.595.84" ]]; then
  echo "Matching NVIDIA 595.84 user libraries are missing: $driver_libs" >&2
  exit 2
fi
export LD_LIBRARY_PATH="$driver_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$@"
