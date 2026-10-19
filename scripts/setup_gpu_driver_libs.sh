#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
package="$repo_dir/.tools/nvidia-59584/libnvidia-compute-595.deb"
root="$repo_dir/.tools/nvidia-59584/root"
expected="256a06d566091e36ee2cc215edbb290562f3c507361eaec210468e462a8465fc"
url="https://snapshot.ubuntu.com/ubuntu/20260801T000000Z/pool/multiverse/n/nvidia-graphics-drivers-595/libnvidia-compute-595_595.84-0ubuntu0.24.04.1_amd64.deb"

mkdir -p "$(dirname "$package")"
if [[ ! -f "$package" ]]; then
  wget -q -O "$package.part" "$url"
  mv "$package.part" "$package"
fi
actual="$(sha256sum "$package" | cut -d ' ' -f 1)"
if [[ "$actual" != "$expected" ]]; then
  echo "NVIDIA 595.84 package SHA-256 mismatch" >&2
  exit 2
fi
if [[ ! -f "$root/usr/lib/x86_64-linux-gnu/libcuda.so.595.84" ]]; then
  dpkg-deb -x "$package" "$root"
fi
"$repo_dir/scripts/with_gpu.sh" nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
