#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
exec nice -n 15 .venv-gpu/bin/python -u scripts/watch_gaze_v10_videos.py "$@"
