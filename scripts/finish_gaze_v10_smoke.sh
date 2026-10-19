#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/python" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p runs/v10_smoke
unity/Build/TpsTreasure.x86_64 -batchmode -logFile "$PWD/runs/v10_smoke/probe_composite.player.log" --tps-task=treasure --tps-split=validation --tps-scripted-probe=jump > runs/v10_smoke/probe_composite.stdout.log 2>&1
.venv-gpu/bin/python -u scripts/evaluate_multi_pilot.py --checkpoint runs/v10_smoke/ppo/checkpoint_0002.pt --layout-manifest manifests/gaze_v10_layouts.json --episodes-per-agent 1 --seed 9000 --worker-id 611 --output runs/v10_smoke/primitive_validation.json > runs/v10_smoke/primitive_validation.log 2>&1
.venv-gpu/bin/python -u scripts/train_downstream_pilot.py --pretrained runs/v10_smoke/ppo/checkpoint_0002.pt --downstream-task treasure --conditions P,S,F --seeds 42 --worker-id 612 --rollout 256 --updates 1 --checkpoint-every 1 --epochs 2 --minibatch 32 --output runs/v10_smoke/downstream > runs/v10_smoke/downstream.log 2>&1
.venv-gpu/bin/python -u scripts/evaluate_multi_pilot.py --checkpoint runs/v10_smoke/downstream/F_seed42/checkpoint_0001.pt --layout-manifest manifests/gaze_v10_layouts.json --episodes-per-agent 1 --seed 9000 --worker-id 613 --output runs/v10_smoke/composite_validation.json > runs/v10_smoke/composite_validation.log 2>&1
.venv-gpu/bin/python scripts/verify_gaze_v10_smoke.py > runs/v10_smoke/verification.log 2>&1
date -Is > runs/v10_smoke/COMPLETE
