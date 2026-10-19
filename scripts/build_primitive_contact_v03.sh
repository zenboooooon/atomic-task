#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
editor="$repo_dir/.tools/unity-editors/6000.0.84f1/Editor/Unity"
source_manifest=manifests/primitive_contact_v03_layouts.json
streaming=unity/Assets/StreamingAssets/pilot_layouts.json
export PYTHONPATH="$repo_dir/python${PYTHONPATH:+:$PYTHONPATH}"

"$repo_dir/.venv/bin/python" scripts/derive_primitive_contact_layouts.py \
  --source manifests/pilot_layouts.json --output "$source_manifest"
treasure_hash_before=$("$repo_dir/.venv/bin/python" -c \
  'from pathlib import Path; from tps_transfer.build_identity import unity_build_sha256; print(unity_build_sha256(Path("unity/Build/TpsTreasure.x86_64")))')

saved_streaming=$(mktemp)
cp "$streaming" "$saved_streaming"
restore_streaming() {
  cp "$saved_streaming" "$streaming"
  rm -f "$saved_streaming"
}
trap restore_streaming EXIT
cp "$source_manifest" "$streaming"
"$editor" -batchmode -nographics -quit -projectPath "$repo_dir/unity" \
  -executeMethod TpsTransfer.Editor.TpsMultiSceneBuilder.BuildPretrainThreeLinuxPlayer \
  -logFile "$repo_dir/build-primitive-contact-v03.log"
test -x unity/Build/TpsPretrainThree.x86_64
"$repo_dir/.venv/bin/python" -c \
  'from pathlib import Path; from tps_transfer.build_identity import layout_manifest_sha256; print(layout_manifest_sha256(Path("unity/Build/TpsPretrainThree.x86_64"), Path("manifests/primitive_contact_v03_layouts.json")))'
treasure_hash_after=$("$repo_dir/.venv/bin/python" -c \
  'from pathlib import Path; from tps_transfer.build_identity import unity_build_sha256; print(unity_build_sha256(Path("unity/Build/TpsTreasure.x86_64")))')
if [[ "$treasure_hash_before" != "$treasure_hash_after" ]]; then
  echo "Treasure player changed while building source tasks" >&2
  exit 1
fi
printf 'Treasure build unchanged: %s\n' "$treasure_hash_after"
