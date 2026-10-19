"""Verify rendered look starts across all layouts and both multi-arena builds."""

import json
from pathlib import Path

import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

from record_gaze_v06_preview import target_pixels
from tps_transfer.build_identity import layout_manifest_sha256
from tps_transfer.smoke import _rgb_at

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/gaze_v06/preview"
MANIFEST = ROOT / "manifests/gaze_v06_layouts.json"


def check_look(split: str, expected_count: int) -> dict:
    executable = ROOT / "unity/Build/TpsPretrainTwo.x86_64"
    layout_manifest_sha256(executable, MANIFEST)
    log = OUT / f"check-look-{split}.player.log"
    env = UnityEnvironment(file_name=str(executable), worker_id=161, seed=0,
                           no_graphics=False, additional_args=["-logFile", str(log),
                                                               "--tps-task=pretrain", f"--tps-split={split}"])
    starts = []
    checked_layouts = set()
    ready = set()
    try:
        env.reset()
        behaviors = {name.split("?", 1)[0]: name for name in env.behavior_specs}
        assert set(behaviors) == {"TPS_look", "TPS_approach"}
        for name in behaviors.values():
            decisions, _ = env.get_steps(name)
            assert len(decisions) == 6
            assert env.behavior_specs[name].action_spec.continuous_size == 4
        initial, _ = env.get_steps(behaviors["TPS_look"])
        ready.update(map(int, initial.agent_id))
        for calls in range(400):
            for kind, behavior in behaviors.items():
                decisions, terminals = env.get_steps(behavior)
                action = np.zeros((len(decisions), 4), dtype=np.float32)
                if kind == "TPS_look":
                    ready.update(map(int, terminals.agent_id))
                    for row, agent in enumerate(decisions.agent_id):
                        if int(agent) not in ready:
                            continue
                        rgb = _rgb_at(decisions.obs, row)
                        assert rgb.shape == (224, 224, 3)
                        pixels_y, pixels_x = target_pixels(rgb, False)
                        if len(pixels_x) < 5:
                            raise AssertionError(f"missing initial target: {split} agent={agent}")
                        x = float(pixels_x.mean() / 224)
                        assert .08 < x < .18 or .82 < x < .92, (split, x)
                        assert pixels_x.min() > 0 and pixels_x.max() < 223
                        starts.append(x)
                        ready.remove(int(agent))
                    action[:, 1] = 1.0  # End look episodes quickly through its movement rule.
                env.set_actions(behavior, ActionTuple(continuous=action))
            env.step()
            if calls % 10 == 0:
                for line in log.read_text(errors="replace").splitlines():
                    if "TPS_EPISODE" in line and " task=look " in line:
                        fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
                        checked_layouts.add(fields["layout"])
                if len(checked_layouts) == expected_count:
                    break
        assert len(checked_layouts) == expected_count, (split, len(checked_layouts))
    finally:
        env.close()
    return {"split": split, "unique_layouts_checked": len(checked_layouts),
            "rendered_start_frames": len(starts), "min_center_x": min(starts),
            "max_center_x": max(starts), "all_targets_visible_at_edges": True}


def check_treasure() -> dict:
    exe = ROOT / "unity/Build/TpsTreasure.x86_64"
    layout_manifest_sha256(exe, MANIFEST)
    env = UnityEnvironment(file_name=str(exe), worker_id=162, seed=0, no_graphics=False,
                           additional_args=["-logFile", str(OUT / "check-treasure.player.log"),
                                            "--tps-task=treasure", "--tps-split=validation"])
    try:
        env.reset()
        (behavior,) = env.behavior_specs
        decisions, terminals = env.get_steps(behavior)
        assert len(decisions) == 16 and len(terminals) == 0
        assert env.behavior_specs[behavior].action_spec.continuous_size == 4
        for row in range(16):
            rgb = _rgb_at(decisions.obs, row)
            mask = (rgb[..., 0] > .5) & (rgb[..., 1] > .4) & (rgb[..., 2] < .3)
            assert np.count_nonzero(mask) > 20, "yellow box avatar is not visible"
        return {"treasure_agents": 16, "yellow_avatar_visible_for_all": True}
    finally:
        env.close()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"look": [check_look("train", 128), check_look("validation", 32)],
              "treasure": check_treasure()}
    (OUT / "player_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
