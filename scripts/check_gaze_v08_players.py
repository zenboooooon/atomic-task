"""Check six real Unity behaviors, distinct appearances, and look success rules."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

from tps_transfer.build_identity import layout_manifest_sha256
from tps_transfer.primitive_variants import COLOR_TASKS, PALETTES
from tps_transfer.smoke import _rgb_at

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/gaze_v08/preflight"


def target_mask(rgb, variant):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if variant == 0:
        return (r > .25) & (r > 1.4 * g) & (g > 1.5 * b)
    if variant == 1:
        return (g > .25) & (g > 1.5 * r) & (g > 1.5 * b)
    return (r > .1) & (r < .75) & (r > 1.4 * g) & (g > 1.5 * b)


def avatar_mask(rgb, variant):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if variant == 0:
        return (g > 2 * r) & (b > 2 * r) & (g > .3)
    if variant == 1:
        return (b > 2 * r) & (b > 1.5 * g) & (b > .3)
    return (np.min(rgb, axis=-1) > .65) & (np.max(rgb, axis=-1) < 1.15 * np.min(rgb, axis=-1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    executable = ROOT / "unity/Build/TpsPretrainColors.x86_64"
    layout_manifest_sha256(executable, ROOT / "manifests/gaze_v08_layouts.json")
    log = OUT / "colors.player.log"
    env = UnityEnvironment(file_name=str(executable), worker_id=185, seed=0,
                           no_graphics=False, additional_args=["-logFile", str(log),
                           "--tps-task=pretrain", "--tps-split=validation"])
    images, counts, completed = {}, {}, {}
    try:
        env.reset()
        behaviors = {name.split("?", 1)[0].removeprefix("TPS_"): name
                     for name in env.behavior_specs}
        assert set(behaviors) == set(COLOR_TASKS)
        for head, behavior in behaviors.items():
            decisions, terminals = env.get_steps(behavior)
            assert len(decisions) == 6 and len(terminals) == 0
            spec = env.behavior_specs[behavior].action_spec
            assert spec.continuous_size == 4 and spec.discrete_size == 0
            variant = int(head[-1])
            rows = []
            completed[head] = set()
            for row, agent in enumerate(decisions.agent_id):
                rgb = _rgb_at(decisions.obs, row)
                target = target_mask(rgb, variant)
                avatar_pixels = int(avatar_mask(rgb, variant).sum())
                assert avatar_pixels > 20, (head, avatar_pixels)
                if head.startswith("look"):
                    y, x = np.nonzero(target)
                    assert len(x) >= 5, (head, row, "missing target")
                    center = float(x.mean() / 224)
                    assert .08 < center < .18 or .82 < center < .92, (head, center)
                    assert x.min() > 0 and x.max() < 223
                rows.append({"agent": int(agent), "target_pixels": int(target.sum()),
                             "avatar_pixels": avatar_pixels})
                if row == 0:
                    images[head] = rgb
            counts[head] = rows
        focal = 112 / math.tan(math.radians(35))
        for step in range(180):
            for head, behavior in behaviors.items():
                decisions, terminals = env.get_steps(behavior)
                for row, agent in enumerate(terminals.agent_id):
                    if head.startswith("look"):
                        assert not terminals.interrupted[row] and terminals.reward[row] > .5
                        completed[head].add(int(agent))
                action = np.zeros((len(decisions), 4), dtype=np.float32)
                if head.startswith("look"):
                    for row in range(len(decisions)):
                        y, x = np.nonzero(target_mask(_rgb_at(decisions.obs, row), int(head[-1])))
                        assert len(x) >= 5, (head, step, "target lost")
                        angles = [math.degrees(math.atan((float(v.mean()) - 111.5) / focal))
                                  for v in (x, y)]
                        action[row, 2:] = np.clip(.7 * np.array(angles) / 9,
                                                 [-.35, -.25], [.35, .25])
                env.set_actions(behavior, ActionTuple(continuous=action))
            if all(len(completed[head]) == 6 for head in COLOR_TASKS if head.startswith("look")):
                break
            env.step()
        assert all(len(completed[head]) == 6 for head in COLOR_TASKS if head.startswith("look"))
    finally:
        env.close()
    diagnostics = []
    initial_layouts = {head: [] for head in COLOR_TASKS if head.startswith("look")}
    for line in log.read_text(errors="replace").splitlines():
        if "TPS_LOOK_START " in line:
            fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
            if len(initial_layouts[fields["head"]]) < 6:
                initial_layouts[fields["head"]].append(fields["layout"])
        if "TPS_EPISODE" not in line or " task=look " not in line:
            continue
        fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
        if fields["reason"] == "checkpoint":
            continue
        assert fields["reason"] == "success" and int(fields["hold_ticks"]) == 25, fields
        assert float(fields["penalty_look_command"]) == float(fields["penalty_look_movement"]) == 0
        assert fields["head"] in COLOR_TASKS
        diagnostics.append(fields)
    assert set(row["head"] for row in diagnostics) == {h for h in COLOR_TASKS if h.startswith("look")}
    assert all(len(ids) == 6 for ids in initial_layouts.values())
    assert len({tuple(sorted(ids)) for ids in initial_layouts.values()}) == 1, initial_layouts

    treasure = ROOT / "unity/Build/TpsTreasure.x86_64"
    layout_manifest_sha256(treasure, ROOT / "manifests/gaze_v08_layouts.json")
    env = UnityEnvironment(file_name=str(treasure), worker_id=186, seed=0,
                           no_graphics=False, additional_args=["-logFile", str(OUT / "treasure.player.log"),
                           "--tps-task=treasure", "--tps-split=validation"])
    try:
        env.reset()
        (behavior,) = env.behavior_specs
        decisions, _ = env.get_steps(behavior)
        assert len(decisions) == 16
        treasure_frames = []
        for row in range(len(decisions)):
            rgb = _rgb_at(decisions.obs, row)
            r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            assert ((r > 1.8 * b) & (g > 1.8 * b) & (r > .5)).sum() > 20
            magenta_pixels = int(((r > .4) & (b > .4) & (g < .55 * np.minimum(r, b))).sum())
            treasure_frames.append((magenta_pixels, rgb))
        # Treasure begins with the goal outside the view. Scan the camera to
        # capture its appearance without treating the scripted scan as an eval.
        for scan_step in range(90):
            if max(item[0] for item in treasure_frames) > 5:
                break
            action = np.zeros((len(decisions), 4), dtype=np.float32)
            action[:, 2] = .5
            env.set_actions(behavior, ActionTuple(continuous=action))
            env.step()
            decisions, _ = env.get_steps(behavior)
            for row in range(len(decisions)):
                rgb = _rgb_at(decisions.obs, row)
                r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
                magenta_pixels = int(((r > .4) & (b > .4) & (g < .55 * np.minimum(r, b))).sum())
                treasure_frames.append((magenta_pixels, rgb))
        magenta_pixels, images["treasure"] = max(treasure_frames, key=lambda item: item[0])
        assert magenta_pixels > 5
    finally:
        env.close()

    sheet = np.zeros((2 * 280, 4 * 224, 3), dtype=np.uint8)
    for index, head in enumerate(COLOR_TASKS + ("treasure",)):
        row, col = (0, index) if index < 3 else (1, index - 3) if index < 6 else (0, 3)
        x, y = col * 224, row * 280
        rgb = np.rint(images[head] * 255).clip(0, 255).astype(np.uint8)
        sheet[y+56:y+280, x:x+224] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(sheet, head, (x+6, y+20), cv2.FONT_HERSHEY_SIMPLEX, .47, (245,245,245), 1)
        label = PALETTES[int(head[-1])]["label"] if head != "treasure" else "magenta / yellow"
        cv2.putText(sheet, label, (x+6, y+43), cv2.FONT_HERSHEY_SIMPLEX, .43, (220,220,220), 1)
    cv2.imwrite(str(OUT / "color-variants.png"), sheet)
    report = {"palettes": list(PALETTES),
              "layout_manifest_sha256": layout_manifest_sha256(
                  executable, ROOT / "manifests/gaze_v08_layouts.json"),
              "behaviors": counts, "look_scripted_successes_by_head":
              {head: len(ids) for head, ids in completed.items() if head.startswith("look")},
              "look_hold_ticks": 25, "look_movement_penalties": False,
              "initial_look_layouts_match_across_colors": True,
              "treasure_agents": 16, "training_started": False}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
