"""Color variants identify independent primitive heads; layouts use base tasks."""

from copy import deepcopy

PROTOCOL_VERSION = "gaze-v0.8-color-variants"
PALETTES = (
    {"id": "color0", "label": "orange / cyan",
     "target_rgb": [1.0, 0.38, 0.03], "avatar_rgb": [0.03, 0.82, 0.9]},
    {"id": "color1", "label": "green / blue",
     "target_rgb": [0.12, 0.9, 0.16], "avatar_rgb": [0.08, 0.22, 0.95]},
    {"id": "color2", "label": "brown / white",
     "target_rgb": [0.45, 0.22, 0.08], "avatar_rgb": [0.94, 0.94, 0.94]},
)
COMPOSITE_COLORS = {"target_rgb": [0.85, 0.08, 0.9],
                    "avatar_rgb": [1.0, 0.82, 0.08]}
COLOR_TASKS = tuple(f"{task}_{palette['id']}"
                    for task in ("look", "approach") for palette in PALETTES)
COLOR_AGENTS_PER_HEAD = 6


def base_task(head: str) -> str:
    if head in COLOR_TASKS:
        return head.rsplit("_", 1)[0]
    if head in ("look", "approach", "stop", "avoid", "jump"):
        return head
    raise ValueError(f"unknown primitive head: {head}")


def task_metadata(tasks: tuple[str, ...]) -> dict:
    result = {}
    for head in tasks:
        task = base_task(head)
        palette = next((p for p in PALETTES if head == f"{task}_{p['id']}"), None)
        result[head] = {"task": task, "variant": palette["id"] if palette else None,
                        "appearance": deepcopy(palette)}
    return result


def appearance_protocol() -> dict:
    return {"protocol_version": PROTOCOL_VERSION,
            "primitive_tasks": ["look", "approach"],
            "variants": deepcopy(list(PALETTES)),
            "heads": task_metadata(COLOR_TASKS),
            "composite": deepcopy(COMPOSITE_COLORS),
            "head_initialization": "independent fresh actor and critic per task and color",
            "shared_encoder": "one shared ViT for all six heads"}


def pretraining_player(tasks: tuple[str, ...]) -> tuple[str, int]:
    if tasks == ("look", "approach", "jump"):
        return "TpsPretrainJump", 6
    if tasks == COLOR_TASKS:
        return "TpsPretrainColors", COLOR_AGENTS_PER_HEAD
    if tasks == ("look", "approach"):
        return "TpsPretrainTwo", 6
    if tasks == ("look", "approach", "avoid"):
        return "TpsPretrainThree", 4
    if tasks == ("look", "approach", "stop", "avoid"):
        return "TpsPretrain", 4
    raise ValueError(f"no player for primitive task set: {tasks}")
