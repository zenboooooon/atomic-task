"""Versioned layout IDs and deterministic pilot geometry.

These layouts are generated before training and serialized. Unity must use the
serialized coordinates, not reimplement Python's random-number generator.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Task = Literal["look", "approach", "stop", "avoid", "composite", "treasure"]
Split = Literal["train", "validation", "test"]
TASKS: tuple[Task, ...] = ("look", "approach", "stop", "avoid", "composite", "treasure")
SPLITS: tuple[Split, ...] = ("train", "validation", "test")
ARENA_HALF_WIDTH = 6.0
TREASURE_ARENA_HALF_WIDTH = 9.0
PLAYER_RADIUS = 0.35
OBSTACLE_RADIUS = 0.42


@dataclass(frozen=True)
class Obstacle:
    x: float
    z: float
    radius: float = OBSTACLE_RADIUS
    height: float = 1.0


@dataclass(frozen=True)
class Layout:
    layout_id: str
    task: Task
    split: Split
    player_x: float
    player_z: float
    target_x: float
    target_z: float
    camera_yaw_degrees: float
    initial_speed_mps: float
    obstacles: tuple[Obstacle, ...]

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def _identity(protocol_version: str, seed: int, task: Task, split: Split, index: int) -> tuple[str, int]:
    if not protocol_version or seed < 0 or index < 0 or task not in TASKS or split not in SPLITS:
        raise ValueError("invalid layout identity")
    payload = f"{protocol_version}|{seed}|{task}|{split}|{index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return digest.hex()[:20], int.from_bytes(digest[:8], "big")


def _distance(x0: float, z0: float, x1: float, z1: float) -> float:
    return math.hypot(x1 - x0, z1 - z0)


def _inside(x: float, z: float, clearance: float,
            half_width: float = ARENA_HALF_WIDTH) -> bool:
    limit = half_width - clearance
    return -limit <= x <= limit and -limit <= z <= limit


def _reachable(layout: Layout) -> bool:
    """Grid flood fill with player radius, reaching the task's goal region."""
    spacing = 0.25
    half_width = TREASURE_ARENA_HALF_WIDTH if layout.task == "treasure" else ARENA_HALF_WIDTH
    limit = half_width - PLAYER_RADIUS
    width = int(round(2 * limit / spacing)) + 1

    def position(cell: tuple[int, int]) -> tuple[float, float]:
        return -limit + cell[0] * spacing, -limit + cell[1] * spacing

    def cell(x: float, z: float) -> tuple[int, int]:
        return round((x + limit) / spacing), round((z + limit) / spacing)

    def free(candidate: tuple[int, int]) -> bool:
        if not (0 <= candidate[0] < width and 0 <= candidate[1] < width):
            return False
        x, z = position(candidate)
        return all(
            _distance(x, z, obstacle.x, obstacle.z)
            > PLAYER_RADIUS + obstacle.radius + 0.05
            for obstacle in layout.obstacles
        )

    start = cell(layout.player_x, layout.player_z)
    if not free(start):
        return False
    queue = deque([start])
    visited = {start}
    goal_distance = (0.55 if layout.task in ("approach", "avoid", "treasure") else
                     1.5 if layout.task in ("stop", "composite") else 1.7)
    if layout.task == "look":
        return True
    while queue:
        current = queue.popleft()
        x, z = position(current)
        distance = _distance(x, z, layout.target_x, layout.target_z)
        if (abs(distance - 1.5) <= 0.125 if layout.task in ("stop", "composite")
                else distance <= goal_distance):
            return True
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (current[0] + dx, current[1] + dz)
            if neighbor not in visited and free(neighbor):
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def validate_layout(layout: Layout) -> None:
    if layout.task not in TASKS or layout.split not in SPLITS:
        raise ValueError("unknown task or split")
    half_width = TREASURE_ARENA_HALF_WIDTH if layout.task == "treasure" else ARENA_HALF_WIDTH
    if not _inside(layout.player_x, layout.player_z, PLAYER_RADIUS, half_width):
        raise ValueError("player outside arena")
    if not _inside(layout.target_x, layout.target_z, 0.4, half_width):
        raise ValueError("target outside arena")
    distance = _distance(layout.player_x, layout.player_z, layout.target_x, layout.target_z)
    low, high = {
        "look": (3.0, 6.0),
        "approach": (4.0, 8.0),
        "stop": (2.5, 4.0),
        "avoid": (4.0, 8.0),
        "composite": (4.0, 8.0),
        "treasure": (4.0, 10.0),
    }[layout.task]
    if not low <= distance <= high:
        raise ValueError("target distance outside task range")
    if layout.task == "stop":
        if not 0.5 <= layout.initial_speed_mps <= 2.0:
            raise ValueError("stop initial speed outside range")
    elif layout.initial_speed_mps != 0:
        raise ValueError("only stop may have initial speed")
    if layout.task in ("avoid", "composite", "treasure"):
        if not 1 <= len(layout.obstacles) <= 3:
            raise ValueError("avoid/composite/treasure require one to three obstacles")
    elif layout.obstacles:
        raise ValueError("obstacles are only used for avoid/composite")
    for index, obstacle in enumerate(layout.obstacles):
        if not _inside(obstacle.x, obstacle.z, obstacle.radius, half_width):
            raise ValueError("obstacle outside arena")
        if _distance(obstacle.x, obstacle.z, layout.player_x, layout.player_z) <= obstacle.radius + PLAYER_RADIUS + 0.1:
            raise ValueError("obstacle overlaps player")
        if _distance(obstacle.x, obstacle.z, layout.target_x, layout.target_z) <= obstacle.radius + 0.4:
            raise ValueError("obstacle overlaps target")
        for prior in layout.obstacles[:index]:
            if _distance(obstacle.x, obstacle.z, prior.x, prior.z) <= obstacle.radius + prior.radius + 0.05:
                raise ValueError("obstacles overlap")
    if not _reachable(layout):
        raise ValueError("goal region unreachable")


def generate_layout(
    protocol_version: str, seed: int, task: Task, split: Split, index: int
) -> Layout:
    layout_id, random_seed = _identity(protocol_version, seed, task, split, index)
    if protocol_version == "pilot-v0.2":
        # Keep geometry and obstacle draws paired with v0.1 so the visibility
        # change in look can be diagnosed without shifting other task layouts.
        _, random_seed = _identity("pilot-v0.1", seed, task, split, index)
    elif protocol_version in ("treasure-v0.1", "treasure-v0.2") and task != "treasure":
        # Keep the source-task geometry paired with the established pilot.
        _, random_seed = _identity("pilot-v0.1", seed, task, split, index)
    rng = random.Random(random_seed)
    for _ in range(1000):
        player_x = rng.uniform(-1.0, 1.0)
        player_z = rng.uniform(-1.0, 1.0)
        low, high = {
            "look": (3.0, 6.0),
            "approach": (4.0, 7.5),
            "stop": (2.5, 4.0),
            "avoid": (4.0, 7.5),
            "composite": (4.0, 7.5),
            "treasure": ((6.0, 9.5) if protocol_version == "treasure-v0.2"
                         else (4.0, 7.5)),
        }[task]
        distance = rng.uniform(low + 0.01, high - 0.01)
        angle = rng.uniform(-math.pi, math.pi)
        target_x = player_x + distance * math.cos(angle)
        target_z = player_z + distance * math.sin(angle)
        half_width = TREASURE_ARENA_HALF_WIDTH if task == "treasure" else ARENA_HALF_WIDTH
        if not _inside(target_x, target_z, 0.4, half_width):
            continue

        target_yaw = math.degrees(math.atan2(target_x - player_x, target_z - player_z))
        if task == "look":
            # v0.1 used 120 degrees and often began with no target pixels.
            # v0.2 keeps the marker in the camera's safe viewing region.
            max_offset = 120 if protocol_version == "pilot-v0.1" else 40
            offset = rng.uniform(20, max_offset) * rng.choice((-1, 1))
        elif task == "treasure":
            # Every new treasure layout requires an initial camera search.
            offset = (rng.uniform(85, 120) if protocol_version == "treasure-v0.2"
                      else rng.uniform(10, 20) if index % 2 == 0 else rng.uniform(85, 120))
            offset *= rng.choice((-1, 1))
        else:
            offset = rng.uniform(-12, 12)
        obstacles: list[Obstacle] = []
        if task == "treasure" and protocol_version == "treasure-v0.2":
            count = rng.randint(2, 3)
            lateral_x, lateral_z = -math.sin(angle), math.cos(angle)
            # One obstacle blocks the direct route; the others sit on different
            # sides and at independent depths instead of forming a straight row.
            placements = [(rng.uniform(0.36, 0.58), rng.uniform(-0.2, 0.2))]
            side = rng.choice((-1, 1))
            placements.append((rng.uniform(0.30, 0.78), side * rng.uniform(1.8, 2.7)))
            if count == 3:
                placements.append((rng.uniform(0.30, 0.78), -side * rng.uniform(1.8, 2.7)))
            for fraction, lateral in placements:
                obstacles.append(Obstacle(
                    x=player_x + fraction * (target_x - player_x) + lateral * lateral_x,
                    z=player_z + fraction * (target_z - player_z) + lateral * lateral_z,
                ))
        elif task in ("avoid", "composite", "treasure"):
            count = rng.randint(1, 3)
            lateral_x, lateral_z = -math.sin(angle), math.cos(angle)
            for obstacle_index in range(count):
                fraction = (obstacle_index + 1) / (count + 1)
                lateral = rng.uniform(-0.18, 0.18)
                obstacles.append(
                    Obstacle(
                        x=player_x + fraction * (target_x - player_x) + lateral * lateral_x,
                        z=player_z + fraction * (target_z - player_z) + lateral * lateral_z,
                    )
                )
        layout = Layout(
            layout_id=layout_id,
            task=task,
            split=split,
            player_x=player_x,
            player_z=player_z,
            target_x=target_x,
            target_z=target_z,
            camera_yaw_degrees=target_yaw + offset,
            initial_speed_mps=rng.uniform(0.5, 2.0) if task == "stop" else 0.0,
            obstacles=tuple(obstacles),
        )
        try:
            validate_layout(layout)
        except ValueError:
            continue
        return layout
    raise RuntimeError(f"could not generate a valid {task} layout after 1000 attempts")


def write_layout_manifest(
    path: Path,
    protocol_version: str,
    seed: int,
    counts: dict[Split, int],
) -> str:
    if (set(counts) != set(SPLITS) or counts["train"] < 1
            or counts["validation"] < 1 or counts["test"] < 0):
        raise ValueError("counts must include train and validation; test may be zero")
    layouts = [
        generate_layout(protocol_version, seed, task, split, index).to_json()
        for task in TASKS
        for split in SPLITS
        for index in range(counts[split])
        if task in ("composite", "treasure") or split != "test"
    ]
    ids = [item["layout_id"] for item in layouts]
    if len(ids) != len(set(ids)):
        raise RuntimeError("layout ID collision")
    document = {
        "protocol_version": protocol_version,
        "seed": seed,
        "counts_per_task": counts,
        "layouts": layouts,
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    document["sha256"] = digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    return digest
