"""Map the unchanged six v0.8 rollout groups to two actual task heads."""
from .primitive_variants import COLOR_TASKS, base_task


def share_color_heads(actors, critics):
    if tuple(actors) != COLOR_TASKS or tuple(critics) != COLOR_TASKS:
        raise ValueError('sharing is defined for the six v0.8 color groups')
    # Retain color0's fresh head for each task. Allocate all v0.8 heads before
    # calling this, preserving the initial encoder and post-init RNG state.
    task_heads = ('look', 'approach')
    shared_actors = {task: actors[f'{task}_color0'] for task in task_heads}
    shared_critics = {task: critics[f'{task}_color0'] for task in task_heads}
    mapping = {group: base_task(group) for group in COLOR_TASKS}
    return shared_actors, shared_critics, mapping


def heads_for_groups(heads, mapping):
    if set(heads) != set(mapping.values()):
        raise ValueError('head keys must match the group mapping')
    return {group: heads[head] for group, head in mapping.items()}
