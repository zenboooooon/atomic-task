"""Randomly initialized visual-temporal shared model and task-specific heads."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 96
    channels: int = 4
    patch_size: int = 8
    width: int = 128
    spatial_layers: int = 4
    temporal_layers: int = 2
    heads: int = 4
    mlp_ratio: int = 4
    history_images: int = 4
    history_actions: int = 3
    action_size: int = 5


class SpatialViT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.image_size % config.patch_size:
            raise ValueError("image size must be divisible by patch size")
        self.config = config
        patch_count = (config.image_size // config.patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            config.channels, config.width, config.patch_size, stride=config.patch_size
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, config.width))
        self.position = nn.Parameter(torch.zeros(1, patch_count + 1, config.width))
        layer = nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=config.heads,
            dim_feedforward=config.width * config.mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.spatial_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(config.width)
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, images: Tensor) -> Tensor:
        expected = (self.config.channels, self.config.image_size, self.config.image_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(f"expected (N, {expected}), got {tuple(images.shape)}")
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls = self.cls.expand(images.shape[0], -1, -1)
        tokens = torch.cat((cls, patches), dim=1) + self.position
        return self.output_norm(self.blocks(tokens)[:, 0])


class SharedEncoder(nn.Module):
    """Produces z_t from four images and the three actions between them."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.spatial = SpatialViT(config)
        self.action_embed = nn.Linear(config.action_size, config.width)
        self.position = nn.Parameter(torch.zeros(1, 7, config.width))
        self.kind_embed = nn.Embedding(2, config.width)
        layer = nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=config.heads,
            dim_feedforward=config.width * config.mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=config.temporal_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(config.width)
        self.register_buffer(
            "causal_mask", torch.triu(torch.ones(7, 7, dtype=torch.bool), diagonal=1)
        )
        self.register_buffer(
            "kind_indices", torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.long)
        )
        nn.init.normal_(self.position, std=0.02)

    def forward(self, images: Tensor, actions: Tensor, valid: Tensor) -> Tensor:
        batch = images.shape[0]
        image_shape = (
            batch, self.config.history_images, self.config.channels,
            self.config.image_size, self.config.image_size
        )
        if tuple(images.shape) != image_shape:
            raise ValueError(f"expected images {image_shape}, got {tuple(images.shape)}")
        if tuple(actions.shape) != (batch, self.config.history_actions, self.config.action_size):
            raise ValueError("expected actions (B, 3, 5)")
        if tuple(valid.shape) != (batch, 7) or valid.dtype != torch.bool:
            raise ValueError("expected boolean valid mask (B, 7)")
        if not torch.all(valid[:, -1]):
            raise ValueError("the current observation must be valid")

        frame_features = self.spatial(images.reshape(-1, *images.shape[-3:]))
        frame_features = frame_features.reshape(batch, 4, self.config.width)
        tokens = torch.empty(batch, 7, self.config.width, device=images.device, dtype=images.dtype)
        tokens[:, 0::2] = frame_features
        tokens[:, 1::2] = self.action_embed(actions)
        tokens = tokens + self.position + self.kind_embed(self.kind_indices).unsqueeze(0)
        # A fully padded query would have no attention keys on CUDA and can
        # produce NaNs that leak through later temporal layers. Let each padded
        # query attend only to itself; valid queries still exclude every pad.
        attention_mask = self.causal_mask.unsqueeze(0) | (~valid[:, None, :])
        diagonal = torch.arange(7, device=images.device)
        attention_mask[:, diagonal, diagonal] = False
        attention_mask = attention_mask.repeat_interleave(self.config.heads, dim=0)
        encoded = self.temporal(tokens, mask=attention_mask)
        return self.output_norm(encoded[:, -1])


class ActorHead(nn.Module):
    def __init__(self, width: int = 128, zero_output: bool = False) -> None:
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
        )
        self.mean_output = nn.Linear(width, 5)
        self.log_std = nn.Parameter(torch.full((5,), -0.5))
        if zero_output:
            nn.init.zeros_(self.mean_output.weight)
            nn.init.zeros_(self.mean_output.bias)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        mean = self.mean_output(self.hidden(features))
        return mean, self.log_std.expand_as(mean)


class CriticHead(nn.Module):
    def __init__(self, width: int = 128, zero_output: bool = False) -> None:
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
        )
        self.value_output = nn.Linear(width, 1)
        if zero_output:
            nn.init.zeros_(self.value_output.weight)
            nn.init.zeros_(self.value_output.bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.value_output(self.hidden(features)).squeeze(-1)


class TanhGaussian:
    """Keep pre-tanh samples so PPO can recompute exact action log densities."""

    def __init__(self, mean: Tensor, log_std: Tensor) -> None:
        if mean.shape != log_std.shape or mean.shape[-1] != 5:
            raise ValueError("mean and log_std must have matching (..., 5) shapes")
        self.mean = mean
        self.log_std = log_std
        self.base = Normal(mean, log_std.exp())

    def sample(self) -> tuple[Tensor, Tensor]:
        pre_tanh = self.base.rsample()
        return torch.tanh(pre_tanh), pre_tanh

    def deterministic(self) -> Tensor:
        return torch.tanh(self.mean)

    def log_prob(self, pre_tanh: Tensor) -> Tensor:
        if pre_tanh.shape != self.mean.shape:
            raise ValueError("pre-tanh action shape mismatch")
        log_jacobian = 2.0 * (
            torch.log(torch.tensor(2.0, dtype=pre_tanh.dtype, device=pre_tanh.device))
            - pre_tanh - F.softplus(-2.0 * pre_tanh)
        )
        return (self.base.log_prob(pre_tanh) - log_jacobian).sum(dim=-1)

    def entropy_estimate(self) -> Tensor:
        _, pre_tanh = self.sample()
        return -self.log_prob(pre_tanh)


def make_primitive_heads(tasks: tuple[str, ...], width: int,
                         device: torch.device, *, zero_output: bool = False
                         ) -> tuple[dict[str, ActorHead], dict[str, CriticHead]]:
    """Allocate new modules for every task/variant; never copy another head."""
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("primitive head names must be unique and nonempty")
    actors = {task: ActorHead(width, zero_output=zero_output).to(device) for task in tasks}
    critics = {task: CriticHead(width, zero_output=zero_output).to(device) for task in tasks}
    return actors, critics


def make_downstream_heads(seed: int, width: int = 128) -> tuple[ActorHead, CriticHead]:
    """Same hidden-layer initialization for corresponding downstream runs."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = ActorHead(width, zero_output=True)
        critic = CriticHead(width, zero_output=True)
    return actor, critic


def freeze_shared(module: nn.Module) -> None:
    module.requires_grad_(False)
    module.eval()


def module_hash(module: nn.Module) -> str:
    """Hash both parameters and persistent buffers, including their names."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()
