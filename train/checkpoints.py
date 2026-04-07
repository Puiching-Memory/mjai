from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch

from train.inference_spec import ACTION_DIM, INPUT_DIM
from train.policy_net import PolicyNet, PolicyNetConfig


def default_policy_config(
    hidden_dims: tuple[int, ...] = (256, 256),
    dropout: float = 0.0,
    value_hidden_dims: tuple[int, ...] = (),
) -> PolicyNetConfig:
    return PolicyNetConfig(
        input_dim=INPUT_DIM,
        action_dim=ACTION_DIM,
        hidden_dims=hidden_dims,
        dropout=dropout,
        value_hidden_dims=value_hidden_dims,
    )


def initialize_checkpoint(
    checkpoint_path: Path,
    *,
    hidden_dims: tuple[int, ...] = (256, 256),
    dropout: float = 0.0,
    value_hidden_dims: tuple[int, ...] = (),
    seed: int = 0,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    config = default_policy_config(
        hidden_dims=hidden_dims,
        dropout=dropout,
        value_hidden_dims=value_hidden_dims,
    )
    model = PolicyNet(config)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "model_type": "async_actor_critic_policy",
        "config": config.to_dict(),
        "model_state_dict": model.state_dict(),
        "step": 0,
        "policy_version": 0,
        "metrics": {},
    }
    torch.save(payload, checkpoint_path)
    return payload


def load_checkpoint_payload(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"expected checkpoint payload to be a dict: {checkpoint_path}")
    return payload


def checkpoint_config_from_payload(payload: dict[str, Any]) -> PolicyNetConfig:
    embedded = payload.get("config")
    if isinstance(embedded, dict):
        return PolicyNetConfig.from_dict(embedded)
    raise ValueError("checkpoint payload does not contain a config")


def build_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[PolicyNet, PolicyNetConfig, dict[str, Any]]:
    payload = load_checkpoint_payload(checkpoint_path, map_location=device)
    config = checkpoint_config_from_payload(payload)
    model = PolicyNet(config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config, payload


def save_checkpoint(
    checkpoint_path: Path,
    *,
    model: PolicyNet,
    config: PolicyNetConfig,
    step: int,
    policy_version: int | None = None,
    optimizer_state_dict: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 2,
        "model_type": "async_actor_critic_policy",
        "config": config.to_dict(),
        "model_state_dict": model.state_dict(),
        "step": step,
        "policy_version": step if policy_version is None else policy_version,
        "metrics": metrics or {},
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = optimizer_state_dict
    torch.save(payload, checkpoint_path)


def copy_checkpoint(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)