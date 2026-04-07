from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class PolicyNetConfig:
    input_dim: int
    action_dim: int
    hidden_dims: tuple[int, ...] = (256, 256)
    dropout: float = 0.0
    value_hidden_dims: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict) -> "PolicyNetConfig":
        hidden_dims = payload.get("hidden_dims", (256, 256))
        value_hidden_dims = payload.get("value_hidden_dims", ())
        return cls(
            input_dim=int(payload["input_dim"]),
            action_dim=int(payload["action_dim"]),
            hidden_dims=tuple(int(value) for value in hidden_dims),
            dropout=float(payload.get("dropout", 0.0)),
            value_hidden_dims=tuple(int(value) for value in value_hidden_dims),
        )

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "action_dim": self.action_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "value_hidden_dims": list(self.value_hidden_dims),
        }


def _build_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    *,
    dropout: float,
) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        current_dim = hidden_dim
    return nn.Sequential(*layers), current_dim


class PolicyNet(nn.Module):
    """Shared-trunk actor-critic network whose forward path stays deployment-friendly."""

    def __init__(self, config: PolicyNetConfig) -> None:
        super().__init__()
        self.config = config

        self.backbone, backbone_dim = _build_mlp(
            config.input_dim,
            config.hidden_dims,
            dropout=config.dropout,
        )
        self.policy_head = nn.Linear(backbone_dim, config.action_dim)

        value_trunk, value_dim = _build_mlp(
            backbone_dim,
            config.value_hidden_dims,
            dropout=config.dropout,
        )
        self.value_head = nn.Sequential(
            value_trunk,
            nn.Linear(value_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        logits, _ = self.policy_and_value(features)
        return logits

    def policy_and_value(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.backbone(features)
        logits = self.policy_head(hidden)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values

    def value(self, features: Tensor) -> Tensor:
        _, values = self.policy_and_value(features)
        return values

    def masked_logits(self, features: Tensor, legal_action_mask: Tensor) -> Tensor:
        logits = self(features)
        return logits.masked_fill(~legal_action_mask.to(dtype=torch.bool), torch.finfo(logits.dtype).min)