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

    @classmethod
    def from_dict(cls, payload: dict) -> "PolicyNetConfig":
        hidden_dims = payload.get("hidden_dims", (256, 256))
        return cls(
            input_dim=int(payload["input_dim"]),
            action_dim=int(payload["action_dim"]),
            hidden_dims=tuple(int(value) for value in hidden_dims),
            dropout=float(payload.get("dropout", 0.0)),
        )

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "action_dim": self.action_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
        }


class PolicyNet(nn.Module):
    """Small MLP policy head intended for easy ONNX export and CPU inference."""

    def __init__(self, config: PolicyNetConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        current_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(p=config.dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, config.action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)

    def masked_logits(self, features: Tensor, legal_action_mask: Tensor) -> Tensor:
        logits = self(features)
        return logits.masked_fill(~legal_action_mask.to(dtype=torch.bool), torch.finfo(logits.dtype).min)