from __future__ import annotations

import os
from dataclasses import dataclass

import torch


def auto_train_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(slots=True)
class SupervisedPretrainConfig:
    enabled: bool = False
    dataset_path: str | None = None
    batch_size: int = 256
    max_steps: int = 0


@dataclass(slots=True)
class RewardConfig:
    rank_reward_weight: float = 1.0
    score_reward_scale: float = 0.00005
    placement_rewards: tuple[float, float, float, float] = (1.0, 0.3, -0.3, -1.0)

    def reward_for_result(self, rank: int, score: int) -> float:
        placement = self.placement_rewards[rank - 1]
        score_term = (score - 25000) * self.score_reward_scale
        return self.rank_reward_weight * placement + score_term


@dataclass(slots=True)
class SelfPlayConfig:
    matches_per_iteration: int = 8
    workers: int = max(1, (os.cpu_count() or 2) - 1)
    temperature: float = 1.0
    seed: int = 0


@dataclass(slots=True)
class OptimizerConfig:
    device: str = auto_train_device()
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.0
    entropy_coef: float = 0.01
    grad_clip_norm: float = 1.0
    minibatch_size: int = 256
    epochs_per_iteration: int = 1


@dataclass(slots=True)
class EvaluationConfig:
    matches: int = 8
    workers: int = max(1, (os.cpu_count() or 2) - 1)
    deterministic: bool = True
