from __future__ import annotations

from dataclasses import dataclass

import torch

from bot import BasicMahjongBot
from rust_mjai_bot import to_rank
from train.training_config import RewardConfig


@dataclass(slots=True)
class TrajectoryStep:
    features: list[float]
    legal_actions: list[bool]
    action_index: int
    action_type: str


@dataclass(slots=True)
class EpisodeResult:
    player_id: int
    final_score: int
    rank: int
    reward: float
    steps: list[TrajectoryStep]


class SelfPlayBot(BasicMahjongBot):
    def __init__(
        self,
        *,
        player_id: int,
        policy_model,
        reward_config: RewardConfig,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> None:
        super().__init__(player_id=player_id)
        self._policy_model = policy_model
        self._policy_device = next(policy_model.parameters()).device
        self._reward_config = reward_config
        self._temperature = max(temperature, 1.0e-3)
        self._deterministic = deterministic
        self._last_action_source = "torch"
        self._trajectory: list[TrajectoryStep] = []
        self._episode_result: EpisodeResult | None = None

    def _select_native_action_candidate(self):
        candidates = self._build_action_candidates()
        if not candidates:
            raise RuntimeError("no legal action candidates were produced for torch self-play")

        features, legal_actions = self._build_runtime_features(candidates)
        feature_tensor = torch.tensor(
            features,
            dtype=torch.float32,
            device=self._policy_device,
        ).unsqueeze(0)
        legal_mask = torch.tensor(
            legal_actions,
            dtype=torch.bool,
            device=self._policy_device,
        ).unsqueeze(0)

        with torch.no_grad():
            logits = self._policy_model(feature_tensor)
            masked_logits = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)

            if self._deterministic:
                action_index = int(masked_logits.argmax(dim=-1).item())
            else:
                dist = torch.distributions.Categorical(logits=masked_logits / self._temperature)
                action_index = int(dist.sample().item())

        candidate = candidates[action_index]
        self._validate_native_action_candidate(candidate)
        self._trajectory.append(
            TrajectoryStep(
                features=features,
                legal_actions=legal_actions,
                action_index=action_index,
                action_type=candidate.action_type,
            )
        )
        self._last_action_source = "torch"
        return candidate

    def on_game_end(self, scores: list[int]) -> None:
        rank = to_rank(scores)[self.player_id]
        final_score = int(scores[self.player_id])
        reward = self._reward_config.reward_for_result(rank, final_score)
        self._episode_result = EpisodeResult(
            player_id=self.player_id,
            final_score=final_score,
            rank=rank,
            reward=reward,
            steps=list(self._trajectory),
        )

    def export_episode_result(self) -> EpisodeResult:
        if self._episode_result is None:
            raise RuntimeError("episode result is not available before game end")
        return self._episode_result