from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


from bot import BasicMahjongBot
from rust_mjai_bot import to_rank
from train.training_config import RewardConfig


@dataclass(slots=True)
class ActionSelection:
    action_index: int
    logprob: float
    value: float
    policy_version: int


class PolicyClient(Protocol):
    def select_action(
        self,
        features: list[float],
        legal_actions: list[bool],
        *,
        deterministic: bool,
        temperature: float,
    ) -> ActionSelection: ...


@dataclass(slots=True)
class AsyncEpisodeResult:
    player_id: int
    final_score: int
    rank: int
    reward: float
    features: list[list[float]]
    legal_actions: list[list[bool]]
    actions: list[int]
    behavior_logprobs: list[float]
    behavior_values: list[float]
    policy_versions: list[int]
    step_shantens: list[int]


class AsyncSelfPlayBot(BasicMahjongBot):
    def __init__(
        self,
        *,
        player_id: int,
        policy_client: PolicyClient,
        reward_config: RewardConfig,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> None:
        super().__init__(player_id=player_id)
        self._policy_client = policy_client
        self._reward_config = reward_config
        self._temperature = max(temperature, 1.0e-3)
        self._deterministic = deterministic
        self._last_action_source = "async"
        self._feature_rows: list[list[float]] = []
        self._legal_action_rows: list[list[bool]] = []
        self._action_indices: list[int] = []
        self._behavior_logprobs: list[float] = []
        self._behavior_values: list[float] = []
        self._policy_versions: list[int] = []
        self._step_shantens: list[int] = []
        self._episode_result: AsyncEpisodeResult | None = None

    def _select_native_action_candidate(self):
        candidates = self._build_action_candidates()
        if not candidates:
            raise RuntimeError("no legal action candidates were produced for async self-play")

        features, legal_actions = self._build_runtime_features(candidates)
        selection = self._policy_client.select_action(
            features,
            legal_actions,
            deterministic=self._deterministic,
            temperature=self._temperature,
        )

        action_index = int(selection.action_index)
        if action_index < 0 or action_index >= len(candidates):
            raise RuntimeError(f"async policy client returned out-of-range action index: {action_index}")
        if not legal_actions[action_index]:
            raise RuntimeError(f"async policy client selected a masked action: {action_index}")

        candidate = candidates[action_index]
        self._validate_native_action_candidate(candidate)
        self._feature_rows.append(features)
        self._legal_action_rows.append(legal_actions)
        self._action_indices.append(action_index)
        self._behavior_logprobs.append(float(selection.logprob))
        self._behavior_values.append(float(selection.value))
        self._policy_versions.append(int(selection.policy_version))
        self._step_shantens.append(self.shanten)
        self._last_action_source = "async"
        return candidate

    def on_game_end(self, scores: list[int]) -> None:
        rank = to_rank(scores)[self.player_id]
        final_score = int(scores[self.player_id])
        reward = self._reward_config.reward_for_result(rank, final_score)
        self._episode_result = AsyncEpisodeResult(
            player_id=self.player_id,
            final_score=final_score,
            rank=rank,
            reward=reward,
            features=self._feature_rows,
            legal_actions=self._legal_action_rows,
            actions=self._action_indices,
            behavior_logprobs=self._behavior_logprobs,
            behavior_values=self._behavior_values,
            policy_versions=self._policy_versions,
            step_shantens=self._step_shantens,
        )

    def export_episode_result(self) -> AsyncEpisodeResult:
        if self._episode_result is None:
            raise RuntimeError("episode result is not available before game end")
        return self._episode_result