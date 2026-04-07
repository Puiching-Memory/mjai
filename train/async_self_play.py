from __future__ import annotations

import multiprocessing as mp
import queue
import shutil
import tempfile
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from rust_mjai_arena import Match
from rust_mjai_engine import InProcessMjaiBotEngine

from train.async_training_bot import ActionSelection, AsyncEpisodeResult, AsyncSelfPlayBot
from train.inference_spec import ACTION_DIM, INPUT_DIM
from train.policy_net import PolicyNet, PolicyNetConfig
from train.profiling import profile_scope
from train.training_config import GAEConfig, RewardConfig


@dataclass(slots=True)
class InferenceSharedState:
    request_features: torch.Tensor
    request_legal_actions: torch.Tensor
    request_temperature: torch.Tensor
    request_deterministic: torch.Tensor
    response_action: torch.Tensor
    response_logprob: torch.Tensor
    response_value: torch.Tensor
    response_policy_version: torch.Tensor


@dataclass(slots=True)
class EpisodeTensorBatch:
    actor_id: int
    match_index: int
    duration_sec: float
    episode_count: int
    decision_count: int
    average_rank: float
    average_score: float
    average_reward: float
    features: torch.Tensor
    legal_actions: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    behavior_logprobs: torch.Tensor
    behavior_values: torch.Tensor
    policy_versions: torch.Tensor

    @property
    def max_policy_version(self) -> int:
        if self.policy_versions.numel() == 0:
            return 0
        return int(self.policy_versions.max().item())


@dataclass(slots=True)
class ActorFailure:
    actor_id: int
    message: str
    traceback: str


def create_shared_inference_state(actor_processes: int) -> InferenceSharedState:
    return InferenceSharedState(
        request_features=torch.zeros((actor_processes, INPUT_DIM), dtype=torch.float32).share_memory_(),
        request_legal_actions=torch.zeros((actor_processes, ACTION_DIM), dtype=torch.bool).share_memory_(),
        request_temperature=torch.ones(actor_processes, dtype=torch.float32).share_memory_(),
        request_deterministic=torch.zeros(actor_processes, dtype=torch.bool).share_memory_(),
        response_action=torch.zeros(actor_processes, dtype=torch.long).share_memory_(),
        response_logprob=torch.zeros(actor_processes, dtype=torch.float32).share_memory_(),
        response_value=torch.zeros(actor_processes, dtype=torch.float32).share_memory_(),
        response_policy_version=torch.zeros(actor_processes, dtype=torch.long).share_memory_(),
    )


class SharedMemoryPolicyClient:
    def __init__(
        self,
        *,
        actor_id: int,
        request_queue: mp.Queue[int],
        response_events: list[Any],
        shared_state: InferenceSharedState,
    ) -> None:
        self.actor_id = actor_id
        self.request_queue = request_queue
        self.response_events = response_events
        self.shared_state = shared_state

    def select_action(
        self,
        features: list[float],
        legal_actions: list[bool],
        *,
        deterministic: bool,
        temperature: float,
    ) -> ActionSelection:
        with profile_scope("actor.pack_request"):
            self.shared_state.request_features[self.actor_id].copy_(
                torch.as_tensor(features, dtype=torch.float32)
            )
            self.shared_state.request_legal_actions[self.actor_id].copy_(
                torch.as_tensor(legal_actions, dtype=torch.bool)
            )
            self.shared_state.request_temperature[self.actor_id] = max(float(temperature), 1.0e-3)
            self.shared_state.request_deterministic[self.actor_id] = bool(deterministic)

        event = self.response_events[self.actor_id]
        with profile_scope("actor.enqueue_request"):
            event.clear()
            self.request_queue.put(self.actor_id)
        with profile_scope("actor.wait_response"):
            event.wait()

        return ActionSelection(
            action_index=int(self.shared_state.response_action[self.actor_id].item()),
            logprob=float(self.shared_state.response_logprob[self.actor_id].item()),
            value=float(self.shared_state.response_value[self.actor_id].item()),
            policy_version=int(self.shared_state.response_policy_version[self.actor_id].item()),
        )


class AsyncInferenceServer:
    def __init__(
        self,
        *,
        config: PolicyNetConfig,
        device: str,
        request_queue: mp.Queue[int],
        response_events: list[Any],
        shared_state: InferenceSharedState,
        max_batch_size: int,
        batch_timeout_ms: float,
    ) -> None:
        self.device = torch.device(device)
        self.request_queue = request_queue
        self.response_events = response_events
        self.shared_state = shared_state
        self.max_batch_size = max(1, int(max_batch_size))
        self.batch_timeout_sec = max(float(batch_timeout_ms), 0.1) / 1000.0
        self.pin_memory = self.device.type == "cuda"

        self.model = PolicyNet(config)
        self.model.to(self.device)
        self.model.eval()

        self._policy_version = 0
        self._stop_event = threading.Event()
        self._weights_queue: queue.SimpleQueue[tuple[int, dict[str, torch.Tensor]]] = queue.SimpleQueue()
        self._fatal_error: str | None = None
        self._thread = threading.Thread(target=self._serve_loop, name="async-inference", daemon=True)

        self._metrics_lock = threading.Lock()
        self._total_requests = 0
        self._total_batches = 0
        self._total_inference_sec = 0.0
        self._max_seen_batch = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)

    def publish(self, state_dict: dict[str, torch.Tensor], policy_version: int) -> None:
        cpu_state_dict = {
            key: value.detach().to("cpu").clone()
            for key, value in state_dict.items()
        }
        self._weights_queue.put((int(policy_version), cpu_state_dict))

    def check_health(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(f"async inference server crashed:\n{self._fatal_error}")
        if not self._thread.is_alive() and not self._stop_event.is_set():
            raise RuntimeError("async inference server thread exited unexpectedly")

    def snapshot_metrics(self) -> dict[str, float | int]:
        with self._metrics_lock:
            avg_batch_size = self._total_requests / max(self._total_batches, 1)
            avg_inference_ms = 1000.0 * self._total_inference_sec / max(self._total_batches, 1)
            return {
                "policy_version": self._policy_version,
                "requests": self._total_requests,
                "batches": self._total_batches,
                "avg_batch_size": avg_batch_size,
                "avg_inference_ms": avg_inference_ms,
                "max_batch_size": self._max_seen_batch,
            }

    def _apply_latest_weights(self) -> None:
        latest: tuple[int, dict[str, torch.Tensor]] | None = None
        while True:
            try:
                latest = self._weights_queue.get_nowait()
            except queue.Empty:
                break

        if latest is None:
            return

        with profile_scope("inference.apply_weights"):
            policy_version, state_dict = latest
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self._policy_version = policy_version

    def _serve_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._apply_latest_weights()

                with profile_scope("inference.wait_request"):
                    try:
                        first_actor_id = self.request_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                with profile_scope("inference.collect_batch"):
                    actor_ids = [int(first_actor_id)]
                    deadline = perf_counter() + self.batch_timeout_sec
                    while len(actor_ids) < self.max_batch_size:
                        timeout = deadline - perf_counter()
                        if timeout <= 0.0:
                            break
                        try:
                            actor_ids.append(int(self.request_queue.get(timeout=timeout)))
                        except queue.Empty:
                            break

                started = perf_counter()
                with profile_scope("inference.run_batch"):
                    self._run_batch(actor_ids)
                duration = perf_counter() - started
                with self._metrics_lock:
                    self._total_requests += len(actor_ids)
                    self._total_batches += 1
                    self._total_inference_sec += duration
                    self._max_seen_batch = max(self._max_seen_batch, len(actor_ids))
        except Exception:
            self._fatal_error = traceback.format_exc()

    def _run_batch(self, actor_ids: list[int]) -> None:
        with profile_scope("inference.read_shared_state"):
            actor_index = torch.tensor(actor_ids, dtype=torch.long)
            features_cpu = self.shared_state.request_features.index_select(0, actor_index).contiguous()
            legal_actions_cpu = self.shared_state.request_legal_actions.index_select(0, actor_index).contiguous()
            temperatures_cpu = self.shared_state.request_temperature.index_select(0, actor_index).contiguous()
            deterministic_cpu = self.shared_state.request_deterministic.index_select(0, actor_index).contiguous()

        if self.pin_memory:
            with profile_scope("inference.pin_memory"):
                features_cpu = features_cpu.pin_memory()
                legal_actions_cpu = legal_actions_cpu.pin_memory()
                temperatures_cpu = temperatures_cpu.pin_memory()
                deterministic_cpu = deterministic_cpu.pin_memory()

        with profile_scope("inference.h2d"):
            features = features_cpu.to(self.device, non_blocking=self.pin_memory)
            legal_actions = legal_actions_cpu.to(self.device, non_blocking=self.pin_memory)
            temperatures = temperatures_cpu.to(self.device, non_blocking=self.pin_memory).clamp(min=1.0e-3)
            deterministic = deterministic_cpu.to(self.device, non_blocking=self.pin_memory)

        with profile_scope("inference.forward"):
            with torch.inference_mode():
                logits, values = self.model.policy_and_value(features)
                masked_logits = logits.masked_fill(~legal_actions, torch.finfo(logits.dtype).min)
                scaled_logits = masked_logits / temperatures.unsqueeze(1)
                sampled_actions = torch.distributions.Categorical(logits=scaled_logits).sample()
                greedy_actions = masked_logits.argmax(dim=-1)
                actions = torch.where(deterministic, greedy_actions, sampled_actions)
                log_probs = torch.log_softmax(scaled_logits, dim=-1)
                selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        with profile_scope("inference.d2h"):
            actions_cpu = actions.to("cpu")
            log_probs_cpu = selected_log_probs.to("cpu")
            values_cpu = values.to("cpu")

        with profile_scope("inference.signal_responses"):
            for offset, actor_id in enumerate(actor_ids):
                self.shared_state.response_action[actor_id] = actions_cpu[offset]
                self.shared_state.response_logprob[actor_id] = log_probs_cpu[offset]
                self.shared_state.response_value[actor_id] = values_cpu[offset]
                self.shared_state.response_policy_version[actor_id] = self._policy_version
                self.response_events[actor_id].set()


def _seed_for_match(base_seed: int, actor_id: int, match_index: int) -> tuple[int, int]:
    stride = actor_id * 1_000_000 + match_index * 2
    return (base_seed + stride + 1, base_seed + stride + 2)


def _compute_shaped_rewards(
    episode: AsyncEpisodeResult,
    gae_config: GAEConfig,
) -> list[float]:
    """Compute potential-based shaped rewards for one episode.

    Uses Φ(s) = -α * shanten(s) so that shanten improvement yields positive reward.
    shaped_r_t = r_t + γ·Φ(s_{t+1}) − Φ(s_t)
    """
    T = len(episode.actions)
    if T == 0:
        return []

    alpha = gae_config.shanten_shaping_weight
    gamma = gae_config.gamma
    rewards: list[float] = []

    for t in range(T):
        real_reward = episode.reward if t == T - 1 else 0.0
        phi_current = -alpha * episode.step_shantens[t]
        if t < T - 1:
            phi_next = -alpha * episode.step_shantens[t + 1]
        else:
            phi_next = 0.0  # terminal state potential
        shaped_r = real_reward + gamma * phi_next - phi_current
        rewards.append(shaped_r)

    return rewards


def _compute_gae(
    shaped_rewards: list[float],
    values: list[float],
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and value-function targets for one episode."""
    T = len(shaped_rewards)
    if T == 0:
        return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.float32)

    advantages = torch.zeros(T, dtype=torch.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0.0
        delta = shaped_rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * gae_lambda * last_gae
        advantages[t] = last_gae
    gae_returns = advantages + torch.as_tensor(values[:T], dtype=torch.float32)
    return advantages, gae_returns


def _pack_episode_batch(
    *,
    actor_id: int,
    match_index: int,
    duration_sec: float,
    episodes: list[AsyncEpisodeResult],
    gae_config: GAEConfig,
) -> EpisodeTensorBatch:
    total_decisions = sum(len(episode.actions) for episode in episodes)

    if total_decisions > 0:
        features = torch.empty((total_decisions, INPUT_DIM), dtype=torch.float32)
        legal_actions = torch.empty((total_decisions, ACTION_DIM), dtype=torch.bool)
        action_tensor = torch.empty((total_decisions,), dtype=torch.long)
        return_tensor = torch.empty((total_decisions,), dtype=torch.float32)
        advantage_tensor = torch.empty((total_decisions,), dtype=torch.float32)
        behavior_logprob_tensor = torch.empty((total_decisions,), dtype=torch.float32)
        behavior_value_tensor = torch.empty((total_decisions,), dtype=torch.float32)
        policy_version_tensor = torch.empty((total_decisions,), dtype=torch.long)

        offset = 0
        for episode in episodes:
            episode_decisions = len(episode.actions)
            if episode_decisions == 0:
                continue

            shaped_rewards = _compute_shaped_rewards(episode, gae_config)
            advantages, gae_returns = _compute_gae(
                shaped_rewards,
                episode.behavior_values,
                gae_config.gamma,
                gae_config.gae_lambda,
            )

            next_offset = offset + episode_decisions
            features[offset:next_offset].copy_(torch.as_tensor(episode.features, dtype=torch.float32))
            legal_actions[offset:next_offset].copy_(
                torch.as_tensor(episode.legal_actions, dtype=torch.bool)
            )
            action_tensor[offset:next_offset].copy_(torch.as_tensor(episode.actions, dtype=torch.long))
            return_tensor[offset:next_offset].copy_(gae_returns)
            advantage_tensor[offset:next_offset].copy_(advantages)
            behavior_logprob_tensor[offset:next_offset].copy_(
                torch.as_tensor(episode.behavior_logprobs, dtype=torch.float32)
            )
            behavior_value_tensor[offset:next_offset].copy_(
                torch.as_tensor(episode.behavior_values, dtype=torch.float32)
            )
            policy_version_tensor[offset:next_offset].copy_(
                torch.as_tensor(episode.policy_versions, dtype=torch.long)
            )
            offset = next_offset
    else:
        features = torch.empty((0, INPUT_DIM), dtype=torch.float32)
        legal_actions = torch.empty((0, ACTION_DIM), dtype=torch.bool)
        action_tensor = torch.empty((0,), dtype=torch.long)
        return_tensor = torch.empty((0,), dtype=torch.float32)
        advantage_tensor = torch.empty((0,), dtype=torch.float32)
        behavior_logprob_tensor = torch.empty((0,), dtype=torch.float32)
        behavior_value_tensor = torch.empty((0,), dtype=torch.float32)
        policy_version_tensor = torch.empty((0,), dtype=torch.long)

    episode_count = len(episodes)
    average_rank = sum(float(episode.rank) for episode in episodes) / max(episode_count, 1)
    average_score = sum(float(episode.final_score) for episode in episodes) / max(episode_count, 1)
    average_reward = sum(float(episode.reward) for episode in episodes) / max(episode_count, 1)

    return EpisodeTensorBatch(
        actor_id=actor_id,
        match_index=match_index,
        duration_sec=duration_sec,
        episode_count=episode_count,
        decision_count=int(action_tensor.numel()),
        average_rank=average_rank,
        average_score=average_score,
        average_reward=average_reward,
        features=features,
        legal_actions=legal_actions,
        actions=action_tensor,
        returns=return_tensor,
        advantages=advantage_tensor,
        behavior_logprobs=behavior_logprob_tensor,
        behavior_values=behavior_value_tensor,
        policy_versions=policy_version_tensor,
    )


def actor_process_main(
    *,
    actor_id: int,
    seed: int,
    temperature: float,
    deterministic: bool,
    reward_config: RewardConfig,
    gae_config: GAEConfig,
    request_queue: mp.Queue[int],
    response_events: list[Any],
    shared_state: InferenceSharedState,
    episode_queue: mp.Queue[EpisodeTensorBatch | ActorFailure],
    stop_event: Any,
) -> None:
    try:
        with profile_scope("actor.initialize"):
            policy_client = SharedMemoryPolicyClient(
                actor_id=actor_id,
                request_queue=request_queue,
                response_events=response_events,
                shared_state=shared_state,
            )

        with tempfile.TemporaryDirectory(prefix=f"mjai-actor-{actor_id}-") as log_root:
            log_root_path = Path(log_root)
            match_index = 0
            consecutive_failures = 0
            max_consecutive_failures = 5
            while not stop_event.is_set():
                try:
                    with profile_scope("actor.match_lifecycle"):
                        with profile_scope("actor.match_setup"):
                            match_log_dir = log_root_path / f"match-{match_index:06d}"
                            match_log_dir.mkdir(parents=True, exist_ok=True)
                            started = perf_counter()

                            env = Match(log_dir=str(match_log_dir))
                            bots: list[AsyncSelfPlayBot] = []
                            agents: list[InProcessMjaiBotEngine] = []

                            for seat in range(4):
                                bot = AsyncSelfPlayBot(
                                    player_id=seat,
                                    policy_client=policy_client,
                                    reward_config=reward_config,
                                    temperature=temperature,
                                    deterministic=deterministic,
                                )
                                bots.append(bot)
                                agents.append(InProcessMjaiBotEngine(name=f"actor-{actor_id}-{seat}", player=bot))

                        with profile_scope("actor.match_run"):
                            env.py_match(
                                agents[0],
                                agents[1],
                                agents[2],
                                agents[3],
                                seed_start=_seed_for_match(seed, actor_id, match_index),
                            )

                        with profile_scope("actor.pack_episode"):
                            episodes = [bot.export_episode_result() for bot in bots]
                            batch = _pack_episode_batch(
                                actor_id=actor_id,
                                match_index=match_index,
                                duration_sec=perf_counter() - started,
                                episodes=episodes,
                                gae_config=gae_config,
                            )

                        with profile_scope("actor.enqueue_episode"):
                            episode_queue.put(batch)

                        with profile_scope("actor.cleanup_logs"):
                            shutil.rmtree(match_log_dir, ignore_errors=True)
                        consecutive_failures = 0
                        match_index += 1
                except Exception as match_exc:
                    shutil.rmtree(match_log_dir, ignore_errors=True)
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        raise RuntimeError(
                            f"actor {actor_id} hit {max_consecutive_failures} consecutive match failures, last: {match_exc}"
                        ) from match_exc
                    match_index += 1
    except Exception as exc:
        episode_queue.put(
            ActorFailure(
                actor_id=actor_id,
                message=str(exc),
                traceback=traceback.format_exc(),
            )
        )