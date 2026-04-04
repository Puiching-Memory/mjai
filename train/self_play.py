from __future__ import annotations

import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mjai.engine import DockerMjaiLogEngine
from mjai.mlibriichi.arena import Match  # type: ignore

from train.checkpoints import build_model_from_checkpoint
from train.training_bot import EpisodeResult, SelfPlayBot
from train.training_config import RewardConfig


@dataclass(frozen=True, slots=True)
class PolicyMatchSpec:
    checkpoint_path: str
    policy_name: str
    deterministic: bool = False
    temperature: float = 1.0


@dataclass(slots=True)
class PolicyMetrics:
    policy_name: str
    games: int
    average_rank: float
    average_score: float
    top1_rate: float
    last_rate: float
    average_reward: float
    average_decisions: float


class InProcessMjaiBotEngine(DockerMjaiLogEngine):
    def end_game(self, game_idx: int, scores: list[int]):
        if hasattr(self.player, "on_game_end"):
            self.player.on_game_end(scores)
        super().end_game(game_idx, scores)


def _run_single_match(
    policy_specs: tuple[PolicyMatchSpec, PolicyMatchSpec, PolicyMatchSpec, PolicyMatchSpec],
    seed: tuple[int, int],
    reward_config: RewardConfig,
) -> dict[str, Any]:
    model_cache: dict[str, Any] = {}

    with tempfile.TemporaryDirectory() as log_dir:
        env = Match(log_dir=log_dir)
        bots: list[SelfPlayBot] = []
        agents: list[InProcessMjaiBotEngine] = []

        for seat, spec in enumerate(policy_specs):
            model = model_cache.get(spec.checkpoint_path)
            if model is None:
                model, _, _ = build_model_from_checkpoint(Path(spec.checkpoint_path), device="cpu")
                model.eval()
                model_cache[spec.checkpoint_path] = model

            bot = SelfPlayBot(
                player_id=seat,
                policy_model=model,
                reward_config=reward_config,
                temperature=spec.temperature,
                deterministic=spec.deterministic,
            )
            bots.append(bot)
            agents.append(InProcessMjaiBotEngine(name=f"{spec.policy_name}-{seat}", player=bot))

        env.py_match(
            agents[0],
            agents[1],
            agents[2],
            agents[3],
            seed_start=seed,
        )

    player_results = []
    training_examples = []
    for seat, (spec, bot) in enumerate(zip(policy_specs, bots)):
        episode: EpisodeResult = bot.export_episode_result()
        player_results.append(
            {
                "policy_name": spec.policy_name,
                "seat": seat,
                "score": episode.final_score,
                "rank": episode.rank,
                "reward": episode.reward,
                "decisions": len(episode.steps),
            }
        )
        for step in episode.steps:
            training_examples.append(
                {
                    "policy_name": spec.policy_name,
                    "features": step.features,
                    "legal_actions": step.legal_actions,
                    "action_index": step.action_index,
                    "reward": episode.reward,
                    "rank": episode.rank,
                    "score": episode.final_score,
                    "action_type": step.action_type,
                }
            )

    return {
        "seed": seed,
        "players": player_results,
        "training_examples": training_examples,
    }


def _seed_for_match(base_seed: int, match_index: int) -> tuple[int, int]:
    return (base_seed + match_index * 2 + 1, base_seed + match_index * 2 + 2)


def _run_single_match_job(job: tuple[tuple[PolicyMatchSpec, PolicyMatchSpec, PolicyMatchSpec, PolicyMatchSpec], tuple[int, int], RewardConfig]) -> dict[str, Any]:
    return _run_single_match(*job)


def run_match_series(
    policy_specs_by_match: list[tuple[PolicyMatchSpec, PolicyMatchSpec, PolicyMatchSpec, PolicyMatchSpec]],
    *,
    reward_config: RewardConfig,
    workers: int,
    seed: int,
) -> list[dict[str, Any]]:
    jobs = [
        (policy_specs, _seed_for_match(seed, match_index), reward_config)
        for match_index, policy_specs in enumerate(policy_specs_by_match)
    ]

    if workers <= 1:
        return [_run_single_match(*job) for job in jobs]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_run_single_match_job, jobs))


def summarize_matches(match_results: list[dict[str, Any]]) -> dict[str, PolicyMetrics]:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for match in match_results:
        for player in match["players"]:
            bucket = buckets[player["policy_name"]]
            bucket["rank"].append(float(player["rank"]))
            bucket["score"].append(float(player["score"]))
            bucket["reward"].append(float(player["reward"]))
            bucket["decisions"].append(float(player["decisions"]))
            bucket["top1"].append(1.0 if player["rank"] == 1 else 0.0)
            bucket["last"].append(1.0 if player["rank"] == 4 else 0.0)

    summary: dict[str, PolicyMetrics] = {}
    for policy_name, bucket in buckets.items():
        games = len(bucket["rank"])
        summary[policy_name] = PolicyMetrics(
            policy_name=policy_name,
            games=games,
            average_rank=sum(bucket["rank"]) / games,
            average_score=sum(bucket["score"]) / games,
            top1_rate=sum(bucket["top1"]) / games,
            last_rate=sum(bucket["last"]) / games,
            average_reward=sum(bucket["reward"]) / games,
            average_decisions=sum(bucket["decisions"]) / games,
        )
    return summary


def flatten_training_examples(
    match_results: list[dict[str, Any]],
    *,
    policy_name: str,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for match in match_results:
        for example in match["training_examples"]:
            if example["policy_name"] == policy_name:
                examples.append(example)
    return examples