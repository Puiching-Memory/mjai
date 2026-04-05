from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

from train.self_play import PolicyMatchSpec, run_match_series, summarize_matches
from train.training_config import RewardConfig


def build_selfplay_match_specs(
    checkpoint_path: Path,
    *,
    matches: int,
    deterministic: bool,
) -> list[tuple[PolicyMatchSpec, PolicyMatchSpec, PolicyMatchSpec, PolicyMatchSpec]]:
    spec = PolicyMatchSpec(
        checkpoint_path=str(checkpoint_path),
        policy_name="candidate",
        deterministic=deterministic,
        temperature=1.0,
    )
    return [(spec, spec, spec, spec) for _ in range(matches)]


def build_candidate_vs_baseline_specs(
    candidate_checkpoint: Path,
    baseline_checkpoint: Path,
    *,
    matches: int,
    deterministic: bool,
) -> list[tuple[PolicyMatchSpec, PolicyMatchSpec, PolicyMatchSpec, PolicyMatchSpec]]:
    specs = []
    for match_index in range(matches):
        candidate_seat = match_index % 4
        seats = []
        for seat in range(4):
            if seat == candidate_seat:
                seats.append(
                    PolicyMatchSpec(
                        checkpoint_path=str(candidate_checkpoint),
                        policy_name="candidate",
                        deterministic=deterministic,
                        temperature=1.0,
                    )
                )
            else:
                seats.append(
                    PolicyMatchSpec(
                        checkpoint_path=str(baseline_checkpoint),
                        policy_name="baseline",
                        deterministic=deterministic,
                        temperature=1.0,
                    )
                )
        specs.append(tuple(seats))
    return specs


def evaluate_policy_paths(
    *,
    candidate_checkpoint: Path,
    baseline_checkpoint: Path | None,
    matches: int,
    workers: int,
    seed: int,
    deterministic: bool,
    reward_config: RewardConfig,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    if baseline_checkpoint is None:
        match_specs = build_selfplay_match_specs(
            candidate_checkpoint,
            matches=matches,
            deterministic=deterministic,
        )
    else:
        match_specs = build_candidate_vs_baseline_specs(
            candidate_checkpoint,
            baseline_checkpoint,
            matches=matches,
            deterministic=deterministic,
        )

    match_results = run_match_series(
        match_specs,
        reward_config=reward_config,
        workers=workers,
        seed=seed,
        progress_callback=progress_callback,
    )
    metrics = summarize_matches(match_results)
    return {
        "matches": match_results,
        "metrics": {
            policy_name: asdict(metrics_)
            for policy_name, metrics_ in metrics.items()
        },
    }