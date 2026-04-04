from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.checkpoints import (
    build_model_from_checkpoint,
    copy_checkpoint,
    initialize_checkpoint,
    save_checkpoint,
)
from train.evaluation import evaluate_policy_paths
from train.self_play import PolicyMatchSpec, flatten_training_examples, run_match_series, summarize_matches
from train.training_config import EvaluationConfig, OptimizerConfig, RewardConfig, SelfPlayConfig, SupervisedPretrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal REINFORCE trainer using CPU self-play workers and an optional GPU learner."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/policy.pt"), help="Training checkpoint path.")
    parser.add_argument("--best-checkpoint", type=Path, default=Path("artifacts/policy.best.pt"), help="Best checkpoint path.")
    parser.add_argument("--metrics-jsonl", type=Path, default=Path("artifacts/training_metrics.jsonl"), help="Metrics JSONL output.")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations.")
    parser.add_argument("--matches-per-iteration", type=int, default=8, help="Self-play matches per iteration.")
    parser.add_argument("--workers", type=int, default=1, help="CPU self-play worker count.")
    parser.add_argument("--self-play-temperature", type=float, default=1.0, help="Sampling temperature during self-play.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Learner device: auto/cpu/cuda.")
    parser.add_argument("--lr", type=float, default=1.0e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--minibatch-size", type=int, default=256, help="Learner minibatch size.")
    parser.add_argument("--epochs-per-iteration", type=int, default=1, help="Learner epochs per self-play batch.")
    parser.add_argument("--evaluation-matches", type=int, default=8, help="Candidate vs best evaluation matches.")
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[256, 256], help="Used only when initializing a missing checkpoint.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Used only when initializing a missing checkpoint.")
    parser.add_argument("--enable-supervised-pretrain", action="store_true", help="Reserved interface for future supervised pretraining.")
    parser.add_argument("--supervised-dataset", type=Path, default=None, help="Reserved dataset path for future supervised pretraining.")
    return parser.parse_args()


def select_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def maybe_run_supervised_pretraining(config: SupervisedPretrainConfig) -> None:
    if not config.enabled:
        return
    raise NotImplementedError(
        "supervised pretraining interface is reserved but intentionally disabled until curated match data is available"
    )


def selfplay_match_spec(checkpoint_path: Path, temperature: float) -> tuple[PolicyMatchSpec, PolicyMatchSpec, PolicyMatchSpec, PolicyMatchSpec]:
    spec = PolicyMatchSpec(
        checkpoint_path=str(checkpoint_path),
        policy_name="candidate",
        deterministic=False,
        temperature=temperature,
    )
    return (spec, spec, spec, spec)


def expand_selfplay_specs(
    checkpoint_path: Path,
    *,
    matches: int,
    temperature: float,
) -> list[tuple[PolicyMatchSpec, PolicyMatchSpec, PolicyMatchSpec, PolicyMatchSpec]]:
    spec = selfplay_match_spec(checkpoint_path, temperature)
    return [spec for _ in range(matches)]


def rewards_tensor(examples: list[dict[str, Any]]) -> torch.Tensor:
    rewards = torch.tensor([example["reward"] for example in examples], dtype=torch.float32)
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1.0e-6)


def train_policy_iteration(
    model,
    *,
    examples: list[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: str,
    minibatch_size: int,
    epochs_per_iteration: int,
    entropy_coef: float,
    grad_clip_norm: float,
) -> dict[str, float]:
    features = torch.tensor([example["features"] for example in examples], dtype=torch.float32)
    legal_masks = torch.tensor([example["legal_actions"] for example in examples], dtype=torch.bool)
    actions = torch.tensor([example["action_index"] for example in examples], dtype=torch.long)
    rewards = rewards_tensor(examples)

    features = features.to(device)
    legal_masks = legal_masks.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)

    model.train()
    losses: list[float] = []
    entropies: list[float] = []

    sample_count = features.shape[0]
    for _ in range(epochs_per_iteration):
        indices = torch.randperm(sample_count, device=device)
        for start in range(0, sample_count, minibatch_size):
            batch_indices = indices[start : start + minibatch_size]
            batch_features = features[batch_indices]
            batch_legal_masks = legal_masks[batch_indices]
            batch_actions = actions[batch_indices]
            batch_rewards = rewards[batch_indices]

            logits = model(batch_features)
            masked_logits = logits.masked_fill(~batch_legal_masks, torch.finfo(logits.dtype).min)
            log_probs = torch.log_softmax(masked_logits, dim=-1)
            probs = torch.softmax(masked_logits, dim=-1)
            selected_log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
            entropy = -(probs * log_probs).sum(dim=-1)
            loss = -(selected_log_probs * batch_rewards).mean() - entropy_coef * entropy.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            losses.append(float(loss.detach().cpu().item()))
            entropies.append(float(entropy.mean().detach().cpu().item()))

    model.eval()
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "entropy": sum(entropies) / max(len(entropies), 1),
    }


def append_metrics(metrics_path: Path, payload: dict[str, Any]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def candidate_is_better(evaluation_result: dict[str, Any]) -> bool:
    candidate = evaluation_result["metrics"].get("candidate")
    baseline = evaluation_result["metrics"].get("baseline")
    if candidate is None or baseline is None:
        return True
    if candidate["average_rank"] != baseline["average_rank"]:
        return candidate["average_rank"] < baseline["average_rank"]
    return candidate["average_score"] > baseline["average_score"]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists():
        initialize_checkpoint(
            checkpoint_path,
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
            seed=args.seed,
        )

    supervised_pretrain = SupervisedPretrainConfig(
        enabled=args.enable_supervised_pretrain,
        dataset_path=str(args.supervised_dataset) if args.supervised_dataset is not None else None,
    )
    maybe_run_supervised_pretraining(supervised_pretrain)

    learner_device = select_device(args.device)
    reward_config = RewardConfig()
    self_play_config = SelfPlayConfig(
        matches_per_iteration=args.matches_per_iteration,
        workers=args.workers,
        temperature=args.self_play_temperature,
        seed=args.seed,
    )
    optimizer_config = OptimizerConfig(
        device=learner_device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        entropy_coef=args.entropy_coef,
        grad_clip_norm=args.grad_clip,
        minibatch_size=args.minibatch_size,
        epochs_per_iteration=args.epochs_per_iteration,
    )
    evaluation_config = EvaluationConfig(
        matches=args.evaluation_matches,
        workers=args.workers,
    )

    model, config, payload = build_model_from_checkpoint(checkpoint_path, device=learner_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
    )
    if isinstance(payload.get("optimizer_state_dict"), dict):
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    step = int(payload.get("step", 0))
    if not args.best_checkpoint.exists():
        copy_checkpoint(checkpoint_path, args.best_checkpoint)

    for iteration in range(1, args.iterations + 1):
        current_seed = self_play_config.seed + iteration * 1000
        match_specs = expand_selfplay_specs(
            checkpoint_path,
            matches=self_play_config.matches_per_iteration,
            temperature=self_play_config.temperature,
        )
        match_results = run_match_series(
            match_specs,
            reward_config=reward_config,
            workers=self_play_config.workers,
            seed=current_seed,
        )
        self_play_summary = summarize_matches(match_results)
        examples = flatten_training_examples(match_results, policy_name="candidate")
        if not examples:
            raise RuntimeError("self-play produced no training examples")

        train_metrics = train_policy_iteration(
            model,
            examples=examples,
            optimizer=optimizer,
            device=optimizer_config.device,
            minibatch_size=optimizer_config.minibatch_size,
            epochs_per_iteration=optimizer_config.epochs_per_iteration,
            entropy_coef=optimizer_config.entropy_coef,
            grad_clip_norm=optimizer_config.grad_clip_norm,
        )

        step += 1
        latest_metrics = {
            "iteration": step,
            "self_play": {
                key: asdict(value)
                for key, value in self_play_summary.items()
            },
            "train": train_metrics,
            "examples": len(examples),
            "device": optimizer_config.device,
        }
        save_checkpoint(
            checkpoint_path,
            model=model,
            config=config,
            step=step,
            optimizer_state_dict=optimizer.state_dict(),
            metrics=latest_metrics,
        )

        evaluation = evaluate_policy_paths(
            candidate_checkpoint=checkpoint_path,
            baseline_checkpoint=args.best_checkpoint,
            matches=evaluation_config.matches,
            workers=evaluation_config.workers,
            seed=current_seed + 500,
            deterministic=evaluation_config.deterministic,
            reward_config=reward_config,
        )
        improved = candidate_is_better(evaluation)
        if improved:
            copy_checkpoint(checkpoint_path, args.best_checkpoint)

        metrics_row = {
            **latest_metrics,
            "evaluation": evaluation["metrics"],
            "improved": improved,
        }
        append_metrics(args.metrics_jsonl, metrics_row)

        print(
            json.dumps(
                {
                    "iteration": step,
                    "examples": len(examples),
                    "loss": train_metrics["loss"],
                    "entropy": train_metrics["entropy"],
                    "candidate_eval": evaluation["metrics"].get("candidate"),
                    "baseline_eval": evaluation["metrics"].get("baseline"),
                    "improved": improved,
                },
                ensure_ascii=True,
            )
        )


if __name__ == "__main__":
    main()