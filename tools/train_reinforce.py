from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import json
import multiprocessing as mp
import os
import queue
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("MJAI_LOG_LEVEL", "WARNING")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.async_self_play import (  # noqa: E402
    ActorFailure,
    AsyncInferenceServer,
    EpisodeTensorBatch,
    actor_process_main,
    create_shared_inference_state,
)
from train.checkpoints import (  # noqa: E402
    build_model_from_checkpoint,
    copy_checkpoint,
    initialize_checkpoint,
    save_checkpoint,
)
from train.evaluation import evaluate_policy_paths  # noqa: E402
from train.profiling import profile_scope  # noqa: E402
from train.training_config import GAEConfig, RewardConfig  # noqa: E402
from train.training_ui import TrainingDashboard, resolve_rich_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fully asynchronous actor-learner trainer with shared-memory actors and batched GPU inference."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/policy.pt"), help="Training checkpoint path.")
    parser.add_argument(
        "--best-checkpoint",
        type=Path,
        default=Path("artifacts/policy.best.pt"),
        help="Best-checkpoint path maintained by async background evaluation.",
    )
    parser.add_argument(
        "--metrics-jsonl",
        type=Path,
        default=Path("artifacts/training_metrics.jsonl"),
        help="Metrics JSONL output path.",
    )
    parser.add_argument(
        "--log-format",
        choices=("auto", "rich", "text", "json"),
        default="auto",
        help="Console log format.",
    )
    parser.add_argument("--run-label", type=str, default=None, help="Optional label shown in the Rich dashboard.")
    parser.add_argument("--learner-device", type=str, default="auto", help="Learner device: auto/cpu/cuda/cuda:N.")
    parser.add_argument("--inference-device", type=str, default="auto", help="Inference device: auto/cpu/cuda/cuda:N.")
    parser.add_argument("--total-learner-steps", type=int, default=1000, help="Number of learner optimizer steps.")
    parser.add_argument("--actor-processes", type=int, default=max(1, (os.cpu_count() or 4) // 2), help="Number of persistent self-play actor processes.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Actor sampling temperature.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--replay-capacity", type=int, default=32768, help="Replay capacity measured in decision steps.")
    parser.add_argument("--warmup-steps", type=int, default=4096, help="Replay warmup threshold before learner starts.")
    parser.add_argument("--max-policy-lag", type=int, default=32, help="Drop samples older than this many policy versions.")
    parser.add_argument(
        "--min-fresh-replay-steps",
        type=int,
        default=0,
        help="Minimum lag-bounded replay steps required before a learner update. 0 selects an automatic threshold.",
    )
    parser.add_argument(
        "--min-replay-growth-steps",
        type=int,
        default=0,
        help="New replay steps required to earn one learner update. 0 selects an automatic threshold.",
    )
    parser.add_argument("--minibatch-size", type=int, default=2048, help="Learner minibatch size sampled from replay.")
    parser.add_argument("--updates-per-step", type=int, default=1, help="Optimizer updates per scheduler loop.")
    parser.add_argument("--lr", type=float, default=3.0e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1.0e-5, help="AdamW weight decay.")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient.")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient.")
    parser.add_argument("--ppo-clip", type=float, default=0.2, help="Clipped ratio for stale-policy PPO style updates.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[256, 256], help="Policy trunk hidden dims for a fresh checkpoint.")
    parser.add_argument("--value-hidden-dims", type=int, nargs="*", default=[], help="Optional extra hidden dims for the value head.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")
    parser.add_argument("--inference-batch-size", type=int, default=512, help="Max batched inference size across all actors.")
    parser.add_argument("--inference-timeout-ms", type=float, default=2.0, help="Dynamic micro-batch timeout in milliseconds.")
    parser.add_argument("--policy-sync-interval", type=int, default=4, help="Publish learner weights to inference service every N learner steps.")
    parser.add_argument("--checkpoint-interval", type=int, default=50, help="Checkpoint every N learner steps.")
    parser.add_argument("--log-interval", type=int, default=10, help="Print a training summary every N learner steps.")
    parser.add_argument("--evaluation-matches", type=int, default=0, help="Background evaluation matches. Set 0 to disable async evaluation.")
    parser.add_argument("--evaluation-workers", type=int, default=1, help="CPU workers for background evaluation.")
    parser.add_argument("--evaluation-interval", type=int, default=100, help="Launch background evaluation every N learner steps.")
    parser.add_argument("--gae-gamma", type=float, default=0.999, help="GAE discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda for bias-variance tradeoff.")
    parser.add_argument("--shanten-shaping-weight", type=float, default=0.05, help="Potential-based reward shaping weight for shanten improvement.")
    return parser.parse_args(argv)


def select_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def resolve_threshold(raw_value: int, automatic_value: int) -> int:
    if raw_value > 0:
        return raw_value
    return max(1, automatic_value)


def ensure_spawn_start_method() -> None:
    mp.set_start_method("spawn", force=True)


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


def ingest_episode_batch(
    queued_item: EpisodeTensorBatch | ActorFailure,
    *,
    replay: "EpisodeReplayBuffer",
) -> tuple[int, int]:
    if isinstance(queued_item, ActorFailure):
        raise RuntimeError(
            f"actor {queued_item.actor_id} failed: {queued_item.message}\n{queued_item.traceback}"
        )
    replay.add(queued_item)
    return 1, queued_item.decision_count


def drain_episode_queue(
    episode_queue: mp.Queue[EpisodeTensorBatch | ActorFailure],
    *,
    replay: "EpisodeReplayBuffer",
    max_items: int,
    block_timeout: float | None,
) -> tuple[int, int]:
    drained_matches = 0
    drained_decisions = 0

    if block_timeout is not None:
        try:
            queued_item = episode_queue.get(timeout=block_timeout)
        except queue.Empty:
            return 0, 0
        match_count, decision_count = ingest_episode_batch(queued_item, replay=replay)
        drained_matches += match_count
        drained_decisions += decision_count

    while drained_matches < max_items:
        try:
            queued_item = episode_queue.get_nowait()
        except queue.Empty:
            break
        match_count, decision_count = ingest_episode_batch(queued_item, replay=replay)
        drained_matches += match_count
        drained_decisions += decision_count

    return drained_matches, drained_decisions


class EpisodeReplayBuffer:
    def __init__(self, capacity_steps: int) -> None:
        self.capacity_steps = max(1, int(capacity_steps))
        self._batches: deque[EpisodeTensorBatch] = deque()
        self.total_steps = 0
        self.ingested_steps_total = 0
        self.ingested_batches_total = 0

    def add(self, batch: EpisodeTensorBatch) -> None:
        if batch.decision_count <= 0:
            return
        self._batches.append(batch)
        self.total_steps += batch.decision_count
        self.ingested_steps_total += batch.decision_count
        self.ingested_batches_total += 1
        while self.total_steps > self.capacity_steps and self._batches:
            removed = self._batches.popleft()
            self.total_steps -= removed.decision_count

    def prune_stale(self, *, current_policy_version: int, max_policy_lag: int) -> None:
        stale_limit = max_policy_lag * 2
        while self._batches and current_policy_version - self._batches[0].max_policy_version > stale_limit:
            removed = self._batches.popleft()
            self.total_steps -= removed.decision_count

    def fresh_steps(self, *, current_policy_version: int, max_policy_lag: int) -> int:
        fresh_steps, _ = self.fresh_stats(
            current_policy_version=current_policy_version,
            max_policy_lag=max_policy_lag,
        )
        return fresh_steps

    def fresh_batch_count(self, *, current_policy_version: int, max_policy_lag: int) -> int:
        _, fresh_batches = self.fresh_stats(
            current_policy_version=current_policy_version,
            max_policy_lag=max_policy_lag,
        )
        return fresh_batches

    def fresh_stats(
        self,
        *,
        current_policy_version: int,
        max_policy_lag: int,
    ) -> tuple[int, int]:
        fresh_steps = 0
        fresh_batches = 0
        for batch in self._batches:
            if current_policy_version - batch.max_policy_version > max_policy_lag:
                continue
            fresh_steps += batch.decision_count
            fresh_batches += 1
        return fresh_steps, fresh_batches

    def sample(
        self,
        *,
        batch_size: int,
        current_policy_version: int,
        max_policy_lag: int,
        rng: random.Random,
    ) -> dict[str, Any] | None:
        valid_batches = [
            batch
            for batch in self._batches
            if current_policy_version - batch.max_policy_version <= max_policy_lag * 2
        ]
        if not valid_batches:
            return None

        weights = [max(batch.decision_count, 1) for batch in valid_batches]
        selected_batches: list[EpisodeTensorBatch] = []
        selected_steps = 0
        attempts = 0
        while selected_steps < batch_size and attempts < len(valid_batches) * 4 + 32:
            batch = rng.choices(valid_batches, weights=weights, k=1)[0]
            selected_batches.append(batch)
            selected_steps += batch.decision_count
            attempts += 1

        if not selected_batches:
            return None

        if len(selected_batches) == 1:
            selected_batch = selected_batches[0]
            features = selected_batch.features
            legal_actions = selected_batch.legal_actions
            actions = selected_batch.actions
            returns = selected_batch.returns
            advantages = selected_batch.advantages
            behavior_logprobs = selected_batch.behavior_logprobs
            behavior_values = selected_batch.behavior_values
            policy_versions = selected_batch.policy_versions
        else:
            features = torch.cat([batch.features for batch in selected_batches], dim=0)
            legal_actions = torch.cat([batch.legal_actions for batch in selected_batches], dim=0)
            actions = torch.cat([batch.actions for batch in selected_batches], dim=0)
            returns = torch.cat([batch.returns for batch in selected_batches], dim=0)
            advantages = torch.cat([batch.advantages for batch in selected_batches], dim=0)
            behavior_logprobs = torch.cat([batch.behavior_logprobs for batch in selected_batches], dim=0)
            behavior_values = torch.cat([batch.behavior_values for batch in selected_batches], dim=0)
            policy_versions = torch.cat([batch.policy_versions for batch in selected_batches], dim=0)

        lag_tensor = current_policy_version - policy_versions
        fresh_mask = lag_tensor <= max_policy_lag
        if not bool(fresh_mask.any().item()):
            return None

        features = features[fresh_mask]
        legal_actions = legal_actions[fresh_mask]
        actions = actions[fresh_mask]
        returns = returns[fresh_mask]
        advantages = advantages[fresh_mask]
        behavior_logprobs = behavior_logprobs[fresh_mask]
        behavior_values = behavior_values[fresh_mask]
        policy_versions = policy_versions[fresh_mask]
        lag_tensor = lag_tensor[fresh_mask]

        sample_size = actions.shape[0]
        if sample_size > batch_size:
            indices = torch.randperm(sample_size)[:batch_size]
            features = features[indices]
            legal_actions = legal_actions[indices]
            actions = actions[indices]
            returns = returns[indices]
            advantages = advantages[indices]
            behavior_logprobs = behavior_logprobs[indices]
            behavior_values = behavior_values[indices]
            policy_versions = policy_versions[indices]
            lag_tensor = lag_tensor[indices]

        return {
            "features": features,
            "legal_actions": legal_actions,
            "actions": actions,
            "returns": returns,
            "advantages": advantages,
            "behavior_logprobs": behavior_logprobs,
            "behavior_values": behavior_values,
            "policy_versions": policy_versions,
            "mean_policy_lag": float(lag_tensor.float().mean().item()),
            "steps": int(actions.shape[0]),
            "batch_count": len(selected_batches),
        }


def learner_update(
    model,
    optimizer: torch.optim.Optimizer,
    *,
    batch: dict[str, Any],
    device: str,
    entropy_coef: float,
    value_coef: float,
    ppo_clip: float,
    grad_clip_norm: float,
) -> dict[str, float]:
    with profile_scope("learner.update"):
        use_non_blocking = device.startswith("cuda")
        with profile_scope("learner.transfer_batch"):
            features = batch["features"].to(device, non_blocking=use_non_blocking)
            legal_actions = batch["legal_actions"].to(device, non_blocking=use_non_blocking)
            actions = batch["actions"].to(device, non_blocking=use_non_blocking)
            returns = batch["returns"].to(device, non_blocking=use_non_blocking)
            gae_advantages = batch["advantages"].to(device, non_blocking=use_non_blocking)
            behavior_logprobs = batch["behavior_logprobs"].to(device, non_blocking=use_non_blocking)
            behavior_values = batch["behavior_values"].to(device, non_blocking=use_non_blocking)

        with profile_scope("learner.forward"):
            logits, values = model.policy_and_value(features)
            masked_logits = logits.masked_fill(~legal_actions, torch.finfo(logits.dtype).min)
            log_probs = torch.log_softmax(masked_logits, dim=-1)
            probs = torch.softmax(masked_logits, dim=-1)
            selected_logprobs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            entropy = -(probs * log_probs).sum(dim=-1)

        with profile_scope("learner.compute_loss"):
            advantages = gae_advantages
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-6)

            ratios = torch.exp(selected_logprobs - behavior_logprobs)
            clipped_ratios = torch.clamp(ratios, 1.0 - ppo_clip, 1.0 + ppo_clip)
            policy_loss = -torch.minimum(ratios * advantages, clipped_ratios * advantages).mean()
            value_loss = F.mse_loss(values, returns)
            entropy_loss = entropy.mean()
            total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

        with profile_scope("learner.backward_step"):
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm).item())
            optimizer.step()

        with profile_scope("learner.metrics"):
            approx_kl = 0.5 * torch.mean((selected_logprobs - behavior_logprobs) ** 2)
            clip_fraction = torch.mean((torch.abs(ratios - clipped_ratios) > 1.0e-6).to(dtype=torch.float32))

    return {
        "loss": float(total_loss.detach().cpu().item()),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "entropy": float(entropy_loss.detach().cpu().item()),
        "approx_kl": float(approx_kl.detach().cpu().item()),
        "clip_fraction": float(clip_fraction.detach().cpu().item()),
        "grad_norm": grad_norm,
        "mean_return": float(returns.mean().detach().cpu().item()),
        "mean_behavior_value": float(behavior_values.mean().detach().cpu().item()),
        "mean_value": float(values.mean().detach().cpu().item()),
        "mean_policy_lag": float(batch["mean_policy_lag"]),
        "sample_steps": float(batch["steps"]),
        "sample_batches": float(batch["batch_count"]),
    }


def run_background_evaluation(
    *,
    snapshot_path: Path,
    baseline_checkpoint: Path,
    matches: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    with profile_scope("trainer.background_evaluation"):
        reward_config = RewardConfig()
        evaluation = evaluate_policy_paths(
            candidate_checkpoint=snapshot_path,
            baseline_checkpoint=baseline_checkpoint if baseline_checkpoint.exists() else None,
            matches=matches,
            workers=workers,
            seed=seed,
            deterministic=True,
            reward_config=reward_config,
        )
    return {
        "snapshot_path": str(snapshot_path),
        "metrics": evaluation["metrics"],
    }


def format_payload_text(payload: dict[str, Any]) -> str:
    learner = payload["learner"]
    replay = payload["replay"]
    actors = payload["actors"]
    inference = payload["inference"]
    parts = [
        f"step {payload['step']}",
        f"loss {learner['loss']:.4f}",
        f"policy {learner['policy_loss']:.4f}",
        f"value {learner['value_loss']:.4f}",
        f"entropy {learner['entropy']:.4f}",
        f"replay {replay['steps']} steps",
        f"fresh {replay.get('fresh_steps', 0)}",
        f"credit {replay.get('growth_credit', 0)}",
        f"actor {actors['decisions_per_sec']:.1f} d/s",
        f"infer batch {inference['avg_batch_size']:.1f}",
    ]
    if payload.get("evaluation") is not None:
        evaluation = payload["evaluation"]
        candidate = evaluation["metrics"].get("candidate") or {}
        parts.append(f"eval.rank {candidate.get('average_rank', 0.0):.3f}")
        parts.append(f"best {'yes' if evaluation['improved'] else 'no'}")
    return " | ".join(parts)


def log_payload(payload: dict[str, Any], *, log_format: str) -> None:
    if log_format == "json":
        print(json.dumps(payload, ensure_ascii=True))
        return
    print(format_payload_text(payload))


def main(argv: list[str] | None = None) -> int:
    with profile_scope("trainer.main"):
        args = parse_args(argv)
        rich_logging = resolve_rich_logging(args.log_format)
        effective_log_format = "rich" if rich_logging else ("json" if args.log_format == "json" else "text")
        ensure_spawn_start_method()

        learner_device = select_device(args.learner_device)
        inference_device = select_device(args.inference_device)
        rng = random.Random(args.seed)
        torch.manual_seed(args.seed)
        if learner_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        checkpoint_path = args.checkpoint
        if not checkpoint_path.exists():
            with profile_scope("trainer.initialize_checkpoint"):
                initialize_checkpoint(
                    checkpoint_path,
                    hidden_dims=tuple(args.hidden_dims),
                    dropout=args.dropout,
                    value_hidden_dims=tuple(args.value_hidden_dims),
                    seed=args.seed,
                )

        with profile_scope("trainer.load_checkpoint"):
            model, config, payload = build_model_from_checkpoint(checkpoint_path, device=learner_device)
            model.train()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            if isinstance(payload.get("optimizer_state_dict"), dict):
                optimizer.load_state_dict(payload["optimizer_state_dict"])

        learner_step = int(payload.get("step", 0))
        policy_version = int(payload.get("policy_version", learner_step))

        if not args.best_checkpoint.exists():
            with profile_scope("trainer.bootstrap_best_checkpoint"):
                copy_checkpoint(checkpoint_path, args.best_checkpoint)

        dashboard = TrainingDashboard(
            enabled=rich_logging,
            total_steps=args.total_learner_steps,
            warmup_steps=args.warmup_steps,
            learner_device=learner_device,
            inference_device=inference_device,
            actor_processes=args.actor_processes,
            checkpoint_path=checkpoint_path,
            best_checkpoint_path=args.best_checkpoint,
            metrics_path=args.metrics_jsonl,
            run_label=args.run_label,
        )

        reward_config = RewardConfig()
        gae_config = GAEConfig(
            gamma=args.gae_gamma,
            gae_lambda=args.gae_lambda,
            shanten_shaping_weight=args.shanten_shaping_weight,
        )
        replay = EpisodeReplayBuffer(args.replay_capacity)
        min_fresh_replay_steps = resolve_threshold(
            args.min_fresh_replay_steps,
            max(args.minibatch_size, args.warmup_steps // 2),
        )
        min_replay_growth_steps = resolve_threshold(
            args.min_replay_growth_steps,
            args.minibatch_size,
        )

        with dashboard:
            if rich_logging:
                dashboard.add_event(
                    f"run {dashboard.run_label} | learner {learner_device} | "
                    f"inference {inference_device} | actors {args.actor_processes}",
                    style="bold cyan",
                    refresh=False,
                )
                dashboard.update_status(
                    phase="warmup",
                    detail=f"replay 0/{args.warmup_steps} | waiting for actor data",
                    completed=0,
                    total=args.warmup_steps,
                )

            with profile_scope("trainer.create_runtime"):
                shared_state = create_shared_inference_state(args.actor_processes)
                request_queue: mp.Queue[int] = mp.Queue(maxsize=max(args.actor_processes * 8, 32))
                episode_queue: mp.Queue[EpisodeTensorBatch | ActorFailure] = mp.Queue(
                    maxsize=max(args.actor_processes * 4, 16)
                )
                response_events = [mp.Event() for _ in range(args.actor_processes)]
                stop_event = mp.Event()

                inference_server = AsyncInferenceServer(
                    config=config,
                    device=inference_device,
                    request_queue=request_queue,
                    response_events=response_events,
                    shared_state=shared_state,
                    max_batch_size=args.inference_batch_size,
                    batch_timeout_ms=args.inference_timeout_ms,
                )
                inference_server.publish(model.state_dict(), policy_version)
                inference_server.start()

                actor_processes: list[mp.Process] = []
                for actor_id in range(args.actor_processes):
                    process = mp.Process(
                        name=f"mjai-actor-{actor_id}",
                        target=actor_process_main,
                        kwargs={
                            "actor_id": actor_id,
                            "seed": args.seed,
                            "temperature": args.temperature,
                            "deterministic": False,
                            "reward_config": reward_config,
                            "gae_config": gae_config,
                            "request_queue": request_queue,
                            "response_events": response_events,
                            "shared_state": shared_state,
                            "episode_queue": episode_queue,
                            "stop_event": stop_event,
                        },
                        daemon=True,
                    )
                    process.start()
                    actor_processes.append(process)

            if rich_logging:
                dashboard.add_event("actors and inference server started", style="green")

            eval_executor = ThreadPoolExecutor(max_workers=1)
            pending_eval: Future[dict[str, Any]] | None = None
            last_evaluation: dict[str, Any] | None = None

            total_matches = 0
            total_decisions = 0
            replay_growth_credit = 0
            last_log_matches = 0
            last_log_decisions = 0
            last_log_time = perf_counter()
            warmup_notice_time = 0.0
            stall_notice_time = 0.0
            completed_normally = False

            try:
                while learner_step < args.total_learner_steps:
                    with profile_scope("trainer.scheduler_tick"):
                        with profile_scope("trainer.health_check"):
                            inference_server.check_health()
                            for process in actor_processes:
                                if process.exitcode not in (None, 0):
                                    raise RuntimeError(
                                        f"actor process {process.pid} exited with code {process.exitcode}"
                                    )

                        with profile_scope("trainer.drain_episode_queue"):
                            drained_matches, drained_decisions = drain_episode_queue(
                                episode_queue,
                                replay=replay,
                                max_items=args.actor_processes * 4,
                                block_timeout=0.2 if replay.total_steps < args.warmup_steps else None,
                            )
                            replay_growth_credit += drained_decisions
                            total_matches += drained_matches
                            total_decisions += drained_decisions

                        with profile_scope("trainer.collect_evaluation"):
                            if pending_eval is not None and pending_eval.done():
                                evaluation_result = pending_eval.result()
                                evaluation_payload = {"metrics": evaluation_result["metrics"]}
                                improved = candidate_is_better(evaluation_payload)
                                snapshot_path = Path(evaluation_result["snapshot_path"])
                                if improved:
                                    copy_checkpoint(snapshot_path, args.best_checkpoint)
                                last_evaluation = {
                                    "metrics": evaluation_result["metrics"],
                                    "improved": improved,
                                }
                                snapshot_path.unlink(missing_ok=True)
                                pending_eval = None
                                if rich_logging:
                                    dashboard.record_evaluation(last_evaluation)

                        if replay.total_steps < args.warmup_steps:
                            with profile_scope("trainer.warmup_wait"):
                                now = perf_counter()
                                if now - warmup_notice_time >= 5.0:
                                    message = (
                                        f"warmup replay {replay.total_steps}/{args.warmup_steps} steps | "
                                        f"matches {total_matches} | actors {args.actor_processes}"
                                    )
                                    if rich_logging:
                                        dashboard.update_status(
                                            phase="warmup",
                                            detail=message,
                                            completed=replay.total_steps,
                                            total=args.warmup_steps,
                                        )
                                        dashboard.add_event(message, style="yellow", refresh=False)
                                        dashboard.refresh()
                                    elif effective_log_format == "text":
                                        print(message)
                                    warmup_notice_time = now
                            continue

                        with profile_scope("trainer.prune_replay"):
                            replay.prune_stale(
                                current_policy_version=policy_version,
                                max_policy_lag=args.max_policy_lag,
                            )

                        with profile_scope("trainer.check_fresh_replay"):
                            fresh_replay_steps, fresh_replay_batches = replay.fresh_stats(
                                current_policy_version=policy_version,
                                max_policy_lag=args.max_policy_lag,
                            )

                        if fresh_replay_steps < min_fresh_replay_steps:
                            with profile_scope("trainer.wait_fresh_replay"):
                                now = perf_counter()
                                if now - stall_notice_time >= 5.0:
                                    message = (
                                        f"waiting fresh replay {fresh_replay_steps}/{min_fresh_replay_steps} steps | "
                                        f"fresh batches {fresh_replay_batches} | policy lag <= {args.max_policy_lag}"
                                    )
                                    if rich_logging:
                                        dashboard.update_status(
                                            phase="wait-fresh",
                                            detail=message,
                                            completed=fresh_replay_steps,
                                            total=min_fresh_replay_steps,
                                        )
                                        dashboard.add_event(message, style="yellow", refresh=False)
                                        dashboard.refresh()
                                    elif effective_log_format == "text":
                                        print(message)
                                    stall_notice_time = now
                            with profile_scope("trainer.block_for_fresh_replay"):
                                drained_matches, drained_decisions = drain_episode_queue(
                                    episode_queue,
                                    replay=replay,
                                    max_items=args.actor_processes * 4,
                                    block_timeout=1.0,
                                )
                                replay_growth_credit += drained_decisions
                                total_matches += drained_matches
                                total_decisions += drained_decisions
                            continue

                        if replay_growth_credit < min_replay_growth_steps:
                            with profile_scope("trainer.wait_replay_growth"):
                                now = perf_counter()
                                if now - stall_notice_time >= 5.0:
                                    message = (
                                        f"waiting replay growth credit {replay_growth_credit}/{min_replay_growth_steps} | "
                                        f"replay {replay.total_steps} | fresh {fresh_replay_steps}"
                                    )
                                    if rich_logging:
                                        dashboard.update_status(
                                            phase="wait-growth",
                                            detail=message,
                                            completed=replay_growth_credit,
                                            total=min_replay_growth_steps,
                                        )
                                        dashboard.add_event(message, style="yellow", refresh=False)
                                        dashboard.refresh()
                                    elif effective_log_format == "text":
                                        print(message)
                                    stall_notice_time = now
                            with profile_scope("trainer.block_for_replay_growth"):
                                drained_matches, drained_decisions = drain_episode_queue(
                                    episode_queue,
                                    replay=replay,
                                    max_items=args.actor_processes * 4,
                                    block_timeout=1.0,
                                )
                                replay_growth_credit += drained_decisions
                                total_matches += drained_matches
                                total_decisions += drained_decisions
                            continue

                        with profile_scope("trainer.sample_replay"):
                            sampled_batch = replay.sample(
                                batch_size=args.minibatch_size,
                                current_policy_version=policy_version,
                                max_policy_lag=args.max_policy_lag,
                                rng=rng,
                            )
                        if sampled_batch is None:
                            continue

                        optimizer_metrics = learner_update(
                            model,
                            optimizer,
                            batch=sampled_batch,
                            device=learner_device,
                            entropy_coef=args.entropy_coef,
                            value_coef=args.value_coef,
                            ppo_clip=args.ppo_clip,
                            grad_clip_norm=args.grad_clip,
                        )

                        learner_step += 1
                        policy_version += 1
                        replay_growth_credit = max(0, replay_growth_credit - min_replay_growth_steps)

                        if learner_step % args.policy_sync_interval == 0:
                            with profile_scope("trainer.publish_weights"):
                                inference_server.publish(model.state_dict(), policy_version)

                        if learner_step % args.checkpoint_interval == 0:
                            with profile_scope("trainer.save_checkpoint"):
                                save_checkpoint(
                                    checkpoint_path,
                                    model=model,
                                    config=config,
                                    step=learner_step,
                                    policy_version=policy_version,
                                    optimizer_state_dict=optimizer.state_dict(),
                                    metrics={
                                        "step": learner_step,
                                        "policy_version": policy_version,
                                        "replay_steps": replay.total_steps,
                                    },
                                )
                            if rich_logging:
                                dashboard.add_event(
                                    f"checkpoint saved | step {learner_step} | {checkpoint_path.name}",
                                    style="green",
                                )

                        if (
                            args.evaluation_matches > 0
                            and pending_eval is None
                            and learner_step % args.evaluation_interval == 0
                        ):
                            with profile_scope("trainer.launch_evaluation"):
                                snapshot_path = checkpoint_path.with_name(
                                    f"{checkpoint_path.stem}.eval-step-{learner_step}.pt"
                                )
                                save_checkpoint(
                                    snapshot_path,
                                    model=model,
                                    config=config,
                                    step=learner_step,
                                    policy_version=policy_version,
                                    metrics={"step": learner_step, "policy_version": policy_version},
                                )
                                pending_eval = eval_executor.submit(
                                    run_background_evaluation,
                                    snapshot_path=snapshot_path,
                                    baseline_checkpoint=args.best_checkpoint,
                                    matches=args.evaluation_matches,
                                    workers=args.evaluation_workers,
                                    seed=args.seed + learner_step * 17,
                                )
                            if rich_logging:
                                dashboard.add_event(
                                    f"background evaluation started | step {learner_step} | "
                                    f"matches {args.evaluation_matches}",
                                    style="magenta",
                                )

                        with profile_scope("trainer.snapshot_metrics"):
                            inference_metrics = inference_server.snapshot_metrics()
                            metrics_row = {
                                "step": learner_step,
                                "policy_version": policy_version,
                                "learner": optimizer_metrics,
                                "replay": {
                                    "steps": replay.total_steps,
                                    "capacity": args.replay_capacity,
                                    "warmup_steps": args.warmup_steps,
                                    "fresh_steps": fresh_replay_steps,
                                    "fresh_batches": fresh_replay_batches,
                                    "growth_credit": replay_growth_credit,
                                    "min_fresh_replay_steps": min_fresh_replay_steps,
                                    "min_replay_growth_steps": min_replay_growth_steps,
                                    "ingested_steps_total": replay.ingested_steps_total,
                                },
                                "actors": {
                                    "matches": total_matches,
                                    "decisions": total_decisions,
                                },
                                "inference": inference_metrics,
                                "evaluation": last_evaluation,
                            }
                            append_metrics(args.metrics_jsonl, metrics_row)

                        if learner_step % args.log_interval == 0:
                            with profile_scope("trainer.log_payload"):
                                now = perf_counter()
                                elapsed = max(now - last_log_time, 1.0e-6)
                                match_delta = total_matches - last_log_matches
                                decision_delta = total_decisions - last_log_decisions
                                payload = {
                                    "step": learner_step,
                                    "learner": optimizer_metrics,
                                    "replay": {
                                        "steps": replay.total_steps,
                                        "capacity": args.replay_capacity,
                                        "fresh_steps": fresh_replay_steps,
                                        "growth_credit": replay_growth_credit,
                                    },
                                    "actors": {
                                        "matches_per_sec": match_delta / elapsed,
                                        "decisions_per_sec": decision_delta / elapsed,
                                        "matches_total": total_matches,
                                        "decisions_total": total_decisions,
                                    },
                                    "inference": inference_metrics,
                                    "evaluation": last_evaluation,
                                }
                                if rich_logging:
                                    dashboard.record_snapshot(
                                        payload,
                                        phase="learn",
                                        detail=(
                                            f"step {learner_step}/{args.total_learner_steps} | replay {replay.total_steps} | "
                                            f"fresh {fresh_replay_steps}"
                                        ),
                                    )
                                    dashboard.add_event(format_payload_text(payload), style="cyan")
                                else:
                                    log_payload(payload, log_format=effective_log_format)
                                last_log_time = now
                                last_log_matches = total_matches
                                last_log_decisions = total_decisions

                with profile_scope("trainer.final_checkpoint"):
                    save_checkpoint(
                        checkpoint_path,
                        model=model,
                        config=config,
                        step=learner_step,
                        policy_version=policy_version,
                        optimizer_state_dict=optimizer.state_dict(),
                        metrics={
                            "step": learner_step,
                            "policy_version": policy_version,
                            "replay_steps": replay.total_steps,
                            "evaluation": last_evaluation,
                        },
                    )
                completed_normally = True
                if rich_logging:
                    dashboard.finish(final_step=learner_step)
            finally:
                with profile_scope("trainer.cleanup"):
                    stop_event.set()
                    for process in actor_processes:
                        process.join(timeout=10)
                    for process in actor_processes:
                        if process.is_alive():
                            process.terminate()
                            process.join(timeout=2)
                    inference_server.stop()
                    if pending_eval is not None:
                        try:
                            evaluation_result = pending_eval.result(timeout=1)
                            Path(evaluation_result["snapshot_path"]).unlink(missing_ok=True)
                        except Exception:
                            pass
                    eval_executor.shutdown(wait=False, cancel_futures=True)
                    if rich_logging and not completed_normally:
                        dashboard.add_event("training aborted", style="bold red")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())