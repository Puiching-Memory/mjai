from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.evaluation import evaluate_policy_paths
from train.training_config import EvaluationConfig, RewardConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint by average rank / average score with in-process self-play matches."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Candidate checkpoint path.")
    parser.add_argument("--baseline-checkpoint", type=Path, default=None, help="Optional baseline checkpoint path.")
    parser.add_argument("--matches", type=int, default=8, help="Number of evaluation matches.")
    parser.add_argument("--workers", type=int, default=1, help="Number of CPU self-play workers.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for match seeds.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    evaluation_config = EvaluationConfig(matches=args.matches, workers=args.workers)
    reward_config = RewardConfig()
    result = evaluate_policy_paths(
        candidate_checkpoint=args.checkpoint,
        baseline_checkpoint=args.baseline_checkpoint,
        matches=evaluation_config.matches,
        workers=evaluation_config.workers,
        seed=args.seed,
        deterministic=evaluation_config.deterministic,
        reward_config=reward_config,
    )

    payload = json.dumps(result["metrics"], ensure_ascii=True, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Wrote evaluation metrics to {args.output}")
    else:
        print(payload)


if __name__ == "__main__":
    main()