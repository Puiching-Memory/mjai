from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.inference_spec import ACTION_DIM, INPUT_DIM
from train.policy_net import PolicyNet, PolicyNetConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a randomly initialized checkpoint for the deployment-friendly policy network."
    )
    parser.add_argument("--output", type=Path, required=True, help="Output checkpoint path.")
    parser.add_argument(
        "--input-dim",
        type=int,
        default=INPUT_DIM,
        help="Feature vector length. Defaults to the current runtime protocol size.",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=ACTION_DIM,
        help="Number of action logits. Defaults to the current runtime protocol size.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="*",
        default=[256, 256],
        help="Hidden layer widths.",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic init.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    config = PolicyNetConfig(
        input_dim=args.input_dim,
        action_dim=args.action_dim,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    )
    model = PolicyNet(config)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
        },
        output_path,
    )

    print(f"Initialized checkpoint written to {output_path}")


if __name__ == "__main__":
    main()