from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.policy_net import PolicyNet, PolicyNetConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a deployment-friendly PyTorch policy checkpoint to ONNX."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a .pt checkpoint file.")
    parser.add_argument("--onnx", type=Path, required=True, help="Output ONNX file path.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional output metadata JSON path. Defaults to <onnx>.json.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    return parser.parse_args()


def checkpoint_state_dict(payload: dict) -> dict:
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint does not contain model_state_dict")
    return state_dict


def checkpoint_config(payload: dict) -> PolicyNetConfig:
    embedded = payload.get("config")
    if isinstance(embedded, dict):
        return PolicyNetConfig.from_dict(embedded)
    raise ValueError("checkpoint does not contain config")


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("expected the checkpoint payload to be a dict")

    config = checkpoint_config(payload)
    model = PolicyNet(config)
    model.load_state_dict(checkpoint_state_dict(payload))
    model.eval()

    onnx_path = args.onnx
    metadata_path = args.metadata or onnx_path.with_suffix(".json")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.zeros((1, config.input_dim), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamo=False,
    )

    metadata = {
        "model_type": "async_actor_critic_policy",
        "checkpoint": str(args.checkpoint.name),
        "input_dim": config.input_dim,
        "action_dim": config.action_dim,
        "hidden_dims": list(config.hidden_dims),
        "value_hidden_dims": list(config.value_hidden_dims),
        "dropout": config.dropout,
        "onnx_opset": args.opset,
        "onnx_file": onnx_path.name,
        "export_outputs": ["logits"],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Exported ONNX model to {onnx_path}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()