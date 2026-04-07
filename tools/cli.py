from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) in sys.path:
    sys.path.remove(str(TOOLS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot import main as bot_main  # noqa: E402
from bot import resolve_runtime_paths  # noqa: E402
from rust_mjai_bot import resolve_bot_decision_path  # noqa: E402
from rich.console import Group  # noqa: E402
from rich.table import Table  # noqa: E402
from tools import train_reinforce  # noqa: E402
from train.training_ui import (  # noqa: E402
    create_rich_console,
    render_example_panel,
    render_info_panel,
    render_note_panel,
)


@dataclass(frozen=True, slots=True)
class TrainPreset:
    name: str
    description: str
    base_args: tuple[str, ...]


SCRIPT_PATH = Path("tools/cli.py")
SINGLE_TRAIN_EXAMPLE = f"uv run python {SCRIPT_PATH} train --preset single"
INFER_EXAMPLE = f"uv run python {SCRIPT_PATH} infer --seat 0"
DDP8_EXAMPLE = (
    "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
    "torchrun --nproc_per_node=8 tools/train_reinforce.py "
    "--checkpoint artifacts/policy.pt --best-checkpoint artifacts/policy.best.pt "
    "--learner-device cuda --inference-device cuda:0"
)

TRAIN_PRESETS: dict[str, TrainPreset] = {
    "single": TrainPreset(
        name="single",
        description="单卡训练默认预设，直接落到当前 async actor-learner trainer。",
        base_args=(
            "--run-label",
            "single-gpu",
            "--log-format",
            "rich",
            "--learner-device",
            "cuda:0",
            "--inference-device",
            "cuda:0",
            "--actor-processes",
            "8",
            "--total-learner-steps",
            "1000",
            "--warmup-steps",
            "4096",
            "--minibatch-size",
            "2048",
            "--evaluation-matches",
            "8",
        ),
    ),
}


def parse_train_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog=f"python {SCRIPT_PATH} train",
        description="Launch the unified training entrypoint.",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(TRAIN_PRESETS),
        default="single",
        help="Named training preset. Additional unknown flags are forwarded to tools/train_reinforce.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved trainer configuration and exit without starting training.",
    )
    known, trainer_args = parser.parse_known_args(argv)
    if trainer_args and trainer_args[0] == "--":
        trainer_args = trainer_args[1:]
    return known, trainer_args


def parse_infer_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"python {SCRIPT_PATH} infer",
        description="Launch the unified inference entrypoint.",
    )
    parser.add_argument("--seat", type=int, default=0, help="Player seat id passed to bot.py.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the startup summary only and exit without starting the bot protocol loop.",
    )
    return parser.parse_args(argv)


def build_train_argv(preset_name: str, trainer_args: Sequence[str]) -> list[str]:
    preset = TRAIN_PRESETS[preset_name]
    return [*preset.base_args, *trainer_args]


def render_command_hub() -> None:
    console = create_rich_console()

    commands_panel = render_info_panel(
        title="Commands",
        rows=[
            ("train", f"uv run python {SCRIPT_PATH} train --preset single"),
            ("infer", f"uv run python {SCRIPT_PATH} infer --seat 0"),
        ],
        border_style="bright_cyan",
    )
    examples_panel = render_example_panel(
        title="Train Examples",
        rows=[
            ("single train", SINGLE_TRAIN_EXAMPLE),
            ("8-card ddp", DDP8_EXAMPLE),
            ("infer", INFER_EXAMPLE),
            (
                "override",
                f"uv run python {SCRIPT_PATH} train --preset single --total-learner-steps 200 --actor-processes 12",
            ),
        ],
        border_style="green",
    )
    notes_panel = render_note_panel(
        title="Notes",
        lines=[
            "train 会把未识别参数直接转发给 tools/train_reinforce.py。",
            "infer 会在 stderr 打印 Rich 启动摘要，stdout 仍保留给 mjai 协议。",
            "Rich 首页默认展示两条训练示例：单卡和 8 卡 DDP。",
            "当前主线训练器是 async actor-learner；8 卡 DDP 仍是历史参考命令，不会伪装成 unified preset。",
        ],
        border_style="yellow",
    )

    grid = Table.grid(expand=True)
    grid.add_column(ratio=5)
    grid.add_column(ratio=7)
    grid.add_row(commands_panel, examples_panel)

    console.print(Group(grid, notes_panel))


def render_train_banner(
    *,
    preset_name: str,
    trainer_namespace: argparse.Namespace,
    resolved_argv: Sequence[str],
    dry_run: bool,
) -> None:
    console = create_rich_console()
    info_panel = render_info_panel(
        title="Train Launch",
        rows=[
            ("preset", preset_name),
            ("learner", trainer_namespace.learner_device),
            ("inference", trainer_namespace.inference_device),
            ("actors", str(trainer_namespace.actor_processes)),
            ("steps", str(trainer_namespace.total_learner_steps)),
            ("warmup", str(trainer_namespace.warmup_steps)),
            ("entry", "tools/train_reinforce.py"),
        ],
        border_style="cyan",
    )
    example_panel = render_example_panel(
        title="Commands",
        rows=[
            ("train", SINGLE_TRAIN_EXAMPLE),
            ("8-card ddp", DDP8_EXAMPLE),
            ("infer", INFER_EXAMPLE),
            ("argv", " ".join(resolved_argv)),
        ],
        border_style="magenta",
    )
    note_lines = [
        TRAIN_PRESETS[preset_name].description,
        "可以直接在统一命令后追加底层 trainer 参数覆盖默认值。",
        "8 卡 DDP 示例会和单卡示例一起展示，但它当前仍是参考命令，不是统一入口 preset。",
    ]
    if dry_run:
        note_lines.append("当前是 dry-run；不会真正启动训练。")
    notes_panel = render_note_panel(title="Notes", lines=note_lines, border_style="yellow")

    grid = Table.grid(expand=True)
    grid.add_column(ratio=5)
    grid.add_column(ratio=7)
    grid.add_row(info_panel, example_panel)
    console.print(Group(grid, notes_panel))


def render_infer_banner(*, seat: int, dry_run: bool) -> None:
    console = create_rich_console(stderr=True)
    runtime_rows: list[tuple[str, str]] = [("seat", str(seat)), ("entry", "bot.py")]
    note_lines = [
        "infer 启动摘要打印到 stderr；stdout 仍由 mjai Bot 协议占用。",
        "默认会从 artifacts/ 和 native_runtime/ 下自动查找 runtime、onnx、metadata。",
    ]
    try:
        runtime_paths = resolve_runtime_paths()
        runtime_rows.extend(
            [
                ("runtime", str(runtime_paths["binary_path"])),
                ("model", str(runtime_paths["model_path"])),
                ("meta", str(runtime_paths["metadata_path"])),
            ]
        )
    except Exception as exc:
        note_lines.append(f"runtime resolution failed: {exc}")

    try:
        runtime_rows.append(("decision", str(resolve_bot_decision_path())))
    except Exception as exc:
        note_lines.append(f"rust decision runtime resolution failed: {exc}")

    if dry_run:
        note_lines.append("当前是 dry-run；不会真正进入 bot 协议循环。")

    info_panel = render_info_panel(title="Infer Launch", rows=runtime_rows, border_style="cyan")
    example_panel = render_example_panel(
        title="Commands",
        rows=[
            ("infer", INFER_EXAMPLE),
            ("single train", SINGLE_TRAIN_EXAMPLE),
            ("8-card ddp", DDP8_EXAMPLE),
            ("seat 2", f"uv run python {SCRIPT_PATH} infer --seat 2"),
        ],
        border_style="green",
    )
    notes_panel = render_note_panel(title="Notes", lines=note_lines, border_style="yellow")

    grid = Table.grid(expand=True)
    grid.add_column(ratio=5)
    grid.add_column(ratio=7)
    grid.add_row(info_panel, example_panel)
    console.print(Group(grid, notes_panel))


def handle_train(argv: list[str]) -> int:
    known_args, trainer_args = parse_train_args(argv)
    resolved_argv = build_train_argv(known_args.preset, trainer_args)
    trainer_namespace = train_reinforce.parse_args(resolved_argv)
    render_train_banner(
        preset_name=known_args.preset,
        trainer_namespace=trainer_namespace,
        resolved_argv=resolved_argv,
        dry_run=known_args.dry_run,
    )
    if known_args.dry_run:
        return 0
    return train_reinforce.main(resolved_argv)


def handle_infer(argv: list[str]) -> int:
    args = parse_infer_args(argv)
    render_infer_banner(seat=args.seat, dry_run=args.dry_run)
    if args.dry_run:
        return 0
    return bot_main([str(args.seat)])


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        render_command_hub()
        return 0

    command = args[0]
    remainder = args[1:]
    if command == "train":
        return handle_train(remainder)
    if command == "infer":
        return handle_infer(remainder)

    render_command_hub()
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())