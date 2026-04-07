from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run async training under NVIDIA Nsight Systems and export a text summary.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/nsys_async_profile"), help="Output prefix for .nsys-rep and summary files.")
    parser.add_argument("--nsys-bin", type=str, default=shutil.which("nsys") or "nsys", help="Path to the nsys executable.")
    parser.add_argument("--cuda-visible-devices", type=str, default=None, help="Optional CUDA_VISIBLE_DEVICES override.")
    parser.add_argument("--trace", type=str, default="cuda,nvtx,osrt", help="nsys trace domains.")
    parser.add_argument("--sample", type=str, default="process-tree", help="CPU sampling mode.")
    parser.add_argument("--cpuctxsw", type=str, default="process-tree", help="CPU context-switch collection mode.")
    parser.add_argument("--cuda-memory-usage", action="store_true", help="Enable CUDA memory usage collection.")
    parser.add_argument("--summary-lines", type=int, default=160, help="Maximum summary lines to print to stdout.")
    parser.add_argument(
        "--stats-reports",
        nargs="+",
        default=["nvtx_pushpop_sum", "osrt_sum", "cuda_api_sum", "cuda_gpu_kern_sum"],
        help="nsys stats report names.",
    )
    parser.add_argument(
        "trainer_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to tools/train_reinforce.py. Use '--' before trainer args.",
    )
    return parser.parse_args()


def strip_remainder_separator(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def main() -> None:
    args = parse_args()
    trainer_args = strip_remainder_separator(args.trainer_args)
    if not trainer_args:
        raise SystemExit("trainer args are required; pass them after '--'")

    report_prefix = args.output if args.output.is_absolute() else ROOT / args.output
    report_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = report_prefix.with_suffix(".summary.txt")

    env = os.environ.copy()
    env.setdefault("MJAI_ENABLE_NVTX", "1")
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    python_executable = Path(sys.executable)
    train_script = ROOT / "tools" / "train_reinforce.py"
    profile_command = [
        args.nsys_bin,
        "profile",
        "--force-overwrite=true",
        f"--trace={args.trace}",
        f"--sample={args.sample}",
        f"--cpuctxsw={args.cpuctxsw}",
        f"--cuda-memory-usage={'true' if args.cuda_memory_usage else 'false'}",
        f"--output={report_prefix}",
        str(python_executable),
        str(train_script),
        *trainer_args,
    ]

    print("profiling command:")
    print(" ".join(profile_command))
    subprocess.run(profile_command, cwd=ROOT, env=env, check=True)

    stats_command = [
        args.nsys_bin,
        "stats",
        "--force-export=true",
        "--report",
        ",".join(args.stats_reports),
        str(report_prefix.with_suffix('.nsys-rep')),
    ]
    stats_run = subprocess.run(
        stats_command,
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if stats_run.returncode != 0:
        details = []
        if stats_run.stdout:
            details.append(stats_run.stdout)
        if stats_run.stderr:
            details.append(stats_run.stderr)
        raise RuntimeError("nsys stats failed:\n" + "\n".join(details).strip())
    full_summary_text = stats_run.stdout
    summary_text = "\n".join(full_summary_text.splitlines()[: args.summary_lines])
    summary_path.write_text(full_summary_text, encoding="utf-8")

    print(f"report: {report_prefix.with_suffix('.nsys-rep')}")
    print(f"summary: {summary_path}")
    print(summary_text)


if __name__ == "__main__":
    main()