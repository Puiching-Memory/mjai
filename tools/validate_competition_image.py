from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rust_mjai_bot import resolve_bot_decision_path
from train.inference_spec import ACTION_DIM, INPUT_DIM

COMPETITION_IMAGE = "smly/mjai-client:v3"
SUBMISSION_ZIP_DEFAULT = Path("artifacts/submission.zip")
SUBMISSION_MAX_BYTES = 1_000_000_000


@dataclass(frozen=True, slots=True)
class BotFixtureCase:
    name: str
    fixture_path: Path
    expected_types: tuple[tuple[str, ...], ...]


DEFAULT_CASES = (
    BotFixtureCase(
        name="call-choice",
        fixture_path=ROOT / "tools" / "fixtures" / "competition_call_choice.jsonl",
        expected_types=(("none",), ("none", "pon")),
    ),
    BotFixtureCase(
        name="riichi-discard",
        fixture_path=ROOT / "tools" / "fixtures" / "competition_riichi_discard.jsonl",
        expected_types=(("none",), ("reach",), ("dahai",)),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a submission ZIP and validate it end-to-end inside the competition Docker image."
    )
    parser.add_argument("--image", default=COMPETITION_IMAGE, help="Competition image to run.")
    parser.add_argument(
        "--submission-zip",
        type=Path,
        default=SUBMISSION_ZIP_DEFAULT,
        help="Output ZIP path for the submission bundle.",
    )
    parser.add_argument(
        "--runtime-path",
        type=Path,
        default=Path("artifacts/mjai-tract-runtime"),
        help="Linux runtime binary path inside the workspace.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("artifacts/policy.onnx"),
        help="ONNX model path inside the workspace.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("artifacts/policy.json"),
        help="Metadata JSON path inside the workspace.",
    )
    parser.add_argument(
        "--skip-runtime-smoke",
        action="store_true",
        help="Skip the direct native runtime smoke test inside the competition image.",
    )
    parser.add_argument(
        "--skip-bot-smoke",
        action="store_true",
        help="Skip bot end-to-end fixture replay inside the competition image.",
    )
    parser.add_argument(
        "--seat",
        type=int,
        default=1,
        help="Seat id passed to bot.py during fixture replay.",
    )
    return parser.parse_args()


def resolve_workspace_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def submission_zip_container_path() -> str:
    return "/tmp/submission.zip"


def submission_root_container_path() -> str:
    return "/tmp/submission"


def submission_runtime_container_path() -> str:
    return f"{submission_root_container_path()}/artifacts/mjai-tract-runtime"


def submission_onnx_container_path() -> str:
    return f"{submission_root_container_path()}/artifacts/policy.onnx"


def submission_metadata_container_path() -> str:
    return f"{submission_root_container_path()}/artifacts/policy.json"


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
    description: str,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{description} failed with exit code {result.returncode}.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result


def ensure_exists(path: Path, description: str, hint: str) -> None:
    if path.exists():
        return
    raise FileNotFoundError(f"missing {description}: {path}. {hint}")


def ensure_docker_ready() -> None:
    try:
        result = subprocess.run(
            ["docker", "info"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "docker CLI is not installed or not on PATH. Install Docker Desktop or another Docker-compatible engine first."
        ) from exc

    if result.returncode == 0:
        return

    stderr = result.stderr.strip()
    if "dockerDesktopLinuxEngine" in stderr or "The system cannot find the file specified" in stderr:
        raise RuntimeError(
            "docker daemon is not running. Start Docker Desktop (Linux containers) and rerun the validation script.\n"
            f"docker info stderr:\n{stderr}"
        )

    raise RuntimeError(f"docker is unavailable. docker info stderr:\n{stderr}")


def build_submission_zip(
    submission_zip_path: Path,
    *,
    runtime_path: Path,
    onnx_path: Path,
    metadata_path: Path,
) -> None:
    submission_zip_path.parent.mkdir(parents=True, exist_ok=True)

    files_to_package = {
        ROOT / "bot.py": "bot.py",
        ROOT / "rust_mjai_bot.py": "rust_mjai_bot.py",
        ROOT / "train" / "__init__.py": "train/__init__.py",
        ROOT / "train" / "inference_spec.py": "train/inference_spec.py",
        resolve_bot_decision_path(): "artifacts/mjai-bot-decision",
        runtime_path: "artifacts/mjai-tract-runtime",
        onnx_path: "artifacts/policy.onnx",
        metadata_path: "artifacts/policy.json",
    }

    with zipfile.ZipFile(
        submission_zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for source_path, archive_name in files_to_package.items():
            archive.write(source_path, arcname=archive_name)

    size_bytes = submission_zip_path.stat().st_size
    if size_bytes >= SUBMISSION_MAX_BYTES:
        raise RuntimeError(
            f"submission zip is too large: {size_bytes} bytes. The competition limit is under 1GB."
        )


def create_container(image: str, shell_command: str) -> str:
    container_name = f"mjai-competition-validate-{uuid.uuid4().hex[:12]}"
    run_command(
        [
            "docker",
            "create",
            "-i",
            "--platform",
            "linux/amd64",
            "--name",
            container_name,
            image,
            "sh",
            "-lc",
            shell_command,
        ],
        cwd=ROOT,
        description="Creating temporary competition validation container",
    )
    return container_name


def remove_container(container_name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def copy_into_container(container_name: str, source: Path, destination: str) -> None:
    run_command(
        ["docker", "cp", str(source), f"{container_name}:{destination}"],
        cwd=ROOT,
        description=f"Copying {source.name} into container",
    )


def start_container(container_name: str, *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return run_command(
        ["docker", "start", "-ai", container_name],
        cwd=ROOT,
        input_text=input_text,
        description="Starting temporary competition validation container",
    )


def copy_submission_zip_into_container(container_name: str, submission_zip_path: Path) -> None:
    copy_into_container(container_name, submission_zip_path, submission_zip_container_path())


def runtime_shell_path() -> str:
    return shlex.quote(submission_runtime_container_path())


def onnx_shell_path() -> str:
    return shlex.quote(submission_onnx_container_path())


def metadata_shell_path() -> str:
    return shlex.quote(submission_metadata_container_path())


def extract_submission_shell_steps() -> list[str]:
    return [
        f"rm -rf {shlex.quote(submission_root_container_path())}",
        f"mkdir -p {shlex.quote(submission_root_container_path())}",
        (
            f"python3 -m zipfile -e {shlex.quote(submission_zip_container_path())} "
            f"{shlex.quote(submission_root_container_path())}"
        ),
    ]


def validate_runtime_smoke(image: str, submission_zip_path: Path) -> None:
    request = json.dumps(
        {
            "features": [0.0] * INPUT_DIM,
            "legal_actions": [index == 0 for index in range(ACTION_DIM)],
        },
        ensure_ascii=True,
    )

    shell_command = " && ".join(
        [
            *extract_submission_shell_steps(),
            (
                "python3 -c \"import os; path='"
                f"{submission_runtime_container_path()}"
                "'; mode=os.stat(path).st_mode; os.chmod(path, mode | 0o111)\""
            ),
            (
                f"printf '%s\\n' {shlex.quote(request)} | "
                f"{runtime_shell_path()} {onnx_shell_path()} {metadata_shell_path()}"
            ),
        ]
    )
    container_name = create_container(image, shell_command)
    try:
        copy_submission_zip_into_container(container_name, submission_zip_path)
        result = start_container(container_name)
    finally:
        remove_container(container_name)

    response = json.loads(result.stdout.strip())
    action = response.get("action")
    if action != 0:
        raise RuntimeError(
            "competition image runtime smoke test selected an unexpected action.\n"
            f"Response: {result.stdout}"
        )


def validate_bot_fixture(
    image: str,
    submission_zip_path: Path,
    case: BotFixtureCase,
    seat: int,
) -> None:
    shell_command = " && ".join(
        [
            *extract_submission_shell_steps(),
            f"cd {shlex.quote(submission_root_container_path())}",
            f"python3 bot.py {seat}",
        ]
    )
    container_name = create_container(image, shell_command)
    try:
        copy_submission_zip_into_container(container_name, submission_zip_path)
        result = start_container(
            container_name,
            input_text=case.fixture_path.read_text(encoding="utf-8"),
        )
    finally:
        remove_container(container_name)

    response_lines = [line for line in result.stdout.splitlines() if line.strip()]
    responses = [json.loads(line) for line in response_lines]
    if len(responses) != len(case.expected_types):
        raise RuntimeError(
            f"fixture '{case.name}' produced {len(responses)} responses, expected {len(case.expected_types)}.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    for index, (response, allowed_types) in enumerate(zip(responses, case.expected_types)):
        response_type = response.get("type")
        if response_type not in allowed_types:
            raise RuntimeError(
                f"fixture '{case.name}' response {index} had unexpected type '{response_type}'.\n"
                f"Allowed: {allowed_types}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )


def main() -> None:
    args = parse_args()
    submission_zip_path = resolve_workspace_path(args.submission_zip)
    runtime_path = resolve_workspace_path(args.runtime_path)
    onnx_path = resolve_workspace_path(args.onnx_path)
    metadata_path = resolve_workspace_path(args.metadata_path)

    ensure_docker_ready()

    ensure_exists(
        onnx_path,
        "ONNX model",
        "Run tools/export_onnx.py first.",
    )
    ensure_exists(
        metadata_path,
        "metadata JSON",
        "Run tools/export_onnx.py first.",
    )
    ensure_exists(
        runtime_path,
        "Linux native runtime binary",
        "Build it in your development environment first, then place it at artifacts/mjai-tract-runtime or pass --runtime-path.",
    )
    for case in DEFAULT_CASES:
        ensure_exists(case.fixture_path, f"fixture '{case.name}'", "The repository fixture file is missing.")

    print(f"[1/3] Building submission zip: {submission_zip_path}")
    build_submission_zip(
        submission_zip_path,
        runtime_path=runtime_path,
        onnx_path=onnx_path,
        metadata_path=metadata_path,
    )

    if not args.skip_runtime_smoke:
        print(f"[2/3] Running native runtime smoke test in {args.image}...")
        validate_runtime_smoke(args.image, submission_zip_path)
    else:
        print("[2/3] Skipping native runtime smoke test.")

    if not args.skip_bot_smoke:
        print(f"[3/3] Replaying bot fixtures in {args.image}...")
        for case in DEFAULT_CASES:
            validate_bot_fixture(args.image, submission_zip_path, case, args.seat)
            print(f"  - {case.name} fixture passed")
    else:
        print("[3/3] Skipping bot fixture replay.")

    print("Competition image validation passed.")


if __name__ == "__main__":
    main()