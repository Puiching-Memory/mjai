from __future__ import annotations

from collections import deque
from datetime import datetime
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_HISTORY_LIMIT = 64
_LOG_HISTORY_LIMIT = 512
_VISIBLE_LOG_LINES = 10
_CHART_HEIGHT = 8
_CHART_WIDTH = 36


def resolve_rich_logging(log_format: str) -> bool:
    if log_format == "rich":
        return True
    if log_format in {"json", "text"}:
        return False
    return sys.stdout.isatty()


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _trim(values: Sequence[float], limit: int = _HISTORY_LIMIT) -> list[float]:
    if len(values) <= limit:
        return list(values)
    return list(values[-limit:])


def _padded_bounds(values: Sequence[float]) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if abs(high - low) < 1.0e-12:
        pad = max(abs(low) * 0.1, 1.0)
        return low - pad, high + pad
    pad = (high - low) * 0.1
    return low - pad, high + pad


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _bresenham_points(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        twice_err = 2 * err
        if twice_err >= dy:
            err += dy
            x0 += sx
        if twice_err <= dx:
            err += dx
            y0 += sy
    return points


def _format_axis_label(value: float, digits: int) -> str:
    return f"{value:.{digits}f}" if digits > 0 else f"{value:.0f}"


def _format_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _segment_char(x0: int, y0: int, x1: int, y1: int) -> str:
    if y0 == y1:
        return "-"
    if x0 == x1:
        return "|"
    return "\\" if (x1 - x0) * (y1 - y0) > 0 else "/"


def create_rich_console(*, stderr: bool = False) -> Console:
    return Console(stderr=stderr)


def render_info_panel(
    *,
    title: str,
    rows: Sequence[tuple[str, str]],
    border_style: str = "cyan",
) -> Panel:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(style="bold cyan", width=14)
    table.add_column()
    for label, value in rows:
        table.add_row(str(label), str(value))
    return Panel(table, title=title, border_style=border_style)


def render_example_panel(
    *,
    title: str,
    rows: Sequence[tuple[str, str]],
    border_style: str = "magenta",
) -> Panel:
    if not rows:
        return Panel(Align.left(Text("no examples", style="dim")), title=title, border_style=border_style)

    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(style="bold magenta", width=14)
    table.add_column()
    for label, value in rows:
        table.add_row(str(label), str(value))
    return Panel(table, title=title, border_style=border_style)


def render_note_panel(
    *,
    title: str,
    lines: Sequence[str],
    border_style: str = "yellow",
) -> Panel:
    if not lines:
        return Panel(Align.left(Text("-", style="dim")), title=title, border_style=border_style)

    content = Group(*(Text(f"- {line}") for line in lines))
    return Panel(content, title=title, border_style=border_style)


class TrainingDashboard:
    def __init__(
        self,
        *,
        enabled: bool,
        total_steps: int,
        warmup_steps: int,
        learner_device: str,
        inference_device: str,
        actor_processes: int,
        checkpoint_path: Path,
        best_checkpoint_path: Path,
        metrics_path: Path,
        run_label: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.total_steps = max(int(total_steps), 1)
        self.warmup_steps = max(int(warmup_steps), 1)
        self.learner_device = learner_device
        self.inference_device = inference_device
        self.actor_processes = actor_processes
        self.checkpoint_path = checkpoint_path
        self.best_checkpoint_path = best_checkpoint_path
        self.metrics_path = metrics_path
        self.run_label = run_label or "custom"
        self.event_log_path = metrics_path.with_suffix(".events.log")

        self.console = Console(force_terminal=enabled)
        self.live: Live | None = None
        self.started_at = perf_counter()
        self.phase = "starting"
        self.detail = "bootstrapping runtime"
        self.current_step = 0
        self.progress_completed = 0.0
        self.progress_total = float(self.total_steps)
        self.latest_payload: dict[str, Any] | None = None
        self.latest_evaluation: dict[str, Any] | None = None
        self.events: deque[tuple[str, str, str]] = deque(maxlen=_LOG_HISTORY_LIMIT)
        self.event_count = 0
        self.history: dict[str, list[float]] = {
            "step": [],
            "loss": [],
            "decisions_per_sec": [],
            "matches_per_sec": [],
            "avg_batch_size": [],
            "avg_inference_ms": [],
        }
        self.layout = self._build_layout()

    def __enter__(self) -> TrainingDashboard:
        if not self.enabled:
            return self

        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n== training session {datetime.now().isoformat(timespec='seconds')} ==\n")
        self._update_all_panels()
        self.live = Live(self.layout, console=self.console, transient=False, auto_refresh=False)
        self.live.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.live is not None:
            self.live.stop()
            self.live = None

    def refresh(self) -> None:
        if self.live is not None:
            self.live.refresh()

    def add_event(self, message: str, *, style: str = "white", refresh: bool = True) -> None:
        if not self.enabled:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.events.append((timestamp, message, style))
        self.event_count += 1
        with self.event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")
        self._update_log_panel()
        if refresh:
            self.refresh()

    def update_status(
        self,
        *,
        phase: str,
        detail: str,
        completed: int | float,
        total: int | float,
    ) -> None:
        if not self.enabled:
            return
        self.phase = phase
        self.detail = detail
        self.progress_completed = float(max(completed, 0))
        self.progress_total = float(max(total, 1))
        self._update_overview_panel()
        self._update_phase_panel()
        self.refresh()

    def record_snapshot(self, payload: dict[str, Any], *, phase: str, detail: str) -> None:
        if not self.enabled:
            return
        self.latest_payload = payload
        self.latest_evaluation = payload.get("evaluation")
        self.current_step = int(payload.get("step", self.current_step))
        self.phase = phase
        self.detail = detail
        self.progress_completed = float(self.current_step)
        self.progress_total = float(self.total_steps)
        self._append_history(payload)
        self._update_all_panels()
        self.refresh()

    def record_evaluation(self, evaluation: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.latest_evaluation = evaluation
        candidate = evaluation["metrics"].get("candidate") or {}
        baseline = evaluation["metrics"].get("baseline") or {}
        self.add_event(
            "evaluation done"
            f" | cand rank {_format_float(candidate.get('average_rank'), digits=3)}"
            f" | base rank {_format_float(baseline.get('average_rank'), digits=3)}"
            f" | {'best updated' if evaluation.get('improved') else 'best kept'}",
            style="green" if evaluation.get("improved") else "yellow",
            refresh=False,
        )
        self._update_overview_panel()
        self._update_metrics_panel()
        self.refresh()

    def finish(self, *, final_step: int) -> None:
        if not self.enabled:
            return
        self.current_step = final_step
        self.phase = "done"
        self.detail = "training complete"
        self.progress_completed = float(self.total_steps)
        self.progress_total = float(self.total_steps)
        self.add_event("training complete", style="bold green", refresh=False)
        self._update_all_panels()
        self.refresh()

    def _append_history(self, payload: dict[str, Any]) -> None:
        actors = payload.get("actors") or {}
        inference = payload.get("inference") or {}
        learner = payload.get("learner") or {}
        self.history["step"].append(float(payload.get("step", 0)))
        self.history["loss"].append(float(learner.get("loss", 0.0)))
        self.history["decisions_per_sec"].append(float(actors.get("decisions_per_sec", 0.0)))
        self.history["matches_per_sec"].append(float(actors.get("matches_per_sec", 0.0)))
        self.history["avg_batch_size"].append(float(inference.get("avg_batch_size", 0.0)))
        self.history["avg_inference_ms"].append(float(inference.get("avg_inference_ms", 0.0)))

    def _build_layout(self) -> Layout:
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="summary", size=9),
            Layout(name="log", size=11),
            Layout(name="charts", ratio=1),
        )
        layout["summary"].split_row(
            Layout(name="overview", ratio=6),
            Layout(name="metrics", ratio=7),
            Layout(name="phase", ratio=4),
        )
        layout["charts"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )
        layout["left"].split_column(
            Layout(name="throughput_chart"),
            Layout(name="loss_chart"),
        )
        layout["right"].split_column(
            Layout(name="batch_chart"),
            Layout(name="latency_chart"),
        )
        return layout

    def _update_all_panels(self) -> None:
        self._update_overview_panel()
        self._update_metrics_panel()
        self._update_phase_panel()
        self._update_log_panel()
        self._update_chart_panels()

    def _update_overview_panel(self) -> None:
        self.layout["overview"].update(self._render_overview())

    def _update_metrics_panel(self) -> None:
        self.layout["metrics"].update(self._render_metrics())

    def _update_phase_panel(self) -> None:
        self.layout["phase"].update(self._render_phase())

    def _update_log_panel(self) -> None:
        self.layout["log"].update(self._render_event_log())

    def _update_chart_panels(self) -> None:
        self.layout["throughput_chart"].update(
            self._render_history_chart(
                title="Throughput",
                series=[
                    ("d/s", self.history["decisions_per_sec"], "cyan"),
                    ("m/s", self.history["matches_per_sec"], "green"),
                ],
                digits=2,
                border_style="bright_cyan",
            )
        )
        self.layout["loss_chart"].update(
            self._render_history_chart(
                title="Loss",
                series=[("loss", self.history["loss"], "magenta")],
                digits=4,
                border_style="magenta",
            )
        )
        self.layout["batch_chart"].update(
            self._render_history_chart(
                title="Batch Size",
                series=[("batch", self.history["avg_batch_size"], "yellow")],
                digits=2,
                border_style="yellow",
            )
        )
        self.layout["latency_chart"].update(
            self._render_history_chart(
                title="Inference Ms",
                series=[("ms", self.history["avg_inference_ms"], "red")],
                digits=2,
                border_style="red",
            )
        )

    def _render_overview(self) -> Panel:
        elapsed = perf_counter() - self.started_at
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style="bold cyan", width=10)
        table.add_column(no_wrap=True)
        table.add_column(style="bold cyan", width=10)
        table.add_column(no_wrap=True)
        table.add_row("run", self.run_label, "phase", self.phase)
        table.add_row("step", f"{self.current_step}/{self.total_steps}", "elapsed", _format_elapsed(elapsed))
        table.add_row("learner", self.learner_device, "inference", self.inference_device)
        table.add_row("actors", str(self.actor_processes), "warmup", str(self.warmup_steps))
        table.add_row("ckpt", self.checkpoint_path.name, "best", self.best_checkpoint_path.name)
        return Panel(table, title="Run", border_style="cyan")

    def _render_metrics(self) -> Panel:
        if self.latest_payload is None:
            placeholder = Align.left(Text("waiting for first learner snapshot", style="dim"))
            return Panel(placeholder, title="Latest Metrics", border_style="green")

        learner = self.latest_payload.get("learner") or {}
        replay = self.latest_payload.get("replay") or {}
        actors = self.latest_payload.get("actors") or {}
        inference = self.latest_payload.get("inference") or {}
        evaluation = self.latest_evaluation or {}
        candidate = (evaluation.get("metrics") or {}).get("candidate") or {}
        baseline = (evaluation.get("metrics") or {}).get("baseline") or {}

        table = Table.grid(expand=True, padding=(0, 1))
        for _ in range(6):
            table.add_column()
        table.add_row(
            "loss",
            _format_float(learner.get("loss")),
            "policy",
            _format_float(learner.get("policy_loss")),
            "value",
            _format_float(learner.get("value_loss")),
        )
        table.add_row(
            "entropy",
            _format_float(learner.get("entropy")),
            "kl",
            _format_float(learner.get("approx_kl")),
            "grad",
            _format_float(learner.get("grad_norm"), digits=3),
        )
        table.add_row(
            "replay",
            f"{int(replay.get('steps', 0))}/{int(replay.get('capacity', 0))}",
            "fresh",
            str(int(replay.get("fresh_steps", 0))),
            "credit",
            str(int(replay.get("growth_credit", 0))),
        )
        table.add_row(
            "actor d/s",
            _format_float(actors.get("decisions_per_sec"), digits=2),
            "actor m/s",
            _format_float(actors.get("matches_per_sec"), digits=2),
            "actor total",
            str(int(actors.get("decisions_total", 0))),
        )
        table.add_row(
            "infer batch",
            _format_float(inference.get("avg_batch_size"), digits=2),
            "infer ms",
            _format_float(inference.get("avg_inference_ms"), digits=2),
            "max batch",
            str(int(inference.get("max_batch_size", 0))),
        )
        table.add_row(
            "eval rank",
            self._format_eval_metric(candidate, baseline, "average_rank", digits=3),
            "eval score",
            self._format_eval_metric(candidate, baseline, "average_score", digits=0),
            "best",
            "updated" if evaluation.get("improved") else ("kept" if evaluation else "-"),
        )
        return Panel(table, title="Latest Metrics", border_style="green")

    def _format_eval_metric(
        self,
        candidate: dict[str, Any],
        baseline: dict[str, Any],
        key: str,
        *,
        digits: int,
    ) -> str:
        candidate_text = _format_float(candidate.get(key), digits=digits)
        if not baseline:
            return candidate_text
        return f"{candidate_text} / {_format_float(baseline.get(key), digits=digits)}"

    def _render_phase(self) -> Panel:
        pct = 100.0 * self.progress_completed / max(self.progress_total, 1.0)
        pct = max(0.0, min(pct, 100.0))
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style="bold cyan")
        table.add_column()
        table.add_row("phase", self.phase)
        table.add_row(
            "progress",
            f"{int(self.progress_completed)}/{int(self.progress_total)} ({pct:.1f}%)",
        )
        table.add_row("detail", self.detail)

        bar_width = 20
        filled = round(bar_width * pct / 100.0)
        bar = Text("[", style="dim")
        bar.append("=" * filled, style="bold cyan")
        bar.append("-" * max(bar_width - filled, 0), style="grey35")
        bar.append("]", style="dim")

        return Panel(Group(table, bar), title="Phase", border_style="cyan")

    def _render_event_log(self) -> Panel:
        if not self.events:
            placeholder = Align.left(Text("waiting for training events", style="dim"))
            return Panel(placeholder, title="Recent Log", border_style="yellow")

        lines = []
        for timestamp, message, style in list(self.events)[-_VISIBLE_LOG_LINES:]:
            line = Text()
            line.append(timestamp, style="dim")
            line.append(" ")
            line.append(message, style=style)
            lines.append(line)
        footer = Text(
            f"showing {min(len(self.events), _VISIBLE_LOG_LINES)}/{self.event_count} events | "
            f"full history: {self.event_log_path.name}",
            style="dim",
        )
        return Panel(Group(*lines, footer), title="Recent Log", border_style="yellow")

    def _render_history_chart(
        self,
        *,
        title: str,
        series: list[tuple[str, Sequence[float], str]],
        digits: int,
        border_style: str,
    ) -> Panel:
        populated = [(name, _trim(values), color) for name, values, color in series if values]
        if not populated:
            placeholder = Align.left(Text("waiting for samples", style="dim"))
            return Panel(placeholder, title=title, border_style=border_style)

        trimmed_steps = _trim(self.history["step"])
        bounds_source = [value for _, values, _ in populated for value in values]
        lower, upper = _padded_bounds(bounds_source)
        lines = self._build_chart_lines(
            steps=trimmed_steps,
            series=populated,
            lower=lower,
            upper=upper,
            digits=digits,
        )

        legends: list[Text] = []
        if trimmed_steps:
            legends.append(Text(f"steps {int(trimmed_steps[0])}->{int(trimmed_steps[-1])}", style="dim"))
        legend = Text()
        for index, (name, values, color) in enumerate(populated):
            if index > 0:
                legend.append("  ")
            legend.append("#", style=f"bold {color}")
            legend.append(f" {name} {_format_float(values[-1], digits=digits)}")
        legends.append(legend)
        return Panel(Group(*lines, *legends), title=title, border_style=border_style)

    def _build_chart_lines(
        self,
        *,
        steps: Sequence[float],
        series: list[tuple[str, Sequence[float], str]],
        lower: float,
        upper: float,
        digits: int,
    ) -> list[Text]:
        width = max(_CHART_WIDTH, 2)
        height = max(_CHART_HEIGHT, 2)
        grid = [[(" ", "") for _ in range(width)] for _ in range(height)]

        def put_cell(x_pos: int, y_pos: int, char: str, style: str) -> None:
            if 0 <= x_pos < width and 0 <= y_pos < height:
                grid[y_pos][x_pos] = (char, style)

        span = max(upper - lower, 1.0e-12)
        step_count = max(len(steps), 1)

        for _, values, color in series:
            coordinates: list[tuple[int, int]] = []
            for index, value in enumerate(values):
                if step_count == 1:
                    x_pos = 0
                else:
                    x_pos = round(index * (width - 1) / max(len(values) - 1, 1))
                normalized = (float(value) - lower) / span
                y_pos = round((1.0 - normalized) * (height - 1))
                coordinates.append((x_pos, _clamp(y_pos, 0, height - 1)))

            if len(coordinates) == 1:
                put_cell(coordinates[0][0], coordinates[0][1], "*", f"bold {color}")
                continue

            for start, end in zip(coordinates, coordinates[1:]):
                segment = _bresenham_points(start[0], start[1], end[0], end[1])
                for point_index, (x_pos, y_pos) in enumerate(segment[:-1]):
                    next_x, next_y = segment[point_index + 1]
                    put_cell(x_pos, y_pos, _segment_char(x_pos, y_pos, next_x, next_y), f"bold {color}")
                put_cell(segment[-1][0], segment[-1][1], "*", f"bold {color}")

        label_rows = {
            0: upper,
            height // 2: (lower + upper) / 2.0,
            height - 1: lower,
        }
        rendered: list[Text] = []
        for row_index, row in enumerate(grid):
            label = label_rows.get(row_index)
            line = Text()
            if label is None:
                line.append(" " * 7, style="dim")
            else:
                line.append(f"{_format_axis_label(label, digits):>7}", style="dim")
            line.append(" ")
            for char, style in row:
                line.append(char, style=style)
            rendered.append(line)
        return rendered