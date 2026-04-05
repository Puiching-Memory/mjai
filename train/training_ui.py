from __future__ import annotations

from collections import deque
from datetime import datetime
import sys
from pathlib import Path
from typing import Any, Sequence

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

_HISTORY_LIMIT = 64
_LOG_LIMIT = 10
_CHART_HEIGHT = 8
_CHART_WIDTH = 34


def resolve_rich_logging(log_format: str) -> bool:
    if log_format == "rich":
        return True
    if log_format == "json":
        return False
    return sys.stdout.isatty()


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _format_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100.0:.1f}%"


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


class TrainingDashboard:
    def __init__(
        self,
        *,
        enabled: bool,
        total_iterations: int,
        device: str,
        checkpoint_path: Path,
        best_checkpoint_path: Path,
        metrics_path: Path,
    ) -> None:
        self.enabled = enabled
        self.total_iterations = total_iterations
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.best_checkpoint_path = best_checkpoint_path
        self.metrics_path = metrics_path

        self.console = Console(force_terminal=enabled)
        self.live: Live | None = None
        self.progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[bright_black]{task.fields[detail]}", justify="right"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            expand=True,
        )
        self.phase_task = self.progress.add_task("waiting", total=1, completed=0, detail="-")

        self.run_iteration = 0
        self.checkpoint_step = 0
        self.phase = "waiting"
        self.detail = "-"
        self.examples = 0
        self.improved: bool | None = None
        self.self_play_summary: dict[str, dict[str, Any]] = {}
        self.train_metrics: dict[str, float] = {}
        self.evaluation_metrics: dict[str, dict[str, Any]] = {}
        self.history: dict[str, list[float]] = {
            "step": [],
            "loss": [],
            "entropy": [],
            "candidate_rank": [],
            "baseline_rank": [],
            "candidate_score": [],
            "baseline_score": [],
        }
        self.events: deque[tuple[str, str, str]] = deque(maxlen=_LOG_LIMIT)
        self.layout = self._build_layout()

    def __enter__(self) -> TrainingDashboard:
        self._update_all_panels()
        if self.enabled:
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

    def start_iteration(self, *, run_iteration: int, checkpoint_step: int) -> None:
        self.run_iteration = run_iteration
        self.checkpoint_step = checkpoint_step
        self.add_event(
            f"iteration {run_iteration}/{self.total_iterations} started | checkpoint step {checkpoint_step}",
            style="bold cyan",
            refresh=False,
        )
        self._update_overview_panel()
        self.refresh()

    def set_phase(self, phase: str, *, total: int, completed: int = 0, detail: str = "-", announce: bool = True) -> None:
        self.phase = phase
        self.detail = detail
        self.progress.update(
            self.phase_task,
            description=phase,
            total=max(total, 1),
            completed=min(completed, max(total, 1)),
            detail=detail,
        )
        if announce:
            self.add_event(f"phase {phase} | {detail}", style="yellow", refresh=False)
        self._update_overview_panel()
        self._update_progress_panel()
        self.refresh()

    def update_phase_progress(self, completed: int, total: int, *, detail: str | None = None) -> None:
        if detail is not None:
            self.detail = detail
        self.progress.update(
            self.phase_task,
            total=max(total, 1),
            completed=min(completed, max(total, 1)),
            detail=self.detail,
        )
        self._update_overview_panel()
        self._update_progress_panel()
        self.refresh()

    def record_self_play(self, summary: dict[str, Any], examples: int) -> None:
        self.self_play_summary = {
            key: dict(value)
            for key, value in summary.items()
        }
        self.examples = examples
        candidate = self.self_play_summary.get("candidate", {})
        self.add_event(
            "self-play done"
            f" | examples {examples}"
            f" | avg rank {_format_float(candidate.get('average_rank'), digits=3)}"
            f" | avg score {_format_float(candidate.get('average_score'), digits=0)}",
            style="green",
            refresh=False,
        )
        self._update_overview_panel()
        self._update_metrics_panel()
        self.refresh()

    def record_train(self, metrics: dict[str, float]) -> None:
        self.train_metrics = dict(metrics)
        self.add_event(
            f"optimize done | loss {_format_float(metrics.get('loss'))} | entropy {_format_float(metrics.get('entropy'))}",
            style="magenta",
            refresh=False,
        )
        self._update_metrics_panel()
        self.refresh()

    def record_evaluation(self, metrics: dict[str, Any], improved: bool) -> None:
        self.evaluation_metrics = {
            key: dict(value)
            for key, value in metrics.items()
        }
        self.improved = improved
        candidate = self.evaluation_metrics.get("candidate", {})
        baseline = self.evaluation_metrics.get("baseline", {})
        self.add_event(
            "evaluation done"
            f" | cand {_format_float(candidate.get('average_rank'), digits=3)}"
            f" vs base {_format_float(baseline.get('average_rank'), digits=3)}"
            f" | {'best updated' if improved else 'kept current best'}",
            style="cyan" if improved else "red",
            refresh=False,
        )
        self._update_overview_panel()
        self._update_metrics_panel()
        self.refresh()

    def record_iteration_history(self, metrics_row: dict[str, Any]) -> None:
        candidate_eval = metrics_row["evaluation"].get("candidate") or {}
        baseline_eval = metrics_row["evaluation"].get("baseline") or {}
        self.history["step"].append(float(metrics_row["iteration"]))
        self.history["loss"].append(float(metrics_row["train"]["loss"]))
        self.history["entropy"].append(float(metrics_row["train"]["entropy"]))
        self.history["candidate_rank"].append(float(candidate_eval.get("average_rank", 0.0)))
        self.history["baseline_rank"].append(float(baseline_eval.get("average_rank", 0.0)))
        self.history["candidate_score"].append(float(candidate_eval.get("average_score", 0.0)))
        self.history["baseline_score"].append(float(baseline_eval.get("average_score", 0.0)))
        self._update_chart_panels()
        self.refresh()

    def finish(self) -> None:
        self.phase = "done"
        self.detail = "training complete"
        self.progress.update(
            self.phase_task,
            description="done",
            total=1,
            completed=1,
            detail="training complete",
        )
        self.add_event("training complete", style="bold green", refresh=False)
        self._update_overview_panel()
        self._update_progress_panel()
        self.refresh()

    def log_iteration_summary(self, payload: dict[str, Any]) -> None:
        candidate = payload.get("candidate_eval") or {}
        baseline = payload.get("baseline_eval") or {}
        parts = [
            f"step {payload['iteration']}",
            f"loss {_format_float(payload.get('loss'))}",
            f"entropy {_format_float(payload.get('entropy'))}",
            f"cand.rank {_format_float(candidate.get('average_rank'), digits=3)}",
        ]
        if baseline:
            parts.append(f"base.rank {_format_float(baseline.get('average_rank'), digits=3)}")
        parts.append(f"improved {'yes' if payload.get('improved') else 'no'}")
        self.add_event(" | ".join(parts), style="bold cyan")

    def add_event(self, message: str, *, style: str = "white", refresh: bool = True) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.events.append((timestamp, message, style))
        self._update_log_panel()
        if refresh:
            self.refresh()

    def _build_layout(self) -> Layout:
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="summary", size=11),
            Layout(name="status", size=10),
            Layout(name="charts", ratio=1),
        )
        layout["summary"].split_row(
            Layout(name="overview"),
            Layout(name="metrics"),
        )
        layout["status"].split_row(
            Layout(name="progress", size=44),
            Layout(name="log", ratio=1),
        )
        layout["charts"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )
        layout["left"].split_column(
            Layout(name="rank_chart"),
            Layout(name="score_chart"),
        )
        layout["right"].split_column(
            Layout(name="loss_chart"),
            Layout(name="entropy_chart"),
        )
        return layout

    def _update_all_panels(self) -> None:
        self._update_overview_panel()
        self._update_metrics_panel()
        self._update_progress_panel()
        self._update_log_panel()
        self._update_chart_panels()

    def _update_overview_panel(self) -> None:
        self.layout["overview"].update(self._render_overview())

    def _update_metrics_panel(self) -> None:
        self.layout["metrics"].update(self._render_metrics())

    def _update_progress_panel(self) -> None:
        self.layout["progress"].update(self._render_progress())

    def _update_log_panel(self) -> None:
        self.layout["log"].update(self._render_event_log())

    def _update_chart_panels(self) -> None:
        self.layout["rank_chart"].update(
            self._render_history_chart(
                title="Rank Trend",
                series=[
                    ("candidate", self.history["candidate_rank"], "cyan"),
                    ("baseline", self.history["baseline_rank"], "white"),
                ],
                digits=3,
                border_style="bright_cyan",
                invert=True,
                fixed_bounds=(1.0, 4.0),
            )
        )
        self.layout["score_chart"].update(
            self._render_history_chart(
                title="Score Trend",
                series=[
                    ("candidate", self.history["candidate_score"], "green"),
                    ("baseline", self.history["baseline_score"], "yellow"),
                ],
                digits=0,
                border_style="green",
            )
        )
        self.layout["loss_chart"].update(
            self._render_history_chart(
                title="Loss Trend",
                series=[("loss", self.history["loss"], "magenta")],
                digits=4,
                border_style="magenta",
            )
        )
        self.layout["entropy_chart"].update(
            self._render_history_chart(
                title="Entropy Trend",
                series=[("entropy", self.history["entropy"], "bright_yellow")],
                digits=4,
                border_style="yellow",
            )
        )

    def _render_overview(self) -> Panel:
        table = Table.grid(expand=True, padding=(0, 2))
        table.add_column(style="bold cyan")
        table.add_column()
        table.add_column(style="bold cyan")
        table.add_column()
        table.add_row("run", f"{self.run_iteration}/{self.total_iterations}", "step", str(self.checkpoint_step))
        table.add_row("phase", self.phase, "device", self.device)
        table.add_row("examples", str(self.examples) if self.examples else "-", "improved", self._format_improved())
        table.add_row("checkpoint", self.checkpoint_path.name, "best", self.best_checkpoint_path.name)
        table.add_row("metrics", self.metrics_path.name, "detail", self.detail)
        return Panel(table, title="Training", border_style="blue")

    def _render_metrics(self) -> Panel:
        table = Table(
            show_header=True,
            expand=True,
            box=box.SIMPLE_HEAVY,
            show_edge=False,
            pad_edge=False,
        )
        table.add_column("metric", style="bold cyan")
        table.add_column("candidate", justify="right")
        table.add_column("baseline", justify="right")

        selfplay_candidate = self.self_play_summary.get("candidate", {})
        eval_candidate = self.evaluation_metrics.get("candidate", {})
        eval_baseline = self.evaluation_metrics.get("baseline", {})

        table.add_row("selfplay rank", _format_float(selfplay_candidate.get("average_rank"), digits=3), "-")
        table.add_row("selfplay score", _format_float(selfplay_candidate.get("average_score"), digits=0), "-")
        table.add_row("train loss", _format_float(self.train_metrics.get("loss")), "-")
        table.add_row("train entropy", _format_float(self.train_metrics.get("entropy")), "-")
        table.add_row(
            "eval rank",
            _format_float(eval_candidate.get("average_rank"), digits=3),
            _format_float(eval_baseline.get("average_rank"), digits=3),
        )
        table.add_row(
            "eval score",
            _format_float(eval_candidate.get("average_score"), digits=0),
            _format_float(eval_baseline.get("average_score"), digits=0),
        )
        table.add_row(
            "top1 rate",
            _format_pct(eval_candidate.get("top1_rate")),
            _format_pct(eval_baseline.get("top1_rate")),
        )
        table.add_row(
            "last rate",
            _format_pct(eval_candidate.get("last_rate")),
            _format_pct(eval_baseline.get("last_rate")),
        )
        return Panel(table, title="Latest Metrics", border_style="green")

    def _render_progress(self) -> Panel:
        return Panel(self.progress, title="Phase Progress", border_style="cyan")

    def _render_event_log(self) -> Panel:
        if not self.events:
            return Panel(Align.left(Text("waiting for training events", style="dim")), title="Log", border_style="yellow")

        lines = []
        for timestamp, message, style in self.events:
            line = Text()
            line.append(timestamp, style="dim")
            line.append(" ")
            line.append(message, style=style)
            lines.append(line)
        return Panel(Group(*lines), title="Log", border_style="yellow")

    def _render_history_chart(
        self,
        *,
        title: str,
        series: list[tuple[str, list[float], str]],
        digits: int,
        border_style: str,
        invert: bool = False,
        fixed_bounds: tuple[float, float] | None = None,
    ) -> Panel:
        trimmed_steps = _trim(self.history["step"])
        trimmed_series = [
            (name, _trim(values), color)
            for name, values, color in series
        ]

        populated = [(name, values, color) for name, values, color in trimmed_series if values]
        if not populated:
            return Panel(
                Align.center(Text("waiting for completed iteration", style="dim"), vertical="middle"),
                title=title,
                border_style=border_style,
            )

        all_values = [value for _, values, _ in populated for value in values]
        low, high = fixed_bounds if fixed_bounds is not None else _padded_bounds(all_values)
        lines = self._build_chart_lines(
            steps=trimmed_steps,
            series=populated,
            low=low,
            high=high,
            digits=digits,
            invert=invert,
        )

        legend = Text()
        if trimmed_steps:
            legend.append(f"steps {int(trimmed_steps[0])}->{int(trimmed_steps[-1])}", style="dim")
        for index, (name, values, color) in enumerate(populated):
            if index == 0 and trimmed_steps:
                legend.append("  ")
            elif index > 0:
                legend.append("  ")
            legend.append("●", style=f"bold {color}")
            legend.append(
                f" {name} {_format_float(values[-1], digits=digits)}",
                style="white",
            )

        return Panel(Group(*lines, legend), title=title, border_style=border_style)

    def _build_chart_lines(
        self,
        *,
        steps: list[float],
        series: list[tuple[str, list[float], str]],
        low: float,
        high: float,
        digits: int,
        invert: bool,
    ) -> list[Text]:
        width = _CHART_WIDTH
        height = _CHART_HEIGHT
        chars = [[" " for _ in range(width)] for _ in range(height)]
        styles = [["white" for _ in range(width)] for _ in range(height)]
        guide_rows = {0, height // 2, height - 1}

        for row in guide_rows:
            for col in range(width):
                chars[row][col] = "─"
                styles[row][col] = "grey27"

        span = max(high - low, 1.0e-12)
        point_count = max((len(values) for _, values, _ in series), default=0)

        def x_for_index(index: int, count: int) -> int:
            if count <= 1:
                return width // 2
            return round(index * (width - 1) / (count - 1))

        def row_for_value(value: float) -> int:
            normalized = (value - low) / span
            if invert:
                normalized = 1.0 - normalized
            return _clamp(height - 1 - round(normalized * (height - 1)), 0, height - 1)

        def put_cell(x: int, y: int, char: str, style: str) -> None:
            current = chars[y][x]
            if current == " " or current == "─":
                chars[y][x] = char
                styles[y][x] = style
                return
            if current == char and styles[y][x] == style:
                return
            chars[y][x] = "◉"
            styles[y][x] = "bold white"

        for _, values, color in series:
            count = len(values)
            coordinates = [
                (x_for_index(index, count), row_for_value(value))
                for index, value in enumerate(values)
            ]
            if len(coordinates) == 1:
                x_pos, y_pos = coordinates[0]
                put_cell(x_pos, y_pos, "●", f"bold {color}")
                continue

            for start, end in zip(coordinates, coordinates[1:]):
                segment = _bresenham_points(start[0], start[1], end[0], end[1])
                for point_index, (x_pos, y_pos) in enumerate(segment):
                    endpoint = point_index == 0 or point_index == len(segment) - 1
                    put_cell(x_pos, y_pos, "●" if endpoint else "•", f"bold {color}")

        label_rows = {
            0: high,
            height // 2: (high + low) / 2.0,
            height - 1: low,
        }

        lines: list[Text] = []
        for row in range(height):
            label = _format_axis_label(label_rows[row], digits) if row in label_rows else ""
            line = Text(f"{label:>9} ", style="dim")
            line.append("┤" if row in guide_rows else "│", style="dim")
            for col in range(width):
                line.append(chars[row][col], style=styles[row][col])
            lines.append(line)

        axis = Text(" " * 10 + "└" + "─" * width, style="dim")
        labels = Text(" " * 11, style="dim")
        if steps:
            left = str(int(steps[0]))
            right = str(int(steps[-1]))
            labels.append(left)
            gap = max(width - len(left) - len(right), 1)
            labels.append(" " * gap)
            if len(steps) > 1:
                labels.append(right)
        lines.extend([axis, labels])
        return lines

    def _format_improved(self) -> str:
        if self.improved is None:
            return "-"
        return "yes" if self.improved else "no"