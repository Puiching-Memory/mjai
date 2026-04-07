from __future__ import annotations

from typing import Any


def _resolve_match_type():
    try:
        from mjai.mlibriichi.arena import Match as UpstreamMatch  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError(
            "Rust arena Match binding is not available. Build or install the Rust mjai arena binding before running self-play."
        ) from exc
    return UpstreamMatch


class Match:
    def __init__(self, *, log_dir: str | None = None):
        self._match = _resolve_match_type()(log_dir=log_dir)

    def py_match(
        self,
        agent1: Any,
        agent2: Any,
        agent3: Any,
        agent4: Any,
        *,
        seed_start: tuple[int, int],
    ):
        return self._match.py_match(agent1, agent2, agent3, agent4, seed_start=seed_start)

    def py_match_continue(
        self,
        agent1: Any,
        agent2: Any,
        agent3: Any,
        agent4: Any,
        *,
        scores: list[int],
        kyoku: int,
        honba: int,
        kyotaku: int,
        seed_start: tuple[int, int],
    ):
        return self._match.py_match_continue(
            agent1,
            agent2,
            agent3,
            agent4,
            scores=scores,
            kyoku=kyoku,
            honba=honba,
            kyotaku=kyotaku,
            seed_start=seed_start,
        )


__all__ = ["Match"]