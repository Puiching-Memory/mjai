from __future__ import annotations

import unittest
from unittest.mock import patch

from rust_mjai_arena import Match


class _FakeUpstreamMatch:
    def __init__(self, *, log_dir: str | None = None) -> None:
        self.log_dir = log_dir
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def py_match(self, *args, **kwargs):
        self.calls.append(("py_match", args, kwargs))
        return [1, 1, 1, 1]

    def py_match_continue(self, *args, **kwargs):
        self.calls.append(("py_match_continue", args, kwargs))
        return [0, 0, 0, 1]


class RustMjaiArenaTest(unittest.TestCase):
    def test_match_delegates_to_localized_arena_binding(self) -> None:
        with patch("rust_mjai_arena._resolve_match_type", return_value=_FakeUpstreamMatch):
            match = Match(log_dir="/tmp/logs")
            result = match.py_match("a", "b", "c", "d", seed_start=(1, 2))

        self.assertEqual(result, [1, 1, 1, 1])
        self.assertEqual(match._match.log_dir, "/tmp/logs")
        self.assertEqual(match._match.calls[0][0], "py_match")
        self.assertEqual(match._match.calls[0][1], ("a", "b", "c", "d"))
        self.assertEqual(match._match.calls[0][2], {"seed_start": (1, 2)})

    def test_match_continue_delegates_all_arguments(self) -> None:
        with patch("rust_mjai_arena._resolve_match_type", return_value=_FakeUpstreamMatch):
            match = Match(log_dir=None)
            result = match.py_match_continue(
                "a",
                "b",
                "c",
                "d",
                scores=[25000, 25000, 25000, 25000],
                kyoku=4,
                honba=1,
                kyotaku=2,
                seed_start=(3, 4),
            )

        self.assertEqual(result, [0, 0, 0, 1])
        self.assertEqual(match._match.calls[0][0], "py_match_continue")
        self.assertEqual(
            match._match.calls[0][2],
            {
                "scores": [25000, 25000, 25000, 25000],
                "kyoku": 4,
                "honba": 1,
                "kyotaku": 2,
                "seed_start": (3, 4),
            },
        )


if __name__ == "__main__":
    unittest.main()