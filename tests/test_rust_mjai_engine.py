from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from rust_mjai_engine import DockerMjaiLogEngine


class _FakePlayer:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []

    def react(self, events_json: str) -> str:
        self.calls.append(json.loads(events_json))
        return '{"type":"none"}'


class RustMjaiEngineTest(unittest.TestCase):
    def test_react_batch_redacts_hidden_information(self) -> None:
        player = _FakePlayer()
        engine = DockerMjaiLogEngine(name="test", player=player)
        engine.set_player_ids([1])

        start_kyoku = {
            "type": "start_kyoku",
            "tehais": [
                ["1m"] * 13,
                ["2m"] * 13,
                ["3m"] * 13,
                ["4m"] * 13,
            ],
        }
        events_json = json.dumps(
            [
                start_kyoku,
                {"type": "tsumo", "actor": 0, "pai": "5p"},
                {"type": "tsumo", "actor": 1, "pai": "6p"},
            ]
        )
        game_state = SimpleNamespace(game_index=0, events_json=events_json)

        self.assertEqual(engine.react_batch([game_state]), ['{"type":"none"}'])
        delivered = player.calls[0]
        self.assertEqual(delivered[0]["tehais"][0], ["?"] * 13)
        self.assertEqual(delivered[0]["tehais"][1], ["2m"] * 13)
        self.assertEqual(delivered[1]["pai"], "?")
        self.assertEqual(delivered[2]["pai"], "6p")

    def test_react_batch_returns_none_when_no_new_events(self) -> None:
        player = _FakePlayer()
        engine = DockerMjaiLogEngine(name="test", player=player)
        engine.set_player_ids([0])

        events_json = json.dumps([{"type": "start_kyoku", "tehais": [["1m"] * 13] * 4}])
        game_state = SimpleNamespace(game_index=0, events_json=events_json)

        engine.react_batch([game_state])
        self.assertEqual(engine.react_batch([game_state]), ['{"type":"none"}'])


if __name__ == "__main__":
    unittest.main()