from __future__ import annotations

import json


class BaseMjaiLogEngine:
    def __init__(self, name: str):
        self.engine_type = "mjai-log"
        self.name = name
        self.player_ids: list[int] = []

    def set_player_ids(self, player_ids: list[int]) -> None:
        self.player_ids = player_ids

    def react_batch(self, game_states):
        responses = []
        for game_state in game_states:
            game_idx = game_state.game_index
            state = game_state.state
            events_json = game_state.events_json

            events = json.loads(events_json)
            assert events[0]["type"] == "start_kyoku"

            player_id = self.player_ids[game_idx]
            cans = state.last_cans
            if cans.can_discard:
                tile = state.last_self_tsumo()
                responses.append(
                    json.dumps(
                        {
                            "type": "dahai",
                            "actor": player_id,
                            "pai": tile,
                            "tsumogiri": True,
                        }
                    )
                )
            else:
                responses.append('{"type":"none"}')

        return responses

    def start_game(self, game_idx: int):
        return None

    def end_kyoku(self, game_idx: int):
        return None

    def end_game(self, game_idx: int, scores: list[int]):
        return None


class DockerMjaiLogEngine(BaseMjaiLogEngine):
    def __init__(self, name: str, player):
        super().__init__(name)
        self.engine_type = "mjai-log"
        self.player = player
        self.player_ids: list[int] = []
        self.last_event_idx = 0
        self.player_id = 0

    def react_batch(self, game_states):
        events = []
        for game_state in game_states:
            self.player_id = self.player_ids[game_state.game_index]
            events_json = game_state.events_json
            events += json.loads(events_json)

        if self.last_event_idx > len(events):
            self.last_event_idx = 0

        event_buffer = []
        for event in events[self.last_event_idx :]:
            if event["type"] == "tsumo" and event["actor"] != self.player_id:
                event["pai"] = "?"
            if event["type"] == "start_kyoku":
                for player_id in range(4):
                    if self.player_id != player_id:
                        event["tehais"][player_id] = ["?"] * 13
            event_buffer.append(event)

        if len(event_buffer) == 0:
            self.last_event_idx = 0
            return ['{"type":"none"}']

        self.last_event_idx = len(events)
        response = self.player.react(
            json.dumps(event_buffer, indent=0, separators=(",", ":")).replace("\n", "")
        )
        json.loads(response)
        return [response]

    def start_game(self, game_idx: int) -> None:
        events = [{"type": "start_game", "names": ["0", "1", "2", "3"], "id": game_idx}]
        self.player.react(json.dumps(events, indent=0, separators=(",", ":")).replace("\n", ""))

    def end_kyoku(self, game_idx: int):
        events = [{"type": "end_kyoku"}]
        self.player.react(json.dumps(events, indent=0, separators=(",", ":")).replace("\n", ""))
        self.last_event_idx = 0


class InProcessMjaiBotEngine(DockerMjaiLogEngine):
    def end_game(self, game_idx: int, scores: list[int]):
        if hasattr(self.player, "on_game_end"):
            self.player.on_game_end(scores)
        events = [{"type": "end_game"}]
        self.player.react(json.dumps(events, indent=0, separators=(",", ":")).replace("\n", ""))
        self.last_event_idx = 0