from __future__ import annotations

import unittest

from bot import ActionCandidate
from tools.build_mjai_dataset import (
    DecisionPoint,
    _parse_json_payload,
    candidate_matches_event,
    match_logged_action_index,
    redact_event_for_player,
)


def _candidate(
    *,
    action_type: str,
    discard_tile: str | None = None,
    consumed_tiles: tuple[str, ...] = (),
    is_tsumogiri: bool = False,
) -> ActionCandidate:
    return ActionCandidate(
        action_type=action_type,
        action_label=action_type,
        primary_tile=None,
        discard_tile=discard_tile,
        consumed_tiles=consumed_tiles,
        next_shanten=0,
        next_ukeire=0,
        ukeire=0,
        improving_count=0,
        discard_candidate_count=0,
        baseline_score=0,
        discard_bonus=0,
        tile_seen=0,
        tile_count=0,
        tile_dora=0,
        is_tsumogiri=is_tsumogiri,
    )


class BuildMjaiDatasetTest(unittest.TestCase):
    def test_parse_json_payload_accepts_json_lines(self) -> None:
        payload = '{"type":"start_game"}\n{"type":"end_game"}\n'
        parsed = _parse_json_payload(payload)

        self.assertEqual(parsed, [{"type": "start_game"}, {"type": "end_game"}])

    def test_redact_event_for_player_masks_hidden_information(self) -> None:
        start_kyoku = {
            "type": "start_kyoku",
            "tehais": [
                ["1m"] * 13,
                ["2m"] * 13,
                ["3m"] * 13,
                ["4m"] * 13,
            ],
        }
        redacted = redact_event_for_player(start_kyoku, player_id=1)

        self.assertEqual(redacted["tehais"][0], ["?"] * 13)
        self.assertEqual(redacted["tehais"][1], ["2m"] * 13)
        self.assertEqual(redacted["tehais"][2], ["?"] * 13)
        self.assertEqual(start_kyoku["tehais"][0], ["1m"] * 13)

        hidden_tsumo = {"type": "tsumo", "actor": 0, "pai": "5p"}
        visible_tsumo = {"type": "tsumo", "actor": 1, "pai": "6p"}
        self.assertEqual(redact_event_for_player(hidden_tsumo, player_id=1)["pai"], "?")
        self.assertEqual(redact_event_for_player(visible_tsumo, player_id=1)["pai"], "6p")

    def test_candidate_matches_event_for_discards_and_calls(self) -> None:
        discard = _candidate(action_type="discard", discard_tile="5p", is_tsumogiri=False)
        riichi_discard = _candidate(
            action_type="riichi_discard",
            discard_tile="9p",
            is_tsumogiri=True,
        )
        chi = _candidate(action_type="chi", consumed_tiles=("2m", "3m"))

        self.assertTrue(
            candidate_matches_event(
                discard,
                "discard",
                {"type": "dahai", "actor": 1, "pai": "5p", "tsumogiri": False},
                1,
            )
        )
        self.assertTrue(
            candidate_matches_event(
                riichi_discard,
                "riichi_discard",
                {"type": "dahai", "actor": 1, "pai": "9p", "tsumogiri": True},
                1,
            )
        )
        self.assertTrue(
            candidate_matches_event(
                chi,
                "call",
                {"type": "chi", "actor": 1, "consumed": ["3m", "2m"]},
                1,
            )
        )

    def test_match_logged_action_index_falls_back_to_pass_only_when_no_claim_happened(self) -> None:
        decision = DecisionPoint(
            phase="call",
            candidates=[
                _candidate(action_type="pass"),
                _candidate(action_type="pon", consumed_tiles=("8m", "8m")),
                _candidate(action_type="chi", consumed_tiles=("2m", "3m")),
            ],
            features=[],
            legal_actions=[],
        )

        pass_index = match_logged_action_index(
            decision,
            {"type": "reach_accepted", "actor": 3},
            player_id=1,
        )
        chi_index = match_logged_action_index(
            decision,
            {"type": "chi", "actor": 1, "consumed": ["3m", "2m"]},
            player_id=1,
        )
        daiminkan_index = match_logged_action_index(
            decision,
            {"type": "daiminkan", "actor": 1, "consumed": ["8m", "8m", "8m"]},
            player_id=1,
        )

        self.assertEqual(pass_index, 0)
        self.assertEqual(chi_index, 2)
        self.assertIsNone(daiminkan_index)


if __name__ == "__main__":
    unittest.main()