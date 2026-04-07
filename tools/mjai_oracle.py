from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mjai import Bot

from train.inference_spec import base_tile, tile_sort_key


def _normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_normalize_json(item) for item in value]
    if isinstance(value, list):
        return [_normalize_json(item) for item in value]
    return value


def _safe_call(callback: Callable[[], Any]) -> Any:
    try:
        return _normalize_json(callback())
    except Exception as exc:  # pragma: no cover - oracle should preserve failure detail
        return {
            "__error__": type(exc).__name__,
            "message": str(exc),
        }


def _optional_call(callback: Callable[[], Any], default: Any) -> Any:
    try:
        return _normalize_json(callback())
    except Exception:
        return default


def _read_fixture_lines(fixture_path: Path) -> list[str]:
    return [line.strip() for line in fixture_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _infer_player_id(lines: list[str]) -> int:
    for line in lines:
        events = json.loads(line)
        for event in events:
            if event.get("type") != "start_kyoku":
                continue
            visible_players = []
            for player_id, tehai in enumerate(event.get("tehais", [])):
                if any(tile != "?" for tile in tehai):
                    visible_players.append(player_id)
            if len(visible_players) == 1:
                return visible_players[0]
    raise ValueError("failed to infer player_id from fixture; expected exactly one visible hand")


class OracleProbeBot(Bot):
    def phase(self) -> str:
        if self.can_discard and self.self_riichi_declared and not self.self_riichi_accepted:
            return "riichi_discard"
        if self.can_pon or self.can_chi:
            return "call"
        if self.can_discard:
            return "discard"
        return "idle"

    def snapshot(self, reaction: str) -> dict[str, Any]:
        capabilities = {
            "can_act": _safe_call(lambda: self.can_act),
            "can_agari": _safe_call(lambda: self.can_agari),
            "can_ankan": _safe_call(lambda: self.can_ankan),
            "can_chi": _safe_call(lambda: self.can_chi),
            "can_chi_high": _safe_call(lambda: self.can_chi_high),
            "can_chi_low": _safe_call(lambda: self.can_chi_low),
            "can_chi_mid": _safe_call(lambda: self.can_chi_mid),
            "can_daiminkan": _safe_call(lambda: self.can_daiminkan),
            "can_discard": _safe_call(lambda: self.can_discard),
            "can_kakan": _safe_call(lambda: self.can_kakan),
            "can_kan": _safe_call(lambda: self.can_kan),
            "can_pass": _safe_call(lambda: self.can_pass),
            "can_pon": _safe_call(lambda: self.can_pon),
            "can_riichi": _safe_call(lambda: self.can_riichi),
            "can_ron_agari": _safe_call(lambda: self.can_ron_agari),
            "can_ryukyoku": _safe_call(lambda: self.can_ryukyoku),
            "can_tsumo_agari": _safe_call(lambda: self.can_tsumo_agari),
        }
        can_discard = capabilities["can_discard"] is True
        can_pon = capabilities["can_pon"] is True
        can_chi = capabilities["can_chi"] is True

        state = {
            "player_id": self.player_id,
            "phase": _safe_call(self.phase),
            "reaction": reaction,
            "bakaze": _optional_call(lambda: self.bakaze, None),
            "jikaze": _optional_call(lambda: self.jikaze, None),
            "honba": _safe_call(lambda: self.honba),
            "kyoku": _safe_call(lambda: self.kyoku),
            "kyotaku": _safe_call(lambda: self.kyotaku),
            "scores": _safe_call(lambda: self.scores),
            "target_actor": _optional_call(lambda: self.target_actor, None),
            "target_actor_rel": _optional_call(lambda: self.target_actor_rel, None),
            "last_self_tsumo": _optional_call(lambda: self.last_self_tsumo, ""),
            "last_kawa_tile": _optional_call(lambda: self.last_kawa_tile, ""),
            "self_riichi_declared": _safe_call(lambda: self.self_riichi_declared),
            "self_riichi_accepted": _safe_call(lambda: self.self_riichi_accepted),
            "at_furiten": _safe_call(lambda: self.at_furiten),
            "is_oya": _safe_call(lambda: self.is_oya),
            "akas_in_hand": _safe_call(lambda: self.akas_in_hand),
            "tehai": _optional_call(lambda: self.tehai, ""),
            "tehai_mjai": _optional_call(lambda: self.tehai_mjai, []),
            "tehai_vec34": _optional_call(lambda: self.tehai_vec34, []),
            "shanten": _optional_call(lambda: self.shanten, None),
            "dora_indicators": _safe_call(lambda: self.dora_indicators),
            "tiles_seen": _safe_call(lambda: self.tiles_seen),
            "forbidden_tiles": _safe_call(lambda: self.forbidden_tiles),
            "discarded_tiles_all": _safe_call(lambda: self.discarded_tiles()),
            "discarded_tiles_self": _safe_call(lambda: self.discarded_tiles(self.player_id)),
            "call_events_all": _safe_call(lambda: self.get_call_events()),
            "call_events_self": _safe_call(lambda: self.get_call_events(self.player_id)),
        }

        queries = {
            "discardable_tiles": _safe_call(
                lambda: sorted(self.discardable_tiles, key=tile_sort_key)
            ),
            "discardable_tiles_riichi_declaration": _safe_call(
                lambda: sorted(self.discardable_tiles_riichi_declaration, key=tile_sort_key)
            ),
            "improving_tiles": _safe_call(self.find_improving_tiles) if can_discard else [],
            "pon_candidates": _safe_call(self.find_pon_candidates) if can_pon else [],
            "chi_candidates": _safe_call(self.find_chi_candidates) if can_chi else [],
        }

        return {
            "capabilities": capabilities,
            "state": state,
            "queries": queries,
        }


def build_fixture_transcript(fixture_path: Path) -> dict[str, Any]:
    lines = _read_fixture_lines(fixture_path)
    player_id = _infer_player_id(lines)
    bot = OracleProbeBot(player_id=player_id)
    steps = []

    for step_index, line in enumerate(lines):
        reaction = bot.react(line)
        steps.append(
            {
                "step_index": step_index,
                "events": json.loads(line),
                "snapshot": bot.snapshot(reaction),
            }
        )

    return {
        "fixture": str(fixture_path),
        "player_id": player_id,
        "steps": steps,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the installed Python mjai package as an oracle over fixture event streams."
    )
    parser.add_argument(
        "--fixture",
        required=True,
        type=Path,
        help="Path to a JSONL fixture where each line is an mjai event batch.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    transcript = build_fixture_transcript(args.fixture)
    print(json.dumps(transcript, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())