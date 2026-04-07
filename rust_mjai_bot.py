from __future__ import annotations

import atexit
import json
import os
import stat
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
BOT_DECISION_BIN_ENV = "MJAI_BOT_DECISION_BIN"


# ---------------------------------------------------------------------------
# Game utility functions (merged from rust_mjai_game.py)
# ---------------------------------------------------------------------------


def to_rank(scores: list[int]) -> list[int]:
    adjusted_scores = scores.copy()
    adjusted_scores[0] = -adjusted_scores[0] - 0.3
    adjusted_scores[1] = -adjusted_scores[1] - 0.2
    adjusted_scores[2] = -adjusted_scores[2] - 0.1
    adjusted_scores[3] *= -1

    player_idx_by_rank = [
        idx for _, idx in sorted(zip(adjusted_scores, list(range(4))))
    ]
    player_rank_map = {
        player_idx: rank_idx for rank_idx, player_idx in enumerate(player_idx_by_rank)
    }
    return [player_rank_map[player_idx] + 1 for player_idx in range(4)]


def kyoku_to_zero_indexed_kyoku(bakaze: str, kyoku: int) -> int:
    if bakaze == "E":
        return kyoku - 1
    if bakaze == "S":
        return kyoku - 1 + 4
    return kyoku - 1 + 8


def _default_binary_candidates(
    base_dir: Path, binary_name: str, *, include_debug: bool = True
) -> list[Path]:
    candidates = [
        base_dir / "artifacts" / f"{binary_name}.exe",
        base_dir / "artifacts" / binary_name,
        base_dir / "native_runtime" / "target" / "release" / f"{binary_name}.exe",
        base_dir / "native_runtime" / "target" / "release" / binary_name,
    ]
    if include_debug:
        candidates.extend(
            [
                base_dir / "native_runtime" / "target" / "debug" / f"{binary_name}.exe",
                base_dir / "native_runtime" / "target" / "debug" / binary_name,
            ]
        )
    return candidates


def resolve_binary_path(
    description: str,
    env_name: str,
    defaults: list[Path],
) -> Path:
    configured = os.environ.get(env_name)
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"configured {description} from {env_name} does not exist: {candidate}"
        )

    for candidate in defaults:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in defaults)
    raise FileNotFoundError(
        f"missing {description}; searched: {searched}. "
        f"You can override it with {env_name}."
    )


def ensure_binary_executable(binary_path: Path) -> None:
    if os.name == "nt":
        return
    mode = binary_path.stat().st_mode
    if mode & stat.S_IXUSR:
        return
    try:
        binary_path.chmod(mode | stat.S_IXUSR)
    except OSError as exc:
        raise RuntimeError(
            f"failed to mark binary as executable: {binary_path}"
        ) from exc


def resolve_bot_decision_path(root_dir: Path | None = None) -> Path:
    base_dir = ROOT_DIR if root_dir is None else root_dir
    return resolve_binary_path(
        "Rust mjai decision binary",
        BOT_DECISION_BIN_ENV,
        _default_binary_candidates(base_dir, "mjai-bot-decision"),
    )


class SubprocessJsonClient:
    """Base class for JSON-over-stdio subprocess communication."""

    def __init__(self, args: list[str], *, label: str) -> None:
        self._label = label
        self._disabled = False
        self._process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        atexit.register(self.close)

    def _request(self, payload: object) -> dict[str, Any]:
        if self._disabled:
            raise RuntimeError(f"{self._label} has been disabled after a previous fatal error")
        if self._process.poll() is not None:
            self._disabled = True
            raise RuntimeError(f"{self._label} exited unexpectedly")

        request = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        try:
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            self._process.stdin.write(request + "\n")
            self._process.stdin.flush()
            response_line = self._process.stdout.readline()
        except OSError as exc:
            self._disabled = True
            raise RuntimeError(f"{self._label} communication failed: {exc}") from exc

        if not response_line:
            self._disabled = True
            raise RuntimeError(f"{self._label} returned no response")

        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as exc:
            self._disabled = True
            raise RuntimeError(f"{self._label} returned invalid JSON: {exc}") from exc

        if not isinstance(response, dict):
            raise RuntimeError(f"{self._label} returned a non-object response")
        return response

    def close(self) -> None:
        if getattr(self, "_process", None) is None:
            return
        if self._process.poll() is not None:
            return
        try:
            self._process.terminate()
            self._process.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            self._process.kill()


class RustMjaiStateClient(SubprocessJsonClient):
    def __init__(self, binary_path: Path, player_id: int) -> None:
        self.binary_path = binary_path
        self.player_id = player_id
        super().__init__(
            [str(binary_path), "--player-id", str(player_id)],
            label="Rust mjai decision runtime",
        )

    @classmethod
    def from_environment(cls, *, player_id: int, root_dir: Path | None = None) -> "RustMjaiStateClient":
        base_dir = ROOT_DIR if root_dir is None else root_dir
        binary_path = resolve_bot_decision_path(base_dir)
        ensure_binary_executable(binary_path)
        try:
            return cls(binary_path, player_id)
        except OSError as exc:
            raise RuntimeError(f"failed to start Rust mjai state runtime: {exc}") from exc

    def react(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        response = self._request({"kind": "react", "events": events})
        if response.get("ok") is not True:
            raise RuntimeError(str(response.get("error", "Rust mjai decision runtime failed to react")))
        snapshot = response.get("snapshot")
        if not isinstance(snapshot, dict):
            raise RuntimeError("Rust mjai decision runtime response is missing a snapshot object")
        return {"snapshot": snapshot, "decision": response.get("decision")}

    def validate_reaction(self, reaction: str) -> None:
        response = self._request({"kind": "validate_reaction", "reaction": reaction})
        if response.get("ok") is not True:
            raise RuntimeError(str(response.get("error", "reaction validation failed")))

    def brief_info(self) -> str:
        response = self._request({"kind": "brief_info"})
        if response.get("ok") is not True:
            raise RuntimeError(str(response.get("error", "brief_info failed")))
        brief_info = response.get("brief_info")
        if not isinstance(brief_info, str):
            raise RuntimeError("Rust mjai state runtime returned an invalid brief_info payload")
        return brief_info


class RustMjaiBot:
    def __init__(self, player_id: int = 0):
        self.player_id = player_id
        self._state_client = RustMjaiStateClient.from_environment(player_id=player_id, root_dir=ROOT_DIR)
        self._snapshot: dict[str, Any] | None = None
        self._last_decision: dict[str, Any] | None = None
        self.__discard_events: list[dict[str, Any]] = []
        self.__call_events: list[dict[str, Any]] = []
        self.__dora_indicators: list[str] = []

    def _require_snapshot(self) -> dict[str, Any]:
        if self._snapshot is None:
            raise RuntimeError("mjai bot state is not initialized before the first react() call")
        return self._snapshot

    def _capabilities(self) -> dict[str, Any]:
        capabilities = self._require_snapshot().get("capabilities")
        if not isinstance(capabilities, dict):
            raise RuntimeError("mjai bot snapshot is missing capabilities")
        return capabilities

    def _state(self) -> dict[str, Any]:
        state = self._require_snapshot().get("state")
        if not isinstance(state, dict):
            raise RuntimeError("mjai bot snapshot is missing state")
        return state

    def _queries(self) -> dict[str, Any]:
        queries = self._require_snapshot().get("queries")
        if not isinstance(queries, dict):
            raise RuntimeError("mjai bot snapshot is missing queries")
        return queries

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self._require_snapshot())

    @property
    def can_discard(self) -> bool:
        return bool(self._capabilities().get("can_discard", False))

    @property
    def can_riichi(self) -> bool:
        return bool(self._capabilities().get("can_riichi", False))

    @property
    def can_agari(self) -> bool:
        return bool(self._capabilities().get("can_agari", False))

    @property
    def can_tsumo_agari(self) -> bool:
        return bool(self._capabilities().get("can_tsumo_agari", False))

    @property
    def can_ron_agari(self) -> bool:
        return bool(self._capabilities().get("can_ron_agari", False))

    @property
    def can_ryukyoku(self) -> bool:
        return bool(self._capabilities().get("can_ryukyoku", False))

    @property
    def can_kakan(self) -> bool:
        return bool(self._capabilities().get("can_kakan", False))

    @property
    def can_daiminkan(self) -> bool:
        return bool(self._capabilities().get("can_daiminkan", False))

    @property
    def can_kan(self) -> bool:
        return bool(self._capabilities().get("can_kan", False))

    @property
    def can_ankan(self) -> bool:
        return bool(self._capabilities().get("can_ankan", False))

    @property
    def can_pon(self) -> bool:
        return bool(self._capabilities().get("can_pon", False))

    @property
    def can_chi(self) -> bool:
        return bool(self._capabilities().get("can_chi", False))

    @property
    def can_chi_low(self) -> bool:
        return bool(self._capabilities().get("can_chi_low", False))

    @property
    def can_chi_mid(self) -> bool:
        return bool(self._capabilities().get("can_chi_mid", False))

    @property
    def can_chi_high(self) -> bool:
        return bool(self._capabilities().get("can_chi_high", False))

    @property
    def can_act(self) -> bool:
        return bool(self._capabilities().get("can_act", False))

    @property
    def can_pass(self) -> bool:
        return bool(self._capabilities().get("can_pass", False))

    @property
    def target_actor(self) -> int:
        return int(self._state().get("target_actor", self.player_id))

    @property
    def target_actor_rel(self) -> int:
        return int(self._state().get("target_actor_rel", 0))

    def validate_reaction(self, reaction: str) -> None:
        self._state_client.validate_reaction(reaction)

    def brief_info(self) -> str:
        try:
            return self._state_client.brief_info()
        except Exception:
            if self._snapshot is None:
                return "no snapshot available"
            return json.dumps(self._snapshot, ensure_ascii=True, sort_keys=True, indent=2)

    @property
    def kyotaku(self) -> int:
        return int(self._state().get("kyotaku", 0))

    @property
    def at_furiten(self) -> bool:
        return bool(self._state().get("at_furiten", False))

    @property
    def is_oya(self) -> bool:
        return bool(self._state().get("is_oya", False))

    @property
    def last_self_tsumo(self) -> str:
        return str(self._state().get("last_self_tsumo", ""))

    @property
    def last_kawa_tile(self) -> str | None:
        value = self._state().get("last_kawa_tile")
        return None if value is None else str(value)

    @property
    def self_riichi_declared(self) -> bool:
        return bool(self._state().get("self_riichi_declared", False))

    @property
    def self_riichi_accepted(self) -> bool:
        return bool(self._state().get("self_riichi_accepted", False))

    @property
    def tehai_vec34(self) -> list[int]:
        return [int(value) for value in self._state().get("tehai_vec34", [])]

    @property
    def tehai_mjai(self) -> list[str]:
        return [str(tile) for tile in self._state().get("tehai_mjai", [])]

    @property
    def tehai(self) -> str:
        return str(self._state().get("tehai", ""))

    @property
    def akas_in_hand(self) -> list[bool]:
        return [bool(value) for value in self._state().get("akas_in_hand", [])]

    @property
    def shanten(self) -> int:
        return int(self._state().get("shanten", -1))

    @property
    def discardable_tiles(self) -> list[str]:
        return [str(tile) for tile in self._queries().get("discardable_tiles", [])]

    @property
    def discardable_tiles_riichi_declaration(self) -> list[str]:
        return [str(tile) for tile in self._queries().get("discardable_tiles_riichi_declaration", [])]

    @property
    def dora_indicators(self) -> list[str]:
        return list(self.__dora_indicators)

    def discarded_tiles(self, player_id: int | None = None) -> list[str]:
        if player_id is not None:
            return [ev["pai"] for ev in self.__discard_events if ev["actor"] == player_id]
        return [ev["pai"] for ev in self.__discard_events]

    def get_call_events(self, player_id: int | None = None) -> list[dict]:
        if player_id is not None:
            return [ev for ev in self.__call_events if ev["actor"] == player_id]
        return list(self.__call_events)

    @property
    def honba(self) -> int:
        return int(self._state().get("honba", 0))

    @property
    def kyoku(self) -> int:
        return int(self._state().get("kyoku", 0))

    @property
    def scores(self) -> list[int]:
        return [int(score) for score in self._state().get("scores", [])]

    @property
    def jikaze(self) -> str | None:
        value = self._state().get("jikaze")
        return None if value is None else str(value)

    @property
    def bakaze(self) -> str | None:
        value = self._state().get("bakaze")
        return None if value is None else str(value)

    @property
    def tiles_seen(self) -> dict[str, int]:
        return {str(tile): int(count) for tile, count in self._state().get("tiles_seen", {}).items()}

    @property
    def forbidden_tiles(self) -> dict[str, bool]:
        return {str(tile): bool(value) for tile, value in self._state().get("forbidden_tiles", {}).items()}

    def action_discard(self, tile_str: str) -> str:
        return json.dumps(
            {
                "type": "dahai",
                "pai": tile_str,
                "actor": self.player_id,
                "tsumogiri": tile_str == self.last_self_tsumo,
            },
            separators=(",", ":"),
        )

    def action_nothing(self) -> str:
        return json.dumps({"type": "none"}, separators=(",", ":"))

    def action_tsumo_agari(self) -> str:
        return json.dumps(
            {
                "type": "hora",
                "actor": self.player_id,
                "target": self.target_actor,
                "pai": self.last_self_tsumo,
            },
            separators=(",", ":"),
        )

    def action_ron_agari(self) -> str:
        return json.dumps(
            {
                "type": "hora",
                "actor": self.player_id,
                "target": self.target_actor,
                "pai": self.last_kawa_tile,
            },
            separators=(",", ":"),
        )

    def action_riichi(self) -> str:
        return json.dumps({"type": "reach", "actor": self.player_id}, separators=(",", ":"))

    def action_ankan(self, consumed: list[str]) -> str:
        return json.dumps({"type": "ankan", "actor": self.player_id, "consumed": consumed}, separators=(",", ":"))

    def action_kakan(self, pai: str) -> str:
        consumed = [pai.replace("r", "")] * 3
        if pai[0] == "5" and not pai.endswith("r"):
            consumed[0] = consumed[0] + "r"
        return json.dumps(
            {"type": "kakan", "actor": self.player_id, "pai": pai, "consumed": consumed},
            separators=(",", ":"),
        )

    def action_daiminkan(self, consumed: list[str]) -> str:
        return json.dumps(
            {
                "type": "daiminkan",
                "actor": self.player_id,
                "target": self.target_actor,
                "pai": self.last_kawa_tile,
                "consumed": consumed,
            },
            separators=(",", ":"),
        )

    def action_pon(self, consumed: list[str]) -> str:
        return json.dumps(
            {
                "type": "pon",
                "actor": self.player_id,
                "target": self.target_actor,
                "pai": self.last_kawa_tile,
                "consumed": consumed,
            },
            separators=(",", ":"),
        )

    def action_chi(self, consumed: list[str]) -> str:
        return json.dumps(
            {
                "type": "chi",
                "actor": self.player_id,
                "target": self.target_actor,
                "pai": self.last_kawa_tile,
                "consumed": consumed,
            },
            separators=(",", ":"),
        )

    def action_ryukyoku(self) -> str:
        return json.dumps({"type": "ryukyoku"}, separators=(",", ":"))

    def think(self) -> str:
        if self.can_discard:
            return self.action_discard(self.last_self_tsumo)
        return self.action_nothing()

    def _record_events(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            event_type = event.get("type")
            if event_type == "start_game":
                self.__discard_events = []
                self.__call_events = []
                self.__dora_indicators = []
            if event_type in {"start_kyoku", "dora"} and isinstance(event.get("dora_marker"), str):
                self.__dora_indicators.append(str(event["dora_marker"]))
            if event_type == "dahai":
                self.__discard_events.append(event)
            if event_type in {"chi", "pon", "daiminkan", "kakan", "ankan"}:
                self.__call_events.append(event)

    def react(self, input_str: str) -> str:
        try:
            events = json.loads(input_str)
            if not isinstance(events, list) or len(events) == 0:
                raise ValueError("Empty events")
            if any(not isinstance(event, dict) for event in events):
                raise ValueError("events must be a JSON array of objects")

            response = self._state_client.react(events)
            self._snapshot = response["snapshot"]
            self._last_decision = response.get("decision")
            self._record_events(events)

            if self.self_riichi_accepted and not (self.can_agari or self.can_kakan or self.can_ankan) and self.can_discard:
                return self.action_discard(self.last_self_tsumo)
            return self.think()
        except Exception as exc:
            print("===========================================", file=sys.stderr)
            print(f"Exception: {str(exc)}", file=sys.stderr)
            print("Brief info:", file=sys.stderr)
            print(self.brief_info(), file=sys.stderr)
            print("", file=sys.stderr)
            return json.dumps({"type": "none"}, separators=(",", ":"))

    def start(self) -> None:
        while line := sys.stdin.readline():
            line = line.strip()
            resp = self.react(line)
            sys.stdout.write(resp + "\n")
            sys.stdout.flush()

    def is_yakuhai(self, tile: str) -> bool:
        return tile in [self.jikaze, self.bakaze] or self.is_dragon(tile)

    def is_dragon(self, tile: str) -> bool:
        return tile in ["P", "F", "C"]

    def find_pon_candidates(self) -> list[dict[str, Any]]:
        return [deepcopy(candidate) for candidate in self._queries().get("pon_candidates", [])]

    def find_chi_candidates(self) -> list[dict[str, Any]]:
        return [deepcopy(candidate) for candidate in self._queries().get("chi_candidates", [])]

    def find_improving_tiles(self) -> list[dict[str, Any]]:
        return [deepcopy(candidate) for candidate in self._queries().get("improving_tiles", [])]

    @property
    def last_decision(self) -> dict[str, Any] | None:
        return self._last_decision