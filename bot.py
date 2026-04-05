import atexit
import json
import math
import os
import stat
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from mjai import Bot
from train.inference_spec import (
    ACTION_DIM,
    CANDIDATE_FEATURE_DIM,
    INPUT_DIM,
    MAX_ACTION_CANDIDATES,
    TILE_TYPES,
    action_type_one_hot,
    base_tile,
    tile_histogram,
    tile_one_hot,
    tile_sort_key,
)

logger.remove()
logger.add(
    sys.stderr,
    level=os.environ.get("MJAI_LOG_LEVEL", "INFO").upper(),
    format="<green>{time:HH:mm:ss}</green> <level>{level} {message}</level>",
)  # noqa

HONOR_TILES = {"E", "S", "W", "N", "P", "F", "C"}
WIND_TILES = ["E", "S", "W", "N"]
DRAGON_TILES = ["P", "F", "C"]
TERMINAL_OR_HONOR_TILES = {
    "1m",
    "9m",
    "1p",
    "9p",
    "1s",
    "9s",
    "E",
    "S",
    "W",
    "N",
    "P",
    "F",
    "C",
}
ROOT_DIR = Path(__file__).resolve().parent
NATIVE_RUNTIME_BIN_ENV = "MJAI_NATIVE_RUNTIME_BIN"
NATIVE_RUNTIME_ONNX_ENV = "MJAI_NATIVE_RUNTIME_ONNX"
NATIVE_RUNTIME_META_ENV = "MJAI_NATIVE_RUNTIME_META"


@dataclass(slots=True)
class ActionCandidate:
    action_type: str
    action_label: str
    primary_tile: str | None
    discard_tile: str | None
    consumed_tiles: tuple[str, ...]
    next_shanten: int
    next_ukeire: int
    ukeire: int
    improving_count: int
    discard_candidate_count: int
    baseline_score: int
    discard_bonus: int
    tile_seen: int
    tile_count: int
    tile_dora: int
    is_tsumogiri: bool


class NativeRuntimeClient:
    def __init__(self, binary_path: Path, model_path: Path, metadata_path: Path) -> None:
        self.binary_path = binary_path
        self.model_path = model_path
        self.metadata_path = metadata_path
        self._disabled = False
        self._process = subprocess.Popen(
            [str(binary_path), str(model_path), str(metadata_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        atexit.register(self.close)

    @classmethod
    def from_environment(cls, root_dir: Path) -> "NativeRuntimeClient":
        binary_path = cls._resolve_required_path(
            "native runtime binary",
            NATIVE_RUNTIME_BIN_ENV,
            [
                root_dir / "artifacts" / "mjai-tract-runtime.exe",
                root_dir / "artifacts" / "mjai-tract-runtime",
                root_dir / "native_runtime" / "target" / "release" / "mjai-tract-runtime.exe",
                root_dir / "native_runtime" / "target" / "release" / "mjai-tract-runtime",
            ],
        )
        model_path = cls._resolve_required_path(
            "native runtime ONNX model",
            NATIVE_RUNTIME_ONNX_ENV,
            [root_dir / "artifacts" / "policy.onnx"],
        )
        metadata_path = cls._resolve_required_path(
            "native runtime metadata",
            NATIVE_RUNTIME_META_ENV,
            [root_dir / "artifacts" / "policy.json"],
        )

        cls._ensure_binary_executable(binary_path)

        try:
            runtime = cls(binary_path, model_path, metadata_path)
        except OSError as exc:
            raise RuntimeError(f"failed to start native runtime: {exc}") from exc

        logger.info(
            "native action runtime enabled: "
            f"{binary_path.name} | {model_path.name} | {metadata_path.name}"
        )
        return runtime

    @staticmethod
    def _resolve_required_path(
        description: str, env_name: str, defaults: list[Path]
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

    @staticmethod
    def _ensure_binary_executable(binary_path: Path) -> None:
        if os.name == "nt":
            return

        mode = binary_path.stat().st_mode
        if mode & stat.S_IXUSR:
            return

        try:
            binary_path.chmod(mode | stat.S_IXUSR)
        except OSError as exc:
            raise RuntimeError(
                f"failed to mark native runtime binary as executable: {binary_path}"
            ) from exc

    def infer(self, features: list[float], legal_actions: list[bool]) -> dict:
        if self._disabled:
            raise RuntimeError("native runtime has been disabled after a previous fatal error")
        if self._process.poll() is not None:
            self._disabled = True
            raise RuntimeError("native runtime exited unexpectedly")

        payload = json.dumps(
            {"features": features, "legal_actions": legal_actions},
            ensure_ascii=True,
            separators=(",", ":"),
        )

        try:
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            self._process.stdin.write(payload + "\n")
            self._process.stdin.flush()
            response_line = self._process.stdout.readline()
        except OSError as exc:
            self._disabled = True
            raise RuntimeError(f"native runtime communication failed: {exc}") from exc

        if not response_line:
            self._disabled = True
            raise RuntimeError("native runtime returned no response")

        try:
            return json.loads(response_line)
        except json.JSONDecodeError as exc:
            self._disabled = True
            raise RuntimeError(f"native runtime returned invalid JSON: {exc}") from exc

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


class BasicMahjongBot(Bot):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._native_runtime = NativeRuntimeClient.from_environment(ROOT_DIR)
        self._last_action_source = "native"

    def think(self) -> str:
        if self.can_tsumo_agari:
            return self.action_tsumo_agari()
        if self.can_ron_agari:
            return self.action_ron_agari()

        if self.can_ryukyoku and self._should_abort_nine_terminals():
            return self.action_ryukyoku()

        if self._is_riichi_discard_phase():
            candidate = self._select_native_action_candidate()
            discard_tile = candidate.discard_tile
            logger.info(
                f"{self.bakaze}{self.kyoku}-{self.honba}: riichi-discard {self.tehai} | {self.last_self_tsumo} -> {discard_tile} [{self._last_action_source}]"  # noqa
            )
            return self.action_discard(discard_tile)

        if self.can_riichi:
            return self.action_riichi()

        if self.can_pon or self.can_chi:
            candidate = self._select_native_action_candidate()
            logger.info(
                f"{self.bakaze}{self.kyoku}-{self.honba}: call-choice {self.last_kawa_tile} -> {candidate.action_label} [{self._last_action_source}]"  # noqa
            )
            return self._action_from_candidate(candidate)

        if self.can_discard:
            if self.self_riichi_accepted:
                logger.info(
                    f"{self.bakaze}{self.kyoku}-{self.honba}: {self.tehai} | {self.last_self_tsumo} -> {self.last_self_tsumo} [riichi-tsumogiri]"  # noqa
                )
                return self.action_discard(self.last_self_tsumo)

            candidate = self._select_native_action_candidate()
            discard_tile = candidate.discard_tile
            logger.info(
                f"{self.bakaze}{self.kyoku}-{self.honba}: {self.tehai} | {self.last_self_tsumo} -> {discard_tile} [{self._last_action_source}]"  # noqa
            )
            return self.action_discard(discard_tile)

        return self.action_nothing()

    def _is_riichi_discard_phase(self) -> bool:
        return bool(
            self.can_discard
            and getattr(self, "self_riichi_declared", False)
            and not self.self_riichi_accepted
        )

    def _select_native_action_candidate(self) -> ActionCandidate:
        candidates = self._build_action_candidates()
        if not candidates:
            raise RuntimeError("no legal action candidates were produced for native runtime")

        features, legal_actions = self._build_runtime_features(candidates)
        response = self._native_runtime.infer(features, legal_actions)

        action = response.get("action")
        if not isinstance(action, int):
            raise RuntimeError("native runtime returned a non-integer action index")
        if action < 0 or action >= len(candidates):
            raise RuntimeError(f"native runtime returned out-of-range action index: {action}")
        if not legal_actions[action]:
            raise RuntimeError(f"native runtime selected a masked action: {action}")

        candidate = candidates[action]
        self._validate_native_action_candidate(candidate)
        self._last_action_source = "native"
        return candidate

    def _validate_native_action_candidate(self, candidate: ActionCandidate) -> None:
        if candidate.action_type in {"discard", "riichi_discard"}:
            if candidate.discard_tile is None:
                raise RuntimeError("native runtime selected a discard action without a discard tile")
            if self.forbidden_tiles.get(base_tile(candidate.discard_tile), True):
                raise RuntimeError(
                    f"native runtime selected a forbidden tile: {candidate.discard_tile}"
                )
            return

        if candidate.action_type == "pon" and not self.can_pon:
            raise RuntimeError("native runtime selected pon when pon is not legal")
        if candidate.action_type == "chi" and not self.can_chi:
            raise RuntimeError("native runtime selected chi when chi is not legal")
        if candidate.action_type == "pass" and not (self.can_pon or self.can_chi):
            raise RuntimeError("native runtime selected pass when no optional call exists")

    def _action_from_candidate(self, candidate: ActionCandidate) -> str:
        if candidate.action_type == "pass":
            return self.action_nothing()
        if candidate.action_type in {"discard", "riichi_discard"}:
            assert candidate.discard_tile is not None
            return self.action_discard(candidate.discard_tile)
        if candidate.action_type == "pon":
            return self.action_pon(consumed=list(candidate.consumed_tiles))
        if candidate.action_type == "chi":
            return self.action_chi(consumed=list(candidate.consumed_tiles))
        raise RuntimeError(f"unsupported action type selected by native runtime: {candidate.action_type}")

    def _build_action_candidates(self) -> list[ActionCandidate]:
        if self._is_riichi_discard_phase():
            return self._build_riichi_discard_action_candidates()
        if self.can_pon or self.can_chi:
            return self._build_call_action_candidates()
        if self.can_discard:
            return self._build_discard_action_candidates()
        return []

    def _build_discard_action_candidates(self) -> list[ActionCandidate]:
        selected_by_tile: dict[str, ActionCandidate] = {}
        tile_counts = Counter(base_tile(hand_tile) for hand_tile in self.tehai_mjai)

        for candidate in self.find_improving_tiles():
            discard_tile = candidate["discard_tile"]
            if not discard_tile:
                continue
            if self.forbidden_tiles.get(base_tile(discard_tile), True):
                continue

            compiled = self._compile_discard_action_candidate(
                candidate,
                tile_counts,
                action_type="discard",
            )
            current = selected_by_tile.get(discard_tile)
            if current is None or compiled.baseline_score > current.baseline_score:
                selected_by_tile[discard_tile] = compiled

        if not selected_by_tile:
            for tile in sorted(self.tehai_mjai, key=tile_sort_key):
                if self.forbidden_tiles.get(base_tile(tile), True):
                    continue
                if tile in selected_by_tile:
                    continue
                selected_by_tile[tile] = self._compile_fallback_discard_action_candidate(
                    tile,
                    tile_counts,
                    action_type="discard",
                )

        candidates = sorted(
            selected_by_tile.values(),
            key=lambda candidate: tile_sort_key(candidate.discard_tile),
        )
        return candidates[:MAX_ACTION_CANDIDATES]

    def _build_riichi_discard_action_candidates(self) -> list[ActionCandidate]:
        tile_counts = Counter(base_tile(hand_tile) for hand_tile in self.tehai_mjai)
        improving_by_tile = {
            candidate["discard_tile"]: candidate
            for candidate in self.find_improving_tiles()
            if candidate.get("discard_tile")
        }

        candidates: list[ActionCandidate] = []
        for discard_tile in sorted(self.discardable_tiles_riichi_declaration, key=tile_sort_key):
            if self.forbidden_tiles.get(base_tile(discard_tile), True):
                continue
            improving_candidate = improving_by_tile.get(discard_tile)
            if improving_candidate is not None:
                candidates.append(
                    self._compile_discard_action_candidate(
                        improving_candidate,
                        tile_counts,
                        action_type="riichi_discard",
                    )
                )
            else:
                candidates.append(
                    self._compile_fallback_discard_action_candidate(
                        discard_tile,
                        tile_counts,
                        action_type="riichi_discard",
                    )
                )

        return candidates[:MAX_ACTION_CANDIDATES]

    def _build_call_action_candidates(self) -> list[ActionCandidate]:
        candidates = [
            ActionCandidate(
                action_type="pass",
                action_label="pass",
                primary_tile=self.last_kawa_tile,
                discard_tile=None,
                consumed_tiles=(),
                next_shanten=self.shanten,
                next_ukeire=self._current_best_ukeire(),
                ukeire=self._current_best_ukeire(),
                improving_count=0,
                discard_candidate_count=0,
                baseline_score=0,
                discard_bonus=0,
                tile_seen=self.tiles_seen.get(base_tile(self.last_kawa_tile), 0),
                tile_count=0,
                tile_dora=self._tile_dora_value(self.last_kawa_tile),
                is_tsumogiri=False,
            )
        ]

        if self.can_pon:
            for candidate in self.find_pon_candidates():
                if self._should_call(candidate, action_type="pon"):
                    candidates.append(self._compile_call_action_candidate(candidate, "pon"))

        if self.can_chi:
            for candidate in self.find_chi_candidates():
                if self._should_call(candidate, action_type="chi"):
                    candidates.append(self._compile_call_action_candidate(candidate, "chi"))

        passthrough = candidates[0]
        call_candidates = sorted(
            candidates[1:],
            key=lambda candidate: (
                candidate.action_type,
                [tile_sort_key(tile) for tile in candidate.consumed_tiles],
            ),
        )
        return [passthrough, *call_candidates[: MAX_ACTION_CANDIDATES - 1]]

    def _compile_discard_action_candidate(
        self, candidate: dict, tile_counts: Counter[str], action_type: str
    ) -> ActionCandidate:
        discard_tile = candidate["discard_tile"]
        discard_bonus = self._tile_discard_bonus(discard_tile)
        return ActionCandidate(
            action_type=action_type,
            action_label=discard_tile,
            primary_tile=discard_tile,
            discard_tile=discard_tile,
            consumed_tiles=(),
            next_shanten=self.shanten,
            next_ukeire=int(candidate.get("ukeire", 0)),
            ukeire=int(candidate.get("ukeire", 0)),
            improving_count=len(candidate.get("improving_tiles", [])),
            discard_candidate_count=0,
            baseline_score=self._discard_candidate_score(candidate),
            discard_bonus=discard_bonus,
            tile_seen=self.tiles_seen.get(base_tile(discard_tile), 0),
            tile_count=tile_counts[base_tile(discard_tile)],
            tile_dora=self._tile_dora_value(discard_tile),
            is_tsumogiri=discard_tile == self.last_self_tsumo,
        )

    def _compile_fallback_discard_action_candidate(
        self, discard_tile: str, tile_counts: Counter[str], action_type: str
    ) -> ActionCandidate:
        discard_bonus = self._tile_discard_bonus(discard_tile)
        baseline_score = discard_bonus + self.tiles_seen.get(base_tile(discard_tile), 0) * 6
        if discard_tile == self.last_self_tsumo:
            baseline_score += 1
        return ActionCandidate(
            action_type=action_type,
            action_label=discard_tile,
            primary_tile=discard_tile,
            discard_tile=discard_tile,
            consumed_tiles=(),
            next_shanten=self.shanten,
            next_ukeire=0,
            ukeire=0,
            improving_count=0,
            discard_candidate_count=0,
            baseline_score=baseline_score,
            discard_bonus=discard_bonus,
            tile_seen=self.tiles_seen.get(base_tile(discard_tile), 0),
            tile_count=tile_counts[base_tile(discard_tile)],
            tile_dora=self._tile_dora_value(discard_tile),
            is_tsumogiri=discard_tile == self.last_self_tsumo,
        )

    def _compile_call_action_candidate(self, candidate: dict, action_type: str) -> ActionCandidate:
        consumed_tiles = tuple(sorted(candidate["consumed"], key=tile_sort_key))
        primary_tile = self.last_kawa_tile
        primary_base_tile = base_tile(primary_tile)
        hand_counts = Counter(base_tile(tile) for tile in self.tehai_mjai)
        return ActionCandidate(
            action_type=action_type,
            action_label=f"{action_type}:{'/'.join(consumed_tiles)}",
            primary_tile=primary_tile,
            discard_tile=None,
            consumed_tiles=consumed_tiles,
            next_shanten=int(candidate.get("next_shanten", self.shanten)),
            next_ukeire=int(candidate.get("next_ukeire", 0)),
            ukeire=int(candidate.get("next_ukeire", 0)),
            improving_count=0,
            discard_candidate_count=len(candidate.get("discard_candidates", [])),
            baseline_score=self._call_candidate_score(candidate, action_type),
            discard_bonus=0,
            tile_seen=self.tiles_seen.get(primary_base_tile, 0),
            tile_count=hand_counts.get(primary_base_tile, 0),
            tile_dora=self._tile_dora_value(primary_tile),
            is_tsumogiri=False,
        )

    def _build_runtime_features(
        self, candidates: list[ActionCandidate]
    ) -> tuple[list[float], list[bool]]:
        features: list[float] = []
        features.extend(self._global_runtime_features())

        legal_actions = [False] * ACTION_DIM
        for index in range(MAX_ACTION_CANDIDATES):
            if index < len(candidates):
                legal_actions[index] = True
                features.extend(self._candidate_runtime_features(candidates[index]))
            else:
                features.extend([0.0] * CANDIDATE_FEATURE_DIM)

        if len(features) != INPUT_DIM:
            raise ValueError(
                f"runtime feature vector length {len(features)} does not match expected {INPUT_DIM}"
            )
        return features, legal_actions

    def _global_runtime_features(self) -> list[float]:
        hand_counts = Counter(base_tile(tile) for tile in self.tehai_mjai)
        dora_targets = Counter(
            self._dora_from_indicator(indicator) for indicator in self.dora_indicators
        )

        features = [
            self.shanten / 6.0,
            min(self._current_best_ukeire(), 40) / 40.0,
            min(self._total_dora_in_hand(), 13) / 13.0,
            float(self._has_open_hand()),
            float(self.self_riichi_accepted),
            float(self.can_riichi),
            max(self.kyoku - 1, 0) / 3.0,
            min(self.honba, 10) / 10.0,
            min(getattr(self, "kyotaku", 0), 10) / 10.0,
        ]

        features.extend(1.0 if self.bakaze == wind else 0.0 for wind in WIND_TILES)
        features.extend(
            1.0 if self.player_id == player_index else 0.0 for player_index in range(4)
        )
        features.extend(hand_counts.get(tile, 0) / 4.0 for tile in TILE_TYPES)
        features.extend(min(self.tiles_seen.get(tile, 0), 4) / 4.0 for tile in TILE_TYPES)
        features.extend(min(dora_targets.get(tile, 0), 4) / 4.0 for tile in TILE_TYPES)
        return features

    def _candidate_runtime_features(self, candidate: ActionCandidate) -> list[float]:
        primary_tile = candidate.primary_tile or candidate.discard_tile
        if primary_tile is None:
            raise ValueError("action candidate is missing a primary tile")

        features = [1.0]
        features.extend(action_type_one_hot(candidate.action_type))
        features.extend(
            [
                max(min(candidate.next_shanten, 6), 0) / 6.0,
                min(candidate.ukeire, 40) / 40.0,
                min(candidate.improving_count, 34) / 34.0,
                min(candidate.discard_candidate_count, MAX_ACTION_CANDIDATES)
                / float(MAX_ACTION_CANDIDATES),
                math.tanh(candidate.baseline_score / 200.0),
                math.tanh(candidate.discard_bonus / 80.0),
                min(candidate.tile_seen, 4) / 4.0,
                min(candidate.tile_dora, 4) / 4.0,
                float(candidate.is_tsumogiri),
            ]
        )
        features.extend(tile_one_hot(primary_tile))
        features.extend(tile_histogram(candidate.consumed_tiles))
        return features

    def _should_call(self, candidate: dict, action_type: str) -> bool:
        current_shanten = self.shanten
        current_ukeire = self._current_best_ukeire()
        next_shanten = candidate["next_shanten"]
        next_ukeire = candidate["next_ukeire"]
        has_open_hand = self._has_open_hand()
        is_value_pon = action_type == "pon" and self.is_yakuhai(self.last_kawa_tile[:2])

        if next_shanten > current_shanten:
            return False

        if next_shanten < current_shanten:
            if is_value_pon:
                return True
            if has_open_hand:
                return next_ukeire + 2 >= current_ukeire
            if action_type == "pon":
                return next_shanten == 0 and next_ukeire >= max(6, current_ukeire - 2)
            return next_shanten == 0 and next_ukeire >= max(8, current_ukeire)

        if is_value_pon and next_shanten <= 1 and next_ukeire >= current_ukeire + 2:
            return True

        if has_open_hand and next_shanten == 0 and next_ukeire >= current_ukeire + 4:
            return True

        if has_open_hand and next_shanten <= 1 and next_ukeire >= current_ukeire + 6:
            return True

        return False

    def _call_candidate_score(self, candidate: dict, action_type: str) -> int:
        score = 0
        score += (self.shanten - candidate["next_shanten"]) * 100
        score += candidate["next_ukeire"] * 10
        score += len(candidate["discard_candidates"])

        if action_type == "pon" and self.is_yakuhai(self.last_kawa_tile[:2]):
            score += 25
        if candidate["next_shanten"] == 0:
            score += 20
        if self._has_open_hand():
            score += 10

        return score

    def _discard_candidate_score(self, candidate: dict) -> int:
        discard_tile = candidate["discard_tile"]
        normalized = base_tile(discard_tile)

        score = 0
        score += candidate["ukeire"] * 100
        score += len(candidate["improving_tiles"]) * 3
        score += self.tiles_seen.get(normalized, 0) * 6
        score += self._tile_discard_bonus(discard_tile)

        if discard_tile == self.last_self_tsumo:
            score += 1

        return score

    def _tile_discard_bonus(self, tile: str) -> int:
        tile_counts = Counter(base_tile(hand_tile) for hand_tile in self.tehai_mjai)
        normalized = base_tile(tile)
        tile_count = tile_counts[normalized]

        score = 0
        score -= self._tile_dora_value(tile) * 60

        if tile.endswith("r"):
            score -= 40

        if normalized in HONOR_TILES:
            if self.is_yakuhai(normalized):
                score -= 35 if tile_count >= 2 else 15
            elif tile_count >= 2:
                score -= 12
            else:
                score += 28

            score += self.tiles_seen.get(normalized, 0) * 4
            return score

        number = int(normalized[0])
        score += {
            1: 18,
            2: 10,
            3: 4,
            4: 0,
            5: -4,
            6: 0,
            7: 4,
            8: 10,
            9: 18,
        }[number]

        if tile_count >= 2:
            score -= 12

        score += self._tile_isolation_bonus(normalized, tile_counts)
        return score

    def _tile_isolation_bonus(self, tile: str, tile_counts: Counter[str]) -> int:
        if tile in HONOR_TILES:
            return 0

        number = int(tile[0])
        color = tile[1]
        close_connections = 0
        wide_connections = 0

        for offset in (-1, 1):
            neighbor = number + offset
            if 1 <= neighbor <= 9 and tile_counts.get(f"{neighbor}{color}", 0):
                close_connections += 1

        for offset in (-2, 2):
            neighbor = number + offset
            if 1 <= neighbor <= 9 and tile_counts.get(f"{neighbor}{color}", 0):
                wide_connections += 1

        if close_connections == 0 and wide_connections == 0:
            return 26
        if close_connections == 0:
            return 14
        if close_connections == 1:
            return 6 - wide_connections * 2
        return -8

    def _should_abort_nine_terminals(self) -> bool:
        distinct_terminal_tiles = {
            base_tile(tile)
            for tile in self.tehai_mjai
            if base_tile(tile) in TERMINAL_OR_HONOR_TILES
        }
        return len(distinct_terminal_tiles) >= 9

    def _current_best_ukeire(self) -> int:
        candidates = self.find_improving_tiles()
        if not candidates:
            return 0
        return max(candidate["ukeire"] for candidate in candidates)

    def _has_open_hand(self) -> bool:
        return any(
            event["type"] in {"chi", "pon", "daiminkan", "kakan"}
            for event in self.get_call_events(self.player_id)
        )

    def _total_dora_in_hand(self) -> int:
        return sum(self._tile_dora_value(tile) for tile in self.tehai_mjai)

    def _tile_dora_value(self, tile: str) -> int:
        normalized = base_tile(tile)
        value = 1 if tile.endswith("r") else 0
        value += sum(
            1
            for indicator in self.dora_indicators
            if self._dora_from_indicator(indicator) == normalized
        )
        return value

    def _dora_from_indicator(self, indicator: str) -> str:
        normalized = base_tile(indicator)

        if normalized in WIND_TILES:
            return WIND_TILES[(WIND_TILES.index(normalized) + 1) % len(WIND_TILES)]
        if normalized in DRAGON_TILES:
            return DRAGON_TILES[(DRAGON_TILES.index(normalized) + 1) % len(DRAGON_TILES)]

        number = int(normalized[0])
        color = normalized[1]
        next_number = 1 if number == 9 else number + 1
        return f"{next_number}{color}"


if __name__ == "__main__":
    BasicMahjongBot(player_id=int(sys.argv[1])).start()
