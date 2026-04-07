import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from rust_mjai_bot import (
    ROOT_DIR,
    RustMjaiBot as Bot,
    SubprocessJsonClient,
    _default_binary_candidates,
    ensure_binary_executable,
    resolve_binary_path,
)
from train.inference_spec import (
    ACTION_DIM,
    INPUT_DIM,
    base_tile,
)

logger.remove()
logger.add(
    sys.stderr,
    level=os.environ.get("MJAI_LOG_LEVEL", "INFO").upper(),
    format="<green>{time:HH:mm:ss}</green> <level>{level} {message}</level>",
)  # noqa

TERMINAL_OR_HONOR_TILES = {
    "1m", "9m", "1p", "9p", "1s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
}
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


class NativeRuntimeClient(SubprocessJsonClient):
    def __init__(self, binary_path: Path, model_path: Path, metadata_path: Path) -> None:
        self.binary_path = binary_path
        self.model_path = model_path
        self.metadata_path = metadata_path
        super().__init__(
            [str(binary_path), str(model_path), str(metadata_path)],
            label="native runtime",
        )

    @classmethod
    def from_environment(cls, root_dir: Path) -> "NativeRuntimeClient":
        runtime_paths = resolve_runtime_paths(root_dir)
        binary_path = runtime_paths["binary_path"]
        model_path = runtime_paths["model_path"]
        metadata_path = runtime_paths["metadata_path"]

        ensure_binary_executable(binary_path)

        try:
            runtime = cls(binary_path, model_path, metadata_path)
        except OSError as exc:
            raise RuntimeError(f"failed to start native runtime: {exc}") from exc

        logger.info(
            "native action runtime enabled: "
            f"{binary_path.name} | {model_path.name} | {metadata_path.name}"
        )
        return runtime

    def infer(self, features: list[float], legal_actions: list[bool]) -> dict:
        return self._request({"features": features, "legal_actions": legal_actions})


def resolve_runtime_paths(root_dir: Path | None = None) -> dict[str, Path]:
    base_dir = ROOT_DIR if root_dir is None else root_dir
    binary_path = resolve_binary_path(
        "native runtime binary",
        NATIVE_RUNTIME_BIN_ENV,
        _default_binary_candidates(base_dir, "mjai-tract-runtime", include_debug=False),
    )
    model_path = resolve_binary_path(
        "native runtime ONNX model",
        NATIVE_RUNTIME_ONNX_ENV,
        [base_dir / "artifacts" / "policy.onnx"],
    )
    metadata_path = resolve_binary_path(
        "native runtime metadata",
        NATIVE_RUNTIME_META_ENV,
        [base_dir / "artifacts" / "policy.json"],
    )
    return {
        "binary_path": binary_path,
        "model_path": model_path,
        "metadata_path": metadata_path,
    }


def _action_candidate_from_payload(payload: object) -> ActionCandidate:
    if not isinstance(payload, dict):
        raise RuntimeError("decision binary returned a non-object candidate payload")

    action_type = payload.get("action_type")
    action_label = payload.get("action_label")
    primary_tile = payload.get("primary_tile")
    discard_tile = payload.get("discard_tile")
    consumed_tiles = payload.get("consumed_tiles", [])
    is_tsumogiri = payload.get("is_tsumogiri")

    if not isinstance(action_type, str):
        raise RuntimeError("decision binary returned a candidate without action_type")
    if not isinstance(action_label, str):
        raise RuntimeError("decision binary returned a candidate without action_label")
    if not isinstance(is_tsumogiri, bool):
        raise RuntimeError("decision binary returned a candidate with invalid is_tsumogiri")

    return ActionCandidate(
        action_type=action_type,
        action_label=action_label,
        primary_tile=primary_tile if isinstance(primary_tile, str) else None,
        discard_tile=discard_tile if isinstance(discard_tile, str) else None,
        consumed_tiles=tuple(str(t) for t in consumed_tiles) if isinstance(consumed_tiles, list) else (),
        next_shanten=int(payload.get("next_shanten", 0)),
        next_ukeire=int(payload.get("next_ukeire", 0)),
        ukeire=int(payload.get("ukeire", 0)),
        improving_count=int(payload.get("improving_count", 0)),
        discard_candidate_count=int(payload.get("discard_candidate_count", 0)),
        baseline_score=int(payload.get("baseline_score", 0)),
        discard_bonus=int(payload.get("discard_bonus", 0)),
        tile_seen=int(payload.get("tile_seen", 0)),
        tile_count=int(payload.get("tile_count", 0)),
        tile_dora=int(payload.get("tile_dora", 0)),
        is_tsumogiri=is_tsumogiri,
    )


class BasicMahjongBot(Bot):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.__native_runtime: NativeRuntimeClient | None = None
        self._last_action_source = "native"

    @property
    def _native_runtime(self) -> NativeRuntimeClient:
        if self.__native_runtime is None:
            self.__native_runtime = NativeRuntimeClient.from_environment(ROOT_DIR)
        return self.__native_runtime

    def think(self) -> str:
        try:
            if self.can_tsumo_agari:
                return self.action_tsumo_agari()
            if self.can_ron_agari:
                return self.action_ron_agari()

            if self.can_ryukyoku and self._should_abort_nine_terminals():
                return self.action_ryukyoku()

            if self._is_riichi_discard_phase():
                candidate = self._select_native_action_candidate()
                logger.info(
                    f"{self.bakaze}{self.kyoku}-{self.honba}: riichi-discard {self.tehai} | {self.last_self_tsumo} -> {candidate.discard_tile} [{self._last_action_source}]"  # noqa
                )
                return self.action_discard(candidate.discard_tile)

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
                logger.info(
                    f"{self.bakaze}{self.kyoku}-{self.honba}: {self.tehai} | {self.last_self_tsumo} -> {candidate.discard_tile} [{self._last_action_source}]"  # noqa
                )
                return self.action_discard(candidate.discard_tile)

            return self.action_nothing()
        finally:
            pass

    def _is_riichi_discard_phase(self) -> bool:
        return bool(
            self.can_discard
            and getattr(self, "self_riichi_declared", False)
            and not self.self_riichi_accepted
        )

    def _select_native_action_candidate(self) -> ActionCandidate:
        candidates = self._build_action_candidates()
        if not candidates:
            raise RuntimeError("no legal action candidates were produced")

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
        raise RuntimeError(f"unsupported action type: {candidate.action_type}")

    def _build_action_candidates(self) -> list[ActionCandidate]:
        decision = self._last_decision
        if decision is None:
            raise RuntimeError("no decision available from the Rust decision binary")
        raw_candidates = decision.get("candidates")
        if not isinstance(raw_candidates, list):
            raise RuntimeError("decision binary returned invalid candidates")
        return [_action_candidate_from_payload(c) for c in raw_candidates]

    def _build_runtime_features(
        self, candidates: list[ActionCandidate]
    ) -> tuple[list[float], list[bool]]:
        decision = self._last_decision
        if decision is None:
            raise RuntimeError("no decision available from the Rust decision binary")
        features = decision.get("features")
        legal_actions = decision.get("legal_actions")
        if not isinstance(features, list) or len(features) != INPUT_DIM:
            raise RuntimeError(
                f"decision binary returned invalid features (expected {INPUT_DIM}, "
                f"got {len(features) if isinstance(features, list) else 'none'})"
            )
        if not isinstance(legal_actions, list) or len(legal_actions) != ACTION_DIM:
            raise RuntimeError(
                f"decision binary returned invalid legal_actions (expected {ACTION_DIM}, "
                f"got {len(legal_actions) if isinstance(legal_actions, list) else 'none'})"
            )
        return [float(v) for v in features], [bool(v) for v in legal_actions]

    def _should_abort_nine_terminals(self) -> bool:
        distinct_terminal_tiles = {
            base_tile(tile)
            for tile in self.tehai_mjai
            if base_tile(tile) in TERMINAL_OR_HONOR_TILES
        }
        return len(distinct_terminal_tiles) >= 9


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mjai inference bot.")
    parser.add_argument("seat", nargs="?", type=int, default=0, help="Player seat id.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    BasicMahjongBot(player_id=args.seat).start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
