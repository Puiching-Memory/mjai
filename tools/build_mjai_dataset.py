from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from json import JSONDecodeError
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) in sys.path:
    sys.path.remove(str(TOOLS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot import ActionCandidate, BasicMahjongBot  # noqa: E402
from rust_mjai_bot import to_rank  # noqa: E402
from train.training_config import RewardConfig  # noqa: E402


MJLOG_BASE_URL = "https://storage.googleapis.com/mjlog"
DEFAULT_RAW_DIR = ROOT / "artifacts" / "mjai_raw"
GAME_URL_RE = re.compile(
    r"^https?://mjai\.app/games/(?P<batch_id>\d+)/(?P<match_id>\d+)/(?P<game_idx>\d+)/?$"
)


@dataclass(frozen=True, slots=True)
class GameRef:
    batch_id: int
    match_id: int
    game_idx: int

    @property
    def page_url(self) -> str:
        return f"https://mjai.app/games/{self.batch_id}/{self.match_id}/{self.game_idx}"


@dataclass(slots=True)
class DecisionPoint:
    phase: str
    candidates: list[ActionCandidate]
    features: list[float]
    legal_actions: list[bool]


@dataclass(slots=True)
class ConversionStats:
    games_processed: int = 0
    samples_written: int = 0
    unmatched_actions: int = 0
    failed_games: int = 0


class ReplayObservationBot(BasicMahjongBot):
    def think(self) -> str:
        return self.action_nothing()


def parse_game_url(url: str) -> GameRef:
    match = GAME_URL_RE.match(url.strip())
    if match is None:
        raise ValueError(f"invalid game URL: {url}")
    return GameRef(
        batch_id=int(match.group("batch_id")),
        match_id=int(match.group("match_id")),
        game_idx=int(match.group("game_idx")),
    )


def redact_event_for_player(event: dict[str, Any], player_id: int) -> dict[str, Any]:
    redacted = deepcopy(event)
    event_type = redacted.get("type")

    if event_type == "start_kyoku":
        tehais = redacted.get("tehais")
        if isinstance(tehais, list):
            masked_tehais = []
            for actor, tehai in enumerate(tehais):
                if actor == player_id:
                    masked_tehais.append(list(tehai) if isinstance(tehai, list) else tehai)
                else:
                    tile_count = len(tehai) if isinstance(tehai, list) else 13
                    masked_tehais.append(["?"] * tile_count)
            redacted["tehais"] = masked_tehais

    if event_type == "tsumo" and redacted.get("actor") != player_id and "pai" in redacted:
        redacted["pai"] = "?"

    return redacted


def _fetch_json(url: str, cache_path: Path) -> Any:
    if cache_path.exists():
        return _parse_json_payload(cache_path.read_text(encoding="utf-8"))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "mjai-dataset-builder/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except Exception:
        curl_path = shutil.which("curl")
        if curl_path is None:
            raise
        result = subprocess.run(
            [
                curl_path,
                "--http1.1",
                "--retry",
                "3",
                "--retry-delay",
                "1",
                "-fsSL",
                url,
            ],
            capture_output=True,
            check=True,
        )
        payload = result.stdout
    cache_path.write_bytes(payload)
    return _parse_json_payload(payload.decode("utf-8"))


def _parse_json_payload(text: str) -> Any:
    try:
        return json.loads(text)
    except JSONDecodeError:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise
        return [json.loads(line) for line in lines]


def _batch_games_index_cache_path(raw_dir: Path, batch_id: int) -> Path:
    return raw_dir / "games" / f"{batch_id}.json"


def _game_summary_cache_path(raw_dir: Path, game_ref: GameRef) -> Path:
    return raw_dir / "games" / str(game_ref.batch_id) / f"{game_ref.match_id}_{game_ref.game_idx}.json"


def _game_events_cache_path(raw_dir: Path, game_ref: GameRef) -> Path:
    return raw_dir / "games" / str(game_ref.batch_id) / f"{game_ref.match_id}_{game_ref.game_idx}_mjai.json"


def fetch_batch_games_index(batch_id: int, raw_dir: Path) -> dict[str, Any]:
    return _fetch_json(
        f"{MJLOG_BASE_URL}/games/{batch_id}.json",
        _batch_games_index_cache_path(raw_dir, batch_id),
    )


def fetch_game_assets(game_ref: GameRef, raw_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = _fetch_json(
        f"{MJLOG_BASE_URL}/games/{game_ref.batch_id}/{game_ref.match_id}_{game_ref.game_idx}_mjai.json",
        _game_events_cache_path(raw_dir, game_ref),
    )
    summary = _fetch_json(
        f"{MJLOG_BASE_URL}/games/{game_ref.batch_id}/{game_ref.match_id}_{game_ref.game_idx}.json",
        _game_summary_cache_path(raw_dir, game_ref),
    )
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise RuntimeError(f"invalid mjai event stream for {game_ref.page_url}")
    if not isinstance(summary, dict):
        raise RuntimeError(f"invalid game summary for {game_ref.page_url}")
    return events, summary


def _infer_games_per_match(match_entry: dict[str, Any]) -> int:
    ranks = match_entry.get("ranks")
    if not isinstance(ranks, list) or not ranks:
        return 0
    row_lengths = [len(row) for row in ranks if isinstance(row, list)]
    if not row_lengths:
        return 0
    return min(row_lengths)


def list_game_refs_for_batch(
    batch_id: int,
    raw_dir: Path,
    *,
    match_ids: set[int] | None = None,
    game_indices: set[int] | None = None,
) -> list[GameRef]:
    payload = fetch_batch_games_index(batch_id, raw_dir)
    matches = payload.get("matches")
    if not isinstance(matches, list):
        raise RuntimeError(f"batch games index is missing matches for batch {batch_id}")

    refs: list[GameRef] = []
    for match_entry in matches:
        if not isinstance(match_entry, dict):
            continue
        match_id = match_entry.get("match_id")
        if not isinstance(match_id, int):
            continue
        if match_ids is not None and match_id not in match_ids:
            continue

        game_count = _infer_games_per_match(match_entry)
        if game_count <= 0:
            continue

        available_game_indices = range(game_count)
        selected_game_indices = game_indices if game_indices is not None else set(available_game_indices)
        for game_idx in sorted(selected_game_indices):
            if 0 <= game_idx < game_count:
                refs.append(GameRef(batch_id=batch_id, match_id=match_id, game_idx=game_idx))
    return refs


def extract_decision_point(bot: ReplayObservationBot) -> DecisionPoint | None:
    if bot.can_tsumo_agari or bot.can_ron_agari:
        return None

    if bot.can_ryukyoku and bot._should_abort_nine_terminals():
        return None

    if bot._is_riichi_discard_phase():
        phase = "riichi_discard"
    elif bot.can_riichi:
        return None
    elif bot.can_pon or bot.can_chi:
        phase = "call"
    elif bot.can_discard:
        if bot.self_riichi_accepted:
            return None
        phase = "discard"
    else:
        return None

    try:
        candidates = bot._build_action_candidates()
        features, legal_actions = bot._build_runtime_features(candidates)
    except RuntimeError:
        return None

    if not candidates:
        return None
    return DecisionPoint(
        phase=phase,
        candidates=candidates,
        features=features,
        legal_actions=legal_actions,
    )


def _tile_counter(tiles: Sequence[str]) -> Counter[str]:
    return Counter(str(tile) for tile in tiles)


def candidate_matches_event(
    candidate: ActionCandidate,
    phase: str,
    event: dict[str, Any],
    player_id: int,
) -> bool:
    event_type = event.get("type")
    actor = event.get("actor")

    if phase in {"discard", "riichi_discard"}:
        return bool(
            event_type == "dahai"
            and actor == player_id
            and candidate.action_type == phase
            and candidate.discard_tile == event.get("pai")
            and candidate.is_tsumogiri == bool(event.get("tsumogiri", False))
        )

    if phase == "call":
        if actor != player_id or event_type not in {"pon", "chi"}:
            return False
        consumed = event.get("consumed")
        if not isinstance(consumed, list):
            return False
        return bool(
            candidate.action_type == event_type
            and _tile_counter(candidate.consumed_tiles) == _tile_counter(consumed)
        )

    return False


def match_logged_action_index(
    decision: DecisionPoint,
    next_event: dict[str, Any] | None,
    player_id: int,
) -> int | None:
    if next_event is None:
        return None

    for index, candidate in enumerate(decision.candidates):
        if candidate_matches_event(candidate, decision.phase, next_event, player_id):
            return index

    if decision.phase != "call":
        return None

    next_event_is_player_claim = bool(
        next_event.get("actor") == player_id
        and next_event.get("type") in {"pon", "chi", "daiminkan", "hora"}
    )
    if next_event_is_player_claim:
        return None

    for index, candidate in enumerate(decision.candidates):
        if candidate.action_type == "pass":
            return index
    return None


def _final_scores_from_summary(summary: dict[str, Any]) -> list[int]:
    kyoku = summary.get("kyoku")
    if not isinstance(kyoku, list) or not kyoku:
        raise RuntimeError("game summary is missing kyoku score history")
    last_round = kyoku[-1]
    if not isinstance(last_round, dict):
        raise RuntimeError("game summary has an invalid final round entry")
    end_scores = last_round.get("end_kyoku_scores")
    if not isinstance(end_scores, list) or len(end_scores) != 4:
        raise RuntimeError("game summary is missing final scores")
    return [int(score) for score in end_scores]


def _ranks_from_summary(summary: dict[str, Any], final_scores: list[int]) -> list[int]:
    raw_ranks = summary.get("rank")
    if isinstance(raw_ranks, list) and len(raw_ranks) == 4:
        return [int(rank) for rank in raw_ranks]
    return to_rank(final_scores)


def convert_game_to_samples(
    game_ref: GameRef,
    events: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    reward_config: RewardConfig,
) -> tuple[list[dict[str, Any]], int]:
    usernames = summary.get("usernames")
    if not isinstance(usernames, list) or len(usernames) != 4:
        usernames = [f"seat-{seat}" for seat in range(4)]

    final_scores = _final_scores_from_summary(summary)
    ranks = _ranks_from_summary(summary, final_scores)

    samples: list[dict[str, Any]] = []
    unmatched_actions = 0

    for seat in range(4):
        bot = ReplayObservationBot(player_id=seat)
        reward = reward_config.reward_for_result(ranks[seat], final_scores[seat])

        for event_index, event in enumerate(events[:-1]):
            visible_event = redact_event_for_player(event, seat)
            bot.react(json.dumps([visible_event], ensure_ascii=True, separators=(",", ":")))

            decision = extract_decision_point(bot)
            if decision is None:
                continue

            next_event = events[event_index + 1]
            action_index = match_logged_action_index(decision, next_event, seat)
            if action_index is None:
                unmatched_actions += 1
                continue

            action_type = decision.candidates[action_index].action_type
            samples.append(
                {
                    "policy_name": str(usernames[seat]),
                    "features": decision.features,
                    "legal_actions": decision.legal_actions,
                    "action_index": action_index,
                    "reward": reward,
                    "rank": ranks[seat],
                    "score": final_scores[seat],
                    "action_type": action_type,
                    "batch_id": game_ref.batch_id,
                    "match_id": game_ref.match_id,
                    "game_idx": game_ref.game_idx,
                    "seat": seat,
                    "event_index": event_index,
                    "source_url": game_ref.page_url,
                }
            )

    return samples, unmatched_actions


def _dedupe_game_refs(game_refs: Sequence[GameRef]) -> list[GameRef]:
    deduped: list[GameRef] = []
    seen: set[GameRef] = set()
    for game_ref in game_refs:
        if game_ref in seen:
            continue
        seen.add(game_ref)
        deduped.append(game_ref)
    return deduped


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download archived mjai.app replays and convert them into the current project's supervised JSONL format.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSONL path for converted training samples.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory used to cache downloaded mjlog JSON files.",
    )
    parser.add_argument(
        "--batch-id",
        type=int,
        help="Batch id from mjai.app weekly archives. If set without --match-id, the whole batch is processed.",
    )
    parser.add_argument(
        "--match-id",
        action="append",
        type=int,
        default=None,
        help="Optional match id filter within --batch-id. Can be repeated.",
    )
    parser.add_argument(
        "--game-idx",
        action="append",
        type=int,
        choices=(0, 1, 2, 3),
        default=None,
        help="Optional game index filter within each selected match. Can be repeated.",
    )
    parser.add_argument(
        "--game-url",
        action="append",
        default=None,
        help="Explicit mjai.app game URL such as https://mjai.app/games/137/250/0. Can be repeated.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="Optional limit for debugging. Zero means no limit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output JSONL file.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on the first failed download or conversion.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.batch_id is None and not args.game_url:
        raise SystemExit("either --batch-id or --game-url is required")
    if args.match_id and args.batch_id is None:
        raise SystemExit("--match-id requires --batch-id")
    if args.game_idx and args.batch_id is None and not args.game_url:
        raise SystemExit("--game-idx requires --batch-id or --game-url")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {args.output}")

    raw_dir = args.raw_dir.resolve()
    reward_config = RewardConfig()
    selected_refs: list[GameRef] = []

    if args.game_url:
        selected_refs.extend(parse_game_url(url) for url in args.game_url)

    if args.batch_id is not None:
        match_ids = set(args.match_id) if args.match_id else None
        game_indices = set(args.game_idx) if args.game_idx else None
        selected_refs.extend(
            list_game_refs_for_batch(
                args.batch_id,
                raw_dir,
                match_ids=match_ids,
                game_indices=game_indices,
            )
        )

    selected_refs = _dedupe_game_refs(selected_refs)
    if args.max_games > 0:
        selected_refs = selected_refs[: args.max_games]
    if not selected_refs:
        raise SystemExit("no games matched the provided filters")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    stats = ConversionStats()
    action_counts: Counter[str] = Counter()

    with args.output.open("w", encoding="utf-8") as output_handle:
        for index, game_ref in enumerate(selected_refs, start=1):
            try:
                events, summary = fetch_game_assets(game_ref, raw_dir)
                samples, unmatched_actions = convert_game_to_samples(
                    game_ref,
                    events,
                    summary,
                    reward_config=reward_config,
                )
            except Exception as exc:
                stats.failed_games += 1
                print(
                    f"[{index}/{len(selected_refs)}] failed {game_ref.page_url}: {exc}",
                    file=sys.stderr,
                )
                if args.fail_fast:
                    raise
                continue

            for sample in samples:
                output_handle.write(json.dumps(sample, ensure_ascii=True, separators=(",", ":")) + "\n")
                action_counts.update([sample["action_type"]])

            stats.games_processed += 1
            stats.samples_written += len(samples)
            stats.unmatched_actions += unmatched_actions
            print(
                f"[{index}/{len(selected_refs)}] {game_ref.batch_id}/{game_ref.match_id}/{game_ref.game_idx}: "
                f"wrote {len(samples)} samples (unmatched {unmatched_actions})",
                file=sys.stderr,
            )

    print(
        (
            f"completed {stats.games_processed} games, wrote {stats.samples_written} samples, "
            f"unmatched {stats.unmatched_actions}, failed {stats.failed_games}. "
            f"action counts: {dict(sorted(action_counts.items()))}"
        ),
        file=sys.stderr,
    )
    return 0 if stats.failed_games == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())