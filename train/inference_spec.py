from __future__ import annotations

TILE_TYPES = [
    *(f"{number}m" for number in range(1, 10)),
    *(f"{number}p" for number in range(1, 10)),
    *(f"{number}s" for number in range(1, 10)),
    "E",
    "S",
    "W",
    "N",
    "P",
    "F",
    "C",
]
TILE_INDEX = {tile: index for index, tile in enumerate(TILE_TYPES)}

ACTION_TYPES = [
    "pass",
    "discard",
    "riichi_discard",
    "pon",
    "chi",
]
ACTION_TYPE_INDEX = {action: index for index, action in enumerate(ACTION_TYPES)}

MAX_ACTION_CANDIDATES = 14

GLOBAL_SCALAR_FEATURES = 9
GLOBAL_ONE_HOT_FEATURES = 8
GLOBAL_HISTOGRAM_FEATURES = len(TILE_TYPES) * 3
GLOBAL_FEATURE_DIM = (
    GLOBAL_SCALAR_FEATURES + GLOBAL_ONE_HOT_FEATURES + GLOBAL_HISTOGRAM_FEATURES
)

CANDIDATE_PRESENCE_FEATURES = 1
CANDIDATE_ACTION_TYPE_FEATURES = len(ACTION_TYPES)
CANDIDATE_SCALAR_FEATURES = 9
CANDIDATE_PRIMARY_TILE_FEATURES = len(TILE_TYPES)
CANDIDATE_CONSUMED_TILE_FEATURES = len(TILE_TYPES)
CANDIDATE_FEATURE_DIM = (
    CANDIDATE_PRESENCE_FEATURES
    + CANDIDATE_ACTION_TYPE_FEATURES
    + CANDIDATE_SCALAR_FEATURES
    + CANDIDATE_PRIMARY_TILE_FEATURES
    + CANDIDATE_CONSUMED_TILE_FEATURES
)

INPUT_DIM = GLOBAL_FEATURE_DIM + MAX_ACTION_CANDIDATES * CANDIDATE_FEATURE_DIM
ACTION_DIM = MAX_ACTION_CANDIDATES


def base_tile(tile: str) -> str:
    return tile[:2]


def tile_sort_key(tile: str) -> tuple[int, int]:
    normalized = base_tile(tile)
    return TILE_INDEX[normalized], 1 if tile.endswith("r") else 0


def tile_one_hot(tile: str) -> list[float]:
    vector = [0.0] * len(TILE_TYPES)
    vector[TILE_INDEX[base_tile(tile)]] = 1.0
    return vector


def tile_histogram(tiles: list[str] | tuple[str, ...]) -> list[float]:
    vector = [0.0] * len(TILE_TYPES)
    for tile in tiles:
        vector[TILE_INDEX[base_tile(tile)]] += 1.0
    return vector


def action_type_one_hot(action_type: str) -> list[float]:
    vector = [0.0] * len(ACTION_TYPES)
    vector[ACTION_TYPE_INDEX[action_type]] = 1.0
    return vector