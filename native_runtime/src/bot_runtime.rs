use std::collections::{HashMap, HashSet};

use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};

pub const TILE_TYPES: [&str; 34] = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p",
    "6p", "7p", "8p", "9p", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E",
    "S", "W", "N", "P", "F", "C",
];
pub const WIND_TILES: [&str; 4] = ["E", "S", "W", "N"];
pub const ACTION_TYPES: [ActionType; 5] = [
    ActionType::Pass,
    ActionType::Discard,
    ActionType::RiichiDiscard,
    ActionType::Pon,
    ActionType::Chi,
];

pub const MAX_ACTION_CANDIDATES: usize = 14;
pub const GLOBAL_SCALAR_FEATURES: usize = 9;
pub const GLOBAL_ONE_HOT_FEATURES: usize = 8;
pub const GLOBAL_HISTOGRAM_FEATURES: usize = TILE_TYPES.len() * 3;
pub const GLOBAL_FEATURE_DIM: usize =
    GLOBAL_SCALAR_FEATURES + GLOBAL_ONE_HOT_FEATURES + GLOBAL_HISTOGRAM_FEATURES;
pub const CANDIDATE_PRESENCE_FEATURES: usize = 1;
pub const CANDIDATE_ACTION_TYPE_FEATURES: usize = ACTION_TYPES.len();
pub const CANDIDATE_SCALAR_FEATURES: usize = 9;
pub const CANDIDATE_PRIMARY_TILE_FEATURES: usize = TILE_TYPES.len();
pub const CANDIDATE_CONSUMED_TILE_FEATURES: usize = TILE_TYPES.len();
pub const CANDIDATE_FEATURE_DIM: usize = CANDIDATE_PRESENCE_FEATURES
    + CANDIDATE_ACTION_TYPE_FEATURES
    + CANDIDATE_SCALAR_FEATURES
    + CANDIDATE_PRIMARY_TILE_FEATURES
    + CANDIDATE_CONSUMED_TILE_FEATURES;
pub const INPUT_DIM: usize = GLOBAL_FEATURE_DIM + MAX_ACTION_CANDIDATES * CANDIDATE_FEATURE_DIM;
pub const ACTION_DIM: usize = MAX_ACTION_CANDIDATES;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionType {
    Pass,
    Discard,
    RiichiDiscard,
    Pon,
    Chi,
}

impl ActionType {
    fn one_hot(self) -> Vec<f32> {
        ACTION_TYPES
            .iter()
            .map(|candidate| if *candidate == self { 1.0 } else { 0.0 })
            .collect()
    }

    fn label(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Discard => "discard",
            Self::RiichiDiscard => "riichi_discard",
            Self::Pon => "pon",
            Self::Chi => "chi",
        }
    }

    fn call_sort_key(self) -> usize {
        match self {
            Self::Chi => 0,
            Self::Pon => 1,
            _ => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionPhase {
    Idle,
    Discard,
    RiichiDiscard,
    Call,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionCandidate {
    pub action_type: ActionType,
    pub action_label: String,
    pub primary_tile: Option<String>,
    pub discard_tile: Option<String>,
    #[serde(default)]
    pub consumed_tiles: Vec<String>,
    pub next_shanten: i32,
    pub next_ukeire: i32,
    pub ukeire: i32,
    pub improving_count: i32,
    pub discard_candidate_count: i32,
    pub baseline_score: i32,
    pub discard_bonus: i32,
    pub tile_seen: i32,
    pub tile_count: i32,
    pub tile_dora: i32,
    pub is_tsumogiri: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRequest {
    pub shanten: i32,
    pub best_ukeire: i32,
    pub bakaze: String,
    pub kyoku: i32,
    pub honba: i32,
    pub kyotaku: i32,
    pub player_id: usize,
    pub self_riichi_accepted: bool,
    pub can_riichi: bool,
    pub has_open_hand: bool,
    #[serde(default)]
    pub hand_tiles: Vec<String>,
    #[serde(default)]
    pub tiles_seen: HashMap<String, i32>,
    #[serde(default)]
    pub dora_indicators: Vec<String>,
    #[serde(default)]
    pub candidates: Vec<ActionCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawDiscardCandidate {
    pub discard_tile: Option<String>,
    pub ukeire: i32,
    pub improving_count: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawCallCandidate {
    #[serde(default)]
    pub consumed_tiles: Vec<String>,
    pub next_shanten: i32,
    pub next_ukeire: i32,
    pub discard_candidate_count: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateCompileRequest {
    pub phase: DecisionPhase,
    pub shanten: i32,
    pub best_ukeire: i32,
    pub has_open_hand: bool,
    #[serde(default)]
    pub hand_tiles: Vec<String>,
    #[serde(default)]
    pub tiles_seen: HashMap<String, i32>,
    #[serde(default)]
    pub dora_indicators: Vec<String>,
    #[serde(default)]
    pub forbidden_tiles: HashMap<String, bool>,
    #[serde(default)]
    pub yakuhai_tiles: Vec<String>,
    pub last_self_tsumo: Option<String>,
    pub last_kawa_tile: Option<String>,
    #[serde(default)]
    pub riichi_discardable_tiles: Vec<String>,
    #[serde(default)]
    pub improving_tiles: Vec<RawDiscardCandidate>,
    #[serde(default)]
    pub pon_candidates: Vec<RawCallCandidate>,
    #[serde(default)]
    pub chi_candidates: Vec<RawCallCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedDecision {
    pub features: Vec<f32>,
    pub legal_actions: Vec<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledCandidates {
    pub candidates: Vec<ActionCandidate>,
}

#[derive(Debug, Clone)]
struct DecisionState {
    hand_tiles: Vec<String>,
    hand_counts: HashMap<String, i32>,
    tiles_seen: HashMap<String, i32>,
    dora_targets: HashMap<String, i32>,
    total_dora_in_hand: i32,
}

impl DecisionState {
    fn from_request(request: &DecisionRequest) -> Result<Self> {
        Self::from_parts(&request.hand_tiles, &request.tiles_seen, &request.dora_indicators)
    }

    fn from_compile_request(request: &CandidateCompileRequest) -> Result<Self> {
        Self::from_parts(&request.hand_tiles, &request.tiles_seen, &request.dora_indicators)
    }

    fn from_parts(
        hand_tiles: &[String],
        tiles_seen: &HashMap<String, i32>,
        dora_indicators: &[String],
    ) -> Result<Self> {
        let mut hand_counts = HashMap::new();
        for tile in hand_tiles {
            ensure_known_tile(tile)?;
            *hand_counts.entry(base_tile(tile).to_string()).or_insert(0) += 1;
        }

        let mut normalized_tiles_seen = HashMap::new();
        for (tile, count) in tiles_seen {
            ensure_known_tile(tile)?;
            normalized_tiles_seen.insert(base_tile(tile).to_string(), *count);
        }

        let mut dora_targets = HashMap::new();
        for indicator in dora_indicators {
            ensure_known_tile(indicator)?;
            let dora = dora_from_indicator(indicator)?;
            *dora_targets.entry(dora).or_insert(0) += 1;
        }

        let total_dora_in_hand = hand_tiles
            .iter()
            .map(|tile| tile_dora_value(tile, &dora_targets))
            .sum();

        Ok(Self {
            hand_tiles: hand_tiles.to_vec(),
            hand_counts,
            tiles_seen: normalized_tiles_seen,
            dora_targets,
            total_dora_in_hand,
        })
    }
}

pub fn encode_decision(request: DecisionRequest) -> Result<EncodedDecision> {
    validate_request(&request)?;
    let state = DecisionState::from_request(&request)?;

    let mut features = global_runtime_features(&request, &state);
    let mut legal_actions = vec![false; ACTION_DIM];

    for index in 0..MAX_ACTION_CANDIDATES {
        if let Some(candidate) = request.candidates.get(index) {
            legal_actions[index] = true;
            features.extend(candidate_runtime_features(candidate)?);
        } else {
            features.extend(vec![0.0; CANDIDATE_FEATURE_DIM]);
        }
    }

    if features.len() != INPUT_DIM {
        bail!(
            "encoded feature length {} does not match expected {}",
            features.len(),
            INPUT_DIM
        );
    }

    Ok(EncodedDecision {
        features,
        legal_actions,
    })
}

pub fn compile_candidates(request: CandidateCompileRequest) -> Result<CompiledCandidates> {
    validate_compile_request(&request)?;
    let state = DecisionState::from_compile_request(&request)?;
    let yakuhai_tiles = request
        .yakuhai_tiles
        .iter()
        .map(|tile| base_tile(tile).to_string())
        .collect::<HashSet<_>>();

    let candidates = match request.phase {
        DecisionPhase::Idle => Vec::new(),
        DecisionPhase::Discard => {
            compile_discard_candidates(&request, &state, &yakuhai_tiles, ActionType::Discard)
        }
        DecisionPhase::RiichiDiscard => compile_riichi_discard_candidates(
            &request,
            &state,
            &yakuhai_tiles,
        ),
        DecisionPhase::Call => compile_call_candidates(&request, &state, &yakuhai_tiles)?,
    };

    Ok(CompiledCandidates { candidates })
}

fn validate_request(request: &DecisionRequest) -> Result<()> {
    if request.player_id >= 4 {
        bail!("player_id {} is out of range", request.player_id);
    }
    if request.candidates.len() > MAX_ACTION_CANDIDATES {
        bail!(
            "candidate count {} exceeds max {}",
            request.candidates.len(),
            MAX_ACTION_CANDIDATES
        );
    }
    for candidate in &request.candidates {
        if let Some(primary_tile) = &candidate.primary_tile {
            ensure_known_tile(primary_tile)?;
        }
        if let Some(discard_tile) = &candidate.discard_tile {
            ensure_known_tile(discard_tile)?;
        }
        for tile in &candidate.consumed_tiles {
            ensure_known_tile(tile)?;
        }
    }
    Ok(())
}

fn validate_compile_request(request: &CandidateCompileRequest) -> Result<()> {
    for tile in &request.hand_tiles {
        ensure_known_tile(tile)?;
    }
    for (tile, _) in &request.tiles_seen {
        ensure_known_tile(tile)?;
    }
    for indicator in &request.dora_indicators {
        ensure_known_tile(indicator)?;
    }
    for (tile, _) in &request.forbidden_tiles {
        ensure_known_tile(tile)?;
    }
    for tile in &request.yakuhai_tiles {
        ensure_known_tile(tile)?;
    }
    if let Some(tile) = &request.last_self_tsumo {
        ensure_known_tile(tile)?;
    }
    if let Some(tile) = &request.last_kawa_tile {
        ensure_known_tile(tile)?;
    }
    for tile in &request.riichi_discardable_tiles {
        ensure_known_tile(tile)?;
    }
    for candidate in &request.improving_tiles {
        if let Some(discard_tile) = &candidate.discard_tile {
            ensure_known_tile(discard_tile)?;
        }
    }
    for candidate in request
        .pon_candidates
        .iter()
        .chain(request.chi_candidates.iter())
    {
        for tile in &candidate.consumed_tiles {
            ensure_known_tile(tile)?;
        }
    }
    if matches!(request.phase, DecisionPhase::Call) && request.last_kawa_tile.is_none() {
        bail!("call candidate compilation requires last_kawa_tile")
    }
    Ok(())
}

fn compile_discard_candidates(
    request: &CandidateCompileRequest,
    state: &DecisionState,
    yakuhai_tiles: &HashSet<String>,
    action_type: ActionType,
) -> Vec<ActionCandidate> {
    let mut selected_by_tile: HashMap<String, ActionCandidate> = HashMap::new();

    for candidate in &request.improving_tiles {
        let Some(discard_tile) = candidate.discard_tile.as_deref() else {
            continue;
        };
        if is_forbidden_tile(discard_tile, &request.forbidden_tiles) {
            continue;
        }

        let compiled = compile_discard_action_candidate(
            candidate,
            state,
            request,
            yakuhai_tiles,
            action_type,
        );
        match selected_by_tile.get(discard_tile) {
            Some(current) if compiled.baseline_score <= current.baseline_score => {}
            _ => {
                selected_by_tile.insert(discard_tile.to_string(), compiled);
            }
        }
    }

    if selected_by_tile.is_empty() {
        let mut fallback_tiles = state.hand_tiles.clone();
        fallback_tiles.sort_by_key(|tile| tile_sort_key(tile));
        for tile in fallback_tiles {
            if is_forbidden_tile(&tile, &request.forbidden_tiles) {
                continue;
            }
            if selected_by_tile.contains_key(&tile) {
                continue;
            }
            let compiled = compile_fallback_discard_action_candidate(
                &tile,
                state,
                request,
                yakuhai_tiles,
                action_type,
            );
            selected_by_tile.insert(tile, compiled);
        }
    }

    let mut candidates = selected_by_tile.into_values().collect::<Vec<_>>();
    candidates.sort_by_key(|candidate| tile_sort_key(candidate.discard_tile.as_deref().unwrap()));
    candidates.truncate(MAX_ACTION_CANDIDATES);
    candidates
}

fn compile_riichi_discard_candidates(
    request: &CandidateCompileRequest,
    state: &DecisionState,
    yakuhai_tiles: &HashSet<String>,
) -> Vec<ActionCandidate> {
    let improving_by_tile = request
        .improving_tiles
        .iter()
        .filter_map(|candidate| {
            candidate
                .discard_tile
                .as_ref()
                .map(|discard_tile| (discard_tile.clone(), candidate))
        })
        .collect::<HashMap<_, _>>();

    let mut riichi_tiles = request.riichi_discardable_tiles.clone();
    riichi_tiles.sort_by_key(|tile| tile_sort_key(tile));

    let mut candidates = Vec::new();
    for discard_tile in riichi_tiles {
        if is_forbidden_tile(&discard_tile, &request.forbidden_tiles) {
            continue;
        }

        let compiled = if let Some(candidate) = improving_by_tile.get(&discard_tile) {
            compile_discard_action_candidate(
                candidate,
                state,
                request,
                yakuhai_tiles,
                ActionType::RiichiDiscard,
            )
        } else {
            compile_fallback_discard_action_candidate(
                &discard_tile,
                state,
                request,
                yakuhai_tiles,
                ActionType::RiichiDiscard,
            )
        };
        candidates.push(compiled);
    }

    candidates.truncate(MAX_ACTION_CANDIDATES);
    candidates
}

fn compile_call_candidates(
    request: &CandidateCompileRequest,
    state: &DecisionState,
    yakuhai_tiles: &HashSet<String>,
) -> Result<Vec<ActionCandidate>> {
    let primary_tile = request
        .last_kawa_tile
        .as_deref()
        .ok_or_else(|| anyhow!("call candidate compilation requires last_kawa_tile"))?;
    let primary_base_tile = base_tile(primary_tile);

    let mut candidates = vec![ActionCandidate {
        action_type: ActionType::Pass,
        action_label: ActionType::Pass.label().to_string(),
        primary_tile: Some(primary_tile.to_string()),
        discard_tile: None,
        consumed_tiles: Vec::new(),
        next_shanten: request.shanten,
        next_ukeire: request.best_ukeire,
        ukeire: request.best_ukeire,
        improving_count: 0,
        discard_candidate_count: 0,
        baseline_score: 0,
        discard_bonus: 0,
        tile_seen: *state.tiles_seen.get(primary_base_tile).unwrap_or(&0),
        tile_count: 0,
        tile_dora: tile_dora_value(primary_tile, &state.dora_targets),
        is_tsumogiri: false,
    }];

    for candidate in &request.pon_candidates {
        if should_call(candidate, ActionType::Pon, request, yakuhai_tiles, primary_tile) {
            candidates.push(compile_call_action_candidate(
                candidate,
                ActionType::Pon,
                request,
                state,
                yakuhai_tiles,
                primary_tile,
            ));
        }
    }
    for candidate in &request.chi_candidates {
        if should_call(candidate, ActionType::Chi, request, yakuhai_tiles, primary_tile) {
            candidates.push(compile_call_action_candidate(
                candidate,
                ActionType::Chi,
                request,
                state,
                yakuhai_tiles,
                primary_tile,
            ));
        }
    }

    let passthrough = candidates.remove(0);
    candidates.sort_by_key(|candidate| {
        (
            candidate.action_type.call_sort_key(),
            candidate
                .consumed_tiles
                .iter()
                .map(|tile| tile_sort_key(tile))
                .collect::<Vec<_>>(),
        )
    });
    candidates.truncate(MAX_ACTION_CANDIDATES.saturating_sub(1));

    let mut output = Vec::with_capacity(candidates.len() + 1);
    output.push(passthrough);
    output.extend(candidates);
    Ok(output)
}

fn compile_discard_action_candidate(
    candidate: &RawDiscardCandidate,
    state: &DecisionState,
    request: &CandidateCompileRequest,
    yakuhai_tiles: &HashSet<String>,
    action_type: ActionType,
) -> ActionCandidate {
    let discard_tile = candidate.discard_tile.as_deref().unwrap();
    let discard_bonus = tile_discard_bonus(discard_tile, state, yakuhai_tiles);
    let normalized = base_tile(discard_tile);

    ActionCandidate {
        action_type,
        action_label: discard_tile.to_string(),
        primary_tile: Some(discard_tile.to_string()),
        discard_tile: Some(discard_tile.to_string()),
        consumed_tiles: Vec::new(),
        next_shanten: request.shanten,
        next_ukeire: candidate.ukeire,
        ukeire: candidate.ukeire,
        improving_count: candidate.improving_count,
        discard_candidate_count: 0,
        baseline_score: discard_candidate_score(candidate, state, request, yakuhai_tiles),
        discard_bonus,
        tile_seen: *state.tiles_seen.get(normalized).unwrap_or(&0),
        tile_count: *state.hand_counts.get(normalized).unwrap_or(&0),
        tile_dora: tile_dora_value(discard_tile, &state.dora_targets),
        is_tsumogiri: request.last_self_tsumo.as_deref() == Some(discard_tile),
    }
}

fn compile_fallback_discard_action_candidate(
    discard_tile: &str,
    state: &DecisionState,
    request: &CandidateCompileRequest,
    yakuhai_tiles: &HashSet<String>,
    action_type: ActionType,
) -> ActionCandidate {
    let normalized = base_tile(discard_tile);
    let discard_bonus = tile_discard_bonus(discard_tile, state, yakuhai_tiles);
    let mut baseline_score = discard_bonus + state.tiles_seen.get(normalized).unwrap_or(&0) * 6;
    if request.last_self_tsumo.as_deref() == Some(discard_tile) {
        baseline_score += 1;
    }

    ActionCandidate {
        action_type,
        action_label: discard_tile.to_string(),
        primary_tile: Some(discard_tile.to_string()),
        discard_tile: Some(discard_tile.to_string()),
        consumed_tiles: Vec::new(),
        next_shanten: request.shanten,
        next_ukeire: 0,
        ukeire: 0,
        improving_count: 0,
        discard_candidate_count: 0,
        baseline_score,
        discard_bonus,
        tile_seen: *state.tiles_seen.get(normalized).unwrap_or(&0),
        tile_count: *state.hand_counts.get(normalized).unwrap_or(&0),
        tile_dora: tile_dora_value(discard_tile, &state.dora_targets),
        is_tsumogiri: request.last_self_tsumo.as_deref() == Some(discard_tile),
    }
}

fn compile_call_action_candidate(
    candidate: &RawCallCandidate,
    action_type: ActionType,
    request: &CandidateCompileRequest,
    state: &DecisionState,
    yakuhai_tiles: &HashSet<String>,
    primary_tile: &str,
) -> ActionCandidate {
    let mut consumed_tiles = candidate.consumed_tiles.clone();
    consumed_tiles.sort_by_key(|tile| tile_sort_key(tile));
    let primary_base_tile = base_tile(primary_tile);

    ActionCandidate {
        action_type,
        action_label: format!("{}:{}", action_type.label(), consumed_tiles.join("/")),
        primary_tile: Some(primary_tile.to_string()),
        discard_tile: None,
        consumed_tiles,
        next_shanten: candidate.next_shanten,
        next_ukeire: candidate.next_ukeire,
        ukeire: candidate.next_ukeire,
        improving_count: 0,
        discard_candidate_count: candidate.discard_candidate_count,
        baseline_score: call_candidate_score(
            candidate,
            action_type,
            request,
            yakuhai_tiles,
            primary_tile,
        ),
        discard_bonus: 0,
        tile_seen: *state.tiles_seen.get(primary_base_tile).unwrap_or(&0),
        tile_count: *state.hand_counts.get(primary_base_tile).unwrap_or(&0),
        tile_dora: tile_dora_value(primary_tile, &state.dora_targets),
        is_tsumogiri: false,
    }
}

fn should_call(
    candidate: &RawCallCandidate,
    action_type: ActionType,
    request: &CandidateCompileRequest,
    yakuhai_tiles: &HashSet<String>,
    primary_tile: &str,
) -> bool {
    let current_shanten = request.shanten;
    let current_ukeire = request.best_ukeire;
    let next_shanten = candidate.next_shanten;
    let next_ukeire = candidate.next_ukeire;
    let has_open_hand = request.has_open_hand;
    let is_value_pon = action_type == ActionType::Pon && is_yakuhai_tile(primary_tile, yakuhai_tiles);

    if next_shanten > current_shanten {
        return false;
    }

    if next_shanten < current_shanten {
        if is_value_pon {
            return true;
        }
        if has_open_hand {
            return next_ukeire + 2 >= current_ukeire;
        }
        if action_type == ActionType::Pon {
            return next_shanten == 0 && next_ukeire >= std::cmp::max(6, current_ukeire - 2);
        }
        return next_shanten == 0 && next_ukeire >= std::cmp::max(8, current_ukeire);
    }

    if is_value_pon && next_shanten <= 1 && next_ukeire >= current_ukeire + 2 {
        return true;
    }

    if has_open_hand && next_shanten == 0 && next_ukeire >= current_ukeire + 4 {
        return true;
    }

    if has_open_hand && next_shanten <= 1 && next_ukeire >= current_ukeire + 6 {
        return true;
    }

    false
}

fn call_candidate_score(
    candidate: &RawCallCandidate,
    action_type: ActionType,
    request: &CandidateCompileRequest,
    yakuhai_tiles: &HashSet<String>,
    primary_tile: &str,
) -> i32 {
    let mut score = 0;
    score += (request.shanten - candidate.next_shanten) * 100;
    score += candidate.next_ukeire * 10;
    score += candidate.discard_candidate_count;

    if action_type == ActionType::Pon && is_yakuhai_tile(primary_tile, yakuhai_tiles) {
        score += 25;
    }
    if candidate.next_shanten == 0 {
        score += 20;
    }
    if request.has_open_hand {
        score += 10;
    }

    score
}

fn discard_candidate_score(
    candidate: &RawDiscardCandidate,
    state: &DecisionState,
    request: &CandidateCompileRequest,
    yakuhai_tiles: &HashSet<String>,
) -> i32 {
    let discard_tile = candidate.discard_tile.as_deref().unwrap();
    let normalized = base_tile(discard_tile);

    let mut score = 0;
    score += candidate.ukeire * 100;
    score += candidate.improving_count * 3;
    score += state.tiles_seen.get(normalized).unwrap_or(&0) * 6;
    score += tile_discard_bonus(discard_tile, state, yakuhai_tiles);

    if request.last_self_tsumo.as_deref() == Some(discard_tile) {
        score += 1;
    }

    score
}

fn tile_discard_bonus(tile: &str, state: &DecisionState, yakuhai_tiles: &HashSet<String>) -> i32 {
    let normalized = base_tile(tile);
    let tile_count = *state.hand_counts.get(normalized).unwrap_or(&0);

    let mut score = 0;
    score -= tile_dora_value(tile, &state.dora_targets) * 60;

    if tile.ends_with('r') {
        score -= 40;
    }

    if is_honor_tile(normalized) {
        if is_yakuhai_tile(normalized, yakuhai_tiles) {
            score -= if tile_count >= 2 { 35 } else { 15 };
        } else if tile_count >= 2 {
            score -= 12;
        } else {
            score += 28;
        }

        score += state.tiles_seen.get(normalized).unwrap_or(&0) * 4;
        return score;
    }

    let number = normalized
        .chars()
        .next()
        .and_then(|value| value.to_digit(10))
        .unwrap_or(0) as i32;
    score += match number {
        1 => 18,
        2 => 10,
        3 => 4,
        4 => 0,
        5 => -4,
        6 => 0,
        7 => 4,
        8 => 10,
        9 => 18,
        _ => 0,
    };

    if tile_count >= 2 {
        score -= 12;
    }

    score + tile_isolation_bonus(normalized, &state.hand_counts)
}

fn tile_isolation_bonus(tile: &str, tile_counts: &HashMap<String, i32>) -> i32 {
    if is_honor_tile(tile) {
        return 0;
    }

    let mut chars = tile.chars();
    let number = chars.next().and_then(|value| value.to_digit(10)).unwrap_or(0) as i32;
    let color = chars.next().unwrap_or('m');
    let mut close_connections = 0;
    let mut wide_connections = 0;

    for offset in [-1, 1] {
        let neighbor = number + offset;
        if (1..=9).contains(&neighbor) && tile_counts.get(&format!("{neighbor}{color}")).copied().unwrap_or(0) > 0 {
            close_connections += 1;
        }
    }
    for offset in [-2, 2] {
        let neighbor = number + offset;
        if (1..=9).contains(&neighbor) && tile_counts.get(&format!("{neighbor}{color}")).copied().unwrap_or(0) > 0 {
            wide_connections += 1;
        }
    }

    if close_connections == 0 && wide_connections == 0 {
        return 26;
    }
    if close_connections == 0 {
        return 14;
    }
    if close_connections == 1 {
        return 6 - wide_connections * 2;
    }
    -8
}

fn is_forbidden_tile(tile: &str, forbidden_tiles: &HashMap<String, bool>) -> bool {
    *forbidden_tiles.get(base_tile(tile)).unwrap_or(&true)
}

fn is_yakuhai_tile(tile: &str, yakuhai_tiles: &HashSet<String>) -> bool {
    yakuhai_tiles.contains(base_tile(tile))
}

fn is_honor_tile(tile: &str) -> bool {
    matches!(tile, "E" | "S" | "W" | "N" | "P" | "F" | "C")
}

fn global_runtime_features(request: &DecisionRequest, state: &DecisionState) -> Vec<f32> {
    let mut features = vec![
        clamp_non_negative(request.shanten, 6) as f32 / 6.0,
        clamp_non_negative(request.best_ukeire, 40) as f32 / 40.0,
        clamp_non_negative(state.total_dora_in_hand, 13) as f32 / 13.0,
        request.has_open_hand as u8 as f32,
        request.self_riichi_accepted as u8 as f32,
        request.can_riichi as u8 as f32,
        request.kyoku.saturating_sub(1).clamp(0, 3) as f32 / 3.0,
        clamp_non_negative(request.honba, 10) as f32 / 10.0,
        clamp_non_negative(request.kyotaku, 10) as f32 / 10.0,
    ];

    for wind in WIND_TILES {
        features.push((request.bakaze == wind) as u8 as f32);
    }
    for player_index in 0..4 {
        features.push((request.player_id == player_index) as u8 as f32);
    }
    for tile in TILE_TYPES {
        let count = *state.hand_counts.get(tile).unwrap_or(&0);
        features.push(clamp_non_negative(count, 4) as f32 / 4.0);
    }
    for tile in TILE_TYPES {
        let count = *state.tiles_seen.get(tile).unwrap_or(&0);
        features.push(clamp_non_negative(count, 4) as f32 / 4.0);
    }
    for tile in TILE_TYPES {
        let count = *state.dora_targets.get(tile).unwrap_or(&0);
        features.push(clamp_non_negative(count, 4) as f32 / 4.0);
    }

    features
}

fn candidate_runtime_features(candidate: &ActionCandidate) -> Result<Vec<f32>> {
    let primary_tile = candidate
        .primary_tile
        .as_deref()
        .or(candidate.discard_tile.as_deref())
        .ok_or_else(|| anyhow::anyhow!("candidate is missing both primary_tile and discard_tile"))?;

    let mut features = vec![1.0];
    features.extend(candidate.action_type.one_hot());
    features.extend([
        clamp_non_negative(candidate.next_shanten, 6) as f32 / 6.0,
        clamp_non_negative(candidate.ukeire, 40) as f32 / 40.0,
        clamp_non_negative(candidate.improving_count, 34) as f32 / 34.0,
        clamp_non_negative(candidate.discard_candidate_count, MAX_ACTION_CANDIDATES as i32) as f32
            / MAX_ACTION_CANDIDATES as f32,
        (candidate.baseline_score as f32 / 200.0).tanh(),
        (candidate.discard_bonus as f32 / 80.0).tanh(),
        clamp_non_negative(candidate.tile_seen, 4) as f32 / 4.0,
        clamp_non_negative(candidate.tile_dora, 4) as f32 / 4.0,
        candidate.is_tsumogiri as u8 as f32,
    ]);
    features.extend(tile_one_hot(primary_tile)?);
    features.extend(tile_histogram(&candidate.consumed_tiles)?);
    Ok(features)
}

fn tile_one_hot(tile: &str) -> Result<Vec<f32>> {
    let mut vector = vec![0.0; TILE_TYPES.len()];
    vector[tile_index(tile)?] = 1.0;
    Ok(vector)
}

fn tile_histogram(tiles: &[String]) -> Result<Vec<f32>> {
    let mut vector = vec![0.0; TILE_TYPES.len()];
    for tile in tiles {
        vector[tile_index(tile)?] += 1.0;
    }
    Ok(vector)
}

fn tile_index(tile: &str) -> Result<usize> {
    let normalized = base_tile(tile);
    TILE_TYPES
        .iter()
        .position(|candidate| *candidate == normalized)
        .ok_or_else(|| anyhow!("unknown tile: {tile}"))
}

fn tile_sort_key(tile: &str) -> (usize, usize) {
    (
        tile_index(tile).expect("validated tile sort key"),
        usize::from(tile.ends_with('r')),
    )
}

pub fn dora_from_indicator(indicator: &str) -> Result<String> {
    let normalized = base_tile(indicator);
    match normalized {
        "E" => Ok("S".to_string()),
        "S" => Ok("W".to_string()),
        "W" => Ok("N".to_string()),
        "N" => Ok("E".to_string()),
        "P" => Ok("F".to_string()),
        "F" => Ok("C".to_string()),
        "C" => Ok("P".to_string()),
        _ => {
            let mut chars = normalized.chars();
            let number = chars
                .next()
                .and_then(|value| value.to_digit(10))
                .ok_or_else(|| anyhow::anyhow!("invalid suit tile indicator: {indicator}"))?;
            let color = chars
                .next()
                .ok_or_else(|| anyhow::anyhow!("invalid suit tile indicator: {indicator}"))?;
            let next_number = if number == 9 { 1 } else { number + 1 };
            Ok(format!("{next_number}{color}"))
        }
    }
}

pub fn tile_dora_value(tile: &str, dora_targets: &HashMap<String, i32>) -> i32 {
    let mut value = *dora_targets.get(base_tile(tile)).unwrap_or(&0);
    if tile.ends_with('r') {
        value += 1;
    }
    value
}

pub fn base_tile(tile: &str) -> &str {
    if tile.len() >= 2 {
        &tile[..2]
    } else {
        tile
    }
}

fn ensure_known_tile(tile: &str) -> Result<()> {
    tile_index(tile).map(|_| ())
}

fn clamp_non_negative(value: i32, upper: i32) -> i32 {
    value.clamp(0, upper)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_tile_strips_red_suffix() {
        assert_eq!(base_tile("5mr"), "5m");
        assert_eq!(base_tile("E"), "E");
    }

    #[test]
    fn dora_indicator_wraps_correctly() {
        assert_eq!(dora_from_indicator("9m").unwrap(), "1m");
        assert_eq!(dora_from_indicator("N").unwrap(), "E");
        assert_eq!(dora_from_indicator("C").unwrap(), "P");
    }

    #[test]
    fn compile_discard_candidates_matches_python_ordering() {
        let request = CandidateCompileRequest {
            phase: DecisionPhase::Discard,
            shanten: 1,
            best_ukeire: 8,
            has_open_hand: false,
            hand_tiles: vec![
                "1m".to_string(),
                "2m".to_string(),
                "3m".to_string(),
                "4p".to_string(),
                "5pr".to_string(),
                "6p".to_string(),
                "7s".to_string(),
                "8s".to_string(),
                "9s".to_string(),
                "E".to_string(),
                "E".to_string(),
                "P".to_string(),
                "F".to_string(),
            ],
            tiles_seen: HashMap::from([("1m".to_string(), 1), ("E".to_string(), 2)]),
            dora_indicators: vec!["4p".to_string(), "N".to_string()],
            forbidden_tiles: HashMap::from([("1m".to_string(), false), ("E".to_string(), false)]),
            yakuhai_tiles: vec!["E".to_string()],
            last_self_tsumo: Some("F".to_string()),
            last_kawa_tile: None,
            riichi_discardable_tiles: Vec::new(),
            improving_tiles: vec![
                RawDiscardCandidate {
                    discard_tile: Some("E".to_string()),
                    ukeire: 8,
                    improving_count: 5,
                },
                RawDiscardCandidate {
                    discard_tile: Some("1m".to_string()),
                    ukeire: 6,
                    improving_count: 4,
                },
            ],
            pon_candidates: Vec::new(),
            chi_candidates: Vec::new(),
        };

        let compiled = compile_candidates(request).unwrap();
        let labels = compiled
            .candidates
            .iter()
            .map(|candidate| candidate.action_label.as_str())
            .collect::<Vec<_>>();
        assert_eq!(labels, vec!["1m", "E"]);
    }

    #[test]
    fn compile_call_candidates_keeps_pass_first() {
        let request = CandidateCompileRequest {
            phase: DecisionPhase::Call,
            shanten: 1,
            best_ukeire: 8,
            has_open_hand: false,
            hand_tiles: vec![
                "1m".to_string(),
                "2m".to_string(),
                "3m".to_string(),
                "4p".to_string(),
                "5p".to_string(),
                "6p".to_string(),
                "7s".to_string(),
                "8s".to_string(),
                "9s".to_string(),
                "E".to_string(),
                "E".to_string(),
                "P".to_string(),
                "F".to_string(),
            ],
            tiles_seen: HashMap::from([("E".to_string(), 2)]),
            dora_indicators: vec!["4p".to_string()],
            forbidden_tiles: HashMap::new(),
            yakuhai_tiles: vec!["E".to_string()],
            last_self_tsumo: None,
            last_kawa_tile: Some("E".to_string()),
            riichi_discardable_tiles: Vec::new(),
            improving_tiles: Vec::new(),
            pon_candidates: vec![RawCallCandidate {
                consumed_tiles: vec!["E".to_string(), "E".to_string()],
                next_shanten: 0,
                next_ukeire: 10,
                discard_candidate_count: 2,
            }],
            chi_candidates: vec![RawCallCandidate {
                consumed_tiles: vec!["1m".to_string(), "2m".to_string()],
                next_shanten: 2,
                next_ukeire: 3,
                discard_candidate_count: 1,
            }],
        };

        let compiled = compile_candidates(request).unwrap();
        assert_eq!(compiled.candidates[0].action_type, ActionType::Pass);
        assert_eq!(compiled.candidates[1].action_type, ActionType::Pon);
    }

    #[test]
    fn encode_decision_matches_contract_dimensions() {
        let request = DecisionRequest {
            shanten: 1,
            best_ukeire: 8,
            bakaze: "E".to_string(),
            kyoku: 1,
            honba: 0,
            kyotaku: 0,
            player_id: 0,
            self_riichi_accepted: false,
            can_riichi: true,
            has_open_hand: false,
            hand_tiles: vec![
                "1m".to_string(),
                "2m".to_string(),
                "3m".to_string(),
                "4p".to_string(),
                "5pr".to_string(),
                "6p".to_string(),
                "7s".to_string(),
                "8s".to_string(),
                "9s".to_string(),
                "E".to_string(),
                "E".to_string(),
                "P".to_string(),
                "F".to_string(),
            ],
            tiles_seen: HashMap::from([("1m".to_string(), 1), ("E".to_string(), 2)]),
            dora_indicators: vec!["4p".to_string(), "N".to_string()],
            candidates: vec![ActionCandidate {
                action_type: ActionType::Discard,
                action_label: "E".to_string(),
                primary_tile: Some("E".to_string()),
                discard_tile: Some("E".to_string()),
                consumed_tiles: vec![],
                next_shanten: 1,
                next_ukeire: 8,
                ukeire: 8,
                improving_count: 5,
                discard_candidate_count: 0,
                baseline_score: 120,
                discard_bonus: 14,
                tile_seen: 2,
                tile_count: 2,
                tile_dora: 0,
                is_tsumogiri: false,
            }],
        };

        let encoded = encode_decision(request).unwrap();
        assert_eq!(encoded.features.len(), INPUT_DIM);
        assert_eq!(encoded.legal_actions.len(), ACTION_DIM);
        assert!(encoded.legal_actions[0]);
        assert!(!encoded.legal_actions[1]);
    }
}