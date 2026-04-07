use std::collections::BTreeMap;

use anyhow::{Context, Result, anyhow, bail};
use mjai_core::mjai::Event;
use mjai_core::state::PlayerState;
use mjai_core::tools::{calc_shanten, find_improving_tiles};
use serde_json::{Value, json};

use crate::bot_runtime::{
    ActionCandidate, CandidateCompileRequest, DecisionPhase, DecisionRequest,
    RawCallCandidate, RawDiscardCandidate, compile_candidates, encode_decision,
};

const TILE_TYPES: [&str; 34] = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
    "5p", "6p", "7p", "8p", "9p", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s",
    "9s", "E", "S", "W", "N", "P", "F", "C",
];

#[derive(Debug)]
pub struct RustMjaiBotState {
    player_id: u8,
    player_state: PlayerState,
    discard_events: Vec<Value>,
    call_events: Vec<Value>,
    dora_indicators: Vec<String>,
}

impl RustMjaiBotState {
    pub fn new(player_id: u8) -> Self {
        Self {
            player_id,
            player_state: PlayerState::new(player_id),
            discard_events: Vec::new(),
            call_events: Vec::new(),
            dora_indicators: Vec::new(),
        }
    }

    pub fn react_batch_value(&mut self, events_value: &Value) -> Result<Value> {
        let event_values = events_value
            .as_array()
            .cloned()
            .ok_or_else(|| anyhow!("react payload must contain an events array"))?;
        if event_values.is_empty() {
            bail!("react payload cannot contain an empty events array");
        }

        for event_value in &event_values {
            self.record_event_value(event_value);
            let event: Event = serde_json::from_value(event_value.clone())
                .context("failed to parse mjai event")?;
            let _ = self.player_state.update(&event);
        }

        let reaction = self.default_reaction()?;
        self.snapshot(&reaction)
    }

    pub fn validate_reaction_json(&self, reaction_json: &str) -> Result<()> {
        let action: Event = serde_json::from_str(reaction_json)
            .context("failed to parse reaction JSON")?;
        self.player_state.validate_reaction(&action)
    }

    pub fn brief_info(&self) -> String {
        self.player_state.brief_info()
    }

    /// Produce a full decision after reacting to events: snapshot + compiled candidates + encoded features.
    pub fn react_and_decide(&mut self, events_value: &Value) -> Result<Value> {
        let event_values = events_value
            .as_array()
            .cloned()
            .ok_or_else(|| anyhow!("react payload must contain an events array"))?;
        if event_values.is_empty() {
            bail!("react payload cannot contain an empty events array");
        }

        for event_value in &event_values {
            self.record_event_value(event_value);
            let event: Event = serde_json::from_value(event_value.clone())
                .context("failed to parse mjai event")?;
            let _ = self.player_state.update(&event);
        }

        let reaction = self.default_reaction()?;
        let snapshot = self.snapshot(&reaction)?;

        let cans = self.player_state.last_cans();
        let phase = self.phase();

        // For non-actionable states, return snapshot only (no decision)
        if phase == "idle" || (!cans.can_discard && !cans.can_pon && !cans.can_chi()) {
            return Ok(json!({
                "ok": true,
                "snapshot": snapshot,
                "decision": null,
            }));
        }

        // Auto-actions: riichi-tsumogiri, agari, etc. are handled in Python; we only
        // produce decisions for states that need candidate compilation + feature encoding.
        let compile_request = self.build_compile_request()?;
        let compiled = compile_candidates(compile_request)?;

        if compiled.candidates.is_empty() {
            return Ok(json!({
                "ok": true,
                "snapshot": snapshot,
                "decision": null,
            }));
        }

        let encode_request = self.build_encode_request(&compiled.candidates)?;
        let encoded = encode_decision(encode_request)?;

        Ok(json!({
            "ok": true,
            "snapshot": snapshot,
            "decision": {
                "phase": phase,
                "candidates": compiled.candidates,
                "features": encoded.features,
                "legal_actions": encoded.legal_actions,
            },
        }))
    }

    fn build_compile_request(&self) -> Result<CandidateCompileRequest> {
        let cans = self.player_state.last_cans();
        let phase = match self.phase() {
            "riichi_discard" => DecisionPhase::RiichiDiscard,
            "call" => DecisionPhase::Call,
            "discard" => DecisionPhase::Discard,
            _ => DecisionPhase::Idle,
        };

        let improving_tiles_json = if cans.can_discard {
            self.find_improving_tiles_json()?
        } else {
            Value::Array(Vec::new())
        };
        let improving_tiles = self.json_to_raw_discard_candidates(&improving_tiles_json);
        let best_ukeire = improving_tiles.iter().map(|c| c.ukeire).max().unwrap_or(0);

        let pon_candidates = if cans.can_pon {
            let json = self.find_pon_candidates_json()?;
            self.json_to_raw_call_candidates(&json)
        } else {
            Vec::new()
        };
        let chi_candidates = if cans.can_chi() {
            let json = self.find_chi_candidates_json()?;
            self.json_to_raw_call_candidates(&json)
        } else {
            Vec::new()
        };

        let has_open_hand = self.has_open_hand();
        let yakuhai_tiles = self.yakuhai_tiles();

        Ok(CandidateCompileRequest {
            phase,
            shanten: i32::from(self.player_state.shanten()) - 1,
            best_ukeire,
            has_open_hand,
            hand_tiles: self.tehai_mjai(),
            tiles_seen: self.tiles_seen_map().into_iter().map(|(k, v)| (k, i32::from(v))).collect(),
            dora_indicators: self.dora_indicators.clone(),
            forbidden_tiles: self.forbidden_tiles_map().into_iter().collect(),
            yakuhai_tiles,
            last_self_tsumo: self.player_state.last_self_tsumo().map(|t| t.to_string()),
            last_kawa_tile: self.player_state.last_kawa_tile().map(|t| t.to_string()),
            riichi_discardable_tiles: if phase == DecisionPhase::RiichiDiscard {
                self.discardable_tiles_riichi_declaration()
            } else {
                Vec::new()
            },
            improving_tiles,
            pon_candidates,
            chi_candidates,
        })
    }

    fn build_encode_request(&self, candidates: &[ActionCandidate]) -> Result<DecisionRequest> {
        let improving_tiles_json = self.find_improving_tiles_json()?;
        let improving_tiles = self.json_to_raw_discard_candidates(&improving_tiles_json);
        let best_ukeire = improving_tiles.iter().map(|c| c.ukeire).max().unwrap_or(0);

        Ok(DecisionRequest {
            shanten: i32::from(self.player_state.shanten()) - 1,
            best_ukeire,
            bakaze: wind_name(self.player_state.bakaze()).unwrap_or("E").to_string(),
            kyoku: i32::from(self.player_state.kyoku()) + 1,
            honba: i32::from(self.player_state.honba()),
            kyotaku: i32::from(self.player_state.kyotaku()),
            player_id: usize::from(self.player_id),
            self_riichi_accepted: self.player_state.self_riichi_accepted(),
            can_riichi: self.player_state.last_cans().can_riichi,
            has_open_hand: self.has_open_hand(),
            hand_tiles: self.tehai_mjai(),
            tiles_seen: self.tiles_seen_map().into_iter().map(|(k, v)| (k, i32::from(v))).collect(),
            dora_indicators: self.dora_indicators.clone(),
            candidates: candidates.to_vec(),
        })
    }

    fn has_open_hand(&self) -> bool {
        self.call_events(Some(self.player_id)).iter().any(|event| {
            matches!(
                event.get("type").and_then(Value::as_str),
                Some("chi") | Some("pon") | Some("daiminkan") | Some("kakan")
            )
        })
    }

    fn yakuhai_tiles(&self) -> Vec<String> {
        let jikaze = wind_name(self.player_state.jikaze());
        let bakaze = wind_name(self.player_state.bakaze());
        let mut tiles = Vec::new();
        for tile in &["E", "S", "W", "N", "P", "F", "C"] {
            if Some(*tile) == jikaze || Some(*tile) == bakaze || matches!(*tile, "P" | "F" | "C") {
                tiles.push(tile.to_string());
            }
        }
        tiles
    }

    fn json_to_raw_discard_candidates(&self, json: &Value) -> Vec<RawDiscardCandidate> {
        let Some(array) = json.as_array() else {
            return Vec::new();
        };
        array
            .iter()
            .map(|item| RawDiscardCandidate {
                discard_tile: item.get("discard_tile").and_then(Value::as_str).map(str::to_owned),
                ukeire: item.get("ukeire").and_then(Value::as_i64).unwrap_or(0) as i32,
                improving_count: item
                    .get("improving_count")
                    .and_then(Value::as_i64)
                    .or_else(|| {
                        item.get("improving_tiles")
                            .and_then(Value::as_array)
                            .map(|a| a.len() as i64)
                    })
                    .unwrap_or(0) as i32,
            })
            .collect()
    }

    fn json_to_raw_call_candidates(&self, json: &Value) -> Vec<RawCallCandidate> {
        let Some(array) = json.as_array() else {
            return Vec::new();
        };
        array
            .iter()
            .map(|item| {
                let consumed_tiles = item
                    .get("consumed")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                RawCallCandidate {
                    consumed_tiles,
                    next_shanten: item.get("next_shanten").and_then(Value::as_i64).unwrap_or(0) as i32,
                    next_ukeire: item.get("next_ukeire").and_then(Value::as_i64).unwrap_or(0) as i32,
                    discard_candidate_count: item
                        .get("discard_candidate_count")
                        .and_then(Value::as_i64)
                        .or_else(|| {
                            item.get("discard_candidates")
                                .and_then(Value::as_array)
                                .map(|a| a.len() as i64)
                        })
                        .unwrap_or(0) as i32,
                }
            })
            .collect()
    }

    pub fn snapshot(&self, reaction: &str) -> Result<Value> {
        let cans = self.player_state.last_cans();
        let tehai_mjai = self.tehai_mjai();
        let tiles_seen = self.tiles_seen_map();
        let forbidden_tiles = self.forbidden_tiles_map();

        let capabilities = json!({
            "can_act": cans.can_act(),
            "can_agari": cans.can_agari(),
            "can_ankan": cans.can_ankan,
            "can_chi": cans.can_chi(),
            "can_chi_high": cans.can_chi_high,
            "can_chi_low": cans.can_chi_low,
            "can_chi_mid": cans.can_chi_mid,
            "can_daiminkan": cans.can_daiminkan,
            "can_discard": cans.can_discard,
            "can_kakan": cans.can_kakan,
            "can_kan": cans.can_kan(),
            "can_pass": cans.can_pass(),
            "can_pon": cans.can_pon,
            "can_riichi": cans.can_riichi,
            "can_ron_agari": cans.can_ron_agari,
            "can_ryukyoku": cans.can_ryukyoku,
            "can_tsumo_agari": cans.can_tsumo_agari,
        });

        let state = json!({
            "player_id": self.player_id,
            "phase": self.phase(),
            "reaction": reaction,
            "bakaze": wind_name(self.player_state.bakaze()).map(str::to_owned),
            "jikaze": wind_name(self.player_state.jikaze()).map(str::to_owned),
            "honba": self.player_state.honba(),
            "kyoku": self.player_state.kyoku() + 1,
            "kyotaku": self.player_state.kyotaku(),
            "scores": self.player_state.scores().to_vec(),
            "target_actor": cans.target_actor,
            "target_actor_rel": (cans.target_actor + 4 - self.player_id) % 4,
            "last_self_tsumo": self.player_state.last_self_tsumo().map(|tile| tile.to_string()).unwrap_or_default(),
            "last_kawa_tile": self.player_state.last_kawa_tile().map(|tile| tile.to_string()),
            "self_riichi_declared": self.player_state.self_riichi_declared(),
            "self_riichi_accepted": self.player_state.self_riichi_accepted(),
            "at_furiten": self.player_state.at_furiten(),
            "is_oya": self.player_state.is_oya(),
            "akas_in_hand": self.player_state.akas_in_hand().to_vec(),
            "tehai": self.tehai_short(),
            "tehai_mjai": tehai_mjai,
            "tehai_vec34": self.player_state.tehai().to_vec(),
            "shanten": i32::from(self.player_state.shanten()) - 1,
            "dora_indicators": self.dora_indicators.clone(),
            "tiles_seen": tiles_seen,
            "forbidden_tiles": forbidden_tiles,
            "discarded_tiles_all": self.discarded_tiles(None),
            "discarded_tiles_self": self.discarded_tiles(Some(self.player_id)),
            "call_events_all": self.call_events(None),
            "call_events_self": self.call_events(Some(self.player_id)),
        });

        let queries = json!({
            "discardable_tiles": self.discardable_tiles(),
            "discardable_tiles_riichi_declaration": self.discardable_tiles_riichi_declaration(),
            "improving_tiles": if cans.can_discard { self.find_improving_tiles_json()? } else { Value::Array(Vec::new()) },
            "pon_candidates": if cans.can_pon { self.find_pon_candidates_json()? } else { Value::Array(Vec::new()) },
            "chi_candidates": if cans.can_chi() { self.find_chi_candidates_json()? } else { Value::Array(Vec::new()) },
        });

        Ok(json!({
            "capabilities": capabilities,
            "state": state,
            "queries": queries,
        }))
    }

    pub fn default_reaction(&self) -> Result<String> {
        let cans = self.player_state.last_cans();
        if self.player_state.self_riichi_accepted()
            && !(cans.can_agari() || cans.can_kakan || cans.can_ankan)
            && cans.can_discard
        {
            return self.action_discard(
                &self
                    .player_state
                    .last_self_tsumo()
                    .map(|tile| tile.to_string())
                    .unwrap_or_default(),
            );
        }
        if cans.can_discard {
            return self.action_discard(
                &self
                    .player_state
                    .last_self_tsumo()
                    .map(|tile| tile.to_string())
                    .unwrap_or_default(),
            );
        }
        Ok(json!({"type":"none"}).to_string())
    }

    fn record_event_value(&mut self, event_value: &Value) {
        let event_type = event_value.get("type").and_then(Value::as_str);
        match event_type {
            Some("start_game") => {
                self.discard_events.clear();
                self.call_events.clear();
                self.dora_indicators.clear();
            }
            Some("start_kyoku") | Some("dora") => {
                if let Some(dora_marker) = event_value.get("dora_marker").and_then(Value::as_str) {
                    self.dora_indicators.push(dora_marker.to_string());
                }
            }
            Some("dahai") => self.discard_events.push(event_value.clone()),
            Some("chi") | Some("pon") | Some("daiminkan") | Some("kakan") | Some("ankan") => {
                self.call_events.push(event_value.clone())
            }
            _ => {}
        }
    }

    fn phase(&self) -> &'static str {
        let cans = self.player_state.last_cans();
        if cans.can_discard && self.player_state.self_riichi_declared() && !self.player_state.self_riichi_accepted() {
            return "riichi_discard";
        }
        if cans.can_pon || cans.can_chi() {
            return "call";
        }
        if cans.can_discard {
            return "discard";
        }
        "idle"
    }

    fn action_discard(&self, tile_str: &str) -> Result<String> {
        let last_self_tsumo = self
            .player_state
            .last_self_tsumo()
            .map(|tile| tile.to_string())
            .unwrap_or_default();
        Ok(format!(
            "{{\"type\":\"dahai\",\"pai\":\"{}\",\"actor\":{},\"tsumogiri\":{}}}",
            tile_str,
            self.player_id,
            if tile_str == last_self_tsumo { "true" } else { "false" }
        ))
    }

    fn tehai_mjai(&self) -> Vec<String> {
        let tehai = self.player_state.tehai();
        let akas = self.player_state.akas_in_hand();
        let mut tiles = Vec::new();
        for (tile_idx, mut tile_count) in tehai.into_iter().enumerate() {
            if tile_idx == 4 && akas[0] {
                tile_count = tile_count.saturating_sub(1);
                tiles.push("5mr".to_string());
            } else if tile_idx == 4 + 9 && akas[1] {
                tile_count = tile_count.saturating_sub(1);
                tiles.push("5pr".to_string());
            } else if tile_idx == 4 + 18 && akas[2] {
                tile_count = tile_count.saturating_sub(1);
                tiles.push("5sr".to_string());
            }
            for _ in 0..tile_count {
                tiles.push(TILE_TYPES[tile_idx].to_string());
            }
        }
        tiles
    }

    fn tehai_short(&self) -> String {
        let mut tehai = convert_vec34_to_short(&self.player_state.tehai(), &self.player_state.akas_in_hand());
        tehai.push_str(&self.fmt_calls(Some(self.player_id)));
        tehai
    }

    fn tiles_seen_map(&self) -> BTreeMap<String, u8> {
        TILE_TYPES
            .iter()
            .enumerate()
            .map(|(index, tile)| ((*tile).to_string(), self.player_state.tiles_seen()[index]))
            .collect()
    }

    fn forbidden_tiles_map(&self) -> BTreeMap<String, bool> {
        TILE_TYPES
            .iter()
            .enumerate()
            .map(|(index, tile)| ((*tile).to_string(), self.player_state.forbidden_tiles()[index]))
            .collect()
    }

    fn discarded_tiles(&self, player_id: Option<u8>) -> Vec<String> {
        self.discard_events
            .iter()
            .filter(|event| match player_id {
                Some(player_id) => event.get("actor").and_then(Value::as_u64) == Some(u64::from(player_id)),
                None => true,
            })
            .filter_map(|event| event.get("pai").and_then(Value::as_str))
            .map(str::to_owned)
            .collect()
    }

    fn call_events(&self, player_id: Option<u8>) -> Vec<Value> {
        self.call_events
            .iter()
            .filter(|event| match player_id {
                Some(player_id) => event.get("actor").and_then(Value::as_u64) == Some(u64::from(player_id)),
                None => true,
            })
            .cloned()
            .collect()
    }

    fn discardable_tiles(&self) -> Vec<String> {
        let forbidden_tiles = self.forbidden_tiles_map();
        let mut output = Vec::new();
        for tile in self.tehai_mjai() {
            if *forbidden_tiles.get(base_tile(&tile)).unwrap_or(&false) {
                continue;
            }
            if !output.contains(&tile) {
                output.push(tile);
            }
        }
        output.sort_by_key(|tile| tile_sort_key(tile));
        output
    }

    fn discardable_tiles_riichi_declaration(&self) -> Vec<String> {
        let tehai_mjai = self.tehai_mjai();
        let mut output = Vec::new();
        for discard_index in 0..tehai_mjai.len() {
            let mut reduced = tehai_mjai.clone();
            let discarded_tile = reduced.remove(discard_index);
            let reduced_vec34 = convert_mjai_to_vec34(&reduced);
            let reduced_short = convert_vec34_to_short(&reduced_vec34, &self.player_state.akas_in_hand());
            if calc_shanten(&reduced_short) == 0 && !output.contains(&discarded_tile) {
                output.push(discarded_tile);
            }
        }
        output.sort_by_key(|tile| tile_sort_key(tile));
        output
    }

    fn find_improving_tiles_json(&self) -> Result<Value> {
        let tehai_short = self.tehai_short();
        let tehai_vec34 = self.player_state.tehai();
        let tiles_seen = self.tiles_seen_map();
        let mut raw_candidates = find_improving_tiles(&tehai_short);
        raw_candidates.sort_by(|left, right| right.1.len().cmp(&left.1.len()));

        let mut candidates = raw_candidates
            .into_iter()
            .map(|(discard_index, improving_indices)| {
                let discard_tile = if discard_index < 34 {
                    aka_tile_if_needed(TILE_TYPES[discard_index as usize], &tehai_vec34, &self.player_state.akas_in_hand())
                } else {
                    String::new()
                };
                let improving_tiles = improving_indices
                    .into_iter()
                    .map(|tile_index| TILE_TYPES[tile_index as usize].to_string())
                    .collect::<Vec<_>>();
                let ukeire = improving_tiles
                    .iter()
                    .map(|tile| 4_i32 - i32::from(*tiles_seen.get(base_tile(tile)).unwrap_or(&0)))
                    .sum::<i32>();
                json!({
                    "discard_tile": discard_tile,
                    "improving_tiles": improving_tiles,
                    "ukeire": ukeire,
                })
            })
            .collect::<Vec<_>>();

        candidates.sort_by(|left, right| {
            right
                .get("ukeire")
                .and_then(Value::as_i64)
                .cmp(&left.get("ukeire").and_then(Value::as_i64))
        });
        Ok(Value::Array(candidates))
    }

    fn find_pon_candidates_json(&self) -> Result<Value> {
        let current_shanten = calc_shanten(&self.tehai_short());
        let current_improving_tiles = self.find_improving_tiles_json()?;
        let mut current_ukeire = 0_i32;
        if let Some(improving_tiles) = current_improving_tiles.as_array() {
            for improving in improving_tiles {
                current_ukeire = improving.get("ukeire").and_then(Value::as_i64).unwrap_or(0) as i32;
            }
        }

        let tehai_mjai = self.tehai_mjai();
        let last_kawa_tile = self
            .player_state
            .last_kawa_tile()
            .map(|tile| tile.to_string())
            .unwrap_or_default();
        if last_kawa_tile.is_empty() {
            return Ok(Value::Array(Vec::new()));
        }

        let mut pon_candidates = Vec::new();
        if last_kawa_tile.starts_with('5') && !last_kawa_tile.ends_with('z') {
            let base_tile_value = base_tile(&last_kawa_tile).to_string();
            let base_count = tehai_mjai.iter().filter(|tile| tile.as_str() == base_tile_value).count();
            if base_count >= 2 {
                pon_candidates.push(self.new_pon_candidate(vec![base_tile_value.clone(), base_tile_value], current_shanten, current_ukeire)?);
            } else {
                let red_tile = format!("{}r", base_tile(&last_kawa_tile));
                let red_count = tehai_mjai.iter().filter(|tile| tile.as_str() == red_tile).count();
                if red_count == 1 {
                    pon_candidates.push(self.new_pon_candidate(vec![base_tile(&last_kawa_tile).to_string(), red_tile], current_shanten, current_ukeire)?);
                }
            }
            return Ok(Value::Array(pon_candidates));
        }

        pon_candidates.push(self.new_pon_candidate(vec![last_kawa_tile.clone(), last_kawa_tile], current_shanten, current_ukeire)?);
        Ok(Value::Array(pon_candidates))
    }

    fn new_pon_candidate(&self, consumed: Vec<String>, current_shanten: i8, current_ukeire: i32) -> Result<Value> {
        self.new_call_candidate("pon", consumed, current_shanten, current_ukeire)
    }

    fn find_chi_candidates_json(&self) -> Result<Value> {
        let cans = self.player_state.last_cans();
        let current_shanten = calc_shanten(&self.tehai_short());
        let current_improving_tiles = self.find_improving_tiles_json()?;
        let mut current_ukeire = 0_i32;
        if let Some(improving_tiles) = current_improving_tiles.as_array() {
            for improving in improving_tiles {
                current_ukeire = improving.get("ukeire").and_then(Value::as_i64).unwrap_or(0) as i32;
            }
        }

        let last_kawa_tile = self
            .player_state
            .last_kawa_tile()
            .map(|tile| tile.to_string())
            .unwrap_or_default();
        if last_kawa_tile.len() < 2 || last_kawa_tile.ends_with('z') {
            return Ok(Value::Array(Vec::new()));
        }
        let color = last_kawa_tile.chars().nth(1).unwrap_or('m');
        let number = last_kawa_tile.chars().next().and_then(|value| value.to_digit(10)).unwrap_or(0) as i32;

        let mut chi_candidates = Vec::new();
        let tehai_mjai = self.tehai_mjai();
        let mut push_if_present = |tiles: [&str; 2]| -> Result<()> {
            if tiles.iter().all(|tile| tehai_mjai.contains(&tile.to_string())) {
                chi_candidates.push(self.new_call_candidate(
                    "chi",
                    vec![tiles[0].to_string(), tiles[1].to_string()],
                    current_shanten,
                    current_ukeire,
                )?);
            }
            Ok(())
        };

        if cans.can_chi_high && number >= 3 {
            push_if_present([
                &format!("{}{}", number - 2, color),
                &format!("{}{}", number - 1, color),
            ])?;
            push_if_present([
                &format!("{}{}r", number - 2, color),
                &format!("{}{}", number - 1, color),
            ])?;
            push_if_present([
                &format!("{}{}", number - 2, color),
                &format!("{}{}r", number - 1, color),
            ])?;
        }
        if cans.can_chi_low && number <= 7 {
            push_if_present([
                &format!("{}{}", number + 1, color),
                &format!("{}{}", number + 2, color),
            ])?;
            push_if_present([
                &format!("{}{}r", number + 1, color),
                &format!("{}{}", number + 2, color),
            ])?;
            push_if_present([
                &format!("{}{}", number + 1, color),
                &format!("{}{}r", number + 2, color),
            ])?;
        }
        if cans.can_chi_mid && (2..=8).contains(&number) {
            push_if_present([
                &format!("{}{}", number - 1, color),
                &format!("{}{}", number + 1, color),
            ])?;
            push_if_present([
                &format!("{}{}r", number - 1, color),
                &format!("{}{}", number + 1, color),
            ])?;
            push_if_present([
                &format!("{}{}", number - 1, color),
                &format!("{}{}r", number + 1, color),
            ])?;
        }

        Ok(Value::Array(chi_candidates))
    }

    fn new_call_candidate(
        &self,
        call_type: &str,
        consumed: Vec<String>,
        current_shanten: i8,
        current_ukeire: i32,
    ) -> Result<Value> {
        let mut new_tehai_mjai = self.tehai_mjai();
        for consumed_tile in &consumed {
            remove_first_match(&mut new_tehai_mjai, consumed_tile);
        }
        let target_actor = self.player_state.last_cans().target_actor;
        let last_kawa_tile = self
            .player_state
            .last_kawa_tile()
            .map(|tile| tile.to_string())
            .unwrap_or_default();
        let call_event = json!({
            "type": call_type,
            "consumed": consumed,
            "pai": last_kawa_tile,
            "target": target_actor,
            "actor": self.player_id,
        });
        let mut tehai_str = convert_vec34_to_short(
            &convert_mjai_to_vec34(&new_tehai_mjai),
            &self.player_state.akas_in_hand(),
        );
        tehai_str.push_str(&self.fmt_calls(Some(self.player_id)));
        let new_call_str = fmt_call_value(&call_event, self.player_id);
        let tehai_with_call = format!("{tehai_str}{new_call_str}");

        let new_shanten = calc_shanten(&tehai_with_call);
        let mut raw_candidates = find_improving_tiles(&tehai_with_call);
        raw_candidates.sort_by(|left, right| right.1.len().cmp(&left.1.len()));

        let tiles_seen = self.tiles_seen_map();
        let mut discard_candidates = Vec::new();
        let mut next_best_ukeire = 0_i32;
        for (discard_index, improving_indices) in raw_candidates {
            let discard_tile = if discard_index < 34 {
                TILE_TYPES[discard_index as usize].to_string()
            } else {
                String::new()
            };
            let improving_tiles = improving_indices
                .into_iter()
                .map(|tile_index| TILE_TYPES[tile_index as usize].to_string())
                .collect::<Vec<_>>();
            let next_ukeire = improving_tiles
                .iter()
                .map(|tile| 4_i32 - i32::from(*tiles_seen.get(base_tile(tile)).unwrap_or(&0)))
                .sum::<i32>();
            next_best_ukeire = next_best_ukeire.max(next_ukeire);
            discard_candidates.push(json!({
                "discard_tile": discard_tile,
                "improving_tiles": improving_tiles,
                "ukeire": next_ukeire,
                "shanten": new_shanten,
            }));
        }

        Ok(json!({
            "consumed": consumed,
            "current_shanten": current_shanten,
            "current_ukeire": current_ukeire,
            "discard_candidates": discard_candidates,
            "next_shanten": new_shanten,
            "next_ukeire": next_best_ukeire,
        }))
    }

    fn fmt_calls(&self, player_id: Option<u8>) -> String {
        let filtered = self.call_events(player_id);
        let mut calls = Vec::new();
        let mut kakan_calls = Vec::new();
        for event in filtered {
            let call = fmt_call_value(&event, self.player_id);
            match event.get("type").and_then(Value::as_str) {
                Some("kakan") => kakan_calls.push(call),
                Some("chi") | Some("pon") | Some("daiminkan") | Some("ankan") => calls.push(call),
                _ => {}
            }
        }
        for kakan_call in kakan_calls {
            let tile = &kakan_call[2..4.min(kakan_call.len())];
            if let Some(index) = calls.iter().position(|call| call.len() >= 5 && &call[2..4] == tile && call.as_bytes()[1] == b'p') {
                let rel_pos = calls[index].chars().nth(4).unwrap_or('0');
                let mut chars = kakan_call.chars().collect::<Vec<_>>();
                if chars.len() > 4 {
                    chars[4] = rel_pos;
                }
                calls[index] = chars.into_iter().collect();
            }
        }
        calls.join("")
    }
}

fn convert_mjai_to_vec34(mjai_tiles: &[String]) -> [u8; 34] {
    let mut vec34_tiles = [0_u8; 34];
    for tile in mjai_tiles {
        let normalized = tile.replace('r', "");
        if let Some(index) = TILE_TYPES.iter().position(|candidate| *candidate == normalized) {
            vec34_tiles[index] += 1;
        }
    }
    vec34_tiles
}

fn convert_vec34_to_short(tehai_vec34: &[u8; 34], akas_in_hand: &[bool; 3]) -> String {
    let mut ms = Vec::new();
    let mut ps = Vec::new();
    let mut ss = Vec::new();
    let mut zs = Vec::new();
    for (tile_idx, tile_count) in tehai_vec34.iter().enumerate() {
        let mut count = usize::from(*tile_count);
        if tile_idx == 4 {
            if akas_in_hand[0] {
                ms.push('0');
                count = count.saturating_sub(1);
            }
            ms.extend(std::iter::repeat_n('5', count));
        } else if tile_idx == 13 {
            if akas_in_hand[1] {
                ps.push('0');
                count = count.saturating_sub(1);
            }
            ps.extend(std::iter::repeat_n('5', count));
        } else if tile_idx == 22 {
            if akas_in_hand[2] {
                ss.push('0');
                count = count.saturating_sub(1);
            }
            ss.extend(std::iter::repeat_n('5', count));
        } else if tile_idx < 9 {
            ms.extend(std::iter::repeat_n(char::from_digit((tile_idx + 1) as u32, 10).unwrap(), count));
        } else if tile_idx < 18 {
            ps.extend(std::iter::repeat_n(char::from_digit((tile_idx - 8) as u32, 10).unwrap(), count));
        } else if tile_idx < 27 {
            ss.extend(std::iter::repeat_n(char::from_digit((tile_idx - 17) as u32, 10).unwrap(), count));
        } else {
            zs.extend(std::iter::repeat_n(char::from_digit((tile_idx - 26) as u32, 10).unwrap(), count));
        }
    }

    let mut parts = Vec::new();
    if !ms.is_empty() {
        let mut part = ms.into_iter().collect::<String>();
        part.push('m');
        parts.push(part);
    }
    if !ps.is_empty() {
        let mut part = ps.into_iter().collect::<String>();
        part.push('p');
        parts.push(part);
    }
    if !ss.is_empty() {
        let mut part = ss.into_iter().collect::<String>();
        part.push('s');
        parts.push(part);
    }
    if !zs.is_empty() {
        let mut part = zs.into_iter().collect::<String>();
        part.push('z');
        parts.push(part);
    }
    parts.concat()
}

fn fmt_call_value(event: &Value, player_id: u8) -> String {
    match event.get("type").and_then(Value::as_str) {
        Some("pon") => {
            let rel_pos = rel_target(event, player_id);
            let call_tiles = [
                mjai_tile_to_short(event.get("pai").and_then(Value::as_str).unwrap_or_default()),
                mjai_tile_to_short(array_string(event, "consumed", 0)),
                mjai_tile_to_short(array_string(event, "consumed", 1)),
            ];
            format!(
                "(p{}{}{})",
                deaka_short_tile(&call_tiles[0]),
                rel_pos,
                if call_tiles.iter().any(|tile| is_aka_short_tile(tile)) { "r" } else { "" }
            )
        }
        Some("chi") => {
            let pai = event.get("pai").and_then(Value::as_str).unwrap_or_default();
            let color = pai.chars().nth(1).unwrap_or('m');
            let called = mjai_tile_to_short(pai);
            let mut nums = [
                called.chars().next().unwrap_or('0'),
                mjai_tile_to_short(array_string(event, "consumed", 0)).chars().next().unwrap_or('0'),
                mjai_tile_to_short(array_string(event, "consumed", 1)).chars().next().unwrap_or('0'),
            ];
            nums.sort_unstable();
            let consecutive_nums = nums.into_iter().collect::<String>();
            let called_tile_idx = if consecutive_nums.chars().nth(0) == called.chars().next() {
                0
            } else if consecutive_nums.chars().nth(1) == called.chars().next() {
                1
            } else {
                2
            };
            format!("({consecutive_nums}{color}{called_tile_idx})")
        }
        Some("daiminkan") => {
            let rel_pos = rel_target(event, player_id);
            let call_tiles = [
                mjai_tile_to_short(event.get("pai").and_then(Value::as_str).unwrap_or_default()),
                mjai_tile_to_short(array_string(event, "consumed", 0)),
                mjai_tile_to_short(array_string(event, "consumed", 1)),
                mjai_tile_to_short(array_string(event, "consumed", 2)),
            ];
            format!(
                "(k{}{}{})",
                deaka_short_tile(&call_tiles[0]),
                rel_pos,
                if call_tiles.iter().any(|tile| is_aka_short_tile(tile)) { "r" } else { "" }
            )
        }
        Some("ankan") => {
            let rel_pos = rel_target(event, player_id);
            let call_tiles = [
                mjai_tile_to_short(array_string(event, "consumed", 0)),
                mjai_tile_to_short(array_string(event, "consumed", 1)),
                mjai_tile_to_short(array_string(event, "consumed", 2)),
                mjai_tile_to_short(array_string(event, "consumed", 3)),
            ];
            format!(
                "(k{}{}{})",
                deaka_short_tile(&call_tiles[0]),
                rel_pos,
                if call_tiles.iter().any(|tile| is_aka_short_tile(tile)) { "r" } else { "" }
            )
        }
        Some("kakan") => {
            let call_tiles = [
                mjai_tile_to_short(event.get("pai").and_then(Value::as_str).unwrap_or_default()),
                mjai_tile_to_short(array_string(event, "consumed", 0)),
                mjai_tile_to_short(array_string(event, "consumed", 1)),
                mjai_tile_to_short(array_string(event, "consumed", 2)),
            ];
            format!(
                "(s{}0{})",
                deaka_short_tile(&call_tiles[0]),
                if call_tiles.iter().any(|tile| is_aka_short_tile(tile)) { "r" } else { "" }
            )
        }
        _ => String::new(),
    }
}

fn rel_target(event: &Value, player_id: u8) -> u8 {
    let target = event.get("target").and_then(Value::as_u64).unwrap_or(u64::from(player_id)) as u8;
    (target + 4 - player_id) % 4
}

fn mjai_tile_to_short(tile: &str) -> String {
    match tile {
        "5mr" => "0m".to_string(),
        "5pr" => "0s".to_string(),
        "5sr" => "0p".to_string(),
        "E" => "1z".to_string(),
        "S" => "2z".to_string(),
        "W" => "3z".to_string(),
        "N" => "4z".to_string(),
        "P" => "5z".to_string(),
        "F" => "6z".to_string(),
        "C" => "7z".to_string(),
        _ => tile.to_string(),
    }
}

fn is_aka_short_tile(tile: &str) -> bool {
    matches!(tile, "0m" | "0p" | "0s")
}

fn deaka_short_tile(tile: &str) -> String {
    if is_aka_short_tile(tile) {
        format!("5{}", tile.chars().nth(1).unwrap_or('m'))
    } else {
        tile.to_string()
    }
}

fn array_string<'a>(value: &'a Value, key: &str, index: usize) -> &'a str {
    value
        .get(key)
        .and_then(Value::as_array)
        .and_then(|items| items.get(index))
        .and_then(Value::as_str)
        .unwrap_or_default()
}

fn aka_tile_if_needed(tile: &str, tehai_vec34: &[u8; 34], akas_in_hand: &[bool; 3]) -> String {
    match tile {
        "5m" if tehai_vec34[4] == 1 && akas_in_hand[0] => "5mr".to_string(),
        "5p" if tehai_vec34[13] == 1 && akas_in_hand[1] => "5pr".to_string(),
        "5s" if tehai_vec34[22] == 1 && akas_in_hand[2] => "5sr".to_string(),
        _ => tile.to_string(),
    }
}

fn wind_name(tile_id: u8) -> Option<&'static str> {
    match tile_id {
        27 => Some("E"),
        28 => Some("S"),
        29 => Some("W"),
        30 => Some("N"),
        _ => None,
    }
}

fn base_tile(tile: &str) -> &str {
    if tile.len() >= 2 {
        &tile[..2]
    } else {
        tile
    }
}

fn tile_sort_key(tile: &str) -> (usize, usize) {
    let normalized = base_tile(tile);
    let index = TILE_TYPES
        .iter()
        .position(|candidate| *candidate == normalized)
        .unwrap_or(TILE_TYPES.len());
    (index, usize::from(tile.ends_with('r')))
}

fn remove_first_match(tiles: &mut Vec<String>, target: &str) {
    if let Some(index) = tiles.iter().position(|tile| tile == target) {
        tiles.remove(index);
    }
}