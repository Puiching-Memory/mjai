use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::Value;

use crate::mjai_bot_state::RustMjaiBotState;

#[derive(Debug, Serialize)]
pub struct Transcript {
    pub fixture: String,
    pub player_id: u8,
    pub steps: Vec<TranscriptStep>,
}

#[derive(Debug, Serialize)]
pub struct TranscriptStep {
    pub step_index: usize,
    pub events: Value,
    pub snapshot: Value,
}

pub fn build_transcript(fixture_path: &Path) -> Result<Transcript> {
    let lines = read_fixture_lines(fixture_path)?;
    let player_id = infer_player_id(&lines)?;
    let mut bot = RustMjaiBotState::new(player_id);
    let mut steps = Vec::with_capacity(lines.len());

    for (step_index, line) in lines.iter().enumerate() {
        steps.push(react_line(&mut bot, step_index, line)?);
    }

    Ok(Transcript {
        fixture: fixture_path.display().to_string(),
        player_id,
        steps,
    })
}

fn react_line(bot: &mut RustMjaiBotState, step_index: usize, line: &str) -> Result<TranscriptStep> {
    let events_value: Value =
        serde_json::from_str(line).with_context(|| format!("invalid fixture JSON at step {step_index}"))?;
    let event_values = events_value
        .as_array()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("fixture line {step_index} must be a JSON array"))?;
    if event_values.is_empty() {
        bail!("fixture line {step_index} cannot be empty");
    }

    let snapshot = bot.react_batch_value(&Value::Array(event_values.clone()))?;

    Ok(TranscriptStep {
        step_index,
        events: Value::Array(event_values),
        snapshot,
    })
}

fn read_fixture_lines(fixture_path: &Path) -> Result<Vec<String>> {
    Ok(fs::read_to_string(fixture_path)
        .with_context(|| format!("failed to read fixture {}", fixture_path.display()))?
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect())
}

fn infer_player_id(lines: &[String]) -> Result<u8> {
    for line in lines {
        let events: Value = serde_json::from_str(line)?;
        let Some(events_array) = events.as_array() else {
            continue;
        };
        for event in events_array {
            if event.get("type").and_then(Value::as_str) != Some("start_kyoku") {
                continue;
            }
            let Some(tehais) = event.get("tehais").and_then(Value::as_array) else {
                continue;
            };
            let visible_players = tehais
                .iter()
                .enumerate()
                .filter_map(|(player_id, tehai)| {
                    let tiles = tehai.as_array()?;
                    tiles.iter().any(|tile| tile.as_str().is_some_and(|value| value != "?"))
                        .then_some(player_id as u8)
                })
                .collect::<Vec<_>>();
            if visible_players.len() == 1 {
                return Ok(visible_players[0]);
            }
        }
    }
    bail!("failed to infer player_id from fixture")
}