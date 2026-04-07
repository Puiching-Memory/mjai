use std::io::{self, BufRead, Write};

use anyhow::{Context, Result};
use mjai_tract_runtime::mjai_bot_state::RustMjaiBotState;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Request {
    /// Update state and produce compiled candidates + encoded features in one shot.
    React { events: Value },
    /// Validate a reaction string (unchanged from mjai-bot-state).
    ValidateReaction { reaction: String },
    /// Print brief state info (unchanged from mjai-bot-state).
    BriefInfo,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(flag) = args.next() else {
        anyhow::bail!("usage: mjai-bot-decision --player-id <seat>");
    };
    if flag != "--player-id" {
        anyhow::bail!("usage: mjai-bot-decision --player-id <seat>");
    }
    let Some(player_id) = args.next() else {
        anyhow::bail!("usage: mjai-bot-decision --player-id <seat>");
    };
    if args.next().is_some() {
        anyhow::bail!("usage: mjai-bot-decision --player-id <seat>");
    }
    let player_id: u8 = player_id.parse().context("player id must be an integer in [0, 3]")?;

    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut bot = RustMjaiBotState::new(player_id);

    for line in stdin.lock().lines() {
        let line = line.context("failed to read stdin")?;
        if line.trim().is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<Request>(&line) {
            Ok(Request::React { events }) => match bot.react_and_decide(&events) {
                Ok(result) => result,
                Err(error) => json!({"ok": false, "error": error.to_string()}),
            },
            Ok(Request::ValidateReaction { reaction }) => {
                match bot.validate_reaction_json(&reaction) {
                    Ok(()) => json!({"ok": true}),
                    Err(error) => json!({"ok": false, "error": error.to_string()}),
                }
            }
            Ok(Request::BriefInfo) => json!({"ok": true, "brief_info": bot.brief_info()}),
            Err(error) => json!({"ok": false, "error": error.to_string()}),
        };

        serde_json::to_writer(&mut stdout, &response).context("failed to serialize response")?;
        stdout.write_all(b"\n").context("failed to write newline")?;
        stdout.flush().context("failed to flush stdout")?;
    }

    Ok(())
}
