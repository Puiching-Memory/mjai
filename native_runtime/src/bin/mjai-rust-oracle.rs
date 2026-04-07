use std::path::PathBuf;

use anyhow::Result;
use mjai_tract_runtime::mjai_oracle_adapter::build_transcript;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(flag) = args.next() else {
        anyhow::bail!("usage: mjai-rust-oracle --fixture <path>");
    };
    if flag != "--fixture" {
        anyhow::bail!("usage: mjai-rust-oracle --fixture <path>");
    }
    let Some(path) = args.next() else {
        anyhow::bail!("usage: mjai-rust-oracle --fixture <path>");
    };
    if args.next().is_some() {
        anyhow::bail!("usage: mjai-rust-oracle --fixture <path>");
    }

    let transcript = build_transcript(&PathBuf::from(path))?;
    println!("{}", serde_json::to_string(&transcript)?);
    Ok(())
}