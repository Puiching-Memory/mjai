use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

fn contains_error_marker(value: &Value) -> bool {
    match value {
        Value::Object(map) => {
            map.contains_key("__error__") || map.values().any(contains_error_marker)
        }
        Value::Array(values) => values.iter().any(contains_error_marker),
        _ => false,
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("native_runtime should live directly under repo root")
        .to_path_buf()
}

fn python_bin() -> PathBuf {
    if let Ok(value) = env::var("MJAI_PYTHON_BIN") {
        return PathBuf::from(value);
    }
    repo_root().join(".venv/bin/python")
}

fn fixture_paths() -> [PathBuf; 2] {
    [
        repo_root().join("tools/fixtures/competition_call_choice.jsonl"),
        repo_root().join("tools/fixtures/competition_riichi_discard.jsonl"),
    ]
}

fn run_adapter(binary: &Path, fixture: &Path) -> Value {
    let output = Command::new(binary)
        .arg("--fixture")
        .arg(fixture)
        .output()
        .unwrap_or_else(|error| panic!("failed to start adapter {}: {error}", binary.display()));

    assert!(
        output.status.success(),
        "adapter {} failed for {}: {}",
        binary.display(),
        fixture.display(),
        String::from_utf8_lossy(&output.stderr)
    );

    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "adapter {} returned invalid JSON for {}: {error}",
            binary.display(),
            fixture.display()
        )
    })
}

fn run_python_oracle(fixture: &Path) -> Value {
    let python = python_bin();
    let script = repo_root().join("tools/mjai_oracle.py");
    let output = Command::new(&python)
        .arg(&script)
        .arg("--fixture")
        .arg(fixture)
        .output()
        .unwrap_or_else(|error| {
            panic!(
                "failed to start Python oracle {} via {}: {error}",
                script.display(),
                python.display()
            )
        });

    assert!(
        output.status.success(),
        "Python oracle failed for {}: {}",
        fixture.display(),
        String::from_utf8_lossy(&output.stderr)
    );

    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "Python oracle returned invalid JSON for {}: {error}",
            fixture.display()
        )
    })
}

fn rust_adapter_bin() -> Option<PathBuf> {
    std::env::var_os("MJAI_RUST_ORACLE_BIN")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("CARGO_BIN_EXE_mjai-rust-oracle").map(PathBuf::from))
}

fn transcript_steps(transcript: &Value) -> &[Value] {
    transcript
        .get("steps")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .expect("transcript must contain a steps array")
}

#[test]
fn python_oracle_transcripts_are_well_formed() {
    for fixture in fixture_paths() {
        let transcript = run_python_oracle(&fixture);
        assert_eq!(
            transcript.get("fixture").and_then(Value::as_str),
            Some(fixture.to_string_lossy().as_ref())
        );

        let steps = transcript_steps(&transcript);
        assert!(!steps.is_empty(), "fixture {} produced no steps", fixture.display());

        let saw_decision_phase = steps.iter().any(|step| {
            step.get("snapshot")
                .and_then(|snapshot| snapshot.get("state"))
                .and_then(|state| state.get("phase"))
                .and_then(Value::as_str)
                .is_some_and(|phase| phase != "idle")
        });
        assert!(
            saw_decision_phase,
            "fixture {} never reached a decision phase",
            fixture.display()
        );

        for step in steps {
            let snapshot = step
                .get("snapshot")
                .expect("transcript step must contain snapshot");
            assert!(snapshot.get("capabilities").is_some());
            assert!(snapshot.get("state").is_some());
            assert!(snapshot.get("queries").is_some());
            assert!(
                !contains_error_marker(snapshot),
                "fixture {} produced error markers in snapshot: {}",
                fixture.display(),
                snapshot
            );
        }
    }
}

#[test]
fn rust_adapter_matches_python_oracle_when_configured() {
    let Some(adapter_bin) = rust_adapter_bin() else {
        eprintln!("Rust oracle adapter binary is not available; skipping Rust/Python differential run");
        return;
    };

    for fixture in fixture_paths() {
        let python_transcript = run_python_oracle(&fixture);
        let rust_transcript = run_adapter(&adapter_bin, &fixture);
        assert_eq!(
            rust_transcript, python_transcript,
            "Rust adapter diverged from Python oracle for {}",
            fixture.display()
        );
    }
}