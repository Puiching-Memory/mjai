use std::env;
use std::fs;
use std::io::{self, BufRead, Write};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use tract_onnx::prelude::*;

const MASKED_LOGIT: f32 = -1.0e30;

#[derive(Debug, Deserialize)]
struct ModelMetadata {
    input_dim: usize,
    action_dim: usize,
}

#[derive(Debug, Deserialize)]
struct InferenceRequest {
    features: Vec<f32>,
    legal_actions: Vec<bool>,
}

#[derive(Debug, Serialize)]
struct InferenceResponse {
    action: usize,
    logits: Vec<f32>,
    masked_logits: Vec<f32>,
}

fn load_model(model_path: &str) -> Result<SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>> {
    tract_onnx::onnx()
        .model_for_path(model_path)
        .with_context(|| format!("failed to load ONNX model at {model_path}"))?
        .into_optimized()?
        .into_runnable()
        .context("failed to build runnable tract plan")
}

fn read_metadata(path: &str) -> Result<ModelMetadata> {
    let content = fs::read_to_string(path).with_context(|| format!("failed to read metadata at {path}"))?;
    serde_json::from_str(&content).with_context(|| format!("failed to parse metadata JSON at {path}"))
}

fn apply_mask(logits: &[f32], mask: &[bool]) -> Result<(usize, Vec<f32>)> {
    if logits.len() != mask.len() {
        bail!("logit count {} does not match mask count {}", logits.len(), mask.len());
    }

    let mut best_index: Option<usize> = None;
    let mut best_value = f32::NEG_INFINITY;
    let mut masked_logits = Vec::with_capacity(logits.len());
    let mut saw_legal = false;

    for (index, (logit, allowed)) in logits.iter().zip(mask.iter()).enumerate() {
        let masked = if *allowed { *logit } else { MASKED_LOGIT };
        if *allowed {
            saw_legal = true;
            if masked > best_value {
                best_value = masked;
                best_index = Some(index);
            }
        }
        masked_logits.push(masked);
    }

    if !saw_legal {
        bail!("no legal action was provided");
    }

    let action = best_index.ok_or_else(|| anyhow!("no legal action was provided"))?;
    Ok((action, masked_logits))
}

fn main() -> Result<()> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        bail!("usage: mjai-tract-runtime <model.onnx> <model.json>");
    }

    let model_path = &args[1];
    let metadata = read_metadata(&args[2])?;
    let model = load_model(model_path)?;

    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());

    for line in stdin.lock().lines() {
        let line = line.context("failed to read stdin")?;
        if line.trim().is_empty() {
            continue;
        }

        let request: InferenceRequest = serde_json::from_str(&line).context("failed to parse inference request JSON")?;
        if request.features.len() != metadata.input_dim {
            bail!(
                "feature length {} does not match metadata input_dim {}",
                request.features.len(),
                metadata.input_dim
            );
        }
        if request.legal_actions.len() != metadata.action_dim {
            bail!(
                "legal action count {} does not match metadata action_dim {}",
                request.legal_actions.len(),
                metadata.action_dim
            );
        }

        let input = tract_ndarray::Array2::from_shape_vec((1, metadata.input_dim), request.features)
            .context("failed to build tract input tensor")?;
        let input = Tensor::from(input);
        let result = model.run(tvec!(input.into()))?;
        let logits = result[0]
            .to_array_view::<f32>()
            .context("tract returned a non-f32 logits tensor")?
            .iter()
            .copied()
            .collect::<Vec<_>>();

        let (action, masked_logits) = apply_mask(&logits, &request.legal_actions)?;
        let response = InferenceResponse {
            action,
            logits,
            masked_logits,
        };
        serde_json::to_writer(&mut stdout, &response).context("failed to serialize inference response")?;
        stdout.write_all(b"\n").context("failed to write newline")?;
        stdout.flush().context("failed to flush stdout")?;
    }

    Ok(())
}