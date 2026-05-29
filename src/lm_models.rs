use serde::{Deserialize, Serialize};

use crate::models::Accelerator;

/// Options for loading a `.litertlm` LLM model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadLmModelOptions {
    /// Filesystem path to the `.litertlm` model file.
    pub model_path: String,
    /// Stable identifier used in subsequent calls.
    pub model_id: String,
    #[serde(default)]
    pub accelerator: Accelerator,
    /// Maximum context window in tokens. `None` = model default.
    pub max_tokens: Option<i32>,
    /// Directory for runtime caches (KV cache, compiled shaders).
    pub cache_dir: Option<String>,
    /// Enable vision (multimodal) backend. Required for models like Gemma 4 E2B/E4B
    /// that support image input. Passed as `visionBackend = Backend.GPU()` on Android.
    #[serde(default)]
    pub vision: bool,
}

/// Metadata returned after a model is loaded.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LmModelInfo {
    pub model_id: String,
    pub model_path: String,
    pub accelerator: Accelerator,
}

/// Input for a single-turn or multi-turn generation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateInput {
    pub model_id: String,
    pub prompt: String,
    #[serde(default)]
    pub sampler: SamplerOptions,
    /// Optional system instruction prepended to the conversation.
    pub system_instruction: Option<String>,
    /// Optional base64-encoded image bytes (no data-URL prefix) for multimodal models.
    pub image: Option<String>,
}

/// Sampling parameters exposed to the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SamplerOptions {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_top_k")]
    pub top_k: i32,
}

impl Default for SamplerOptions {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            top_p: default_top_p(),
            top_k: default_top_k(),
        }
    }
}

fn default_temperature() -> f32 { 0.8 }
fn default_top_p() -> f32 { 0.95 }
fn default_top_k() -> i32 { 40 }

/// Full response from a blocking generation call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateOutput {
    pub model_id: String,
    pub text: String,
    pub latency_ms: f64,
}

/// A single streamed token chunk emitted via Tauri events.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamChunk {
    pub model_id: String,
    /// The token text fragment.
    pub chunk: String,
    /// True on the final (empty) chunk signalling end-of-stream.
    pub done: bool,
    /// Set on the final chunk when generation succeeded.
    pub latency_ms: Option<f64>,
    /// Set on the final chunk when generation failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}
