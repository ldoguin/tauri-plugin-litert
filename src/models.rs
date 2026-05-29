use serde::{Deserialize, Serialize};

/// Hardware accelerator to use for inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Accelerator {
    Cpu,
    Gpu,
    Npu,
}

impl Default for Accelerator {
    fn default() -> Self {
        Self::Cpu
    }
}

/// Options supplied when loading a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadModelOptions {
    /// Filesystem path or asset URI to the `.tflite` model file.
    pub model_path: String,
    /// Human-readable identifier used to reference this model in subsequent calls.
    pub model_id: String,
    #[serde(default)]
    pub accelerator: Accelerator,
}

/// Metadata returned after a model is loaded.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub model_id: String,
    pub model_path: String,
    pub accelerator: Accelerator,
    /// Number of input tensors.
    pub input_count: usize,
    /// Number of output tensors.
    pub output_count: usize,
    /// Shape of each input tensor.
    pub input_shapes: Vec<Vec<i32>>,
    /// Shape of each output tensor.
    pub output_shapes: Vec<Vec<i32>>,
}

/// Input payload for a single inference run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceInput {
    pub model_id: String,
    /// Flat float32 arrays, one per input tensor (in order).
    pub inputs: Vec<Vec<f32>>,
    /// Optional element type per tensor: "float" (default) or "int32".
    /// On Android, "int32" tensors are written via writeInt instead of writeFloat.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_types: Option<Vec<String>>,
}

/// Output of a single inference run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceOutput {
    pub model_id: String,
    /// Flat float32 arrays, one per output tensor (in order).
    pub outputs: Vec<Vec<f32>>,
    /// Wall-clock inference time in milliseconds.
    pub latency_ms: f64,
}

/// Input payload for embedding creation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingInput {
    pub model_id: String,
    /// Raw float32 input (e.g. tokenised text, image pixels).
    pub input: Vec<f32>,
}

/// Embedding vector returned by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingOutput {
    pub model_id: String,
    pub embedding: Vec<f32>,
    pub latency_ms: f64,
}
