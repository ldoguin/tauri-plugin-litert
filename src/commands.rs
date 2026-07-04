use tauri::{command, AppHandle, Runtime};

use crate::{
    error::Result,
    models::{
        EmbeddingInput, EmbeddingOutput, InferenceInput, InferenceOutput, LoadModelOptions,
        ModelInfo,
    },
    LiteRtExt,
};

/// Load a `.tflite` model from disk and register it under `model_id`.
#[command]
pub fn load_model<R: Runtime>(app: AppHandle<R>, opts: LoadModelOptions) -> Result<ModelInfo> {
    app.litert().load_model(opts)
}

/// Release a previously loaded model and free its resources.
#[command]
pub fn unload_model<R: Runtime>(app: AppHandle<R>, model_id: String) -> Result<()> {
    app.litert().unload_model(&model_id)
}

/// Return metadata for all currently loaded models.
#[command]
pub fn list_models<R: Runtime>(app: AppHandle<R>) -> Result<Vec<ModelInfo>> {
    app.litert().list_models()
}

/// Return metadata for a single loaded model.
#[command]
pub fn get_model_info<R: Runtime>(app: AppHandle<R>, model_id: String) -> Result<ModelInfo> {
    app.litert().get_model_info(&model_id)
}

/// Run a forward pass through the model with the supplied input tensors.
#[command]
pub fn run_inference<R: Runtime>(app: AppHandle<R>, input: InferenceInput) -> Result<InferenceOutput> {
    app.litert().run_inference(input)
}

/// Run the model and return the first output tensor as an embedding vector.
#[command]
pub fn create_embedding<R: Runtime>(app: AppHandle<R>, input: EmbeddingInput) -> Result<EmbeddingOutput> {
    app.litert().create_embedding(input)
}

#[command]
pub fn tts_speak<R: Runtime>(
    app: AppHandle<R>,
    text: String,
    #[allow(unused_variables)] rate: Option<f32>,
    #[allow(unused_variables)] pitch: Option<f32>,
) -> Result<()> {
    app.litert().tts_speak(text, rate.unwrap_or(1.0), pitch.unwrap_or(1.0))
}

#[command]
pub fn tts_cancel<R: Runtime>(app: AppHandle<R>) -> Result<()> {
    app.litert().tts_cancel()
}

/// On Android: uses NpuCompatibilityChecker to determine the best available
/// accelerator (npu > gpu) at runtime, based on the SDK's embedded SoC lists.
/// On desktop: always returns "gpu" (the standard desktop LLM accelerator).
#[command]
pub fn query_accelerator_support<R: Runtime>(app: AppHandle<R>) -> Result<crate::AcceleratorSupport> {
    app.litert().query_accelerator_support()
}
