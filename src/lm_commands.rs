use tauri::{command, AppHandle, Runtime};
#[cfg(mobile)]
use tauri::Emitter;

use crate::{
    error::Result,
    lm_models::{GenerateInput, GenerateOutput, LmModelInfo, LoadLmModelOptions},
    LiteRtLmExt,
};
#[cfg(mobile)]
use crate::lm_models::StreamChunk;

/// Load a `.litertlm` LLM model.
#[command]
pub fn load_lm_model<R: Runtime>(
    app: AppHandle<R>,
    opts: LoadLmModelOptions,
) -> Result<LmModelInfo> {
    app.litert_lm().load_lm_model(opts)
}

/// Release a loaded LLM model.
#[command]
pub fn unload_lm_model<R: Runtime>(app: AppHandle<R>, model_id: String) -> Result<()> {
    app.litert_lm().unload_lm_model(&model_id)
}

/// Return metadata for all loaded LLM models.
#[command]
pub fn list_lm_models<R: Runtime>(app: AppHandle<R>) -> Result<Vec<LmModelInfo>> {
    app.litert_lm().list_lm_models()
}

/// Run a blocking generation and return the full response.
#[command]
pub fn generate<R: Runtime>(app: AppHandle<R>, input: GenerateInput) -> Result<GenerateOutput> {
    app.litert_lm().generate(input)
}

/// Start a streaming generation. Tokens are emitted as `litert-lm://chunk`
/// Tauri events on the app handle. The final event has `done: true`.
#[command]
pub fn generate_stream<R: Runtime>(app: AppHandle<R>, input: GenerateInput) -> Result<()> {
    #[cfg(mobile)]
    {
        // On mobile the Kotlin plugin emits chunks via a Tauri Channel.
        // We create the channel here (where AppHandle is available) so its
        // on_message closure can forward each payload to the global event bus,
        // which is what the JS listen("litert-lm://chunk") handler reads.
        let channel = {
            let app2 = app.clone();
            tauri::ipc::Channel::<serde_json::Value>::new(move |body| {
                if let tauri::ipc::InvokeResponseBody::Json(json) = body {
                    if let Ok(chunk) = serde_json::from_str::<StreamChunk>(&json) {
                        let _ = app2.emit("litert-lm://chunk", chunk);
                    }
                }
                Ok(())
            })
        };
        return app.litert_lm().generate_stream(input, channel);
    }
    #[cfg(not(mobile))]
    app.litert_lm().generate_stream(input)
}
