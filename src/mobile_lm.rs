use tauri::{
    ipc::Channel,
    plugin::{PluginApi, PluginHandle},
    AppHandle, Runtime,
};

use crate::{
    error::Result,
    lm_models::{GenerateInput, GenerateOutput, LmModelInfo, LoadLmModelOptions},
};

pub fn init<R: Runtime, C: serde::de::DeserializeOwned>(
    _app: &AppHandle<R>,
    api: &PluginApi<R, C>,
) -> Result<LiteRtLm<R>> {
    #[cfg(target_os = "android")]
    let handle = api
        // Both LiteRt (inference) and LiteRtLm (generation) are handled by the
    // merged LiteRtPlugin class. Registering LiteRtLmPlugin separately would
    // overwrite the LiteRt registration in PluginManager's name-keyed map.
    .register_android_plugin("com.plugin.litert", "LiteRtPlugin")
        .map_err(|e| crate::error::Error::Backend(e.to_string()))?;

    #[cfg(target_os = "ios")]
    let handle = api
        .register_ios_plugin(init_plugin_litert_lm)
        .map_err(|e| crate::error::Error::Backend(e.to_string()))?;

    Ok(LiteRtLm(handle))
}

pub struct LiteRtLm<R: Runtime>(PluginHandle<R>);

impl<R: Runtime> LiteRtLm<R> {
    pub fn load_lm_model(&self, opts: LoadLmModelOptions) -> Result<LmModelInfo> {
        self.0
            .run_mobile_plugin("loadLmModel", opts)
            .map_err(|e| crate::error::Error::Backend(e.to_string()))
    }

    pub fn unload_lm_model(&self, model_id: &str) -> Result<()> {
        self.0
            .run_mobile_plugin("unloadLmModel", serde_json::json!({ "modelId": model_id }))
            .map_err(|e| crate::error::Error::Backend(e.to_string()))
    }

    pub fn list_lm_models(&self) -> Result<Vec<LmModelInfo>> {
        #[derive(serde::Deserialize)]
        struct Resp { models: Vec<LmModelInfo> }
        self.0
            .run_mobile_plugin::<Resp>("listLmModels", ())
            .map(|r| r.models)
            .map_err(|e| crate::error::Error::Backend(e.to_string()))
    }

    pub fn generate(&self, input: GenerateInput) -> Result<GenerateOutput> {
        self.0
            .run_mobile_plugin("generate", input)
            .map_err(|e| crate::error::Error::Backend(e.to_string()))
    }

    pub fn generate_stream(&self, input: GenerateInput, channel: Channel<serde_json::Value>) -> Result<()> {
        // Merge GenerateInput fields with the channel so Kotlin receives them together.
        // Channel<T> serializes as "__CHANNEL__:<id>" which Kotlin's ChannelDeserializer
        // understands natively.
        let mut payload = serde_json::to_value(&input)
            .map_err(|e| crate::error::Error::Backend(e.to_string()))?;
        payload["channel"] = serde_json::to_value(&channel)
            .map_err(|e| crate::error::Error::Backend(e.to_string()))?;
        self.0
            .run_mobile_plugin::<()>("generateStream", payload)
            .map_err(|e| crate::error::Error::Backend(e.to_string()))
    }
}

#[cfg(target_os = "ios")]
tauri::ios_plugin_binding!(init_plugin_litert_lm);
