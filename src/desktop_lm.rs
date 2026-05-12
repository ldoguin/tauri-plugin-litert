//! Desktop LiteRT-LM backend — uses the `litertlm` crate which downloads
//! prebuilt `libLiteRtLm.{so,dylib,dll}` on first `cargo build`.

use std::{collections::HashMap, sync::{Arc, Mutex}, time::Instant};

use litertlm::{Backend, Conversation, Engine, EngineSettings, SamplerParams};
use tauri::{AppHandle, Emitter, Runtime};

use crate::{
    error::{Error, Result},
    lm_models::{
        GenerateInput, GenerateOutput, LmModelInfo, LoadLmModelOptions, SamplerOptions,
        StreamChunk,
    },
    models::Accelerator,
};

// ---------------------------------------------------------------------------
// Internal record — Engine is wrapped in Arc so it can be cloned out of the
// Mutex before the blocking decode loop runs.
// ---------------------------------------------------------------------------

struct LoadedLmModel {
    info: LmModelInfo,
    engine: Arc<Engine>,
}

// ---------------------------------------------------------------------------
// Plugin state
// ---------------------------------------------------------------------------

pub struct LiteRtLm<R: Runtime> {
    #[allow(dead_code)]
    app: AppHandle<R>,
    models: Mutex<HashMap<String, LoadedLmModel>>,
}

impl<R: Runtime> LiteRtLm<R> {
    pub fn new(app: AppHandle<R>) -> Self {
        Self {
            app,
            models: Mutex::new(HashMap::new()),
        }
    }

    // -----------------------------------------------------------------------
    // load_lm_model
    // -----------------------------------------------------------------------
    pub fn load_lm_model(&self, opts: LoadLmModelOptions) -> Result<LmModelInfo> {
        // Check for duplicates under the lock, then drop it before the blocking
        // Engine::new() call so other commands aren't starved during model loading.
        {
            let models = self.models.lock().unwrap();
            if models.contains_key(&opts.model_id) {
                return Err(Error::ModelAlreadyLoaded(opts.model_id));
            }
        }

        let backend = accel_to_backend(&opts.accelerator);
        let mut settings = EngineSettings::new(&opts.model_path).backend(backend);
        if let Some(n) = opts.max_tokens {
            settings = settings.max_num_tokens(n);
        }
        if let Some(ref dir) = opts.cache_dir {
            settings = settings.cache_dir(dir);
        }

        let engine = Engine::new(settings)
            .map_err(|e| Error::Backend(format!("Engine::new: {e}")))?;

        let info = LmModelInfo {
            model_id: opts.model_id.clone(),
            model_path: opts.model_path,
            accelerator: opts.accelerator,
        };

        // Re-acquire to insert. A duplicate could have been inserted concurrently
        // while the lock was dropped — treat that as an error.
        let mut models = self.models.lock().unwrap();
        if models.contains_key(&info.model_id) {
            return Err(Error::ModelAlreadyLoaded(info.model_id));
        }
        models.insert(opts.model_id, LoadedLmModel { info: info.clone(), engine: Arc::new(engine) });
        Ok(info)
    }

    // -----------------------------------------------------------------------
    // unload_lm_model
    // -----------------------------------------------------------------------
    pub fn unload_lm_model(&self, model_id: &str) -> Result<()> {
        let mut models = self.models.lock().unwrap();
        models
            .remove(model_id)
            .ok_or_else(|| Error::ModelNotFound(model_id.to_string()))?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // list_lm_models
    // -----------------------------------------------------------------------
    pub fn list_lm_models(&self) -> Result<Vec<LmModelInfo>> {
        let models = self.models.lock().unwrap();
        Ok(models.values().map(|m| m.info.clone()).collect())
    }

    // -----------------------------------------------------------------------
    // generate (blocking)
    // Clone Arc<Engine> out of the Mutex before the blocking decode so the
    // lock is not held for the entire generation.
    // -----------------------------------------------------------------------
    pub fn generate(&self, input: GenerateInput) -> Result<GenerateOutput> {
        let (engine, params, prompt) = {
            let models = self.models.lock().unwrap();
            let loaded = models
                .get(&input.model_id)
                .ok_or_else(|| Error::ModelNotFound(input.model_id.clone()))?;
            (Arc::clone(&loaded.engine), sampler_params(&input.sampler), build_prompt(&input))
        }; // mutex released here

        let mut conv = engine
            .create_conversation(params)
            .map_err(|e| Error::Backend(format!("create_conversation: {e}")))?;

        let t0 = Instant::now();
        let text = conv
            .send_message(&prompt)
            .map_err(|e| Error::InferenceFailed(e.to_string()))?;
        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

        Ok(GenerateOutput {
            model_id: input.model_id,
            text,
            latency_ms,
        })
    }

    // -----------------------------------------------------------------------
    // generate_stream
    // Spawns a background thread so the Tauri command thread is not blocked
    // for the duration of generation. Returns immediately; tokens arrive as
    // `litert-lm://chunk` events. The final event always has `done: true`
    // (whether generation succeeded or failed).
    // -----------------------------------------------------------------------
    pub fn generate_stream(&self, input: GenerateInput) -> Result<()> {
        let (engine, params, prompt) = {
            let models = self.models.lock().unwrap();
            let loaded = models
                .get(&input.model_id)
                .ok_or_else(|| Error::ModelNotFound(input.model_id.clone()))?;
            (Arc::clone(&loaded.engine), sampler_params(&input.sampler), build_prompt(&input))
        }; // mutex released here

        let model_id = input.model_id.clone();
        let app = self.app.clone();

        std::thread::spawn(move || {
            let conv_result = engine
                .create_conversation(params)
                .map_err(|e| Error::Backend(format!("create_conversation: {e}")));

            let mut conv: Conversation = match conv_result {
                Ok(c) => c,
                Err(e) => {
                    let _ = app.emit(
                        "litert-lm://chunk",
                        StreamChunk {
                            model_id,
                            chunk: String::new(),
                            done: true,
                            latency_ms: None,
                            error: Some(e.to_string()),
                        },
                    );
                    return;
                }
            };

            let t0 = Instant::now();
            let stream_result = conv.send_message_stream(&prompt, |chunk| {
                let _ = app.emit(
                    "litert-lm://chunk",
                    StreamChunk {
                        model_id: model_id.clone(),
                        chunk: chunk.to_string(),
                        done: false,
                        latency_ms: None,
                        error: None,
                    },
                );
            });

            let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let _ = app.emit(
                "litert-lm://chunk",
                StreamChunk {
                    model_id,
                    chunk: String::new(),
                    done: true,
                    latency_ms: Some(latency_ms),
                    error: stream_result
                        .err()
                        .map(|e| Error::InferenceFailed(e.to_string()).to_string()),
                },
            );
        });

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn accel_to_backend(a: &Accelerator) -> Backend {
    match a {
        Accelerator::Gpu => Backend::Gpu,
        Accelerator::Npu => Backend::Npu,
        Accelerator::Cpu => Backend::Cpu,
    }
}

fn sampler_params(opts: &SamplerOptions) -> SamplerParams {
    SamplerParams::default()
        .temperature(opts.temperature)
        .top_p(opts.top_p)
        .top_k(opts.top_k)
}

fn build_prompt(input: &GenerateInput) -> String {
    match &input.system_instruction {
        Some(sys) => format!("[System: {}]\n\n{}", sys, input.prompt),
        None => input.prompt.clone(),
    }
}
