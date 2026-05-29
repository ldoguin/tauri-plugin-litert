//! Desktop backend — uses the `litert` crate which downloads prebuilt
//! `libLiteRt.{so,dylib,dll}` on first `cargo build`.

use std::{collections::HashMap, sync::Mutex, time::Instant};

use litert::{Accelerators, CompiledModel, CompilationOptions, ElementType, Environment, Model,
             TensorBuffer, TensorShape};
use tauri::{AppHandle, Runtime};

use crate::{
    error::{Error, Result},
    models::{
        Accelerator, EmbeddingInput, EmbeddingOutput, InferenceInput, InferenceOutput,
        LoadModelOptions, ModelInfo,
    },
};

// ---------------------------------------------------------------------------
// Internal model record
// ---------------------------------------------------------------------------

struct LoadedModel {
    info: ModelInfo,
    compiled: CompiledModel,
    /// Separate environment kept for TensorBuffer allocation.
    /// CompiledModel takes ownership of its own env; we keep a second one
    /// here because LiteRT internally reference-counts the hardware context.
    buf_env: Environment,
    /// Per-tensor element types, needed to allocate correctly-typed buffers.
    input_element_types: Vec<ElementType>,
    output_element_types: Vec<ElementType>,
}

// ---------------------------------------------------------------------------
// Plugin state
// ---------------------------------------------------------------------------

pub struct LiteRt<R: Runtime> {
    #[allow(dead_code)]
    app: AppHandle<R>,
    models: Mutex<HashMap<String, LoadedModel>>,
}

impl<R: Runtime> LiteRt<R> {
    pub fn new(app: AppHandle<R>) -> Self {
        Self {
            app,
            models: Mutex::new(HashMap::new()),
        }
    }

    // -----------------------------------------------------------------------
    // load_model
    // -----------------------------------------------------------------------
    pub fn load_model(&self, opts: LoadModelOptions) -> Result<ModelInfo> {
        // Check for duplicates under the lock, then drop it before the blocking
        // I/O and compilation so other commands aren't starved.
        {
            let models = self.models.lock().unwrap();
            if models.contains_key(&opts.model_id) {
                return Err(Error::ModelAlreadyLoaded(opts.model_id));
            }
        }

        let env_for_model = Environment::new()
            .map_err(|e| Error::Backend(format!("Environment::new: {e}")))?;
        let buf_env = Environment::new()
            .map_err(|e| Error::Backend(format!("Environment::new (buf): {e}")))?;

        let model = Model::from_file(&opts.model_path)
            .map_err(|e| Error::Backend(format!("Model::from_file: {e}")))?;

        // Map plugin Accelerator → litert Accelerators bitset.
        // Always include CPU as fallback so ops unsupported by the requested
        // backend still run rather than failing compilation.
        let hw = match opts.accelerator {
            Accelerator::Gpu => Accelerators::GPU | Accelerators::CPU,
            Accelerator::Npu => Accelerators::NPU | Accelerators::CPU,
            Accelerator::Cpu => Accelerators::CPU,
        };

        let compile_opts = CompilationOptions::new()
            .map_err(|e| Error::Backend(format!("CompilationOptions::new: {e}")))?
            .with_accelerators(hw)
            .map_err(|e| Error::Backend(format!("with_accelerators: {e}")))?;

        // Introspect shapes before moving model into CompiledModel.
        let sig = model
            .signature(0)
            .map_err(|e| Error::Backend(format!("signature(0): {e}")))?;

        let input_count = sig
            .input_count()
            .map_err(|e| Error::Backend(e.to_string()))?;
        let output_count = sig
            .output_count()
            .map_err(|e| Error::Backend(e.to_string()))?;

        let mut input_shapes = Vec::with_capacity(input_count);
        let mut input_element_types = Vec::with_capacity(input_count);
        for i in 0..input_count {
            let shape = sig
                .input_shape(i)
                .map_err(|e| Error::Backend(format!("input_shape({i}): {e}")))?;
            input_element_types.push(shape.element_type);
            input_shapes.push(shape.dims.clone());
        }

        let mut output_shapes = Vec::with_capacity(output_count);
        let mut output_element_types = Vec::with_capacity(output_count);
        for i in 0..output_count {
            let shape = sig
                .output_shape(i)
                .map_err(|e| Error::Backend(format!("output_shape({i}): {e}")))?;
            output_element_types.push(shape.element_type);
            output_shapes.push(shape.dims.clone());
        }

        let compiled = CompiledModel::new(env_for_model, model, &compile_opts)
            .map_err(|e| Error::Backend(format!("CompiledModel::new: {e}")))?;

        let info = ModelInfo {
            model_id: opts.model_id.clone(),
            model_path: opts.model_path,
            accelerator: opts.accelerator,
            input_count,
            output_count,
            input_shapes,
            output_shapes,
        };

        // Re-acquire to insert. A duplicate could have been inserted concurrently
        // while the lock was dropped — treat that as an error.
        let mut models = self.models.lock().unwrap();
        if models.contains_key(&info.model_id) {
            return Err(Error::ModelAlreadyLoaded(info.model_id));
        }
        models.insert(
            opts.model_id,
            LoadedModel {
                info: info.clone(),
                compiled,
                buf_env,
                input_element_types,
                output_element_types,
            },
        );
        Ok(info)
    }

    // -----------------------------------------------------------------------
    // unload_model
    // -----------------------------------------------------------------------
    pub fn unload_model(&self, model_id: &str) -> Result<()> {
        let mut models = self.models.lock().unwrap();
        models
            .remove(model_id)
            .ok_or_else(|| Error::ModelNotFound(model_id.to_string()))?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // list_models
    // -----------------------------------------------------------------------
    pub fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let models = self.models.lock().unwrap();
        Ok(models.values().map(|m| m.info.clone()).collect())
    }

    // -----------------------------------------------------------------------
    // get_model_info
    // -----------------------------------------------------------------------
    pub fn get_model_info(&self, model_id: &str) -> Result<ModelInfo> {
        let models = self.models.lock().unwrap();
        models
            .get(model_id)
            .map(|m| m.info.clone())
            .ok_or_else(|| Error::ModelNotFound(model_id.to_string()))
    }

    // -----------------------------------------------------------------------
    // run_inference
    // -----------------------------------------------------------------------
    pub fn run_inference(&self, input: InferenceInput) -> Result<InferenceOutput> {
        // Shared (non-exclusive) borrow: CompiledModel::run takes &self.
        let models = self.models.lock().unwrap();
        let loaded = models
            .get(&input.model_id)
            .ok_or_else(|| Error::ModelNotFound(input.model_id.clone()))?;

        let info = &loaded.info;

        if input.inputs.len() != info.input_count {
            return Err(Error::InvalidInput(format!(
                "expected {} input tensors, got {}",
                info.input_count,
                input.inputs.len()
            )));
        }

        // Allocate input buffers using the actual element type from the model.
        let mut input_buffers: Vec<TensorBuffer> = info
            .input_shapes
            .iter()
            .zip(loaded.input_element_types.iter())
            .zip(input.inputs.iter())
            .map(|((dims, &elem_type), data)| {
                let shape = TensorShape { element_type: elem_type, dims: dims.clone() };
                let expected = shape.num_elements();
                if data.len() != expected {
                    return Err(Error::InvalidInput(format!(
                        "tensor expects {expected} elements, got {}",
                        data.len()
                    )));
                }
                let mut buf = TensorBuffer::managed_host(&loaded.buf_env, &shape)
                    .map_err(|e| Error::Backend(e.to_string()))?;
                write_f32_to_buf(&mut buf, elem_type, data)?;
                Ok(buf)
            })
            .collect::<Result<_>>()?;

        // Allocate output buffers using the actual element type.
        let mut output_buffers: Vec<TensorBuffer> = info
            .output_shapes
            .iter()
            .zip(loaded.output_element_types.iter())
            .map(|(dims, &elem_type)| {
                let shape = TensorShape { element_type: elem_type, dims: dims.clone() };
                TensorBuffer::managed_host(&loaded.buf_env, &shape)
                    .map_err(|e| Error::Backend(e.to_string()))
            })
            .collect::<Result<_>>()?;

        let t0 = Instant::now();
        loaded
            .compiled
            .run(&mut input_buffers, &mut output_buffers)
            .map_err(|e| Error::InferenceFailed(e.to_string()))?;
        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Read outputs — always returned as f32 for a uniform JS API.
        let outputs: Vec<Vec<f32>> = output_buffers
            .iter()
            .zip(loaded.output_element_types.iter())
            .map(|(buf, &elem_type)| read_buf_as_f32(buf, elem_type))
            .collect::<Result<_>>()?;

        Ok(InferenceOutput {
            model_id: input.model_id,
            outputs,
            latency_ms,
        })
    }

    // -----------------------------------------------------------------------
    // create_embedding
    // -----------------------------------------------------------------------
    pub fn create_embedding(&self, input: EmbeddingInput) -> Result<EmbeddingOutput> {
        let result = self.run_inference(InferenceInput {
            model_id: input.model_id.clone(),
            inputs: vec![input.input],
            input_types: None,
        })?;

        let embedding = result
            .outputs
            .into_iter()
            .next()
            .ok_or_else(|| Error::InferenceFailed("model produced no output".into()))?;

        Ok(EmbeddingOutput {
            model_id: input.model_id,
            embedding,
            latency_ms: result.latency_ms,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Write caller-supplied f32 data into a buffer, casting to the tensor's type.
fn write_f32_to_buf(buf: &mut TensorBuffer, elem_type: ElementType, data: &[f32]) -> Result<()> {
    match elem_type {
        ElementType::Float32 => {
            let mut g = buf.lock_for_write::<f32>().map_err(|e| Error::Backend(e.to_string()))?;
            g.copy_from_slice(data);
        }
        ElementType::Int8 => {
            let mut g = buf.lock_for_write::<i8>().map_err(|e| Error::Backend(e.to_string()))?;
            for (dst, &src) in g.iter_mut().zip(data) { *dst = src as i8; }
        }
        ElementType::UInt8 => {
            let mut g = buf.lock_for_write::<u8>().map_err(|e| Error::Backend(e.to_string()))?;
            for (dst, &src) in g.iter_mut().zip(data) { *dst = src as u8; }
        }
        ElementType::Int32 => {
            let mut g = buf.lock_for_write::<i32>().map_err(|e| Error::Backend(e.to_string()))?;
            for (dst, &src) in g.iter_mut().zip(data) { *dst = src as i32; }
        }
        _ => {
            // Best-effort: write as f32 and let LiteRT handle it.
            let mut g = buf.lock_for_write::<f32>().map_err(|e| Error::Backend(e.to_string()))?;
            g.copy_from_slice(data);
        }
    }
    Ok(())
}

/// Read a buffer and return its contents as f32, casting from the tensor's type.
fn read_buf_as_f32(buf: &TensorBuffer, elem_type: ElementType) -> Result<Vec<f32>> {
    match elem_type {
        ElementType::Float32 => buf
            .lock_for_read::<f32>()
            .map(|g| g.to_vec())
            .map_err(|e| Error::Backend(e.to_string())),
        ElementType::Int8 => buf
            .lock_for_read::<i8>()
            .map(|g| g.iter().map(|&v| v as f32).collect())
            .map_err(|e| Error::Backend(e.to_string())),
        ElementType::UInt8 => buf
            .lock_for_read::<u8>()
            .map(|g| g.iter().map(|&v| v as f32).collect())
            .map_err(|e| Error::Backend(e.to_string())),
        ElementType::Int32 => buf
            .lock_for_read::<i32>()
            .map(|g| g.iter().map(|&v| v as f32).collect())
            .map_err(|e| Error::Backend(e.to_string())),
        _ => buf
            .lock_for_read::<f32>()
            .map(|g| g.to_vec())
            .map_err(|e| Error::Backend(e.to_string())),
    }
}
