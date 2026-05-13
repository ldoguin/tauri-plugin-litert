# tauri-plugin-litert

Tauri plugin for [Google LiteRT](https://ai.google.dev/edge/litert) on-device ML inference.

Supports **desktop** (Linux, macOS, Windows), **Android**, and **web** from a single TypeScript API.

| Platform | Runtime |
|---|---|
| Desktop | [`litert`](https://crates.io/crates/litert) Rust crate — downloads prebuilt `libLiteRt` automatically |
| Android | LiteRT `CompiledModel` Kotlin API (CPU / GPU / NPU) |
| Web | [`@litertjs/core`](https://ai.google.dev/edge/litert/web) (WebAssembly / WebGPU) |

## Features

- **Model management** – load, unload, list, inspect `.tflite` models
- **Inference** – multi-tensor forward pass with latency reporting
- **Embeddings** – convenience wrapper for single-output embedding models
- **LLM generation** – load `.litertlm` models (Gemma, Llama, Phi, Qwen…) and generate text
- **Streaming generation** – token-by-token streaming via Tauri events (`litert-lm://chunk`)
- **Accelerator selection** – `cpu` | `gpu` | `npu` for both inference and LLM backends

## Installation

### Rust (Tauri app)

```toml
# src-tauri/Cargo.toml
[dependencies]
tauri-plugin-litert = { git = "https://github.com/ldoguin/tauri-plugin-litert" }
```

Register the plugin in `src-tauri/src/main.rs`:

```rust
fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_litert::init())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### JavaScript / TypeScript

```sh
npm install tauri-plugin-litert-api
```

### Android

Add the LiteRT Maven dependencies to your app's `build.gradle`:

```kotlin
dependencies {
    implementation("com.google.ai.edge.litert:litert:2.1.0")
    implementation("com.google.ai.edge.litert:litert-gpu:2.1.0")
    // LLM generation (optional — only needed if using loadLmModel / generate)
    implementation("com.google.ai.edge.litertlm:litertlm-android:latest.release")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}
```

Register both plugins in your `MainActivity.kt`:

```kotlin
class MainActivity : TauriActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        registerPlugin(LiteRtPlugin::class.java)
        registerPlugin(LiteRtLmPlugin::class.java)
        super.onCreate(savedInstanceState)
    }
}
```

For GPU acceleration, add the following to your app's `AndroidManifest.xml` inside the `<application>` tag:

```xml
<application>
    <uses-native-library android:name="libvndksupport.so" android:required="false"/>
    <uses-native-library android:name="libOpenCL.so" android:required="false"/>
</application>
```

### Permissions

Add to your app's capability file (e.g. `src-tauri/capabilities/default.json`):

```json
{
  "permissions": [
    "litert:default"
  ]
}
```

Or grant individual permissions:

```json
{
  "permissions": [
    "litert:allow-load-model",
    "litert:allow-run-inference",
    "litert:allow-create-embedding"
  ]
}
```

## Usage

### Load a model

```typescript
import { loadModel } from "tauri-plugin-litert-api";

const info = await loadModel({
  modelId: "mobilenet",
  modelPath: "/path/to/mobilenet_v2.tflite", // URL on web
  accelerator: "cpu",                         // "cpu" | "gpu" | "npu"
});

console.log(info.inputShapes);  // e.g. [[1, 224, 224, 3]]
console.log(info.outputShapes); // e.g. [[1, 1001]]
```

### Run inference

```typescript
import { runInference } from "tauri-plugin-litert-api";

// Flat Float32 array – one entry per input tensor
const pixels = new Array(1 * 224 * 224 * 3).fill(0.5);

const result = await runInference({
  modelId: "mobilenet",
  inputs: [pixels],
});

console.log(result.outputs[0]); // [0.002, 0.001, …] – 1001 class scores
console.log(`Inference took ${result.latencyMs.toFixed(1)} ms`);
```

### Create an embedding

```typescript
import { createEmbedding } from "tauri-plugin-litert-api";

const tokens = [101, 2023, 2003, 1037, 3231, 102]; // tokenised input

const { embedding, latencyMs } = await createEmbedding({
  modelId: "embedding-model",
  input: tokens.map(Number),
});

console.log(`Embedding dim: ${embedding.length}, latency: ${latencyMs.toFixed(1)} ms`);
```

### List / inspect loaded models

```typescript
import { listModels, getModelInfo, unloadModel } from "tauri-plugin-litert-api";

const models = await listModels();
models.forEach(m => console.log(m.modelId, m.accelerator));

const info = await getModelInfo("mobilenet");
console.log(info);

await unloadModel("mobilenet");
```

### Web-only setup

> **Required headers** — the `@litertjs/core` WebAssembly backend uses `SharedArrayBuffer`,
> which requires cross-origin isolation. Your server (or dev server) must send:
> ```
> Cross-Origin-Opener-Policy: same-origin
> Cross-Origin-Embedder-Policy: require-corp
> ```
> With Vite, add these under `server.headers` in `vite.config.ts`. For production,
> configure them in your CDN or reverse proxy.

On web the plugin uses [`@litertjs/core`](https://ai.google.dev/edge/litert/web) —
Google's official LiteRT.js runtime. Install it alongside the plugin bindings:

```sh
npm install @litertjs/core
```

The Wasm files are loaded from jsDelivr by default. To self-host them, copy
`node_modules/@litertjs/core/wasm/` to your public directory and call
`setWasmPath` before the first `loadModel`:

```typescript
import { setWasmPath } from "tauri-plugin-litert-api";
setWasmPath("/wasm/");
```

## API Reference

### Inference / embedding

| Function | Description |
|---|---|
| `loadModel(opts)` | Load a `.tflite` model; returns `ModelInfo` |
| `unloadModel(modelId)` | Release a loaded model |
| `listModels()` | Return `ModelInfo[]` for all loaded models |
| `getModelInfo(modelId)` | Return `ModelInfo` for one model |
| `runInference(input)` | Forward pass; returns `InferenceOutput` |
| `createEmbedding(input)` | Embedding convenience wrapper; returns `EmbeddingOutput` |

### LLM generation (LiteRT-LM)

| Function | Description |
|---|---|
| `loadLmModel(opts)` | Load a `.litertlm` LLM; returns `LmModelInfo` |
| `unloadLmModel(modelId)` | Release a loaded LLM |
| `listLmModels()` | Return `LmModelInfo[]` for all loaded LLMs |
| `generate(input)` | Blocking generation; returns `GenerateOutput` |
| `generateStream(input)` | Streaming generation; emits `litert-lm://chunk` events |

#### Streaming example

```typescript
import { loadLmModel, generateStream } from "tauri-plugin-litert-api";
import { listen } from "@tauri-apps/api/core";

await loadLmModel({
  modelId: "gemma",
  modelPath: "/path/to/gemma4.litertlm",
  accelerator: "gpu",
});

// Register the listener BEFORE calling generateStream to avoid missing early chunks.
// The final event always has done: true; check error to distinguish success from failure.
const unlistenRef: { fn: (() => void) | null } = { fn: null };
unlistenRef.fn = await listen("litert-lm://chunk", (event) => {
  const { chunk, done, latencyMs, error } = event.payload;
  if (done) {
    unlistenRef.fn?.();
    if (error) console.error("generation failed:", error);
    else console.log(`Done in ${latencyMs}ms`);
  } else {
    process.stdout.write(chunk);
  }
});

await generateStream({
  modelId: "gemma",
  prompt: "Explain RAG in one paragraph.",
  sampler: { temperature: 0.8, topP: 0.95, topK: 40 },
});
```

### Types

```typescript
type Accelerator = "cpu" | "gpu" | "npu";

interface LoadModelOptions {
  modelPath: string;   // path (desktop/Android) or URL (web)
  modelId: string;
  accelerator?: Accelerator;
}

interface ModelInfo {
  modelId: string;
  modelPath: string;
  accelerator: Accelerator;
  inputCount: number;
  outputCount: number;
  inputShapes: number[][];
  outputShapes: number[][];
}

interface InferenceInput {
  modelId: string;
  inputs: number[][];  // one flat array per input tensor
}

interface InferenceOutput {
  modelId: string;
  outputs: number[][];
  latencyMs: number;
}

interface EmbeddingInput {
  modelId: string;
  input: number[];
}

interface EmbeddingOutput {
  modelId: string;
  embedding: number[];
  latencyMs: number;
}
```

#### LiteRT-LM types

```typescript
interface LoadLmModelOptions {
  modelPath: string;   // path to .litertlm file
  modelId: string;
  accelerator?: Accelerator;
  maxTokens?: number;
  cacheDir?: string;
}

interface LmModelInfo {
  modelId: string;
  modelPath: string;
  accelerator: Accelerator;
}

interface GenerateInput {
  modelId: string;
  prompt: string;
  systemInstruction?: string;
  sampler?: {
    temperature?: number;  // default 0.8
    topP?: number;         // default 0.95
    topK?: number;         // default 40
  };
}

interface GenerateOutput {
  modelId: string;
  text: string;
  latencyMs: number;
}

// Emitted as "litert-lm://chunk" Tauri events during generateStream()
interface StreamChunk {
  modelId: string;
  chunk: string;
  done: boolean;
  latencyMs?: number;  // set on the final chunk when generation succeeded
  error?: string;      // set on the final chunk when generation failed
}
```

## Model sources

**Inference / embedding (`.tflite`)**
- [Kaggle Models — TFLite](https://www.kaggle.com/models?framework=tfLite)
- [Hugging Face — TFLite](https://huggingface.co/models?library=tflite)
- [MediaPipe Models](https://developers.google.com/mediapipe/solutions/guide)

**LLM generation (`.litertlm`)**
- [LiteRT community on Hugging Face](https://huggingface.co/litert-community) — Gemma, Llama, Phi, Qwen, SmolLM
- Models are downloaded and converted using the [LiteRT-LM CLI](https://ai.google.dev/edge/litert-lm/convert)

> **Platform support**: LiteRT-LM runs on **desktop** (Linux, macOS, Windows) and **Android**.
> There is no web runtime for LiteRT-LM yet. The web-chat example uses a three-tier
> fallback for LLM generation on web:
> 1. **MediaPipe GenAI** (`@mediapipe/tasks-genai`) — on-device inference via WebAssembly/WebGPU
> 2. **OpenAI-compatible API** — any endpoint (Groq, OpenRouter, Ollama) configured via the toolbar
> 3. **Mock** — echoes the prompt back, used when no other backend is available

## License

Apache-2.0
