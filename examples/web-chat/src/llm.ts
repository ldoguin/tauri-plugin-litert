/**
 * LLM generation backend with three tiers:
 *
 * 1. **Tauri IPC** — when running inside a Tauri app, delegates to the
 *    Rust/Kotlin plugin which uses LiteRT-LM (Gemma, Llama, Phi…).
 *
 * 2. **MediaPipe LLM Inference** (`@mediapipe/tasks-genai`) — on-device LLM
 *    in the browser via WebGPU/Wasm. Loads a `.litertlm` or `.task` model
 *    (e.g. Gemma3-1B-IT web variant). Configure via `loadWebLlm()`.
 *
 * 3. **OpenAI-compatible API** — any remote endpoint (Groq, OpenRouter,
 *    Ollama). Configure via `setApiConfig()`.
 *
 * 4. **Mock** — word-by-word echo used when nothing else is configured.
 */

import type { LlmInference } from "@mediapipe/tasks-genai";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface LmModelInfo {
  modelId: string;
  modelPath: string;
  accelerator: string;
}

export interface GenerateOptions {
  modelId?: string;
  systemInstruction?: string;
  temperature?: number;
  topP?: number;
  topK?: number;
}

export type LlmBackend = "tauri" | "mediapipe" | "api" | "mock";

export interface ApiConfig {
  baseUrl: string;
  apiKey?: string;
  model: string;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let apiConfig: ApiConfig | null = null;
let activeLmModelId: string | null = null;
let webLlm: LlmInference | null = null;
let webLlmLoading = false;

export function setApiConfig(config: ApiConfig): void {
  apiConfig = config;
}

export function getApiConfig(): ApiConfig | null {
  return apiConfig;
}

export function setActiveLmModel(modelId: string | null): void {
  activeLmModelId = modelId;
}

export function getActiveLmModel(): string | null {
  return activeLmModelId;
}

export function getWebLlm(): LlmInference | null {
  return webLlm;
}

export function isWebLlmLoading(): boolean {
  return webLlmLoading;
}

function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

export function getActiveBackend(): LlmBackend {
  if (isTauri() && activeLmModelId) return "tauri";
  if (webLlm) return "mediapipe";
  if (apiConfig) return "api";
  return "mock";
}

// ---------------------------------------------------------------------------
// MediaPipe LLM Inference loader
// ---------------------------------------------------------------------------

export interface WebLlmOptions {
  /** URL to the .litertlm or .task model file. */
  modelUrl: string;
  maxTokens?: number;
  topK?: number;
  temperature?: number;
  /** Override the MediaPipe Wasm CDN path. */
  wasmPath?: string;
}

/**
 * Load a `.litertlm` / `.task` model for in-browser LLM inference via
 * `@mediapipe/tasks-genai` (WebGPU / Wasm).
 *
 * Recommended model: Gemma3-1B-IT web variant (~700 MB, 133 tok/s on WebGPU)
 * https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4-web.task
 */
export async function loadWebLlm(opts: WebLlmOptions): Promise<void> {
  if (webLlmLoading) throw new Error("Already loading a web LLM model");
  webLlmLoading = true;

  try {
    const { FilesetResolver, LlmInference } = await import("@mediapipe/tasks-genai");

    const wasmPath =
      opts.wasmPath ??
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai/wasm";

    const genai = await FilesetResolver.forGenAiTasks(wasmPath);

    // Prefer WebGPU; fall back to Wasm CPU if the adapter is unavailable.
    const gpuAvailable =
      typeof navigator !== "undefined" &&
      "gpu" in navigator &&
      (await (navigator as any).gpu.requestAdapter()) !== null;

    webLlm = await LlmInference.createFromOptions(genai, {
      baseOptions: { modelAssetPath: opts.modelUrl },
      maxTokens: opts.maxTokens ?? 1024,
      topK: opts.topK ?? 40,
      temperature: opts.temperature ?? 0.8,
      delegate: gpuAvailable ? "GPU" : "CPU",
    });
  } finally {
    webLlmLoading = false;
  }
}

export function unloadWebLlm(): void {
  webLlm?.close();
  webLlm = null;
}

// ---------------------------------------------------------------------------
// Streaming generation — public entry point
// ---------------------------------------------------------------------------

export interface StreamCallbacks {
  onChunk: (text: string) => void;
  onDone: (latencyMs: number) => void;
  onError: (err: string) => void;
}

export async function generateStream(
  history: Array<{ role: string; content: string }>,
  ragContext: string,
  opts: GenerateOptions,
  callbacks: StreamCallbacks
): Promise<void> {
  const backend = getActiveBackend();
  const messages = buildMessages(history, ragContext, opts.systemInstruction);

  if (backend === "tauri") return generateViaTauri(messages, opts, callbacks);
  if (backend === "mediapipe") return generateViaMediaPipe(messages, callbacks);
  if (backend === "api") return generateViaApi(messages, opts, callbacks);
  return generateMock(messages, callbacks);
}

// ---------------------------------------------------------------------------
// Tauri IPC backend
// ---------------------------------------------------------------------------

async function generateViaTauri(
  messages: Array<{ role: string; content: string }>,
  opts: GenerateOptions,
  callbacks: StreamCallbacks
): Promise<void> {
  // Use Function() to escape Vite's static import analysis.
  // @tauri-apps/api is injected by the Tauri webview at runtime and is
  // never present in a plain browser — this code only runs when isTauri().
  const tauriImport = new Function("specifier", "return import(specifier)");
  const { invoke, listen } = await tauriImport("@tauri-apps/api/core");

  // Use a holder so the callback can reference unlisten before it's assigned.
  const unlistenHolder: { fn: (() => void) | null } = { fn: null };

  unlistenHolder.fn = await (listen as Function)(
    "litert-lm://chunk",
    (event: { payload: { chunk: string; done: boolean; latencyMs?: number; error?: string } }) => {
      const { chunk, done, latencyMs, error } = event.payload;
      if (done) {
        // Always unlisten on the final event, then report error or completion.
        unlistenHolder.fn?.();
        if (error) callbacks.onError(error);
        else callbacks.onDone(latencyMs ?? 0);
        return;
      }
      if (error) { unlistenHolder.fn?.(); callbacks.onError(error); return; }
      callbacks.onChunk(chunk);
    }
  );

  try {
    const prompt = messages
      .filter((m) => m.role !== "system")
      .map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`)
      .join("\n") + "\nAssistant:";

    const systemInstruction = messages.find((m) => m.role === "system")?.content;

    await invoke("plugin:litert|generate_stream", {
      input: {
        modelId: opts.modelId ?? activeLmModelId,
        prompt,
        systemInstruction,
        sampler: {
          temperature: opts.temperature ?? 0.8,
          topP: opts.topP ?? 0.95,
          topK: opts.topK ?? 40,
        },
      },
    });
  } catch (e) {
    unlistenHolder.fn?.();
    throw e;
  }
}

// ---------------------------------------------------------------------------
// MediaPipe LLM Inference backend (WebGPU / Wasm)
// ---------------------------------------------------------------------------

async function generateViaMediaPipe(
  messages: Array<{ role: string; content: string }>,
  callbacks: StreamCallbacks
): Promise<void> {
  if (!webLlm) { callbacks.onError("No web LLM loaded"); return; }

  // Format history as a simple turn-based prompt.
  // MediaPipe handles the model's chat template internally.
  const prompt = messages
    .map((m) => {
      if (m.role === "system") return `<start_of_turn>system\n${m.content}<end_of_turn>`;
      if (m.role === "user") return `<start_of_turn>user\n${m.content}<end_of_turn>`;
      return `<start_of_turn>model\n${m.content}<end_of_turn>`;
    })
    .join("\n") + "\n<start_of_turn>model\n";

  const t0 = performance.now();

  await webLlm.generateResponse(
    prompt,
    (partialResult: string, done: boolean) => {
      if (done) {
        callbacks.onDone(performance.now() - t0);
      } else {
        callbacks.onChunk(partialResult);
      }
    }
  );
}

// ---------------------------------------------------------------------------
// OpenAI-compatible API backend
// ---------------------------------------------------------------------------

async function generateViaApi(
  messages: Array<{ role: string; content: string }>,
  opts: GenerateOptions,
  callbacks: StreamCallbacks
): Promise<void> {
  if (!apiConfig) { callbacks.onError("No API config set"); return; }

  const t0 = performance.now();

  const response = await fetch(`${apiConfig.baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(apiConfig.apiKey ? { Authorization: `Bearer ${apiConfig.apiKey}` } : {}),
    },
    body: JSON.stringify({
      model: apiConfig.model,
      messages,
      stream: true,
      temperature: opts.temperature ?? 0.8,
      top_p: opts.topP ?? 0.95,
    }),
  });

  if (!response.ok) {
    callbacks.onError(`API error ${response.status}: ${await response.text()}`);
    return;
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const line of decoder.decode(value).split("\n")) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6).trim();
      if (data === "[DONE]") { callbacks.onDone(performance.now() - t0); return; }
      try {
        const chunk = JSON.parse(data).choices?.[0]?.delta?.content;
        if (chunk) callbacks.onChunk(chunk);
      } catch { /* ignore malformed SSE */ }
    }
  }

  callbacks.onDone(performance.now() - t0);
}

// ---------------------------------------------------------------------------
// Mock backend
// ---------------------------------------------------------------------------

async function generateMock(
  messages: Array<{ role: string; content: string }>,
  callbacks: StreamCallbacks
): Promise<void> {
  const lastUser = [...messages].reverse().find((m) => m.role === "user");
  const hasRag = messages.some((m) => m.role === "system" && m.content.includes("context"));

  const reply =
    (hasRag ? "*(RAG context injected)*\n\n" : "") +
    `You said: "${lastUser?.content ?? ""}"\n\n` +
    `Configure an LLM backend to get real responses:\n` +
    `• **On-device (web)**: click "Load LLM" and enter a .litertlm / .task URL\n` +
    `  e.g. https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4-web.task\n` +
    `• **Tauri app**: load a LiteRT-LM model via loadLmModel()\n` +
    `• **API**: click "API config" and enter a Groq / OpenRouter / Ollama endpoint`;

  const t0 = performance.now();
  for (const word of reply.split(" ")) {
    await new Promise((r) => setTimeout(r, 25 + Math.random() * 35));
    callbacks.onChunk(word + " ");
  }
  callbacks.onDone(performance.now() - t0);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function buildMessages(
  history: Array<{ role: string; content: string }>,
  ragContext: string,
  systemInstruction?: string
): Array<{ role: string; content: string }> {
  const messages: Array<{ role: string; content: string }> = [];
  const systemParts: string[] = [];
  if (systemInstruction) systemParts.push(systemInstruction);
  if (ragContext) systemParts.push(ragContext);
  if (systemParts.length > 0) {
    messages.push({ role: "system", content: systemParts.join("\n\n") });
  }
  messages.push(...history);
  return messages;
}
