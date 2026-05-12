/**
 * Embedding engine with two backends:
 *
 * 1. **LiteRT** (`@litertjs/core`) — primary. Uses a `.tflite` embedding model
 *    loaded from a URL you supply via `initLiteRtEmbeddings()`.
 *
 * 2. **TensorFlow.js Universal Sentence Encoder** — automatic fallback when
 *    no LiteRT model URL is provided, or when LiteRT fails to load.
 *
 * Both backends expose the same `embed(text): Promise<number[]>` surface.
 */

// ---------------------------------------------------------------------------
// Cosine similarity (used by the RAG retriever)
// ---------------------------------------------------------------------------

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

// ---------------------------------------------------------------------------
// Backend types
// ---------------------------------------------------------------------------

type Backend = "litert" | "use" | "none";

let activeBackend: Backend = "none";
let liteRtModel: import("@litertjs/core").CompiledModel | null = null;
let useModel: { embed(sentences: string[]): Promise<{ arraySync(): number[][] }> } | null = null;

// Simple whitespace tokeniser — produces a fixed-length bag-of-words vector
// used only when both ML backends are unavailable (offline / blocked CDN).
function bowEmbed(text: string, dim = 512): number[] {
  const vec = new Float32Array(dim);
  const words = text.toLowerCase().split(/\s+/);
  for (const w of words) {
    let h = 5381;
    for (let i = 0; i < w.length; i++) h = ((h << 5) + h) ^ w.charCodeAt(i);
    vec[Math.abs(h) % dim] += 1;
  }
  // L2-normalise
  let norm = 0;
  for (const v of vec) norm += v * v;
  norm = Math.sqrt(norm);
  return norm === 0 ? Array.from(vec) : Array.from(vec).map((v) => v / norm);
}

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

export type EmbeddingStatus =
  | { backend: "litert"; modelUrl: string }
  | { backend: "use" }
  | { backend: "bow"; reason: string };

let initPromise: Promise<EmbeddingStatus> | null = null;

/**
 * Initialise the embedding engine.
 *
 * @param liteRtModelUrl  Optional URL to a `.tflite` embedding model.
 *                        When omitted the TF.js Universal Sentence Encoder
 *                        is used instead.
 */
export function initEmbeddings(liteRtModelUrl?: string): Promise<EmbeddingStatus> {
  // Re-initialise when a LiteRT URL is provided and we're not already using LiteRT.
  if (initPromise && !(liteRtModelUrl && activeBackend !== "litert")) return initPromise;
  initPromise = _init(liteRtModelUrl);
  return initPromise;
}

async function _init(liteRtModelUrl?: string): Promise<EmbeddingStatus> {
  // --- Try LiteRT first ---
  if (liteRtModelUrl) {
    try {
      const { loadLiteRt, loadAndCompile } = await import("@litertjs/core");
      await loadLiteRt("https://cdn.jsdelivr.net/npm/@litertjs/core/wasm/");
      liteRtModel = await loadAndCompile(liteRtModelUrl, { accelerator: "wasm" });
      activeBackend = "litert";
      return { backend: "litert", modelUrl: liteRtModelUrl };
    } catch (err) {
      console.warn("[embeddings] LiteRT init failed, falling back to USE:", err);
    }
  }

  // --- Try TF.js Universal Sentence Encoder ---
  try {
    const tf = await import("@tensorflow/tfjs");
    await tf.ready();
    const use = await import("@tensorflow-models/universal-sentence-encoder");
    useModel = await use.load() as typeof useModel;
    activeBackend = "use";
    return { backend: "use" };
  } catch (err) {
    console.warn("[embeddings] USE init failed, using bag-of-words fallback:", err);
    activeBackend = "none";
    return { backend: "bow", reason: String(err) };
  }
}

// ---------------------------------------------------------------------------
// Embed
// ---------------------------------------------------------------------------

/**
 * Embed a single text string.
 * Returns a normalised float32 vector.
 */
export async function embed(text: string): Promise<number[]> {
  if (!initPromise) await initEmbeddings();

  if (activeBackend === "litert" && liteRtModel) {
    return embedWithLiteRt(text);
  }
  if (activeBackend === "use" && useModel) {
    return embedWithUse(text);
  }
  return bowEmbed(text);
}

// ---------------------------------------------------------------------------
// LiteRT backend
// ---------------------------------------------------------------------------

// LiteRT embedding models typically expect a fixed-length int32 token sequence.
// We use a simple character-level hash tokeniser that maps text → int32[128].
function tokenise(text: string, seqLen = 128): number[] {
  const tokens = new Array<number>(seqLen).fill(0);
  const words = text.toLowerCase().split(/\s+/).slice(0, seqLen);
  for (let i = 0; i < words.length; i++) {
    let h = 5381;
    for (let j = 0; j < words[i].length; j++)
      h = ((h << 5) + h) ^ words[i].charCodeAt(j);
    tokens[i] = Math.abs(h) % 30000; // vocab size cap
  }
  return tokens;
}

async function embedWithLiteRt(text: string): Promise<number[]> {
  if (!liteRtModel) throw new Error("LiteRT model not loaded");
  const { Tensor } = await import("@litertjs/core");

  const inputDetails = liteRtModel.getInputDetails() as Array<{ shape: number[] }>;
  const shape = inputDetails[0]?.shape ?? [1, 128];
  const seqLen = shape[shape.length - 1];

  const tokens = tokenise(text, seqLen);
  const inputTensor = new Tensor(new Int32Array(tokens), shape);

  let outputs: Array<{ data(): Promise<Float32Array>; delete(): void }>;
  try {
    outputs = (await liteRtModel.run(inputTensor)) as typeof outputs;
  } finally {
    inputTensor.delete();
  }

  try {
    const arr = await outputs[0].data();
    return l2Normalise(Array.from(arr));
  } finally {
    outputs.forEach((t) => t.delete());
  }
}

// ---------------------------------------------------------------------------
// TF.js USE backend
// ---------------------------------------------------------------------------

async function embedWithUse(text: string): Promise<number[]> {
  if (!useModel) throw new Error("USE model not loaded");
  const embeddings = await useModel.embed([text]);
  const arr = embeddings.arraySync();
  return l2Normalise(arr[0]);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function l2Normalise(vec: number[]): number[] {
  let norm = 0;
  for (const v of vec) norm += v * v;
  norm = Math.sqrt(norm);
  return norm === 0 ? vec : vec.map((v) => v / norm);
}

export function getActiveBackend(): Backend {
  return activeBackend;
}
