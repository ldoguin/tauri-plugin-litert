package com.plugin.litert

import android.app.Activity
import android.speech.tts.TextToSpeech
import android.util.Base64
import android.util.Log
import app.tauri.annotation.Command
import app.tauri.annotation.InvokeArg
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Channel
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSObject
import app.tauri.plugin.Plugin
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.NpuCompatibilityChecker
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.SamplerConfig
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch
import org.json.JSONArray
import java.util.concurrent.ConcurrentHashMap
import kotlin.system.measureNanoTime

// ---------------------------------------------------------------------------
// Argument data classes — inference (LiteRT)
// ---------------------------------------------------------------------------

@InvokeArg
class LoadModelArgs {
    lateinit var modelPath: String
    lateinit var modelId: String
    var accelerator: String = "cpu"
}

@InvokeArg
class ModelIdArgs {
    lateinit var modelId: String
}

@InvokeArg
class RunInferenceArgs {
    lateinit var modelId: String
    lateinit var inputs: Array<FloatArray>
    var inputTypes: Array<String>? = null
}

@InvokeArg
class CreateEmbeddingArgs {
    lateinit var modelId: String
    lateinit var input: FloatArray
}

// ---------------------------------------------------------------------------
// Argument data classes — LM (LiteRT-LM)
// ---------------------------------------------------------------------------

@InvokeArg
class LoadLmModelArgs {
    lateinit var modelPath: String
    lateinit var modelId: String
    var accelerator: String = "gpu"
    var cacheDir: String? = null
    /// Set to true for multimodal (vision-capable) models such as Gemma 4 E2B/E4B.
    var vision: Boolean = false
}

@InvokeArg
class LmModelIdArgs {
    lateinit var modelId: String
}

@InvokeArg
class ExtractBundledModelsArgs {
    lateinit var targetDir: String
}

@InvokeArg
class TtsSpeakArgs {
    lateinit var text: String
    var rate: Float = 1.0f
    var pitch: Float = 1.0f
}

@InvokeArg
class SamplerArgs {
    var temperature: Float = 0.8f
    var topP: Float = 0.95f
    var topK: Int = 40
}

@InvokeArg
class GenerateArgs {
    lateinit var modelId: String
    lateinit var prompt: String
    var sampler: SamplerArgs = SamplerArgs()
    var systemInstruction: String? = null
    var channel: Channel? = null
    /// Base64-encoded image bytes (no data-URL prefix). Optional — text-only if absent.
    var image: String? = null
}

// ---------------------------------------------------------------------------
// Internal model records
// ---------------------------------------------------------------------------

private data class LoadedModel(
    val modelId: String,
    val modelPath: String,
    val accelerator: String,
    val compiledModel: CompiledModel,
    val inputCount: Int,
    val outputCount: Int,
    val inputShapes: List<List<Int>>,
    val outputShapes: List<List<Int>>,
)

private data class LoadedLmModel(
    val modelId: String,
    val modelPath: String,
    val accelerator: String,
    val engine: Engine,
)

// ---------------------------------------------------------------------------
// Merged plugin — handles all LiteRT (inference/embedding) and LiteRT-LM
// (generation) commands under the single "litert" plugin name.
//
// Two separate Kotlin classes cannot share the same Tauri plugin name because
// PluginManager stores plugins in a map keyed by name; the second registration
// would overwrite the first.
// ---------------------------------------------------------------------------

@TauriPlugin
class LiteRtPlugin(private val activity: Activity) : Plugin(activity) {

    private val models   = ConcurrentHashMap<String, LoadedModel>()
    private val lmModels = ConcurrentHashMap<String, LoadedLmModel>()
    private val scope    = CoroutineScope(Dispatchers.IO)

    // ── Android TextToSpeech ────────────────────────────────────────────────

    private var tts: TextToSpeech? = null
    private var ttsReady = false

    private fun ensureTts(onReady: () -> Unit) {
        // TextToSpeech must be constructed on the main thread on some devices.
        activity.runOnUiThread {
            if (ttsReady && tts != null) { onReady(); return@runOnUiThread }
            tts?.shutdown()
            tts = TextToSpeech(activity) { status ->
                ttsReady = (status == TextToSpeech.SUCCESS)
                if (ttsReady) {
                    activity.runOnUiThread { onReady() }
                } else {
                    Log.e("LiteRtPlugin", "TextToSpeech init failed with status=$status")
                }
            }
        }
    }

    @Command
    fun ttsSpeak(invoke: Invoke) {
        try {
            val args = invoke.parseArgs(TtsSpeakArgs::class.java)
            Log.d("LiteRtPlugin", "ttsSpeak: text length=${args.text.length}")
            ensureTts {
                tts?.setSpeechRate(args.rate)
                tts?.setPitch(args.pitch)
                val result = tts?.speak(args.text, TextToSpeech.QUEUE_FLUSH, null, "tts_utt")
                Log.d("LiteRtPlugin", "tts.speak result=$result")
            }
            invoke.resolve()
        } catch (e: Exception) {
            Log.e("LiteRtPlugin", "ttsSpeak error: ${e.message}", e)
            invoke.reject(e.message ?: "ttsSpeak failed")
        }
    }

    @Command
    fun ttsCancel(invoke: Invoke) {
        try {
            tts?.stop()
            invoke.resolve()
        } catch (e: Exception) {
            invoke.reject(e.message ?: "ttsCancel failed")
        }
    }

    // ── queryAcceleratorSupport ───────────────────────────────────────────
    // Uses the litert SDK's built-in NpuCompatibilityChecker (which checks
    // Build.SOC_MANUFACTURER / Build.SOC_MODEL against its embedded SoC lists)
    // to determine the best available accelerator at runtime — no hardcoded
    // device lists needed in application code.
    //
    // Returns: { accelerator: "npu"|"gpu"|"cpu", vendor: string|null }
    //   "npu"  — supported Qualcomm Hexagon, MediaTek APU, or Google Tensor NPU
    //   "gpu"  — NPU not supported but GPU is assumed available (all modern phones)
    //   "cpu"  — explicit CPU-only request (never returned by auto-detect)

    @Command
    fun queryAcceleratorSupport(invoke: Invoke) {
        try {
            val (accelerator, vendor) = when {
                NpuCompatibilityChecker.Companion.Qualcomm.isDeviceSupported()     -> "npu" to "Qualcomm"
                NpuCompatibilityChecker.Companion.Mediatek.isDeviceSupported()     -> "npu" to "MediaTek"
                NpuCompatibilityChecker.Companion.GoogleTensor.isDeviceSupported() -> "npu" to "GoogleTensor"
                else                                                                -> "gpu" to null
            }
            val result = JSObject()
            result.put("accelerator", accelerator)
            if (vendor != null) result.put("vendor", vendor)
            Log.d("LiteRtPlugin", "queryAcceleratorSupport: accelerator=$accelerator vendor=$vendor")
            invoke.resolve(result)
        } catch (e: Exception) {
            Log.e("LiteRtPlugin", "queryAcceleratorSupport error: ${e.message}", e)
            // Fall back to GPU on any error (safer than NPU for an unknown device)
            val result = JSObject()
            result.put("accelerator", "gpu")
            invoke.resolve(result)
        }
    }

    // ── extractBundledModels ──────────────────────────────────────────────
    // Copies APK assets under assets/bundled-models/ (small task models that
    // can't be auto-downloaded — gated/dead upstream URLs) into targetDir,
    // skipping files that already exist there. Called once at app startup;
    // idempotent so repeat calls after the first launch are cheap no-ops.
    // Runs on Dispatchers.IO — the bundled set can be hundreds of MB, and
    // Rust's run_mobile_plugin call blocks on whichever thread resolves this,
    // so doing the copy on IO instead of the invoking thread avoids an ANR
    // if that thread turns out to be the main/UI thread.

    @Command
    fun extractBundledModels(invoke: Invoke) {
        val args = invoke.parseArgs(ExtractBundledModelsArgs::class.java)
        scope.launch {
            try {
                val assetDir = "bundled-models"
                val names = try {
                    activity.assets.list(assetDir) ?: emptyArray()
                } catch (e: Exception) {
                    Log.d("LiteRtPlugin", "extractBundledModels: no $assetDir/ in APK assets")
                    emptyArray()
                }

                val targetDir = java.io.File(args.targetDir)
                targetDir.mkdirs()

                var copied = 0
                for (name in names) {
                    val dest = java.io.File(targetDir, name)
                    if (dest.exists()) continue
                    activity.assets.open("$assetDir/$name").use { input ->
                        dest.outputStream().use { output -> input.copyTo(output) }
                    }
                    copied++
                    Log.d("LiteRtPlugin", "extractBundledModels: extracted $name -> ${dest.absolutePath}")
                }

                val result = JSObject()
                result.put("copied", copied)
                invoke.resolve(result)
            } catch (e: Exception) {
                Log.e("LiteRtPlugin", "extractBundledModels error: ${e.message}", e)
                invoke.reject(e.message ?: "extractBundledModels failed")
            }
        }
    }

    // ── loadModel ──────────────────────────────────────────────────────────

    @Command
    fun loadModel(invoke: Invoke) {
        val args = invoke.parseArgs(LoadModelArgs::class.java)

        if (models.containsKey(args.modelId)) {
            invoke.reject("model already loaded: ${args.modelId}")
            return
        }

        scope.launch {
            try {
                val accel = when (args.accelerator.lowercase()) {
                    "gpu" -> Accelerator.GPU
                    "npu" -> Accelerator.NPU
                    else  -> Accelerator.CPU
                }

                val compiledModel = CompiledModel.create(args.modelPath, CompiledModel.Options(accel))

                val inputCount: Int  = compiledModel.createInputBuffers(0).size
                val outputCount: Int = compiledModel.createOutputBuffers(0).size

                if (models.containsKey(args.modelId)) {
                    compiledModel.close()
                    invoke.reject("model already loaded: ${args.modelId}")
                    return@launch
                }

                val loaded = LoadedModel(
                    modelId       = args.modelId,
                    modelPath     = args.modelPath,
                    accelerator   = args.accelerator,
                    compiledModel = compiledModel,
                    inputCount    = inputCount,
                    outputCount   = outputCount,
                    inputShapes   = emptyList(),
                    outputShapes  = emptyList(),
                )
                models[args.modelId] = loaded
                invoke.resolve(loaded.toJSObject())
            } catch (e: Exception) {
                invoke.reject("load_model failed: ${e.message}")
            }
        }
    }

    // ── unloadModel ────────────────────────────────────────────────────────

    @Command
    fun unloadModel(invoke: Invoke) {
        val args = invoke.parseArgs(ModelIdArgs::class.java)
        val removed = models.remove(args.modelId)
        if (removed == null) {
            invoke.reject("model not found: ${args.modelId}")
        } else {
            try { removed.compiledModel.close() } catch (_: Exception) {}
            invoke.resolve()
        }
    }

    // ── listModels ─────────────────────────────────────────────────────────

    @Command
    fun listModels(invoke: Invoke) {
        val arr = JSONArray()
        models.values.forEach { arr.put(it.toJSObject()) }
        val result = JSObject()
        result.put("models", arr)
        invoke.resolve(result)
    }

    // ── getModelInfo ───────────────────────────────────────────────────────

    @Command
    fun getModelInfo(invoke: Invoke) {
        val args = invoke.parseArgs(ModelIdArgs::class.java)
        val loaded = models[args.modelId]
        if (loaded == null) {
            invoke.reject("model not found: ${args.modelId}")
        } else {
            invoke.resolve(loaded.toJSObject())
        }
    }

    // ── runInference ───────────────────────────────────────────────────────

    @Command
    fun runInference(invoke: Invoke) {
        val args = invoke.parseArgs(RunInferenceArgs::class.java)
        val loaded = models[args.modelId]
        if (loaded == null) {
            invoke.reject("model not found: ${args.modelId}")
            return
        }

        if (args.inputs.size != loaded.inputCount) {
            invoke.reject("expected ${loaded.inputCount} input tensors, got ${args.inputs.size}")
            return
        }

        scope.launch {
            try {
                val inputBuffers  = loaded.compiledModel.createInputBuffers(0)
                val outputBuffers = loaded.compiledModel.createOutputBuffers(0)
                args.inputs.forEachIndexed { i, data ->
                    val type = args.inputTypes?.getOrNull(i) ?: "float"
                    when (type) {
                        "int32" -> inputBuffers[i].writeInt(data.map { it.toInt() }.toIntArray())
                        // Quantized uint8 input tensors (e.g. MoveNet Lightning) expect raw
                        // 0-255 byte values. JVM Byte is signed (-128..127); toByte() truncates
                        // to the low 8 bits, which is the correct bit pattern for unsigned byte
                        // data — the native side reads the bytes raw, not as a signed value.
                        "int8", "uint8" -> inputBuffers[i].writeInt8(data.map { it.toInt().toByte() }.toByteArray())
                        else -> inputBuffers[i].writeFloat(data)
                    }
                }

                val nanos = measureNanoTime {
                    loaded.compiledModel.run(inputBuffers, outputBuffers, 0)
                }

                val outputsArr = JSONArray()
                outputBuffers.forEach { buf ->
                    val inner = JSONArray()
                    buf.readFloat().forEach { inner.put(it) }
                    outputsArr.put(inner)
                }

                val result = JSObject()
                result.put("modelId",   args.modelId)
                result.put("outputs",   outputsArr)
                result.put("latencyMs", nanos / 1_000_000.0)
                invoke.resolve(result)
            } catch (e: Exception) {
                invoke.reject("run_inference failed: ${e.message}")
            }
        }
    }

    // ── createEmbedding ────────────────────────────────────────────────────

    @Command
    fun createEmbedding(invoke: Invoke) {
        val args = invoke.parseArgs(CreateEmbeddingArgs::class.java)
        val loaded = models[args.modelId]
        if (loaded == null) {
            invoke.reject("model not found: ${args.modelId}")
            return
        }

        scope.launch {
            try {
                val inputBuffers  = loaded.compiledModel.createInputBuffers(0)
                val outputBuffers = loaded.compiledModel.createOutputBuffers(0)
                inputBuffers[0].writeFloat(args.input)

                val nanos = measureNanoTime {
                    loaded.compiledModel.run(inputBuffers, outputBuffers, 0)
                }

                val embArr = JSONArray()
                outputBuffers[0].readFloat().forEach { embArr.put(it) }

                val result = JSObject()
                result.put("modelId",   args.modelId)
                result.put("embedding", embArr)
                result.put("latencyMs", nanos / 1_000_000.0)
                invoke.resolve(result)
            } catch (e: Exception) {
                invoke.reject("create_embedding failed: ${e.message}")
            }
        }
    }

    // ── loadLmModel ────────────────────────────────────────────────────────
    // Attempts backends in priority order and falls back automatically:
    //   npu  → [NPU, GPU, CPU]
    //   gpu  → [GPU, CPU]
    //   cpu  → [CPU]
    // Vision (multimodal) requires GPU; it is silently dropped when falling
    // back to CPU so the user still gets text-only inference rather than an
    // error. The actually-used accelerator is reported back in the response.

    @Command
    fun loadLmModel(invoke: Invoke) {
        val args = invoke.parseArgs(LoadLmModelArgs::class.java)

        if (lmModels.containsKey(args.modelId)) {
            invoke.reject("model already loaded: ${args.modelId}")
            return
        }

        data class Attempt(val backend: Backend, val name: String, val vision: Boolean)

        val attempts: List<Attempt> = when (args.accelerator.lowercase()) {
            "npu" -> listOf(
                Attempt(Backend.NPU(activity.applicationInfo.nativeLibraryDir), "npu", args.vision),
                Attempt(Backend.GPU(),                                           "gpu", args.vision),
                Attempt(Backend.CPU(),                                           "cpu", false),
            )
            "gpu" -> listOf(
                Attempt(Backend.GPU(), "gpu", args.vision),
                Attempt(Backend.CPU(), "cpu", false),
            )
            else -> listOf(
                Attempt(Backend.CPU(), "cpu", false),
            )
        }

        scope.launch {
            var lastError = "no backends attempted"
            for ((idx, attempt) in attempts.withIndex()) {
                try {
                    val config = if (attempt.vision) {
                        EngineConfig(
                            modelPath     = args.modelPath,
                            backend       = attempt.backend,
                            cacheDir      = args.cacheDir ?: activity.cacheDir.path,
                            visionBackend = Backend.GPU(),
                        )
                    } else {
                        EngineConfig(
                            modelPath = args.modelPath,
                            backend   = attempt.backend,
                            cacheDir  = args.cacheDir ?: activity.cacheDir.path,
                        )
                    }

                    val engine = Engine(config)
                    engine.initialize()

                    if (idx > 0) {
                        Log.w("LiteRtPlugin",
                            "loadLmModel: ${args.accelerator} unavailable, using ${attempt.name}" +
                            if (!attempt.vision && args.vision) " (vision disabled)" else "")
                    }

                    val loaded = LoadedLmModel(
                        modelId     = args.modelId,
                        modelPath   = args.modelPath,
                        accelerator = attempt.name,   // actual accelerator, not the requested one
                        engine      = engine,
                    )
                    lmModels[args.modelId] = loaded
                    invoke.resolve(loaded.toJSObject())
                    return@launch
                } catch (e: Exception) {
                    lastError = e.message ?: "unknown error"
                    Log.w("LiteRtPlugin", "loadLmModel: ${attempt.name} failed — $lastError")
                }
            }
            invoke.reject("load_lm_model failed on all backends: $lastError")
        }
    }

    // ── unloadLmModel ──────────────────────────────────────────────────────

    @Command
    fun unloadLmModel(invoke: Invoke) {
        val args = invoke.parseArgs(LmModelIdArgs::class.java)
        val removed = lmModels.remove(args.modelId)
        if (removed == null) {
            invoke.reject("model not found: ${args.modelId}")
        } else {
            try { removed.engine.close() } catch (_: Exception) {}
            invoke.resolve()
        }
    }

    // ── listLmModels ───────────────────────────────────────────────────────

    @Command
    fun listLmModels(invoke: Invoke) {
        val arr = JSONArray()
        lmModels.values.forEach { arr.put(it.toJSObject()) }
        val result = JSObject()
        result.put("models", arr)
        invoke.resolve(result)
    }

    // ── generate ───────────────────────────────────────────────────────────

    @Command
    fun generate(invoke: Invoke) {
        val args = invoke.parseArgs(GenerateArgs::class.java)
        val loaded = lmModels[args.modelId]
        if (loaded == null) {
            invoke.reject("model not found: ${args.modelId}")
            return
        }

        scope.launch {
            try {
                val convConfig = buildConversationConfig(args)
                loaded.engine.createConversation(convConfig).use { conv ->
                    var text = ""
                    val contents = buildContents(args)
                    val nanos = measureNanoTime {
                        text = conv.sendMessage(contents).extractText()
                    }
                    val result = JSObject()
                    result.put("modelId",   args.modelId)
                    result.put("text",      text)
                    result.put("latencyMs", nanos / 1_000_000.0)
                    invoke.resolve(result)
                }
            } catch (e: Exception) {
                invoke.reject("generate failed: ${e.message}")
            }
        }
    }

    // ── generateStream ─────────────────────────────────────────────────────

    @Command
    fun generateStream(invoke: Invoke) {
        val args = invoke.parseArgs(GenerateArgs::class.java)
        val loaded = lmModels[args.modelId]
        if (loaded == null) {
            invoke.reject("model not found: ${args.modelId}")
            return
        }

        val ch = args.channel
        invoke.resolve()

        scope.launch {
            try {
                val convConfig = buildConversationConfig(args)
                loaded.engine.createConversation(convConfig).use { conv ->
                    val startNs = System.nanoTime()
                    val contents = buildContents(args)

                    conv.sendMessageAsync(contents)
                        .catch { e ->
                            val err = JSObject()
                            err.put("modelId",   args.modelId)
                            err.put("chunk",     "")
                            err.put("done",      true)
                            err.put("latencyMs", null)
                            err.put("error",     e.message)
                            ch?.send(err)
                        }
                        .collect { message ->
                            val chunk = JSObject()
                            chunk.put("modelId",   args.modelId)
                            chunk.put("chunk",     message.extractText())
                            chunk.put("done",      false)
                            chunk.put("latencyMs", null)
                            ch?.send(chunk)
                        }

                    val latencyMs = (System.nanoTime() - startNs) / 1_000_000.0
                    val done = JSObject()
                    done.put("modelId",   args.modelId)
                    done.put("chunk",     "")
                    done.put("done",      true)
                    done.put("latencyMs", latencyMs)
                    ch?.send(done)
                }
            } catch (e: Exception) {
                val err = JSObject()
                err.put("modelId",   args.modelId)
                err.put("chunk",     "")
                err.put("done",      true)
                err.put("latencyMs", null)
                err.put("error",     e.message)
                ch?.send(err)
            }
        }
    }

    // ── Lifecycle ──────────────────────────────────────────────────────────

    override fun onDestroy() {
        models.values.forEach { try { it.compiledModel.close() } catch (_: Exception) {} }
        models.clear()
        lmModels.values.forEach { try { it.engine.close() } catch (_: Exception) {} }
        lmModels.clear()
        super.onDestroy()
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    private fun buildContents(args: GenerateArgs): Contents {
        val imageB64 = args.image
        Log.d("LiteRtPlugin", "buildContents: image=${if (imageB64 == null) "null" else "len=${imageB64.length}"}")
        if (imageB64 != null && imageB64.isNotEmpty()) {
            // Write image bytes to a temp file so we can use Content.ImageFile,
            // which is more reliable than Content.ImageBytes for Gemma 4 multimodal.
            try {
                val bytes = Base64.decode(imageB64, Base64.DEFAULT)
                Log.d("LiteRtPlugin", "buildContents: decoded ${bytes.size} bytes")
                val ext = when {
                    bytes.size >= 4 &&
                        bytes[0] == 0x89.toByte() && bytes[1] == 0x50.toByte() -> "png"
                    bytes.size >= 4 &&
                        bytes[0] == 0x52.toByte() && bytes[1] == 0x49.toByte() -> "webp"
                    else -> "jpg"
                }
                val tmp = java.io.File.createTempFile("litert_img_", ".$ext", activity.cacheDir)
                tmp.writeBytes(bytes)
                tmp.deleteOnExit()
                Log.d("LiteRtPlugin", "buildContents: temp file=${tmp.absolutePath} size=${tmp.length()} readable=${tmp.canRead()}")
                Log.d("LiteRtPlugin", "buildContents: using Content.ImageFile+Text")
                return Contents.of(Content.ImageFile(tmp.absolutePath), Content.Text(args.prompt))
            } catch (e: Exception) {
                Log.e("LiteRtPlugin", "buildContents: error processing image, falling back to text-only", e)
                // Fall through to text-only on any error
            }
        }
        Log.d("LiteRtPlugin", "buildContents: text-only path")
        return Contents.of(args.prompt)
    }

    private fun buildConversationConfig(args: GenerateArgs): ConversationConfig {
        val samplerConfig = SamplerConfig(
            topK        = args.sampler.topK,
            topP        = args.sampler.topP.toDouble(),
            temperature = args.sampler.temperature.toDouble(),
        )
        return if (args.systemInstruction != null) {
            ConversationConfig(
                systemInstruction = Contents.of(args.systemInstruction!!),
                samplerConfig = samplerConfig,
            )
        } else {
            ConversationConfig(samplerConfig = samplerConfig)
        }
    }

    private fun Message.extractText(): String =
        contents.contents.filterIsInstance<Content.Text>().joinToString("") { it.text }

    private fun LoadedModel.toJSObject(): JSObject {
        val obj = JSObject()
        obj.put("modelId",     modelId)
        obj.put("modelPath",   modelPath)
        obj.put("accelerator", accelerator)
        obj.put("inputCount",  inputCount)
        obj.put("outputCount", outputCount)
        val inShapes = JSONArray()
        inputShapes.forEach { shape -> val arr = JSONArray(); shape.forEach { arr.put(it) }; inShapes.put(arr) }
        obj.put("inputShapes", inShapes)
        val outShapes = JSONArray()
        outputShapes.forEach { shape -> val arr = JSONArray(); shape.forEach { arr.put(it) }; outShapes.put(arr) }
        obj.put("outputShapes", outShapes)
        return obj
    }

    private fun LoadedLmModel.toJSObject(): JSObject {
        val obj = JSObject()
        obj.put("modelId",     modelId)
        obj.put("modelPath",   modelPath)
        obj.put("accelerator", accelerator)
        return obj
    }
}
