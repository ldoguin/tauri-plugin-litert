package com.plugin.litert

import android.app.Activity
import app.tauri.annotation.Command
import app.tauri.annotation.InvokeArg
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Channel
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSObject
import app.tauri.plugin.Plugin
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch
import org.json.JSONArray
import java.util.concurrent.ConcurrentHashMap
import kotlin.system.measureNanoTime

// ---------------------------------------------------------------------------
// Argument data classes
// ---------------------------------------------------------------------------

@InvokeArg
class LoadLmModelArgs {
    lateinit var modelPath: String
    lateinit var modelId: String
    var accelerator: String = "gpu"
    var cacheDir: String? = null
}

@InvokeArg
class LmModelIdArgs {
    lateinit var modelId: String
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
    // Present only for generateStream — Tauri's ChannelDeserializer populates this
    // from the "__CHANNEL__:<id>" string passed by the Rust mobile_lm.rs bridge.
    var channel: Channel? = null
}

// ---------------------------------------------------------------------------
// Internal record
// ---------------------------------------------------------------------------

private data class LoadedLmModel(
    val modelId: String,
    val modelPath: String,
    val accelerator: String,
    val engine: Engine,
)

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

@TauriPlugin
class LiteRtLmPlugin(private val activity: Activity) : Plugin(activity) {

    private val models = ConcurrentHashMap<String, LoadedLmModel>()
    private val scope = CoroutineScope(Dispatchers.IO)

    // -----------------------------------------------------------------------
    // loadLmModel
    // -----------------------------------------------------------------------
    @Command
    fun loadLmModel(invoke: Invoke) {
        val args = invoke.parseArgs(LoadLmModelArgs::class.java)

        if (models.containsKey(args.modelId)) {
            invoke.reject("model already loaded: ${args.modelId}")
            return
        }

        scope.launch {
            try {
                val backend = when (args.accelerator.lowercase()) {
                    "gpu" -> Backend.GPU()
                    "npu" -> Backend.NPU(
                        nativeLibraryDir = activity.applicationInfo.nativeLibraryDir
                    )
                    else  -> Backend.CPU()
                }

                val config = EngineConfig(
                    modelPath = args.modelPath,
                    backend = backend,
                    cacheDir = args.cacheDir ?: activity.cacheDir.path,
                )

                val engine = Engine(config)
                engine.initialize()

                val loaded = LoadedLmModel(
                    modelId     = args.modelId,
                    modelPath   = args.modelPath,
                    accelerator = args.accelerator,
                    engine      = engine,
                )
                models[args.modelId] = loaded

                invoke.resolve(loaded.toJSObject())
            } catch (e: Exception) {
                invoke.reject("load_lm_model failed: ${e.message}")
            }
        }
    }

    // -----------------------------------------------------------------------
    // unloadLmModel
    // -----------------------------------------------------------------------
    @Command
    fun unloadLmModel(invoke: Invoke) {
        val args = invoke.parseArgs(LmModelIdArgs::class.java)
        val removed = models.remove(args.modelId)
        if (removed == null) {
            invoke.reject("model not found: ${args.modelId}")
        } else {
            try { removed.engine.close() } catch (_: Exception) {}
            invoke.resolve()
        }
    }

    // -----------------------------------------------------------------------
    // listLmModels
    // -----------------------------------------------------------------------
    @Command
    fun listLmModels(invoke: Invoke) {
        val arr = JSONArray()
        models.values.forEach { arr.put(it.toJSObject()) }
        val result = JSObject()
        result.put("models", arr)
        invoke.resolve(result)
    }

    // -----------------------------------------------------------------------
    // generate (blocking, returns full text)
    // -----------------------------------------------------------------------
    @Command
    fun generate(invoke: Invoke) {
        val args = invoke.parseArgs(GenerateArgs::class.java)
        val loaded = models[args.modelId]
        if (loaded == null) {
            invoke.reject("model not found: ${args.modelId}")
            return
        }

        scope.launch {
            try {
                val convConfig = buildConversationConfig(args)

                loaded.engine.createConversation(convConfig).use { conv ->
                    var text = ""
                    val nanos = measureNanoTime {
                        text = conv.sendMessage(args.prompt).text ?: ""
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

    // -----------------------------------------------------------------------
    // generateStream — emits Tauri events for each token chunk
    // -----------------------------------------------------------------------
    @Command
    fun generateStream(invoke: Invoke) {
        val args = invoke.parseArgs(GenerateArgs::class.java)
        val loaded = models[args.modelId]
        if (loaded == null) {
            invoke.reject("model not found: ${args.modelId}")
            return
        }

        // Acknowledge the command immediately; tokens arrive via the Tauri Channel
        // which forwards each payload through the Rust event bus to JS listen().
        val ch = args.channel
        invoke.resolve()

        scope.launch {
            try {
                val convConfig = buildConversationConfig(args)

                loaded.engine.createConversation(convConfig).use { conv ->
                    val startNs = System.nanoTime()

                    conv.sendMessageAsync(args.prompt)
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
                            chunk.put("chunk",     message.text ?: "")
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

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------
    override fun onDestroy() {
        models.values.forEach { try { it.engine.close() } catch (_: Exception) {} }
        models.clear()
        super.onDestroy()
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private fun buildConversationConfig(args: GenerateArgs): ConversationConfig {
        val samplerConfig = SamplerConfig(
            topK        = args.sampler.topK,
            topP        = args.sampler.topP,
            temperature = args.sampler.temperature,
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

    private fun LoadedLmModel.toJSObject(): JSObject {
        val obj = JSObject()
        obj.put("modelId",     modelId)
        obj.put("modelPath",   modelPath)
        obj.put("accelerator", accelerator)
        return obj
    }
}
