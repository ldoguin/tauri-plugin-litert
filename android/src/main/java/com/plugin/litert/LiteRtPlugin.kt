package com.plugin.litert

import android.app.Activity
import app.tauri.annotation.Command
import app.tauri.annotation.InvokeArg
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSObject
import app.tauri.plugin.Plugin
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Accelerator
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.json.JSONArray
import java.util.concurrent.ConcurrentHashMap
import kotlin.system.measureNanoTime

// ---------------------------------------------------------------------------
// Argument data classes (deserialised from JS via @InvokeArg)
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
}

@InvokeArg
class CreateEmbeddingArgs {
    lateinit var modelId: String
    lateinit var input: FloatArray
}

// ---------------------------------------------------------------------------
// Internal model record
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

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

@TauriPlugin
class LiteRtPlugin(private val activity: Activity) : Plugin(activity) {

    // ConcurrentHashMap for thread-safe access from concurrent Tauri commands.
    private val models = ConcurrentHashMap<String, LoadedModel>()
    private val scope = CoroutineScope(Dispatchers.IO)

    // -----------------------------------------------------------------------
    // loadModel
    // -----------------------------------------------------------------------
    @Command
    fun loadModel(invoke: Invoke) {
        val args = invoke.parseArgs(LoadModelArgs::class.java)

        if (models.containsKey(args.modelId)) {
            invoke.reject("model already loaded: ${args.modelId}")
            return
        }

        // CompiledModel.create() loads the model from disk — run off the main thread.
        scope.launch {
            try {
                val accel = when (args.accelerator.lowercase()) {
                    "gpu" -> Accelerator.GPU
                    "npu" -> Accelerator.NPU
                    else  -> Accelerator.CPU
                }

                val compiledModel = CompiledModel.create(args.modelPath, CompiledModel.Options(accel))

                // Count inputs/outputs by allocating buffers briefly.
                // TensorBuffer has no shape property in the 2.x API, so inputShapes /
                // outputShapes are left empty — they are informational only and not
                // required for inference.
                val inputCount: Int
                val outputCount: Int

                compiledModel.createInputBuffers().use { inputBuffers ->
                    inputCount = inputBuffers.size
                }
                compiledModel.createOutputBuffers().use { outputBuffers ->
                    outputCount = outputBuffers.size
                }

                val inputShapes: List<List<Int>> = emptyList()
                val outputShapes: List<List<Int>> = emptyList()

                // Re-check for duplicate after the blocking load (concurrent call may have raced).
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
                    inputShapes   = inputShapes,
                    outputShapes  = outputShapes,
                )
                models[args.modelId] = loaded

                invoke.resolve(loaded.toJSObject())
            } catch (e: Exception) {
                invoke.reject("load_model failed: ${e.message}")
            }
        }
    }

    // -----------------------------------------------------------------------
    // unloadModel
    // -----------------------------------------------------------------------
    @Command
    fun unloadModel(invoke: Invoke) {
        val args = invoke.parseArgs(ModelIdArgs::class.java)
        val removed = models.remove(args.modelId)
        if (removed == null) {
            invoke.reject("model not found: ${args.modelId}")
        } else {
            // Release native resources held by the CompiledModel.
            try { removed.compiledModel.close() } catch (_: Exception) {}
            invoke.resolve()
        }
    }

    // -----------------------------------------------------------------------
    // listModels
    // -----------------------------------------------------------------------
    @Command
    fun listModels(invoke: Invoke) {
        val arr = JSONArray()
        models.values.forEach { arr.put(it.toJSObject()) }
        // Wrap in { "models": [...] } to match the Rust ListModelsResponse wrapper.
        val result = JSObject()
        result.put("models", arr)
        invoke.resolve(result)
    }

    // -----------------------------------------------------------------------
    // getModelInfo
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // runInference
    // -----------------------------------------------------------------------
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
                // Allocate fresh buffers per call and close them when done.
                val nanos: Long
                val outputsArr = JSONArray()

                loaded.compiledModel.createInputBuffers().use { inputBuffers ->
                    loaded.compiledModel.createOutputBuffers().use { outputBuffers ->
                        args.inputs.forEachIndexed { i, data ->
                            inputBuffers[i].writeFloat(data)
                        }

                        nanos = measureNanoTime {
                            loaded.compiledModel.run(inputBuffers, outputBuffers)
                        }

                        outputBuffers.forEach { buf ->
                            val floats = buf.readFloat()
                            val inner = JSONArray()
                            floats.forEach { inner.put(it) }
                            outputsArr.put(inner)
                        }
                    }
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

    // -----------------------------------------------------------------------
    // createEmbedding
    // -----------------------------------------------------------------------
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
                val nanos: Long
                val embArr = JSONArray()

                loaded.compiledModel.createInputBuffers().use { inputBuffers ->
                    loaded.compiledModel.createOutputBuffers().use { outputBuffers ->
                        inputBuffers[0].writeFloat(args.input)

                        nanos = measureNanoTime {
                            loaded.compiledModel.run(inputBuffers, outputBuffers)
                        }

                        outputBuffers[0].readFloat().forEach { embArr.put(it) }
                    }
                }

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

    // -----------------------------------------------------------------------
    // Lifecycle: close all models when the plugin is destroyed
    // -----------------------------------------------------------------------
    override fun onDestroy() {
        models.values.forEach { loaded ->
            try { loaded.compiledModel.close() } catch (_: Exception) {}
        }
        models.clear()
        super.onDestroy()
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private fun LoadedModel.toJSObject(): JSObject {
        val obj = JSObject()
        obj.put("modelId",     modelId)
        obj.put("modelPath",   modelPath)
        obj.put("accelerator", accelerator)
        obj.put("inputCount",  inputCount)
        obj.put("outputCount", outputCount)

        val inShapes = JSONArray()
        inputShapes.forEach { shape ->
            val arr = JSONArray(); shape.forEach { arr.put(it) }; inShapes.put(arr)
        }
        obj.put("inputShapes", inShapes)

        val outShapes = JSONArray()
        outputShapes.forEach { shape ->
            val arr = JSONArray(); shape.forEach { arr.put(it) }; outShapes.put(arr)
        }
        obj.put("outputShapes", outShapes)

        return obj
    }
}
