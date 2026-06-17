import Foundation
import Tauri
import UIKit
import WebKit
import LiteRTLM

// ---------------------------------------------------------------------------
// LiteRtPlugin — iOS implementation of tauri-plugin-litert.
//
// LLM generation (loadLmModel/unloadLmModel/listLmModels/generate/
// generateStream) is REAL, backed by Google's LiteRT-LM Swift API (the same
// engine + .litertlm model format as the Android Kotlin plugin).
//
// Raw tensor inference / embeddings (loadModel/runInference/createEmbedding)
// are NOT implemented yet on iOS — they need a TFLite interpreter binding
// (no official SwiftPM distribution as of LiteRT 2.x; would require vendoring
// the TensorFlowLiteC xcframework). The app's RAG falls back to its JS
// bag-of-words embedder when these commands reject.
// ---------------------------------------------------------------------------

private let kInferenceNotImplemented =
    "tauri-plugin-litert: raw inference/embeddings are not yet implemented on iOS (LLM generation is)"

// ── Argument types (mirror the Kotlin @InvokeArg classes) ──────────────────

class LoadModelArgs: Decodable {
    let modelPath: String
    let modelId: String
    var accelerator: String? = "cpu"
}

class ModelIdArgs: Decodable {
    let modelId: String
}

class RunInferenceArgs: Decodable {
    let modelId: String
    let inputs: [[Float]]
    var inputTypes: [String]? = nil
}

class CreateEmbeddingArgs: Decodable {
    let modelId: String
    let input: [Float]
}

class LoadLmModelArgs: Decodable {
    let modelPath: String
    let modelId: String
    var accelerator: String? = "gpu"
    var cacheDir: String? = nil
    var vision: Bool? = false
}

class SamplerArgs: Decodable {
    var temperature: Float? = 0.8
    var topP: Float? = 0.95
    var topK: Int? = 40
}

class GenerateArgs: Decodable {
    let modelId: String
    let prompt: String
    var sampler: SamplerArgs? = nil
    var systemInstruction: String? = nil
    var channel: Channel? = nil
    /// Base64-encoded image bytes (no data-URL prefix). Optional.
    var image: String? = nil
}

// ── Stream chunk payload (matches the Kotlin chunk JSObject shape) ──────────

private struct StreamChunk: Encodable {
    let modelId: String
    let chunk: String
    let done: Bool
    let latencyMs: Double?
    var error: String? = nil
}

// ── Loaded model record ─────────────────────────────────────────────────────

private final class LoadedLmModel {
    let modelId: String
    let modelPath: String
    let accelerator: String
    let engine: Engine

    init(modelId: String, modelPath: String, accelerator: String, engine: Engine) {
        self.modelId = modelId
        self.modelPath = modelPath
        self.accelerator = accelerator
        self.engine = engine
    }

    var asJSObject: JSObject {
        return [
            "modelId": modelId,
            "modelPath": modelPath,
            "accelerator": accelerator,
        ]
    }
}

// ---------------------------------------------------------------------------

class LiteRtPlugin: Plugin {

    private var lmModels: [String: LoadedLmModel] = [:]
    private let lock = NSLock()

    private func getLmModel(_ id: String) -> LoadedLmModel? {
        lock.lock()
        defer { lock.unlock() }
        return lmModels[id]
    }

    // ── LiteRT raw inference / embeddings — not yet implemented ─────────────

    @objc public func loadModel(_ invoke: Invoke) throws {
        _ = try invoke.parseArgs(LoadModelArgs.self)
        invoke.reject(kInferenceNotImplemented)
    }

    @objc public func unloadModel(_ invoke: Invoke) throws {
        _ = try invoke.parseArgs(ModelIdArgs.self)
        invoke.reject(kInferenceNotImplemented)
    }

    @objc public func listModels(_ invoke: Invoke) throws {
        invoke.resolve(["models": [] as [JSObject]])
    }

    @objc public func getModelInfo(_ invoke: Invoke) throws {
        _ = try invoke.parseArgs(ModelIdArgs.self)
        invoke.reject(kInferenceNotImplemented)
    }

    @objc public func runInference(_ invoke: Invoke) throws {
        _ = try invoke.parseArgs(RunInferenceArgs.self)
        invoke.reject(kInferenceNotImplemented)
    }

    @objc public func createEmbedding(_ invoke: Invoke) throws {
        _ = try invoke.parseArgs(CreateEmbeddingArgs.self)
        invoke.reject(kInferenceNotImplemented)
    }

    // ── loadLmModel ──────────────────────────────────────────────────────────

    @objc public func loadLmModel(_ invoke: Invoke) throws {
        let args = try invoke.parseArgs(LoadLmModelArgs.self)

        if getLmModel(args.modelId) != nil {
            invoke.reject("model already loaded: \(args.modelId)")
            return
        }

        // Engine initialization is heavy (reads the whole model); Engine is an
        // actor, so drive it from a Task. invoke resolution is thread-safe.
        Task { [weak self] in
            guard let self else { return }
            do {
                let accelerator = (args.accelerator ?? "gpu").lowercased()
                let backend: Backend = accelerator == "gpu" ? .gpu : .cpu()
                let cacheDir = args.cacheDir
                    ?? FileManager.default.temporaryDirectory.path

                let config = try EngineConfig(
                    modelPath: args.modelPath,
                    backend: backend,
                    visionBackend: (args.vision ?? false) ? .gpu : nil,
                    cacheDir: cacheDir
                )
                let engine = Engine(engineConfig: config)
                try await engine.initialize()

                let loaded = LoadedLmModel(
                    modelId: args.modelId,
                    modelPath: args.modelPath,
                    accelerator: accelerator,
                    engine: engine
                )
                self.lock.lock()
                self.lmModels[args.modelId] = loaded
                self.lock.unlock()
                invoke.resolve(loaded.asJSObject)
            } catch {
                invoke.reject("load_lm_model failed: \(error)")
            }
        }
    }

    // ── unloadLmModel ────────────────────────────────────────────────────────

    @objc public func unloadLmModel(_ invoke: Invoke) throws {
        let args = try invoke.parseArgs(ModelIdArgs.self)
        lock.lock()
        let removed = lmModels.removeValue(forKey: args.modelId)
        lock.unlock()
        if removed == nil {
            invoke.reject("model not found: \(args.modelId)")
        } else {
            invoke.resolve()
        }
    }

    // ── listLmModels ─────────────────────────────────────────────────────────

    @objc public func listLmModels(_ invoke: Invoke) throws {
        lock.lock()
        let models = lmModels.values.map { $0.asJSObject }
        lock.unlock()
        invoke.resolve(["models": models])
    }

    // ── generate ─────────────────────────────────────────────────────────────

    @objc public func generate(_ invoke: Invoke) throws {
        let args = try invoke.parseArgs(GenerateArgs.self)
        guard let loaded = getLmModel(args.modelId) else {
            invoke.reject("model not found: \(args.modelId)")
            return
        }

        Task {
            do {
                let conversation = try await loaded.engine.createConversation(
                    with: Self.conversationConfig(for: args))
                let message = Self.buildMessage(args)
                let start = DispatchTime.now()
                let response = try await conversation.sendMessage(message)
                let latencyMs =
                    Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000.0
                invoke.resolve([
                    "modelId": args.modelId,
                    "text": response.toString,
                    "latencyMs": latencyMs,
                ] as JSObject)
            } catch {
                invoke.reject("generate failed: \(error)")
            }
        }
    }

    // ── generateStream ───────────────────────────────────────────────────────

    @objc public func generateStream(_ invoke: Invoke) throws {
        let args = try invoke.parseArgs(GenerateArgs.self)
        guard let loaded = getLmModel(args.modelId) else {
            invoke.reject("model not found: \(args.modelId)")
            return
        }
        let channel = args.channel
        // Mirror Kotlin: resolve immediately; tokens flow over the channel.
        invoke.resolve()

        Task {
            let start = DispatchTime.now()
            do {
                let conversation = try await loaded.engine.createConversation(
                    with: Self.conversationConfig(for: args))
                let message = Self.buildMessage(args)

                for try await chunk in conversation.sendMessageStream(message) {
                    try? channel?.send(StreamChunk(
                        modelId: args.modelId,
                        chunk: chunk.toString,
                        done: false,
                        latencyMs: nil
                    ))
                }

                let latencyMs =
                    Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000.0
                try? channel?.send(StreamChunk(
                    modelId: args.modelId,
                    chunk: "",
                    done: true,
                    latencyMs: latencyMs
                ))
            } catch {
                try? channel?.send(StreamChunk(
                    modelId: args.modelId,
                    chunk: "",
                    done: true,
                    latencyMs: nil,
                    error: "\(error)"
                ))
            }
        }
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    private static func conversationConfig(for args: GenerateArgs) -> ConversationConfig {
        let sampler = try? SamplerConfig(
            topK: args.sampler?.topK ?? 40,
            topP: args.sampler?.topP ?? 0.95,
            temperature: args.sampler?.temperature ?? 0.8
        )
        var systemMessage: Message? = nil
        if let sys = args.systemInstruction, !sys.isEmpty {
            systemMessage = Message(sys, role: .system)
        }
        return ConversationConfig(systemMessage: systemMessage, samplerConfig: sampler)
    }

    private static func buildMessage(_ args: GenerateArgs) -> Message {
        if let imageB64 = args.image, !imageB64.isEmpty,
           let data = Data(base64Encoded: imageB64) {
            return Message(of: .imageData(data), .text(args.prompt))
        }
        return Message(args.prompt)
    }
}

// ── C entry points expected by the Rust side ───────────────────────────────
// mobile.rs    → ios_plugin_binding!(init_plugin_litert)
// mobile_lm.rs → ios_plugin_binding!(init_plugin_litert_lm)
// Both command sets are served by the same class; each Rust handle routes only
// its own commands to its instance.

@_cdecl("init_plugin_litert")
func initPluginLitert() -> Plugin {
    return LiteRtPlugin()
}

@_cdecl("init_plugin_litert_lm")
func initPluginLitertLm() -> Plugin {
    return LiteRtPlugin()
}
