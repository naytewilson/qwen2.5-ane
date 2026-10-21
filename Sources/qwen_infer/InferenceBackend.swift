import Foundation

struct BackendGenerationResult: Sendable {
    let tokenIds: [Int32]
    let decodeTimesMs: [Double]?
    let includesPrompt: Bool
}

protocol InferenceBackend: AnyObject, Sendable {
    var backendName: String { get }

    func generate(
        promptIds: [Int32],
        promptText: String,
        systemPrompt: String?,
        maxTokens: Int,
        onToken: @escaping @Sendable (Int32, String?) -> Void
    ) async throws -> BackendGenerationResult
}

// ─────────────────────────────────────────────────────────────────────────────
// MLX-Server backend — proxies to mlx_lm.server OpenAI-compatible endpoint.
// Fastest path: continuous batching, NAX-enabled, 30+ tok/s with 4-way concurrency.
// Start server: mlx_lm.server --model <path> --port 8080 --decode-concurrency 4
// ─────────────────────────────────────────────────────────────────────────────
final class MLXServerBackend: InferenceBackend, @unchecked Sendable {
    let backendName = "mlx-server"

    private let baseURL: String
    private let modelId: String

    init() {
        let env = ProcessInfo.processInfo.environment
        self.baseURL = env["QWEN_MLX_SERVER_URL"] ?? "http://localhost:8080"
        self.modelId = env["QWEN_MLX_SERVER_MODEL"]
            ?? env["QWEN_MLX_MODEL_DIR"]
            ?? "local"
    }

    init(port: Int, modelDir: String) {
        self.baseURL = "http://localhost:\(port)"
        self.modelId = modelDir
    }

    func generate(
        promptIds: [Int32],
        promptText: String,
        systemPrompt: String?,
        maxTokens: Int,
        onToken: @escaping @Sendable (Int32, String?) -> Void
    ) async throws -> BackendGenerationResult {
        let system = systemPrompt ?? "You are a helpful assistant."
        let messages: [[String: String]] = [
            ["role": "system", "content": system],
            ["role": "user", "content": promptText]
        ]
        let body: [String: Any] = [
            "model": modelId,
            "messages": messages,
            "max_tokens": maxTokens,
            "temperature": 0,
            "stream": true
        ]

        let url = URL(string: "\(baseURL)/v1/chat/completions")!
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: body)

        let startTime = CFAbsoluteTimeGetCurrent()
        var generated = [Int32]()
        var decodeTimes = [Double]()
        var fullText = ""

        let (asyncBytes, response) = try await URLSession.shared.bytes(for: req)
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw NSError(domain: "MLXServer", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "mlx_lm.server returned non-200"])
        }

        for try await line in asyncBytes.lines {
            guard line.hasPrefix("data: "), line != "data: [DONE]" else { continue }
            let jsonStr = String(line.dropFirst(6))
            guard let data = jsonStr.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let choices = obj["choices"] as? [[String: Any]],
                  let delta = choices.first?["delta"] as? [String: Any],
                  let piece = delta["content"] as? String,
                  !piece.isEmpty
            else { continue }

            fullText += piece
            // Approximate: emit one token per chunk (mlx-lm streams word-pieces)
            let fakeTokenId = Int32(generated.count + 1)
            generated.append(fakeTokenId)
            let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
            decodeTimes.append(elapsed / Double(generated.count))
            onToken(fakeTokenId, piece)
        }

        return BackendGenerationResult(tokenIds: generated, decodeTimesMs: decodeTimes, includesPrompt: false)
    }
}

final class CortexBackendAdapter: InferenceBackend, @unchecked Sendable {
    let backendName = "ane"
    private let engine: CortexEngine
    private let tokenizer: NativeTokenizer

    init(engine: CortexEngine, tokenizer: NativeTokenizer) {
        self.engine = engine
        self.tokenizer = tokenizer
    }

    func generate(
        promptIds: [Int32],
        promptText _: String,
        systemPrompt _: String?,
        maxTokens: Int,
        onToken: @escaping @Sendable (Int32, String?) -> Void
    ) async throws -> BackendGenerationResult {
        let env = ProcessInfo.processInfo.environment
        let enforcer: FormatEnforcer? = env["QWEN_JSON_MODE"] == "1"
            ? FormatEnforcer.json(tokenizer: tokenizer)
            : nil
        let result = try engine.generate(promptIds: promptIds, maxTokens: maxTokens, formatEnforcer: enforcer) { tokenId in
            onToken(tokenId, nil)
        }
        return BackendGenerationResult(tokenIds: result.0, decodeTimesMs: result.1, includesPrompt: false)
    }
}

final class ChunkedBackendAdapter: InferenceBackend, @unchecked Sendable {
    let backendName = "ane"
    private let engine: ChunkedCortexEngine

    init(engine: ChunkedCortexEngine) {
        self.engine = engine
    }

    func generate(
        promptIds: [Int32],
        promptText _: String,
        systemPrompt _: String?,
        maxTokens: Int,
        onToken: @escaping @Sendable (Int32, String?) -> Void
    ) async throws -> BackendGenerationResult {
        let result = try engine.generate(promptIds: promptIds, maxTokens: maxTokens) { tokenId in
            onToken(tokenId, nil)
        }
        return BackendGenerationResult(tokenIds: result.0, decodeTimesMs: result.1, includesPrompt: false)
    }
}

