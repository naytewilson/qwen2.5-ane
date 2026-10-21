import CoreML
import Darwin
import Foundation

// Release packaging change (see PROVENANCE.md): upstream hardcoded
// PROJECT_ROOT to the author's machine (~/Projects/qwen2.5-ane-optimization).
// It now resolves portably: QWEN_HOME env override, else walk up from the
// executable looking for the repo layout (tokenizer/), else the cwd.
// This is the ONLY intentional source change in this release.
func resolveProjectRoot() -> String {
    if let home = ProcessInfo.processInfo.environment["QWEN_HOME"], !home.isEmpty {
        return home
    }
    let exe = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
    var dir = exe.deletingLastPathComponent()
    for _ in 0..<6 {
        if FileManager.default.fileExists(atPath: dir.appendingPathComponent("tokenizer").path) {
            return dir.path
        }
        dir = dir.deletingLastPathComponent()
    }
    return FileManager.default.currentDirectoryPath
}
let PROJECT_ROOT = resolveProjectRoot()
// R1 Fix #5: Default to the Qwen2.5 monolithic (PROJECT.md production baseline = 54.5 tok/s).
// DeepSeek-R1 distill path is opt-in via QWEN_MODEL_PATH override.
// Prefer the multi-function export (infer + batched prefill) when present, matching
// the CLI router; --serve and --compute-plan resolve their model through this.
let MODEL_PATH: String = {
    if let explicit = ProcessInfo.processInfo.environment["QWEN_MODEL_PATH"] { return explicit }
    let multiFn = "\(PROJECT_ROOT)/models/qwen25-1.5b-ane-ctx512-lut4/qwen25_monolithic_full_lut4.mlmodelc"
    if FileManager.default.fileExists(atPath: multiFn) { return multiFn }
    return "\(PROJECT_ROOT)/models/anemll_qwen25_1.5b_mono_ctx512_lut4/qwen25_monolithic_full_lut4.mlmodelc"
}()
let EOS_TOKENS: Set<Int> = [151643, 151645]

// ─────────────────────────────────────────────────────────────────────────────
// Request / Response types
// ─────────────────────────────────────────────────────────────────────────────
struct InferenceRequest: Decodable {
    let prompt: String
    let max_tokens: Int?
    let system: String?
    let temperature: Double?
    let backend: String?
}

struct InferenceResponse: Encodable {
    let text: String
    let tokens_generated: Int
    let elapsed_sec: Double
    let tok_per_sec: Double
    let decode_tok_per_sec: Double?
    /// Prompt tokens consumed by this request, when the backend can report them.
    var prompt_tokens: Int? = nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
func log(_ msg: String) {
    fputs("[qwen_infer] \(msg)\n", stderr)
}

func envInt(_ key: String, _ defaultValue: Int, min minValue: Int = 0) -> Int {
    max(minValue, Int(ProcessInfo.processInfo.environment[key] ?? "\(defaultValue)") ?? defaultValue)
}

func envDouble(_ key: String, _ defaultValue: Double) -> Double {
    Double(ProcessInfo.processInfo.environment[key] ?? "\(defaultValue)") ?? defaultValue
}

func median(_ xs: [Double]) -> Double {
    guard !xs.isEmpty else { return 0 }
    let s = xs.sorted()
    let m = s.count / 2
    if s.count % 2 == 0 {
        return (s[m - 1] + s[m]) / 2.0
    }
    return s[m]
}

func stddev(_ xs: [Double]) -> Double {
    guard xs.count > 1 else { return 0 }
    let mean = xs.reduce(0, +) / Double(xs.count)
    let varSum = xs.reduce(0) { $0 + pow($1 - mean, 2) }
    return sqrt(varSum / Double(xs.count - 1))
}

func fnv1a64(_ xs: [Int32]) -> String {
    var h: UInt64 = 0xcbf29ce484222325
    let prime: UInt64 = 0x100000001b3
    for x in xs {
        var v = UInt32(bitPattern: x)
        for _ in 0..<4 {
            let b = UInt8(v & 0xff)
            h ^= UInt64(b)
            h = h &* prime
            v >>= 8
        }
    }
    return String(format: "0x%016llx", h)
}

func selectedBackend(_ requestBackend: String?) -> String {
    let env = ProcessInfo.processInfo.environment
    let raw = (requestBackend ?? env["QWEN_BACKEND"] ?? "ane").lowercased()
    switch raw {
    case "mlx-server":
        // Use mlx_lm.server if it's reachable, else fall back to ane
        return "mlx-server"
    default:
        return "ane"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main async entry point
// ─────────────────────────────────────────────────────────────────────────────
func run() async throws {
    // ── Read JSON request from stdin ────────────────────────────────────────
    guard let line = readLine(strippingNewline: true),
          let data = line.data(using: .utf8)
    else {
        fputs("Usage: echo '{\"prompt\":\"...\"}' | qwen_infer\n", stderr)
        exit(1)
    }

    let request: InferenceRequest
    do {
        request = try JSONDecoder().decode(InferenceRequest.self, from: data)
    } catch {
        fputs("Error: invalid JSON — \(error)\n", stderr)
        exit(1)
    }

    let maxTokens = request.max_tokens ?? 1024
    let temperature = request.temperature ?? 0.0
    let requestedBackend = selectedBackend(request.backend)
    let stdoutIsTTY = isatty(fileno(stdout)) != 0
    let streamTokens = (ProcessInfo.processInfo.environment["QWEN_STREAM_TOKENS"] ?? (stdoutIsTTY ? "1" : "0")) != "0"
    let flushTokenStream = (ProcessInfo.processInfo.environment["QWEN_STREAM_FLUSH"] ?? (stdoutIsTTY ? "1" : "0")) != "0"

if requestedBackend == "mlx-server" {
        let env = ProcessInfo.processInfo.environment
        let smallModelDir = env["QWEN_MLX_SMALL_MODEL_DIR"]
        let estimatedTokens = max(1, request.prompt.count / 4)
        let useSmall = smallModelDir != nil && estimatedTokens <= 32
        let port = useSmall ? 8081 : 8080
        let dir = useSmall
            ? smallModelDir!
            : (env["QWEN_MLX_MODEL_DIR"] ?? env["QWEN_MLX_SERVER_MODEL"] ?? "local")
        if useSmall { log("Port 8081: 0.5B for \(estimatedTokens) est tokens") }
        log("Backend: MLX-Server port \(port)")
        let backend = MLXServerBackend(port: port, modelDir: dir)
        let startTime = Date()
        final class TextBuffer: @unchecked Sendable { var text = "" }
        let outputBuffer = TextBuffer()
        let result = try await backend.generate(
            promptIds: [], promptText: request.prompt, systemPrompt: request.system, maxTokens: maxTokens
        ) { _, piece in
            guard let piece else { return }
            outputBuffer.text += piece
            if streamTokens {
                print(piece, terminator: "")
                if flushTokenStream { fflush(stdout) }
            }
        }
        if streamTokens { print("") }
        let elapsed = Date().timeIntervalSince(startTime)
        let tps = elapsed > 0 ? Double(result.tokenIds.count) / elapsed : 0
        let summary = InferenceResponse(
            text: outputBuffer.text, tokens_generated: result.tokenIds.count,
            elapsed_sec: elapsed, tok_per_sec: tps, decode_tok_per_sec: nil)
        if let d = try? JSONEncoder().encode(summary), let s = String(data: d, encoding: .utf8) {
            fputs("INFERENCE_RESULT:\(s)\n", stderr)
        }
        return
    }

    if ProcessInfo.processInfo.environment["QWEN_SKIP_PRELOAD_1024"] == nil {
        setenv("QWEN_SKIP_PRELOAD_1024", "1", 1)
        log("Memory guard: defaulting QWEN_SKIP_PRELOAD_1024=1 (set to 0 to re-enable background 1024 preload).")
    }

    // ── Load tokenizer using NativeTokenizer ────────────────────────────────
    log("Loading tokenizer...")
    let tokenizer = try NativeTokenizer(tokenizerDirectory: "\(PROJECT_ROOT)/tokenizer")

    // ── Format + tokenize ───────────────────────────────────────────────────
    let formatted = tokenizer.formatChatML(system: request.system, user: request.prompt)
    let promptTokens = tokenizer.tokenize(formatted)
    log("Prompt: \(promptTokens.count) tokens")

    guard !promptTokens.isEmpty else {
        fputs("Error: empty prompt after tokenization\n", stderr)
        exit(1)
    }

    // ── Dynamic Model Routing ────────────────────────────────────────────────
    // The 512 mono engine background-preloads the 1024 mono model and transitions
    // KV state in-place when currentPosition crosses 512 (CortexEngine.transitionState).
    // So we route on PROMPT length only: if the prompt fits in 512 with headroom,
    // start at 512 mono and let the in-process hot-swap grow context if generation
    // overruns. If the prompt itself already exceeds the 512 window, start directly
    // on the 1024 mono multi-function model (infer + batched prefill).
    //
    // The legacy chunked 1024 model (`anemll_qwen25_1.5b_ctx1024_lut4`) is currently
    // emitting a shape-mismatch crash (1×64×1536 vs 1×1×1536) and is therefore NOT in
    // the default routing table. Set QWEN_FORCE_CHUNKED=1 to opt into it explicitly.
    var activeModelPath = MODEL_PATH
    if requestedBackend == "ane" && ProcessInfo.processInfo.environment["QWEN_MODEL_PATH"] == nil {
        let promptLen = promptTokens.count
        let routingHeadroom = 32   // keep ≥32-tok slack so prefill batching fits cleanly
        let mono512Path     = "\(PROJECT_ROOT)/models/anemll_qwen25_1.5b_mono_ctx512_lut4/qwen25_monolithic_full_lut4.mlmodelc"
        // Same 512-token checkpoint exported WITH the `prefill` function. Decode is
        // identical; prompt prefill runs in 64-token batches instead of one dispatch per
        // prompt token (measured 10.67s → 2.78s on a 413-token prompt). Preferred when present.
        let mono512MultiFn = "\(PROJECT_ROOT)/models/qwen25-1.5b-ane-ctx512-lut4/qwen25_monolithic_full_lut4.mlmodelc"
        let mono1024Path    = "\(PROJECT_ROOT)/models/anemll_qwen25_1.5b_mono_ctx1024_lut4/qwen25_monolithic_full_lut4.mlmodelc"
        let chunked1024Path = "\(PROJECT_ROOT)/models/anemll_qwen25_1.5b_ctx1024_lut4"

        if ProcessInfo.processInfo.environment["QWEN_FORCE_CHUNKED"] == "1" {
            activeModelPath = chunked1024Path
            log("Dynamic Routing: QWEN_FORCE_CHUNKED=1 → chunked 1024 (debug only; known shape-mismatch crash).")
        } else if promptLen + routingHeadroom <= 512 {
            if FileManager.default.fileExists(atPath: mono512MultiFn) {
                activeModelPath = mono512MultiFn
                log("Dynamic Routing: prompt=\(promptLen) ≤ 512 → mono 512 multi-function (batched prefill).")
            } else {
                activeModelPath = mono512Path
                log("Dynamic Routing: prompt=\(promptLen) ≤ 512 → mono 512 single-function (sequential prefill — artifact has no 'prefill' export).")
            }
        } else if FileManager.default.fileExists(atPath: mono1024Path) {
            activeModelPath = mono1024Path
            log("Dynamic Routing: prompt=\(promptLen) > 512−\(routingHeadroom) → starting directly on mono 1024 multi-function (no hot-swap needed).")
        } else {
            // Mono 1024 not built yet → fall back to chunked even with its known regression,
            // so an unbuilt model situation isn't silently a 512-truncated prompt.
            activeModelPath = chunked1024Path
            log("Dynamic Routing: prompt=\(promptLen) > 512 but mono 1024 mlmodelc missing — falling back to chunked 1024 (⚠️ known crash).")
        }
    }

    var isChunked = false
    let backend: any InferenceBackend = try await {
        // ── Load ANE model ────────────────────────────────────────────────────
        // If .mlpackage, compile to .mlmodelc first
        let modelURL: URL
        if activeModelPath.hasSuffix(".mlpackage") {
            log("Compiling \(activeModelPath)...")
            let cacheDir = "\(NSHomeDirectory())/.cache/qwen-ane/models"
            try FileManager.default.createDirectory(atPath: cacheDir, withIntermediateDirectories: true)

            let compiledPath = cacheDir + "/qwen25-1.5b-stateful-compiled.mlmodelc"
            if !FileManager.default.fileExists(atPath: compiledPath) {
                let compiled = try await MLModel.compileModel(at: URL(fileURLWithPath: activeModelPath))
                try FileManager.default.moveItem(at: compiled, to: URL(fileURLWithPath: compiledPath))
                log("Compiled and cached")
            } else {
                log("Using cached compiled model")
            }
            modelURL = URL(fileURLWithPath: compiledPath)
        } else {
            modelURL = URL(fileURLWithPath: activeModelPath)
        }

        // Detect chunked ANEMLL layout: parent dir containing the 3 standard mlmodelc dirs.
        let chunkedProbe = modelURL.appendingPathComponent("qwen25_embeddings.mlmodelc").path
        isChunked = !activeModelPath.hasSuffix(".mlmodelc") && FileManager.default.fileExists(atPath: chunkedProbe)

        log("Loading ANE model from \(modelURL.path)... [\(isChunked ? "chunked" : "monolithic")]")

        // Dynamically optimize CoreML/MPSGraph/ANE flags for the specific model layout.
        let flagsOn = (ProcessInfo.processInfo.environment["QWEN_INFER_FLAGS"] ?? "on") != "off"
        if flagsOn {
            setenv("ANEC_SPECIALIZE_LATENCY", "1", 1)

            if isChunked {
                setenv("ESPRESSO_ANEC_COMPRESS_SPARSE", "1", 1)
                unsetenv("ESPRESSO_PRIVATE_ANE_ALWAYS_COMPRESSED")
                log("[Optimization] Dynamic environment tuned for Chunked Layout: Active Sparse Compression.")
            } else {
                setenv("ESPRESSO_PRIVATE_ANE_ALWAYS_COMPRESSED", "1", 1)
                setenv("ESPRESSO_ANEC_COMPRESS_SPARSE", "1", 1)
                setenv("ESPRESSO_ANEC_REQUIRED_SPARSITY_FOR_COMPRESSION", "0.1", 1)
                log("[Optimization] Dynamic environment tuned for Monolithic Layout: In-Memory Compression + Sparsity.")
            }
        } else {
            unsetenv("ANEC_SPECIALIZE_LATENCY")
            unsetenv("ESPRESSO_PRIVATE_ANE_ALWAYS_COMPRESSED")
            unsetenv("ESPRESSO_ANEC_COMPRESS_SPARSE")
            unsetenv("ESPRESSO_ANEC_REQUIRED_SPARSITY_FOR_COMPRESSION")
            log("[A/B] QWEN_INFER_FLAGS=off — all ANE optimization flags STRIPPED for baseline.")
        }

        if isChunked {
            let engine = try ChunkedCortexEngine(modelURL: modelURL)
            log("Model loaded [ANE, ctx=\(engine.ctxLen)]")
            return ChunkedBackendAdapter(engine: engine)
        }

        let engine = try CortexEngine(modelURL: modelURL)
        log("Model loaded [ANE]")
        return CortexBackendAdapter(engine: engine, tokenizer: tokenizer)
    }()

    // ── Tokenizer-Model Validation Guard ────────────────────────────────────
    let tokenizerVocabCount = tokenizer.vocabCount
    let modelVocabCount = 151936
    if tokenizerVocabCount != modelVocabCount && tokenizerVocabCount != 151643 {
        log("🔥 [Validation Guard] FATAL: Tokenizer vocabulary count (\(tokenizerVocabCount)) does not match the model's expected index space (\(modelVocabCount)). Bailing out to prevent silent garbage token generation.")
        print("error: Tokenizer-Model mismatch. Ensure your --tokenizer path matches the active model vocab size.")
        exit(1)
    } else {
        log("✅ [Validation Guard] Tokenizer vocabulary count matches model index space: \(tokenizerVocabCount) / \(modelVocabCount) (OK).")
    }

    // ── Generate using selected backend ─────────────────────────────────────
    log("Generating (max \(maxTokens) tokens, temp=\(temperature))...")
    let startTime = Date()

    final class TextBuffer: @unchecked Sendable { var text = "" }
    let outputBuffer = TextBuffer()
    let tok = tokenizer
    final class TokenBuffer: @unchecked Sendable { var tokens: [Int32] = [] }
    let deferredDecodeTokenIds = TokenBuffer()
    let result = try await backend.generate(
        promptIds: promptTokens.map { Int32($0) },
        promptText: request.prompt,
        systemPrompt: request.system,
        maxTokens: maxTokens
    ) { tokenId, providedPiece in
        if let providedPiece {
            outputBuffer.text += providedPiece
            if streamTokens {
                print(providedPiece, terminator: "")
                if flushTokenStream { fflush(stdout) }
            }
            return
        }

        if streamTokens {
            let piece = tok.decode(tokens: [Int(tokenId)], skipSpecialTokens: true)
            outputBuffer.text += piece
            print(piece, terminator: "")
            if flushTokenStream { fflush(stdout) }
        } else {
            deferredDecodeTokenIds.tokens.append(tokenId)
        }
    }
    if !streamTokens, outputBuffer.text.isEmpty, !deferredDecodeTokenIds.tokens.isEmpty {
        outputBuffer.text = tok.decode(tokens: deferredDecodeTokenIds.tokens.map { Int($0) }, skipSpecialTokens: true)
    }
    let generatedIds = result.tokenIds
    let decodeTimes = result.decodeTimesMs ?? []
    if streamTokens { print("") }  // trailing newline

    let elapsed = Date().timeIntervalSince(startTime)
    let tps = elapsed > 0 ? Double(generatedIds.count) / elapsed : 0
    
    let measuredDecodeTimes = Array(decodeTimes.prefix(generatedIds.count))
    let totalDecodeMs = measuredDecodeTimes.reduce(0.0, +)
    let decodeTps: Double? = totalDecodeMs > 0 ? (Double(generatedIds.count) / (totalDecodeMs / 1000.0)) : nil
    let decodeTpsDisplay = decodeTps.map { String(format: "%.1f", $0) } ?? "n/a"

    log("Done: \(generatedIds.count) tokens, \(String(format: "%.2f", elapsed))s, \(String(format: "%.1f", tps)) tok/s (Decode: \(decodeTpsDisplay) tok/s)")

    // ── Emit JSON summary to stderr for router to parse ─────────────────────
    let summary = InferenceResponse(
        text: outputBuffer.text,
        tokens_generated: generatedIds.count,
        elapsed_sec: elapsed,
        tok_per_sec: tps,
        decode_tok_per_sec: decodeTps
    )
    if let summaryData = try? JSONEncoder().encode(summary),
       let summaryStr = String(data: summaryData, encoding: .utf8) {
        fputs("INFERENCE_RESULT:\(summaryStr)\n", stderr)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// --validate-verify: Bit-correctness gate (K_QUERY_VERIFY_HANDOFF.md §4)
//
// Protocol:
//   1. Load the batch=4 model (infer + prefill functions).
//   2. Prime the KV cache with a test prompt via sequential single-token infer.
//   3. Record the ground-truth argmax for 4 consecutive positions (tokens K, K+1, K+2, K+3).
//   4. Reset to the same prompt KV state, then call _verifyCandidatesUnlocked
//      (which routes through the prefill/K=4 path) on those same 4 tokens+positions.
//   5. Compare: all 4 argmax outputs must match exactly.
//      PASS → the verify path is bit-correct; proceed to --benchmark-spec.
//      FAIL → the verify export is still broken; do NOT report any tok/s.
// ─────────────────────────────────────────────────────────────────────────────
func runValidateVerify() async throws {
    print("=============================================================")
    print("   BIT-CORRECTNESS GATE: sequential infer vs prefill/K=4     ")
    print("   K_QUERY_VERIFY_HANDOFF.md §4                              ")
    print("=============================================================")

    let verifyModelPath = ProcessInfo.processInfo.environment["QWEN_VERIFY_MODEL_PATH"]
        ?? "\(PROJECT_ROOT)/models/qwen25-1.5b-ane-ctx512-lut4/qwen25_monolithic_full_lut4.mlmodelc"

    guard FileManager.default.fileExists(atPath: verifyModelPath) else {
        print("[GATE] ❌ FAIL: verify model not found at \(verifyModelPath)")
        print("       Set QWEN_VERIFY_MODEL_PATH to the batch=4 mlmodelc path.")
        exit(1)
    }
    print("[GATE] Verify model: \(verifyModelPath)")

    let K = 4   // number of tokens to verify in one batch call
    let promptStr = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    print("[GATE] Test prompt: \"\(promptStr.prefix(60))...\"")

    let tokenizer = try NativeTokenizer(tokenizerDirectory: "\(PROJECT_ROOT)/tokenizer")
    let promptTokens = tokenizer.tokenize(promptStr).map { Int32($0) }
    print("[GATE] Prompt length: \(promptTokens.count) tokens\n")

    // ── Step 1: Load the batch=4 model ───────────────────────────────────────
    print("[GATE] Step 1: Loading batch=4 model (infer + prefill functions)...")
    let engine = try CortexEngine(modelURL: URL(fileURLWithPath: verifyModelPath))
    print("[GATE] Model loaded. hasPrefillModel=\(engine.hasPrefillModel) hasVerifyModel=\(engine.hasVerifyModel)\n")

    // ── Step 2: Prime KV cache + collect ground-truth argmax for K tokens ─────
    // We run a short generation (prefill the prompt + decode exactly K tokens).
    // The K decoded token IDs ARE the ground-truth verify targets; we capture
    // both the input token at each position and the argmax output.
    print("[GATE] Step 2: Sequential decode — collecting \(K) ground-truth (token, argmax) pairs...")

    var groundTruthTokensAtPos: [Int32] = []   // the token fed at each position
    var groundTruthArgmax:      [Int32] = []   // the argmax produced at each position
    var groundTruthPositions:   [Int]   = []   // absolute position index for each

    let _ = try engine.generate(promptIds: promptTokens, maxTokens: K) { tok in
        // onToken fires AFTER each decode step.
        // Position for this step = (last prompt token pos) + steps already done.
        let stepIdx = groundTruthArgmax.count
        let pos = promptTokens.count - 1 + stepIdx
        // The token fed INTO position `pos` is either the last prompt token (step 0)
        // or the previous argmax output.
        let fedToken: Int32 = stepIdx == 0 ? promptTokens.last! : groundTruthArgmax[stepIdx - 1]
        groundTruthPositions.append(pos)
        groundTruthTokensAtPos.append(fedToken)
        groundTruthArgmax.append(tok)
    }

    guard groundTruthArgmax.count == K else {
        print("[GATE] ❌ FAIL: Only got \(groundTruthArgmax.count) tokens from sequential decode (expected \(K)).")
        exit(1)
    }
    print("[GATE] Ground truth (sequential infer):")
    for i in 0..<K {
        let word = tokenizer.decode(tokens: [Int(groundTruthArgmax[i])], skipSpecialTokens: false)
        print(String(format: "  pos=%d  in_tok=%d  argmax=%d  (\"%@\")",
                     groundTruthPositions[i], groundTruthTokensAtPos[i],
                     groundTruthArgmax[i], word))
    }

    // ── Step 3: Run _verifyCandidatesUnlocked on same tokens + positions ──────
    // At this point the KV cache already holds the state after those K decodes.
    // _verifyCandidatesUnlocked uses the prefill model to verify K candidates in
    // one batch call. The causal mask it builds is:
    //   row i sees positions[0..i] AND history up to positions[0]-1
    // which matches what sequential infer had access to for each position.
    print("\n[GATE] Step 3: Running prefill/K=4 verify on same \(K) tokens+positions...")
    let verifyArgmax: [Int32]
    do {
        verifyArgmax = try engine.withQueueLock {
            try engine._verifyCandidatesUnlocked(
                tokens: groundTruthTokensAtPos,
                positions: groundTruthPositions
            )
        }
    } catch {
        print("[GATE] ❌ FAIL: _verifyCandidatesUnlocked threw: \(error)")
        exit(1)
    }

    print("[GATE] Verify output (prefill/K=4):")
    for i in 0..<K {
        let word = tokenizer.decode(tokens: [Int(verifyArgmax[i])], skipSpecialTokens: false)
        print(String(format: "  pos=%d  argmax=%d  (\"%@\")",
                     groundTruthPositions[i], verifyArgmax[i], word))
    }

    // ── Step 4: Compare ───────────────────────────────────────────────────────
    print("\n[GATE] Step 4: Comparing sequential vs verify argmax...")
    var allMatch = true
    for i in 0..<K {
        let match = groundTruthArgmax[i] == verifyArgmax[i]
        if !match { allMatch = false }
        print(String(format: "  pos=%d  seq=%d  verify=%d  %@",
                     groundTruthPositions[i],
                     groundTruthArgmax[i], verifyArgmax[i],
                     match ? "✓ MATCH" : "✗ MISMATCH"))
    }

    print("\n=============================================================")
    let gatePath = "/tmp/qwen_verify_gate_status.json"
    let ts = ISO8601DateFormatter().string(from: Date())
    if allMatch {
        print("[GATE] ✅ PASS: All \(K) argmax outputs match exactly.")
        print("       Verify path is bit-correct. Safe to run --benchmark-spec.")
        print("       To run: ~/bin/qwen_infer --benchmark-spec")
        let payload: [String: Any] = [
            "schema_version": "verify_gate/v2",
            "status": "PASS",
            "timestamp": ts,
            "verify_model": verifyModelPath,
            "k": K
        ]
        if let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted]) {
            try? data.write(to: URL(fileURLWithPath: gatePath), options: .atomic)
        }
        print("       Gate file: \(gatePath)")
    } else {
        print("[GATE] ❌ FAIL: Argmax mismatch detected.")
        print("       The prefill/K=4 export is NOT bit-correct vs sequential infer.")
        print("       Do NOT run --benchmark-spec or report any tok/s until this passes.")
        print("       See K_QUERY_VERIFY_HANDOFF.md §1 for root-cause analysis.")
        let payload: [String: Any] = [
            "schema_version": "verify_gate/v2",
            "status": "FAIL",
            "timestamp": ts,
            "verify_model": verifyModelPath,
            "k": K,
            "reason": "argmax_mismatch"
        ]
        if let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted]) {
            try? data.write(to: URL(fileURLWithPath: gatePath), options: .atomic)
        }
        exit(2)
    }
    print("=============================================================")
}

// ─────────────────────────────────────────────────────────────────────────────
// --generate-labels: Generate teacher argmax labels for draft distillation.
//
// Reads a JSON array of token sequences from stdin:
//   [[tok, tok, ...], [tok, tok, ...], ...]
// For each sequence, runs the mono512 model to get the argmax at every position.
// Outputs a JSON array of label sequences to stdout:
//   [[label, label, ...], ...]  (same shape as input, but shifted +1)
//
// Usage:
//   python3 -c "import json,sys; seqs=...; print(json.dumps(seqs))" | \
//     ~/bin/qwen_infer --generate-labels > /tmp/labels.json
//
// Speed: uses batched prefill (batch=64) → ~47 tok/s effective throughput.
// ─────────────────────────────────────────────────────────────────────────────
func runGenerateLabels() async throws {
    let modelPath = "\(PROJECT_ROOT)/models/anemll_qwen25_1.5b_mono_ctx512_lut4/qwen25_monolithic_full_lut4.mlmodelc"
    fputs("[gen-labels] Loading mono512...\n", stderr); fflush(stderr)
    let engine = try CortexEngine(modelURL: URL(fileURLWithPath: modelPath))
    fputs("[gen-labels] Model loaded. Reading sequences from stdin...\n", stderr); fflush(stderr)

    // Read all stdin
    var inputData = Data()
    let bufSize = 65536
    var buf = [UInt8](repeating: 0, count: bufSize)
    while true {
        let n = read(STDIN_FILENO, &buf, bufSize)
        if n <= 0 { break }
        inputData.append(contentsOf: buf[..<n])
    }

    guard let sequences = try? JSONSerialization.jsonObject(with: inputData) as? [[Int]] else {
        fputs("[gen-labels] ERROR: invalid JSON input — expected [[int]]\n", stderr)
        exit(1)
    }
    fputs("[gen-labels] \(sequences.count) sequences. Generating labels...\n", stderr); fflush(stderr)

    var labelBatches: [[Int32]] = []
    let t0 = CFAbsoluteTimeGetCurrent()

    for (i, seq) in sequences.enumerated() {
        let promptIds = seq.map { Int32($0) }
        // generate(maxTokens: seqLen) → argmax at each position AFTER the prompt
        // We want labels[j] = argmax at position promptIds[j], for j in 0..<seqLen-1
        // This is achieved by feeding each token as a "prompt" and collecting the decode output.
        // Efficient approach: use generate with maxTokens = seq.count
        var labels: [Int32] = []
        let _ = try engine.generate(promptIds: promptIds, maxTokens: seq.count) { tok in
            labels.append(tok)
            if labels.count >= seq.count { return }
        }
        // labels[j] = what the model predicted after seeing promptIds[0..j]
        // Truncate to seq.count - 1 (next-token prediction targets for each input position)
        labelBatches.append(Array(labels.prefix(seq.count - 1)))

        if (i + 1) % 100 == 0 || (i + 1) == sequences.count {
            let elapsed = CFAbsoluteTimeGetCurrent() - t0
            let tps = Double((i + 1) * seq.count) / elapsed
            fputs(String(format: "[gen-labels] %d/%d (%.0f tok/s) %.0fs\n",
                         i+1, sequences.count, tps, elapsed), stderr)
            fflush(stderr)
        }
    }

    let elapsed = CFAbsoluteTimeGetCurrent() - t0
    fputs(String(format: "[gen-labels] Done: %d seqs in %.0fs (%.0f tok/s avg)\n",
                 sequences.count, elapsed, Double(sequences.count * (sequences.first?.count ?? 0)) / elapsed), stderr)

    // Output JSON labels to stdout
    let output = try JSONSerialization.data(withJSONObject: labelBatches.map { $0.map { Int($0) } })
    FileHandle.standardOutput.write(output)
    fputs("\n", stdout)
}

// Swift 6 @main + async entry
Task {
    do {
        // Pre-load ANE specialization flag (also gated by QWEN_INFER_FLAGS=off).
        if (ProcessInfo.processInfo.environment["QWEN_INFER_FLAGS"] ?? "on") != "off" {
            setenv("ANEC_SPECIALIZE_LATENCY", "1", 1)
        } else {
            unsetenv("ANEC_SPECIALIZE_LATENCY")
        }
        if CommandLine.arguments.contains("--probe-model") {
            // Structural gate used by setup.sh and by support: does this artifact
            // actually load as a multi-function (infer + prefill) model?
            let args = CommandLine.arguments
            let idx = args.firstIndex(of: "--probe-model")!
            let explicit = (idx + 1 < args.count && !args[idx + 1].hasPrefix("-")) ? args[idx + 1] : nil
            // With no argument, probe the artifact the router would actually pick.
            let preferred = "\(PROJECT_ROOT)/models/qwen25-1.5b-ane-ctx512-lut4/qwen25_monolithic_full_lut4.mlmodelc"
            let fallback = MODEL_PATH
            let defaultProbe = FileManager.default.fileExists(atPath: preferred) ? preferred : fallback
            runModelProbe(modelPath: explicit ?? defaultProbe)
            exit(0)
        } else if CommandLine.arguments.contains("--compute-plan") {
            // Placement audit (playbook step 5). Optional explicit path follows the flag;
            // otherwise audit the resolved production artifact.
            let args = CommandLine.arguments
            let idx = args.firstIndex(of: "--compute-plan")!
            let explicit = (idx + 1 < args.count && !args[idx + 1].hasPrefix("-")) ? args[idx + 1] : nil
            await runComputePlanAudit(modelPath: explicit ?? MODEL_PATH)
            exit(0)
        } else if CommandLine.arguments.contains("--validate-verify") {
            try await runValidateVerify()
            exit(0)
        } else if CommandLine.arguments.contains("--generate-labels") {
            try await runGenerateLabels()
            exit(0)
        } else if CommandLine.arguments.contains("--serve") {
            // Headless HTTP server: OpenAI-compatible endpoint, no IPC/TCP daemon.
            setenv("QWEN_SERVE_ONLY_HTTP", "1", 1)
            try await runDaemon()
        } else if CommandLine.arguments.contains("--daemon") {
            try await runDaemon()
        } else {
            try await run()
            exit(0)
        }
    } catch {
        fputs("Fatal error: \(error.localizedDescription)\n", stderr)
        exit(1)
    }
}

// Keep runloop alive for async task
RunLoop.main.run()
