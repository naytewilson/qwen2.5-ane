import Foundation
@preconcurrency import Network
@preconcurrency import CoreML
import Dispatch

// ─────────────────────────────────────────────────────────────────────────────
// DaemonServer — ANE Inference Server on port 45281
//
// Wire protocol (external client compatibility):
//   Request  → [0x00][UInt32 BE len][JSON: InferenceRequest]
//   Response → [0x01][UInt32 BE len][token piece]  (repeated)
//              [0x02][UInt32 BE len][INFERENCE_RESULT:{...}]
//
// Backend is loaded ONCE at startup (NativeTokenizer only for ANE). Inference is serialized
// through InferenceActor to keep CortexEngine single-threaded.
//
// DESIGN CONSTRAINT: Under MPSGRAPH_ENABLE_ANE_SYNC_RUN=1, the Apple Neural
// Engine execution synchronously blocks the calling thread during inference.
// Because CortexEngine is not thread-safe and sync-run ties up the calling thread,
// all inference requests are strictly serialized and queued via InferenceActor
// to run one at a time. This prevents concurrent model corruption and minimizes
// thread pool/head-of-line blocking under load.
// ─────────────────────────────────────────────────────────────────────────────

// Serializes all inference requests — CortexEngine is not thread-safe.
actor InferenceActor {
    let backend: any InferenceBackend
    let tokenizer: NativeTokenizer?

    init(backend: any InferenceBackend, tokenizer: NativeTokenizer?) {
        self.backend = backend
        self.tokenizer = tokenizer
    }

    func infer(
        request: InferenceRequest,
        streamTokens: Bool,
        onToken: @escaping @Sendable (String) -> Void
    ) async throws -> InferenceResponse {
        var maxTokens = request.max_tokens ?? 128
        if maxTokens > 256 {
            maxTokens = 256
        }

        let enrichedSystem = request.system ?? "You are a helpful assistant."
        var alignedTokens: [Int] = []
        if backend.backendName == "ane" {
            guard let tokenizer else {
                throw NSError(domain: "qwen_infer", code: 2, userInfo: [NSLocalizedDescriptionKey: "ANE backend missing tokenizer"])
            }
            // Local Dynamic Prompt Injection (DPI) disabled to avoid prefill latency.
            let formatted = tokenizer.formatChatML(system: enrichedSystem, user: request.prompt)
            let promptTokens = tokenizer.tokenize(formatted)

            guard !promptTokens.isEmpty else {
                throw NSError(domain: "qwen_infer", code: 1, userInfo: [NSLocalizedDescriptionKey: "empty prompt"])
            }

            log("[Daemon] Prompt: \(promptTokens.count) tokens")

            // Dynamic truncation to fit perfectly within the 512-context monolithic limit
            alignedTokens = promptTokens
            let maxAllowedPrompt = 512 - maxTokens - 8
            if alignedTokens.count > maxAllowedPrompt {
                log("[Daemon] Prompt exceeds maximum allowed tokens (\(promptTokens.count) > \(maxAllowedPrompt)). Truncating to keep the most recent \(maxAllowedPrompt) tokens.")
                alignedTokens = Array(alignedTokens.suffix(maxAllowedPrompt))
            }
        }

        // Thread-safe buffers — generate callback runs on a background thread
        final class TokenBuffer: @unchecked Sendable {
            private let lock = NSLock()
            private var text = ""
            func append(_ s: String) {
                lock.lock()
                text += s
                lock.unlock()
            }
            func set(_ s: String) {
                lock.lock()
                text = s
                lock.unlock()
            }
            func isEmpty() -> Bool {
                lock.lock()
                let empty = text.isEmpty
                lock.unlock()
                return empty
            }
            func snapshot() -> String {
                lock.lock()
                let copy = text
                lock.unlock()
                return copy
            }
        }
        final class TokenIdBuffer: @unchecked Sendable {
            private let lock = NSLock()
            private var ids: [Int32] = []
            func append(_ id: Int32) {
                lock.lock()
                ids.append(id)
                lock.unlock()
            }
            func snapshot() -> [Int32] {
                lock.lock()
                let copy = ids
                lock.unlock()
                return copy
            }
        }
        let buf = TokenBuffer()
        let tok = self.tokenizer
        let emittedTokenIds = TokenIdBuffer()
        let decodeBatchSize = max(1, Int(ProcessInfo.processInfo.environment["QWEN_DAEMON_DECODE_BATCH"] ?? "4") ?? 4)

        final class PendingTokenDecoder: @unchecked Sendable {
            private let lock = NSLock()
            private var pending: [Int32] = []
            private let tokenizer: NativeTokenizer?
            init(tokenizer: NativeTokenizer?) { self.tokenizer = tokenizer }
            func append(_ id: Int32, batchSize: Int) -> String? {
                lock.lock()
                pending.append(id)
                if pending.count < batchSize {
                    lock.unlock()
                    return nil
                }
                let ids = pending
                pending.removeAll(keepingCapacity: true)
                lock.unlock()
                return tokenizer?.decode(tokens: ids.map { Int($0) }, skipSpecialTokens: true)
            }
            func flush() -> String {
                lock.lock()
                let ids = pending
                pending.removeAll(keepingCapacity: true)
                lock.unlock()
                guard !ids.isEmpty else { return "" }
                return tokenizer?.decode(tokens: ids.map { Int($0) }, skipSpecialTokens: true) ?? ""
            }
        }
        let pendingDecoder = PendingTokenDecoder(tokenizer: tok)

        let startTime = Date()
        let generated = try await backend.generate(
            promptIds: alignedTokens.map { Int32($0) },
            promptText: request.prompt,
            systemPrompt: enrichedSystem,
            maxTokens: maxTokens
        ) { tokenId, providedPiece in
            if let providedPiece {
                if streamTokens {
                    let flushed = pendingDecoder.flush()
                    if !flushed.isEmpty {
                        buf.append(flushed)
                        onToken(flushed)
                    }
                    if !providedPiece.isEmpty {
                        buf.append(providedPiece)
                        onToken(providedPiece)
                    }
                }
                return
            }

            if streamTokens {
                let decoded = pendingDecoder.append(tokenId, batchSize: decodeBatchSize) ?? ""
                if !decoded.isEmpty {
                    buf.append(decoded)
                    onToken(decoded)
                }
            } else {
                emittedTokenIds.append(tokenId)
            }
        }

        if streamTokens {
            let tail = pendingDecoder.flush()
            if !tail.isEmpty {
                buf.append(tail)
                onToken(tail)
            }
        }

        let emitted = emittedTokenIds.snapshot()
        if !streamTokens, buf.isEmpty(), !emitted.isEmpty {
            let decoded = tok?.decode(tokens: emitted.map { Int($0) }, skipSpecialTokens: true) ?? ""
            buf.set(decoded)
        }

        let elapsed = Date().timeIntervalSince(startTime)
        let generatedCount = generated.includesPrompt
            ? max(0, generated.tokenIds.count - alignedTokens.count)
            : generated.tokenIds.count
        let tps = elapsed > 0 ? Double(generatedCount) / elapsed : 0

        // Report the mode actually in use. The production ANE path is a plain single-token
        // decode loop; only an explicitly configured EAGLE head makes it speculative.
        let eagleEnabled = (ProcessInfo.processInfo.environment["QWEN_EAGLE_HEAD_PATH"] ?? "").isEmpty == false
        let mode: String
        if backend.backendName == "ane" {
            mode = eagleEnabled ? "Speculative Decode" : "ANE Decode"
        } else {
            mode = "MLX Decode"
        }
        log("[Daemon] Done: \(generatedCount) tokens, \(String(format: "%.1f", tps)) tok/s (\(mode))")

        return InferenceResponse(
            text: buf.snapshot(),
            tokens_generated: generatedCount,
            elapsed_sec: elapsed,
            tok_per_sec: tps,
            decode_tok_per_sec: tps,
            prompt_tokens: alignedTokens.count
        )
    }

    // Direct IPC call - fully bypasses tokenization and string building.
    // Serialized by the actor to prevent concurrency with TCP clients.
    func inferRaw(
        promptIds: [Int32],
        maxTokens: Int,
        onToken: @escaping @Sendable (Int32) -> Void
    ) async throws {
        _ = try await backend.generate(
            promptIds: promptIds,
            promptText: "[IPC_PROMPT]", // Fast path skips swift-side tokenizer
            systemPrompt: nil,
            maxTokens: maxTokens
        ) { tokenId, _ in
            onToken(tokenId)
        }
    }
}

final class ConnectionBatchSender: @unchecked Sendable {
    private let connection: NWConnection
    private let queue: DispatchQueue
    private let batchSize: Int
    private let lock = NSLock()
    private var buffered: [Data] = []

    init(connection: NWConnection, queueLabel: String, batchSize: Int) {
        self.connection = connection
        self.queue = DispatchQueue(label: queueLabel)
        self.batchSize = max(1, batchSize)
    }

    func sendRaw(_ data: Data) {
        queue.async {
            self.connection.send(content: data, completion: .idempotent)
        }
    }

    func enqueue(_ data: Data) {
        var payload: Data?
        lock.lock()
        buffered.append(data)
        if buffered.count >= batchSize {
            payload = drainLocked()
        }
        lock.unlock()
        if let payload {
            sendRaw(payload)
        }
    }

    func flush() {
        var payload: Data?
        lock.lock()
        payload = drainLocked()
        lock.unlock()
        if let payload {
            sendRaw(payload)
        }
    }

    func sendAndClose(_ data: Data) {
        flush()
        queue.async {
            self.connection.send(content: data, completion: .contentProcessed({ _ in
                self.connection.cancel()
            }))
        }
    }

    func drainLocked() -> Data? {
        guard !buffered.isEmpty else { return nil }
        var payload = Data()
        payload.reserveCapacity(buffered.reduce(0) { $0 + $1.count })
        for frame in buffered {
            payload.append(frame)
        }
        buffered.removeAll(keepingCapacity: true)
        return payload
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Frame helpers
// ─────────────────────────────────────────────────────────────────────────────

func makeFrame(type: UInt8, text: String) -> Data {
    let payload = text.data(using: .utf8) ?? Data()
    var frame = Data()
    frame.append(type)
    var len = UInt32(payload.count).bigEndian
    withUnsafeBytes(of: &len) { frame.append(contentsOf: $0) }
    frame.append(payload)
    return frame
}

// ─────────────────────────────────────────────────────────────────────────────
// Connection handler
// ─────────────────────────────────────────────────────────────────────────────

func handleConnection(_ connection: NWConnection, actor inferActor: InferenceActor) {
    connection.start(queue: .global(qos: .userInitiated))

    // Read 5-byte header
    connection.receive(minimumIncompleteLength: 5, maximumLength: 5) { headerData, _, _, error in
        guard let headerData = headerData, headerData.count == 5, error == nil else {
            log("[Daemon] Header read error: \(error?.localizedDescription ?? "incomplete")")
            connection.cancel()
            return
        }

        // type byte (0 = request) — we accept any type
        let payloadLen = (UInt32(headerData[1]) << 24) |
                         (UInt32(headerData[2]) << 16) |
                         (UInt32(headerData[3]) << 8)  |
                          UInt32(headerData[4])

        guard payloadLen > 0 && payloadLen < 1_000_000 else {
            log("[Daemon] Invalid payload length: \(payloadLen)")
            connection.cancel()
            return
        }

        // Read JSON payload
        connection.receive(minimumIncompleteLength: Int(payloadLen), maximumLength: Int(payloadLen)) { bodyData, _, _, error in
            struct DecodableRequest: Decodable {
                let prompt: String
                let system: String?
                let max_tokens: Int?
                let backend: String?
            }
            
            guard let bodyData = bodyData, error == nil,
                  let rawReq = try? JSONDecoder().decode(DecodableRequest.self, from: bodyData)
            else {
                log("[Daemon] JSON decode error")
                connection.cancel()
                return
            }

            let request = InferenceRequest(
                prompt: rawReq.prompt,
                max_tokens: rawReq.max_tokens,
                system: rawReq.system,
                temperature: nil,
                backend: rawReq.backend
            )
            
            log("[Daemon] Request: \"\(String(request.prompt.prefix(60)))\"")

            // Run inference on actor (serialized)
            Task {
                do {
                    let streamTokens = (ProcessInfo.processInfo.environment["QWEN_DAEMON_STREAM_TOKENS"] ?? "1") != "0"
                    let batchSize = max(1, Int(ProcessInfo.processInfo.environment["QWEN_DAEMON_STREAM_BATCH"] ?? "1") ?? 1)
                    let sender = ConnectionBatchSender(
                        connection: connection,
                        queueLabel: "qwen.daemon.send.\(UUID().uuidString)",
                        batchSize: batchSize
                    )

                    let response = try await inferActor.infer(request: request, streamTokens: streamTokens) { tokenPiece in
                        if streamTokens {
                            sender.enqueue(makeFrame(type: 1, text: tokenPiece))
                        }
                    }

                    // Final frame
                    if let summaryData = try? JSONEncoder().encode(response),
                       let summaryStr = String(data: summaryData, encoding: .utf8) {
                        let finalFrame = makeFrame(type: 2, text: "INFERENCE_RESULT:\(summaryStr)")
                        sender.sendAndClose(finalFrame)
                    } else {
                        connection.cancel()
                    }
                } catch {
                    log("[Daemon] Inference error: \(error)")
                    let errFrame = makeFrame(type: 2, text: "ERROR:\(error.localizedDescription)")
                    let sender = ConnectionBatchSender(
                        connection: connection,
                        queueLabel: "qwen.daemon.send.err.\(UUID().uuidString)",
                        batchSize: 1
                    )
                    sender.sendAndClose(errFrame)
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main daemon entry point
// ─────────────────────────────────────────────────────────────────────────────

func runDaemon() async throws {
    let daemonPortValue = Int(ProcessInfo.processInfo.environment["QWEN_DAEMON_PORT"] ?? "45281") ?? 45281
    let ssePortValue = Int(ProcessInfo.processInfo.environment["QWEN_SSE_PORT"] ?? String(daemonPortValue + 1)) ?? (daemonPortValue + 1)
    guard
        let daemonPort = NWEndpoint.Port(rawValue: UInt16(daemonPortValue)),
        let ssePort = NWEndpoint.Port(rawValue: UInt16(ssePortValue))
    else {
        fputs("[Daemon] Invalid port configuration. Use 1-65535 for QWEN_DAEMON_PORT / QWEN_SSE_PORT.\n", stderr)
        _exit(1)
    }

    let serveOnly = ProcessInfo.processInfo.environment["QWEN_SERVE_ONLY_HTTP"] == "1"
    if serveOnly {
        log("[Daemon] Starting OpenAI-compatible HTTP server on port \(ssePortValue)…")
    } else {
        log("[Daemon] Starting ANE inference server (IPC \(daemonPortValue), HTTP/SSE \(ssePortValue))…")
    }

    let daemonBackend = selectedBackend(nil)

    if daemonBackend == "ane" && ProcessInfo.processInfo.environment["QWEN_SKIP_PRELOAD_1024"] == nil {
        setenv("QWEN_SKIP_PRELOAD_1024", "1", 1)
        log("[Daemon] Memory guard: defaulting QWEN_SKIP_PRELOAD_1024=1 (set to 0 to re-enable background 1024 preload)")
    }
    if ProcessInfo.processInfo.environment["QWEN_COLLECT_DECODE_TIMES"] == nil {
        // Keep per-token timing ON. The measured overhead of the two timer reads plus one
        // array append is ~0.02 ms/token (0.08% of a 26 ms dispatch), and without it the
        // API cannot report a true `decode_tok_per_sec` separate from end-to-end time.
        setenv("QWEN_COLLECT_DECODE_TIMES", "1", 1)
    }

    // Load tokenizer only if ANE backend is active.
    let tokenizer: NativeTokenizer?
    if daemonBackend == "ane" {
        log("[Daemon] Loading tokenizer...")
        tokenizer = try NativeTokenizer(tokenizerDirectory: "\(PROJECT_ROOT)/tokenizer")
    } else {
        tokenizer = nil
        log("[Daemon] Backend selected: MLX (experimental) — skipping NativeTokenizer preload")
    }

    let backend: any InferenceBackend
    if daemonBackend == "mlx-server" {
        log("[Daemon] Backend: MLX-Server — proxying to mlx_lm.server at \(ProcessInfo.processInfo.environment["QWEN_MLX_SERVER_URL"] ?? "http://localhost:8080")")
        backend = MLXServerBackend()
    } else if daemonBackend == "mlx" {
        throw NSError(domain: "DaemonServer", code: 1, userInfo: [NSLocalizedDescriptionKey: "MLX backend is disabled in this build."])
    } else {
        let modelURL: URL
        if MODEL_PATH.hasSuffix(".mlpackage") {
            let cacheDir = "\(NSHomeDirectory())/.cache/qwen-ane/models"
            try FileManager.default.createDirectory(atPath: cacheDir, withIntermediateDirectories: true)
            let compiledPath = cacheDir + "/qwen25-1.5b-stateful-compiled.mlmodelc"
            if !FileManager.default.fileExists(atPath: compiledPath) {
                log("[Daemon] Compiling model (one-time)...")
                let compiled = try await MLModel.compileModel(at: URL(fileURLWithPath: MODEL_PATH))
                try FileManager.default.moveItem(at: compiled, to: URL(fileURLWithPath: compiledPath))
            }
            modelURL = URL(fileURLWithPath: compiledPath)
        } else {
            modelURL = URL(fileURLWithPath: MODEL_PATH)
        }

        if ProcessInfo.processInfo.environment["QWEN_B4_BATCH"] == "1" {
            // Request-coalescing B=4 batched decode (D030). Equal-length cohorts only (shared
            // position/mask export → lock-step lanes; PROVEN per-lane correct, spi/lane_isolation.swift).
            let b4Path = ProcessInfo.processInfo.environment["QWEN_B4_MODEL_PATH"]
                ?? "\(PROJECT_ROOT)/models/anemll_qwen25_1.5b_mono_ctx512_lut4_batch4/qwen25_b4_monolithic_lut4.mlmodelc"
            let bSize = Int(ProcessInfo.processInfo.environment["QWEN_B4_BATCH_SIZE"] ?? "")
                ?? BatchInferenceActor.autoBatchSize()
            let windowMs = UInt64(ProcessInfo.processInfo.environment["QWEN_B4_WINDOW_MS"] ?? "8") ?? 8
            log("[Daemon] Loading B=\(bSize) batched-decode engine (QWEN_B4_BATCH=1, window=\(windowMs)ms) from \(b4Path)...")
            backend = try BatchBackendAdapter(modelURL: URL(fileURLWithPath: b4Path), batchSize: bSize, windowMs: windowMs)
            log("[Daemon] Engines loaded [B=\(bSize) request-coalescing ANE] ✓")
        } else {
            log("[Daemon] Loading co-processor engines...")
            let targetEngine = try CortexEngine(modelURL: modelURL)

            backend = CortexBackendAdapter(engine: targetEngine, tokenizer: tokenizer!)
            log("[Daemon] Engines loaded [Target ANE — pure, no speculation] ✓")
        }
    }

    let inferActor = InferenceActor(backend: backend, tokenizer: tokenizer)
    
    // Serve mode: HTTP endpoint only. The IPC and TCP daemon listeners exist for
    // desktop-app integration and are not part of a headless inference server.
    let serveOnlyHTTP = ProcessInfo.processInfo.environment["QWEN_SERVE_ONLY_HTTP"] == "1"

    if serveOnlyHTTP {
        log("[Daemon] Serve mode: HTTP endpoint only (IPC/TCP listeners disabled)")
    } else {
        // Launch Zero-Copy IPC Server alongside TCP
        let ipcServer: IPCServer?
        do {
            let server = try IPCServer(actor: inferActor)
            server.start()
            ipcServer = server
            log("[Daemon] IPCServer started and retained.")
        } catch {
            log("[Daemon] Failed to start IPCServer: \(error)")
            ipcServer = nil
        }
        _ = ipcServer

        // Start TCP listener with Nagle's algorithm disabled (noDelay)
        let tcpOptions = NWProtocolTCP.Options()
        tcpOptions.noDelay = true
        let params = NWParameters(tls: nil, tcp: tcpOptions)
        params.acceptLocalOnly = true

        guard let listener = try? NWListener(using: params, on: daemonPort) else {
            fputs("[Daemon] Failed to bind port \(daemonPortValue)\n", stderr)
            _exit(1)
        }

        listener.newConnectionHandler = { connection in
            log("[Daemon] New connection from \(connection.endpoint)")
            handleConnection(connection, actor: inferActor)
        }

        listener.stateUpdateHandler = { state in
            switch state {
            case .ready:
                log("[Daemon] Listening on port \(daemonPortValue)")
            case .failed(let error):
                fputs("[Daemon] Listener failed: \(error)\n", stderr)
                _exit(1)
            default:
                break
            }
        }

        listener.start(queue: .global(qos: .userInitiated))
    }

    // Keep running — handle SIGTERM gracefully
    signal(SIGTERM) { _ in
        fputs("[Daemon] Shutting down...\n", stderr)
        _exit(0)
    }

    log("[Daemon] Ready. Ctrl+C to stop.")

    // Concurrent HTTP/SSE listener on configured port with keep-alive
    Task {
        do {
            try await runHTTPSSEServer(inferActor: inferActor, port: ssePort.rawValue)
        } catch {
            fputs("[SSE] HTTP server failed: \(error.localizedDescription)\n", stderr)
        }
    }
    
    // Sleep forever asynchronously to retain the listener in ARC scope and keep task active
    while true {
        try? await Task.sleep(nanoseconds: 3600_000_000_000)
    }
}
