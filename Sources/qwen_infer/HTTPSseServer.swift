import Foundation
@preconcurrency import Network

// ─────────────────────────────────────────────────────────────────────────────
// OpenAI-compatible HTTP server for the ANE inference engine.
//
// Endpoints
//   GET  /health                 liveness, model + backend identity, uptime
//   GET  /v1/models              OpenAI model list
//   POST /v1/chat/completions    chat completions (streaming and non-streaming)
//   POST /v1/completions         legacy completions (streaming and non-streaming)
//
// Binding   QWEN_SSE_HOST (default 127.0.0.1), QWEN_SSE_PORT (default 45282)
// Auth      QWEN_API_KEY — when set, requires `Authorization: Bearer <key>`
// Identity  QWEN_MODEL_ID (default qwen2.5-1.5b-ane)
//
// No third-party dependencies. Serialized inference is enforced by InferenceActor;
// this layer only translates HTTP <-> the inference contract.
// ─────────────────────────────────────────────────────────────────────────────

private let CR = "\r"
private let LF = "\n"
private let CRLF = "\r\n"

struct HTTPRequest: Sendable {
    let method: String
    let path: String
    let queryParams: [String: String]
    let headers: [String: String]
    let body: Data
}

struct ServerContext: Sendable {
    let modelId: String
    let backendName: String
    let apiKey: String?
    let startedAt: Date
}

// MARK: - Request parsing

private func parseHTTPRequest(_ data: Data) -> HTTPRequest? {
    guard let separator = data.range(of: Data([13, 10, 13, 10])) else { return nil }

    let headerBytes = data[data.startIndex..<separator.lowerBound]
    guard let headerBlock = String(data: headerBytes, encoding: .utf8) else { return nil }

    var lines = headerBlock.components(separatedBy: CRLF)
    guard !lines.isEmpty else { return nil }

    let requestLine = lines.removeFirst().split(separator: " ")
    guard requestLine.count >= 2 else { return nil }
    let method = String(requestLine[0]).uppercased()
    let pathAndQuery = String(requestLine[1])

    var path = pathAndQuery
    var queryParams: [String: String] = [:]
    if let qIndex = pathAndQuery.firstIndex(of: "?") {
        path = String(pathAndQuery[..<qIndex])
        let queryString = String(pathAndQuery[pathAndQuery.index(after: qIndex)...])
        for pair in queryString.components(separatedBy: "&") where !pair.isEmpty {
            let kv = pair.components(separatedBy: "=")
            guard kv.count >= 1, !kv[0].isEmpty else { continue }
            let key = kv[0].removingPercentEncoding ?? kv[0]
            let value = kv.count > 1 ? (kv[1].removingPercentEncoding ?? kv[1]) : ""
            queryParams[key] = value
        }
    }

    var headers: [String: String] = [:]
    for line in lines where !line.isEmpty {
        guard let colon = line.firstIndex(of: ":") else { continue }
        let key = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
        let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        headers[key] = value
    }

    // Body: honour Content-Length. Requests without a body are complete as parsed.
    let contentLength = Int(headers["content-length"] ?? "0") ?? 0
    let bodyStart = data.distance(from: data.startIndex, to: separator.upperBound)
    let available = data.count - bodyStart
    guard available >= contentLength else { return nil } // body still in flight
    let start = data.startIndex + bodyStart
    let body = data.subdata(in: start..<(start + contentLength))

    return HTTPRequest(method: method, path: path, queryParams: queryParams, headers: headers, body: body)
}

// MARK: - JSON helpers

private func jsonData(_ object: [String: Any]) -> Data {
    (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])) ?? Data("{}".utf8)
}

private func openAIError(_ message: String, type: String = "invalid_request_error", code: String? = nil) -> [String: Any] {
    var detail: [String: Any] = ["message": message, "type": type]
    if let code { detail["code"] = code }
    return ["error": detail]
}

private func statusReason(_ status: Int) -> String {
    switch status {
    case 200: return "OK"
    case 204: return "No Content"
    case 400: return "Bad Request"
    case 401: return "Unauthorized"
    case 404: return "Not Found"
    case 405: return "Method Not Allowed"
    case 408: return "Request Timeout"
    case 413: return "Payload Too Large"
    case 500: return "Internal Server Error"
    default: return "Error"
    }
}

// MARK: - Response helpers

private func sendResponse(
    _ connection: NWConnection,
    status: Int,
    contentType: String,
    body: Data,
    extraHeaders: [String: String] = [:]
) {
    var head = "HTTP/1.1 \(status) \(statusReason(status))" + CRLF
    head += "Content-Type: \(contentType)" + CRLF
    head += "Content-Length: \(body.count)" + CRLF
    head += "Connection: close" + CRLF
    head += "Access-Control-Allow-Origin: *" + CRLF
    head += "Access-Control-Allow-Headers: Authorization, Content-Type" + CRLF
    head += "Access-Control-Allow-Methods: GET, POST, OPTIONS" + CRLF
    for (key, value) in extraHeaders {
        head += "\(key): \(value)" + CRLF
    }
    head += CRLF

    var payload = Data(head.utf8)
    payload.append(body)

    connection.send(content: payload, completion: .contentProcessed { _ in
        connection.cancel()
    })
}

/// Non-standard `timings` block (OpenAI clients ignore unknown fields). Decode rate is
/// measured inside the engine and excludes prompt prefill, unlike wall-clock tok/s.
private func timingsObject(_ response: InferenceResponse) -> [String: Any] {
    var t: [String: Any] = [
        "elapsed_sec": response.elapsed_sec,
        "tok_per_sec": response.tok_per_sec
    ]
    if let decode = response.decode_tok_per_sec { t["decode_tok_per_sec"] = decode }
    return t
}

private func sendJSON(_ connection: NWConnection, status: Int, object: [String: Any]) {
    sendResponse(connection, status: status, contentType: "application/json", body: jsonData(object))
}

private func sendError(
    _ connection: NWConnection,
    status: Int,
    message: String,
    type: String = "invalid_request_error",
    code: String? = nil
) {
    sendJSON(connection, status: status, object: openAIError(message, type: type, code: code))
}

// MARK: - SSE framing

private func sseFrame(_ object: [String: Any]) -> Data {
    var frame = Data("data: ".utf8)
    frame.append(jsonData(object))
    frame.append(Data((LF + LF).utf8))
    return frame
}

private func sseDoneFrame() -> Data {
    Data(("data: [DONE]" + LF + LF).utf8)
}

private func makeCompletionId(_ prefix: String) -> String {
    let raw = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    return prefix + String(raw.prefix(24))
}

private func unixNow() -> Int {
    Int(Date().timeIntervalSince1970)
}

// MARK: - Payload types

private struct ChatMessage: Decodable {
    let role: String
    let content: String?
}

private struct ChatCompletionRequest: Decodable {
    let model: String?
    let messages: [ChatMessage]
    let max_tokens: Int?
    let max_completion_tokens: Int?
    let stream: Bool?
    let temperature: Double?
}

private struct LegacyCompletionRequest: Decodable {
    let model: String?
    let prompt: String?
    let max_tokens: Int?
    let stream: Bool?
    let temperature: Double?
}

// MARK: - Inference helpers

private func runInference(
    _ inferActor: InferenceActor,
    prompt: String,
    system: String?,
    maxTokens: Int,
    streamTokens: Bool,
    onToken: @escaping @Sendable (String) -> Void
) async throws -> InferenceResponse {
    let request = InferenceRequest(
        prompt: prompt,
        max_tokens: maxTokens,
        system: system,
        temperature: nil,
        backend: nil
    )
    return try await inferActor.infer(request: request, streamTokens: streamTokens, onToken: onToken)
}

// MARK: - Routing

private func requiresAuth(_ request: HTTPRequest, _ context: ServerContext) -> Bool {
    guard let apiKey = context.apiKey, !apiKey.isEmpty else { return true }
    guard let header = request.headers["authorization"] else { return false }
    let parts = header.split(separator: " ", maxSplits: 1).map(String.init)
    guard parts.count == 2, parts[0].lowercased() == "bearer" else { return false }
    return parts[1] == apiKey
}

private func route(_ request: HTTPRequest, connection: NWConnection, inferActor: InferenceActor, context: ServerContext) {
    // CORS preflight
    if request.method == "OPTIONS" {
        sendResponse(connection, status: 204, contentType: "text/plain", body: Data())
        return
    }

    switch (request.method, request.path) {
    case ("GET", "/health"):
        let uptime = Int(Date().timeIntervalSince(context.startedAt))
        sendJSON(connection, status: 200, object: [
            "status": "ok",
            "model": context.modelId,
            "backend": context.backendName,
            "accelerator": "Apple Neural Engine",
            "context_tokens": 512,
            "max_output_tokens": 256,
            "uptime_sec": uptime
        ])

    case ("GET", "/v1/models"):
        sendJSON(connection, status: 200, object: [
            "object": "list",
            "data": [[
                "id": context.modelId,
                "object": "model",
                "created": unixNow(),
                "owned_by": "local"
            ]]
        ])

    case ("POST", "/v1/chat/completions"):
        guard requiresAuth(request, context) else {
            sendError(connection, status: 401, message: "Invalid or missing API key", type: "authentication_error", code: "invalid_api_key")
            return
        }
        guard let payload = try? JSONDecoder().decode(ChatCompletionRequest.self, from: request.body) else {
            sendError(connection, status: 400, message: "Invalid JSON body for /v1/chat/completions")
            return
        }
        handleChat(payload, connection: connection, inferActor: inferActor, context: context)

    case ("POST", "/v1/completions"):
        guard requiresAuth(request, context) else {
            sendError(connection, status: 401, message: "Invalid or missing API key", type: "authentication_error", code: "invalid_api_key")
            return
        }
        guard let payload = try? JSONDecoder().decode(LegacyCompletionRequest.self, from: request.body),
              let prompt = payload.prompt, !prompt.isEmpty else {
            sendError(connection, status: 400, message: "Invalid JSON body for /v1/completions: 'prompt' is required")
            return
        }
        handleLegacyCompletion(prompt, payload: payload, connection: connection, inferActor: inferActor, context: context)

    case ("GET", "/"):
        sendError(connection, status: 404, message: "Not found. Use POST /v1/chat/completions.", code: "not_found")

    default:
        sendError(connection, status: 404, message: "Unknown endpoint: \(request.method) \(request.path)", code: "not_found")
    }
}

// MARK: - Chat completions

private func handleChat(
    _ payload: ChatCompletionRequest,
    connection: NWConnection,
    inferActor: InferenceActor,
    context: ServerContext
) {
    let system = payload.messages
        .filter { $0.role == "system" }
        .compactMap { $0.content }
        .joined(separator: LF)
    let prompt = payload.messages
        .filter { $0.role == "user" || $0.role == "assistant" }
        .compactMap { $0.content }
        .joined(separator: LF)

    guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        sendError(connection, status: 400, message: "No message content supplied")
        return
    }

    let maxTokens = payload.max_completion_tokens ?? payload.max_tokens ?? 128
    let stream = payload.stream ?? false
    let id = makeCompletionId("chatcmpl-")
    let created = unixNow()
    let model = payload.model ?? context.modelId

    if !stream {
        Task {
            do {
                let response = try await runInference(
                    inferActor, prompt: prompt, system: system,
                    maxTokens: maxTokens, streamTokens: false, onToken: { _ in }
                )
                let completionTokens = response.tokens_generated
                let promptTokens = response.prompt_tokens ?? 0
                sendJSON(connection, status: 200, object: [
                    "id": id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [[
                        "index": 0,
                        "message": ["role": "assistant", "content": response.text],
                        "finish_reason": "stop"
                    ]],
                    "usage": [
                        "prompt_tokens": promptTokens,
                        "completion_tokens": completionTokens,
                        "total_tokens": promptTokens + completionTokens
                    ],
                    "timings": timingsObject(response)
                ])
            } catch {
                sendError(connection, status: 500, message: error.localizedDescription, type: "server_error")
            }
        }
        return
    }

    // Streaming
    let sender = ConnectionBatchSender(
        connection: connection,
        queueLabel: "qwen.sse.send.\(UUID().uuidString)",
        batchSize: max(1, Int(ProcessInfo.processInfo.environment["QWEN_SSE_STREAM_BATCH"] ?? "1") ?? 1)
    )

    var headers = "HTTP/1.1 200 OK" + CRLF
    headers += "Content-Type: text/event-stream" + CRLF
    headers += "Cache-Control: no-cache" + CRLF
    headers += "Connection: close" + CRLF
    headers += "Access-Control-Allow-Origin: *" + CRLF
    headers += CRLF
    sender.sendRaw(Data(headers.utf8))

    let roleFrame = sseFrame([
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [[
            "index": 0,
            "delta": ["role": "assistant", "content": ""],
            "finish_reason": NSNull()
        ]]
    ])
    sender.enqueue(roleFrame)

    Task {
        do {
            _ = try await runInference(
                inferActor, prompt: prompt, system: system,
                maxTokens: maxTokens, streamTokens: true
            ) { piece in
                guard !piece.isEmpty else { return }
                sender.enqueue(sseFrame([
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [[
                        "index": 0,
                        "delta": ["content": piece],
                        "finish_reason": NSNull()
                    ]]
                ]))
            }
            sender.enqueue(sseFrame([
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [[
                    "index": 0,
                    "delta": [String: Any](),
                    "finish_reason": "stop"
                ]]
            ]))
            sender.sendAndClose(sseDoneFrame())
        } catch {
            sender.enqueue(sseFrame(openAIError(error.localizedDescription, type: "server_error")))
            sender.sendAndClose(sseDoneFrame())
        }
    }
}

// MARK: - Legacy completions

private func handleLegacyCompletion(
    _ prompt: String,
    payload: LegacyCompletionRequest,
    connection: NWConnection,
    inferActor: InferenceActor,
    context: ServerContext
) {
    let maxTokens = payload.max_tokens ?? 128
    let stream = payload.stream ?? false
    let id = makeCompletionId("cmpl-")
    let created = unixNow()
    let model = payload.model ?? context.modelId

    if !stream {
        Task {
            do {
                let response = try await runInference(
                    inferActor, prompt: prompt, system: nil,
                    maxTokens: maxTokens, streamTokens: false, onToken: { _ in }
                )
                let completionTokens = response.tokens_generated
                let promptTokens = response.prompt_tokens ?? 0
                sendJSON(connection, status: 200, object: [
                    "id": id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [[
                        "text": response.text,
                        "index": 0,
                        "logprobs": NSNull(),
                        "finish_reason": "stop"
                    ]],
                    "usage": [
                        "prompt_tokens": promptTokens,
                        "completion_tokens": completionTokens,
                        "total_tokens": promptTokens + completionTokens
                    ],
                    "timings": timingsObject(response)
                ])
            } catch {
                sendError(connection, status: 500, message: error.localizedDescription, type: "server_error")
            }
        }
        return
    }

    let sender = ConnectionBatchSender(
        connection: connection,
        queueLabel: "qwen.sse.send.\(UUID().uuidString)",
        batchSize: max(1, Int(ProcessInfo.processInfo.environment["QWEN_SSE_STREAM_BATCH"] ?? "1") ?? 1)
    )

    var headers = "HTTP/1.1 200 OK" + CRLF
    headers += "Content-Type: text/event-stream" + CRLF
    headers += "Cache-Control: no-cache" + CRLF
    headers += "Connection: close" + CRLF
    headers += "Access-Control-Allow-Origin: *" + CRLF
    headers += CRLF
    sender.sendRaw(Data(headers.utf8))

    Task {
        do {
            _ = try await runInference(
                inferActor, prompt: prompt, system: nil,
                maxTokens: maxTokens, streamTokens: true
            ) { piece in
                guard !piece.isEmpty else { return }
                sender.enqueue(sseFrame([
                    "id": id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [[
                        "text": piece,
                        "index": 0,
                        "logprobs": NSNull(),
                        "finish_reason": NSNull()
                    ]]
                ]))
            }
            sender.enqueue(sseFrame([
                "id": id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [[
                    "text": "",
                    "index": 0,
                    "logprobs": NSNull(),
                    "finish_reason": "stop"
                ]]
            ]))
            sender.sendAndClose(sseDoneFrame())
        } catch {
            sender.enqueue(sseFrame(openAIError(error.localizedDescription, type: "server_error")))
            sender.sendAndClose(sseDoneFrame())
        }
    }
}

// MARK: - Connection handling

private let maxRequestBytes = 8 * 1024 * 1024

private func receiveRequest(
    _ connection: NWConnection,
    buffer: Data,
    inferActor: InferenceActor,
    context: ServerContext
) {
    connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { data, _, isComplete, error in
        var accumulated = buffer
        if let data { accumulated.append(data) }

        if accumulated.count > maxRequestBytes {
            sendError(connection, status: 413, message: "Request exceeds \(maxRequestBytes) bytes")
            return
        }

        if let request = parseHTTPRequest(accumulated) {
            route(request, connection: connection, inferActor: inferActor, context: context)
            return
        }

        if isComplete || error != nil {
            connection.cancel()
            return
        }

        receiveRequest(connection, buffer: accumulated, inferActor: inferActor, context: context)
    }
}

private func handleConnection(_ connection: NWConnection, inferActor: InferenceActor, context: ServerContext) {
    connection.start(queue: .global(qos: .userInitiated))
    receiveRequest(connection, buffer: Data(), inferActor: inferActor, context: context)
}

// MARK: - Server entry point

func runHTTPSSEServer(inferActor: InferenceActor, port: UInt16 = 45282) async throws {
    let env = ProcessInfo.processInfo.environment
    let host = env["QWEN_SSE_HOST"] ?? "127.0.0.1"
    let isLoopback = host == "127.0.0.1" || host == "localhost" || host == "::1"

    let backendName = await inferActor.backend.backendName
    let context = ServerContext(
        modelId: env["QWEN_MODEL_ID"] ?? "qwen2.5-1.5b-ane",
        backendName: backendName,
        apiKey: (env["QWEN_API_KEY"].flatMap { $0.isEmpty ? nil : $0 }),
        startedAt: Date()
    )

    let tcpOptions = NWProtocolTCP.Options()
    tcpOptions.noDelay = true
    tcpOptions.enableKeepalive = true
    tcpOptions.keepaliveIdle = 60

    let params = NWParameters(tls: nil, tcp: tcpOptions)
    params.acceptLocalOnly = isLoopback
    if !isLoopback {
        params.requiredLocalEndpoint = .hostPort(host: NWEndpoint.Host(host), port: NWEndpoint.Port(rawValue: port)!)
    }

    guard let listener = try? NWListener(using: params, on: NWEndpoint.Port(rawValue: port)!) else {
        fputs("[http] Failed to bind \(host):\(port)" + LF, stderr)
        _exit(1)
    }

    listener.newConnectionHandler = { connection in
        handleConnection(connection, inferActor: inferActor, context: context)
    }

    listener.stateUpdateHandler = { state in
        switch state {
        case .ready:
            fputs("[http] Listening on http://\(host):\(port)  model=\(context.modelId)  backend=\(context.backendName)" + LF, stderr)
            fputs("[http] Endpoints: GET /health · GET /v1/models · POST /v1/chat/completions · POST /v1/completions" + LF, stderr)
        case .failed(let error):
            fputs("[http] Listener failed: \(error.localizedDescription)" + LF, stderr)
            _exit(1)
        default:
            break
        }
    }

    listener.start(queue: .global(qos: .userInitiated))

    while true {
        try? await Task.sleep(nanoseconds: 3600_000_000_000)
    }
}
