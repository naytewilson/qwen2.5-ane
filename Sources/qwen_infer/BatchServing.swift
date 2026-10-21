//  BatchServing.swift — request-coalescing B=4 batched-decode serve layer for qwen_infer.
//
//  Turns N concurrent decode dispatches into 1 ANE cohort dispatch.
//  Activated behind QWEN_B4_BATCH=1; B=1 (CortexBackendAdapter) stays the default.
//
//  MEASURED CONSTRAINT (A18 Pro, 2026-06-24):
//    The deployable B=4 export shares ONE scalar position_ids[1] + ONE causal_mask[1,1,1,512]
//    row across all lanes. That makes lanes a STRICT LOCK-STEP cohort: every lane attends the
//    identical KV column set every step. Consequences, both verified on-device:
//      • Equal-length cohorts are per-lane BIT-CORRECT (neighbor-invariance: all 4 lanes PASS).
//      • RAGGED (mixed-length) cohorts are BROKEN — a short lane diverges at step 0 when batched
//        with longer neighbors (shared mask forces it to attend the long lane's columns).
//    => v1 coalesces ONLY equal prompt-length requests. Different lengths fall back to singleton
//       cohorts (B=1, still correct). Variable-length batching needs a re-export with PER-LANE
//       position_ids[4,1] + per-lane mask[4,1,1,512] (Opus converter lane) — then drop the
//       length-bucket key and this layer batches arbitrary traffic unchanged.
//
//  Concurrency: InferenceActor.infer is `async`; at its `await backend.generate` point the actor
//  suspends and services other connections, so concurrent requests pile into this actor's pending
//  queue and coalesce. One cohort in flight at a time (draining guard); each request suspends on
//  exactly one continuation, resumed once when its cohort finishes (lock-step v1).

import Foundation
import CoreML

// MARK: - Engine seam

/// A batched-decode engine: advance up to `batchSize` independent streams, one cohort to completion.
protocol BatchDecodeEngine: Sendable {
    var batchSize: Int { get }
    /// Drive one cohort. `prompts.count` may be < batchSize (the engine pads internally for the ANE).
    /// All prompts in a cohort MUST be equal length (the actor guarantees this for the shared-mask export).
    /// `onToken(slot, token)` fires once per generated token per slot, in decode order.
    func decodeCohort(prompts: [[Int32]],
                      maxNewTokens: [Int],
                      onToken: @escaping @Sendable (_ slot: Int, _ token: Int32) -> Void) async
}

// MARK: - Token sink (mutated only by the engine's serial decode loop, never two cohorts at once)

private final class TokenSink: @unchecked Sendable {
    var tokens: [[Int32]]
    private let userOnToken: [@Sendable (Int32) -> Void]
    init(_ n: Int, onTokens: [@Sendable (Int32) -> Void]) {
        tokens = Array(repeating: [], count: n); userOnToken = onTokens
    }
    func emit(_ slot: Int, _ tok: Int32) {
        guard slot < tokens.count else { return }   // ignore pad slots
        tokens[slot].append(tok)
        userOnToken[slot](tok)
    }
}

// MARK: - The coalescing actor (length-bucketed: equal-length cohorts only)

actor BatchInferenceActor {
    private struct Pending {
        let prompt: [Int32]
        let maxNew: Int
        let onToken: @Sendable (Int32) -> Void
        let cont: CheckedContinuation<[Int32], Never>
    }

    private let engine: BatchDecodeEngine
    private let windowNs: UInt64
    private var pending: [Pending] = []
    private var draining = false
    private var timerScheduled = false
    private(set) var cohortSizes: [Int] = []   // observability (and test assertions)

    init(engine: BatchDecodeEngine, windowMs: UInt64 = 8) {
        self.engine = engine
        self.windowNs = windowMs * 1_000_000
    }

    /// Submit one request. Suspends until its cohort finishes; returns this request's full token list.
    func submit(promptIds: [Int32], maxNewTokens: Int,
                onToken: @escaping @Sendable (Int32) -> Void) async -> [Int32] {
        await withCheckedContinuation { (cont: CheckedContinuation<[Int32], Never>) in
            pending.append(Pending(prompt: promptIds, maxNew: maxNewTokens, onToken: onToken, cont: cont))
            maybeFire()
        }
    }

    /// A length-bucket is "full" when ≥ batchSize requests share one prompt length.
    private func fullLength() -> Int? {
        var counts: [Int: Int] = [:]
        for p in pending {
            let l = p.prompt.count
            counts[l, default: 0] += 1
            if counts[l]! >= engine.batchSize { return l }
        }
        return nil
    }

    private func maybeFire() {
        guard !draining else { return }               // a cohort is already in flight
        if fullLength() != nil {
            draining = true
            Task { await self.runCohort() }           // a length-bucket is full → fire now
        } else if !timerScheduled, !pending.isEmpty {
            timerScheduled = true                      // partial → fire the oldest after the window
            Task { try? await Task.sleep(nanoseconds: windowNs); await self.onTimer() }
        }
    }

    private func onTimer() {
        timerScheduled = false
        guard !draining, !pending.isEmpty else { return }
        draining = true
        Task { await self.runCohort() }
    }

    private func runCohort() async {
        // Choose the cohort length L: a full bucket if one exists, else the OLDEST request's
        // length (so a lone/odd-length request is served within the window and never starves).
        let L = fullLength() ?? pending.first!.prompt.count

        // Gather up to batchSize SAME-length requests, preserving arrival order.
        var cohort: [Pending] = []
        var rest: [Pending] = []
        for p in pending {
            if p.prompt.count == L && cohort.count < engine.batchSize { cohort.append(p) }
            else { rest.append(p) }
        }
        pending = rest
        cohortSizes.append(cohort.count)
        log("[B4] cohort fired: size=\(cohort.count) promptLen=\(L) (\(pending.count) still queued)")

        // Split into Sendable pieces BEFORE the await so the non-Sendable `cohort`
        // (Pending holds closures + continuations) is never held across the suspension point.
        let prompts = cohort.map { $0.prompt }            // [[Int32]] — Sendable
        let maxNews = cohort.map { $0.maxNew }            // [Int] — Sendable
        let conts   = cohort.map { $0.cont }             // [CheckedContinuation] — Sendable
        let sink    = TokenSink(cohort.count, onTokens: cohort.map { $0.onToken })  // @unchecked Sendable

        await engine.decodeCohort(prompts: prompts, maxNewTokens: maxNews) { slot, tok in
            sink.emit(slot, tok)
        }
        for (i, c) in conts.enumerated() { c.resume(returning: sink.tokens[i]) }

        draining = false
        maybeFire()                                    // drain anything that queued while decoding
    }

    /// Heuristic B for the host. 8GB A18 = floor → 4. (The B=4 export pins this to 4 regardless.)
    static func autoBatchSize() -> Int {
        let gb = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        switch gb {
        case ..<10:  return 4     // A18 / 8GB — the floor case
        case ..<24:  return 8
        default:     return 8
        }
    }
}

// MARK: - B=4 ANE engine (the real BatchDecodeEngine — equal-length lock-step decode)

/// Wraps the deployable B=4 monolithic .mlmodelc. Decode contract (D030, PROVEN):
///   input_ids   [B,1] int32  — per-lane new token (only B-leading input)
///   position_ids [1]  int32  — SHARED scalar (lock-step)
///   current_pos  [1]  int32  — SHARED scalar
///   causal_mask [1,1,1,CTX] fp16 — SHARED row
///   state model_model_kv_cache_0 [2*L*B, 2, CTX, HD] — leading-B KV
///   outputs logits1..N each [B,1,CHUNK] (rank-3; vocab stride = strides[2])
final class B4BatchEngine: BatchDecodeEngine, @unchecked Sendable {
    let batchSize: Int
    private let model: MLModel
    private let ctx: Int
    private let nChunk: Int
    private let chunk: Int
    private let graphBatch: Int   // batch dim the GRAPH declares on input_ids ([B,1] deployable vs [1,1] fresh anemll exports)
    private let eosToken: Int32
    private let queue = DispatchQueue(label: "qwen.b4batch", qos: .userInteractive)
    private let predictionOptions: MLPredictionOptions

    init(modelURL: URL, batchSize: Int, ctx: Int = 512, eosToken: Int32 = 151645) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        cfg.allowLowPrecisionAccumulationOnGPU = false
        if #available(macOS 15.0, *) { cfg.optimizationHints.specializationStrategy = .fastPrediction }
        // The deployable is a single-function ML Program; no functionName needed.
        self.model = try MLModel(contentsOf: modelURL, configuration: cfg)
        self.batchSize = batchSize
        self.ctx = ctx
        self.eosToken = eosToken
        self.predictionOptions = MLPredictionOptions()

        // Discover chunked-logits layout from the model's outputs (NCHUNK, CHUNK).
        let outNames = model.modelDescription.outputDescriptionsByName.keys.filter { $0.hasPrefix("logits") }
        guard !outNames.isEmpty,
              let firstShape = model.modelDescription.outputDescriptionsByName["logits1"]?.multiArrayConstraint?.shape.map({ $0.intValue }),
              let lastDim = firstShape.last else {
            throw NSError(domain: "B4BatchEngine", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "model has no chunked logits1..N outputs"])
        }
        self.nChunk = outNames.count
        self.chunk  = lastDim

        // Graph batch dim: the D030 deployable declares input_ids [B,1]; fresh anemll-v0.3.5
        // exports declare [1,1] (single-lane graph, single-lane KV state). Detect it here so
        // the serve path can drive either graph instead of throwing a CoreML shape mismatch
        // on the first prefill prediction (the "0 tokens in ~16ms" failure).
        let idsShape = model.modelDescription.inputDescriptionsByName["input_ids"]?
            .multiArrayConstraint?.shape.map { $0.intValue } ?? []
        self.graphBatch = idsShape.first ?? batchSize
    }

    func decodeCohort(prompts: [[Int32]], maxNewTokens: [Int],
                      onToken: @escaping @Sendable (Int, Int32) -> Void) async {
        // Bridge the synchronous ANE decode onto a serial queue (one cohort at a time, per the actor).
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            queue.async {
                self.decodeCohortSync(prompts: prompts, maxNewTokens: maxNewTokens, onToken: onToken)
                cont.resume()
            }
        }
    }

    private func decodeCohortSync(prompts: [[Int32]], maxNewTokens: [Int],
                                  onToken: @Sendable (Int, Int32) -> Void) {
        let nReal = prompts.count
        guard nReal > 0 else { return }
        if graphBatch == 1 {
            // B=1 graph: there is no batch dim to fill — drive each real lane with its own KV state.
            decodeLanesOnSingleBatchGraphSync(prompts: prompts, maxNewTokens: maxNewTokens, onToken: onToken)
            return
        }
        let L = prompts[0].count
        // All real prompts must be equal length (actor guarantees). Pad lanes to batchSize by
        // duplicating lane 0's prompt — proven neighbor-invariant, so it cannot perturb real lanes.
        var lanes = prompts
        while lanes.count < batchSize { lanes.append(prompts[0]) }

        do {
            let st = model.makeState()
            let ids  = try MLMultiArray(shape: [NSNumber(value: batchSize), 1], dataType: .int32)
            let pos  = try MLMultiArray(shape: [1], dataType: .int32)
            let cur  = try MLMultiArray(shape: [1], dataType: .int32)
            let mask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: ctx)], dataType: .float16)
            let idsP  = ids.dataPointer.assumingMemoryBound(to: Int32.self)
            let posP  = pos.dataPointer.assumingMemoryBound(to: Int32.self)
            let curP  = cur.dataPointer.assumingMemoryBound(to: Int32.self)
            let maskP = mask.dataPointer.assumingMemoryBound(to: Float16.self)
            let fp = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": ids, "position_ids": pos, "current_pos": cur, "causal_mask": mask])

            func step(_ toks: [Int32], _ p: Int) throws -> MLFeatureProvider {
                for b in 0..<batchSize { idsP[b] = toks[b] }
                posP[0] = Int32(p); curP[0] = Int32(p)
                for i in 0..<ctx { maskP[i] = (i <= p) ? Float16(0) : Float16(-40000.0) }
                return try model.prediction(from: fp, using: st, options: predictionOptions)
            }

            // Lock-step prefill (all lanes equal length L).
            for i in 0..<L {
                _ = try step(lanes.map { $0[i] }, i)
            }
            var last = lanes.map { $0.last! }
            var done = [Bool](repeating: false, count: nReal)
            let maxSteps = maxNewTokens.prefix(nReal).max() ?? 0

            for s in 0..<maxSteps {
                let out = try step(last, L + s)
                var next = [Int32](repeating: 0, count: batchSize)
                for b in 0..<batchSize { next[b] = argmaxLane(out, b) }
                for slot in 0..<nReal where !done[slot] && s < maxNewTokens[slot] {
                    let t = next[slot]
                    onToken(slot, t)
                    if t == eosToken { done[slot] = true }
                }
                if (0..<nReal).allSatisfy({ done[$0] || s + 1 >= maxNewTokens[$0] }) { break }
                last = next
            }
        } catch {
            // Decode failure: surface the actual error instead of swallowing it.
            fputs("[B4 DECODE ERROR] \(error)\n", stderr)
            return
        }
    }

    /// Fresh anemll-v0.3.5 exports are B=1 graphs: input_ids [1,1], single-lane KV state
    /// [56,2,CTX,HD], logits1..16 [1,1,CHUNK]. There is no batch dim, so a cohort is served
    /// by threading a SEPARATE MLState per real lane through the one loaded model (the
    /// mono_pplx pattern), round-robin one token per live lane per pass so cohort members
    /// stream fairly. Pad lanes are never decoded — they existed only to fill the ANE batch.
    /// mono_pplx-aligned semantics: the first generated token is the argmax of the FINAL
    /// prefill step's logits (no double-feed of the last prompt token).
    private func decodeLanesOnSingleBatchGraphSync(prompts: [[Int32]], maxNewTokens: [Int],
                                                   onToken: @Sendable (Int, Int32) -> Void) {
        let nReal = prompts.count
        do {
            let ids  = try MLMultiArray(shape: [1, 1], dataType: .int32)
            let pos  = try MLMultiArray(shape: [1], dataType: .int32)
            let cur  = try MLMultiArray(shape: [1], dataType: .int32)
            let mask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: ctx)], dataType: .float16)
            let idsP  = ids.dataPointer.assumingMemoryBound(to: Int32.self)
            let posP  = pos.dataPointer.assumingMemoryBound(to: Int32.self)
            let curP  = cur.dataPointer.assumingMemoryBound(to: Int32.self)
            let maskP = mask.dataPointer.assumingMemoryBound(to: Float16.self)
            let fp = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": ids, "position_ids": pos, "current_pos": cur, "causal_mask": mask])

            func step(_ tok: Int32, _ p: Int, _ st: MLState) throws -> MLFeatureProvider {
                idsP[0] = tok
                posP[0] = Int32(p); curP[0] = Int32(p)
                for i in 0..<ctx { maskP[i] = (i <= p) ? Float16(0) : Float16(-40000.0) }
                return try model.prediction(from: fp, using: st, options: predictionOptions)
            }

            var states: [MLState] = []
            for _ in 0..<nReal { states.append(model.makeState()) }
            var last = [Int32](repeating: 0, count: nReal)
            var plen = [Int](repeating: 0, count: nReal)
            var nGen = [Int](repeating: 0, count: nReal)
            var done = [Bool](repeating: false, count: nReal)

            // Per-lane prefill (no lock-step constraint on a B=1 graph), first token from
            // the final prefill logits.
            for lane in 0..<nReal {
                let p = prompts[lane]
                plen[lane] = p.count
                guard !p.isEmpty, maxNewTokens[lane] > 0 else { done[lane] = true; continue }
                var out: MLFeatureProvider? = nil
                for i in 0..<p.count { out = try step(p[i], i, states[lane]) }
                let t = argmaxLane(out!, 0)
                onToken(lane, t); nGen[lane] = 1
                if t == eosToken { done[lane] = true } else { last[lane] = t }
            }

            // Round-robin decode: generated token #k is fed at position plen + k - 1.
            while true {
                var any = false
                for lane in 0..<nReal where !done[lane] && nGen[lane] < maxNewTokens[lane] {
                    let fpos = plen[lane] + nGen[lane] - 1
                    if fpos >= ctx { done[lane] = true; continue }   // context window exhausted
                    let out = try step(last[lane], fpos, states[lane])
                    let t = argmaxLane(out, 0)
                    onToken(lane, t); nGen[lane] += 1
                    if t == eosToken { done[lane] = true } else { last[lane] = t }
                    any = true
                }
                if !any { break }
            }
        } catch {
            fputs("[B4 DECODE ERROR] \(error)\n", stderr)
            return
        }
    }

    private func argmaxLane(_ out: MLFeatureProvider, _ r: Int) -> Int32 {
        var best = -Float.infinity
        var bestI: Int32 = 0
        for c in 0..<nChunk {
            guard let ch = out.featureValue(for: "logits\(c+1)")?.multiArrayValue else { continue }
            let rowStride = ch.strides[0].intValue
            // rank-3 [B,1,CHUNK]: vocab stride = strides[2]; rank-2 [B,CHUNK]: strides[1].
            let colStride = ch.shape.count == 3 ? ch.strides[2].intValue : ch.strides[1].intValue
            let p = ch.dataPointer.assumingMemoryBound(to: Float16.self) + r * rowStride
            for j in 0..<chunk {
                let v = Float(p[j * colStride])
                if v > best { best = v; bestI = Int32(c * chunk + j) }
            }
        }
        return bestI
    }
}

// MARK: - Backend adapter (routes generate() into the coalescing actor)

/// Drop-in InferenceBackend that batches concurrent requests through BatchInferenceActor.
/// Reuses all of InferenceActor.infer's ChatML / truncation / streaming machinery — only the
/// decode engine changes. Constructed behind QWEN_B4_BATCH=1 in runDaemon.
final class BatchBackendAdapter: InferenceBackend, @unchecked Sendable {
    let backendName = "ane"
    private let actor: BatchInferenceActor

    init(modelURL: URL, batchSize: Int, windowMs: UInt64) throws {
        let engine = try B4BatchEngine(modelURL: modelURL, batchSize: batchSize)
        self.actor = BatchInferenceActor(engine: engine, windowMs: windowMs)
    }

    func generate(
        promptIds: [Int32],
        promptText _: String,
        systemPrompt _: String?,
        maxTokens: Int,
        onToken: @escaping @Sendable (Int32, String?) -> Void
    ) async throws -> BackendGenerationResult {
        let tokens = await actor.submit(promptIds: promptIds, maxNewTokens: maxTokens) { tokenId in
            onToken(tokenId, nil)
        }
        return BackendGenerationResult(tokenIds: tokens, decodeTimesMs: nil, includesPrompt: false)
    }
}
