import Foundation
@preconcurrency import CoreML
import Accelerate
import Darwin

@_silgen_name("dequantizeKV")
func dequantizeKV(_ src: UnsafePointer<Int8>, _ dst: UnsafeMutablePointer<Float16>, _ count: Int, _ scale: Float)

@_silgen_name("quantizeKV")
func quantizeKV(_ src: UnsafePointer<Float16>, _ dst: UnsafeMutablePointer<Int8>, _ count: Int, _ invScale: Float)

// ─────────────────────────────────────────────────────────────────────────────
// ChunkedCortexEngine — ANEMLL 3-part pipeline (embeddings → FFN_PF → lm_head)
// Zero-alloc hot path:
//   - Pre-allocated MLMultiArrays for every tensor (input + intermediate + output)
//   - MLPredictionOptions.outputBackings so CoreML writes INTO our buffers
//   - Embeddings output buffer == FFN_PF hidden_states input buffer (zero-copy)
//   - FFN_PF output buffer == lm_head input buffer (zero-copy)
//   - FeatureProviders built once and reused (the arrays are aliased)
// ─────────────────────────────────────────────────────────────────────────────

private let configureANECocktailOnce: Void = {
    setenv("MPSGRAPH_ENABLE_ANE_REAL_TIME_QOS", "1", 1)
    setenv("MPSGRAPH_ENABLE_ANE_SYNC_RUN", "1", 1)
    setenv("ESPRESSO_PRIVATE_ANE_ALWAYS_COMPRESSED", "1", 1)
    setenv("ESPRESSO_ANEC_COMPRESS_SPARSE", "1", 1)
    setenv("ESPRESSO_PRIVATE_ANE_SRAM_PERSISTENCE", "1", 1)
    setenv("MPSGRAPH_ENABLE_ANE_PREEMPTION_DISABLE", "1", 1)
}()

public final class ChunkedCortexEngine: @unchecked Sendable {
    // Models
    private let embeddings: MLModel
    private let ffnPF: MLModel          // "infer" function — seq=1 decode
    private let ffnPrefill: MLModel     // "prefill" function — seq=batchSize
    private let lmHead: MLModel

    // KV cache state (stateless vs stateful support)
    private let stateless: Bool
    private let state: MLState?         // shared KV cache (stateful mode)
    private let kvCacheSize: Int        // compressed buffer size (stateless mode)
    private let int8KVCache: UnsafeMutablePointer<Int8>?
    private let kvCacheFP16Input: MLMultiArray?
    private let kvCacheFP16Output: MLMultiArray?
    private let kvCacheInputPtr: UnsafeMutablePointer<Float16>?
    private let kvCacheOutputPtr: UnsafeMutablePointer<Float16>?

    // —— Persistent input tensors (mutated per step via raw pointers) ─────────
    private let inputTokens: MLMultiArray          // [1,1] int32 — to embeddings
    private let positionIds: MLMultiArray          // [1] int32 — to FFN_PF
    private let currentPos: MLMultiArray           // [1] int32 — to FFN_PF
    private let causalMask: MLMultiArray           // [1,1,1,ctx] f16 — to FFN_PF

    // ── Persistent intermediate / output tensors (zero-copy chained) ─────────
    private let hiddenBuf: MLMultiArray            // embeddings OUT == FFN_PF IN
    private let ffnOutBuf: MLMultiArray            // FFN_PF OUT == lm_head IN
    private let argmaxValBuf: MLMultiArray         // lm_head OUT (chunked vals)
    private let argmaxIdxBuf: MLMultiArray         // lm_head OUT (chunked idxs)

    // ── Batched prefill tensors (seq=batchSize) ──────────────────────────────
    private let batchSize: Int
    private let prefillTokens: MLMultiArray        // [1,batchSize] int32 — to embeddings
    private let prefillHidden: MLMultiArray        // [1,batchSize,hidden] — emb OUT == FFN prefill IN
    private let prefillHiddenOut: MLMultiArray      // [1,1,hidden] — FFN prefill OUT (discarded)
    private let prefillPosIds: MLMultiArray        // [batchSize] int32
    private let prefillMask: MLMultiArray          // [1,1,batchSize,ctx] f16
    private let prefillTokensPtr: UnsafeMutablePointer<Int32>
    private let prefillPosIdsPtr: UnsafeMutablePointer<Int32>
    private let prefillMaskPtr: UnsafeMutablePointer<Float16>
    private let prefillHiddenPtr: UnsafeMutablePointer<Float16>
    private let hiddenBufPtr: UnsafeMutablePointer<Float16>
    private let embPrefillProvider: MLDictionaryFeatureProvider
    private let ffnPrefillProvider: MLDictionaryFeatureProvider
    private let embPrefillOptions: MLPredictionOptions
    private let ffnPrefillOptions: MLPredictionOptions

    // Hot pointers
    private let inputTokensPtr: UnsafeMutablePointer<Int32>
    private let positionIdsPtr: UnsafeMutablePointer<Int32>
    private let currentPosPtr: UnsafeMutablePointer<Int32>
    private let causalMaskPtr: UnsafeMutablePointer<Float16>
    private let argmaxValPtr: UnsafeMutablePointer<Float16>
    private let argmaxIdxPtr: UnsafeMutablePointer<Int32>

    // ── Pre-built feature providers (aliased to the arrays above) ────────────
    private let embProvider: MLDictionaryFeatureProvider
    private let ffnProvider: MLDictionaryFeatureProvider
    private let headProvider: MLDictionaryFeatureProvider

    // ── Pre-built prediction options with output backings (zero-alloc out) ───
    private let embOptions: MLPredictionOptions
    private let ffnOptions: MLPredictionOptions
    private let headOptions: MLPredictionOptions

    // ── Constants ────────────────────────────────────────────────────────────
    private let contextLength: Int
    private let hiddenSize: Int
    private let numChunks: Int
    private let chunkSize: Int
    private let eosTokens: Set<Int32> = [151645, 151643]

    private let cortexQueue = DispatchQueue(label: "chunked-cortex", qos: .userInteractive)

    public var ctxLen: Int { contextLength }

    // ─────────────────────────────────────────────────────────────────────────
    public init(modelURL: URL) throws {
        // ANE real-time QoS cocktail – kills XPC dispatch latency (unconditional to prevent any regressions)
        _ = configureANECocktailOnce

        // Dynamically optimize CoreML/MPSGraph/ANE flags for chunked execution.
        // NOTE: MPSGRAPH_ENABLE_ANE_SYNC_RUN=1 synchronously blocks the calling thread during
        // Neural Engine execution. This prevents thread switching overhead and maximizes single-thread
        // latency/throughput, but requires that the daemon serializes and queues all incoming
        // requests to prevent head-of-line blocking on the global thread pool.
        if (ProcessInfo.processInfo.environment["QWEN_INFER_FLAGS"] ?? "on") != "off" {
            setenv("ANEC_SPECIALIZE_LATENCY", "1", 1)
            unsetenv("ESPRESSO_ANEC_REQUIRED_SPARSITY_FOR_COMPRESSION")
            fputs("[ChunkedCortexEngine] ANE chunked execution cocktail set in environment.\n", stderr)
        } else {
            fputs("[ChunkedCortexEngine] QWEN_INFER_FLAGS=off — skipping ANE flag setup (A/B baseline).\n", stderr)
        }

        let dir = modelURL
        let embUrl  = dir.appendingPathComponent("qwen25_embeddings.mlmodelc")
        let ffnUrl  = dir.appendingPathComponent("qwen25_FFN_PF_lut4_chunk_01of01.mlmodelc")
        let headUrl = dir.appendingPathComponent("qwen25_lm_head_lut6.mlmodelc")

        // FFN_PF + lm_head do real matmul — ANE is the right home.
        // Embeddings is a single-row table lookup — CPU is faster than the
        // ANE submission round-trip (~6ms vs <1µs in memory).
        let aneConfig = MLModelConfiguration()
        aneConfig.computeUnits = .cpuAndNeuralEngine
        aneConfig.allowLowPrecisionAccumulationOnGPU = true
        if #available(macOS 15.0, *) {
            aneConfig.optimizationHints.reshapeFrequency = .infrequent
            aneConfig.optimizationHints.specializationStrategy = .fastPrediction
        }

        let cpuConfig = MLModelConfiguration()
        cpuConfig.computeUnits = .cpuOnly
        if #available(macOS 15.0, *) {
            cpuConfig.optimizationHints.reshapeFrequency = .infrequent
            cpuConfig.optimizationHints.specializationStrategy = .fastPrediction
        }

        fputs("[ChunkedCortex] Loading embeddings [CPU]...\n", stderr); fflush(stderr)
        self.embeddings = try MLModel(contentsOf: embUrl,  configuration: cpuConfig)
        fputs("[ChunkedCortex] Loading FFN_PF infer [ANE]...\n", stderr); fflush(stderr)
        let inferConfig = MLModelConfiguration()
        inferConfig.computeUnits = .cpuAndNeuralEngine
        inferConfig.allowLowPrecisionAccumulationOnGPU = true
        
        // Expose state schema / KV cache input checks to determine if model is a single-function stateless monolithic program
        let tempModelForCheck = try MLModel(contentsOf: ffnUrl, configuration: inferConfig)
        let isStateless = tempModelForCheck.modelDescription.inputDescriptionsByName["kv_cache"] != nil
        self.stateless = isStateless
        
        if !isStateless {
            inferConfig.functionName = "infer"
        }
        if #available(macOS 15.0, *) {
            inferConfig.optimizationHints.reshapeFrequency = .infrequent
            inferConfig.optimizationHints.specializationStrategy = .fastPrediction
        }
        self.ffnPF      = isStateless ? tempModelForCheck : try MLModel(contentsOf: ffnUrl, configuration: inferConfig)
        
        fputs("[ChunkedCortex] Loading FFN_PF prefill [ANE]...\n", stderr); fflush(stderr)
        let prefillConfig = MLModelConfiguration()
        prefillConfig.computeUnits = .cpuAndNeuralEngine
        prefillConfig.allowLowPrecisionAccumulationOnGPU = true
        if !isStateless {
            prefillConfig.functionName = "prefill"
        }
        if #available(macOS 15.0, *) {
            prefillConfig.optimizationHints.reshapeFrequency = .infrequent
            prefillConfig.optimizationHints.specializationStrategy = .fastPrediction
        }
        self.ffnPrefill = try MLModel(contentsOf: ffnUrl,  configuration: prefillConfig)
        fputs("[ChunkedCortex] Loading lm_head [ANE]...\n", stderr); fflush(stderr)
        self.lmHead     = try MLModel(contentsOf: headUrl, configuration: aneConfig)

        // KV cache state detection: shared stateful MLState vs explicit stateless KV cache inputs
        if isStateless {
            fputs("[ChunkedCortex] Stateful MLState bypassed. Running in stateless KV cache mode.\n", stderr); fflush(stderr)
            guard let kvDesc = ffnPF.modelDescription.inputDescriptionsByName["kv_cache"],
                  let shape = kvDesc.multiArrayConstraint?.shape, shape.count >= 4 else {
                throw NSError(domain: "ChunkedCortexEngine", code: 3, userInfo: [NSLocalizedDescriptionKey: "Stateless model kv_cache input has invalid description shape."])
            }
            let dim0 = shape[0].intValue
            let dim1 = shape[1].intValue
            let dim2 = shape[2].intValue
            let dim3 = shape[3].intValue
            let cSize = dim0 * dim1 * dim2 * dim3
            self.kvCacheSize = cSize
            let cache = UnsafeMutablePointer<Int8>.allocate(capacity: cSize)
            memset(cache, 0, cSize)
            self.int8KVCache = cache

            let kvIn = try MLMultiArray(shape: shape, dataType: .float16)
            let kvOut = try MLMultiArray(shape: shape, dataType: .float16)
            self.kvCacheFP16Input = kvIn
            self.kvCacheFP16Output = kvOut
            self.kvCacheInputPtr = kvIn.dataPointer.assumingMemoryBound(to: Float16.self)
            self.kvCacheOutputPtr = kvOut.dataPointer.assumingMemoryBound(to: Float16.self)
            self.state = nil
        } else {
            self.state = self.ffnPF.makeState()
            self.kvCacheSize = 0
            self.int8KVCache = nil
            self.kvCacheFP16Input = nil
            self.kvCacheFP16Output = nil
            self.kvCacheInputPtr = nil
            self.kvCacheOutputPtr = nil
        }

        // Derive shapes from model descriptions
        var ctxLen = 1024
        if let maskDesc = ffnPF.modelDescription.inputDescriptionsByName["causal_mask"],
           let shape = maskDesc.multiArrayConstraint?.shape, shape.count >= 4 {
            ctxLen = shape[3].intValue
        }
        self.contextLength = ctxLen

        var hSize = 1536
        if let hsDesc = ffnPF.modelDescription.inputDescriptionsByName["hidden_states"],
           let shape = hsDesc.multiArrayConstraint?.shape, shape.count >= 3 {
            hSize = shape[2].intValue
        }
        self.hiddenSize = hSize

        var nChunks = 16
        if let avDesc = lmHead.modelDescription.outputDescriptionsByName["argmax_val"],
           let shape = avDesc.multiArrayConstraint?.shape, !shape.isEmpty {
            guard let lastDim = shape.last else {
                throw NSError(domain: "ChunkedCortexEngine", code: 4, userInfo: [NSLocalizedDescriptionKey: "argmax_val output has empty shape."])
            }
            nChunks = lastDim.intValue
        }
        self.numChunks = nChunks
        self.chunkSize = 151936 / nChunks

        // Batch size for prefill = seq dim of the prefill function's hidden_states input
        var bSize = 64
        if let hsDesc = ffnPrefill.modelDescription.inputDescriptionsByName["hidden_states"],
           let shape = hsDesc.multiArrayConstraint?.shape, shape.count >= 3 {
            bSize = shape[1].intValue
        }
        self.batchSize = bSize

        // ── Allocate persistent tensors ─────────────────────────────────────
        self.inputTokens  = try MLMultiArray(shape: [1, 1], dataType: .int32)
        self.positionIds  = try MLMultiArray(shape: [1], dataType: .int32)
        self.currentPos   = try MLMultiArray(shape: [1], dataType: .int32)
        self.causalMask   = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: ctxLen)], dataType: .float16)
        self.hiddenBuf    = try MLMultiArray(shape: [1, 1, NSNumber(value: hSize)], dataType: .float16)
        self.ffnOutBuf    = try MLMultiArray(shape: [1, 1, NSNumber(value: hSize)], dataType: .float16)
        self.argmaxValBuf = try MLMultiArray(shape: [NSNumber(value: nChunks)], dataType: .float16)
        self.argmaxIdxBuf = try MLMultiArray(shape: [NSNumber(value: nChunks)], dataType: .int32)

        // ── Batched prefill tensors ─────────────────────────────────────────
        let bs = NSNumber(value: bSize)
        self.prefillTokens    = try MLMultiArray(shape: [1, bs], dataType: .int32)
        self.prefillHidden    = try MLMultiArray(shape: [1, bs, NSNumber(value: hSize)], dataType: .float16)
        self.prefillHiddenOut = try MLMultiArray(shape: [1, bs, NSNumber(value: hSize)], dataType: .float16)
        self.prefillPosIds    = try MLMultiArray(shape: [bs], dataType: .int32)
        self.prefillMask      = try MLMultiArray(shape: [1, 1, bs, NSNumber(value: ctxLen)], dataType: .float16)
        self.prefillTokensPtr = self.prefillTokens.dataPointer.assumingMemoryBound(to: Int32.self)
        self.prefillPosIdsPtr = self.prefillPosIds.dataPointer.assumingMemoryBound(to: Int32.self)
        self.prefillMaskPtr   = self.prefillMask.dataPointer.assumingMemoryBound(to: Float16.self)
        self.prefillHiddenPtr = self.prefillHidden.dataPointer.assumingMemoryBound(to: Float16.self)
        self.hiddenBufPtr     = self.hiddenBuf.dataPointer.assumingMemoryBound(to: Float16.self)

        // Hot pointers
        self.inputTokensPtr = self.inputTokens.dataPointer.assumingMemoryBound(to: Int32.self)
        self.positionIdsPtr = self.positionIds.dataPointer.assumingMemoryBound(to: Int32.self)
        self.currentPosPtr  = self.currentPos.dataPointer.assumingMemoryBound(to: Int32.self)
        self.causalMaskPtr  = self.causalMask.dataPointer.assumingMemoryBound(to: Float16.self)
        self.argmaxValPtr   = self.argmaxValBuf.dataPointer.assumingMemoryBound(to: Float16.self)
        self.argmaxIdxPtr   = self.argmaxIdxBuf.dataPointer.assumingMemoryBound(to: Int32.self)

        // ── Pre-build feature providers (arrays are aliased; mutations live-update) ──
        self.embProvider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": inputTokens
        ])
        
        var ffnDict: [String: MLFeatureValue] = [
            "hidden_states": MLFeatureValue(multiArray: hiddenBuf),
            "causal_mask": MLFeatureValue(multiArray: causalMask),
            "position_ids": MLFeatureValue(multiArray: positionIds),
            "current_pos": MLFeatureValue(multiArray: currentPos)
        ]
        if isStateless {
            guard let kvIn = self.kvCacheFP16Input else {
                throw NSError(domain: "ChunkedCortexEngine", code: 6, userInfo: [NSLocalizedDescriptionKey: "kvCacheFP16Input is nil in ffnProvider."])
            }
            ffnDict["kv_cache"] = MLFeatureValue(multiArray: kvIn)
        }
        self.ffnProvider = try MLDictionaryFeatureProvider(dictionary: ffnDict)
        
        self.headProvider = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": ffnOutBuf
        ])
        self.embPrefillProvider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": prefillTokens
        ])
        
        var ffnPrefillDict: [String: MLFeatureValue] = [
            "hidden_states": MLFeatureValue(multiArray: prefillHidden),
            "causal_mask": MLFeatureValue(multiArray: prefillMask),
            "position_ids": MLFeatureValue(multiArray: prefillPosIds),
            "current_pos": MLFeatureValue(multiArray: currentPos)
        ]
        if isStateless {
            guard let kvIn = self.kvCacheFP16Input else {
                throw NSError(domain: "ChunkedCortexEngine", code: 7, userInfo: [NSLocalizedDescriptionKey: "kvCacheFP16Input is nil in ffnPrefillProvider."])
            }
            ffnPrefillDict["kv_cache"] = MLFeatureValue(multiArray: kvIn)
        }
        self.ffnPrefillProvider = try MLDictionaryFeatureProvider(dictionary: ffnPrefillDict)

        // ── Pre-build prediction options with output backings (CoreML writes into our buffers) ──
        let embOpts = MLPredictionOptions()
        embOpts.outputBackings = ["hidden_states": hiddenBuf]
        self.embOptions = embOpts

        let ffnOpts = MLPredictionOptions()
        var ffnBackings = ["output_hidden_states": ffnOutBuf]
        if isStateless {
            guard let kvOut = self.kvCacheFP16Output else {
                throw NSError(domain: "ChunkedCortexEngine", code: 8, userInfo: [NSLocalizedDescriptionKey: "kvCacheFP16Output is nil in ffnOptions."])
            }
            ffnBackings["kv_cache_out"] = kvOut
        }
        ffnOpts.outputBackings = ffnBackings
        self.ffnOptions = ffnOpts

        let headOpts = MLPredictionOptions()
        headOpts.outputBackings = ["argmax_val": argmaxValBuf, "argmax_idx": argmaxIdxBuf]
        self.headOptions = headOpts

        let embPreOpts = MLPredictionOptions()
        embPreOpts.outputBackings = ["hidden_states": prefillHidden]
        self.embPrefillOptions = embPreOpts

        let ffnPreOpts = MLPredictionOptions()
        var ffnPrefillBackings = ["output_hidden_states": prefillHiddenOut]
        if isStateless {
            guard let kvOut = self.kvCacheFP16Output else {
                throw NSError(domain: "ChunkedCortexEngine", code: 9, userInfo: [NSLocalizedDescriptionKey: "kvCacheFP16Output is nil in ffnPrefillOptions."])
            }
            ffnPrefillBackings["kv_cache_out"] = kvOut
        }
        ffnPreOpts.outputBackings = ffnPrefillBackings
        self.ffnPrefillOptions = ffnPreOpts

        fputs("[ChunkedCortex] Ready (ctx=\(ctxLen), hidden=\(hSize), chunks=\(nChunks), zero-alloc=ON)\n", stderr)
        fflush(stderr)

        // Async ANE pre-warm (uses the same hot path)
        cortexQueue.async { [weak self] in
            try? self?.burnIn()
        }
    }

    private func burnIn() throws {
        let zero = Float16(0.0)
        inputTokensPtr[0] = 0
        positionIdsPtr[0] = 0
        currentPosPtr[0]  = 0
        for i in 0..<contextLength { causalMaskPtr[i] = zero }

        _ = try embeddings.prediction(from: embProvider, options: embOptions)
        
        if self.stateless {
            self.dequantizeKVCache(scale: 0.125)
            _ = try ffnPF.prediction(from: ffnProvider, options: ffnOptions)
            self.quantizeKVCache(scale: 0.125)
        } else {
            guard let st = self.state else {
                fputs("[ChunkedCortex] ERROR: MLState is nil in burnIn\n", stderr)
                throw NSError(domain: "ChunkedCortexEngine", code: 10, userInfo: [NSLocalizedDescriptionKey: "MLState is nil in burnIn."])
            }
            _ = try ffnPF.prediction(from: ffnProvider, using: st, options: ffnOptions)
        }
        
        _ = try lmHead.prediction(from: headProvider, options: headOptions)

        fputs("[ChunkedCortex] SRAM Burn-In Complete. ANE is hot.\n", stderr)
        fflush(stderr)
    }

    // ─────────────────────────────────────────────────────────────────────────
    public func generate(promptIds: [Int32], maxTokens: Int, onToken: @escaping (Int32) -> Void) throws -> ([Int32], [Double]) {
        try cortexQueue.sync {
            guard !promptIds.isEmpty else { return ([], []) }

            let zero = Float16(0.0)
            let negInf = Float16(-40000.0)
            for i in 0..<contextLength { causalMaskPtr[i] = negInf }

            // --- BATCHED PRE-FILL (no lm_head) ---
            // Ingest the prompt in windows of `batchSize` tokens per ANE submission
            // via the FFN_PF "prefill" function, instead of one token at a time.
            let startPrefill = CFAbsoluteTimeGetCurrent()
            let prefillCount = promptIds.count - 1   // positions 0..prefillCount-1 (last token feeds decode)
            var pos = 0
            while pos < prefillCount {
                try autoreleasepool {
                    let len = min(batchSize, prefillCount - pos)
                    for r in 0..<batchSize {
                        let real = r < len
                        let qpos = real ? (pos + r) : min(pos + len - 1, contextLength - 1)
                        let tid = real ? promptIds[pos + r] : 0
                        
                        // Call the embedding model for the single token `tid`
                        inputTokensPtr[0] = tid
                        _ = try embeddings.prediction(from: embProvider, options: embOptions)
                        
                        // Copy the 1x1xhidden embedding output from hiddenBufPtr into prefillHiddenPtr at slice r
                        let destSlicePtr = prefillHiddenPtr.advanced(by: r * hiddenSize)
                        destSlicePtr.update(from: hiddenBufPtr, count: hiddenSize)
                        
                        prefillTokensPtr[r] = tid
                        prefillPosIdsPtr[r] = Int32(qpos)
                        let base = r * contextLength
                        for c in 0..<contextLength {
                            prefillMaskPtr[base + c] = (c <= qpos) ? zero : negInf
                        }
                    }
                    currentPosPtr[0] = Int32(pos)
                    if self.stateless {
                        self.dequantizeKVCache(scale: 0.125)
                        _ = try ffnPrefill.prediction(from: ffnPrefillProvider, options: ffnPrefillOptions)
                        self.quantizeKVCache(scale: 0.125)
                    } else {
                        guard let st = self.state else {
                            fputs("[ChunkedCortex] ERROR: MLState is nil during prefill\n", stderr)
                            throw NSError(domain: "ChunkedCortexEngine", code: 11, userInfo: [NSLocalizedDescriptionKey: "MLState is nil during prefill."])
                        }
                        _ = try ffnPrefill.prediction(from: ffnPrefillProvider, using: st, options: ffnPrefillOptions)
                    }
                    pos += len
                }
            }
            // Decode mask must mark every prefilled position as visible.
            for j in 0..<prefillCount { causalMaskPtr[j] = zero }
            let prefillElapsed = (CFAbsoluteTimeGetCurrent() - startPrefill) * 1000
            fputs("[ChunkedCortex] Batched prefill completed in \(String(format: "%.2f", prefillElapsed))ms (\(promptIds.count) tokens, batch=\(batchSize))\n", stderr)
            fflush(stderr)

            // --- DECODE ---
            let startDecode = CFAbsoluteTimeGetCurrent()
            var currentToken = promptIds.last!
            var currentPosition = promptIds.count - 1
            var generatedIds: [Int32] = []
            var decodeTimes: [Double] = []

            while generatedIds.count < maxTokens {
                if currentPosition >= contextLength {
                    fputs("[ChunkedCortex] Reached context limit (\(contextLength))!\n", stderr)
                    fflush(stderr)
                    break
                }

                // Thermal backoff
                let th = ProcessInfo.processInfo.thermalState
                if th == .critical { usleep(100_000) }
                else if th == .serious { usleep(50_000) }

                inputTokensPtr[0] = currentToken
                positionIdsPtr[0] = Int32(currentPosition)
                currentPosPtr[0]  = Int32(currentPosition)
                causalMaskPtr[currentPosition] = zero

                let stepStart = CFAbsoluteTimeGetCurrent()
                _ = try embeddings.prediction(from: embProvider, options: embOptions)
                let t1 = CFAbsoluteTimeGetCurrent()
                
                if self.stateless {
                    self.dequantizeKVCache(scale: 0.125)
                    _ = try ffnPF.prediction(from: ffnProvider, options: ffnOptions)
                    self.quantizeKVCache(scale: 0.125)
                } else {
                    guard let st = self.state else {
                        fputs("[ChunkedCortex] ERROR: MLState is nil during decode\n", stderr)
                        throw NSError(domain: "ChunkedCortexEngine", code: 12, userInfo: [NSLocalizedDescriptionKey: "MLState is nil during decode."])
                    }
                    _ = try ffnPF.prediction(from: ffnProvider, using: st, options: ffnOptions)
                }
                
                let t2 = CFAbsoluteTimeGetCurrent()
                _ = try lmHead.prediction(from: headProvider, options: headOptions)
                let t3 = CFAbsoluteTimeGetCurrent()
                let stepElapsed = (t3 - stepStart) * 1000
                decodeTimes.append(stepElapsed)
                if generatedIds.count < 3 {
                    fputs(String(format: "  [profile] emb=%.2fms ffn=%.2fms head=%.2fms\n",
                        (t1-stepStart)*1000, (t2-t1)*1000, (t3-t2)*1000), stderr)
                    fflush(stderr)
                }

                // Zero-copy argmax over pre-allocated buffers
                var bestVal: Float16 = -Float16.greatestFiniteMagnitude
                var bestChunk = 0
                for c in 0..<numChunks {
                    let v = argmaxValPtr[c]
                    if v > bestVal { bestVal = v; bestChunk = c }
                }
                let nextToken = Int32(bestChunk * chunkSize) + argmaxIdxPtr[bestChunk]

                if eosTokens.contains(nextToken) { break }
                generatedIds.append(nextToken)
                onToken(nextToken)

                currentToken = nextToken
                currentPosition += 1
            }

            let decodeElapsed = (CFAbsoluteTimeGetCurrent() - startDecode) * 1000
            let avgPerTok = decodeTimes.isEmpty ? 0 : decodeTimes.reduce(0, +) / Double(decodeTimes.count)
            let tps = avgPerTok > 0 ? 1000.0 / avgPerTok : 0
            fputs("[ChunkedCortex] Decode loop completed in \(String(format: "%.2f", decodeElapsed))ms (Avg: \(String(format: "%.2f", avgPerTok))ms/tok, Speed: \(String(format: "%.1f", tps)) tok/s)\n", stderr)
            fflush(stderr)

            return (generatedIds, decodeTimes)
        }
    }

    deinit {
        if self.stateless {
            self.int8KVCache?.deallocate()
        }
    }

    // ── High-Performance KV Cache Quantization/Dequantization Loops ───────
    private func dequantizeKVCache(scale: Float16) {
        guard let src = self.int8KVCache,
              let dst = self.kvCacheInputPtr else { return }
        dequantizeKV(src, dst, self.kvCacheSize, Float(scale))
    }

    private func quantizeKVCache(scale: Float16) {
        guard let src = self.kvCacheOutputPtr,
              let dst = self.int8KVCache else { return }
        quantizeKV(src, dst, self.kvCacheSize, 1.0 / Float(scale))
    }
}
