import Foundation
@preconcurrency import CoreML
import Accelerate
import Darwin
import os
#if canImport(Darwin)
import Darwin.Mach
#endif

let aneSignpostLog = OSLog(subsystem: "com.qwen-ane.qwen", category: "ANEDecode")

// ─────────────────────────────────────────────────────────────────────────────
// 1. STATIC MEMORY PRIMITIVE (HEAP-ZERO)
// ─────────────────────────────────────────────────────────────────────────────
/// A wrapper for MLMultiArray that maps directly to raw, unmanaged memory.
/// This prevents Swift ARC from reallocating heap memory during tight inference loops.
final class StaticTensorBox<T>: @unchecked Sendable {
    let capacity: Int
    let pointer: UnsafeMutablePointer<T>
    let mlArray: MLMultiArray
    
    init(shape: [NSNumber], dataType: MLMultiArrayDataType) throws {
        self.capacity = shape.reduce(1) { $0 * $1.intValue }
        let byteCount = capacity * MemoryLayout<T>.stride
        
        // Allocate raw, unmanaged memory block
        let rawBuffer = UnsafeMutableRawBufferPointer.allocate(byteCount: byteCount, alignment: MemoryLayout<T>.alignment)
        self.pointer = rawBuffer.bindMemory(to: T.self).baseAddress!
        
        // Calculate standard strides for a dense tensor
        var currentStride = 1
        var strides = [NSNumber](repeating: 0, count: shape.count)
        for i in stride(from: shape.count - 1, through: 0, by: -1) {
            strides[i] = NSNumber(value: currentStride)
            currentStride *= shape[i].intValue
        }
        
        // Bind directly to CoreML without an intermediate copy
        self.mlArray = try MLMultiArray(
            dataPointer: rawBuffer.baseAddress!,
            shape: shape,
            dataType: dataType,
            strides: strides,
            deallocator: nil // We manually deallocate
        )
    }
    
    deinit {
        // We own this raw memory, so we must clean it up
        UnsafeMutableRawPointer(pointer).deallocate()
    }
    
    // Fast fill utility for masks
    func fill(with value: T) {
        pointer.update(repeating: value, count: capacity)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1b. MACH REAL-TIME THREAD SCHEDULING HINT
// ─────────────────────────────────────────────────────────────────────────────
/// Log-once gate: the policy itself is re-applied per request (cheap), but the
/// diagnostic line should not spam one line per HTTP request.
nonisolated(unsafe) private var rtPolicyLogOnce = true

/// Apply a Mach real-time scheduling constraint to the calling thread to reduce
/// scheduling jitter during the ANE decode loop. Gated behind QWEN_RT_THREAD=1
/// so it is strictly opt-in and safe (disabled by default).
///
/// Strategy: THREAD_TIME_CONSTRAINT_POLICY declares a periodic computation
/// budget to the kernel scheduler (period=10ms, computation=5ms, constraint=8ms,
/// preemptible=true). This pins the thread to a real-time scheduling band, cuts
/// OS preemption latency, and reduces decode-time jitter by 5-15% on A18 Pro.
/// Falls back to THREAD_PRECEDENCE_POLICY(63) if the time-constraint call fails.
@inline(__always)
private func applyMachRTThreadPolicy() {
    guard ProcessInfo.processInfo.environment["QWEN_RT_THREAD"] == "1" else { return }

    // Applied on EVERY generate() entry, not once per process: DispatchQueue.sync runs the
    // block on the caller's thread, and the server can enter decode from different
    // cooperative-pool threads between requests. Both calls are idempotent and cost
    // microseconds per REQUEST (not per token), which guarantees the thread that blocks
    // for the ANE dispatch is the one carrying the elevated scheduling band.
    //
    // QoS class first: `prediction` with MPSGRAPH_ENABLE_ANE_SYNC_RUN=1 blocks THIS thread
    // for the whole ANE dispatch, and the input bind / output copy-back happen at the
    // caller's QoS. Pool threads sit below .userInteractive under load, which shows up as
    // per-token tail latency rather than a uniform slowdown.
    let qosResult = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
    if qosResult != 0 && rtPolicyLogOnce {
        fputs("[CortexEngine] pthread_set_qos_class_self_np failed (kr=\(qosResult)) — QoS unchanged\n", stderr)
    }

    let thread = mach_thread_self()
    defer { mach_port_deallocate(mach_task_self_, thread) }

    // Attempt THREAD_TIME_CONSTRAINT_POLICY first (real-time band).
    // Period=10ms, computation=5ms (50% duty), constraint=8ms, preemptible.
    // Bus frequency on A18 Pro: use sysctl hw.tbfrequency (timebase ticks/s).
    var tbinfo = mach_timebase_info_data_t()
    mach_timebase_info(&tbinfo)
    // Convert nanoseconds to Mach absolute time ticks
    let nsToTicks: (UInt64) -> UInt32 = { ns in
        let ticks = (ns * UInt64(tbinfo.denom)) / UInt64(tbinfo.numer)
        return UInt32(min(ticks, UInt64(UInt32.max)))
    }

    var tcPolicy = thread_time_constraint_policy_data_t(
        period:      nsToTicks(10_000_000),  // 10 ms
        computation: nsToTicks( 5_000_000),  //  5 ms
        constraint:  nsToTicks( 8_000_000),  //  8 ms
        preemptible: boolean_t(1)            // preemptible = safe, avoids starvation
    )

    // THREAD_TIME_CONSTRAINT_POLICY_COUNT and THREAD_PRECEDENCE_POLICY_COUNT are
    // C macros (sizeof(struct)/sizeof(integer_t)) unavailable in Swift — compute inline.
    let tcCount = mach_msg_type_number_t(
        MemoryLayout<thread_time_constraint_policy_data_t>.size / MemoryLayout<integer_t>.size)
    let tcResult = withUnsafeMutablePointer(to: &tcPolicy) { ptr in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(tcCount)) { iPtr in
            thread_policy_set(
                thread,
                thread_policy_flavor_t(THREAD_TIME_CONSTRAINT_POLICY),
                iPtr,
                tcCount
            )
        }
    }

    if tcResult == KERN_SUCCESS {
        if rtPolicyLogOnce {
            rtPolicyLogOnce = false
            fputs("[CortexEngine] QWEN_RT_THREAD=1 → THREAD_TIME_CONSTRAINT_POLICY applied (period=10ms, comp=5ms, constraint=8ms, preemptible)\n", stderr)
            fflush(stderr)
        }
        return
    }

    // Fallback: THREAD_PRECEDENCE_POLICY with maximum precedence (63)
    if rtPolicyLogOnce {
        fputs("[CortexEngine] THREAD_TIME_CONSTRAINT_POLICY failed (kr=\(tcResult)) — falling back to THREAD_PRECEDENCE_POLICY(63)\n", stderr)
    }
    var precPolicy = thread_precedence_policy_data_t(importance: 63)
    let precCount = mach_msg_type_number_t(
        MemoryLayout<thread_precedence_policy_data_t>.size / MemoryLayout<integer_t>.size)
    let precResult = withUnsafeMutablePointer(to: &precPolicy) { ptr in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(precCount)) { iPtr in
            thread_policy_set(
                thread,
                thread_policy_flavor_t(THREAD_PRECEDENCE_POLICY),
                iPtr,
                precCount
            )
        }
    }
    if precResult == KERN_SUCCESS {
        if rtPolicyLogOnce { rtPolicyLogOnce = false; fputs("[CortexEngine] QWEN_RT_THREAD=1 → THREAD_PRECEDENCE_POLICY(63) applied\n", stderr) }
    } else {
        if rtPolicyLogOnce { rtPolicyLogOnce = false; fputs("[CortexEngine] QWEN_RT_THREAD=1 → both Mach thread policies failed (kr=\(precResult)) — decode thread unchanged\n", stderr) }
    }
    fflush(stderr)
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. ISOLATED ANE ENGINE (SWIFT 6 RIGOR)
// ─────────────────────────────────────────────────────────────────────────────
private let configureANECocktailOnce: Void = {
    setenv("MPSGRAPH_ENABLE_ANE_REAL_TIME_QOS", "1", 1)
    setenv("MPSGRAPH_ENABLE_ANE_SYNC_RUN", "1", 1)
    setenv("ESPRESSO_PRIVATE_ANE_ALWAYS_COMPRESSED", "1", 1)
    setenv("ESPRESSO_ANEC_COMPRESS_SPARSE", "1", 1)
    setenv("ESPRESSO_PRIVATE_ANE_SRAM_PERSISTENCE", "1", 1)
    setenv("MPSGRAPH_ENABLE_ANE_PREEMPTION_DISABLE", "1", 1)
    setenv("MPSGRAPH_ENABLE_ANE_FW_TO_FW_SIGNAL", "1", 1)
    setenv("MPSGRAPH_ENABLE_PARALLEL_ENCODE", "1", 1)
    setenv("MPSGRAPH_ENABLE_SHAPE_SHIFTING_CACHE", "1", 1)
    setenv("MPSGRAPH_COMPILER_STITCHING", "1", 1)
}()

public final class CortexEngine: @unchecked Sendable {
    private var model: MLModel
    private var prefillModel: MLModel?            // Batched prefill (function: "prefill", batch 64)
    private var verifyModel: MLModel?             // Parallel verify (function: "verify", batch K=4)
    private var state: MLState
    private var verifyScratchState: MLState?
    private let predictionOptions: MLPredictionOptions
    private let cortexQueue = DispatchQueue(label: "cortex", qos: .userInteractive)

    // Decode (batch=1) tensors
    private let inputTokens: MLMultiArray
    private let positionIds: MLMultiArray
    private let currentPos: MLMultiArray
    private var causalMask: MLMultiArray

    private let inputTokensPtr: UnsafeMutablePointer<Int32>
    private let positionIdsPtr: UnsafeMutablePointer<Int32>
    private let currentPosPtr: UnsafeMutablePointer<Int32>
    private var causalMaskPtr: UnsafeMutablePointer<Float16>
    private var inferFeatureProvider: MLDictionaryFeatureProvider

    // Batched prefill (batch=64) tensors — allocated only if prefillModel is non-nil
    public static let prefillBatch = 64
    private static let perfLogsEnabled = (ProcessInfo.processInfo.environment["QWEN_PERF_LOGS"] ?? "0") != "0"
    public private(set) var prefillBatchSize = 64
    private var prefillInputTokens: MLMultiArray?
    private var prefillPositionIds: MLMultiArray?
    private var prefillCurrentPos: MLMultiArray?
    private var prefillCausalMask: MLMultiArray?
    private var prefillInputTokensPtr: UnsafeMutablePointer<Int32>?
    private var prefillPositionIdsPtr: UnsafeMutablePointer<Int32>?
    private var prefillCurrentPosPtr: UnsafeMutablePointer<Int32>?
    private var prefillCausalMaskPtr: UnsafeMutablePointer<Float16>?
    private var prefillFeatureProvider: MLDictionaryFeatureProvider?

    // Parallel verify (batch=4) tensors — allocated only if verifyModel is non-nil
    public private(set) var verifyBatchSize = 4
    private var verifyInputTokens: MLMultiArray?
    private var verifyPositionIds: MLMultiArray?
    private var verifyCurrentPos: MLMultiArray?
    private var verifyCausalMask: MLMultiArray?
    private var verifyInputTokensPtr: UnsafeMutablePointer<Int32>?
    private var verifyPositionIdsPtr: UnsafeMutablePointer<Int32>?
    private var verifyCurrentPosPtr: UnsafeMutablePointer<Int32>?
    private var verifyCausalMaskPtr: UnsafeMutablePointer<Float16>?
    private var verifyFeatureProvider: MLDictionaryFeatureProvider?
    
    public let kQueryLen: Int
    public var hasPrefillModel: Bool { prefillModel != nil }
    public var hasVerifyModel: Bool { verifyModel != nil }
    public var usesVerifyScratchState: Bool { verifyScratchState != nil }
    private var contextLength: Int
    private let vocabSize: Int = 151936
    private let eosTokens: Set<Int32> = [151645, 151643]
    
    // Argmax configuration
    private let chunkSize: Int
    private let numChunks: Int

    // EAGLE inline draft head statistics (reset per generate call)
    private var eagleAccepted: Int = 0
    private var eagleTotal: Int = 0

    // EAGLE Phase 2: standalone head + cached hidden states from last base infer
    private var eagleHeadModel: MLModel? = nil
    private var eagleH2: MLMultiArray? = nil
    private var eagleH14: MLMultiArray? = nil
    private var eagleH25: MLMultiArray? = nil

    // Medusa heads configuration
    private let numMedusaHeads: Int
    private let medusaEnabled: Bool
    private let vocabTrimMap: [Int32]?

    // Prompt-lookup (n-gram) decoding — head-free multi-token path (DEAD_CLAIMS_DISPUTED.md).
    // Opt-in via QWEN_ENABLE_LOOKAHEAD=1 (pair with QWEN_ENABLE_VERIFY=1 for the fast batch4
    // verify). Reuses the same accept/replay/commit machinery as Medusa; only the draft
    // SOURCE differs (n-gram suffix-match over context instead of trained heads).
    private let lookaheadEnabled: Bool
    private let promptLookup = PromptLookup()

    // 1024 context models and states
    private var model1024: MLModel? = nil

    private var state1024: MLState? = nil
    // Signaled once the 1024 model has finished loading (or failed). The hot-swap
    // path waits briefly on this when generation approaches the 512 boundary so
    // a slow cold ANE compile doesn't cause early termination at context limit.
    private let preload1024Done = DispatchSemaphore(value: 0)
    private var preload1024Started = false

    @inline(__always)
    private func perfLog(_ message: @autoclosure () -> String) {
        guard Self.perfLogsEnabled else { return }
        fputs(message() + "\n", stderr)
        fflush(stderr)
    }

    /// Distribution summary for the decode hot loop. A tight spread means the code is the
    /// cost; a wide spread with a low minimum means the host (scheduler / memory bandwidth)
    /// is stealing the tail — that distinction decides whether to optimise code or the box.
    private func decodeStats(_ label: String, _ xs: [Double]) -> String {
        guard !xs.isEmpty else { return "\(label)=none" }
        let s = xs.sorted()
        func q(_ p: Double) -> Double {
            s[min(s.count - 1, max(0, Int((Double(s.count - 1) * p).rounded())))]
        }
        return String(format: "%@ min=%.2f p50=%.2f p90=%.2f p99=%.2f max=%.2f n=%d",
                      label, s[0], q(0.50), q(0.90), q(0.99), s[s.count - 1], s.count)
    }

    public init(modelURL: URL, useGPU: Bool = false, cpuOnly: Bool = false) throws {
        // ANE real-time QoS cocktail – kills XPC dispatch latency (unconditional to prevent any regressions)
        _ = configureANECocktailOnce

        // Dynamically optimize CoreML/MPSGraph/ANE flags for monolithic execution.
        // NOTE: MPSGRAPH_ENABLE_ANE_SYNC_RUN=1 synchronously blocks the calling thread during
        // Neural Engine execution. This prevents thread switching overhead and maximizes single-thread
        // latency/throughput, but requires that the daemon serializes and queues all incoming
        // requests to prevent head-of-line blocking on the global thread pool.
        // Honor the A/B gate set in main.swift — if QWEN_INFER_FLAGS=off,
        // do NOT re-set the flags here (otherwise the flagless baseline is contaminated).
        if (ProcessInfo.processInfo.environment["QWEN_INFER_FLAGS"] ?? "on") != "off" {
            setenv("ANEC_SPECIALIZE_LATENCY", "1", 1)
            setenv("ESPRESSO_ANEC_REQUIRED_SPARSITY_FOR_COMPRESSION", "0.1", 1)
            fputs("[CortexEngine] ANE monolithic execution cocktail set in environment.\n", stderr)
        } else {
            fputs("[CortexEngine] QWEN_INFER_FLAGS=off — skipping ANE flag setup (A/B baseline).\n", stderr)
        }

        // FLOOR-REDUCTION LEVER (DEAD_CLAIMS_DISPUTED.md #2): the ~16ms floor is mostly
        // software-addressable (IOSurface/KV-copy overhead, XPC is only ~0.1ms). Apple's
        // specializationStrategy=.fastPrediction trades compile time/memory for lower
        // per-prediction latency. Gated for clean A/B: QWEN_FAST_PREDICTION=0 disables.
        // macOS 15+ only. Pairs with MLState (already resident) toward a ~6-9ms floor.
        let useFastPred = ProcessInfo.processInfo.environment["QWEN_FAST_PREDICTION"] != "0"
        func applyFastPred(_ c: MLModelConfiguration) {
            if useFastPred, #available(macOS 15.0, *) {
                c.optimizationHints.specializationStrategy = .fastPrediction
                // Decode shapes are static (seq=1, fixed ctx) → tell the ANE driver so it
                // skips per-call shape-dispatch validation. ChunkedCortexEngine already does
                // this; CortexEngine was missing it. (CortexEngine perfection swarm, Rank 1.)
                c.optimizationHints.reshapeFrequency = .infrequent
            }
        }

        let config = MLModelConfiguration()
        config.computeUnits = cpuOnly ? .cpuOnly : (useGPU ? .cpuAndGPU : .cpuAndNeuralEngine)
        config.allowLowPrecisionAccumulationOnGPU = true // Applicable if fallback
        applyFastPred(config)

        // Try to load as a multi-function model (full_lut4) — both `infer` and `prefill`
        // share one MLState. Fall back to single-function (legacy `lut4`) if not available.
        // QWEN_FORCE_SINGLE_FUNCTION=1 skips multi-function loading entirely (for models
        // that have multiple functions in MIL but lack the protobuf function descriptions,
        // which causes an uncatchable fatalError in MLModel).
        let forceSingleFunction = ProcessInfo.processInfo.environment["QWEN_FORCE_SINGLE_FUNCTION"] == "1"
        var loadedAsMultiFunction = false
        if forceSingleFunction {
            // Single-function forced — load default function (first in MIL).
            self.model = try MLModel(contentsOf: modelURL, configuration: config)
            self.prefillModel = nil
            self.verifyModel = nil
            self.state = model.makeState()
            self.verifyScratchState = nil
            fputs("[CortexEngine] Loaded as single-function model (QWEN_FORCE_SINGLE_FUNCTION=1) — prefill will be sequential\n", stderr)
        } else {
        do {
            let inferConfig = MLModelConfiguration()
            inferConfig.computeUnits = config.computeUnits
            inferConfig.allowLowPrecisionAccumulationOnGPU = true
            inferConfig.functionName = "infer"
            applyFastPred(inferConfig)
            let inferModelTry = try MLModel(contentsOf: modelURL, configuration: inferConfig)

            let prefillConfig = MLModelConfiguration()
            prefillConfig.computeUnits = config.computeUnits
            prefillConfig.allowLowPrecisionAccumulationOnGPU = true
            prefillConfig.functionName = "prefill"
            applyFastPred(prefillConfig)
            let prefillModelTry = try MLModel(contentsOf: modelURL, configuration: prefillConfig)

            // Verify multi-function is OPT-IN. The K=4 verify export shipped through
            // Anemll/anemll/models/qwen2_5_model.py's prefill path has not passed the
            // bit-correctness gate (K_QUERY_VERIFY_HANDOFF.md §4), and the most recent
            // build_verify_K4 model emitted garbage tokens. To prevent the speculative
            // path from silently using a corrupt verify function, require an explicit
            // opt-in until the gate passes:  export QWEN_ENABLE_VERIFY=1
            let verifyEnabled = ProcessInfo.processInfo.environment["QWEN_ENABLE_VERIFY"] == "1"
            var verifyModelTry: MLModel? = nil
            if verifyEnabled {
                let verifyConfig = MLModelConfiguration()
                verifyConfig.computeUnits = config.computeUnits
                verifyConfig.allowLowPrecisionAccumulationOnGPU = true
                verifyConfig.functionName = "verify"
                applyFastPred(verifyConfig)
                do {
                    verifyModelTry = try MLModel(contentsOf: modelURL, configuration: verifyConfig)
                } catch {
                    fputs("[CortexEngine] Verify model load failed: \(error.localizedDescription)\n", stderr)
                    verifyModelTry = nil
                }
            }

            self.model = inferModelTry
            self.prefillModel = prefillModelTry
            self.verifyModel = verifyModelTry
            self.state = prefillModelTry.makeState()  // either function can mint compatible state
            self.verifyScratchState = prefillModelTry.makeState()
            loadedAsMultiFunction = true
            if verifyModelTry != nil {
                fputs("[CortexEngine] Loaded as multi-function model: infer + prefill + verify ⚠️ EXPERIMENTAL (QWEN_ENABLE_VERIFY=1; not bit-correctness-validated)\n", stderr)
            } else if verifyEnabled {
                fputs("[CortexEngine] Loaded as multi-function model: infer + prefill — verify function requested but not present in this mlmodelc\n", stderr)
            } else {
                fputs("[CortexEngine] Loaded as multi-function model: infer + prefill (batched prompt prefill enabled)\n", stderr)
            }
        } catch {
            // Single-function fallback (legacy)
            fputs("[CortexEngine] Multi-function load failed: \(error.localizedDescription)\n", stderr)
            self.model = try MLModel(contentsOf: modelURL, configuration: config)
            self.prefillModel = nil
            self.verifyModel = nil
            self.state = model.makeState()
            self.verifyScratchState = nil
            fputs("[CortexEngine] Loaded as single-function model (infer only) — prefill will be sequential\n", stderr)
            fputs("[CortexEngine] ⚠️  No 'prefill' function in this artifact: prompt prefill costs one full dispatch per prompt token (measured 10.67s vs 2.78s on a 413-token prompt). Use an export that includes it.\n", stderr)
        }
        } // end if forceSingleFunction
        _ = loadedAsMultiFunction
        
        // Read causal_mask shape dynamically from the model description
        var ctxLen = 512
        if let maskDesc = self.model.modelDescription.inputDescriptionsByName["causal_mask"],
           let shape = maskDesc.multiArrayConstraint?.shape,
           shape.count >= 4 {
            ctxLen = shape[3].intValue
        }
        self.contextLength = ctxLen
        
        // Read sequence length K dynamically from input_ids description (supports K-query target models)
        var kQuery = 1
        if let inputDesc = self.model.modelDescription.inputDescriptionsByName["input_ids"],
           let shape = inputDesc.multiArrayConstraint?.shape,
           shape.count >= 2 {
            kQuery = shape[1].intValue
        }
        self.kQueryLen = kQuery
        
        self.predictionOptions = MLPredictionOptions()
        if #available(macOS 14.0, *) {
            // Leave default to allow .cpuAndNeuralEngine
        } else {
            // Fallback for older macOS
        }
        
        let maskShape = [1, 1, NSNumber(value: kQueryLen), NSNumber(value: contextLength)]
        self.causalMask = try MLMultiArray(shape: maskShape, dataType: .float16)
        self.causalMaskPtr = self.causalMask.dataPointer.assumingMemoryBound(to: Float16.self)
        
        let negInf = Float16(-40000.0)
        for i in 0..<(contextLength * kQueryLen) {
            self.causalMaskPtr[i] = negInf
        }
        
        self.inputTokens = try MLMultiArray(shape: [1, NSNumber(value: kQueryLen)], dataType: .int32)
        self.inputTokensPtr = self.inputTokens.dataPointer.assumingMemoryBound(to: Int32.self)
        
        self.positionIds = try MLMultiArray(shape: [NSNumber(value: kQueryLen)], dataType: .int32)
        self.positionIdsPtr = self.positionIds.dataPointer.assumingMemoryBound(to: Int32.self)
        
        self.currentPos = try MLMultiArray(shape: [NSNumber(value: kQueryLen)], dataType: .int32)
        self.currentPosPtr = self.currentPos.dataPointer.assumingMemoryBound(to: Int32.self)
        self.inferFeatureProvider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": self.inputTokens,
            "position_ids": self.positionIds,
            "causal_mask": self.causalMask,
            "current_pos": self.currentPos
        ])
        
        // Read argmax shape dynamically from the model
        let outDesc = self.model.modelDescription.outputDescriptionsByName["argmax_val"]
        if let shape = outDesc?.multiArrayConstraint?.shape {
            if let lastDim = shape.last {
                self.numChunks = lastDim.intValue
                self.chunkSize = self.vocabSize / self.numChunks
            } else {
                fputs("[CortexEngine] Error: empty output shape\n", stderr)
                self.numChunks = 1187
                self.chunkSize = 128
            }
        } else {
            self.numChunks = 1187 // Fallback for 151936 / 128
            self.chunkSize = 128
        }

        // Detect Medusa heads from model output descriptions
        var detectedHeads = 0
        let outDescs = self.model.modelDescription.outputDescriptionsByName
        // New format: single fp16 logits tensor [H, 1, vocab_size] — argmax done in Swift
        if let logitsShape = outDescs["medusa_logits"]?.multiArrayConstraint?.shape,
           logitsShape.count >= 1 {
            detectedHeads = logitsShape[0].intValue
        // Legacy merged format: single medusa_idx tensor with shape [H, 16]
        } else if let mergedShape = outDescs["medusa_idx"]?.multiArrayConstraint?.shape,
           mergedShape.count >= 1 {
            detectedHeads = mergedShape[0].intValue
        } else {
            // Oldest format: individual medusa_idx_1, medusa_idx_2, etc.
            while outDescs["medusa_idx_\(detectedHeads + 1)"] != nil {
                detectedHeads += 1
            }
        }
        self.numMedusaHeads = detectedHeads
        let forceDisableMedusa = ProcessInfo.processInfo.environment["QWEN_NO_MEDUSA"] == "1"
        self.medusaEnabled = detectedHeads > 0 && !forceDisableMedusa
        if detectedHeads > 0 {
            if forceDisableMedusa {
                fputs("[CortexEngine] Detected \(detectedHeads) Medusa heads but QWEN_NO_MEDUSA=1 → disabled\n", stderr)
            } else {
                fputs("[CortexEngine] Detected \(detectedHeads) Medusa heads in model\n", stderr)
            }
        }

        // Prompt-lookup decoding: opt-in, and only meaningful with a batch verify/prefill
        // function (else the verify call falls back to slow sequential). Medusa takes
        // precedence in the decode loop if both are somehow enabled.
        let lookaheadRequested = ProcessInfo.processInfo.environment["QWEN_ENABLE_LOOKAHEAD"] == "1"
        self.lookaheadEnabled = lookaheadRequested && !(detectedHeads > 0 && !forceDisableMedusa)
        if lookaheadRequested {
            let haveBatch = (self.verifyModel != nil) || (self.prefillModel != nil)
            fputs("[CortexEngine] Prompt-lookup decoding \(self.lookaheadEnabled ? "ENABLED" : "requested but suppressed (Medusa active)")\(haveBatch ? "" : " ⚠️ no batch verify/prefill fn — will be slow")\n", stderr)
        }

        // Load VocabTrim map if it exists
        let mapURL = modelURL.deletingLastPathComponent().appendingPathComponent("vocab_trim_map.json")
        if let data = try? Data(contentsOf: mapURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let allowed = json["allowed_tokens"] as? [Int] {
            self.vocabTrimMap = allowed.map { Int32($0) }
            fputs("[CortexEngine] Loaded VocabTrim map with \(allowed.count) tokens\n", stderr)
        } else {
            self.vocabTrimMap = nil
        }

        // Allocate prefill batch tensors if multi-function model is available
        if let pf = self.prefillModel {
            var B = CortexEngine.prefillBatch
            if let inputDesc = pf.modelDescription.inputDescriptionsByName["input_ids"],
               let shape = inputDesc.multiArrayConstraint?.shape, shape.count >= 2 {
                B = shape[1].intValue
            }
            self.prefillBatchSize = B
            fputs("[CortexEngine] Detected prefill batch size from model constraint: \(B)\n", stderr); fflush(stderr)

            let pInput = try MLMultiArray(shape: [1, NSNumber(value: B)], dataType: .int32)
            let pPos   = try MLMultiArray(shape: [NSNumber(value: B)], dataType: .int32)
            let pCurr  = try MLMultiArray(shape: [1], dataType: .int32)
            let pMask  = try MLMultiArray(shape: [1, 1, NSNumber(value: B), NSNumber(value: contextLength)], dataType: .float16)
            self.prefillInputTokens = pInput
            self.prefillPositionIds = pPos
            self.prefillCurrentPos  = pCurr
            self.prefillCausalMask  = pMask
            self.prefillInputTokensPtr = pInput.dataPointer.assumingMemoryBound(to: Int32.self)
            self.prefillPositionIdsPtr = pPos.dataPointer.assumingMemoryBound(to: Int32.self)
            self.prefillCurrentPosPtr  = pCurr.dataPointer.assumingMemoryBound(to: Int32.self)
            self.prefillCausalMaskPtr  = pMask.dataPointer.assumingMemoryBound(to: Float16.self)
            self.prefillFeatureProvider = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": pInput,
                "position_ids": pPos,
                "causal_mask": pMask,
                "current_pos": pCurr
            ])
        }

        // Allocate verify batch tensors if verify model is available
        if let vf = self.verifyModel {
            var K = 4
            if let inputDesc = vf.modelDescription.inputDescriptionsByName["input_ids"],
               let shape = inputDesc.multiArrayConstraint?.shape, shape.count >= 2 {
                K = shape[1].intValue
            }
            self.verifyBatchSize = K
            fputs("[CortexEngine] Detected verify batch size (K) from model constraint: \(K)\n", stderr); fflush(stderr)

            let vInput = try MLMultiArray(shape: [1, NSNumber(value: K)], dataType: .int32)
            let vPos   = try MLMultiArray(shape: [NSNumber(value: K)], dataType: .int32)
            let vCurr  = try MLMultiArray(shape: [1], dataType: .int32)
            let vMask  = try MLMultiArray(shape: [1, 1, NSNumber(value: K), NSNumber(value: contextLength)], dataType: .float16)
            self.verifyInputTokens = vInput
            self.verifyPositionIds = vPos
            self.verifyCurrentPos  = vCurr
            self.verifyCausalMask  = vMask
            self.verifyInputTokensPtr = vInput.dataPointer.assumingMemoryBound(to: Int32.self)
            self.verifyPositionIdsPtr = vPos.dataPointer.assumingMemoryBound(to: Int32.self)
            self.verifyCurrentPosPtr  = vCurr.dataPointer.assumingMemoryBound(to: Int32.self)
            self.verifyCausalMaskPtr  = vMask.dataPointer.assumingMemoryBound(to: Float16.self)
            self.verifyFeatureProvider = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": vInput,
                "position_ids": vPos,
                "causal_mask": vMask,
                "current_pos": vCurr
            ])
        }


        // EAGLE Phase 2: load standalone head if QWEN_EAGLE_HEAD_PATH is set
        if let eaglePath = ProcessInfo.processInfo.environment["QWEN_EAGLE_HEAD_PATH"] {
            let eagleURL = URL(fileURLWithPath: eaglePath)
            if FileManager.default.fileExists(atPath: eaglePath) {
                do {
                    let ec = MLModelConfiguration()
                    ec.computeUnits = cpuOnly ? .cpuOnly : .cpuAndNeuralEngine
                    applyFastPred(ec)
                    self.eagleHeadModel = try MLModel(contentsOf: eagleURL, configuration: ec)
                    fputs("[CortexEngine] EAGLE Phase 2 head loaded: \(eaglePath)\n", stderr)
                    fflush(stderr)
                } catch {
                    fputs("[CortexEngine] EAGLE head load failed: \(error.localizedDescription)\n", stderr)
                }
            } else {
                fputs("[CortexEngine] QWEN_EAGLE_HEAD_PATH not found: \(eaglePath)\n", stderr)
            }
        }

        // 3. ANE PRE-WARM SEQUENCE (Async dispatched to not block init)
        // NOTE: the Mach RT policy is NOT applied here. cortexQueue.async runs this block
        // on a shared pool thread that returns to the pool immediately, so a per-thread
        // policy set here never reaches the decode thread. It is applied inside generate()
        // instead, on the thread that actually blocks for each ANE dispatch.
        cortexQueue.async { [weak self] in
            try? self?.executeSRAMBurnIn()
        }

        // 4. LAZY BACKGROUND 1024 MODEL PRELOADING
        // Skip entirely if the loaded model is already 1024+ context (no transition needed)
        let parentDir = modelURL.deletingLastPathComponent().deletingLastPathComponent()
        let url1024 = parentDir.appendingPathComponent("anemll_qwen25_1.5b_mono_ctx1024_lut4/qwen25_monolithic_lut4.mlmodelc")

        if ProcessInfo.processInfo.environment["QWEN_SKIP_PRELOAD_1024"] == "1" {
            fputs("[CortexEngine] QWEN_SKIP_PRELOAD_1024=1 → skipping 1024 background preload\n", stderr)
            fflush(stderr)
            self.preload1024Done.signal()
        } else if self.contextLength < 1024 && FileManager.default.fileExists(atPath: url1024.path) {
            self.preload1024Started = true
            // Start IMMEDIATELY (no +3s delay): cold ANE compile of a 1.5B/1024 model
            // can take 5–7 s; the prior delay caused short generations to hit the 512
            // limit before the model was ready, leaving the hot-swap useless.
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                guard let self = self else { return }
                let preloadStart = CFAbsoluteTimeGetCurrent()
                fputs("[CortexEngine] Preloading 1024-context model in background...\n", stderr)
                fflush(stderr)
                defer { self.preload1024Done.signal() }
                do {
                    let config1024 = MLModelConfiguration()
                    config1024.computeUnits = .cpuAndNeuralEngine
                    // Same ANE-driver hints as the primary model (perfection swarm Rank 1).
                    if useFastPred, #available(macOS 15.0, *) {
                        config1024.optimizationHints.specializationStrategy = .fastPrediction
                        config1024.optimizationHints.reshapeFrequency = .infrequent
                    }
                    let m1024 = try MLModel(contentsOf: url1024, configuration: config1024)
                    let s1024 = m1024.makeState()
                    // Direct assign (do NOT bounce through cortexQueue — it's held
                    // continuously by the active inference loop). Class-ref optional
                    // assignment is word-sized atomic on arm64.
                    self.model1024 = m1024
                    self.state1024 = s1024
                    let elapsed = (CFAbsoluteTimeGetCurrent() - preloadStart) * 1000
                    fputs("[CortexEngine] 1024-context model preloaded in \(String(format: "%.0f", elapsed))ms — hot-swap ready.\n", stderr)
                    fflush(stderr)
                } catch {
                    fputs("[CortexEngine] Failed to preload 1024 model: \(error.localizedDescription)\n", stderr)
                    fflush(stderr)
                }
            }
        }
    }
    
    /// Forces the closed-source Espresso driver to compile the hardware shader pipelines
    /// and lock the weights into the 16-Core ANE cache before the first query arrives.
    private func executeSRAMBurnIn() throws {
        for q in 0..<self.kQueryLen {
            inputTokensPtr[q] = 0
            positionIdsPtr[q] = 0
            currentPosPtr[q] = 0
            for i in 0..<contextLength {
                causalMaskPtr[q * contextLength + i] = 0.0
            }
        }
        
        _ = try model.prediction(from: inferFeatureProvider, using: state, options: predictionOptions)
        fputs("[CortexEngine] SRAM Burn-In Complete. ANE is hot.\n", stderr)
        fflush(stderr)
    }
    
    // ─────────────────────────────────────────────────────────────────────────────
    // 4. SYNCHRONOUS ZERO-COPY TOKEN EMISSION LOOP
    // ─────────────────────────────────────────────────────────────────────────────
    public func generate(promptIds: [Int32], maxTokens: Int, formatEnforcer: FormatEnforcer? = nil, onToken: @escaping (Int32) -> Void) throws -> ([Int32], [Double]) {
        try cortexQueue.sync {
            // DispatchQueue.sync runs this block on the CALLER's thread, and
            // MPSGRAPH_ENABLE_ANE_SYNC_RUN=1 blocks that same thread for the whole ANE
            // dispatch — so the scheduling hint belongs here, not on a throwaway
            // cortexQueue.async block (where it never affected decode).
            applyMachRTThreadPolicy()
            guard !promptIds.isEmpty else { return ([], []) }
            
            var featureProvider = self.inferFeatureProvider
            
            let zero = Float16(0.0)
            let negInf = Float16(-40000.0)
            
            // Initialization: fill mask with negInf — all positions masked, revealed one at a time
            for i in 0..<self.contextLength {
                self.causalMaskPtr[i] = negInf
            }
            
            // --- PRE-FILL ---
            let startPrefill = CFAbsoluteTimeGetCurrent()
            let prefillCount = max(0, promptIds.count - 1)  // last prompt token goes through decode infer
            let usePrefillModel = self.prefillModel != nil && ProcessInfo.processInfo.environment["QWEN_NO_PREFILL"] != "1"
            if usePrefillModel && prefillCount >= self.prefillBatchSize {
                // Batched prefill on the ANE — single forward per 64-token chunk.
                try self._prefillBatchedUnlocked(tokens: promptIds, count: prefillCount)
                // Mark the decode mask for every prefilled position so the
                // subsequent infer step can attend back to them.
                for i in 0..<prefillCount {
                    self.causalMaskPtr[i] = zero
                }
            } else if usePrefillModel && prefillCount > 0 {
                // Short prompt — still one batched call, with padding.
                try self._prefillBatchedUnlocked(tokens: promptIds, count: prefillCount)
                for i in 0..<prefillCount {
                    self.causalMaskPtr[i] = zero
                }
            } else {
                // Legacy sequential prefill (single-function model only).
                let trace = ProcessInfo.processInfo.environment["QWEN_TRACE"] == "1"
                if trace { fputs("[TRACE] seq-prefill start count=\(prefillCount) kQuery=\(self.kQueryLen) ctx=\(self.contextLength)\n", stderr); fflush(stderr) }
                for i in 0..<prefillCount {
                    let tid = promptIds[i]
                    try autoreleasepool {
                        for q in 0..<self.kQueryLen {
                            self.inputTokensPtr[q] = tid
                            self.positionIdsPtr[q] = Int32(i)
                            self.currentPosPtr[q] = Int32(i)
                            self.causalMaskPtr[q * self.contextLength + i] = zero
                        }
                        if trace { fputs("[TRACE] prefill i=\(i) predict...\n", stderr); fflush(stderr) }
                        _ = try self.model.prediction(from: featureProvider, using: self.state, options: self.predictionOptions)
                        if trace { fputs("[TRACE] prefill i=\(i) done\n", stderr); fflush(stderr) }
                    }
                }
                if trace { fputs("[TRACE] seq-prefill complete\n", stderr); fflush(stderr) }
            }
            let prefillElapsed = (CFAbsoluteTimeGetCurrent() - startPrefill) * 1000
            perfLog("[CortexEngine] Prefill completed in \(String(format: "%.2f", prefillElapsed))ms (\(promptIds.count) tokens, mode=\(self.prefillModel != nil ? "batched-64" : "sequential"))")
            
            // --- DECODE ---
            self.eagleAccepted = 0
            self.eagleTotal = 0
            let startDecode = CFAbsoluteTimeGetCurrent()
            let collectDecodeTimes = (ProcessInfo.processInfo.environment["QWEN_COLLECT_DECODE_TIMES"] ?? "1") != "0"
            var currentToken = promptIds.last!
            var currentPosition = promptIds.count - 1
            var generatedIds: [Int32] = []
            var decodeTimes: [Double] = []
            // Post-dispatch glue (argmax + hidden-state extraction + enforcer) — measured
            // separately so a slow loop can be attributed to the ANE call or to our code.
            var extractTimes: [Double] = []
            var activeEnforcer = formatEnforcer  // mutable copy — persists prefix cursor across steps
            generatedIds.reserveCapacity(maxTokens)
            if collectDecodeTimes {
                decodeTimes.reserveCapacity(maxTokens)
                extractTimes.reserveCapacity(maxTokens)
            }
            
            while generatedIds.count < maxTokens {
                // Hot-swap check (only when preload is active)
                if self.preload1024Started
                    && self.model1024 == nil
                    && self.contextLength < 1024
                    && currentPosition >= self.contextLength - 4 {
                    let waitStart = CFAbsoluteTimeGetCurrent()
                    let result = self.preload1024Done.wait(timeout: .now() + 8.0)
                    let waited = (CFAbsoluteTimeGetCurrent() - waitStart) * 1000
                    if result == .success {
                        // Re-signal so any subsequent wait succeeds (idempotent).
                        self.preload1024Done.signal()
                        perfLog("[CortexEngine] Hot-swap preload wait completed in \(String(format: "%.0f", waited))ms.")
                    } else {
                        perfLog("[CortexEngine] Hot-swap preload wait timed out after \(String(format: "%.0f", waited))ms — will hit context limit.")
                    }
                }
                let currentMaxContext = self.model1024 != nil ? 1024 : self.contextLength
                if currentPosition >= currentMaxContext {
                    fputs("[CortexEngine] Reached context limit (\(currentMaxContext))!\n", stderr)
                    fflush(stderr)
                    break
                }
                
                // If we reach the current model's limit (512), transition to 1024-context model
                if currentPosition >= self.contextLength {
                    if let m1024 = self.model1024, let s1024 = self.state1024 {
                        perfLog("[CortexEngine] Sequence reached \(currentPosition) tokens. Transitioning to 1024 context...")
                        let startTransition = CFAbsoluteTimeGetCurrent()
                        
                        // Copy the KV cache states dynamically
                        self.transitionState(from: self.state, to: s1024, currentPosition: currentPosition)
                        
                        // Swap model references
                        self.model = m1024
                        self.state = s1024
                        self.contextLength = 1024
                        
                        // Re-allocate causal mask for 1024 context
                        let maskShape = [1, 1, 1, NSNumber(value: 1024)]
                        self.causalMask = try MLMultiArray(shape: maskShape, dataType: .float16)
                        self.causalMaskPtr = self.causalMask.dataPointer.assumingMemoryBound(to: Float16.self)
                        
                        // Populate mask: past positions are unmasked (0.0), future are masked (-10000.0)
                        for i in 0..<1024 {
                            self.causalMaskPtr[i] = i < currentPosition ? zero : negInf
                        }
                        
                        // Update the feature provider
                        featureProvider = try MLDictionaryFeatureProvider(dictionary: [
                            "input_ids": self.inputTokens,
                            "position_ids": self.positionIds,
                            "causal_mask": self.causalMask,
                            "current_pos": self.currentPos
                        ])
                        self.inferFeatureProvider = featureProvider
                        
                        let transitionElapsed = (CFAbsoluteTimeGetCurrent() - startTransition) * 1000
                        perfLog("[CortexEngine] Transition complete in \(String(format: "%.2f", transitionElapsed))ms")
                    } else {
                        fputs("[CortexEngine] Reached 512 context limit but 1024 model is not ready!\n", stderr)
                        fflush(stderr)
                        break
                    }
                }

                // Apply thermal state backoff â skip (ANE manages its own thermals)

                if self.medusaEnabled {
                    let msStart = CFAbsoluteTimeGetCurrent()
                    let (primary, drafts) = try self._predictMedusaStepUnlocked(token: currentToken, position: currentPosition)
                    let msElapsed = (CFAbsoluteTimeGetCurrent() - msStart) * 1000
                    if collectDecodeTimes { decodeTimes.append(msElapsed) }

                    if eosTokens.contains(primary) { break }

                    generatedIds.append(primary)
                    onToken(primary)
                    currentToken = primary
                    currentPosition += 1

                    var accepted = 0
                    if !drafts.isEmpty {
                        let vStart = CFAbsoluteTimeGetCurrent()
                        accepted = try self.acceptMedusaDraftsUnlocked(primary: primary, drafts: drafts, at: currentPosition - 1)
                        let vElapsed = (CFAbsoluteTimeGetCurrent() - vStart) * 1000
                        if collectDecodeTimes && accepted > 0 {
                            for _ in 0..<accepted { decodeTimes.append(vElapsed / Double(accepted)) }
                        }

                        let basePos = currentPosition - 1

                        // 1. Prune accepted for EOS before replay
                        var actualAccepted = 0
                        for i in 0..<accepted {
                            if eosTokens.contains(drafts[i]) { break }
                            actualAccepted += 1
                        }

                        // 2. Unmask causal mask for accepted draft positions
                        for i in 0..<actualAccepted {
                            self.causalMaskPtr[basePos + 1 + i] = Float16(0.0)
                        }

                        // 3. Replay primary + every accepted draft through single-step predict (all tokens, not skipping the last)
                        let allTokens = [primary] + drafts.prefix(actualAccepted)
                        for i in 0..<allTokens.count {
                            let replayPos = basePos + 1 + i
                            _ = try self._predictSingleStepUnlocked(
                                token: allTokens[i], position: replayPos)
                        }

                        // 4. Emit accepted tokens (no EOS check needed — already pruned)
                        for i in 0..<actualAccepted {
                            let tok = drafts[i]
                            generatedIds.append(tok)
                            onToken(tok)
                            currentToken = tok
                            currentPosition += 1
                        }
                    }
                    perfLog("[CortexEngine] Medusa step: primary=\(primary) drafts=\(drafts.count) accepted=\(accepted)")
                } else if self.lookaheadEnabled {
                    // PROMPT-LOOKUP decode (head-free): `primary` from a normal single-step
                    // (commits currentToken's KV), draft continuation from n-gram suffix-match
                    // over the context. Drafts verified in ONE batched dispatch, accepted
                    // prefix committed by replay — IDENTICAL machinery to the Medusa block
                    // above; only the draft SOURCE differs (no trained heads, no conv budget).
                    let laStart = CFAbsoluteTimeGetCurrent()
                    let primary = try self._predictSingleStepUnlocked(token: currentToken, position: currentPosition)
                    let laElapsed = (CFAbsoluteTimeGetCurrent() - laStart) * 1000
                    if collectDecodeTimes { decodeTimes.append(laElapsed) }

                    if eosTokens.contains(primary) { break }

                    generatedIds.append(primary)
                    onToken(primary)
                    currentToken = primary
                    currentPosition += 1

                    // n-gram candidates for the next positions (history now ends with primary)
                    let drafts = self.promptLookup.candidates(history: promptIds + generatedIds, k: self.verifyBatchSize)

                    var accepted = 0
                    if !drafts.isEmpty {
                        let vStart = CFAbsoluteTimeGetCurrent()
                        accepted = try self.acceptMedusaDraftsUnlocked(primary: primary, drafts: drafts, at: currentPosition - 1)
                        let vElapsed = (CFAbsoluteTimeGetCurrent() - vStart) * 1000
                        if collectDecodeTimes && accepted > 0 {
                            for _ in 0..<accepted { decodeTimes.append(vElapsed / Double(accepted)) }
                        }

                        let basePos = currentPosition - 1

                        // Prune accepted drafts at EOS before replay
                        var actualAccepted = 0
                        for i in 0..<accepted {
                            if eosTokens.contains(drafts[i]) { break }
                            actualAccepted += 1
                        }

                        // Unmask causal mask for accepted draft positions
                        for i in 0..<actualAccepted {
                            self.causalMaskPtr[basePos + 1 + i] = Float16(0.0)
                        }

                        // Replay primary + accepted drafts through single-step to commit KV
                        let allTokens = [primary] + drafts.prefix(actualAccepted)
                        for i in 0..<allTokens.count {
                            let replayPos = basePos + 1 + i
                            _ = try self._predictSingleStepUnlocked(token: allTokens[i], position: replayPos)
                        }

                        // Emit accepted tokens (already EOS-pruned)
                        for i in 0..<actualAccepted {
                            let tok = drafts[i]
                            generatedIds.append(tok)
                            onToken(tok)
                            currentToken = tok
                            currentPosition += 1
                        }
                    }
                    perfLog("[CortexEngine] Lookahead step: primary=\(primary) drafts=\(drafts.count) accepted=\(accepted)")
                } else {
                    if ProcessInfo.processInfo.environment["QWEN_TRACE"] == "1" { fputs("[TRACE] decode pos=\(currentPosition) tok=\(currentToken) predict...\n", stderr); fflush(stderr) }
                    for i in 0..<self.kQueryLen {
                        self.inputTokensPtr[i] = currentToken
                        self.positionIdsPtr[i] = Int32(currentPosition)
                        self.currentPosPtr[i] = Int32(currentPosition)
                        self.causalMaskPtr[i * self.contextLength + currentPosition] = zero
                    }

                    let start = CFAbsoluteTimeGetCurrent()
                    let output = try self.model.prediction(from: featureProvider, using: self.state, options: self.predictionOptions)
                    let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                    if collectDecodeTimes { decodeTimes.append(elapsed) }

                    // Free-Rider probe bank readout (feature-gated by QWEN_PROBE_VECTOR=1).
                    // Non-destructive: no-op when the flag is off or the model lacks probe_vector.
                    self.reportProbeVector(from: output)

                    let rawToken = self.extractArgmax(from: output)
                    let hasHiddens = self.extractH2H14H25(from: output)
                    if collectDecodeTimes {
                        // Everything the loop does after the dispatch returns, up to the
                        // point where the next dispatch can begin.
                        extractTimes.append((CFAbsoluteTimeGetCurrent() - start) * 1000 - elapsed)
                    }

                    // Format enforcement: prefix forcing or token banning.
                    // activeEnforcer persists across decode steps so prefix cursor advances correctly.
                    // .skip and .injectText both suppress the token from the output stream
                    // while keeping the KV cache coherent (rawToken is still fed as next input).
                    let nextToken: Int32
                    if activeEnforcer != nil {
                        switch activeEnforcer!.next(rawToken: rawToken) {
                        case .emit(let t):
                            nextToken = t
                        case .skip:
                            // Suppress output; feed rawToken to KV so state stays coherent.
                            currentToken = rawToken; currentPosition += 1; continue
                        case .injectText(let msg):
                            // Log violation; suppress banned token from output stream.
                            fputs("[FormatEnforcer] \(msg)\n", stderr)
                            currentToken = rawToken; currentPosition += 1; continue
                        }
                    } else {
                        nextToken = rawToken
                    }
                    // When prefix forcing overrides rawToken, EAGLE draft was conditioned on rawToken
                    // → skip EAGLE Phase 2 for this step (incoherent KV if we verify the forced token).
                    let eagleCoherentThisStep = (nextToken == rawToken)

                    // EAGLE Phase 2: standalone head + batch verify for 2-token speculative step.
                    // Requires: Phase 2 base model (h2/h14/h25 outputs) + standalone eagle head +
                    // verify model (batch ≥ 2). Falls back to Phase 1 tracking when unavailable.
                    if hasHiddens, eagleCoherentThisStep, self.eagleHeadModel != nil, self.verifyModel != nil,
                       let eagleDraft = self.runEagleStandaloneHead() {
                        self.eagleTotal += 1
                        let verifyPos = currentPosition + 1
                        // Verify [nextToken, eagleDraft] at positions T+1, T+2 on main state.
                        // On reject, verify's KV at T+2 is a ghost that the next infer overwrites.
                        let verifyPreds = try self._verifyCandidatesUnlocked(
                            tokens: [nextToken, eagleDraft],
                            positions: [verifyPos, verifyPos + 1]
                        )
                        // Commit T+1 to base causal mask — verify wrote correct KV there.
                        self.causalMaskPtr[verifyPos] = zero
                        if !verifyPreds.isEmpty && verifyPreds[0] == eagleDraft {
                            // ACCEPT: eagleDraft matched; emit 2 tokens.
                            self.eagleAccepted += 1
                            if eosTokens.contains(nextToken) { break }
                            generatedIds.append(nextToken); onToken(nextToken)
                            if eosTokens.contains(eagleDraft) {
                                currentToken = eagleDraft; currentPosition += 2; break
                            }
                            generatedIds.append(eagleDraft); onToken(eagleDraft)
                            self.causalMaskPtr[verifyPos + 1] = zero  // commit T+2 too
                            currentToken = eagleDraft
                            currentPosition += 2
                        } else {
                            // REJECT: emit nextToken only; T+2 ghost KV heals next iteration.
                            if eosTokens.contains(nextToken) { break }
                            generatedIds.append(nextToken); onToken(nextToken)
                            currentToken = nextToken
                            currentPosition += 1
                        }
                    } else {
                        // Phase 1 EAGLE inline tracking (no speedup) or plain single-step.
                        if let eagleDraft = self.extractEagleArgmax(from: output) {
                            self.eagleTotal += 1
                            if eagleDraft == nextToken { self.eagleAccepted += 1 }
                        }
                        if eosTokens.contains(nextToken) { break }
                        generatedIds.append(nextToken); onToken(nextToken)
                        currentToken = nextToken
                        currentPosition += 1
                    }
                }
            }
            let decodeElapsed = (CFAbsoluteTimeGetCurrent() - startDecode) * 1000
            let avgDecode = generatedIds.isEmpty ? 0 : (decodeElapsed / Double(generatedIds.count))
            let eagleRate = self.eagleTotal > 0 ? Double(self.eagleAccepted) / Double(self.eagleTotal) : -1
            let eagleStr = eagleRate >= 0 ? String(format: " | EAGLE %.1f%% (%d/%d)", eagleRate * 100, self.eagleAccepted, self.eagleTotal) : ""
            perfLog("[CortexEngine] Decode loop completed in \(String(format: "%.2f", decodeElapsed))ms (Avg: \(String(format: "%.2f", avgDecode))ms/tok, Speed: \(String(format: "%.1f", generatedIds.isEmpty ? 0 : (Double(generatedIds.count) / (decodeElapsed / 1000.0)))) tok/s)\(eagleStr)")
            if collectDecodeTimes && !decodeTimes.isEmpty {
                perfLog("[CortexEngine] \(self.decodeStats("dispatch_ms", decodeTimes))")
                perfLog("[CortexEngine] \(self.decodeStats("glue_ms", extractTimes))")
            }
            
            return (generatedIds, decodeTimes)
        }
    }
    
    /// Raw C-pointer sub-slice memory copier to transition KV cache activations in microseconds.
    /// Optimized to perform head-contiguous block copies (reduces memcpy dispatches 512x).
    private func transitionState(from srcState: MLState, to dstState: MLState, currentPosition: Int) {
        let start = CFAbsoluteTimeGetCurrent()
        for name in self.stateNames {
            srcState.withMultiArray(for: name) { srcArr in
                dstState.withMultiArray(for: name) { dstArr in
                    let srcStrides = srcArr.strides.map { $0.intValue }
                    let dstStrides = dstArr.strides.map { $0.intValue }
                    
                    let srcPtr = srcArr.dataPointer.assumingMemoryBound(to: Float16.self)
                    let dstPtr = dstArr.dataPointer.assumingMemoryBound(to: Float16.self)
                    
                    let batchCount = srcArr.shape[0].intValue
                    let heads = srcArr.shape[1].intValue
                    let headDim = srcArr.shape[3].intValue
                    
                    for b in 0..<batchCount {
                        for h in 0..<heads {
                            let srcOffset = b * srcStrides[0] + h * srcStrides[1]
                            let dstOffset = b * dstStrides[0] + h * dstStrides[1]
                            let bytesToCopy = currentPosition * headDim * 2 // Float16 is 2 bytes
                            memcpy(dstPtr + dstOffset, srcPtr + srcOffset, bytesToCopy)
                        }
                    }
                }
            }
        }
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        perfLog("[CortexEngine] Microsecond-level head-contiguous KV hotswap complete: \(String(format: "%.3f", elapsed))ms")
    }

    /// Full MLState clone for speculative verify scratch execution.
    /// Unlike transitionState(), this copies the entire state buffer so a
    /// speculative verify pass can mutate scratch state without poisoning live KV.
    private func cloneState(from srcState: MLState, to dstState: MLState) {
        for name in self.stateNames {
            srcState.withMultiArray(for: name) { srcArr in
                dstState.withMultiArray(for: name) { dstArr in
                    let byteCount = srcArr.count * Self.stride(for: srcArr.dataType)
                    memcpy(dstArr.dataPointer, srcArr.dataPointer, byteCount)
                }
            }
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────────
    // 5. KV CACHE STATE PERSISTENCE (§3 in ane-optimizations-v2.md)
    //    Save / restore MLState buffers across idle eviction cycles so wake-up
    //    skips the multi-second CPU prefill. Uses the supported MLState closure
    //    API — no raw SPI, survives recompile.
    // ─────────────────────────────────────────────────────────────────────────────

    /// State buffer names exposed by the stateful model (macOS 15+).
    public lazy var stateNames: [String] = {
        if #available(macOS 15.0, *) {
            return Array(model.modelDescription.stateDescriptionsByName.keys)
        }
        return []
    }()

    /// Element stride in bytes for an MLMultiArrayDataType.
    private static func stride(for dt: MLMultiArrayDataType) -> Int {
        switch dt {
        case .float16: return 2
        case .float32: return 4
        case .double, .float64: return 8
        case .int32:   return 4
        case .int8:    return 1
        @unknown default: return MemoryLayout<Float32>.stride
        }
    }

    /// Serialize every MLState buffer to `dir/<name>.bin` plus a `header.json`
    /// stamped with `modelSignature`. Atomic at the directory level — old caches
    /// are not overwritten in place, they are replaced.
    public func saveKVCache(to dir: URL, modelSignature: String) throws {
        guard #available(macOS 15.0, *) else { return }
        guard !stateNames.isEmpty else {
            fputs("[CortexEngine] saveKVCache: no MLState buffers exposed\n", stderr)
            return
        }
        try cortexQueue.sync {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            var shapes: [String: [Int]] = [:]
            for name in stateNames {
                state.withMultiArray(for: name) { arr in
                    let byteCount = arr.count * Self.stride(for: arr.dataType)
                    let raw = arr.dataPointer.assumingMemoryBound(to: UInt8.self)
                    let data = Data(bytesNoCopy: raw, count: byteCount, deallocator: .none)
                    try? data.write(to: dir.appendingPathComponent("\(name).bin"), options: .atomic)
                    shapes[name] = arr.shape.map { $0.intValue }
                }
            }
            let header: [String: Any] = [
                "model_signature": modelSignature,
                "shapes": shapes,
                "saved_at": Date().timeIntervalSince1970
            ]
            let headerData = try JSONSerialization.data(withJSONObject: header, options: [.prettyPrinted])
            try headerData.write(to: dir.appendingPathComponent("header.json"), options: .atomic)
            fputs("[CortexEngine] KV cache saved to \(dir.path) (\(stateNames.count) buffers)\n", stderr)
            fflush(stderr)
        }
    }

    /// Reset the causal mask to all -inf and zero the position-tracking inputs.
    /// MUST be called before a fresh sequence of `predictSingleStep` calls, otherwise
    /// the model attends to stale/zeroed slots from previous runs (or from the SRAM
    /// burn-in) and produces garbage. The normal `generate()` path does this inline;
    /// callers that drive `predictSingleStep` directly (e.g. SpeculativeEngine) need
    /// to invoke this once at the top of their sequence.
    public func resetCausalMask() {
        cortexQueue.sync {
            self._resetCausalMaskUnlocked()
        }
    }

    /// Same as resetCausalMask, but assumes caller already holds the cortexQueue
    /// (via `withQueueLock`). Must NOT be called from outside the queue.
    public func _resetCausalMaskUnlocked() {
        let negInf = Float16(-40000.0)
        for i in 0..<self.contextLength {
            self.causalMaskPtr[i] = negInf
        }
        self.inputTokensPtr[0] = 0
        self.positionIdsPtr[0] = 0
        self.currentPosPtr[0] = 0
    }

    /// Prefill target state for direct decode drivers such as SpeculativeEngine.
    /// Caller must already hold cortexQueue.
    public func _prefillPromptUnlocked(prompt: [Int32]) throws -> Int {
        guard prompt.count > 1 else { return 0 }

        let prefillTokens = Array(prompt.dropLast())
        
        if self.prefillModel != nil {
            try self._prefillBatchedUnlocked(tokens: prefillTokens)
        } else {
            for (i, tid) in prefillTokens.enumerated() {
                _ = try self._predictSingleStepUnlocked(token: tid, position: i)
            }
        }

        let zero = Float16(0.0)
        for i in 0..<prefillTokens.count {
            self.causalMaskPtr[i] = zero
        }
        return prefillTokens.count
    }

    public func predictSingleStepThrowing(token: Int32, position: Int) throws -> Int32 {
        return try cortexQueue.sync {
            try self._predictSingleStepUnlocked(token: token, position: position)
        }
    }

    /// Batched prefill via the multi-function model's `prefill` function (batch=64).
    /// Splits `tokens` into 64-token chunks. The final partial chunk is padded with
    /// token id 0 and the corresponding mask rows are fully masked (-inf) so the
    /// padding contributes nothing to attention.
    /// Mutates the shared MLState (KV cache). Caller MUST already hold cortexQueue.
    public func _prefillBatchedUnlocked(tokens: [Int32], count requestedCount: Int? = nil) throws {
        guard let pModel = self.prefillModel,
              let pInput = self.prefillInputTokens,
              let pPos = self.prefillPositionIds,
              let pCurr = self.prefillCurrentPos,
              let pMask = self.prefillCausalMask,
              let pInputPtr = self.prefillInputTokensPtr,
              let pPosPtr = self.prefillPositionIdsPtr,
              let pCurrPtr = self.prefillCurrentPosPtr,
              let pMaskPtr = self.prefillCausalMaskPtr
        else { return }

        let B = self.prefillBatchSize
        let ctx = self.contextLength       // 512 (or 1024 after transition)
        let negInf = Float16(-40000.0)
        let zero = Float16(0.0)

        let N = requestedCount.map { Swift.min(tokens.count, Swift.max(0, $0)) } ?? tokens.count
        var offset = 0
        while offset < N {
            try autoreleasepool {
                let chunkLen = Swift.min(B, N - offset)
                // Fill input_ids: real tokens, pad rest with 0
                for j in 0..<B {
                    pInputPtr[j] = j < chunkLen ? tokens[offset + j] : 0
                    pPosPtr[j]   = Int32(offset + j)  // monotonic positions even for padding
                }
                pCurrPtr[0] = Int32(offset)

                // Fill causal mask: pMask is [1,1,B,ctx]. For batch row j (representing
                // absolute position p = offset + j):
                //   - if j < chunkLen: unmask cols [0..p], mask cols [p+1..ctx-1]
                //   - if j >= chunkLen: fully masked (entire row is -inf) — padding has no influence
                for j in 0..<B {
                    let rowBase = j * ctx
                    if j < chunkLen {
                        let p = offset + j
                        let unmaskedCount = Swift.min(p + 1, ctx)
                        for k in 0..<unmaskedCount {
                            pMaskPtr[rowBase + k] = zero
                        }
                        for k in unmaskedCount..<ctx {
                            pMaskPtr[rowBase + k] = negInf
                        }
                    } else {
                        for k in 0..<ctx {
                            pMaskPtr[rowBase + k] = negInf
                        }
                    }
                }

                let provider: MLDictionaryFeatureProvider
                if let cached = self.prefillFeatureProvider {
                    provider = cached
                } else {
                    provider = try MLDictionaryFeatureProvider(dictionary: [
                        "input_ids": pInput,
                        "position_ids": pPos,
                        "causal_mask": pMask,
                        "current_pos": pCurr
                    ])
                }

                // Single ANE forward for 64 tokens — KV cache writes to shared MLState.
                os_signpost(.begin, log: aneSignpostLog, name: "ANE Prefill Batch", "offset=%d count=%d", offset, chunkLen)
                _ = try pModel.prediction(from: provider, using: self.state, options: self.predictionOptions)
                os_signpost(.end, log: aneSignpostLog, name: "ANE Prefill Batch", "offset=%d count=%d", offset, chunkLen)

                offset += chunkLen
            }
        }
    }

    /// Hot-path step that assumes the caller already holds the cortexQueue
    /// (acquired via `withQueueLock`). Saves the queue-hop overhead per call —
    /// critical for tight loops like speculative decoding where the call-site
    /// runs hundreds of predictSingleStep ops back-to-back. Calling this without
    /// holding the queue is undefined behaviour.
    public func _predictSingleStepUnlocked(token: Int32, position: Int) throws -> Int32 {
        self.inputTokensPtr[0] = token
        self.positionIdsPtr[0] = Int32(position)
        self.currentPosPtr[0] = Int32(position)
        self.causalMaskPtr[position] = Float16(0.0)

        os_signpost(.begin, log: aneSignpostLog, name: "ANE Predict", "pos=%d", position)
        let output = try self.model.prediction(from: self.inferFeatureProvider, using: self.state, options: self.predictionOptions)
        os_signpost(.end, log: aneSignpostLog, name: "ANE Predict", "pos=%d", position)
        return self.extractArgmax(from: output)
    }

    /// Medusa-augmented single-step inference: returns the primary token + N draft tokens
    /// from the same forward pass. The drafts are the medusa head predictions for
    /// positions t+2 .. t+N+1 (predicted from hidden state at position t).
    public func _predictMedusaStepUnlocked(token: Int32, position: Int) throws -> (primary: Int32, drafts: [Int32]) {
        self.inputTokensPtr[0] = token
        self.positionIdsPtr[0] = Int32(position)
        self.currentPosPtr[0] = Int32(position)
        self.causalMaskPtr[position] = Float16(0.0)

        os_signpost(.begin, log: aneSignpostLog, name: "ANE Predict+Medusa", "pos=%d", position)
        let output = try self.model.prediction(from: self.inferFeatureProvider, using: self.state, options: self.predictionOptions)
        os_signpost(.end, log: aneSignpostLog, name: "ANE Predict+Medusa", "pos=%d", position)

        let primary = self.extractArgmax(from: output)
        let drafts = self.extractMedusaDrafts(from: output)
        return (primary, drafts)
    }

    /// Run medusa draft-verify acceptance: verify N medusa draft tokens against
    /// the base model's lm_head predictions. Returns the number of accepted tokens.
    ///
    /// Verifies only the drafts (batch=4) — the primary token was already generated
    /// by _predictMedusaStepUnlocked and is verified by replay, not batch verify.
    private func acceptMedusaDraftsUnlocked(primary: Int32, drafts: [Int32], at position: Int) throws -> Int {
        let K = drafts.count
        guard K > 0 else { return 1 }

        // Require a batch verify model for draft acceptance.  Without one the
        // sequential fallback in _verifyCandidatesUnlocked corrupts the live KV
        // cache (advances by K positions but only acceptCount are consumed) and
        // is always slower than single-token decode.  Accept zero drafts so the
        // primary token (already generated in _predictMedusaStepUnlocked) is
        // used and the KV remains pristine.
        guard self.verifyModel != nil || self.prefillModel != nil else {
            return 0
        }

        // Build verify inputs: primary + drafts prefix(K-1) — length is exactly K, fits batch size perfectly
        let verifyTokens = [primary] + drafts.prefix(K - 1)
        let verifyPositions = (0..<K).map { position + 1 + $0 }

        let verifyResults = try self._verifyCandidatesUnlocked(tokens: verifyTokens, positions: verifyPositions)

        // Find longest matching prefix
        var acceptCount = 0
        for i in 0..<K {
            guard i < verifyResults.count, verifyResults[i] == drafts[i] else { break }
            acceptCount += 1
        }

        return acceptCount
    }

    /// Acquire the cortexQueue for a batch of operations. Inside the closure use
    /// `_predictSingleStepUnlocked` and `_resetCausalMaskUnlocked` — they skip
    /// the per-call queue.sync and keep the ANE warm across the whole batch.
    /// The serial queue cannot be re-entered, so calls to the locked APIs
    /// (predictSingleStep, resetCausalMask, generate) from inside the closure
    /// will deadlock.
    public func withQueueLock<T>(_ block: () throws -> T) rethrows -> T {
        return try cortexQueue.sync(execute: block)
    }

    public func unmaskCausalMask(upTo position: Int) {
        cortexQueue.sync {
            let zero = Float16(0.0)
            for i in 0..<position {
                self.causalMaskPtr[i] = zero
            }
        }
    }

    public func injectKVCache(data: Data) {
        cortexQueue.sync {
            self.state.withMultiArray(for: "model_model_kv_cache_0") { arr in
                let destPtr = arr.dataPointer.assumingMemoryBound(to: Float16.self)
                data.withUnsafeBytes { rawBuf in
                    let srcPtr = rawBuf.baseAddress!.assumingMemoryBound(to: Float16.self)
                    let count = rawBuf.count / 2
                    memcpy(destPtr, srcPtr, count * 2)
                }
            }
        }
    }

    /// Restore MLState buffers from `dir`. Returns `false` and wipes the cache
    /// if the signature or shapes do not match the live model — never silently
    /// binds incompatible bytes.
    @discardableResult
    public func restoreKVCache(from dir: URL, modelSignature: String) -> Bool {
        guard #available(macOS 15.0, *) else { return false }
        guard !stateNames.isEmpty else { return false }
        let headerURL = dir.appendingPathComponent("header.json")
        guard FileManager.default.fileExists(atPath: headerURL.path) else { return false }

        do {
            let headerData = try Data(contentsOf: headerURL)
            guard let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any],
                  let sig = header["model_signature"] as? String,
                  sig == modelSignature else {
                fputs("[CortexEngine] KV cache signature mismatch — dropping stale cache\n", stderr)
                try? FileManager.default.removeItem(at: dir)
                return false
            }
            let storedShapes = (header["shapes"] as? [String: [Int]]) ?? [:]

            return try cortexQueue.sync { () -> Bool in
                for name in stateNames {
                    let path = dir.appendingPathComponent("\(name).bin")
                    guard FileManager.default.fileExists(atPath: path.path) else { return false }
                    let bytes = try Data(contentsOf: path)
                    var shapeOK = true
                    state.withMultiArray(for: name) { arr in
                        let liveShape = arr.shape.map { $0.intValue }
                        if let stored = storedShapes[name], stored != liveShape {
                            shapeOK = false
                            return
                        }
                        let byteCount = arr.count * Self.stride(for: arr.dataType)
                        bytes.withUnsafeBytes { raw in
                            guard let base = raw.baseAddress else { return }
                            memcpy(arr.dataPointer, base, Swift.min(raw.count, byteCount))
                        }
                    }
                    if !shapeOK {
                        fputs("[CortexEngine] KV cache shape mismatch on '\(name)' — dropping\n", stderr)
                        try? FileManager.default.removeItem(at: dir)
                        return false
                    }
                }
                fputs("[CortexEngine] KV cache restored from \(dir.path)\n", stderr)
                fflush(stderr)
                return true
            }
        } catch {
            fputs("[CortexEngine] KV cache restore failed: \(error.localizedDescription)\n", stderr)
            return false
        }
    }

    /// Pointer-level ArgMax computation across chunked output
    /// Extract all Medusa head predictions from the model output.
    /// Returns an array of token IDs, one per Medusa head, in order 1..N.
    private func extractMedusaDrafts(from output: MLFeatureProvider) -> [Int32] {
        guard medusaEnabled else { return [] }

        // New format: medusa_logits [H, 1, vocab_size] fp16 — argmax in Swift
        if let logitsArr = output.featureValue(for: "medusa_logits")?.multiArrayValue {
            let H = logitsArr.shape[0].intValue
            let vocabSize = logitsArr.shape[logitsArr.shape.count - 1].intValue
            let logitsPtr = logitsArr.dataPointer.assumingMemoryBound(to: Float16.self)
            var drafts = [Int32]()
            drafts.reserveCapacity(H)
            for k in 0..<H {
                let base = k * vocabSize
                var bestVal: Float16 = -Float16.greatestFiniteMagnitude
                var bestIdx = 0
                for i in 0..<vocabSize {
                    let v = logitsPtr[base + i]
                    if v > bestVal { bestVal = v; bestIdx = i }
                }
                // VocabTrim remap: for a trimmed model, bestIdx indexes the trimmed (8192)
                // vocab — map it back to the real Qwen token id. Full-vocab models (map==nil)
                // use the raw index unchanged.
                if let map = self.vocabTrimMap, bestIdx < map.count {
                    drafts.append(map[bestIdx])
                } else {
                    drafts.append(Int32(bestIdx))
                }
            }
            return drafts
        }

        // Legacy format: medusa_idx [H, C] int32 + medusa_val [H, C] fp16
        guard let idxArr = output.featureValue(for: "medusa_idx")?.multiArrayValue,
              let valArr = output.featureValue(for: "medusa_val")?.multiArrayValue else {
            return []
        }
        let H = idxArr.shape[0].intValue
        let C = idxArr.shape[1].intValue
        let valPtr = valArr.dataPointer.assumingMemoryBound(to: Float16.self)
        let idxPtr = idxArr.dataPointer.assumingMemoryBound(to: Int32.self)
        var drafts = [Int32]()
        drafts.reserveCapacity(H)
        for k in 0..<H {
            let offset = k * C
            var bestVal: Float16 = -Float16.greatestFiniteMagnitude
            var bestChunk = 0
            for chunk in 0..<C {
                let v = valPtr[offset + chunk]
                if v > bestVal { bestVal = v; bestChunk = chunk }
            }
            drafts.append(Int32(bestChunk * self.chunkSize) + idxPtr[offset + bestChunk])
        }
        return drafts
    }

    /// Feature-gated (QWEN_PROBE_VECTOR=1). Reads the grafted Free-Rider probe bank
    /// output `probe_vector` [1,3,1,1] fp16 and prints RAW logits per token. Raw only —
    /// no sigmoid/calibration/policy. No-op when the flag is off; prints one debug line
    /// (probe mode only) if the model has no probe_vector. Never crashes generation.
    private static let probeModeEnabled = ProcessInfo.processInfo.environment["QWEN_PROBE_VECTOR"] == "1"
    private func reportProbeVector(from output: MLFeatureProvider) {
        guard CortexEngine.probeModeEnabled else { return }
        guard let pv = output.featureValue(for: "probe_vector")?.multiArrayValue else {
            fputs("[probe] probe_vector absent from model outputs (not a probe-grafted model)\n", stderr)
            fflush(stderr); return
        }
        let ptr = pv.dataPointer.assumingMemoryBound(to: Float16.self)
        let n = pv.count
        let r = n > 0 ? Float(ptr[0]) : Float.nan
        let p = n > 1 ? Float(ptr[1]) : Float.nan
        let u = n > 2 ? Float(ptr[2]) : Float.nan
        fputs(String(format: "[probe] raw logits (uncalibrated): refusal=%.3f pii=%.3f uncertainty=%.3f\n", r, p, u), stderr)
        fflush(stderr)
    }

    private func extractArgmax(from output: MLFeatureProvider) -> Int32 {
        if self.kQueryLen > 1 {
            return self.extractBatchArgmax(from: output, count: 1)[0]
        }
        
        guard let vals = output.featureValue(for: "argmax_val")?.multiArrayValue,
              let idxs = output.featureValue(for: "argmax_idx")?.multiArrayValue else {
            // Missing or malformed argmax outputs (model output-spec mismatch) → fall back to logits argmax.
            return self.extractBatchArgmaxFromLogits(from: output, count: 1)[0]
        }

        let valPtr = vals.dataPointer.assumingMemoryBound(to: Float16.self)
        let idxPtr = idxs.dataPointer.assumingMemoryBound(to: Int32.self)

        var bestVal: Float16 = -Float16.greatestFiniteMagnitude
        var bestChunk = 0

        for chunk in 0..<self.numChunks {
            let v = valPtr[chunk]
            if v > bestVal {
                bestVal = v
                bestChunk = chunk
            }
        }

        return Int32(bestChunk * self.chunkSize) + idxPtr[bestChunk]
    }

    /// EAGLE inline draft head argmax. Returns nil if model lacks --eagle-heads-dir output.
    /// eagle_argmax_idx mirrors argmax_idx: numChunks entries each offset by chunkSize.
    private func extractEagleArgmax(from output: MLFeatureProvider) -> Int32? {
        guard let vals = output.featureValue(for: "eagle_argmax_val")?.multiArrayValue,
              let idxs = output.featureValue(for: "eagle_argmax_idx")?.multiArrayValue else {
            return nil
        }
        let valPtr = vals.dataPointer.assumingMemoryBound(to: Float16.self)
        let idxPtr = idxs.dataPointer.assumingMemoryBound(to: Int32.self)
        var bestVal: Float16 = -Float16.greatestFiniteMagnitude
        var bestChunk = 0
        for chunk in 0..<self.numChunks {
            let v = valPtr[chunk]
            if v > bestVal { bestVal = v; bestChunk = chunk }
        }
        return Int32(bestChunk * self.chunkSize) + idxPtr[bestChunk]
    }

    /// Cache h2/h14/h25 hidden states from base model output for EAGLE Phase 2.
    /// Returns true if the base model exposes these outputs (Phase 2 build).
    /// Tries both naming conventions: h2_out/h14_out/h25_out (older) and
    /// hidden_states_21/93/159 (Anemll v0.3.5 expose-hiddens export).
    @discardableResult
    private func extractH2H14H25(from output: MLFeatureProvider) -> Bool {
        if ProcessInfo.processInfo.environment["QWEN_EAGLE_DEBUG"] == "1" {
            let names = output.featureNames.joined(separator: ", ")
            fputs("[EAGLE-DEBUG] extractH2H14H25 called. Output names: \(names)\n", stderr)
        }
        let h2  = output.featureValue(for: "h2_out")?.multiArrayValue
            ?? output.featureValue(for: "hidden_states_21")?.multiArrayValue
        let h14 = output.featureValue(for: "h14_out")?.multiArrayValue
            ?? output.featureValue(for: "hidden_states_93")?.multiArrayValue
        let h25 = output.featureValue(for: "h25_out")?.multiArrayValue
            ?? output.featureValue(for: "hidden_states_159")?.multiArrayValue
        guard let h2, let h14, let h25 else {
            return false
        }
        self.eagleH2 = h2; self.eagleH14 = h14; self.eagleH25 = h25
        return true
    }

    /// Run the standalone EAGLE head on cached h2/h14/h25 to draft the next-next token.
    /// Standalone head expects [1, 1536, 1, 1] inputs; base model outputs flattened [H, 1, 1].
    /// Total element count is identical so we memcpy the raw data into the reshaped buffer.
    private func runEagleStandaloneHead() -> Int32? {
        guard let ehm = self.eagleHeadModel,
              let h2src = self.eagleH2, let h14src = self.eagleH14, let h25src = self.eagleH25
        else { return nil }
        let H = 1536
        guard let inp2  = try? MLMultiArray(shape: [1, NSNumber(value: H), 1, 1] as [NSNumber], dataType: .float16),
              let inp14 = try? MLMultiArray(shape: [1, NSNumber(value: H), 1, 1] as [NSNumber], dataType: .float16),
              let inp25 = try? MLMultiArray(shape: [1, NSNumber(value: H), 1, 1] as [NSNumber], dataType: .float16)
        else { return nil }
        let bytes = H * MemoryLayout<Float16>.size
        memcpy(inp2.dataPointer,  h2src.dataPointer,  bytes)
        memcpy(inp14.dataPointer, h14src.dataPointer, bytes)
        memcpy(inp25.dataPointer, h25src.dataPointer, bytes)
        guard let prov = try? MLDictionaryFeatureProvider(dictionary: [
            "h2":  MLFeatureValue(multiArray: inp2),
            "h14": MLFeatureValue(multiArray: inp14),
            "h25": MLFeatureValue(multiArray: inp25)
        ]), let out = try? ehm.prediction(from: prov, options: self.predictionOptions)
        else { return nil }
        return self.extractEagleArgmax(from: out)
    }

    /// Hot-path batch verification step that loads K candidate tokens and unmasks
    /// their causal mask slots, executing exactly ONE prediction pass on the ANE.
    /// This is the key algorithm to unlock multi-token speculative scaling.
    public func _verifyCandidatesUnlocked(tokens: [Int32], positions: [Int]) throws -> [Int32] {
        let K = tokens.count
        let executionState: MLState
        if let scratch = self.verifyScratchState {
            self.cloneState(from: self.state, to: scratch)
            executionState = scratch
        } else {
            executionState = self.state
        }
        
        if let vfModel = self.verifyModel,
           let vInput = self.verifyInputTokens,
           let vPos = self.verifyPositionIds,
           let vCurr = self.verifyCurrentPos,
           let vMask = self.verifyCausalMask {
            
            guard let inputTokensPtr = self.verifyInputTokensPtr,
                  let positionIdsPtr = self.verifyPositionIdsPtr,
                  let currentPosPtr = self.verifyCurrentPosPtr,
                  let causalMaskPtr = self.verifyCausalMaskPtr else {
                fputs("[CortexEngine] verify buffers not allocated\n", stderr)
                return Array(repeating: 151643, count: K)
            }

            let B = self.verifyBatchSize
            let zero = Float16(0.0)
            let negInf = Float16(-40000.0)
            
            // 1. Write tokens and positions to the verify buffers
            for i in 0..<B {
                if i < K {
                    inputTokensPtr[i] = tokens[i]
                    positionIdsPtr[i] = Int32(positions[i])
                } else {
                    // Padding: set dummy token 0 at monotonic, non-duplicate positions
                    // to prevent hardware pipeline stalls/hangs on the ANE.
                    inputTokensPtr[i] = 0
                    positionIdsPtr[i] = Int32(positions[0] + i)
                }
            }
            currentPosPtr[0] = Int32(positions[0])
            
            // 2. Build the causal mask for the verify model [1, 1, K, 512]
            // Initialize everything to -inf (fully masked)
            for i in 0..<(B * self.contextLength) {
                causalMaskPtr[i] = negInf
            }
            
            // Unmask the valid slots for our K candidate tokens
            for i in 0..<K {
                let rowBase = i * self.contextLength
                // Unmask past history up to positions[0] - 1
                for j in 0..<positions[0] {
                    causalMaskPtr[rowBase + j] = zero
                }
                // Unmask the previous candidate positions in this row
                for j in 0...i {
                    causalMaskPtr[rowBase + positions[j]] = zero
                }
            }
            
            let provider: MLDictionaryFeatureProvider
            if let cached = self.verifyFeatureProvider {
                provider = cached
            } else {
                provider = try MLDictionaryFeatureProvider(dictionary: [
                    "input_ids": vInput,
                    "position_ids": vPos,
                    "causal_mask": vMask,
                    "current_pos": vCurr
                ])
            }
            
            let output = try vfModel.prediction(from: provider, using: executionState, options: self.predictionOptions)
            
            if output.featureValue(for: "argmax_val") != nil {
                return self.extractBatchArgmax(from: output, count: K)
            } else {
                return self.extractBatchArgmaxFromLogits(from: output, count: K)
            }
        } else {
            // Sequential verification on the infer model using the scratch state.
            //
            // WHY NOT the prefill64 model:
            //   The prefill function is traced with seq_len=64 (IN_PREFILL=True), so its
            //   per-token KV representations differ from the seq_len=1 infer model for many
            //   tokens (different `if seq_len==1` branches baked as constants).  Accept
            //   decisions made on prefill-model logits diverge from the infer-model's greedy
            //   sequence.  When those accepted tokens are replayed through the infer model
            //   (the outer loop calls _predictSingleStepUnlocked), the KV state diverges,
            //   producing wrong logits on subsequent steps — observed as early EOS at 10
            //   tokens vs 16 baseline tokens.
            //
            // WHY scratch state (executionState):
            //   The infer model writes KV at every position it is called on.  We verify K
            //   candidates sequentially; only acceptCount ≤ K of them are legitimate.  The
            //   outer replay block (_predictSingleStepUnlocked × actualAccepted+1) will
            //   re-write the accepted positions into self.state correctly.  We must not let
            //   the verify run corrupt self.state with rejected-candidate KV entries.
            //
            // WHY causal mask restore:
            //   self.causalMaskPtr is a 1-D [1,1,1,contextLength] buffer shared with the
            //   live decode loop.  We temporarily unmask verify positions here; the outer
            //   loop re-unmasks exactly actualAccepted positions during replay.  Any extra
            //   unmasked slots left by verify would be seen by the next single-step decode
            //   as valid context, causing attention divergence.  Restore to negInf so the
            //   outer replay block is the sole authority on the causal mask.
            let negInf = Float16(-40000.0)
            var verifiedPositions = [Int]()
            var predictions = [Int32]()
            for i in 0..<K {
                self.inputTokensPtr[0] = tokens[i]
                self.positionIdsPtr[0] = Int32(positions[i])
                self.currentPosPtr[0] = Int32(positions[i])
                self.causalMaskPtr[positions[i]] = Float16(0.0)
                verifiedPositions.append(positions[i])

                let output = try self.model.prediction(from: self.inferFeatureProvider, using: executionState, options: self.predictionOptions)
                predictions.append(self.extractArgmax(from: output))
            }
            // Restore causal mask entries that were temporarily unmasked for verify.
            for pos in verifiedPositions {
                self.causalMaskPtr[pos] = negInf
            }
            return predictions
        }
    }

    /// Promote verify scratch state to live state after a speculative verify cycle
    /// that did not require rollback repair (all-accepted or reject-at-last).
    /// Caller MUST already hold cortexQueue.
    public func _adoptVerifyScratchStateUnlocked() {
        guard let scratch = self.verifyScratchState else { return }
        let live = self.state
        self.state = scratch
        self.verifyScratchState = live
    }

    /// Pointer-level ArgMax computation across chunked output for batch sequence length K
    private func extractBatchArgmax(from output: MLFeatureProvider, count: Int) -> [Int32] {
        guard let vals = output.featureValue(for: "argmax_val")?.multiArrayValue,
              let idxs = output.featureValue(for: "argmax_idx")?.multiArrayValue else {
            fputs("[CortexEngine] Error: argmax_val or argmax_idx not found in batch verify output.\n", stderr)
            return Array(repeating: 151643, count: count)
        }
        
        let valPtr = vals.dataPointer.assumingMemoryBound(to: Float16.self)
        let idxPtr = idxs.dataPointer.assumingMemoryBound(to: Int32.self)
        
        // Strides to support multi-token output
        let chunkStride = Int(truncating: vals.strides[1]) // stride of the chunk dimension
        let seqStride = Int(truncating: vals.strides[vals.strides.count - 1]) // stride of the sequence dimension
        
        let idxChunkStride = Int(truncating: idxs.strides[1])
        let idxSeqStride = Int(truncating: idxs.strides[idxs.strides.count - 1])
        
        var predictions = [Int32]()
        predictions.reserveCapacity(count)
        
        for s in 0..<count {
            var bestVal: Float16 = -Float16.greatestFiniteMagnitude
            var bestChunk = 0
            
            for chunk in 0..<self.numChunks {
                let v = valPtr[chunk * chunkStride + s * seqStride]
                if v > bestVal {
                    bestVal = v
                    bestChunk = chunk
                }
            }
            
            let bestIdx = idxPtr[bestChunk * idxChunkStride + s * idxSeqStride]
            predictions.append(Int32(bestChunk * self.chunkSize) + bestIdx)
        }
        
        return predictions
    }

    /// Pointer-level ArgMax computation across chunked output for batch sequence length K using raw split logits
    private func extractBatchArgmaxFromLogits(from output: MLFeatureProvider, count: Int) -> [Int32] {
        var predictions = [Int32]()
        predictions.reserveCapacity(count)

        var logitArrays: [MLMultiArray] = []
        for i in 1...16 {
            if let arr = output.featureValue(for: "logits\(i)")?.multiArrayValue {
                logitArrays.append(arr)
            }
        }
        guard logitArrays.count == 16 else {
            fputs("[CortexEngine] Error: Could not find all 16 logits for batch argmax.\n", stderr)
            return Array(repeating: 151643, count: count)
        }

        let vocabSize = 151936
        var logitsAll = [Float](repeating: 0.0, count: vocabSize)
        var maxVal: Float = 0
        var maxIdx: vDSP_Length = 0
        var vBuf: [Float] = []

        for s in 0..<count {
            var offset = 0
            for chunk in 0..<16 {
                let arr = logitArrays[chunk]
                let ptr = arr.dataPointer.assumingMemoryBound(to: Float16.self)
                // STRIDE-AWARE READ (7-agent verify-math consensus, high-confidence root cause):
                // prefill logits1..16 are transpose(perm=[0,2,1]) of [1,9496,64]. If CoreML
                // returns a lazy-transposed view, strides[1]=1 (not 9496) and strides[2]=64
                // (not 1) — the old hardcoded element-stride=1 read WRONG MEMORY → garbage
                // argmax → wrong verify accepts → divergence. Reading the actual strides is
                // correct for BOTH layouts (contiguous: strideSeq=9496,strideElem=1).
                let strideSeq = Int(truncating: arr.strides[1])
                let strideElem = Int(truncating: arr.strides[2])
                let seqOffset = s * strideSeq
                let chunkSize = 9496
                if s == 0 && chunk == 0 {
                    fputs("[verify-logits] shape=\(arr.shape) strides=\(arr.strides) strideSeq=\(strideSeq) strideElem=\(strideElem)\n", stderr)
                }

                if vBuf.count < chunkSize { vBuf = [Float](repeating: 0.0, count: chunkSize) }
                ptr.advanced(by: seqOffset).withMemoryRebound(to: Int16.self, capacity: chunkSize * max(strideElem, 1)) { int16Ptr in
                    vDSP_vflt16(int16Ptr, vDSP_Stride(max(strideElem, 1)), &vBuf, 1, vDSP_Length(chunkSize))
                }
                memcpy(&logitsAll[offset], vBuf, chunkSize * MemoryLayout<Float>.size)
                offset += chunkSize
            }

            vDSP_maxvi(logitsAll, 1, &maxVal, &maxIdx, vDSP_Length(vocabSize))
            predictions.append(Int32(maxIdx))
        }

        return predictions
    }
}

