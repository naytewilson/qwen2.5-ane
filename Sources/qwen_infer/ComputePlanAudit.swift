import CoreML
import Foundation

// ─────────────────────────────────────────────────────────────────────────────
// MODEL PROBE
// ─────────────────────────────────────────────────────────────────────────────
// Answers one question about an artifact: will this build load it as a
// MULTI-FUNCTION model (infer + prefill), or will prompt prefill fall back to
// one dispatch per prompt token? Function presence in the MIL program is not
// sufficient — CoreML only accepts `configuration.functionName` when it
// classifies the artifact as an ML Program, which is what this actually tests.
//
// Usage:  qwen-ane --probe-model [path/to/model.mlmodelc]

func runModelProbe(modelPath: String) {
    let raw = modelPath.hasPrefix("/")
        ? modelPath
        : FileManager.default.currentDirectoryPath + "/" + modelPath
    let url = URL(fileURLWithPath: raw)

    guard FileManager.default.fileExists(atPath: url.path) else {
        print("[probe] no such path: \(url.path)")
        exit(2)
    }

    // Report the declared opset, because it determines whether CoreML will accept
    // `.functionName` at all on the installed OS.
    let milURL = url.appendingPathComponent("model.mil")
    if let mil = try? String(contentsOf: milURL, encoding: .utf8),
       let firstFunc = mil.split(separator: "\n").first(where: { $0.contains("func ") }) {
        let trimmed = firstFunc.trimmingCharacters(in: .whitespaces)
        print("[probe] \(trimmed.prefix(80))")
    }

    func load(_ function: String?) -> MLModel? {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        cfg.allowLowPrecisionAccumulationOnGPU = true
        cfg.functionName = function
        do {
            return try MLModel(contentsOf: url, configuration: cfg)
        } catch {
            print("[probe] load(function=\(function ?? "default")) failed: \(error.localizedDescription)")
            return nil
        }
    }

    guard let infer = load("infer") else {
        print("[probe] VERDICT: artifact does not expose a loadable 'infer' function.")
        exit(1)
    }
    _ = infer

    guard let prefill = load("prefill") else {
        print("[probe] VERDICT: SINGLE-FUNCTION — prompt prefill runs one dispatch per token.")
        exit(1)
    }

    var batch = 0
    if let desc = prefill.modelDescription.inputDescriptionsByName["input_ids"],
       let shape = desc.multiArrayConstraint?.shape, shape.count >= 2 {
        batch = shape[1].intValue
    }
    print("[probe] VERDICT: MULTI-FUNCTION — batched prompt prefill available (batch \(batch)).")
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPUTE-PLAN AUDIT
// ─────────────────────────────────────────────────────────────────────────────
// ANE playbook step 5: verify placement empirically before claiming a speed
// result. If CoreML schedules any operation of the decode graph on CPU or GPU,
// that cost is per-dispatch, stable across runs, and invisible to host-load
// experiments — exactly the signature of a uniform per-token slowdown that no
// Swift-side optimisation can remove. The only fix at that point is the
// artifact (re-export / recompile), not the runtime.
//
// Usage:  qwen-ane --compute-plan [path/to/model.mlmodelc|.mlpackage]

func runComputePlanAudit(modelPath: String) async {
    let raw = modelPath.hasPrefix("/")
        ? modelPath
        : FileManager.default.currentDirectoryPath + "/" + modelPath
    var url = URL(fileURLWithPath: raw)

    guard FileManager.default.fileExists(atPath: url.path) else {
        print("[audit] no such path: \(url.path)")
        return
    }

    // A .mlpackage must be compiled before a compute plan can be loaded.
    if url.path.hasSuffix(".mlpackage") {
        print("[audit] compiling \(url.lastPathComponent) (this can take a while) …")
        do {
            url = try await MLModel.compileModel(at: url)
        } catch {
            print("[audit] compileModel failed: \(error.localizedDescription)")
            return
        }
    }

    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine

    print("[audit] loading compute plan for \(url.path) …")
    let plan: MLComputePlan
    do {
        plan = try await MLComputePlan.load(contentsOf: url, configuration: config)
    } catch {
        print("[audit] MLComputePlan.load failed: \(error.localizedDescription)")
        return
    }

    guard case let .program(program) = plan.modelStructure else {
        print("[audit] model is not an ML Program; per-operation placement is unavailable.")
        print("[audit] (Expected for .mlmodel neural-network artifacts.)")
        return
    }

    var perDevice: [String: Int] = [:]
    var total = 0

    func deviceLabel(_ device: MLComputeDevice?) -> String {
        guard let device else { return "unassigned" }
        switch device {
        case .neuralEngine: return "neuralEngine"
        case .gpu:          return "gpu"
        case .cpu:          return "cpu"
        @unknown default:   return "other"
        }
    }

    // Constant-folding nodes (const / constexpr_lut_to_dense) carry no device and carry no
    // per-inference cost — they are resolved at load. Everything else that is not on the
    // Neural Engine is real work executed on CPU/GPU for every dispatch.
    let compileTimeOps: Set<String> = ["const", "constexpr_lut_to_dense", "constexpr_affine_dequantize", "constexpr_blockwise_shift_scale"]

    // A plan only assigns devices for the function it was loaded for (the default one when
    // functionName is nil), so every function is audited against its own plan.
    for name in program.functions.keys.sorted() {
        let fnConfig = MLModelConfiguration()
        fnConfig.computeUnits = config.computeUnits
        fnConfig.functionName = name
        guard let fnPlan = try? await MLComputePlan.load(contentsOf: url, configuration: fnConfig),
              case let .program(fnProgram) = fnPlan.modelStructure,
              let fn = fnProgram.functions[name] else {
            print("[audit] function '\(name)': compute plan unavailable")
            continue
        }
        var fnPer: [String: Int] = [:]
        var fnOff: [(String, String)] = []
        var fnTotal = 0
        var fnCompileTime = 0

        func walk(_ block: MLModelStructure.Program.Block) {
            for op in block.operations {
                fnTotal += 1
                total += 1
                let label = deviceLabel(fnPlan.deviceUsage(for: op)?.preferred)
                perDevice[label, default: 0] += 1
                fnPer[label, default: 0] += 1
                if compileTimeOps.contains(op.operatorName) { fnCompileTime += 1 }
                if label != "neuralEngine" && !compileTimeOps.contains(op.operatorName) {
                    fnOff.append((op.operatorName, label))
                }
                for sub in op.blocks { walk(sub) }
            }
        }
        walk(fn.block)

        let runtimeOps = fnTotal - fnCompileTime
        print("[audit] function '\(name)': \(fnTotal) ops (\(fnCompileTime) compile-time constants, \(runtimeOps) runtime)")
        for (device, count) in fnPer.sorted(by: { $0.value > $1.value }) {
            print("[audit]     \(device): \(count)")
        }
        if fnOff.isEmpty {
            print("[audit]     → all runtime ops on the Neural Engine")
        } else {
            let names = Dictionary(grouping: fnOff, by: { "\($0.1): \($0.0)" })
                .map { "\($0.value.count)× \($0.key)" }
                .sorted()
            print("[audit]     → \(fnOff.count) runtime op(s) off-ANE:")
            for n in names.prefix(20) { print("[audit]         \(n)") }
        }
    }

    print("[audit] total operations (all functions): \(total)")
    for (device, count) in perDevice.sorted(by: { $0.value > $1.value }) {
        let pct = total > 0 ? Double(count) / Double(total) * 100.0 : 0
        print(String(format: "[audit]   %-13@ %5d  %5.1f%%", device as NSString, count, pct))
    }
}

