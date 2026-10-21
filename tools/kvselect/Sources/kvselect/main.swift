// kvselect — makes a batch-64 `prefill` function compile for the Neural Engine.
//
// The ANE compiler fails (status -14) on a slice_update into the MLState KV cache whose
// `begin` is a runtime value (current_pos) and whose write is wider than 1 along the
// sequence axis. Each such op in the target function is rewritten to static bounds:
//
//   row    = slice_by_index(cache, [l,0,0,0] → [l+1,·,·,·])            static
//   P      = cast(equal(pos_range[S,1] − k_range[1,K], cur), fp16)    one-hot [S,K], shared
//   keep   = 1 − reduce_sum(P, axis 1)                                [S,1], shared
//   placed = matmul(P, update[1,H,K,D])                               [1,H,S,D]
//   newrow = row·keep + placed                                        exact: one-hot ⇒ each
//                                                                     element is old or new
//   cache' = slice_update(cache, [l,0,0,0] → [l+1,·,·,·], newrow)       static
//
// The rewritten op keeps its output name, so every consumer is untouched. Weights and
// Manifest.json are copied byte-for-byte; only model.mlmodel is re-serialized.
//
// usage: kvselect <in.mlpackage> <out.mlpackage> [function=prefill]
import Foundation
import SwiftProtobuf

typealias Operation      = CoreML_Specification_MILSpec_Operation
typealias NamedValueType = CoreML_Specification_MILSpec_NamedValueType
typealias ValueType      = CoreML_Specification_MILSpec_ValueType
typealias TensorType     = CoreML_Specification_MILSpec_TensorType
typealias Dimension      = CoreML_Specification_MILSpec_Dimension
typealias MILValue       = CoreML_Specification_MILSpec_Value
typealias TensorValue    = CoreML_Specification_MILSpec_TensorValue
typealias Argument       = CoreML_Specification_MILSpec_Argument
typealias MILDataType    = CoreML_Specification_MILSpec_DataType
typealias Model          = CoreML_Specification_Model

func die(_ m: String) -> Never { FileHandle.standardError.write(("kvselect: " + m + "\n").data(using: .utf8)!); exit(1) }

// ── MIL protobuf builders ──
func dim(_ n: Int) -> Dimension { var c = Dimension.ConstantDimension(); c.size = UInt64(n); var d = Dimension(); d.constant = c; return d }
func tensorType(_ dt: MILDataType, _ shape: [Int]) -> ValueType {
    var tt = TensorType(); tt.dataType = dt; tt.rank = Int64(shape.count); tt.dimensions = shape.map(dim)
    var vt = ValueType(); vt.tensorType = tt; return vt
}
func immediate(_ tv: TensorValue, _ type: ValueType) -> MILValue {
    var iv = MILValue.ImmediateValue(); iv.tensor = tv; var v = MILValue(); v.type = type; v.immediateValue = iv; return v
}
func strConst(_ s: String) -> MILValue { var r = TensorValue.RepeatedStrings(); r.values = [s]; var t = TensorValue(); t.strings = r; return immediate(t, tensorType(.string, [])) }
func intConst(_ v: [Int], _ shape: [Int]) -> MILValue { var r = TensorValue.RepeatedInts(); r.values = v.map(Int32.init); var t = TensorValue(); t.ints = r; return immediate(t, tensorType(.int32, shape)) }
func boolConst(_ v: [Bool], _ shape: [Int]) -> MILValue { var r = TensorValue.RepeatedBools(); r.values = v; var t = TensorValue(); t.bools = r; return immediate(t, tensorType(.bool, shape)) }
func fp16Scalar(_ x: Float) -> MILValue {
    var b = Float16(x).bitPattern.littleEndian; var d = Data(); withUnsafeBytes(of: &b) { d.append(contentsOf: $0) }
    var r = TensorValue.RepeatedBytes(); r.values = d; var t = TensorValue(); t.bytes = r; return immediate(t, tensorType(.float16, []))
}
func constOp(_ name: String, _ value: MILValue) -> Operation {
    var op = Operation(); op.type = "const"
    op.attributes["name"] = strConst(name); op.attributes["val"] = value
    var out = NamedValueType(); out.name = name; out.type = value.type; op.outputs = [out]; return op
}
func argName(_ n: String) -> Argument { var b = Argument.Binding(); b.name = n; var a = Argument(); a.arguments = [b]; return a }
func op(_ type: String, _ inputs: [String: String], _ out: String, _ outT: ValueType) -> Operation {
    var o = Operation(); o.type = type
    for (k, v) in inputs { o.inputs[k] = argName(v) }
    o.attributes["name"] = strConst(out)
    var no = NamedValueType(); no.name = out; no.type = outT; o.outputs = [no]; return o
}
func dims(_ vt: ValueType) -> [Int]? {
    guard case .tensorType(let t)? = vt.type else { return nil }
    var r: [Int] = []
    for d in t.dimensions { guard case .constant(let c)? = d.dimension else { return nil }; r.append(Int(c.size)) }
    return r
}
func boundName(_ o: Operation, _ key: String) -> String? { o.inputs[key]?.arguments.first?.name }
func constInts(_ o: Operation) -> [Int]? {
    guard o.type == "const", let v = o.attributes["val"], case .immediateValue(let iv)? = v.value,
          case .tensor(let t)? = iv.value, case .ints(let r)? = t.value else { return nil }
    return r.values.map(Int.init)
}

// ── main ──
let args = CommandLine.arguments
guard args.count >= 3 else { die("usage: kvselect <in.mlpackage> <out.mlpackage> [function]") }
let inPkg = args[1], outPkg = args[2], fnName = args.count > 3 ? args[3] : "prefill"
let fm = FileManager.default
guard !fm.fileExists(atPath: outPkg) else { die("output exists: \(outPkg)") }
let mf = inPkg + "/Data/com.apple.CoreML/model.mlmodel"
guard let bytes = fm.contents(atPath: mf) else { die("missing \(mf)") }
var model = try Model(serializedBytes: bytes)
guard case .mlProgram? = model.type, var fn = model.mlProgram.functions[fnName] else { die("no ML Program function '\(fnName)'") }
let opset = fn.opset
guard var block = fn.blockSpecializations[opset] else { die("no block for opset \(opset)") }

// producer index for every value name
var producer: [String: Int] = [:]
for (i, o) in block.operations.enumerated() { for out in o.outputs { producer[out.name] = i } }
func producerOp(_ name: String) -> Operation? { producer[name].map { block.operations[$0] } }
/// int value of a value name if it is (or concat-of) consts; nil if runtime
func staticInt(_ name: String) -> Int? { producerOp(name).flatMap(constInts)?.first }

var newOps: [Operation] = []
var shared: [String: (p: String, keep: String)] = [:]   // per runtime-cur var
var rewritten = 0
for o in block.operations {
    guard o.type == "slice_update",
          let x = boundName(o, "x"), let upd = boundName(o, "update"),
          let beginN = boundName(o, "begin"), let endN = boundName(o, "end"),
          let out = o.outputs.first, let cacheShape = dims(out.type), cacheShape.count == 4,
          let beginOp = producerOp(beginN), beginOp.type == "concat",
          let endOp = producerOp(endN), endOp.type == "concat",
          let updOutT = producerOp(upd)?.outputs.first(where: { $0.name == upd })?.type,
          let updShape = dims(updOutT), updShape.count == 4
    else { newOps.append(o); continue }
    let bv = beginOp.inputs["values"]?.arguments.map(\.name) ?? []
    let ev = endOp.inputs["values"]?.arguments.map(\.name) ?? []
    guard bv.count == 4, ev.count == 4,
          let l = staticInt(bv[0]), let l1 = staticInt(ev[0]), l1 == l + 1,
          staticInt(bv[2]) == nil                     // runtime sequence begin = the -14 trigger
    else { newOps.append(o); continue }
    let cur = bv[2]
    let (H, S, D) = (cacheShape[1], cacheShape[2], cacheShape[3])
    let K = updShape[2]
    guard updShape[0] == 1, updShape[1] == H, updShape[3] == D, K > 1, K <= S else { newOps.append(o); continue }
    let fp16 = MILDataType.float16
    let tag = "kvsel_\(rewritten)"

    if shared[cur] == nil {
        let s = "kvsel_\(cur)"
        newOps.append(constOp("\(s)_pos", intConst(Array(0..<S), [S, 1])))
        newOps.append(constOp("\(s)_k", intConst(Array(0..<K), [1, K])))
        newOps.append(op("sub", ["x": "\(s)_pos", "y": "\(s)_k"], "\(s)_diff", tensorType(.int32, [S, K])))
        newOps.append(op("equal", ["x": "\(s)_diff", "y": cur], "\(s)_eq", tensorType(.bool, [S, K])))
        newOps.append(constOp("\(s)_dt", strConst("fp16")))
        newOps.append(op("cast", ["x": "\(s)_eq", "dtype": "\(s)_dt"], "\(s)_P", tensorType(fp16, [S, K])))
        newOps.append(constOp("\(s)_ax", intConst([1], [1])))
        newOps.append(constOp("\(s)_kd", boolConst([true], [])))
        newOps.append(op("reduce_sum", ["x": "\(s)_P", "axes": "\(s)_ax", "keep_dims": "\(s)_kd"], "\(s)_m", tensorType(fp16, [S, 1])))
        newOps.append(constOp("\(s)_one", fp16Scalar(1)))
        newOps.append(op("sub", ["x": "\(s)_one", "y": "\(s)_m"], "\(s)_keep", tensorType(fp16, [S, 1])))
        newOps.append(constOp("\(s)_f", boolConst([false], [])))
        shared[cur] = ("\(s)_P", "\(s)_keep")
    }
    let (P, keep) = shared[cur]!
    let s = "kvsel_\(cur)"
    let rowT = tensorType(fp16, [1, H, S, D])
    newOps.append(constOp("\(tag)_b", intConst([l, 0, 0, 0], [4])))
    newOps.append(constOp("\(tag)_e", intConst([l + 1, H, S, D], [4])))
    newOps.append(constOp("\(tag)_bm", boolConst([false, false, false, false], [4])))
    newOps.append(constOp("\(tag)_em", boolConst([false, false, false, false], [4])))
    newOps.append(constOp("\(tag)_sq", boolConst([false, false, false, false], [4])))
    newOps.append(constOp("\(tag)_st", intConst([1, 1, 1, 1], [4])))
    newOps.append(op("slice_by_index", ["x": x, "begin": "\(tag)_b", "end": "\(tag)_e",
                                        "begin_mask": "\(tag)_bm", "end_mask": "\(tag)_em"], "\(tag)_row", rowT))
    newOps.append(op("matmul", ["x": P, "y": upd, "transpose_x": "\(s)_f", "transpose_y": "\(s)_f"], "\(tag)_placed", rowT))
    newOps.append(op("mul", ["x": "\(tag)_row", "y": keep], "\(tag)_kept", rowT))
    newOps.append(op("add", ["x": "\(tag)_kept", "y": "\(tag)_placed"], "\(tag)_new", rowT))
    var su = op("slice_update", ["x": x, "update": "\(tag)_new", "begin": "\(tag)_b", "end": "\(tag)_e",
                                 "begin_mask": "\(tag)_bm", "end_mask": "\(tag)_em",
                                 "squeeze_mask": "\(tag)_sq", "stride": "\(tag)_st"], out.name, out.type)
    su.attributes = o.attributes   // keep the original op name attribute
    newOps.append(su)
    rewritten += 1
}
guard rewritten > 0 else { die("no dynamic-begin KV slice_update found in '\(fnName)'") }
block.operations = newOps
fn.blockSpecializations[opset] = block
model.mlProgram.functions[fnName] = fn

// write: copy package byte-for-byte, then replace model.mlmodel
try fm.copyItem(atPath: inPkg, toPath: outPkg)
let outMf = outPkg + "/Data/com.apple.CoreML/model.mlmodel"
try fm.removeItem(atPath: outMf)
var enc = BinaryEncodingOptions(); enc.useDeterministicOrdering = true   // map order is otherwise per-process random
try model.serializedData(options: enc).write(to: URL(fileURLWithPath: outMf))
_ = try Model(serializedBytes: fm.contents(atPath: outMf)!)   // re-parse validation
print("kvselect: rewrote \(rewritten) dynamic KV slice_update op(s) in '\(fnName)' (opset \(opset)); shared placements: \(shared.count) → \(outPkg)")
