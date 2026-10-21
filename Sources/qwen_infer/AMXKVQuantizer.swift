//
//  AMXKVQuantizer.swift
//  qwen-ane
//
//  Phase 5: AMX-Dequant External INT8 KV Cache Bypass.
//  Replaces CoreML stateful Float16 KV cache with 1-byte INT8 storage,
//  using Accelerate (vDSP) vector routines to dequantize to Float16
//  just before each inference call. 0% GPU footprint.
//  Optimised for Apple A18 Pro (AMX2) via vDSP.
//

import Accelerate
import Foundation

public final class AMXKVQuantizer {

    // MARK: - Constants
    @inline(__always) static let maxInt8: Float = 127.0

    // MARK: - Quantization
    /// Per-tensor symmetric quantization: Float16 -> [Int8] + scale (Float16)
    public static func quantize(_ input: [Float16]) -> (data: [Int8], scale: Float16) {
        let count = input.count
        guard count > 0 else { return ([], 0) }

        // Convert Float16 -> Float32 for math
        var floatBuf = [Float](repeating: 0, count: count)
        for i in 0..<count {
            floatBuf[i] = Float(input[i])
        }

        var maxAbs: Float = 0
        vDSP_maxmgv(floatBuf, 1, &maxAbs, vDSP_Length(count))
        
        guard maxAbs > 0 else {
            return ([Int8](repeating: 0, count: count), 0)
        }

        let scaleFloat = maxAbs / Self.maxInt8
        let scale      = Float16(scaleFloat)

        // Divide by scale
        var scaleVar = scaleFloat
        vDSP_vsdiv(floatBuf, 1, &scaleVar, &floatBuf, 1, vDSP_Length(count))

        // Convert Float32 -> Int8
        var int8Buf = [Int8](repeating: 0, count: count)
        for i in 0..<count {
            // clamp and round
            var val = floatBuf[i].rounded(.toNearestOrEven)
            if val > 127.0 { val = 127.0 }
            if val < -128.0 { val = -128.0 }
            int8Buf[i] = Int8(val)
        }

        return (data: int8Buf, scale: scale)
    }

    // MARK: - Dequantization
    /// In-place dequantization into a pre-allocated Float16 buffer.
    public static func dequantize(data: [Int8], scale: Float16, into output: UnsafeMutablePointer<Float16>) {
        let count = vDSP_Length(data.count)
        guard count > 0 else { return }

        let scaleFloat = Float(scale)

        // Stage 1: Int8 -> Float32
        var floatBuf = [Float](repeating: 0, count: Int(count))
        data.withUnsafeBufferPointer { src in
            vDSP_vflt8(src.baseAddress!, 1, &floatBuf, 1, count)
        }

        // Stage 2: Multiply by scale
        var scaleVar = scaleFloat
        vDSP_vsmul(floatBuf, 1, &scaleVar, &floatBuf, 1, count)

        // Stage 3: Float32 -> Float16
        for i in 0..<Int(count) {
            output[i] = Float16(floatBuf[i])
        }
    }

    /// Convenience: returns a new Float16 array (heap allocated).
    public static func dequantize(data: [Int8], scale: Float16) -> [Float16] {
        var result = [Float16](repeating: 0, count: data.count)
        dequantize(data: data, scale: scale, into: &result)
        return result
    }
}
