import Foundation
import Accelerate

/// Vectorized INT8 KV Cache Quantizer utilizing Accelerate/vDSP for peak CPU/AMX performance.
public final class KVQuantizer {
    
    // Static pre-allocated scratch buffers to prevent runtime heap allocations
    private static let maxBufferElements = 131072 // sufficient for head_dim * context_length (e.g., 128 * 1024)
    private nonisolated(unsafe) static var scratchF32 = [Float](repeating: 0.0, count: maxBufferElements)
    private static let lock = NSLock()

    /// Quantizes a Float16 vector into an INT8 vector using absmax scaling.
    /// Performs operations using hardware-accelerated vDSP vector functions.
    public static func quantize(
        src: UnsafePointer<Float16>,
        dst: UnsafeMutablePointer<Int8>,
        count: Int
    ) -> Float16 {
        guard count > 0 else { return 1.0 }
        
        return lock.withLock {
            let limit = min(count, maxBufferElements)
            
            // Step 1: Upcast Float16 input to Float32 scratch buffer for vDSP execution
            for i in 0..<limit {
                scratchF32[i] = Float(src[i])
            }
            
            // Step 2: Find the absolute maximum value in the vector
            var maxVal: Float = 0.0
            vDSP_maxmgv(scratchF32, 1, &maxVal, vDSP_Length(limit))
            
            // If absolute max is 0, return default scale
            if maxVal < 1e-8 {
                memset(dst, 0, limit)
                return 1.0
            }
            
            // Step 3: Compute the scale factor (maps maxVal to 127.0)
            let scale = 127.0 / maxVal
            let invScale = maxVal / 127.0 // This is the scaling factor to dequantize back
            
            // Step 4: Scale the Float32 vector: scratchF32 * scale
            var s = scale
            vDSP_vsmul(scratchF32, 1, &s, &scratchF32, 1, vDSP_Length(limit))
            
            // Step 5: Convert scaled Float32 vector to INT8 via hardware-accelerated vDSP truncation
            vDSP_vfix8(scratchF32, 1, dst, 1, vDSP_Length(limit))
            
            return Float16(invScale)
        }
    }

    /// Dequantizes an INT8 vector back to a Float16 vector.
    /// Uses vectorized vflt8 and vsmul to achieve peak throughput.
    public static func dequantize(
        src: UnsafePointer<Int8>,
        dst: UnsafeMutablePointer<Float16>,
        scale: Float16,
        count: Int
    ) {
        guard count > 0 else { return }
        
        lock.withLock {
            let limit = min(count, maxBufferElements)
            
            // Step 1: Convert INT8 vector to Float32 scratch vector in one atomic vectorized call
            vDSP_vflt8(src, 1, &scratchF32, 1, vDSP_Length(limit))
            
            // Step 2: Multiply by scale factor: scratchF32 * scale
            var s = Float(scale)
            vDSP_vsmul(scratchF32, 1, &s, &scratchF32, 1, vDSP_Length(limit))
            
            // Step 3: Downcast Float32 results back to the target Float16 output pointer
            for i in 0..<limit {
                dst[i] = Float16(scratchF32[i])
            }
        }
    }
}
