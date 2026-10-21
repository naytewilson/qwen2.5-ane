#include <arm_neon.h>
#include <stdint.h>
#include <math.h>

// Clamping utility for tail loop
static inline int32_t clamp_i32(int32_t val, int32_t min, int32_t max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

// Optimized Dequantize (INT8 -> FP16)
void dequantizeKV(const int8_t *src, __fp16 *dst, int count, float scale) {
    float16x8_t vscale = vdupq_n_f16((__fp16)scale);
    int i = 0;
    for (; i + 8 <= count; i += 8) {
        int8x8_t vsrc = vld1_s8(src + i);
        int16x8_t vs16 = vmovl_s8(vsrc);              // Sign-extend Int8 to Int16
        float16x8_t vf16 = vcvtq_f16_s16(vs16);        // Direct hardware cast Int16 -> Float16
        vst1q_f16(dst + i, vmulq_f16(vf16, vscale));  // Multiply and store
    }
    // Scalar tail
    for (; i < count; ++i) {
        dst[i] = (__fp16)((float)src[i] * scale);
    }
}

// Unbiased, Round-to-Nearest Ties-to-Even Quantize (FP16 -> INT8)
void quantizeKV(const __fp16 *src, int8_t *dst, int count, float invScale) {
    float16x8_t vinv = vdupq_n_f16((__fp16)invScale);
    int i = 0;
    for (; i + 8 <= count; i += 8) {
        float16x8_t vsrc = vld1q_f16(src + i);
        // Use vcvtnq_s16_f16 to round-to-nearest ties-to-even to eliminate cumulative quantization bias (token drift)
        int16x8_t vs16 = vcvtnq_s16_f16(vmulq_f16(vsrc, vinv)); 
        int8x8_t vs8 = vqmovn_s16(vs16);                       // Saturation narrow to Int8
        vst1_s8(dst + i, vs8);
    }
    // Scalar tail
    for (; i < count; ++i) {
        float f = (float)src[i] * invScale;
        int r = (int)roundf(f);                                // matches nearest even/ties away
        dst[i] = (int8_t)clamp_i32(r, -128, 127);
    }
}
