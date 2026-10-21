#ifndef QUANTIZE_NEON_H
#define QUANTIZE_NEON_H

#include <stdint.h>

void dequantizeKV(const int8_t *src, __fp16 *dst, int count, float scale);
void quantizeKV(const __fp16 *src, int8_t *dst, int count, float invScale);

#endif // QUANTIZE_NEON_H
