#pragma once

#include "types.h"

__global__ void abs(f32 *in, f32 *out, u32 n);
__global__ void exp(f32 *in, f32 *out, u32 n);
__global__ void log(f32 *in, f32 *out, u32 n);
__global__ void matmul(f32 *in1, f32 *in2, f32 *out, int m, int n, int k);
__global__ void maximum(f32 *in1, f32 *in2, f32 *out, u32 n);
__global__ void square(f32 *in, f32 *out, u32 n);