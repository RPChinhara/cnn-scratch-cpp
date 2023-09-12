#include "kernels.h"

__global__ void abs(f32 *in, f32 *out, u32 n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
        out[id] = fabs(in[id]);
}

__global__ void exp(f32 *in, f32 *out, u32 n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
        out[id] = expf(in[id]);
}

__global__ void log(f32 *in, f32 *out, u32 n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
        out[id] = logf(in[id]);
}

__global__ void matmul(f32 *in1, f32 *in2, f32 *out, int m, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < k) {
        f32 sum = 0.0;
        for (u32 l = 0; l < n; l++)
            sum += in1[i * n + l] * in2[l * k + j];
        out[i * k + j] = sum;
    }
}

__global__ void maximum(f32 *in1, f32 *in2, f32 *out, u32 n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
        for (u32 i = id; i < n; ++i)
	        out[id] = max(in1[id], in2[id]);
}

__global__ void square(f32 *in, f32 *out, u32 n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
        out[id] = powf(in[id], 2.0f);
}