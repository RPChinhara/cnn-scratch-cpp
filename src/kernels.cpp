#include "kernels.h"

__global__ void abs(float *in, float *out, unsigned int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
        out[id] = fabs(in[id]);
}

__global__ void exp(float *in, float *out, unsigned int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
        out[id] = expf(in[id]);
}

__global__ void log(float *in, float *out, unsigned int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
        out[id] = logf(in[id]);
}

__global__ void matmul(float *in1, float *in2, float *out, int m, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < k) {
        float sum = 0.0;
        for (unsigned int l = 0; l < n; l++)
            sum += in1[i * n + l] * in2[l * k + j];
        out[i * k + j] = sum;
    }
}

__global__ void maximum(float *in1, float *in2, float *out, unsigned int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
        for (unsigned int i = id; i < n; ++i)
	        out[id] = max(in1[id], in2[id]);
}

__global__ void square(float *in, float *out, unsigned int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
        out[id] = powf(in[id], 2.0f);
}

__global__ void tanh(float *in, float *out, unsigned int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
        out[id] = tanhf(in[id]);
}