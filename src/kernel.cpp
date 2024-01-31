#include "kernel.h"

__global__ void Exp(float *in, float *out, size_t n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n)
        out[id] = expf(in[id]);
}

__global__ void Log(float *in, float *out, size_t n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n)
        out[id] = logf(in[id]);
}

__global__ void MatMul(float *in_1, float *in_2, float *out, size_t m, size_t n, size_t k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < k)
    {
        float sum = 0.0;

        for (size_t l = 0; l < n; l++)
            sum += in_1[i * n + l] * in_2[l * k + j];

        out[i * k + j] = sum;
    }
}

// __global__ void OperatorPlus(float *in1, float *in2, float *out, size_t otherShapeBack, size_t n)
// {
//     int id = blockIdx.x * blockDim.x + threadIdx.x;
//     if (id < n)
// 	    out[id] = in1[id] + in2[id];
// }

__global__ void Relu(float *in, float *out, size_t n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n)
        out[id] = fmaxf(0.0f, in[id]);
}