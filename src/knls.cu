#include "knls.h"

__global__ void Exp(float *t, float *t_new, size_t n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n)
        t_new[id] = expf(t[id]);
}

__global__ void Log(float *t, float *t_new, size_t n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n)
        t_new[id] = logf(t[id]);
}

__global__ void MatMul(float *tensor1, float *tensor2, float *t_new, size_t numRowsTensor1, size_t numColsTensor1,
                       size_t numRowsTensor2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numRowsTensor1 && j < numRowsTensor2)
    {
        float sum = 0.0;

        for (auto l = 0; l < numColsTensor1; l++)
            sum += tensor1[i * numColsTensor1 + l] * tensor2[l * numRowsTensor2 + j];

        t_new[i * numRowsTensor2 + j] = sum;
    }
}

// __global__ void OperatorPlus(float *in1, float *in2, float *t_new, size_t otherShapeBack, size_t n)
// {
//     int id = blockIdx.x * blockDim.x + threadIdx.x;
//     if (id < n)
// 	    t_new[id] = in1[id] + in2[id];
// }

__global__ void relu(float *t, float *t_new, size_t n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n)
        t_new[id] = fmaxf(0.0f, t[id]);
}