#include "kernel.h"

__global__ void Exp(float *tensor, float *newTensor, size_t n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n)
        newTensor[id] = expf(tensor[id]);
}

__global__ void Log(float *tensor, float *newTensor, size_t n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n)
        newTensor[id] = logf(tensor[id]);
}

__global__ void MatMul(float *tensor1, float *tensor2, float *newTensor, size_t numRowsTensor1, size_t numColsTensor1,
                       size_t numRowsTensor2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numRowsTensor1 && j < numRowsTensor2)
    {
        float sum = 0.0;

        for (size_t l = 0; l < numColsTensor1; l++)
            sum += tensor1[i * numColsTensor1 + l] * tensor2[l * numRowsTensor2 + j];

        newTensor[i * numRowsTensor2 + j] = sum;
    }
}

// __global__ void OperatorPlus(float *in1, float *in2, float *newTensor, size_t otherShapeBack, size_t n)
// {
//     int id = blockIdx.x * blockDim.x + threadIdx.x;
//     if (id < n)
// 	    newTensor[id] = in1[id] + in2[id];
// }

__global__ void Relu(float *tensor, float *newTensor, size_t n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n)
        newTensor[id] = fmaxf(0.0f, tensor[id]);
}