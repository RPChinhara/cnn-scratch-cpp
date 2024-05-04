#include "knls.h"

__global__ void exp(float *t, float *t_new, size_t n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n)
        t_new[id] = expf(t[id]);
}

__global__ void log(float *t, float *t_new, size_t n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n)
        t_new[id] = logf(t[id]);
}

__global__ void matmul(float *t1, float *t2, float *t_new, size_t num_rows_t1, size_t num_cols_t1,
                       size_t num_rows_t2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < num_rows_t1 && j < num_rows_t2)
    {
        float sum = 0.0;

        for (auto l = 0; l < num_cols_t1; l++)
            sum += t1[i * num_cols_t1 + l] * t2[l * num_rows_t2 + j];

        t_new[i * num_rows_t2 + j] = sum;
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