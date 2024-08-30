#include "knls.h"






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