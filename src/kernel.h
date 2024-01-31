#pragma once

__global__ void Exp(float *in, float *out, size_t n);
__global__ void Log(float *in, float *out, size_t n);
__global__ void MatMul(float *in1, float *in2, float *out, size_t m, size_t n, size_t k);
// __global__ void OperatorPlus(float *in1, float *in2, float *out, size_t otherShapeBack, size_t k);
__global__ void Relu(float *in, float *out, size_t n);