#pragma once

__global__ void Abs(float *in, float *out, size_t n);
__global__ void Exp(float *in, float *out, size_t n);
__global__ void Log(float *in, float *out, size_t n);
__global__ void MatMul(float *in_1, float *in_2, float *out, size_t m, size_t n, size_t k);
__global__ void Maximum(float *in_1, float *in_2, float *out, size_t n);
__global__ void Square(float *in, float *out, size_t n);
__global__ void Tanh(float *in, float *out, size_t n);