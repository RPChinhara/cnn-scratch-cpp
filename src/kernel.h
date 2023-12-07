#pragma once

__global__ void Abs(float *in, float *out, unsigned int n);
__global__ void Exp(float *in, float *out, unsigned int n);
__global__ void Log(float *in, float *out, unsigned int n);
__global__ void MatMul(float *in_1, float *in_2, float *out, int m, int n, int k);
__global__ void Maximum(float *in_1, float *in_2, float *out, unsigned int n);
__global__ void Square(float *in, float *out, unsigned int n);
__global__ void Tanh(float *in, float *out, unsigned int n);