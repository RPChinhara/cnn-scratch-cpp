#pragma once

__global__ void abs(float *in, float *out, unsigned int n);
__global__ void exp(float *in, float *out, unsigned int n);
__global__ void log(float *in, float *out, unsigned int n);
__global__ void matmul(float *in1, float *in2, float *out, int m, int n, int k);
__global__ void maximum(float *in1, float *in2, float *out, unsigned int n);
__global__ void square(float *in, float *out, unsigned int n);