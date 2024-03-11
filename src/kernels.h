#pragma once

__global__ void Exp(float *tensor, float *newTensor, size_t n);
__global__ void Log(float *tensor, float *newTensor, size_t n);
__global__ void MatMul(float *tensor1, float *tensor2, float *newTensor, size_t numRowsTensor1, size_t numColsTensor1, size_t numRowsTensor2);
// __global__ void OperatorPlus(float *in1, float *in2, float *out, size_t otherShapeBack, size_t k);
__global__ void Relu(float *tensor, float *newTensor, size_t n);