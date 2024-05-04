#pragma once

__global__ void exp(float *t, float *t_new, size_t n);
__global__ void log(float *t, float *t_new, size_t n);
__global__ void matmul(float *t1, float *t2, float *t_new, size_t num_rows_t1, size_t num_cols_t1, size_t num_rows_t2);
// __global__ void OperatorPlus(float *in1, float *in2, float *out, size_t otherShapeBack, size_t k);
__global__ void relu(float *t, float *t_new, size_t n);