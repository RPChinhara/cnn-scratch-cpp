#include "linalg.h"
#include "arrs.h"
#include "math.h"
#include "tensor.h"

// TODO: Use FP16 instead of FP32 (float)
__global__ void matmul_kernel(float* a, float* b, float* c, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        float value = 0;
        for (int k = 0; k < n; ++k) {
            value += a[row * n + k] * b[k * p + col];
        }

        c[row * p + col] = value;
    }
}

tensor matmul_cuda(const tensor& t1, const tensor& t2) {
    size_t m = t1.shape.front();
    size_t n = t1.shape.back();
    size_t p = t2.shape.back();

    tensor t_new = zeros({m, p});

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, m * n * sizeof(float));
    cudaMalloc(&d_b, n * p * sizeof(float));
    cudaMalloc(&d_c, m * p * sizeof(float));

    cudaMemcpy(d_a, t1.elems, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, t2.elems, n * p * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((p + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, m, n, p);

    cudaMemcpy(t_new.elems, d_c, m * p * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return t_new;
}

tensor matmul(const tensor& t1, const tensor& t2) {
    if (t1.shape.size() == 2) {
        return matmul_cuda(t1, t2);
    }

    // NOTE: This supports matmul between 3D matrices, but it has not use cases for now. It was created for ffn in transformer encoder.
    size_t batch_size = 1;
    for (size_t i = 0; i < t1.shape.size() - 2; ++i) {
        batch_size *= t1.shape[i];
    }

    size_t t1_row = t1.shape[t1.shape.size() - 2];
    size_t t2_row = t2.shape[t2.shape.size() - 2];
    size_t t2_col = t2.shape.back();

    tensor t_new = zeros({batch_size, t1_row, t2_col});

    for (size_t i = 0; i < batch_size; ++i) {
        tensor mat_t1 = slice(t1, i * t1_row, t1_row);
        tensor mat_t2 = slice(t2, i * t2_row, t2_row);
        tensor output = matmul_cuda(mat_t1, mat_t2);

        std::copy(output.elems, output.elems + output.size, t_new.elems + i * output.size);
    }

    return t_new;
}

tensor transpose(const tensor& t) {
    size_t rows = t.shape[0];
    size_t cols = t.shape.back();

    tensor t_new = zeros({cols, rows});

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            t_new(j, i) = t(i, j);
        }
    }
    return t_new;
}