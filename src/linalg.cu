#include "linalg.h"
#include "arrs.h"
#include "math.hpp"
#include "tensor.h"

#include <cassert>

__global__ void matmul(float* a, float* b, float* c, int m, int n, int p) {
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

tensor matmul(const tensor& t1, const tensor& t2) {
    if (t1.shape.back() != t2.shape.front()) {
        std::cerr << __FILE__ << "(" << __LINE__ << "): error: t1.shape.back() and t2.shape.front() have to much" << std::endl;
        exit(1);
    }

    tensor t_new = zeros({t1.shape.front(), t2.shape.back()});

    int M = t1.shape.front();
    int N = t1.shape.back();
    int P = t2.shape.back();

    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * P * sizeof(float));
    cudaMalloc(&d_C, M * P * sizeof(float));

    cudaMemcpy(d_A, t1.elems, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, t2.elems, N * P * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((P + threadsPerBlock.x - 1) / threadsPerBlock.x,(M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, P);

    cudaMemcpy(t_new.elems, d_C, M * P * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return t_new;
}

static size_t get_batch_size(const std::vector<size_t>& shape) {
    if (!(1 < shape.size())) {
        std::cerr << __FILE__ << "(" << __LINE__ << "): error: shape.size() has to be greater than 1" << std::endl;
        exit(1);
    }

    size_t batchSize = 1;

    for (auto i = 0; i < shape.size() - 2; ++i)
        batchSize *= shape[i];

    return batchSize;
}

tensor transpose(const tensor& t) {
    if (!(2 <= t.shape.size())) {
        std::cerr << __FILE__ << "(" << __LINE__ << "): error: shape of t has to at least 2-dimensional" << std::endl;
        exit(1);
    }

    tensor t_new = zeros({t.shape.back(), t.shape[t.shape.size() - 2]});

    std::vector<size_t> idx_rows;

    for (auto i = 0; i < t.size; ++i)
        idx_rows.push_back(i * t.shape.back());

    size_t batchSize = get_batch_size(t.shape);

    size_t idx = 0;

    for (auto i = 0; i < batchSize; ++i) {
        for (auto j = 0; j < t_new.shape[t_new.shape.size() - 2]; ++j) {
            for (auto k = 0; k < t_new.shape.back(); ++k) {
                t_new[idx] = t[idx_rows[k + (i * t_new.shape.back())]];
                idx_rows[k + (i * t_new.shape.back())] += 1;
                ++idx;
            }
        }
    }

    return t_new;
}