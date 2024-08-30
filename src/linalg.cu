#include "linalg.h"
#include "arrs.h"
#include "tensor.h"

#include <cassert>

__global__ void matmul(float* A, float* B, float* C, int M, int N, int P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < P) {
        float value = 0;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * P + col];
        }
        C[row * P + col] = value;
    }
}

tensor matmul(const tensor &t1, const tensor &t2, dev_type dev) {
    assert(t1.shape.back() == t2.shape.front());

    tensor t_new = zeros({t1.shape.front(), t2.shape.back()});

    switch (dev)
    {
    case CPU: {

        for (auto i = 0; i < t1.shape.front(); ++i)
        {
            for (auto j = 0; j < t2.shape.back(); ++j)
            {
                float sum = 0.0;

                for (auto l = 0; l < t1.shape.back(); ++l)
                    sum += t1[i * t1.shape.back() + l] * t2[l * t2.shape.back() + j];

                t_new[i * t2.shape.back() + j] = sum;
            }
        }

        return t_new;
    }
    case GPU: {
        int M = t1.shape.front();
        int N = t1.shape.back();
        int P = t2.shape.back();

        float* d_A, * d_B, * d_C;
        cudaMalloc(&d_A, M * N * sizeof(float));
        cudaMalloc(&d_B, N * P * sizeof(float));
        cudaMalloc(&d_C, M * P * sizeof(float));

        cudaMemcpy(d_A, t1.elem, M * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, t2.elem, N * P * sizeof(float), cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((P + threadsPerBlock.x - 1) / threadsPerBlock.x,(M + threadsPerBlock.y - 1) / threadsPerBlock.y);

        matmul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, P);

        cudaMemcpy(t_new.elem, d_C, M * P * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        return t_new;
    }
    default:
        std::cout << "Unknown dev." << std::endl;
        return tensor();
    }
}

static size_t get_batch_size(const std::vector<size_t> &shape) {
    assert(1 < shape.size());
    size_t batchSize = 1;

    for (auto i = 0; i < shape.size() - 2; ++i)
        batchSize *= shape[i];

    return batchSize;
}

tensor transpose(const tensor &t) {
    assert(2 <= t.shape.size());

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