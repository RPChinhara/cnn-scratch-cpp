#include "linalg.h"
#include "arrs.h"
#include "math.h"
#include "tensor.h"

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
    tensor t_new = zeros({t1.shape.front(), t2.shape.back()});

    int m = t1.shape.front();
    int n = t1.shape.back();
    int p = t2.shape.back();

    float* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, m * n * sizeof(float));
    cudaMalloc(&d_b, n * p * sizeof(float));
    cudaMalloc(&d_c, m * p * sizeof(float));

    cudaMemcpy(d_a, t1.elems, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, t2.elems, n * p * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((p + threadsPerBlock.x - 1) / threadsPerBlock.x,(m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, m, n, p);

    cudaMemcpy(t_new.elems, d_c, m * p * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return t_new;
}

static size_t get_batch_size(const std::vector<size_t>& shape) {
    size_t batch_size = 1;

    for (auto i = 0; i < shape.size() - 2; ++i)
        batch_size *= shape[i];

    return batch_size;
}

tensor transpose(const tensor& t) {
    tensor t_new = zeros({t.shape.back(), t.shape[t.shape.size() - 2]});

    std::vector<size_t> idx_rows;

    for (auto i = 0; i < t.size; ++i)
        idx_rows.push_back(i * t.shape.back());

    size_t batch_size = get_batch_size(t.shape);

    size_t idx = 0;

    for (auto i = 0; i < batch_size; ++i) {
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