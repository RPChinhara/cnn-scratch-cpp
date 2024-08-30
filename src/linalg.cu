#include "linalg.h"
#include "arrs.h"
#include "tensor.h"

#include <cassert>

__global__ void matmul(float *t1, float *t2, float *t_new, size_t num_rows_t1, size_t num_cols_t1, size_t num_rows_t2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < num_rows_t1 && j < num_rows_t2)
    {
        float sum = 0.0;

        for (auto l = 0; l < num_cols_t1; ++l)
            sum += t1[i * num_cols_t1 + l] * t2[l * num_rows_t2 + j];

        t_new[i * num_rows_t2 + j] = sum;
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
        size_t numRowsTensor1 = t1.shape.front();
        size_t numColsTensor1 = t1.shape.back();
        size_t numRowsTensor2 = t2.shape.back();

        float *t1_gpu, *t2_gpu, *t_gpu_new;
        cudaMalloc(&t1_gpu, numRowsTensor1 * numColsTensor1 * sizeof(float));
        cudaMalloc(&t2_gpu, numColsTensor1 * numRowsTensor2 * sizeof(float));
        cudaMalloc(&t_gpu_new, numRowsTensor1 * numRowsTensor2 * sizeof(float));
        cudaMemcpy(t1_gpu, t1.elem, t1.size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(t2_gpu, t2.elem, t2.size * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockDim(16, 16);
        dim3 gridDim((numRowsTensor1 + blockDim.x - 1) / blockDim.x, (numRowsTensor2 + blockDim.y - 1) / blockDim.y);
        matmul<<<gridDim, blockDim>>>(t1_gpu, t2_gpu, t_gpu_new, numRowsTensor1, numColsTensor1,
                                      numRowsTensor2);

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
            std::cerr << "CUDA knl launch error. " + std::string(cudaGetErrorString(cudaError)) << std::endl;

        cudaMemcpy(t_new.elem, t_gpu_new, t_new.size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(t1_gpu);
        cudaFree(t2_gpu);
        cudaFree(t_gpu_new);

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

    for (auto i = 0; i < batchSize; ++i)
    {
        for (auto j = 0; j < t_new.shape[t_new.shape.size() - 2]; ++j)
        {
            for (auto k = 0; k < t_new.shape.back(); ++k)
            {
                t_new[idx] = t[idx_rows[k + (i * t_new.shape.back())]];
                idx_rows[k + (i * t_new.shape.back())] += 1;
                ++idx;
            }
        }
    }

    return t_new;
}