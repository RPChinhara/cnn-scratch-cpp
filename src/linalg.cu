#include "linalg.h"
#include "arrs.h"
#include "knls.h"
#include "ten.h"

#include <cassert>

ten matmul(const ten &t_1, const ten &t_2, dev_type dev)
{
    ten t_new = zeros({t_1.shape.front(), t_2.shape.back()});

    switch (dev)
    {
    case CPU: {

        for (auto i = 0; i < t_1.shape.front(); ++i)
        {
            for (auto j = 0; j < t_2.shape.back(); ++j)
            {
                float sum = 0.0;

                for (auto l = 0; l < t_1.shape.back(); ++l)
                    sum += t_1[i * t_1.shape.back() + l] * t_2[l * t_2.shape.back() + j];

                t_new[i * t_2.shape.back() + j] = sum;
            }
        }

        return t_new;
    }
    case GPU: {
        assert(t_1.shape.back() == t_2.shape.front());

        size_t numRowsTensor1 = t_1.shape.front();
        size_t numColsTensor1 = t_1.shape.back();
        size_t numRowsTensor2 = t_2.shape.back();

        float *t_gpu_1, *t_gpu_2, *t_gpu_new;
        cudaMalloc(&t_gpu_1, numRowsTensor1 * numColsTensor1 * sizeof(float));
        cudaMalloc(&t_gpu_2, numColsTensor1 * numRowsTensor2 * sizeof(float));
        cudaMalloc(&t_gpu_new, numRowsTensor1 * numRowsTensor2 * sizeof(float));
        cudaMemcpy(t_gpu_1, t_1.elem, t_1.size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(t_gpu_2, t_2.elem, t_2.size * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockDim(16, 16);
        dim3 gridDim((numRowsTensor1 + blockDim.x - 1) / blockDim.x, (numRowsTensor2 + blockDim.y - 1) / blockDim.y);
        matmul<<<gridDim, blockDim>>>(t_gpu_1, t_gpu_2, t_gpu_new, numRowsTensor1, numColsTensor1,
                                      numRowsTensor2);

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
            std::cerr << "CUDA knl launch error. " + std::string(cudaGetErrorString(cudaError)) << std::endl;

        cudaMemcpy(t_new.elem, t_gpu_new, t_new.size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(t_gpu_1);
        cudaFree(t_gpu_2);
        cudaFree(t_gpu_new);

        return t_new;
    }
    default:
        std::cout << "Unknown dev." << std::endl;
        return ten();
    }
}

static size_t get_batch_size(const std::vector<size_t> &shape)
{
    assert(1 < shape.size());
    size_t batchSize = 1;

    for (auto i = 0; i < shape.size() - 2; ++i)
        batchSize *= shape[i];

    return batchSize;
}

ten transpose(const ten &t)
{
    assert(2 <= t.shape.size());

    ten t_new = zeros({t.shape.back(), t.shape[t.shape.size() - 2]});

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