#include "linalg.h"
#include "arrs.h"
#include "knls.h"
#include "ten.h"

#include <cassert>

Ten matmul(const Ten &tensor1, const Ten &tensor2, Dev dev)
{
    Ten newTensor = zeros({tensor1.shape.front(), tensor2.shape.back()});

    switch (dev)
    {
    case Dev::CPU: {

        for (size_t i = 0; i < tensor1.shape.front(); ++i)
        {
            for (size_t j = 0; j < tensor2.shape.back(); ++j)
            {
                float sum = 0.0;

                for (size_t l = 0; l < tensor1.shape.back(); ++l)
                    sum += tensor1[i * tensor1.shape.back() + l] * tensor2[l * tensor2.shape.back() + j];

                newTensor[i * tensor2.shape.back() + j] = sum;
            }
        }

        return newTensor;
    }
    case Dev::GPU: {
        assert(tensor1.shape.back() == tensor2.shape.front());

        size_t numRowsTensor1 = tensor1.shape.front();
        size_t numColsTensor1 = tensor1.shape.back();
        size_t numRowsTensor2 = tensor2.shape.back();

        float *tensorGPU1, *tensorGPU2, *newTensorGPU;
        cudaMalloc(&tensorGPU1, numRowsTensor1 * numColsTensor1 * sizeof(float));
        cudaMalloc(&tensorGPU2, numColsTensor1 * numRowsTensor2 * sizeof(float));
        cudaMalloc(&newTensorGPU, numRowsTensor1 * numRowsTensor2 * sizeof(float));
        cudaMemcpy(tensorGPU1, tensor1.elem, tensor1.size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(tensorGPU2, tensor2.elem, tensor2.size * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockDim(16, 16);
        dim3 gridDim((numRowsTensor1 + blockDim.x - 1) / blockDim.x, (numRowsTensor2 + blockDim.y - 1) / blockDim.y);
        MatMul<<<gridDim, blockDim>>>(tensorGPU1, tensorGPU2, newTensorGPU, numRowsTensor1, numColsTensor1,
                                      numRowsTensor2);

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
            std::cerr << "CUDA knl launch error. " + std::string(cudaGetErrorString(cudaError)) << std::endl;

        cudaMemcpy(newTensor.elem, newTensorGPU, newTensor.size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(tensorGPU1);
        cudaFree(tensorGPU2);
        cudaFree(newTensorGPU);

        return newTensor;
    }
    default:
        std::cout << "Unknown dev." << std::endl;
        return Ten();
    }
}

static size_t get_batch_size(const std::vector<size_t> &shape)
{
    assert(shape.size() > 1);
    size_t batchSize = 1;

    for (size_t i = 0; i < shape.size() - 2; ++i)
        batchSize *= shape[i];

    return batchSize;
}

Ten transpose(const Ten &t)
{
    assert(t.shape.size() >= 2);

    Ten newTensor = zeros({t.shape.back(), t.shape[t.shape.size() - 2]});

    std::vector<size_t> idx_rows;

    for (size_t i = 0; i < t.size; ++i)
        idx_rows.push_back(i * t.shape.back());

    size_t batchSize = get_batch_size(t.shape);

    size_t idx = 0;

    for (size_t i = 0; i < batchSize; ++i)
    {
        for (size_t j = 0; j < newTensor.shape[newTensor.shape.size() - 2]; ++j)
        {
            for (size_t k = 0; k < newTensor.shape.back(); ++k)
            {
                newTensor[idx] = t[idx_rows[k + (i * newTensor.shape.back())]];
                idx_rows[k + (i * newTensor.shape.back())] += 1;
                ++idx;
            }
        }
    }

    return newTensor;
}