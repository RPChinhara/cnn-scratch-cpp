#include "math.hpp"
#include "arrs.h"
#include "knls.h"
#include "ten.h"

#include <cassert>

Ten Argmax(const Ten &ten)
{
    Ten newTensor = zeros({ten.shape.front()});

    size_t idx = 0;
    float max = std::numeric_limits<float>::lowest();
    size_t max_idx = 0;

    for (size_t i = 0; i < ten.shape.front(); ++i)
    {
        for (size_t j = 0; j < ten.shape.back(); ++j)
        {
            if (ten[idx] > max)
            {
                max = ten[idx];
                max_idx = j;
            }
            ++idx;
        }

        newTensor[i] = max_idx;
        max = std::numeric_limits<float>::lowest();
    }

    return newTensor;
}

Ten Exp(const Ten &ten, Dev dev)
{
    Ten newTensor = ten;

    switch (dev)
    {
    case Dev::CPU: {

        for (size_t i = 0; i < ten.size; ++i)
            newTensor.elem[i] = expf(ten.elem[i]);

        return newTensor;
    }
    case Dev::GPU: {
        float *tensorGPU, *newTensorGPU;
        cudaMalloc((void **)&tensorGPU, ten.size * sizeof(float));
        cudaMalloc((void **)&newTensorGPU, ten.size * sizeof(float));
        cudaMemcpy(tensorGPU, ten.elem, ten.size * sizeof(float), cudaMemcpyHostToDevice);

        constexpr int blockSize = 128;
        int gridSize = (ten.size + blockSize - 1) / blockSize;
        Exp<<<gridSize, blockSize>>>(tensorGPU, newTensorGPU, ten.size);

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
            std::cerr << "CUDA knl launch error. " + std::string(cudaGetErrorString(cudaError)) << std::endl;

        cudaMemcpy(newTensor.elem, newTensorGPU, ten.size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(tensorGPU);
        cudaFree(newTensorGPU);

        return newTensor;
    }
    default:
        std::cout << "Unknown dev." << std::endl;
        return Ten();
    }
}

Ten Log(const Ten &ten, Dev dev)
{
    Ten newTensor = ten;

    switch (dev)
    {
    case Dev::CPU: {
        for (size_t i = 0; i < ten.size; ++i)
            newTensor.elem[i] = logf(ten.elem[i]);

        return newTensor;
    }
    case Dev::GPU: {
        float *tensorGPU, *newTensorGPU;
        cudaMalloc((void **)&tensorGPU, ten.size * sizeof(float));
        cudaMalloc((void **)&newTensorGPU, ten.size * sizeof(float));
        cudaMemcpy(tensorGPU, ten.elem, ten.size * sizeof(float), cudaMemcpyHostToDevice);

        constexpr int blockSize = 128;
        int gridSize = (ten.size + blockSize - 1) / blockSize;
        Log<<<gridSize, blockSize>>>(tensorGPU, newTensorGPU, ten.size);

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
            std::cerr << "CUDA knl launch error. " + std::string(cudaGetErrorString(cudaError)) << std::endl;

        cudaMemcpy(newTensor.elem, newTensorGPU, ten.size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(tensorGPU);
        cudaFree(newTensorGPU);

        return newTensor;
    }
    default:
        std::cout << "Unknown dev." << std::endl;
        return Ten();
    }
}

Ten Max(const Ten &ten, const size_t axis)
{
    assert(axis == 0 || axis == 1);
    Ten newTensor;

    if (axis == 0)
    {
        newTensor = zeros({1, ten.shape.back()});

        for (size_t i = 0; i < ten.shape.back(); ++i)
        {
            size_t idx = i;
            float max = std::numeric_limits<float>::lowest();

            for (size_t j = 0; j < ten.shape.front(); ++j)
            {
                if (ten[idx] > max)
                    max = ten[idx];
                idx += ten.shape.back();
            }

            newTensor[i] = max;
        }
    }
    else if (axis == 1)
    {
        newTensor = zeros({ten.shape.front(), 1});
        size_t idx = 0;

        for (size_t i = 0; i < ten.shape.front(); ++i)
        {
            float max = std::numeric_limits<float>::lowest();

            for (size_t j = 0; j < ten.shape.back(); ++j)
            {
                if (ten[idx] > max)
                    max = ten[idx];
                ++idx;
            }

            newTensor[i] = max;
        }
    }

    return newTensor;
}

Ten Min(const Ten &ten)
{
    Ten newTensor = zeros({1, ten.shape.back()});

    for (size_t i = 0; i < ten.shape.back(); ++i)
    {
        size_t idx = i;
        float min = std::numeric_limits<float>::max();

        for (size_t j = 0; j < ten.shape.front(); ++j)
        {
            if (ten[idx] < min)
                min = ten[idx];
            idx += ten.shape.back();
        }

        newTensor[i] = min;
    }

    return newTensor;
}

Ten Sum(const Ten &ten, const size_t axis)
{
    assert(axis == 0 || axis == 1);
    Ten newTensor;

    if (ten.shape.size() == 1 || ten.shape.front() == 1)
    {
        if (axis == 0)
        {
            newTensor = ten;
        }
        else if (axis == 1)
        {
            newTensor = zeros({1, 1});
            float sum = 0.0f;

            for (size_t i = 0; i < ten.size; ++i)
            {
                sum += ten[i];
            }
            newTensor[0] = sum;
        }
    }
    else
    {
        if (axis == 0)
        {
            newTensor = zeros({1, ten.shape.back()});

            for (size_t i = 0; i < ten.shape.back(); ++i)
            {
                size_t idx = i;

                for (size_t j = 0; j < ten.shape.front(); ++j)
                {
                    newTensor[i] += ten[idx];
                    idx += ten.shape.back();
                }
            }
        }
        else if (axis == 1)
        {
            newTensor = zeros({ten.shape.front(), 1});
            size_t idx = 0;

            for (size_t i = 0; i < ten.shape.front(); ++i)
            {
                for (size_t j = 0; j < ten.shape.back(); ++j)
                {
                    newTensor[i] += ten[idx];
                    ++idx;
                }
            }
        }
    }

    return newTensor;
}