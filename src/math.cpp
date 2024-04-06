#include "math.hpp"
#include "arrs.h"
#include "knls.h"
#include "ten.h"

#include <cassert>

Ten Argmax(const Ten &t)
{
    Ten newTensor = zeros({t.shape.front()});

    size_t idx = 0;
    float max = std::numeric_limits<float>::lowest();
    size_t max_idx = 0;

    for (size_t i = 0; i < t.shape.front(); ++i)
    {
        for (size_t j = 0; j < t.shape.back(); ++j)
        {
            if (t[idx] > max)
            {
                max = t[idx];
                max_idx = j;
            }
            ++idx;
        }

        newTensor[i] = max_idx;
        max = std::numeric_limits<float>::lowest();
    }

    return newTensor;
}

Ten Exp(const Ten &t, Dev dev)
{
    Ten newTensor = t;

    switch (dev)
    {
    case Dev::CPU: {

        for (size_t i = 0; i < t.size; ++i)
            newTensor.elem[i] = expf(t.elem[i]);

        return newTensor;
    }
    case Dev::GPU: {
        float *tensorGPU, *newTensorGPU;
        cudaMalloc((void **)&tensorGPU, t.size * sizeof(float));
        cudaMalloc((void **)&newTensorGPU, t.size * sizeof(float));
        cudaMemcpy(tensorGPU, t.elem, t.size * sizeof(float), cudaMemcpyHostToDevice);

        constexpr int blockSize = 128;
        int gridSize = (t.size + blockSize - 1) / blockSize;
        Exp<<<gridSize, blockSize>>>(tensorGPU, newTensorGPU, t.size);

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
            std::cerr << "CUDA knl launch error. " + std::string(cudaGetErrorString(cudaError)) << std::endl;

        cudaMemcpy(newTensor.elem, newTensorGPU, t.size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(tensorGPU);
        cudaFree(newTensorGPU);

        return newTensor;
    }
    default:
        std::cout << "Unknown dev." << std::endl;
        return Ten();
    }
}

Ten Log(const Ten &t, Dev dev)
{
    Ten newTensor = t;

    switch (dev)
    {
    case Dev::CPU: {
        for (size_t i = 0; i < t.size; ++i)
            newTensor.elem[i] = logf(t.elem[i]);

        return newTensor;
    }
    case Dev::GPU: {
        float *tensorGPU, *newTensorGPU;
        cudaMalloc((void **)&tensorGPU, t.size * sizeof(float));
        cudaMalloc((void **)&newTensorGPU, t.size * sizeof(float));
        cudaMemcpy(tensorGPU, t.elem, t.size * sizeof(float), cudaMemcpyHostToDevice);

        constexpr int blockSize = 128;
        int gridSize = (t.size + blockSize - 1) / blockSize;
        Log<<<gridSize, blockSize>>>(tensorGPU, newTensorGPU, t.size);

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
            std::cerr << "CUDA knl launch error. " + std::string(cudaGetErrorString(cudaError)) << std::endl;

        cudaMemcpy(newTensor.elem, newTensorGPU, t.size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(tensorGPU);
        cudaFree(newTensorGPU);

        return newTensor;
    }
    default:
        std::cout << "Unknown dev." << std::endl;
        return Ten();
    }
}

Ten Max(const Ten &t, const size_t axis)
{
    assert(axis == 0 || axis == 1);
    Ten newTensor;

    if (axis == 0)
    {
        newTensor = zeros({1, t.shape.back()});

        for (size_t i = 0; i < t.shape.back(); ++i)
        {
            size_t idx = i;
            float max = std::numeric_limits<float>::lowest();

            for (size_t j = 0; j < t.shape.front(); ++j)
            {
                if (t[idx] > max)
                    max = t[idx];
                idx += t.shape.back();
            }

            newTensor[i] = max;
        }
    }
    else if (axis == 1)
    {
        newTensor = zeros({t.shape.front(), 1});
        size_t idx = 0;

        for (size_t i = 0; i < t.shape.front(); ++i)
        {
            float max = std::numeric_limits<float>::lowest();

            for (size_t j = 0; j < t.shape.back(); ++j)
            {
                if (t[idx] > max)
                    max = t[idx];
                ++idx;
            }

            newTensor[i] = max;
        }
    }

    return newTensor;
}

Ten Min(const Ten &t)
{
    Ten newTensor = zeros({1, t.shape.back()});

    for (size_t i = 0; i < t.shape.back(); ++i)
    {
        size_t idx = i;
        float min = std::numeric_limits<float>::max();

        for (size_t j = 0; j < t.shape.front(); ++j)
        {
            if (t[idx] < min)
                min = t[idx];
            idx += t.shape.back();
        }

        newTensor[i] = min;
    }

    return newTensor;
}

Ten Sum(const Ten &t, const size_t axis)
{
    assert(axis == 0 || axis == 1);
    Ten newTensor;

    if (t.shape.size() == 1 || t.shape.front() == 1)
    {
        if (axis == 0)
        {
            newTensor = t;
        }
        else if (axis == 1)
        {
            newTensor = zeros({1, 1});
            float sum = 0.0f;

            for (size_t i = 0; i < t.size; ++i)
            {
                sum += t[i];
            }
            newTensor[0] = sum;
        }
    }
    else
    {
        if (axis == 0)
        {
            newTensor = zeros({1, t.shape.back()});

            for (size_t i = 0; i < t.shape.back(); ++i)
            {
                size_t idx = i;

                for (size_t j = 0; j < t.shape.front(); ++j)
                {
                    newTensor[i] += t[idx];
                    idx += t.shape.back();
                }
            }
        }
        else if (axis == 1)
        {
            newTensor = zeros({t.shape.front(), 1});
            size_t idx = 0;

            for (size_t i = 0; i < t.shape.front(); ++i)
            {
                for (size_t j = 0; j < t.shape.back(); ++j)
                {
                    newTensor[i] += t[idx];
                    ++idx;
                }
            }
        }
    }

    return newTensor;
}