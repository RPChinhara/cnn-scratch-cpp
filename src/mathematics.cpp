#include "mathematics.h"
#include "array.h"
#include "kernel.h"
#include "tensor.h"

#include <cassert>
#include <windows.h>

Tensor Argmax(const Tensor &tensor)
{
    Tensor newTensor = Zeros({tensor.shape.front()});

    size_t idx = 0;
    float max = std::numeric_limits<float>::lowest();
    size_t max_idx = 0;

    for (size_t i = 0; i < tensor.shape.front(); ++i)
    {
        for (size_t j = 0; j < tensor.shape.back(); ++j)
        {
            if (tensor[idx] > max)
            {
                max = tensor[idx];
                max_idx = j;
            }
            ++idx;
        }

        newTensor[i] = max_idx;
        max = std::numeric_limits<float>::lowest();
    }

    return newTensor;
}

Tensor Exp(const Tensor &tensor, Device device)
{
    Tensor newTensor = tensor;

    switch (device)
    {
    case Device::CPU: {

        for (size_t i = 0; i < tensor.size; ++i)
            newTensor.elem[i] = expf(tensor.elem[i]);

        return newTensor;
    }
    case Device::GPU: {
        float *tensorGPU, *newTensorGPU;
        cudaMalloc((void **)&tensorGPU, tensor.size * sizeof(float));
        cudaMalloc((void **)&newTensorGPU, tensor.size * sizeof(float));
        cudaMemcpy(tensorGPU, tensor.elem, tensor.size * sizeof(float), cudaMemcpyHostToDevice);

        constexpr int blockSize = 128;
        int gridSize = (tensor.size + blockSize - 1) / blockSize;
        Exp<<<gridSize, blockSize>>>(tensorGPU, newTensorGPU, tensor.size);

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
            MessageBox(nullptr, ("CUDA kernel launch error " + std::string(cudaGetErrorString(cudaError))).c_str(),
                       "Error", MB_ICONERROR);

        cudaMemcpy(newTensor.elem, newTensorGPU, tensor.size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(tensorGPU);
        cudaFree(newTensorGPU);

        return newTensor;
    }
    default:
        MessageBox(nullptr, "Unknown device", "Error", MB_ICONERROR);
        return Tensor();
    }
}

Tensor Log(const Tensor &tensor, Device device)
{
    Tensor newTensor = tensor;

    switch (device)
    {
    case Device::CPU: {
        for (size_t i = 0; i < tensor.size; ++i)
            newTensor.elem[i] = logf(tensor.elem[i]);

        return newTensor;
    }
    case Device::GPU: {
        float *tensorGPU, *newTensorGPU;
        cudaMalloc((void **)&tensorGPU, tensor.size * sizeof(float));
        cudaMalloc((void **)&newTensorGPU, tensor.size * sizeof(float));
        cudaMemcpy(tensorGPU, tensor.elem, tensor.size * sizeof(float), cudaMemcpyHostToDevice);

        constexpr int blockSize = 128;
        int gridSize = (tensor.size + blockSize - 1) / blockSize;
        Log<<<gridSize, blockSize>>>(tensorGPU, newTensorGPU, tensor.size);

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
            MessageBox(nullptr, ("CUDA kernel launch error " + std::string(cudaGetErrorString(cudaError))).c_str(),
                       "Error", MB_ICONERROR);

        cudaMemcpy(newTensor.elem, newTensorGPU, tensor.size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(tensorGPU);
        cudaFree(newTensorGPU);

        return newTensor;
    }
    default:
        MessageBox(nullptr, "Unknown device", "Error", MB_ICONERROR);
        return Tensor();
    }
}

Tensor Max(const Tensor &tensor, const size_t axis)
{
    assert(axis == 0 || axis == 1);
    Tensor newTensor;

    if (axis == 0)
    {
        newTensor = Zeros({1, tensor.shape.back()});

        for (size_t i = 0; i < tensor.shape.back(); ++i)
        {
            size_t idx = i;
            float max = std::numeric_limits<float>::lowest();

            for (size_t j = 0; j < tensor.shape.front(); ++j)
            {
                if (tensor[idx] > max)
                    max = tensor[idx];
                idx += tensor.shape.back();
            }

            newTensor[i] = max;
        }
    }
    else if (axis == 1)
    {
        newTensor = Zeros({tensor.shape.front(), 1});
        size_t idx = 0;

        for (size_t i = 0; i < tensor.shape.front(); ++i)
        {
            float max = std::numeric_limits<float>::lowest();

            for (size_t j = 0; j < tensor.shape.back(); ++j)
            {
                if (tensor[idx] > max)
                    max = tensor[idx];
                ++idx;
            }

            newTensor[i] = max;
        }
    }

    return newTensor;
}

Tensor Min(const Tensor &tensor)
{
    Tensor newTensor = Zeros({1, tensor.shape.back()});

    for (size_t i = 0; i < tensor.shape.back(); ++i)
    {
        size_t idx = i;
        float min = std::numeric_limits<float>::max();

        for (size_t j = 0; j < tensor.shape.front(); ++j)
        {
            if (tensor[idx] < min)
                min = tensor[idx];
            idx += tensor.shape.back();
        }

        newTensor[i] = min;
    }

    return newTensor;
}

Tensor Sum(const Tensor &tensor, const size_t axis)
{
    assert(axis == 0 || axis == 1);
    Tensor newTensor;

    if (tensor.shape.size() == 1 || tensor.shape.front() == 1)
    {
        if (axis == 0)
        {
            newTensor = tensor;
        }
        else if (axis == 1)
        {
            newTensor = Zeros({1, 1});
            float sum = 0.0f;

            for (size_t i = 0; i < tensor.size; ++i)
            {
                sum += tensor[i];
            }
            newTensor[0] = sum;
        }
    }
    else
    {
        if (axis == 0)
        {
            newTensor = Zeros({1, tensor.shape.back()});

            for (size_t i = 0; i < tensor.shape.back(); ++i)
            {
                size_t idx = i;

                for (size_t j = 0; j < tensor.shape.front(); ++j)
                {
                    newTensor[i] += tensor[idx];
                    idx += tensor.shape.back();
                }
            }
        }
        else if (axis == 1)
        {
            newTensor = Zeros({tensor.shape.front(), 1});
            size_t idx = 0;

            for (size_t i = 0; i < tensor.shape.front(); ++i)
            {
                for (size_t j = 0; j < tensor.shape.back(); ++j)
                {
                    newTensor[i] += tensor[idx];
                    ++idx;
                }
            }
        }
    }

    return newTensor;
}