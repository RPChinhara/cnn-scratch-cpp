#include "mathematics.h"
#include "array.h"
#include "kernel.h"
#include "tensor.h"

#include <cassert>
#include <windows.h>

Tensor Argmax(const Tensor &in)
{
    Tensor out = Zeros({in.shape.front()});

    size_t idx = 0;
    float max = std::numeric_limits<float>::lowest();
    size_t max_idx = 0;

    for (size_t i = 0; i < in.shape.front(); ++i)
    {
        for (size_t j = 0; j < in.shape.back(); ++j)
        {
            if (in[idx] > max)
            {
                max = in[idx];
                max_idx = j;
            }
            ++idx;
        }
        out[i] = max_idx;
        max = std::numeric_limits<float>::lowest();
    }

    return out;
}

Tensor Exp(const Tensor &in, Device device)
{
    switch (device)
    {
    case Device::CPU: {
        Tensor out = in;

        for (size_t i = 0; i < in.size; ++i)
            out.elem[i] = expf(in.elem[i]);

        return out;
    }
    case Device::GPU: {
        float *in2, *out2;
        cudaMalloc((void **)&in2, in.size * sizeof(float));
        cudaMalloc((void **)&out2, in.size * sizeof(float));
        cudaMemcpy(in2, in.elem, in.size * sizeof(float), cudaMemcpyHostToDevice);

        int blockSize = 128;
        int gridSize = (in.size + blockSize - 1) / blockSize;
        Exp<<<gridSize, blockSize>>>(in2, out2, in.size);

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
            std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(cudaError) << std::endl;

        Tensor out = in;
        cudaMemcpy(out.elem, out2, in.size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(in2);
        cudaFree(out2);

        return out;
    }
    default:
        MessageBox(nullptr, "Unknown device", "Error", MB_ICONERROR);
        return Tensor();
    }
}

Tensor Log(const Tensor &in, Device device)
{
    switch (device)
    {
    case Device::CPU: {
        Tensor out = in;

        for (size_t i = 0; i < in.size; ++i)
            out.elem[i] = logf(in.elem[i]);

        return out;
    }
    case Device::GPU: {
        float *in2, *out2;
        cudaMalloc((void **)&in2, in.size * sizeof(float));
        cudaMalloc((void **)&out2, in.size * sizeof(float));
        cudaMemcpy(in2, in.elem, in.size * sizeof(float), cudaMemcpyHostToDevice);

        int blockSize = 128;
        int gridSize = (in.size + blockSize - 1) / blockSize;
        Log<<<gridSize, blockSize>>>(in2, out2, in.size);

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
            std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(cudaError) << std::endl;

        Tensor out = in;
        cudaMemcpy(out.elem, out2, in.size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(in2);
        cudaFree(out2);

        return out;
    }
    default:
        MessageBox(nullptr, "Unknown device", "Error", MB_ICONERROR);
        return Tensor();
    }
}

Tensor Max(const Tensor &in, const size_t axis)
{
    // DEPRECATED: I may only need axis 1 pattern for Max()
    assert(axis == 0 || axis == 1);
    Tensor out;

    if (axis == 0)
    {
        out = Zeros({1, in.shape.back()});

        for (size_t i = 0; i < in.shape.back(); ++i)
        {
            size_t idx = i;
            float max = std::numeric_limits<float>::lowest();

            for (size_t j = 0; j < in.shape.front(); ++j)
            {
                if (in[idx] > max)
                    max = in[idx];
                idx += in.shape.back();
            }

            out[i] = max;
        }
    }
    else if (axis == 1)
    {
        out = Zeros({in.shape.front(), 1});
        size_t idx = 0;

        for (size_t i = 0; i < in.shape.front(); ++i)
        {
            float max = std::numeric_limits<float>::lowest();

            for (size_t j = 0; j < in.shape.back(); ++j)
            {
                if (in[idx] > max)
                    max = in[idx];
                ++idx;
            }

            out[i] = max;
        }
    }

    return out;
}

Tensor Min(const Tensor &in)
{
    Tensor out = Zeros({1, in.shape.back()});

    for (size_t i = 0; i < in.shape.back(); ++i)
    {
        size_t idx = i;
        float min = std::numeric_limits<float>::max();

        for (size_t j = 0; j < in.shape.front(); ++j)
        {
            if (in[idx] < min)
                min = in[idx];
            idx += in.shape.back();
        }
        out[i] = min;
    }

    return out;
}

Tensor Sum(const Tensor &in, const size_t axis)
{
    assert(axis == 0 || axis == 1);
    Tensor out;

    if (in.shape.size() == 1 || in.shape.front() == 1)
    {
        if (axis == 0)
        {
            out = in;
        }
        else if (axis == 1)
        {
            out = Zeros({1, 1});
            float sum = 0.0f;

            for (size_t i = 0; i < in.size; ++i)
            {
                sum += in[i];
            }
            out[0] = sum;
        }
    }
    else
    {
        if (axis == 0)
        {
            out = Zeros({1, in.shape.back()});

            for (size_t i = 0; i < in.shape.back(); ++i)
            {
                size_t idx = i;

                for (size_t j = 0; j < in.shape.front(); ++j)
                {
                    out[i] += in[idx];
                    idx += in.shape.back();
                }
            }
        }
        else if (axis == 1)
        {
            out = Zeros({in.shape.front(), 1});
            size_t idx = 0;

            for (size_t i = 0; i < in.shape.front(); ++i)
            {
                for (size_t j = 0; j < in.shape.back(); ++j)
                {
                    out[i] += in[idx];
                    ++idx;
                }
            }
        }
    }

    return out;
}