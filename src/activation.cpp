#include "activation.h"
#include "array.h"
#include "device.h"
#include "kernel.h"
#include "mathematics.h"
#include "tensor.h"

#include <windows.h>

Tensor Relu(const Tensor &in, Device device)
{
    // IDEA: To 'tensor' and 'newTensor' from 'in' and 'out'? Also 'gpuTensor' and 'gpuNewTesnor' from 'in2' and 'out2'?
    switch (device)
    {
    case Device::CPU: {
        Tensor out = in;

        for (size_t i = 0; i < in.size; ++i)
            out.elem[i] = std::max(0.0f, in.elem[i]);

        return out;
    }
    case Device::GPU: {
        float *in2, *out2;
        cudaMalloc((void **)&in2, in.size * sizeof(float));
        cudaMalloc((void **)&out2, in.size * sizeof(float));
        cudaMemcpy(in2, in.elem, in.size * sizeof(float), cudaMemcpyHostToDevice);

        constexpr int blockSize = 128;
        int gridSize = (in.size + blockSize - 1) / blockSize;
        Relu<<<gridSize, blockSize>>>(in2, out2, in.size);

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
            MessageBox(nullptr, ("CUDA kernel launch error " + std::string(cudaGetErrorString(cudaError))).c_str(),
                       "Error", MB_ICONERROR);

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

Tensor Softmax(const Tensor &in)
{
    Tensor exp_scores = Exp(in - Max(in, 1), Device::CPU);
    return exp_scores / Sum(exp_scores, 1);
}