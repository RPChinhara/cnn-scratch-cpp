#include "activations.h"
#include "arrays.h"
#include "device.h"
#include "kernels.h"
#include "mathematics.h"
#include "tensor.h"

#include <windows.h>

Tensor Relu(const Tensor &tensor, Device device)
{
    Tensor newTensor = tensor;

    switch (device)
    {
    case Device::CPU: {
        for (size_t i = 0; i < tensor.size; ++i)
            newTensor.elem[i] = std::max(0.0f, tensor.elem[i]);

        return newTensor;
    }
    case Device::GPU: {
        float *tensorGPU, *newTensorGPU;
        cudaMalloc((void **)&tensorGPU, tensor.size * sizeof(float));
        cudaMalloc((void **)&newTensorGPU, tensor.size * sizeof(float));
        cudaMemcpy(tensorGPU, tensor.elem, tensor.size * sizeof(float), cudaMemcpyHostToDevice);

        constexpr int blockSize = 128;
        int gridSize = (tensor.size + blockSize - 1) / blockSize;
        Relu<<<gridSize, blockSize>>>(tensorGPU, newTensorGPU, tensor.size);

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

Tensor Softmax(const Tensor &tensor)
{
    Tensor expScores = Exp(tensor - Max(tensor, 1), Device::CPU);
    return expScores / Sum(expScores, 1);
}