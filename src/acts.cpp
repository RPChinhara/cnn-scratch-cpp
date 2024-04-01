#include "acts.h"
#include "arrs.h"
#include "kernels.h"
#include "math.hpp"
#include "ten.h"

Tensor Relu(const Tensor &tensor, Dev dev)
{
    Tensor newTensor = tensor;

    switch (dev)
    {
    case Dev::CPU: {
        for (size_t i = 0; i < tensor.size; ++i)
            newTensor.elem[i] = std::max(0.0f, tensor.elem[i]);

        return newTensor;
    }
    case Dev::GPU: {
        float *tensorGPU, *newTensorGPU;
        cudaMalloc((void **)&tensorGPU, tensor.size * sizeof(float));
        cudaMalloc((void **)&newTensorGPU, tensor.size * sizeof(float));
        cudaMemcpy(tensorGPU, tensor.elem, tensor.size * sizeof(float), cudaMemcpyHostToDevice);

        constexpr int blockSize = 128;
        int gridSize = (tensor.size + blockSize - 1) / blockSize;
        Relu<<<gridSize, blockSize>>>(tensorGPU, newTensorGPU, tensor.size);

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
            std::cerr << "CUDA kernel launch error." + std::string(cudaGetErrorString(cudaError)) << std::endl;

        cudaMemcpy(newTensor.elem, newTensorGPU, tensor.size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(tensorGPU);
        cudaFree(newTensorGPU);

        return newTensor;
    }
    default:
        std::cout << "Unknown dev." << std::endl;
        return Tensor();
    }
}

Tensor Softmax(const Tensor &tensor)
{
    Tensor expScores = Exp(tensor - Max(tensor, 1), Dev::CPU);
    return expScores / Sum(expScores, 1);
}