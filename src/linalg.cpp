#include "linalg.h"
#include "array.h"
#include "kernel.h"
#include "tensor.h"

#include <cassert>
#include <windows.h>

Tensor MatMul(const Tensor &tensor1, const Tensor &tensor2, Device device)
{
    Tensor newTensor = Zeros({tensor1.shape.front(), tensor2.shape.back()});

    switch (device)
    {
    case Device::CPU: {

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
    case Device::GPU: {
        assert(tensor1.shape.back() == tensor2.shape.front());
        size_t m = tensor1.shape.front();
        size_t n = tensor1.shape.back();
        size_t k = tensor2.shape.back();

        float *tensorGPU1, *tensorGPU2, *newTensorGPU;
        cudaMalloc(&tensorGPU1, m * n * sizeof(float));
        cudaMalloc(&tensorGPU2, n * k * sizeof(float));
        cudaMalloc(&newTensorGPU, m * k * sizeof(float));
        cudaMemcpy(tensorGPU1, tensor1.elem, tensor1.size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(tensorGPU2, tensor2.elem, tensor2.size * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block_dim(16, 16);
        dim3 grid_dim((m + block_dim.x - 1) / block_dim.x, (k + block_dim.y - 1) / block_dim.y);
        MatMul<<<grid_dim, block_dim>>>(tensorGPU1, tensorGPU2, newTensorGPU, m, n, k);

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
            MessageBox(nullptr, ("CUDA kernel launch error " + std::string(cudaGetErrorString(cudaError))).c_str(),
                       "Error", MB_ICONERROR);

        cudaMemcpy(newTensor.elem, newTensorGPU, newTensor.size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(tensorGPU1);
        cudaFree(tensorGPU2);
        cudaFree(newTensorGPU);

        return newTensor;
    }
    default:
        MessageBox(nullptr, "Unknown device", "Error", MB_ICONERROR);
        return Tensor();
    }
}