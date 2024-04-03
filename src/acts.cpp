#include "acts.h"
#include "knls.h"
#include "math.hpp"
#include "ten.h"

Ten act(const Ten &ten, Act act, Dev dev)
{

    switch (act)
    {
    case RELU: {
        Ten newTensor = ten;

        switch (dev)
        {
        case Dev::CPU: {
            for (size_t i = 0; i < ten.size; ++i)
                newTensor.elem[i] = std::max(0.0f, ten.elem[i]);

            return newTensor;
        }
        case Dev::GPU: {
            float *tensorGPU, *newTensorGPU;
            cudaMalloc((void **)&tensorGPU, ten.size * sizeof(float));
            cudaMalloc((void **)&newTensorGPU, ten.size * sizeof(float));
            cudaMemcpy(tensorGPU, ten.elem, ten.size * sizeof(float), cudaMemcpyHostToDevice);

            constexpr int blockSize = 128;
            int gridSize = (ten.size + blockSize - 1) / blockSize;
            Relu<<<gridSize, blockSize>>>(tensorGPU, newTensorGPU, ten.size);

            cudaError_t cudaError = cudaGetLastError();
            if (cudaError != cudaSuccess)
                std::cerr << "CUDA kernel launch error. " + std::string(cudaGetErrorString(cudaError)) << std::endl;

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
    case SOFTMAX: {
        Ten expScores = Exp(ten - Max(ten, 1), Dev::CPU);
        return expScores / Sum(expScores, 1);
    }
    default:
        std::cout << "Unknown act." << std::endl;
        return Ten();
    }
}