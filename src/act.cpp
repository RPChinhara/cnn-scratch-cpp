#include "act.h"
#include "knls.h"
#include "math.hpp"
#include "ten.h"

ten act(const ten &t, act_enum act, dev_type dev)
{

    switch (act)
    {
    case ACT_RELU: {
        ten newTensor = t;

        switch (dev)
        {
        case DEV_CPU: {
            for (size_t i = 0; i < t.size; ++i)
                newTensor.elem[i] = std::max(0.0f, t.elem[i]);

            return newTensor;
        }
        case DEV_GPU: {
            float *tensorGPU, *newTensorGPU;
            cudaMalloc((void **)&tensorGPU, t.size * sizeof(float));
            cudaMalloc((void **)&newTensorGPU, t.size * sizeof(float));
            cudaMemcpy(tensorGPU, t.elem, t.size * sizeof(float), cudaMemcpyHostToDevice);

            constexpr int blockSize = 128;
            int gridSize = (t.size + blockSize - 1) / blockSize;
            Relu<<<gridSize, blockSize>>>(tensorGPU, newTensorGPU, t.size);

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
            return ten();
        }
    }
    case ACT_SOFTMAX: {
        ten expScores = Exp(t - Max(t, 1), DEV_CPU);
        // std::cout << "expScores" << std::endl;
        // std::cout << expScores << std::endl;
        // std::cout << "Sum(expScores, 1)" << std::endl;
        // std::cout << Sum(expScores, 1) << std::endl;
        // std::cout << "expScores / Sum(expScores, 1)" << std::endl;
        // std::cout << expScores / Sum(expScores, 1) << std::endl;

        return expScores / Sum(expScores, 1);
    }
    default:
        std::cout << "Unknown act." << std::endl;
        return ten();
    }
}