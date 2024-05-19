#include "act.h"
#include "knls.h"
#include "math.hpp"
#include "ten.h"

ten act(const ten &z, act_type act, dev_type dev)
{
    switch (act)
    {
    case RELU: {
        ten t_new = z;

        switch (dev)
        {
        case CPU: {
            for (auto i = 0; i < z.size; ++i)
                t_new.elem[i] = std::fmax(0.0f, z.elem[i]);

            return t_new;
        }
        case GPU: {
            float *t_gpu, *t_gpu_new;
            cudaMalloc((void **)&t_gpu, z.size * sizeof(float));
            cudaMalloc((void **)&t_gpu_new, z.size * sizeof(float));
            cudaMemcpy(t_gpu, z.elem, z.size * sizeof(float), cudaMemcpyHostToDevice);

            constexpr int blockSize = 128;
            int gridSize = (z.size + blockSize - 1) / blockSize;
            relu<<<gridSize, blockSize>>>(t_gpu, t_gpu_new, z.size);

            cudaError_t cudaError = cudaGetLastError();
            if (cudaError != cudaSuccess)
                std::cerr << "CUDA knl launch error. " + std::string(cudaGetErrorString(cudaError)) << std::endl;

            cudaMemcpy(t_new.elem, t_gpu_new, z.size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(t_gpu);
            cudaFree(t_gpu_new);

            return t_new;
        }
        default:
            std::cout << "Unknown dev." << std::endl;
            return ten();
        }
    }
    case SIGMOID: {
        ten a = z;

        for (auto i = 0; i < z.size; ++i)
            a.elem[i] = 1.0f / (1.0f + std::expf(-z.elem[i]));

        return a;
    }
    case SOFTMAX: {
        ten exp_scores = exp(z - max(z, 1), CPU);
        return exp_scores / sum(exp_scores, 1);
    }
    case TANH: {
        ten a = z;

        for (auto i = 0; i < z.size; ++i)
            a.elem[i] = std::tanhf(z.elem[i]);

        return a;
    }
    default:
        std::cout << "Unknown act." << std::endl;
        return ten();
    }
}