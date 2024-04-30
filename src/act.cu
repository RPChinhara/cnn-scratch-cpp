#include "act.h"
#include "knls.h"
#include "math.hpp"
#include "ten.h"

ten act(const ten &t, act_enum act, dev_type dev)
{
    switch (act)
    {
    case RELU: {
        ten t_new = t;

        switch (dev)
        {
        case CPU: {
            for (auto i = 0; i < t.size; ++i)
                t_new.elem[i] = std::max(0.0f, t.elem[i]);

            return t_new;
        }
        case GPU: {
            float *t_gpu, *t_gpu_new;
            cudaMalloc((void **)&t_gpu, t.size * sizeof(float));
            cudaMalloc((void **)&t_gpu_new, t.size * sizeof(float));
            cudaMemcpy(t_gpu, t.elem, t.size * sizeof(float), cudaMemcpyHostToDevice);

            constexpr int blockSize = 128;
            int gridSize = (t.size + blockSize - 1) / blockSize;
            relu<<<gridSize, blockSize>>>(t_gpu, t_gpu_new, t.size);

            cudaError_t cudaError = cudaGetLastError();
            if (cudaError != cudaSuccess)
                std::cerr << "CUDA knl launch error. " + std::string(cudaGetErrorString(cudaError)) << std::endl;

            cudaMemcpy(t_new.elem, t_gpu_new, t.size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(t_gpu);
            cudaFree(t_gpu_new);

            return t_new;
        }
        default:
            std::cout << "Unknown dev." << std::endl;
            return ten();
        }
    }
    case SOFTMAX: {
        ten exp_scores = Exp(t - Max(t, 1), CPU);
        return exp_scores / sum(exp_scores, 1);
    }
    default:
        std::cout << "Unknown act." << std::endl;
        return ten();
    }
}