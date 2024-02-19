#include "linalg.h"
#include "array.h"
#include "kernel.h"
#include "tensor.h"

#include <cassert>
#include <windows.h>

Tensor MatMul(const Tensor &in1, const Tensor &in2, Device device)
{
    switch (device)
    {
    case Device::CPU: {
        Tensor out = Zeros({in1.shape.front(), in2.shape.back()});

        for (size_t i = 0; i < in1.shape.front(); ++i)
        {
            for (size_t j = 0; j < in2.shape.back(); ++j)
            {
                float sum = 0.0;

                for (size_t l = 0; l < in1.shape.back(); ++l)
                    sum += in1[i * in1.shape.back() + l] * in2[l * in2.shape.back() + j];

                out[i * in2.shape.back() + j] = sum;
            }
        }

        return out;
    }
    case Device::GPU: {
        assert(in1.shape.back() == in2.shape.front());
        size_t m = in1.shape.front();
        size_t n = in1.shape.back();
        size_t k = in2.shape.back();

        float *A, *B, *C;
        cudaMalloc(&A, m * n * sizeof(float));
        cudaMalloc(&B, n * k * sizeof(float));
        cudaMalloc(&C, m * k * sizeof(float));
        cudaMemcpy(A, in1.elem, in1.size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(B, in2.elem, in2.size * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block_dim(16, 16);
        dim3 grid_dim((m + block_dim.x - 1) / block_dim.x, (k + block_dim.y - 1) / block_dim.y);
        MatMul<<<grid_dim, block_dim>>>(A, B, C, m, n, k);

        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
            MessageBox(nullptr, ("CUDA kernel launch error " + std::string(cudaGetErrorString(cudaError))).c_str(),
                       "Error", MB_ICONERROR);

        Tensor out = Zeros({in1.shape.front(), in2.shape.back()});
        cudaMemcpy(out.elem, C, out.size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);

        return out;
    }
    default:
        MessageBox(nullptr, "Unknown device", "Error", MB_ICONERROR);
        return Tensor();
    }
}