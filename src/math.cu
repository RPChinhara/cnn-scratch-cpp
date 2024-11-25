#include "math.hpp"
#include "arrs.h"
#include "tensor.h"

#include <cassert>

tensor argmax(const tensor& t) {
    tensor t_new = zeros({t.shape.front()});

    size_t idx = 0;
    float max = std::numeric_limits<float>::lowest();
    size_t max_idx = 0;

    for (auto i = 0; i < t.shape.front(); ++i) {
        for (auto j = 0; j < t.shape.back(); ++j) {
            if (t[idx] > max) {
                max = t[idx];
                max_idx = j;
            }
            ++idx;
        }

        t_new[i] = max_idx;
        max = std::numeric_limits<float>::lowest();
    }

    return t_new;
}

__global__ void exp(float* t, float* t_new, size_t n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n)
        t_new[id] = expf(t[id]);
}

tensor exp(const tensor& t) {
    tensor t_new = t;

    float* t_gpu, * t_gpu_new;
    cudaMalloc((void**)&t_gpu, t.size * sizeof(float));
    cudaMalloc((void**)&t_gpu_new, t.size * sizeof(float));
    cudaMemcpy(t_gpu, t.elems, t.size * sizeof(float), cudaMemcpyHostToDevice);

    constexpr int blockSize = 128;
    int gridSize = (t.size + blockSize - 1) / blockSize;
    exp<<<gridSize, blockSize>>>(t_gpu, t_gpu_new, t.size);

    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
        std::cerr << "CUDA knl launch error. " + std::string(cudaGetErrorString(cudaError)) << std::endl;

    cudaMemcpy(t_new.elems, t_gpu_new, t.size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(t_gpu);
    cudaFree(t_gpu_new);

    return t_new;
}

tensor max(const tensor& t, const size_t axis) {
    assert(axis == 0 || axis == 1);
    tensor t_new;

    if (axis == 0) {
        t_new = zeros({1, t.shape.back()});

        for (auto i = 0; i < t.shape.back(); ++i) {
            size_t idx = i;
            float max = std::numeric_limits<float>::lowest();

            for (auto j = 0; j < t.shape.front(); ++j) {
                if (t[idx] > max)
                    max = t[idx];
                idx += t.shape.back();
            }

            t_new[i] = max;
        }
    } else if (axis == 1) {
        t_new = zeros({t.shape.front(), 1});
        size_t idx = 0;

        for (auto i = 0; i < t.shape.front(); ++i) {
            float max = std::numeric_limits<float>::lowest();

            for (auto j = 0; j < t.shape.back(); ++j) {
                if (t[idx] > max)
                    max = t[idx];
                ++idx;
            }

            t_new[i] = max;
        }
    }

    return t_new;
}

tensor min(const tensor& t) {
    tensor t_new = zeros({1, t.shape.back()});

    for (auto i = 0; i < t.shape.back(); ++i) {
        size_t idx = i;
        float min = std::numeric_limits<float>::max();

        for (auto j = 0; j < t.shape.front(); ++j) {
            if (t[idx] < min)
                min = t[idx];
            idx += t.shape.back();
        }

        t_new[i] = min;
    }

    return t_new;
}

tensor sqrt(const tensor& x) {
    tensor y = x;

    for (auto i = 0; i < x.size; ++i)
        y.elems[i] = sqrtf(x.elems[i]);

    return y;
}

tensor square(const tensor& t) {
    tensor y = t;

    for (auto i = 0; i < t.size; ++i)
        y.elems[i] = t.elems[i] * t.elems[i];

    return y;
}

tensor sum(const tensor& t, const size_t axis) {
    assert(axis == 0 || axis == 1);
    tensor t_new;

    if (t.shape.size() == 1 || t.shape.front() == 1) {
        if (axis == 0) {
            t_new = t;
        } else if (axis == 1) {
            t_new = zeros({1, 1});
            float sum = 0.0f;

            for (auto i = 0; i < t.size; ++i) {
                sum += t[i];
            }

            t_new[0] = sum;
        }
    } else {
        if (axis == 0) {
            t_new = zeros({1, t.shape.back()});

            for (auto i = 0; i < t.shape.back(); ++i) {
                size_t idx = i;

                for (auto j = 0; j < t.shape.front(); ++j) {
                    t_new[i] += t[idx];
                    idx += t.shape.back();
                }
            }
        } else if (axis == 1) {
            t_new = zeros({t.shape.front(), 1});
            size_t idx = 0;

            for (auto i = 0; i < t.shape.front(); ++i) {
                for (auto j = 0; j < t.shape.back(); ++j) {
                    t_new[i] += t[idx];
                    ++idx;
                }
            }
        }
    }

    return t_new;
}