#include "math.h"
#include "arrs.h"
#include "tensor.h"

__global__ void add(float *x, float *y, float *z, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        z[idx] = x[idx] + y[idx];
    }
}

tensor add(const tensor& x, const tensor& y) {
    tensor t_new = x;

    float *d_x, *d_y, *d_z;

    cudaMalloc(&d_x, x.size * sizeof(float));
    cudaMalloc(&d_y, x.size * sizeof(float));
    cudaMalloc(&d_z, x.size * sizeof(float));

    cudaMemcpy(d_x, x.elems, x.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.elems, x.size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (x.size + threadsPerBlock - 1) / threadsPerBlock;
    add<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, x.size);

    cudaMemcpy(t_new.elems, d_z, x.size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return t_new;
}

__global__ void subtract(float *x, float *y, float *z, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        z[idx] = x[idx] - y[idx];
    }
}

tensor subtract(const tensor& x, const tensor& y) {
    tensor t_new = x;

    float *d_x, *d_y, *d_z;

    cudaMalloc(&d_x, x.size * sizeof(float));
    cudaMalloc(&d_y, x.size * sizeof(float));
    cudaMalloc(&d_z, x.size * sizeof(float));

    cudaMemcpy(d_x, x.elems, x.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.elems, x.size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (x.size + threadsPerBlock - 1) / threadsPerBlock;
    subtract<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, x.size);

    cudaMemcpy(t_new.elems, d_z, x.size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return t_new;
}

__global__ void multiply(float *x, float *y, float *z, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        z[idx] = x[idx] * y[idx];
    }
}

tensor multiply(const tensor& x, const tensor& y) {
    tensor t_new = x;

    float *d_x, *d_y, *d_z;

    cudaMalloc(&d_x, x.size * sizeof(float));
    cudaMalloc(&d_y, x.size * sizeof(float));
    cudaMalloc(&d_z, x.size * sizeof(float));

    cudaMemcpy(d_x, x.elems, x.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.elems, x.size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (x.size + threadsPerBlock - 1) / threadsPerBlock;
    multiply<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, x.size);

    cudaMemcpy(t_new.elems, d_z, x.size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return t_new;
}

__global__ void divide(float *x, float *y, float *z, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        z[idx] = x[idx] / y[idx];
    }
}

tensor divide(const tensor& x, const tensor& y) {
    tensor t_new = x;

    float *d_x, *d_y, *d_z;

    cudaMalloc(&d_x, x.size * sizeof(float));
    cudaMalloc(&d_y, x.size * sizeof(float));
    cudaMalloc(&d_z, x.size * sizeof(float));

    cudaMemcpy(d_x, x.elems, x.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.elems, x.size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (x.size + threadsPerBlock - 1) / threadsPerBlock;
    divide<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, x.size);

    cudaMemcpy(t_new.elems, d_z, x.size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return t_new;
}

__global__ void exp(float* x, float* y, size_t n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n)
        y[id] = expf(x[id]);
}

tensor exp(const tensor& x) {
    tensor y = x;

    float* d_x, * d_y;

    cudaMalloc((void**)&d_x, x.size * sizeof(float));
    cudaMalloc((void**)&d_y, x.size * sizeof(float));

    cudaMemcpy(d_x, x.elems, x.size * sizeof(float), cudaMemcpyHostToDevice);

    constexpr int blockSize = 128;
    int gridSize = (x.size + blockSize - 1) / blockSize;
    exp<<<gridSize, blockSize>>>(d_x, d_y, x.size);

    cudaMemcpy(y.elems, d_y, x.size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);

    return y;
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

tensor max(const tensor& t, const size_t axis) {
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

tensor sum(const tensor& t, const size_t axis) {
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