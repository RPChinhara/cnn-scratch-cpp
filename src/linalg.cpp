#include "kernel.h"
#include "array.h"
#include "linalg.h"
#include "tensor.h"

#include <cassert>

static void CheckCuda(cudaError_t code, const bool abort = true)
{
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), __FILE__, __LINE__);
      if (abort) 
	  	exit(code);
   }
}

Tensor MatMul(const Tensor& in_1, const Tensor& in_2)
{
    assert(in_1.shape.back() == in_2.shape.front());
    size_t m = in_1.shape.front();
    size_t n = in_1.shape.back();
    size_t k = in_2.shape.back();

    float *A, *B, *C;
    cudaMalloc(&A, m * n * sizeof(float));
    cudaMalloc(&B, n * k * sizeof(float));
    cudaMalloc(&C, m * k * sizeof(float));
	cudaMemcpy(A, in_1.elem, sizeof(float) * in_1.size, cudaMemcpyHostToDevice);
	cudaMemcpy(B, in_2.elem, sizeof(float) * in_2.size, cudaMemcpyHostToDevice);

    dim3 block_dim(16, 16);
    dim3 grid_dim((m + block_dim.x - 1) / block_dim.x, (k + block_dim.y - 1) / block_dim.y);

    MatMul<<<grid_dim, block_dim>>>(A, B, C, m, n, k);

	Tensor out = Zeros({ in_1.shape.front(), in_2.shape.back() });

	CheckCuda(cudaMemcpy(out.elem, C, sizeof(float) * out.size, cudaMemcpyDeviceToHost));
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
    return out;
}

static size_t GetBatchSize(const std::vector<size_t>& shape)
{
    assert(shape.size() > 1);
    size_t batch_size = 1;

    for (size_t i = 0; i < shape.size() - 2; ++i)
        batch_size *= shape[i];
    
    return batch_size;
}

Tensor Transpose(const Tensor& in)
{
    assert(in.shape.size() >= 2);

    Tensor out = Zeros({ in.shape.back(), in.shape[in.shape.size() - 2] });

    out.num_ch_dim = 1;

    for (size_t i = 0; i < out.shape.size() - 1; ++i)
        out.num_ch_dim *= out.shape[i];
    
    std::vector<size_t> idx_rows;
    
    for (size_t i = 0; i < in.num_ch_dim; ++i)
        idx_rows.push_back(i * in.shape.back());

    size_t batch_size = GetBatchSize(in.shape);

    size_t idx{};

    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < out.shape[out.shape.size() - 2]; ++j) {
            for (size_t k = 0; k < out.shape.back(); ++k) {
                out[idx] = in[idx_rows[k + (i * out.shape.back())]];
                idx_rows[k + (i * out.shape.back())] += 1;
                ++idx;
            }
        }
    }
    
	return out;
}