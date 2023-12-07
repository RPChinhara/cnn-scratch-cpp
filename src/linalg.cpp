#include "kernel.h"
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
    int m = in_1.shape.front();
    int n = in_1.shape.back();
    int k = in_2.shape.back();

    float *A, *B, *C;
    cudaMalloc(&A, m * n * sizeof(float));
    cudaMalloc(&B, n * k * sizeof(float));
    cudaMalloc(&C, m * k * sizeof(float));
	cudaMemcpy(A, in_1.elem, sizeof(float) * in_1.size, cudaMemcpyHostToDevice);
	cudaMemcpy(B, in_2.elem, sizeof(float) * in_2.size, cudaMemcpyHostToDevice);

    dim3 block_dim(16, 16);
    dim3 grid_dim((m + block_dim.x - 1) / block_dim.x, (k + block_dim.y - 1) / block_dim.y);

    MatMul<<<grid_dim, block_dim>>>(A, B, C, m, n, k);

	Tensor out = Tensor({ 0.0f }, { in_1.shape.front(), in_2.shape.back() });

	CheckCuda(cudaMemcpy(out.elem, C, sizeof(float) * out.size, cudaMemcpyDeviceToHost));
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
    return out;
}

static unsigned int GetBatchSize(const std::vector<unsigned int>& shape)
{
    assert(shape.size() > 1);
    unsigned int batch_size = 1;

    for (unsigned short i = 0; i < shape.size() - 2; ++i)
        batch_size *= shape[i];
    
    return batch_size;
}

Tensor Transpose(const Tensor& in)
{
    assert(in.shape.size() >= 2);

    Tensor out = Tensor({ 0.0f }, { in.shape.back(), in.shape[in.shape.size() - 2] });

    out.num_ch_dim = 1;

    for (int i = 0; i < out.shape.size() - 1; ++i)
        out.num_ch_dim *= out.shape[i];
    
    std::vector<unsigned short> idx_rows;
    
    for (unsigned short i = 0; i < in.num_ch_dim; ++i)
        idx_rows.push_back(i * in.shape.back());

    unsigned short batch_size = GetBatchSize(in.shape);

    unsigned int idx{};
    for (unsigned int i = 0; i < batch_size; ++i) {
        for (unsigned int j = 0; j < out.shape[out.shape.size() - 2]; ++j) {
            for (unsigned int k = 0; k < out.shape.back(); ++k) {
                out[idx] = in[idx_rows[k + (i * out.shape.back())]];
                idx_rows[k + (i * out.shape.back())] += 1;
                ++idx;
            }
        }
    }
	return out;
}