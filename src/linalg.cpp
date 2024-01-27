#include "array.h"
#include "kernel.h"
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
	cudaMemcpy(A, in_1.elem, in_1.size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B, in_2.elem, in_2.size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_dim(16, 16);
    dim3 grid_dim((m + block_dim.x - 1) / block_dim.x, (k + block_dim.y - 1) / block_dim.y);

    MatMul<<<grid_dim, block_dim>>>(A, B, C, m, n, k);

	Tensor out = Zeros({ in_1.shape.front(), in_2.shape.back() });

	CheckCuda(cudaMemcpy(out.elem, C, out.size * sizeof(float), cudaMemcpyDeviceToHost));
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
    return out;
}