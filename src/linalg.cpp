#include "kernels.h"
#include "linalg.h"
#include "tensor.h"

#include <cassert>

static void chk_cuda(cudaError_t code, const bool abort = true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), __FILE__, __LINE__);
      if (abort) 
	  	exit(code);
   }
}

Tensor matmul(const Tensor& in1, const Tensor& in2) {
    assert(in1._shape.back() == in2._shape.front());
    int m = in1._shape.front();
    int n = in1._shape.back();
    int k = in2._shape.back();

    float *A, *B, *C;
    cudaMalloc(&A, m * n * sizeof(float));
    cudaMalloc(&B, n * k * sizeof(float));
    cudaMalloc(&C, m * k * sizeof(float));
	cudaMemcpy(A, in1._elem, sizeof(float) * in1._size, cudaMemcpyHostToDevice);
	cudaMemcpy(B, in2._elem, sizeof(float) * in2._size, cudaMemcpyHostToDevice);

    dim3 block_dim(16, 16);
    dim3 grid_dim((m + block_dim.x - 1) / block_dim.x, (k + block_dim.y - 1) / block_dim.y);

    MatMul<<<grid_dim, block_dim>>>(A, B, C, m, n, k);

	Tensor out = Tensor({ 0.0f }, { in1._shape.front(), in2._shape.back() });

	chk_cuda(cudaMemcpy(out._elem, C, sizeof(float) * out._size, cudaMemcpyDeviceToHost));
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
    return out;
}

static unsigned int get_batch_size(const std::vector<unsigned int>& shape) {
    assert(shape.size() > 1);
    unsigned int batch_size = 1;

    for (unsigned short i = 0; i < shape.size() - 2; ++i)
        batch_size *= shape[i];
    return batch_size;
}

Tensor transpose(const Tensor& in) {
    assert(in._shape.size() >= 2);

    Tensor out = Tensor({ 0.0f }, { in._shape.back(), in._shape[in._shape.size() - 2] });

    out._num_ch_dim = 1;

    for (int i = 0; i < out._shape.size() - 1; ++i)
        out._num_ch_dim *= out._shape[i];
    
    std::vector<unsigned short> idx_rows;
    
    for (unsigned short i = 0; i < in._num_ch_dim; ++i)
        idx_rows.push_back(i * in._shape.back());

    unsigned short batch_size = get_batch_size(in._shape);

    unsigned int idx{};
    for (unsigned int i = 0; i < batch_size; ++i) {
        for (unsigned int j = 0; j < out._shape[out._shape.size() - 2]; ++j) {
            for (unsigned int k = 0; k < out._shape.back(); ++k) {
                out[idx] = in[idx_rows[k + (i * out._shape.back())]];
                idx_rows[k + (i * out._shape.back())] += 1;
                ++idx;
            }
        }
    }
	return out;
}