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
    int m = in1._shape.front(); // Dimension of A
    int n = in1._shape.back();  // Dimension of B (shared dimension)
    int k = in2._shape.back();  // Dimension of C

    f32 *A, *B, *C;
    cudaMalloc(&A, m * n * sizeof(f32));
    cudaMalloc(&B, n * k * sizeof(f32));
    cudaMalloc(&C, m * k * sizeof(f32));
	cudaMemcpy(A, in1._elem, sizeof(f32) * in1._size, cudaMemcpyHostToDevice);
	cudaMemcpy(B, in2._elem, sizeof(f32) * in2._size, cudaMemcpyHostToDevice);

    dim3 block_dim(16, 16);
    dim3 grid_dim((m + block_dim.x - 1) / block_dim.x, (k + block_dim.y - 1) / block_dim.y);

    matmul<<<grid_dim, block_dim>>>(A, B, C, m, n, k);

	Tensor out = Tensor({ 0.0f }, { in1._shape.front(), in2._shape.back() });

	chk_cuda(cudaMemcpy(out._elem, C, sizeof(f32) * out._size, cudaMemcpyDeviceToHost));
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
    return out;
}

static u32 get_batch_size(const std::vector<u32>& shape) {
    // It must be a matrix.
    assert(shape.size() > 1);
    u32 batch_size = 1;
    for (u16 i = 0; i < shape.size() - 2; ++i)
        // Multiply each digits except digits for most inner matrix e.g., { 2, 2, 4, 3 }, then it'd be 4.
        batch_size *= shape[i];
    return batch_size;
}

Tensor transpose(const Tensor& in) {
    assert(in._shape.size() >= 2);

    Tensor out = Tensor({ 0.0f }, { in._shape });

    // Switch last two dimensions.
    u32 tmp{};
    tmp = out._shape.back();
    out._shape[in._shape.size() - 1] = out._shape[in._shape.size() - 2];
    out._shape[in._shape.size() - 2] = tmp;

    // Reset '_num_ch_dim'.
    out._num_ch_dim = 1;
    for (s32 i = 0; i < out._shape.size() - 1; ++i)
        out._num_ch_dim *= out._shape[i];
    
    // Create first index of each rows e.g., if the Tensor's elements = [1, 2, 3, 4, 5, 6] and shape = [2, 3], then it'd be [0, 3] which is indexes of each first rows.
    std::vector<u16> rows;
    for (u16 i = 0; i < in._num_ch_dim; ++i)
        rows.push_back(i * in._shape.back());

    u16 batch_size = get_batch_size(in._shape);
    
    // Assing values from each elements from rows.
    u32 num_elems{};
    // Loop through number batches of 't'.
    for (u32 i = 0; i < batch_size; ++i) {
        // Loop through number of rows of 't'.
        for (u32 j{}; j < out._shape[out._shape.size() - 2]; ++j) {
            // Loop through number of colums of 't'.
            for (u32 k{}; k < out._shape.back(); ++k) {
                // Multiply by (i * out._shape.back()) so that 'rows' will change for every batches otherwise it will loop through for a same matrix.
                // Assign each elements from rows.
                out[num_elems] = in[rows[k + (i * out._shape.back())]];
                // Increment each index.
                rows[k + (i * out._shape.back())] += 1;
                ++num_elems;
            }
        }
    }
	return out;
}