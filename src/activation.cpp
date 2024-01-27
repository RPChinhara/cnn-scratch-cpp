#include "array.h"
#include "kernel.h"
#include "mathematics.h"
#include "tensor.h"

Tensor Relu(const Tensor& in)
{
    float *in2, *out2;
	cudaMalloc((void**)&in2, in.size * sizeof(float));
	cudaMalloc((void**)&out2, in.size * sizeof(float));
	cudaMemcpy(in2, in.elem, in.size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 128;
    int gridSize = (in.size + blockSize - 1) / blockSize;
	Relu<<<gridSize, blockSize>>>(in2, out2, in.size);

    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(cudaError) << std::endl;

	Tensor out = in;
	cudaMemcpy(out.elem, out2, in.size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(in2);
	cudaFree(out2);
	
	return out;
}

Tensor Softmax(const Tensor& in)
{
    Tensor exp_scores = Exp(in - Max(in, 1));
    return exp_scores / Sum(exp_scores, 1);
}