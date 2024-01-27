#include "array.h"
#include "kernel.h"
#include "mathematics.h"
#include "tensor.h"

Tensor Relu(const Tensor& in)
{
    float **in_out = new float*[sizeof(float *) * 2];
	cudaMalloc((void**) &in_out[0], sizeof(float) * in.size);
	cudaMalloc((void**) &in_out[1], sizeof(float) * in.size);
	cudaMemcpy(in_out[0], in.elem, sizeof(float) * in.size, cudaMemcpyHostToDevice);

    int blockSize = 512;
    int gridSize = (in.size + blockSize - 1) / blockSize;
	Relu<<<gridSize, blockSize>>>(in_out[0], in_out[1], in.size);

    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(cudaError) << std::endl;

	Tensor out = in;
	cudaMemcpy(out.elem, in_out[1], sizeof(float) * in.size, cudaMemcpyDeviceToHost);
	cudaFree(in_out[0]);
	cudaFree(in_out[1]);
	
    delete[] in_out;

	return out;
}

Tensor Softmax(const Tensor& in)
{
    Tensor exp_scores = Exp(in - Max(in, 1));
    return exp_scores / Sum(exp_scores, 1);
}