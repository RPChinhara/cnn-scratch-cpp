#include "kernels.h"
#include "mathematics.h"
#include "tensor.h"

#include <cassert>

static constexpr int NUM_PROCS = 128 + (32 * 1);

static void chk_cuda(cudaError_t code, const bool abort = true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), __FILE__, __LINE__);
      if (abort) 
	  	exit(code);
   }
}

Tensor argmax(const Tensor& in) {
	Tensor out  = Tensor({ 0.0f }, { in._shape.front() });
	unsigned short idx = 0;
	float max = std::numeric_limits<float>::lowest();
	unsigned int max_idx = 0;
	
	for (unsigned int i = 0; i < in._shape.front(); ++i) {
		for (unsigned int j = 0; j < in._shape.back(); ++j) {
			if (in[idx] > max) {
				max     = in[idx];
				max_idx = j;
			}
			++idx;
		}
		out[i] = max_idx;
		max = std::numeric_limits<float>::lowest();
	}
	return out;
}

Tensor exp(const Tensor& in) {
	float **in_out = new float*[sizeof(float *) * 2];
	chk_cuda(cudaMalloc((void**) &in_out[0], sizeof(float) * in._size));
	chk_cuda(cudaMalloc((void**) &in_out[1], sizeof(float) * in._size));
	chk_cuda(cudaMemcpy(in_out[0], in._elem, sizeof(float) * in._size, cudaMemcpyHostToDevice));
	exp<<<in._size / NUM_PROCS + 1, NUM_PROCS>>>(in_out[0], in_out[1], in._size);
	Tensor out = in;
	chk_cuda(cudaMemcpy(out._elem, in_out[1], sizeof(float) * in._size, cudaMemcpyDeviceToHost));
	cudaFree(in_out[0]);
	cudaFree(in_out[1]);
    delete[] in_out;
	return out;
}

Tensor log(const Tensor& in) {
	float **in_out = new float*[sizeof(float *) * 2];
	chk_cuda(cudaMalloc((void**) &in_out[0], sizeof(float) * in._size));
	chk_cuda(cudaMalloc((void**) &in_out[1], sizeof(float) * in._size));
	chk_cuda(cudaMemcpy(in_out[0], in._elem, sizeof(float) * in._size, cudaMemcpyHostToDevice));
	log<<<in._size / NUM_PROCS + 1, NUM_PROCS>>>(in_out[0], in_out[1], in._size);
	Tensor out = in;
	chk_cuda(cudaMemcpy(out._elem, in_out[1], sizeof(float) * in._size, cudaMemcpyDeviceToHost));
	cudaFree(in_out[0]);
	cudaFree(in_out[1]);
    delete[] in_out;
	return out;
}

Tensor max(const Tensor& in, const unsigned short axis) {
	assert(axis == 0 || axis == 1);
	Tensor out;
	if (axis == 0) {
		out = Tensor({ 0.0f }, { 1, in._shape.back()});
		for (unsigned short i = 0; i < in._shape.back(); ++i) {
			unsigned short idx = i;
			float max = std::numeric_limits<float>::lowest();
			for (unsigned short j = 0; j < in._shape.front(); ++j) {
				if (in[idx] > max) 
					max = in[idx];
				idx += in._shape.back();
			}
			out[i] = max;
		}
	} else if (axis == 1) {
		out = Tensor({ 0.0f }, { in._shape.front(), 1});
		unsigned short idx = 0;
		for (unsigned short i = 0; i < in._shape.front(); ++i) {
			float max = std::numeric_limits<float>::lowest();
			for (unsigned short j = 0; j < in._shape.back(); ++j) {
				if (in[idx] > max) 
					max = in[idx];
				++idx;
			}
			out[i] = max;
		}
	}
	return out;
}

Tensor maximum(const Tensor& in1, const Tensor& in2) {
	float **in_out = new float*[sizeof(float *) * 3];
	chk_cuda(cudaMalloc((void**) &in_out[0], sizeof(float) * in1._size));
	chk_cuda(cudaMalloc((void**) &in_out[1], sizeof(float) * in1._size));
	chk_cuda(cudaMalloc((void**) &in_out[2], sizeof(float) * in1._size));
	chk_cuda(cudaMemcpy(in_out[0], in1._elem, sizeof(float) * in1._size, cudaMemcpyHostToDevice));
	chk_cuda(cudaMemcpy(in_out[1], in2._elem, sizeof(float) * in1._size, cudaMemcpyHostToDevice));
	maximum<<<in1._size / NUM_PROCS + 1, NUM_PROCS>>>(in_out[0], in_out[1], in_out[2], in1._size);
	Tensor out = in1;
	chk_cuda(cudaMemcpy(out._elem, in_out[2], sizeof(float) * in1._size, cudaMemcpyDeviceToHost));
	cudaFree(in_out[0]);
	cudaFree(in_out[1]);
	cudaFree(in_out[2]);
	delete[] in_out;
	return out;
}

Tensor mean(const Tensor& in) {
	return Tensor();
}

Tensor min(const Tensor& in) {
	Tensor out = Tensor({ 0.0f }, { 1, in._shape.back() });
	for (unsigned short i = 0; i < in._shape.back(); ++i) {
		unsigned short idx = i;
		float min = std::numeric_limits<float>::max();
		for (unsigned short j = 0; j < in._shape.front(); ++j) {
			if (in[idx] < min) 
				min = in[idx];
			idx += in._shape.back();
		}
		out[i] = min;
	}
	return out;
}

Tensor square(const Tensor& in) {
	float **in_out = new float*[sizeof(float *) * 2];
	chk_cuda(cudaMalloc((void**) &in_out[0], sizeof(float) * in._size));
	chk_cuda(cudaMalloc((void**) &in_out[1], sizeof(float) * in._size));
	chk_cuda(cudaMemcpy(in_out[0], in._elem, sizeof(float) * in._size, cudaMemcpyHostToDevice));
	square<<<in._size / NUM_PROCS + 1, NUM_PROCS>>>(in_out[0], in_out[1], in._size);
	Tensor out = in;
	chk_cuda(cudaMemcpy(out._elem, in_out[1], sizeof(float) * in._size, cudaMemcpyDeviceToHost));
	cudaFree(in_out[0]);
	cudaFree(in_out[1]);
    delete[] in_out;
	return out;
}

Tensor sum(const Tensor& in, const unsigned short axis) {
    // TODO: Add a feature that it can take None like axis = None like np.sum?

	assert(axis == 0 || axis == 1);
	Tensor out;
	if (in._shape.size() == 1 || in._shape.front() == 1) {
		if (axis == 0) {
			out = in;
		} else if (axis == 1) {
			out = Tensor({ 0.0f }, { 1, 1 });
			float sum = 0.0f;
			for (unsigned int i = 0; i < in._size; ++i) {
				sum += in[i];
			}
			out[0] = sum;
		}
	} else {
		if (axis == 0) {
			out = Tensor({ 0.0f }, { 1, in._shape.back() });
			for (unsigned int i = 0; i < in._shape.back(); ++i) {
				unsigned short idx = i;
				for (unsigned int j = 0; j < in._shape.front(); ++j) {
					out[i] += in[idx];
					idx += in._shape.back();
				}
			}
		} else if (axis == 1) {
			out = Tensor({ 0.0f }, { in._shape.front(), 1 });
			unsigned short idx = 0;
			for (unsigned int i = 0; i < in._shape.front(); ++i) {
				for (unsigned int j = 0; j < in._shape.back(); ++j) {
					out[i] += in[idx];
					++idx;
				}
			}
		}
	}
	return out;
}

Tensor variance(const Tensor& in) {
	return Tensor();
}