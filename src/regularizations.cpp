#include "regularizations.h"
#include "tensor.h"

#include <random>

void dropout(const float rate, const Tensor& in) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    for (size_t i = 0; i < in._size; ++i) {
        if (distribution(generator) < rate)
            in[i] = 0.0f;
        else
            in[i] *= 1.0f / (1.0f - rate);
    }
}

float l1(const float lambda, const Tensor& weight) {
    float sum = 0.0f;

    for (unsigned int i = 0; i < weight._size; ++i)
        sum += std::fabs(weight[i]);

    return lambda * sum;
}

float l2(const float lambda, const Tensor& weight) {
    float sum = 0.0f;

    for (unsigned int i = 0; i < weight._size; ++i)
        sum += std::powf(weight[i], 2.0f);
    
    return lambda / 2.0f * sum;
}