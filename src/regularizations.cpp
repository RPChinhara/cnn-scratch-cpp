#include "regularizations.h"
#include "tensor.h"

#include <random>

// TODO: Not generating correctly.
void dropout(const f32 rate, const Tensor& in) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<f32> distribution(0.0f, 1.0f);

    for (size_t i = 0; i < in._size; ++i) {
        if (distribution(generator) < rate)
            in[i] = 0.0f;
        else
            in[i] *= 1.0f / (1.0f - rate);
    }
}

f32 l1(const f32 lambda, const Tensor& weight) {
    f32 sum = 0.0f;
    for (u32 i = 0; i < weight._size; ++i)
        sum += std::fabs(weight[i]);
    return lambda * sum;
}

f32 l2(const f32 lambda, const Tensor& weight) {
    f32 sum = 0.0f;
    for (u32 i = 0; i < weight._size; ++i)
        sum += std::powf(weight[i], 2.0f); 
    return lambda / 2.0f * sum;
}