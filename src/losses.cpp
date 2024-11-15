#include "losses.h"
#include "arrs.h"
#include "math.hpp"
#include "tensor.h"

float categorical_cross_entropy(const tensor& y_true, const tensor& y_pred) {
    float sum = 0.0f;
    constexpr float epsilon = 1e-15f;
    size_t num_samples = y_true.shape.front();
    tensor y_pred_clipped = clip_by_value(y_pred, epsilon, 1.0f - epsilon);
    tensor y_pred_logged = log(y_pred_clipped);

    for (auto i = 0; i < y_true.size; ++i)
        sum += y_true[i] * y_pred_logged[i];

    return -sum / num_samples;
}

float mean_squared_error(const tensor& y_true, const tensor& y_pred) {
    float sum = 0.0f;
    float n = static_cast<float>(y_true.shape.back());

    for (auto i = 0; i < y_true.size; ++i)
        sum += std::powf(y_true[i] - y_pred[i], 2.0f);

    return sum / n;
}