#include "losses.h"
#include "arrs.h"
#include "tensor.h"

float categorical_cross_entropy(const tensor& y_true, const tensor& y_pred) {
    float sum = 0.0f;
    float num_elm = static_cast<float>(y_true.shape.front());
    tensor y_pred_clipped = clip_by_value(y_pred, 1e-12f, 1.0f);

    for (auto i = 0; i < y_true.size; ++i)
        sum += y_true[i] * logf(y_pred_clipped[i]);

    return -sum / num_elm;
}

float mean_squared_error(const tensor& y_true, const tensor& y_pred) {
    float sum = 0.0f;
    float num_elm = static_cast<float>(y_true.shape.front());

    for (auto i = 0; i < y_true.size; ++i)
        sum += std::powf(y_true[i] - y_pred[i], 2.0f);

    return sum / num_elm;
}