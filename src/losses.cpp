#include "losses.h"
#include "tensor.h"

float mean_squared_error(const tensor &y_true, const tensor &y_pred) {
    float sum = 0.0f;
    float n = static_cast<float>(y_true.shape.back());

    for (auto i = 0; i < y_true.size; ++i)
        sum += std::powf(y_true[i] - y_pred[i], 2.0f);

    return sum / n;
}