#include "losses.h"
#include "arrs.h"
#include "dev.h"
#include "math.hpp"
#include "ten.h"

float categorical_cross_entropy(const ten &y_true, const ten &y_pred)
{
    float sum = 0.0f;
    constexpr float epsilon = 1e-15f;
    size_t num_samples = y_true.shape.front();
    ten y_pred_clipped = clip_by_value(y_pred, epsilon, 1.0f - epsilon);
    ten log = Log(y_pred_clipped, CPU);

    for (size_t i = 0; i < y_true.size; ++i)
        sum += y_true[i] * log[i];

    return -sum / num_samples;
}