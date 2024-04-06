#include "losses.h"
#include "arrs.h"
#include "dev.h"
#include "math.hpp"
#include "ten.h"

float categorical_cross_entropy(const ten &y_target, const ten &y_pred)
{
    float sum = 0.0f;
    constexpr float epsilon = 1e-15f;
    size_t num_samples = y_target.shape.front();
    ten y_pred_clipped = clip_by_value(y_pred, epsilon, 1.0f - epsilon);
    ten log = Log(y_pred_clipped, DEV_CPU);

    for (size_t i = 0; i < y_target.size; ++i)
        sum += y_target[i] * log[i];

    return -sum / num_samples;
}