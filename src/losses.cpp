#include "losses.h"
#include "arrs.h"
#include "dev.h"
#include "math.hpp"
#include "ten.h"

float CategoricalCrossEntropy(const Ten &y_target, const Ten &y_pred)
{
    float sum = 0.0f;
    constexpr float epsilon = 1e-15f;
    size_t num_samples = y_target.shape.front();
    Ten y_pred_clipped = ClipByValue(y_pred, epsilon, 1.0f - epsilon);
    Ten log = Log(y_pred_clipped, Dev::CPU);

    for (size_t i = 0; i < y_target.size; ++i)
        sum += y_target[i] * log[i];

    return -sum / num_samples;
}