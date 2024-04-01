#include "losses.h"
#include "arrs.h"
#include "dev.h"
#include "mathematics.h"
#include "ten.h"

float CategoricalCrossEntropy(const Tensor &y_target, const Tensor &y_pred)
{
    float sum = 0.0f;
    constexpr float epsilon = 1e-15f;
    size_t num_samples = y_target.shape.front();
    Tensor y_pred_clipped = ClipByValue(y_pred, epsilon, 1.0f - epsilon);
    Tensor log = Log(y_pred_clipped, Dev::CPU);

    for (size_t i = 0; i < y_target.size; ++i)
        sum += y_target[i] * log[i];

    return -sum / num_samples;
}