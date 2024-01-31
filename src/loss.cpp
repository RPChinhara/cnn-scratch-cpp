#include "loss.h"
#include "array.h"
#include "device.h"
#include "mathematics.h"
#include "tensor.h"

float CategoricalCrossEntropy(const Tensor &y_true, const Tensor &y_pred)
{
    float sum = 0.0f;
    float epsilon = 1e-15f;
    size_t num_samples = y_true.shape.front();
    Tensor y_pred_clipped = ClipByValue(y_pred, epsilon, 1.0f - epsilon);
    Tensor log = Log(y_pred_clipped, Device::CPU);

    for (size_t i = 0; i < y_true.size; ++i)
        sum += y_true[i] * log[i];

    return -sum / num_samples;
}