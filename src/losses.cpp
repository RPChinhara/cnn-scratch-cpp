#include "losses.h"
#include "arrays.h"
#include "device.h"
#include "mathematics.h"
#include "tensor.h"

float CategoricalCrossEntropy(const Tensor &yTrue, const Tensor &yPred)
{
    float sum = 0.0f;
    constexpr float epsilon = 1e-15f;
    size_t numSamples = yTrue.shape.front();
    Tensor yPredClipped = ClipByValue(yPred, epsilon, 1.0f - epsilon);
    Tensor log = Log(yPredClipped, Dev::CPU);

    for (size_t i = 0; i < yTrue.size; ++i)
        sum += yTrue[i] * log[i];

    return -sum / numSamples;
}