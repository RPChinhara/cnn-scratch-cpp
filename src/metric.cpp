#include "metric.h"
#include "mathematics.h"
#include "tensor.h"

float CategoricalAccuracy(const Tensor &yTrue, const Tensor &yPred)
{
    Tensor trueIdx = Argmax(yTrue);
    Tensor predIdx = Argmax(yPred);
    float equal = 0.0f;

    for (size_t i = 0; i < trueIdx.size; ++i)
        if (trueIdx[i] == predIdx[i])
            ++equal;

    return equal / trueIdx.size;
}