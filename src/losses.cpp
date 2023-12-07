#include "losses.h"
#include "arrays.h"
#include "mathematics.h"
#include "tensor.h"

float CategoricalCrossEntropy(const Tensor& yTrue, const Tensor& yPred)
{
    float sum = 0.0f;
    float epsilon = 1e-15f;
    unsigned int numSamples = yTrue._shape.front();
    Tensor yPredClipped = ClipByValue(yPred, epsilon, 1.0f - epsilon);

    for (unsigned int i = 0; i < yTrue._size; ++i)
        sum += yTrue[i] * Log(yPredClipped)[i];
    return -sum / numSamples;
}

float MeanSquaredError(const Tensor& yTrue, const Tensor& yPred)
{
    float sum = 0.0f;
    
    for (unsigned int i = 0; i < yTrue._size; ++i)
        sum += std::powf(yTrue[i] - yPred[i], 2.0f);
    return sum / yTrue._size;
}