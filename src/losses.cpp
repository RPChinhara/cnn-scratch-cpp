#include "losses.h"
#include "arrays.h"
#include "mathematics.h"
#include "tensor.h"

float CategoricalCrossEntropy(const Tensor& y_true, const Tensor& y_pred)
{
    float sum = 0.0f;
    float epsilon = 1e-15f;
    unsigned int num_samples = y_true._shape.front();
    Tensor y_pred_clipped = ClipByValue(y_pred, epsilon, 1.0f - epsilon);

    for (unsigned int i = 0; i < y_true._size; ++i)
        sum += y_true[i] * Log(y_pred_clipped)[i];
    return -sum / num_samples;
}

float MeanSquaredError(const Tensor& y_true, const Tensor& y_pred)
{
    float sum = 0.0f;
    
    for (unsigned int i = 0; i < y_true._size; ++i)
        sum += std::powf(y_true[i] - y_pred[i], 2.0f);
    return sum / y_true._size;
}