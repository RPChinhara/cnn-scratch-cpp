#include "metric.h"
#include "mathematics.h"
#include "tensor.h"

float CategoricalAccuracy(const Tensor& y_true, const Tensor& y_pred)
{
    Tensor true_idx = Argmax(y_true);
    Tensor pred_idx = Argmax(y_pred);
    float equal = 0.0f;

    for (size_t i = 0; i < true_idx.size; ++i)
        if (true_idx[i] == pred_idx[i])
            ++equal;

    return equal / true_idx.size;
}