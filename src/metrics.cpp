#include "metrics.h"
#include "mathematics.h"
#include "ten.h"

float CategoricalAccuracy(const Tensor &y_target, const Tensor &y_pred)
{
    Tensor target_idx = Argmax(y_target);
    Tensor pred_idx = Argmax(y_pred);
    float equal = 0.0f;

    for (size_t i = 0; i < target_idx.size; ++i)
        if (target_idx[i] == pred_idx[i])
            ++equal;

    return equal / target_idx.size;
}