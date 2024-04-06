#include "metrics.h"
#include "math.hpp"
#include "ten.h"

float categorical_accuracy(const ten &y_target, const ten &y_pred)
{
    ten target_idx = Argmax(y_target);
    ten pred_idx = Argmax(y_pred);
    float equal = 0.0f;

    for (size_t i = 0; i < target_idx.size; ++i)
        if (target_idx[i] == pred_idx[i])
            ++equal;

    return equal / target_idx.size;
}