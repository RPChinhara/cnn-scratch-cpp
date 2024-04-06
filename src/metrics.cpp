#include "metrics.h"
#include "math.hpp"
#include "ten.h"

float categorical_accuracy(const ten &y_true, const ten &y_pred)
{
    ten idx_true = Argmax(y_true);
    ten pred_idx = Argmax(y_pred);
    float equal = 0.0f;

    for (size_t i = 0; i < idx_true.size; ++i)
        if (idx_true[i] == pred_idx[i])
            ++equal;

    return equal / idx_true.size;
}