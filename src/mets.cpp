#include "mets.h"
#include "math.hpp"
#include "ten.h"

float categorical_acc(const ten &y_true, const ten &y_pred)
{
    ten idx_true = argmax(y_true);
    ten pred_idx = argmax(y_pred);
    float equal = 0.0f;

    for (auto i = 0; i < idx_true.size; ++i)
        if (idx_true[i] == pred_idx[i])
            ++equal;

    return equal / idx_true.size;
}