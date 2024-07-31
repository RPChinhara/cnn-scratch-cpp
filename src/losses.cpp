#include "losses.h"
#include "arrs.h"
#include "dev.h"
#include "math.hpp"
#include "ten.h"

float mse(const ten &y_true, const ten &y_pred) {
    float sum = 0.0f;

    for (auto i = 0; i < y_true.size; ++i)
        sum += std::powf(y_true[i] - y_pred[i], 2.0f);

    return sum / y_true.size;
}