#pragma once

#include "types.h"

class Tensor;

f32 categorical_crossentropy(const Tensor& y_true, const Tensor& y_pred);
f32 mean_squared_error(const Tensor& y_true, const Tensor& y_pred);