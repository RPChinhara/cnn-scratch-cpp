#pragma once

#include "types.h"

class Tensor;

f32 accuracy(const Tensor& y_true, const Tensor& y_pred);
f32 categorical_accuracy(const Tensor& y_true, const Tensor& y_pred);