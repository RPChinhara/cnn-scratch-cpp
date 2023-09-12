#pragma once

#include "types.h"

class Tensor;

Tensor categorical_crossentropy_prime(const Tensor& y_true, const Tensor& y_pred);
Tensor l1_prime(const f32 lambda, const Tensor& w);
Tensor l2_prime(const f32 lambda, const Tensor& w);
Tensor mean_squared_error_prime(const Tensor& y_true, const Tensor& y_pred);
Tensor relu_prime(const Tensor& in);
Tensor sigmoid_prime(const Tensor& in);