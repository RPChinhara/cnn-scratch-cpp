#pragma once

class Tensor;

float categorical_crossentropy(const Tensor& y_true, const Tensor& y_pred);
float mean_squared_error(const Tensor& y_true, const Tensor& y_pred);