#pragma once

class Tensor;

float accuracy(const Tensor& y_true, const Tensor& y_pred);
float categorical_accuracy(const Tensor& y_true, const Tensor& y_pred);