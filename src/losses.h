#pragma once

class Tensor;

float CategoricalCrossEntropy(const Tensor& y_true, const Tensor& y_pred);
float MeanSquaredError(const Tensor& y_true, const Tensor& y_pred);