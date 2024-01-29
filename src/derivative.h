#pragma once

class Tensor;

Tensor CategoricalCrossEntropyDerivative(const Tensor& y_true, const Tensor& y_pred);
Tensor ReluDerivative(const Tensor& in);