#pragma once

class Tensor;

Tensor PrimeCategoricalCrossEntropy(const Tensor& y_true, const Tensor& y_pred);
Tensor PrimeMeanSquaredError(const Tensor& y_true, const Tensor& y_pred);
Tensor PrimeRelu(const Tensor& in);
Tensor PrimeSigmoid(const Tensor& in);