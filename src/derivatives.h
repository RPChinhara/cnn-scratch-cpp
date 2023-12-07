#pragma once

class Tensor;

Tensor PrimeCategoricalCrossEntropy(const Tensor& yTrue, const Tensor& yPred);
Tensor PrimeMeanSquaredError(const Tensor& yTrue, const Tensor& yPred);
Tensor PrimeRelu(const Tensor& in);
Tensor PrimeSigmoid(const Tensor& in);