#pragma once

class Tensor;

Tensor CategoricalCrossEntropyDerivative(const Tensor &yTrue, const Tensor &yPred);
Tensor ReluDerivative(const Tensor &tensor);