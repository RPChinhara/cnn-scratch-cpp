#pragma once

class Tensor;

float CategoricalCrossEntropy(const Tensor &yTrue, const Tensor &yPred);