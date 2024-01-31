#pragma once

class Tensor;

float CategoricalCrossEntropy(const Tensor &y_true, const Tensor &y_pred);