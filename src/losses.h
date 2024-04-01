#pragma once

class Tensor;

float CategoricalCrossEntropy(const Tensor &y_target, const Tensor &y_pred);