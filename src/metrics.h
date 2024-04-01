#pragma once

class Tensor;

float CategoricalAccuracy(const Tensor &y_target, const Tensor &y_pred);