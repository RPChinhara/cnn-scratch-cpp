#pragma once

class Tensor;

float CategoricalAccuracy(const Tensor& y_true, const Tensor& y_pred);