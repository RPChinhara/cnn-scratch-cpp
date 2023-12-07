#pragma once

class Tensor;

float Accuracy(const Tensor& y_true, const Tensor& y_pred);
float CategoricalAccuracy(const Tensor& y_true, const Tensor& y_pred);