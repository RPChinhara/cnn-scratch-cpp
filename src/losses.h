#pragma once

class tensor;

float categorical_cross_entropy(const tensor& y_true, const tensor& y_pred);
float sparse_categorical_cross_entropy(const tensor& y_true, const tensor& y_pred);
float mean_squared_error(const tensor& y_true, const tensor& y_pred);