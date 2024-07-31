#pragma once

class ten;

float categorical_cross_entropy(const ten &y_true, const ten &y_pred);
float mse(const ten &y_true, const ten &y_pred);