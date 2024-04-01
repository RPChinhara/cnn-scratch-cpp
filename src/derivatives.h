#pragma once

class Tensor;

Tensor dcce_dsoftmax_dsoftmax_dz(const Tensor &y_target, const Tensor &y_pred);
Tensor dmse_da_da_dz(const Tensor &y_target, const Tensor &y_pred);
Tensor drelu_dz(const Tensor &tensor);