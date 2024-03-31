#pragma once

class Tensor;

Tensor dcce_da_da_dz(const Tensor &yTrue, const Tensor &yPred);
Tensor drelu_dz(const Tensor &tensor);