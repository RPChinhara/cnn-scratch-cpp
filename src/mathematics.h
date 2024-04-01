#pragma once

#include "dev.h"

class Tensor;

Tensor Argmax(const Tensor &tensor);
Tensor Exp(const Tensor &tensor, Dev device);
Tensor Log(const Tensor &tensor, Dev device);
Tensor Max(const Tensor &tensor, const size_t axis);
Tensor Min(const Tensor &tensor);
Tensor Sum(const Tensor &tensor, const size_t axis);