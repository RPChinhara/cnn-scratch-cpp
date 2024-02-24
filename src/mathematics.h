#pragma once

#include "device.h"

class Tensor;

Tensor Argmax(const Tensor &tensor);
Tensor Exp(const Tensor &tensor, Device device);
Tensor Log(const Tensor &tensor, Device device);
Tensor Max(const Tensor &tensor, const size_t axis);
Tensor Min(const Tensor &tensor);
Tensor Sum(const Tensor &tensor, const size_t axis);