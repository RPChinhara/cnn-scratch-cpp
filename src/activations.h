#pragma once

#include "device.h"

class Tensor;

Tensor Relu(const Tensor &tensor, Dev device);
Tensor Softmax(const Tensor &tensor);