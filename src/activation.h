#pragma once

#include "device.h"

class Tensor;

Tensor Relu(const Tensor &tensor, Device device);
Tensor Softmax(const Tensor &tensor);