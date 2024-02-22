#pragma once

#include "device.h"

class Tensor;

Tensor MatMul(const Tensor &tensor1, const Tensor &tensor2, Device device);