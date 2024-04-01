#pragma once

#include "device.h"

enum Act
{
    RELU,
    SIGMOID,
    SOFTMAX
};

class Tensor;

Tensor Relu(const Tensor &tensor, Dev device);
Tensor Softmax(const Tensor &tensor);