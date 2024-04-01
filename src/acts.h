#pragma once

#include "dev.h"

enum Act
{
    RELU,
    SIGMOID,
    SOFTMAX
};

class Tensor;

Tensor Relu(const Tensor &tensor, Dev dev);
Tensor Softmax(const Tensor &tensor);