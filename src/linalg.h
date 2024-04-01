#pragma once

#include "dev.h"

class Tensor;

Tensor MatMul(const Tensor &tensor1, const Tensor &tensor2, Dev device);