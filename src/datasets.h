#pragma once

#include "tensor.h"

struct Iris {
    Tensor features;
    Tensor target;
};

Iris load_iris();