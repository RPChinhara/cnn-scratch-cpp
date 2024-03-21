#pragma once

#include "tensor.h"

struct Iris
{
    Tensor features;
    Tensor targets;
};

Iris LoadIris();