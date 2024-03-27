#pragma once

#include "tensor.h"

struct Tripadvisor
{
    Tensor features;
    Tensor targets;
};

Tripadvisor LoadTripadvisor();