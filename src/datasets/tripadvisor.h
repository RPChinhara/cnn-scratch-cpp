#pragma once

#include "ten.h"

struct Tripadvisor
{
    Tensor features;
    Tensor targets;
};

Tripadvisor LoadTripadvisor();