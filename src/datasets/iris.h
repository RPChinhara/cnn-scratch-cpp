#pragma once

#include "ten.h"

struct Iris
{
    Tensor features;
    Tensor targets;
};

Iris LoadIris();