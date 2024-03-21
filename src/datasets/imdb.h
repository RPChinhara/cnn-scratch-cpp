#pragma once

#include "tensor.h"

struct IMDB
{
    Tensor features;
    Tensor targets;
};

IMDB LoadIMDB();