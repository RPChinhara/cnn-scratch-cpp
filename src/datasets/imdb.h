#pragma once

#include "ten.h"

struct IMDB
{
    Tensor features;
    Tensor targets;
};

IMDB LoadIMDB();