#pragma once

#include "ten.h"

struct IMDB
{
    Ten features;
    Ten targets;
};

IMDB LoadIMDB();