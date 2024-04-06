#pragma once

#include "ten.h"

struct IMDB
{
    ten features;
    ten targets;
};

IMDB LoadIMDB();