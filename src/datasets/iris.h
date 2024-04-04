#pragma once

#include "ten.h"

struct Iris
{
    Ten features;
    Ten targets;
};

Iris load_iris();