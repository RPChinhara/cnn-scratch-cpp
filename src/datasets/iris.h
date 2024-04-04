#pragma once

#include "ten.h"

struct iris
{
    Ten features;
    Ten targets;
};

iris load_iris();