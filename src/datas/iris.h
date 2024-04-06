#pragma once

#include "ten.h"

struct iris
{
    ten features;
    ten targets;
};

iris load_iris();