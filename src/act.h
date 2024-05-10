#pragma once

#include "dev.h"

class ten;

enum act_enum
{
    RELU,
    SIGMOID,
    SOFTMAX
};

ten act(const ten &z, act_enum act, dev_type dev);