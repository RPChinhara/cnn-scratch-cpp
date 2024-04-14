#pragma once

#include "dev.h"

enum act_enum
{
    RELU,
    SIGMOID,
    SOFTMAX
};

class ten;

ten act(const ten &t, act_enum act, dev_type dev);