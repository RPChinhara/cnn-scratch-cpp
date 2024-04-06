#pragma once

#include "dev.h"

enum act_enum
{
    ACT_RELU,
    ACT_SIGMOID,
    ACT_SOFTMAX
};

class ten;

ten act(const ten &t, act_enum act, dev_type dev);