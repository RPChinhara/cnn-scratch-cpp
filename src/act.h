#pragma once

#include "dev.h"

class ten;

enum act_type
{
    RELU,
    SIGMOID,
    SOFTMAX,
    TANH
};

ten act(const ten &z, act_type act, dev_type dev);