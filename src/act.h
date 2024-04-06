#pragma once

#include "dev.h"

enum Act
{
    RELU,
    SIGMOID,
    SOFTMAX
};

class ten;

ten act(const ten &t, Act act, dev_type dev);