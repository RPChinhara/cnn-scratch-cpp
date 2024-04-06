#pragma once

#include "dev.h"

enum Act
{
    RELU,
    SIGMOID,
    SOFTMAX
};

class Ten;

Ten act(const Ten &t, Act act, Dev dev);