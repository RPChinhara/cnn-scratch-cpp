#pragma once

#include "dev.h"

enum Act
{
    RELU,
    SIGMOID,
    SOFTMAX
};

class Ten;

Ten act(const Ten &ten, Act act, Dev dev);