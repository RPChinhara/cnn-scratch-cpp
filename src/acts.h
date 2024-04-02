#pragma once

#include "dev.h"

enum Act
{
    RELU,
    SIGMOID,
    SOFTMAX
};

class Ten;

Ten Relu(const Ten &ten, Dev dev);
Ten Softmax(const Ten &ten);