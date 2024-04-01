#pragma once

#include "ten.h"

struct MNIST
{
    Tensor trainImages;
    Tensor trainLabels;
    Tensor testImages;
    Tensor testLabels;
};

MNIST LoadMNIST();