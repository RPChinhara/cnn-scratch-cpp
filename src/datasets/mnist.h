#pragma once

#include "tensor.h"

struct MNIST
{
    Tensor trainImages;
    Tensor trainLabels;
    Tensor testImages;
    Tensor testLabels;
};

MNIST LoadMNIST();