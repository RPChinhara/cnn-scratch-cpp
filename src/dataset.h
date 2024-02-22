#pragma once

#include "tensor.h"

struct Iris
{
    Tensor features;
    Tensor target;
};

struct MNIST
{
    Tensor trainImages;
    Tensor trainLabels;
    Tensor testImages;
    Tensor testLabels;
};

Iris LoadIris();
MNIST LoadMNIST();