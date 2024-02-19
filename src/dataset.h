#pragma once

#include "tensor.h"

#include <string>
#include <vector>

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