#pragma once

#include "tensor.h"

struct IMDb
{
    Tensor features;
    Tensor targets;
};

struct Iris
{
    Tensor features;
    Tensor targets;
};

struct MNIST
{
    Tensor trainImages;
    Tensor trainLabels;
    Tensor testImages;
    Tensor testLabels;
};

IMDb LoadIMDb();
Iris LoadIris();
MNIST LoadMNIST();