#pragma once

#include "tensor.h"

struct IMDB
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

IMDB LoadIMDB();
Iris LoadIris();
MNIST LoadMNIST();