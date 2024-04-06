#pragma once

#include "ten.h"

struct MNIST
{
    ten trainImages;
    ten trainLabels;
    ten testImages;
    ten testLabels;
};

MNIST LoadMNIST();