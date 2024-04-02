#pragma once

#include "ten.h"

struct MNIST
{
    Ten trainImages;
    Ten trainLabels;
    Ten testImages;
    Ten testLabels;
};

MNIST LoadMNIST();