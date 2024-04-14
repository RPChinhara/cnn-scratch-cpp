#pragma once

#include "ten.h"

struct mnist
{
    ten trainImages;
    ten trainLabels;
    ten testImages;
    ten testLabels;
};

mnist load_mnist();