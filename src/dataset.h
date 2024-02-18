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
std::vector<std::vector<uint8_t>> readMNISTImages(const std::string &filePath);