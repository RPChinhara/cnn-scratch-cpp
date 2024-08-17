#pragma once

#include "tensor.h"

#include <string>
#include <vector>

struct en_es
{
    std::vector<std::wstring> x;
    std::vector<std::wstring> y;
};

struct imdb
{
    tensor x;
    tensor y;
};

struct iris
{
    tensor x;
    tensor y;
};

struct mnist
{
    tensor trainImages;
    tensor trainLabels;
    tensor testImages;
    tensor testLabels;
};

tensor load_aapl();
en_es load_en_es();
imdb load_imdb();
iris load_iris();
mnist load_mnist();