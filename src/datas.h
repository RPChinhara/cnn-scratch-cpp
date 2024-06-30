#pragma once

#include "ten.h"

#include <string>
#include <vector>

struct en_es
{
    std::vector<std::wstring> x;
    std::vector<std::wstring> y;
};

struct imdb
{
    ten x;
    ten y;
};

struct iris
{
    ten x;
    ten y;
};

struct mnist
{
    ten trainImages;
    ten trainLabels;
    ten testImages;
    ten testLabels;
};

ten load_aapl();
en_es load_en_es();
imdb load_imdb();
iris load_iris();
mnist load_mnist();