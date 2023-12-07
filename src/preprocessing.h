#pragma once

#include "tensor.h"

struct TrainTest
{
    Tensor x_first;
    Tensor x_second;
    Tensor y_first;
    Tensor y_second;
};

Tensor MinMaxScaler(Tensor& dataset);
Tensor OneHot(const Tensor& in, const unsigned short depth);
TrainTest TrainTestSplit(const Tensor& x, const Tensor& y, const float test_size, const unsigned int random_State);