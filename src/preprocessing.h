#pragma once

#include "tensor.h"

struct TrainTest {
    Tensor xFirst;
    Tensor xSecond;
    Tensor yFirst;
    Tensor ySecond;
};

Tensor MinMaxScaler(Tensor& dataset);
Tensor OneHot(const Tensor& in, const unsigned short depth);
TrainTest TrainTestSplit(const Tensor& x, const Tensor& y, const float testSize, const unsigned int randomState);