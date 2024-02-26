#pragma once

#include "tensor.h"

struct TrainTest
{
    Tensor x_first;  // featuresFirst
    Tensor x_second; // featuresSecond
    Tensor y_first;  // targetsFirst
    Tensor y_second; // targetsSecond
};

Tensor MinMaxScaler(Tensor &dataset);
Tensor OneHot(const Tensor &tensor, const size_t depth);
TrainTest TrainTestSplit(const Tensor &x, const Tensor &y, const float test_size, const size_t random_state);