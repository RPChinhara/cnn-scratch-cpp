#pragma once

#include "tensor.h"

struct TrainTest
{
    Tensor trainFeatures;
    Tensor trainTargets;
    Tensor testFeatures;
    Tensor testTargets;
};

Tensor MinMaxScaler(Tensor &dataset);
Tensor OneHot(const Tensor &tensor, const size_t depth);
TrainTest TrainTestSplit(const Tensor &x, const Tensor &y, const float test_size, const size_t random_state);