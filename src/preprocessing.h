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
TrainTest TrainTestSplit(const Tensor &x, const Tensor &y, const float testSize, const size_t randomState);