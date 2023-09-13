#pragma once

#include "tensor.h"

struct TrainTest {
    Tensor x_first;
    Tensor x_second;
    Tensor y_first;
    Tensor y_second;
};

Tensor min_max_scaler(Tensor& dataset);
Tensor one_hot(const Tensor& in, const unsigned short depth);
TrainTest train_test_split(const Tensor x, const Tensor y, const float test_size, const unsigned int random_state);