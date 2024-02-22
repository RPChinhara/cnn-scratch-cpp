#pragma once

#include <vector>

class Tensor;

Tensor NormalDistribution(const std::vector<size_t> &shape, const float mean = 0.0f, const float stddev = 0.05f);
Tensor Shuffle(const Tensor &tensor, const size_t random_state);