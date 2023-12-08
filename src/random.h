#pragma once

#include <vector>

class Tensor;

Tensor NormalDistribution(const std::vector<size_t>& shape, const float mean = 0.0f, const float stddev = 0.05f);
Tensor Shuffle(const Tensor& in, const unsigned int random_state);
Tensor UniformDistribution(const std::vector<size_t>& shape, const float min_val = -0.05f, const float max_val = 0.05f);