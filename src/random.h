#pragma once

#include <vector>

class Tensor;

Tensor normal_distribution(const std::vector<unsigned int>& shape, const float mean = 0.0f, const float stddev = 0.05f);
Tensor shuffle(const Tensor& in, const unsigned int random_state);
Tensor uniform_distribution(const std::vector<unsigned int>& shape, const float min_val = -0.05f, const float max_val = 0.05f);