#pragma once

#include <vector>

class Tensor;

Tensor glorot_normal_distribution(const std::vector<unsigned int>& shape);
Tensor glorot_uniform_distribution(const std::vector<unsigned int>& shape);
Tensor he_normal_distribution(const std::vector<unsigned int>& shape);
Tensor he_uniform_distribution(const std::vector<unsigned int>& shape);
Tensor lecun_normal_distribution(const std::vector<unsigned int>& shape);
Tensor lecun_uniform_distribution(const std::vector<unsigned int>& shape);
Tensor normal_distribution(const std::vector<unsigned int>& shape, const float mean = 0.0f, const float stddev = 0.05f);
Tensor ones(const std::vector<unsigned int>& shape);
Tensor uniform_distribution(const std::vector<unsigned int>& shape, const float min_val = -0.05f, const float max_val = 0.05f);
Tensor zeros(const std::vector<unsigned int>& shape);