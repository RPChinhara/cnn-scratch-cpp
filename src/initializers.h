#pragma once

#include "types.h"

#include <vector>

class Tensor;

// TODO: Are ones and zeros really needed?
Tensor glorot_normal_distribution(const std::vector<u32>& shape);
Tensor glorot_uniform_distribution(const std::vector<u32>& shape);
Tensor he_normal_distribution(const std::vector<u32>& shape);
Tensor he_uniform_distribution(const std::vector<u32>& shape);
Tensor lecun_normal_distribution(const std::vector<u32>& shape);
Tensor lecun_uniform_distribution(const std::vector<u32>& shape);
Tensor normal_distribution(const std::vector<u32>& shape, const f32 mean = 0.0f, const f32 stddev = 0.05f);
Tensor ones(const std::vector<u32>& shape);
Tensor uniform_distribution(const std::vector<u32>& shape, const f32 min_val = -0.05f, const f32 max_val = 0.05f);
Tensor zeros(const std::vector<u32>& shape);