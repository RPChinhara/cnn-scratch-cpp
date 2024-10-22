#pragma once

#include <vector>

class tensor;

tensor clip_by_value(const tensor &t, float clip_val_min, float clip_val_max);
tensor slice(const tensor &t, const size_t begin, const size_t size);
std::pair<tensor, tensor> split(const tensor &t, const float test_size);
tensor stack(const std::vector<tensor> &ts);
tensor zeros(const std::vector<size_t> &shape);