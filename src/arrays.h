#pragma once

#include <vector>

class Tensor;

Tensor clip_by_value(const Tensor& in, float clip_val_min, float clip_val_max);
Tensor ones(const std::vector<unsigned int>& shape);
Tensor slice(const Tensor& in, const unsigned int begin, const unsigned int size);
Tensor zeros(const std::vector<unsigned int>& shape);