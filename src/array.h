#pragma once

#include <vector>

class Tensor;

Tensor ClipByValue(const Tensor& in, float clip_val_min, float clip_val_max);
Tensor Ones(const std::vector<size_t>& shape);
Tensor Slice(const Tensor& in, const unsigned int begin, const unsigned int size);
Tensor Zeros(const std::vector<size_t>& shape);