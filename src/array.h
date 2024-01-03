#pragma once

#include <vector>

class Tensor;

Tensor ClipByValue(const Tensor& in, float clip_val_min, float clip_val_max);
Tensor Slice(const Tensor& in, const size_t begin, const size_t size);
Tensor Zeros(const std::vector<size_t>& shape);