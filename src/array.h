#pragma once

#include <vector>

class Tensor;

Tensor ClipByValue(const Tensor &tensor, float clip_val_min, float clip_val_max);
Tensor Slice(const Tensor &tensor, const size_t begin, const size_t size);
Tensor Transpose(const Tensor &tensor);
Tensor Zeros(const std::vector<size_t> &shape);
Tensor ZerosLike(const std::vector<size_t> &shape);