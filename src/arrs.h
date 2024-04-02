#pragma once

#include <vector>

class Ten;

Ten ClipByValue(const Ten &tensor, float clipValMin, float clipValMax);
Ten Slice(const Ten &tensor, const size_t begin, const size_t size);
Ten Transpose(const Ten &tensor);
Ten Zeros(const std::vector<size_t> &shape);
Ten ZerosLike(const std::vector<size_t> &shape);