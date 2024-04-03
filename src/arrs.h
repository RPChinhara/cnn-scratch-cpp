#pragma once

#include <vector>

class Ten;

Ten clip_by_value(const Ten &ten, float clipValMin, float clipValMax);
Ten slice(const Ten &ten, const size_t begin, const size_t size);
Ten Transpose(const Ten &ten);
Ten Zeros(const std::vector<size_t> &shape);
Ten ZerosLike(const std::vector<size_t> &shape);