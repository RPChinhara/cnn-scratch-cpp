#pragma once

#include <vector>

class Tensor;

Tensor ClipByValue(const Tensor& in, float clipValMin, float clipValMax);
Tensor Ones(const std::vector<unsigned int>& shape);
Tensor Slice(const Tensor& in, const unsigned int begin, const unsigned int size);
Tensor Zeros(const std::vector<unsigned int>& shape);