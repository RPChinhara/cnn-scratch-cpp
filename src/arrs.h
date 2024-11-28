#pragma once

#include <vector>

class tensor;

tensor clip_by_value(const tensor& t, float clip_val_min, float clip_val_max);
tensor one_hot(const tensor& t, const size_t depth);
tensor pad(const tensor& t, size_t pad_top, size_t pad_bottom, size_t pad_left, size_t pad_right);
tensor slice(const tensor& t, const size_t begin, const size_t size);
std::pair<tensor, tensor> split(const tensor& t, const float test_size);
tensor vslice(const tensor& t, const size_t col);
tensor vstack(const std::vector<tensor>& ts);
tensor zeros(const std::vector<size_t>& shape);