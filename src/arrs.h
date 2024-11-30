#pragma once

#include <vector>

class tensor;

tensor fill(const std::vector<size_t>& shape, float val);
tensor zeros(const std::vector<size_t>& shape);

tensor clip_by_value(const tensor& t, float clip_val_min, float clip_val_max);

tensor slice(const tensor& t, const size_t begin, const size_t size);
tensor vslice(const tensor& t, const size_t col);

tensor broadcast_to(const tensor& t, const std::vector<size_t>& shape);
tensor one_hot(const tensor& t, const size_t depth);
tensor pad(const tensor& t, size_t pad_top, size_t pad_bottom, size_t pad_left, size_t pad_right);
tensor vstack(const std::vector<tensor>& ts);

std::pair<tensor, tensor> split(const tensor& t, const float test_size);