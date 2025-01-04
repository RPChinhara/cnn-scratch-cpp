#pragma once

#include <vector>

class tensor;

tensor fill(const std::vector<size_t>& shape, float val);
tensor zeros(const std::vector<size_t>& shape);

tensor clip_by_value(const tensor& t, float clip_val_min, float clip_val_max);

tensor slice(const tensor& t, const size_t begin, const size_t size); // TODO: Change to slice_2d()?
tensor slice_3d(const tensor& t, const size_t begin, const size_t size); // TODO: Change to slice_3d_3d()?
tensor slice_4d(const tensor& t, const size_t begin, const size_t size); // TODO: Change to slice_4d_4d()?
tensor slice_test(const tensor& t, const std::vector<size_t>& begin, const std::vector<size_t>& size);
tensor vslice(const tensor& t, const size_t col);

tensor broadcast_to(const tensor& t, const std::vector<size_t>& shape);
tensor one_hot(const tensor& t, const size_t depth);
tensor pad(const tensor& t, size_t pad_top, size_t pad_bottom, size_t pad_left, size_t pad_right);
tensor vstack(const std::vector<tensor>& ts);

std::pair<tensor, tensor> split(const tensor& t, const float test_size);