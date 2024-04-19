#pragma once

#include <vector>

class ten;

ten clip_by_value(const ten &t, float clip_val_min, float clip_val_max);
ten slice(const ten &t, const size_t begin, const size_t size);
ten zeros(const std::vector<size_t> &shape);