#pragma once

#include <vector>

class Ten;

Ten clip_by_value(const Ten &t, float clipValMin, float clipValMax);
Ten slice(const Ten &t, const size_t begin, const size_t size);
Ten zeros(const std::vector<size_t> &shape);