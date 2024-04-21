#pragma once

#include <vector>

class ten;

ten normal_dist(const std::vector<size_t> &shape, const float mean = 0.0f, const float stdDev = 0.05f);
ten shuffle(const ten &t, const size_t randomState);