#pragma once

#include <vector>

class ten;

ten normal_dist(const std::vector<size_t> &shape, const float mean = 0.0f, const float std_dev = 0.05f);
ten uniform_dist(const std::vector<size_t> &shape, const float min_val = -0.05f, const float max_val = 0.05f);
ten shuffle(const ten &t, const size_t rd_state);