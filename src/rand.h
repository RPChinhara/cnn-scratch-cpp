#pragma once

#include <vector>

class Ten;

Ten normal_dist(const std::vector<size_t> &shape, const float mean = 0.0f, const float stdDev = 0.05f);
Ten shuffle(const Ten &ten, const size_t randomState);