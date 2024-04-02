#pragma once

#include <vector>

class Ten;

Ten NormalDistribution(const std::vector<size_t> &shape, const float mean = 0.0f, const float stdDev = 0.05f);
Ten Shuffle(const Ten &ten, const size_t randomState);