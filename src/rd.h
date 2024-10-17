#pragma once

#include <vector>

class tensor;

tensor glorot_uniform(const size_t num_out, const size_t num_in);
tensor normal_dist(const std::vector<size_t> &shape, const float mean = 0.0f, const float std_dev = 0.05f);
tensor shuffle(const tensor &t, const size_t rd_state);
tensor uniform_dist(const std::vector<size_t> &shape, const float min_val = -0.05f, const float max_val = 0.05f);