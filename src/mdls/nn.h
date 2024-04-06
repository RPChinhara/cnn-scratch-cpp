#pragma once

#include "act.h"

#include <vector>

class ten;

class nn
{
  public:
    nn(const std::vector<size_t> &lyrs, float const lr);
    void train(const ten &x_train, const ten &y_train, const ten &x_val, const ten &y_val);
    void pred(const ten &x_test, const ten &y_test);

  private:
    std::pair<std::vector<ten>, std::vector<ten>> init_params();
    std::vector<ten> forward_prop(const ten &x, const std::vector<ten> &w, const std::vector<ten> &b);

    std::vector<size_t> lyrs;
    std::vector<Act> acts = {RELU, SOFTMAX};
    std::pair<std::vector<ten>, std::vector<ten>> w_b;
    std::pair<std::vector<ten>, std::vector<ten>> w_b_mom;
    std::vector<ten> a;
    size_t batch_size = 10;
    size_t epochs = 200;
    float lr;
    float grad_clip_threshold = 8.0f;
    float mom = 0.1f;
    size_t patience = 4;
};