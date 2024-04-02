#pragma once

#include <vector>

class Ten;

class NN
{
  public:
    NN(const std::vector<size_t> &lyrs, float const lr);
    void train(const Ten &x_train, const Ten &y_train, const Ten &x_val, const Ten &y_val);
    void predict(const Ten &x_test, const Ten &y_test);

  private:
    std::pair<std::vector<Ten>, std::vector<Ten>> init_params();
    std::vector<Ten> forward_prop(const Ten &x, const std::vector<Ten> &w, const std::vector<Ten> &b);

    std::vector<size_t> lyrs;
    std::pair<std::vector<Ten>, std::vector<Ten>> w_b;
    std::pair<std::vector<Ten>, std::vector<Ten>> w_b_mom;
    std::vector<Ten> a;
    size_t batch_size = 10;
    size_t epochs = 200;
    float lr;
    float grad_clip_threshold = 8.0f;
    float mom = 0.1f;
    size_t patience = 4;
};