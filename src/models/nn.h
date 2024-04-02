#pragma once

#include <vector>

class Tensor;

class NN
{
  public:
    NN(const std::vector<size_t> &lyrs, float const lr);
    void train(const Tensor &x_train, const Tensor &y_train, const Tensor &x_val, const Tensor &y_val);
    void predict(const Tensor &x_test, const Tensor &y_test);

  private:
    std::pair<std::vector<Tensor>, std::vector<Tensor>> init_parameters();
    std::vector<Tensor> forward_prop(const Tensor &x, const std::vector<Tensor> &w, const std::vector<Tensor> &b);

    std::vector<size_t> lyrs;
    std::pair<std::vector<Tensor>, std::vector<Tensor>> w_b;
    std::pair<std::vector<Tensor>, std::vector<Tensor>> w_b_mom;
    std::vector<Tensor> a;
    size_t batch_size = 10;
    size_t epochs = 200;
    float lr;
    float grad_clip_threshold = 8.0f;
    float mom = 0.1f;
    size_t patience = 4;
};