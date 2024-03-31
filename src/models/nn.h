#pragma once

#include <vector>

class Tensor;

class NN
{
  public:
    NN(const std::vector<size_t> &layers, float const lr);

    void Train(const Tensor &x_train, const Tensor &y_train, const Tensor &x_val, const Tensor &y_val);

    void Predict(const Tensor &x_test, const Tensor &y_test);

  private:
    std::pair<std::vector<Tensor>, std::vector<Tensor>> InitParameters();

    std::vector<Tensor> ForwardPropagation(const Tensor &input, const std::vector<Tensor> &weight,
                                           const std::vector<Tensor> &bias);

    std::vector<size_t> layers;
    size_t numForwardBackProps;
    std::pair<std::vector<Tensor>, std::vector<Tensor>> weights_biases;
    std::pair<std::vector<Tensor>, std::vector<Tensor>> weights_biases_momentum;
    std::vector<Tensor> a;
    size_t batchSize = 10;
    size_t epochs = 200;
    float lr;
    float gradientClipThreshold = 8.0f;
    float momentum = 0.1f;
    size_t patience = 4;
};