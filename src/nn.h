#pragma once

#include "tensor.h"

#include <vector>

class NN
{
public:
    NN(const std::vector<size_t>& layers, float learning_rate);
    void Train(const Tensor& x_train, const Tensor& y_train, const Tensor& x_val, const Tensor& y_val);
    void Predict(const Tensor& x_test, const Tensor& y_test);

private:
    std::pair<std::vector<Tensor>, std::vector<Tensor>> InitParameters();
    std::vector<Tensor> ForwardPropagation(const Tensor& input, const std::vector<Tensor>& weights, const std::vector<Tensor>& biases);
    Tensor Relu(const Tensor& in);
    Tensor Softmax(const Tensor& in);
    Tensor PrimeRelu(const Tensor& in);
    Tensor PrimeCategoricalCrossEntropy(const Tensor& y_true, const Tensor& y_pred);
    float CategoricalCrossEntropy(const Tensor& y_true, const Tensor& y_pred);

    std::vector<size_t> layers;
    std::pair<std::vector<Tensor>, std::vector<Tensor>> weights_biases;
    std::pair<std::vector<Tensor>, std::vector<Tensor>> weights_biases_momentum;
    size_t batch_size = 8;
    size_t epochs = 200;
    float learning_rate;
    float gradient_clip_threshold = 8.0f;
    float momentum = 0.1f;
    size_t patience = 4;
};

// activation, derivative, loss, metric