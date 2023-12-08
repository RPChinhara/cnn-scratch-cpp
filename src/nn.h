#pragma once

#include "loss.h"
#include "metric.h"
#include "tensor.h"

#include <vector>

using TensorArray = std::vector<Tensor>;

class NN
{
public:
    NN(const std::vector<unsigned int>& layers, float learning_rate);
    void Train(const Tensor& x_train, const Tensor& y_train, const Tensor& x_val, const Tensor& y_val);
    void Predict(const Tensor& x_test, const Tensor& y_test);

private:
    std::vector<unsigned int> layers;
    std::pair<TensorArray, TensorArray> weights_biases;
    std::pair<TensorArray, TensorArray> weights_biases_momentum;
    unsigned short batch_size = 8;
    unsigned short epochs = 100;
    float learning_rate;
    float gradient_clip_threshold = 8.0f;
    float momentum = 0.1f;
    unsigned char patience = 7;
    
    TensorArray ForwardPropagation(const Tensor& input, const TensorArray& weights, const TensorArray& biases);
    std::pair<TensorArray, TensorArray> InitParameters();
};