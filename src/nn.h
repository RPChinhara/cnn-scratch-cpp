#pragma once

#include "losses.h"
#include "metrics.h"
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
    std::pair<TensorArray, TensorArray> weight_bias;
    std::pair<TensorArray, TensorArray> weight_bias_momentum;
    unsigned short batch_size = 8;
    unsigned short epochs = 100;
    float learning_rate;
    float gradient_clip_threshold = 8.0f;
    float momentum = 0.1f;
    unsigned char patience = 12;
    
    TensorArray ForwardPropagation(const Tensor& input, const TensorArray& weight, const TensorArray& bias);
    std::pair<TensorArray, TensorArray> InitParameters();
};