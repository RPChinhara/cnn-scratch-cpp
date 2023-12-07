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
    void train(const Tensor& train_x, const Tensor& train_y, const Tensor& val_x, const Tensor& val_y);
    void predict(const Tensor& test_x, const Tensor& test_y);

private:
    std::vector<unsigned int> layers;
    std::pair<TensorArray, TensorArray> w_b;
    std::pair<TensorArray, TensorArray> w_b_m;
    
    unsigned short batch_size = 8;
    unsigned short epochs = 100;
    float learning_rate;

    float gradient_clip_threshold = 8.0f;
    float momentum = 0.1f;
    unsigned char patience = 12;
    
    TensorArray forward_propagation(const Tensor& input, const TensorArray& w, const TensorArray& b);
    std::pair<TensorArray, TensorArray> init_parameters();
};