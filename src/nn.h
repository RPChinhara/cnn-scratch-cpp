#pragma once

#include "tensor.h"
#include "losses.h"
#include "metrics.h"

#include <vector>

// Hyperparameters
// TODO: Stop using those, and create functions for each of them? like momentum(), learning_rate_scheduler() or just use booling ?
#define EARLY_STOPPING_ENABLED          1
#define GRADIENT_CLIPPING_ENABLED       1
#define LEARNING_RATE_SCHEDULER_ENABLED 1
#define L1_REGULARIZATION_ENABLED       0
#define L2_REGULARIZATION_ENABLED       0
#define L1L2_REGULARIZATION_ENABLED     1
#define MOMENTUM_ENABLED                1

using TensorArray = std::vector<Tensor>;

class NN {
public:
    NN(const std::vector<unsigned int>& layers, float learning_rate);
    void train(const Tensor& train_x, const Tensor& train_y, const Tensor& val_x, const Tensor& val_y);
    void predict(const Tensor& test_x, const Tensor& test_y);

private:
    float (*ACCURACY)(const Tensor& y_true, const Tensor& y_pred) = &categorical_accuracy;
    float (*LOSS)(const Tensor& y_true, const Tensor& y_pred)     = &categorical_crossentropy;

    std::vector<unsigned int> layers;
    std::pair<TensorArray, TensorArray> w_b;
    std::pair<TensorArray, TensorArray> w_b_m;
    
    unsigned short batch_size = 8;
    unsigned short epochs     = 100;
    float learning_rate;

    float gradient_clip_threshold = 8.0f;
    float momentum                = 0.1f;
    unsigned char patience        = 12;
    
    float l1_lambda = 0.01f;
    float l2_lambda = 0.01f;
    
    TensorArray forward_propagation(const Tensor& input, const TensorArray& w, const TensorArray& b);
    std::pair<TensorArray, TensorArray> init_parameters();
    void log_metrics(const std::string& data, const Tensor& y_true, const Tensor& y_pred, const TensorArray *w = nullptr);
};