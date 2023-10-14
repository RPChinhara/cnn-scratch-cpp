#pragma once

#include "tensor.h"
#include "losses.h"
#include "metrics.h"

#include <vector>

// Hyperparameters
// TODO: Stop using those, and create functions for each of them? like momentum(), learning_rate_scheduler()?
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
    NN(const std::vector<unsigned int>& layers);
    std::pair<TensorArray, TensorArray> train(const Tensor& train_x, const Tensor& train_y, const Tensor& val_x, const Tensor& val_y);
    void predict(const Tensor& test_x, const Tensor& test_y, const TensorArray& w, const TensorArray& b);

private:
    float (*ACCURACY)(const Tensor& y_true, const Tensor& y_pred) = &categorical_accuracy;
    float (*LOSS)(const Tensor& y_true, const Tensor& y_pred)     = &categorical_crossentropy;
    unsigned short batch_size = 8;
    unsigned short epochs     = 100;
    float learning_rate       = 0.01f;
    std::vector<unsigned int> layers;

    // TODO: write in lower case
    float GRADIENT_CLIP_THRESHOLD = 8.0f;
    float MOMENTUM                = 0.1f;
    unsigned char PATIENCE        = 12;
    
    float L1_LAMBDA = 0.01f;
    float L2_LAMBDA = 0.01f;
    
    float BETA_1  = 0.9f;
    float BETA_2  = 0.999f;
    float EPSILON = 1e-8f;
    float M_T     = 0;
    float V_T     = 0;
    float T       = 0;

    TensorArray forward_propagation(const Tensor& input, const TensorArray& w, const TensorArray& b);
    std::pair<TensorArray, TensorArray> init_parameters();
    void log_metrics(const std::string& data, const Tensor& y_true, const Tensor& y_pred, const TensorArray *w = nullptr);
};