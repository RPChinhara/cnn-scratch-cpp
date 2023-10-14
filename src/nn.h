#pragma once

#include "tensor.h"
#include "losses.h"
#include "metrics.h"

#include <vector>

using TensorArray = std::vector<Tensor>;

class NN {
  public:
    NN(const std::vector<unsigned int>& layers);
    std::pair<TensorArray, TensorArray> train(const Tensor& train_x, const Tensor& train_y, const Tensor& val_x, const Tensor& val_y);
    void predict(const Tensor& test_x, const Tensor& test_y, const TensorArray& w, const TensorArray& b);

  private:
    float (*ACCURACY)(const Tensor& y_true, const Tensor& y_pred) = &categorical_accuracy;
    float (*LOSS)(const Tensor& y_true, const Tensor& y_pred)     = &categorical_crossentropy;
    unsigned short epochs     = 100;
    unsigned short batch_size = 100;
    float learning_rate       = 0.01f;
    std::vector<unsigned int> layers;

    TensorArray forward_propagation(const Tensor& input, const TensorArray& w, const TensorArray& b);
    std::pair<TensorArray, TensorArray> init_parameters();
    void log_metrics(const std::string& data, const Tensor& y_true, const Tensor& y_pred, const TensorArray *w = nullptr);
};