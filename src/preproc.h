#pragma once

#include "math.hpp"
#include "tensor.h"

#include <string>
#include <vector>

struct train_test {
    tensor x_train;
    tensor y_train;
    tensor x_test;
    tensor y_test;
};

class min_max_scaler {
  private:
    tensor data_min;
    tensor data_max;
    bool is_fitted;

  public:
    min_max_scaler() : data_min(), data_max(), is_fitted(false) {}

    void fit(const tensor& data) {
        data_min = min(data);
        data_max = max(data, 0);
        is_fitted = true;
    }

    tensor transform(const tensor& data) {
        if (!is_fitted)
            std::cerr << "Scaler not fitted yet." << std::endl;

        return (data - data_min) / (data_max - data_min);
    }

    tensor inverse_transform(const tensor& scaled_data) {
       if (!is_fitted)
            std::cerr << "Scaler not fitted yet." << std::endl;

        return scaled_data * (data_max - data_min) + data_min;
    }

};

std::string lower(const std::string &text);
tensor one_hot(const tensor& t, const size_t depth);
std::string regex_replace(const std::string &in, const std::string &pattern, const std::string &rewrite);
train_test split_dataset(const tensor& x, const tensor& y, const float test_size, const size_t rd_state);
std::vector<std::string> tokenizer(const std::string &text);