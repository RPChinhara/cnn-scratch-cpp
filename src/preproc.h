#pragma once

#include "math.hpp"
#include "tensor.h"

#include <string>
#include <vector>

struct train_test
{
    tensor x_train;
    tensor y_train;
    tensor x_test;
    tensor y_test;
};

class min_max_scaler2 {
  private:
    tensor data_min;
    tensor data_max;
    bool is_fitted;

  public:
    min_max_scaler2() : data_min(), data_max(), is_fitted(false) {}

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

std::wstring join(const std::vector<std::wstring> &strings, const std::wstring &separator);
std::string lower(const std::string &text);
std::wstring lower(const std::wstring &text);
tensor min_max_scaler(tensor &data);
tensor one_hot(const tensor &t, const size_t depth);
std::string regex_replace(const std::string &in, const std::string &pattern, const std::string &rewrite);
std::wstring regex_replace(const std::wstring &in, const std::wstring &pattern, const std::wstring &rewrite);
std::wstring strip(const std::wstring &text);
std::vector<std::string> tokenizer(const std::string &text);
std::vector<std::wstring> tokenizer(const std::wstring &text);
train_test split_dataset(const tensor &x, const tensor &y, const float test_size, const size_t rd_state);