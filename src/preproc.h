#pragma once

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