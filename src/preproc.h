#pragma once

#include "ten.h"

#include <string>
#include <vector>

struct train_test
{
    ten x_train;
    ten y_train;
    ten x_test;
    ten y_test;
};

std::wstring join(const std::vector<std::wstring> &strings, const std::wstring &separator);
std::string lower(const std::string &text);
std::wstring lower(const std::wstring &text);
ten min_max_scaler(ten &data);
ten one_hot(const ten &t, const size_t depth);
std::string regex_replace(const std::string &in, const std::string &pattern, const std::string &rewrite);
std::wstring regex_replace(const std::wstring &in, const std::wstring &pattern, const std::wstring &rewrite);
std::wstring strip(const std::wstring &text);
std::vector<std::string> tokenizer(const std::string &text);
std::vector<std::wstring> tokenizer(const std::wstring &text);
train_test split_dataset(const ten &x, const ten &y, const float test_size, const size_t rd_state);