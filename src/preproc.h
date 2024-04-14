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

ten min_max_scaler(ten &dataset);
ten one_hot(const ten &t, const size_t depth);
std::string regex_replace(const std::string &in, const std::string &pattern, const std::string &rewrite);
std::vector<std::string> tokenizer(const std::string &text);
train_test train_test_split(const ten &x, const ten &y, const float test_size, const size_t rand_state);
std::wstring wjoin(const std::vector<std::wstring> &strings, const std::wstring &separator);
std::wstring wlower(const std::wstring &text);
std::wstring wregex_replace(const std::wstring &in, const std::wstring &pattern, const std::wstring &rewrite);
std::wstring wstrip(const std::wstring &text);

std::vector<std::string> RemoveStopWords(const std::vector<std::string> &tokens);