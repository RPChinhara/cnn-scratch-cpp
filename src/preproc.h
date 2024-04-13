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

std::string AddSpaceBetweenPunct(const std::string &text);
ten min_max_scaler(ten &dataset);
ten one_hot(const ten &t, const size_t depth);
std::string RemoveEmoji(const std::string &text);
std::string RemoveHTML(const std::string &text);
std::string RemoveNonASCII(const std::string &text);
std::string RemoveNumber(const std::string &text);
std::string RemovePunct(const std::string &text);
std::string RemovePunct2(const std::string &text);
std::vector<std::string> RemoveStopWords(const std::vector<std::string> &tokens);
std::string RemoveWhiteSpace(const std::string &text);
std::string SpellCorrection(const std::string &text);
std::string regex_replace(const std::string &in, const std::string &pattern, const std::string &rewrite);
std::wstring wregex_replace(const std::wstring &in, const std::wstring &pattern, const std::wstring &rewrite);
std::wstring wstrip(const std::wstring &text);
std::vector<std::string> Tokenizer(const std::string &text);
std::wstring wlower(const std::wstring &text);
train_test train_test_split(const ten &x, const ten &y, const float test_size, const size_t rand_state);
std::wstring wjoin(const std::vector<std::wstring> &strings, const std::wstring &separator);