#pragma once

#include "tensor.h"

#include <string>
#include <vector>

struct train_test {
    tensor x_train;
    tensor y_train;
    tensor x_test;
    tensor y_test;
};

std::string lower(const std::string& text);
tensor one_hot(const tensor& t, const size_t depth);
std::string regex_replace(const std::string& in, const std::string& pattern, const std::string& rewrite);
std::vector<std::string> tokenizer(const std::string& text);