#include "preproc.h"
#include "arrs.h"
#include "math.hpp"
#include "rand.h"

#include <algorithm>
#include <cwctype>
#include <regex>
#include <sstream>

std::string lower(const std::string& text) {
    std::string result;
    for (auto c : text) {
        result += std::tolower(c);
    }
    return result;
}

tensor one_hot(const tensor& t, const size_t depth) {
    tensor t_new = zeros({t.size, depth});

    std::vector<float> idx;

    for (auto i = 0; i < t.size; ++i) {
        if (i == 0)
            idx.push_back(t[i]);
        else
            idx.push_back(t[i] + (i * depth));
    }

    for (auto i = 0; i < t_new.size; ++i) {
        for (auto j : idx) {
            if (i == j)
                t_new[i] = 1.0f;
        }
    }

    return t_new;
}

std::string regex_replace(const std::string& in, const std::string& pattern, const std::string& rewrite) {
    std::regex re(pattern);
    return std::regex_replace(in, re, rewrite);
}

train_test split_dataset(const tensor& x, const tensor& y, const float test_size, const size_t rd_state) {
    tensor x_shuffled = shuffle(x, rd_state);
    tensor y_shuffled = shuffle(y, rd_state);

    train_test data;
    data.x_train = zeros({static_cast<size_t>(std::floorf(x.shape.front() * (1.0 - test_size))), x.shape.back()});
    data.y_train = zeros({static_cast<size_t>(std::floorf(y.shape.front() * (1.0 - test_size))), y.shape.back()});
    data.x_test = zeros({static_cast<size_t>(std::ceilf(x.shape.front() * test_size)), x.shape.back()});
    data.y_test = zeros({static_cast<size_t>(std::ceilf(y.shape.front() * test_size)), y.shape.back()});

    for (auto i = 0; i < data.x_train.size; ++i)
        data.x_train[i] = x_shuffled[i];

    for (auto i = 0; i < data.y_train.size; ++i)
        data.y_train[i] = y_shuffled[i];

    for (auto i = data.x_train.size; i < x.size; ++i)
        data.x_test[i - data.x_train.size] = x_shuffled[i];

    for (auto i = data.y_train.size; i < y.size; ++i)
        data.y_test[i - data.y_train.size] = y_shuffled[i];

    return data;
}

std::vector<std::string> tokenizer(const std::string& text) {
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;

    while (ss >> token) {
        tokens.push_back(token);
    }

    return tokens;
}