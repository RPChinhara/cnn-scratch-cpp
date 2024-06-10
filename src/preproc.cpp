#include "preproc.h"
#include "arrs.h"
#include "math.hpp"
#include "rd.h"

#include <algorithm>
#include <cwctype>
#include <regex>
#include <sstream>

std::wstring join(const std::vector<std::wstring> &strings, const std::wstring &separator)
{
    if (strings.empty())
    {
        return L"";
    }

    std::wstring result = strings[0];
    for (auto i = 1; i < strings.size(); ++i)
    {
        result += separator + strings[i];
    }

    return result;
}

std::string lower(const std::string &text)
{
    std::string result;
    for (auto c : text)
    {
        result += std::tolower(c);
    }
    return result;
}

std::wstring lower(const std::wstring &text)
{
    std::wstring result;
    for (auto c : text)
    {
        result += std::towlower(c);
    }
    return result;
}

ten min_max_scaler(ten &data)
{
    auto min_vals = min(data);
    auto max_vals = max(data, 0);
    return (data - min_vals) / (max_vals - min_vals);
}

ten one_hot(const ten &t, const size_t depth)
{
    ten t_new = zeros({t.size, depth});

    std::vector<float> idx;

    for (auto i = 0; i < t.size; ++i)
    {
        if (i == 0)
            idx.push_back(t[i]);
        else
            idx.push_back(t[i] + (i * depth));
    }

    for (auto i = 0; i < t_new.size; ++i)
    {
        for (auto j : idx)
        {
            if (i == j)
                t_new[i] = 1.0f;
        }
    }

    return t_new;
}

void pad_sequences()
{
}

std::string regex_replace(const std::string &in, const std::string &pattern, const std::string &rewrite)
{
    std::regex re(pattern);
    return std::regex_replace(in, re, rewrite);
}

std::wstring regex_replace(const std::wstring &in, const std::wstring &pattern, const std::wstring &rewrite)
{
    std::wregex regex(pattern);
    return std::regex_replace(in, regex, rewrite);
}

std::wstring strip(const std::wstring &text)
{
    std::wregex pattern(L"(^\\s+)|(\\s+$)");
    return std::regex_replace(text, pattern, L"");
}

std::vector<std::string> tokenizer(const std::string &text)
{
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;

    while (ss >> token)
    {
        tokens.push_back(token);
    }

    return tokens;
}

std::vector<std::wstring> tokenizer(const std::wstring &text)
{
    std::vector<std::wstring> tokens;
    std::wstringstream ss(text);
    std::wstring token;

    while (ss >> token)
    {
        tokens.push_back(token);
    }

    return tokens;
}

train_test split_dataset(const ten &x, const ten &y, const float test_size, const size_t rd_state)
{
    ten x_shuffled = shuffle(x, rd_state);
    ten y_shuffled = shuffle(y, rd_state);

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