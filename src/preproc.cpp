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

    for (size_t i = 0; i < t.size; ++i) {
        size_t index = t[i] + (i * depth);
        t_new[index] = 1.0f;
    }

    return t_new;
}

std::string regex_replace(const std::string& in, const std::string& pattern, const std::string& rewrite) {
    std::regex re(pattern);
    return std::regex_replace(in, re, rewrite);
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