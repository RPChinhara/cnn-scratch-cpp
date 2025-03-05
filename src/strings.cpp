#include "string.h"

#include <regex>
#include <sstream>

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