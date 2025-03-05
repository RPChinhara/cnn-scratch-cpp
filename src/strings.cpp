#include "string.h"

#include <regex>
#include <sstream>

std::vector<std::string> tokenizer(const std::string& text) {
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;

    while (ss >> token) {
        tokens.push_back(token);
    }

    return tokens;
}