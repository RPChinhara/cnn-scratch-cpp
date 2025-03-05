#pragma once

#include "tensor.h"

#include <string>
#include <vector>

std::string regex_replace(const std::string& in, const std::string& pattern, const std::string& rewrite);
std::vector<std::string> tokenizer(const std::string& text);