#pragma once

#include <string>
#include <vector>

struct EnEs
{
    std::vector<std::string> targetRaw;
    std::vector<std::string> contextRaw;
};

EnEs load_en_es();