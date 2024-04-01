#pragma once

#include <string>
#include <vector>

struct EngSpa
{
    std::vector<std::string> targetRaw;
    std::vector<std::string> contextRaw;
};

EngSpa LoadEngSpa();