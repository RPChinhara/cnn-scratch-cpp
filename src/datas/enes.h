#pragma once

#include <string>
#include <vector>

struct EnEs
{
    std::vector<std::wstring> targetRaw;
    std::vector<std::wstring> contextRaw;
};

EnEs load_en_es();