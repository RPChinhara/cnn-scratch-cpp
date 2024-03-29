#pragma once

#include <string>
#include <vector>

struct EnglishSpanish
{
    std::vector<std::string> targetRaw;
    std::vector<std::string> contextRaw;
};

EnglishSpanish LoadEnglishSpanish();