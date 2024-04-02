#pragma once

#include "ten.h"

#include <string>
#include <vector>

struct TrainTest
{
    Ten trainFeatures;
    Ten trainTargets;
    Ten testFeatures;
    Ten testTargets;
};

std::string AddSpaceBetweenPunct(const std::string &text);

std::vector<std::string> Lemmatizer(const std::vector<std::string> &tokens);

Ten MinMaxScaler(Ten &dataset);

Ten OneHot(const Ten &ten, const size_t depth);

std::string RemoveEmoji(const std::string &text);

std::string RemoveHTML(const std::string &text);

std::string RemoveLink(const std::string &text);

std::string RemoveNonASCII(const std::string &text);

std::string RemoveNumber(const std::string &text);

std::string RemovePunct(const std::string &text);

std::string RemovePunct2(const std::string &text);

std::vector<std::string> RemoveStopWords(const std::vector<std::string> &tokens);

std::string RemoveWhiteSpace(const std::string &text);

std::string SpellCorrection(const std::string &text);

std::vector<std::string> Tokenizer(const std::string &text);

std::string ToLower(const std::string &text);

TrainTest TrainTestSplit(const Ten &x, const Ten &y, const float testSize, const size_t randomState);