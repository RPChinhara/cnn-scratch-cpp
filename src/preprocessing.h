#pragma once

#include "tensor.h"

#include <string>
#include <vector>

struct TrainTest
{
    Tensor trainFeatures;
    Tensor trainTargets;
    Tensor testFeatures;
    Tensor testTargets;
};

std::string AddSpaceBetweenPunct(const std::string &text);

std::vector<std::string> Lemmatizer(const std::vector<std::string> &tokens);

Tensor MinMaxScaler(Tensor &dataset);

Tensor OneHot(const Tensor &tensor, const size_t depth);

std::string RemoveEmoji(const std::string &text);

std::string RemoveHTML(const std::string &text);

std::string RemoveLink(const std::string &text);

std::string RemoveNonASCII(const std::string &text);

std::string RemoveNumber(const std::string &text);

std::string RemovePunct(const std::string &text);

std::vector<std::string> RemoveStopWords(const std::vector<std::string> &tokens);

std::string RemoveWhiteSpace(const std::string &text);

std::string SpellCorrection(const std::string &text);

std::vector<std::string> Tokenizer(const std::string &text);

std::string ToLower(const std::string &text);

TrainTest TrainTestSplit(const Tensor &x, const Tensor &y, const float testSize, const size_t randomState);