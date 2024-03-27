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

std::vector<std::string> Lemmatizer(const std::vector<std::string> &tokens);
Tensor MinMaxScaler(Tensor &dataset);
Tensor OneHot(const Tensor &tensor, const size_t depth);
std::vector<std::string> RemoveStopWords(const std::vector<std::string> &tokens);
std::vector<std::string> Tokenizer(const std::string &text);
TrainTest TrainTestSplit(const Tensor &x, const Tensor &y, const float testSize, const size_t randomState);