#pragma once

#include "ten.h"

#include <string>
#include <vector>

struct TrainTest
{
    Ten train_features;
    Ten train_targets;
    Ten test_features;
    Ten test_targets;
};

std::string AddSpaceBetweenPunct(const std::string &text);

std::vector<std::string> Lemmatizer(const std::vector<std::string> &tokens);

Ten min_max_scaler(Ten &dataset);

Ten one_hot(const Ten &t, const size_t depth);

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

TrainTest train_test_split(const Ten &x, const Ten &y, const float testSize, const size_t randomState);