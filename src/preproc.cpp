#include "preproc.h"
#include "arrs.h"
#include "math.hpp"
#include "rand.h"

#include <cwctype>
#include <regex>
#include <sstream>

ten min_max_scaler(ten &dataset)
{
    auto min_vals = Min(dataset);
    auto max_vals = Max(dataset, 0);
    return (dataset - min_vals) / (max_vals - min_vals);
}

ten one_hot(const ten &t, const size_t depth)
{
    ten newTensor = zeros({t.size, depth});

    std::vector<float> indices;

    for (size_t i = 0; i < t.size; ++i)
    {
        if (i == 0)
            indices.push_back(t[i]);
        else
            indices.push_back(t[i] + (i * depth));
    }

    for (size_t i = 0; i < newTensor.size; ++i)
    {
        for (auto j : indices)
        {
            if (i == j)
                newTensor[i] = 1.0f;
        }
    }

    return newTensor;
}

std::string RemoveEmoji(const std::string &text)
{
    std::regex pattern("[\xE2\x98\x80-\xE2\x9B\xBF]");
    return std::regex_replace(text, pattern, "");
}

std::string RemoveNonASCII(const std::string &text)
{
    std::regex pattern("[^\\x00-\\x7f]");
    return std::regex_replace(text, pattern, " ");
}

std::vector<std::string> RemoveStopWords(const std::vector<std::string> &tokens)
{
    std::vector<std::string> stopWords = {
        "aren't", "when",   "each",    "him",       "after",     "most",     "m",       "ma",         "needn",
        "over",   "during", "few",     "against",   "off",       "he",       "they",    "from",       "i",
        "to",     "wouldn", "between", "other",     "ain",       "this",     "won't",   "you've",     "where",
        "if",     "be",     "at",      "hasn't",    "yourself",  "needn't",  "doing",   "aren",       "same",
        "no",     "o",      "of",      "mightn't",  "will",      "yours",    "my",      "while",      "both",
        "so",     "mightn", "our",     "ll",        "these",     "re",       "and",     "that'll",    "doesn't",
        "isn't",  "wasn",   "should",  "their",     "hasn",      "have",     "it",      "can",        "being",
        "mustn",  "by",     "because", "under",     "were",      "haven",    "them",    "themselves", "himself",
        "those",  "then",   "mustn't", "ourselves", "should've", "too",      "did",     "on",         "again",
        "weren",  "a",      "above",   "couldn",    "weren't",   "doesn",    "what",    "how",        "only",
        "your",   "as",     "we",      "there",     "further",   "ours",     "below",   "couldn't",   "you'd",
        "am",     "s",      "an",      "shouldn't", "just",      "or",       "itself",  "in",         "do",
        "theirs", "she",    "does",    "who",       "why",       "y",        "shouldn", "hers",       "don",
        "up",     "out",    "she's",   "you're",    "down",      "about",    "didn't",  "into",       "own",
        "shan't", "here",   "the",     "having",    "than",      "wasn't",   "ve",      "its",        "with",
        "until",  "you",    "is",      "but",       "some",      "his",      "hadn",    "herself",    "t",
        "that",   "won",    "through", "her",       "d",         "wouldn't", "all",     "whom",       "yourselves",
        "which",  "are",    "been",    "had",       "don't",     "you'll",   "for",     "has",        "haven't",
        "myself", "once",   "any",     "before",    "shan",      "isn",      "more",    "nor",        "now",
        "me",     "hadn't", "such",    "not",       "was",       "very",     "it's",    "didn"};

    // this, yours, themselves, ourselves, ours, hers, She's

    std::vector<std::string> tokensNoStopWords;

    for (const std::string &token : tokens)
    {
        bool found = false;

        for (const std::string &stopWord : stopWords)
        {
            if (stopWord == token)
            {
                found = true;
                break;
            }
        }

        if (!found)
            tokensNoStopWords.push_back(token);
    }

    return tokensNoStopWords;
}

std::string RemoveWhiteSpace(const std::string &text)
{
    std::regex pattern("\\s+");
    return std::regex_replace(text, pattern, " ");
}

std::string SpellCorrection(const std::string &text)
{
    std::regex pattern("(.)\\1+");
    return std::regex_replace(text, pattern, "$1$1");
}

std::vector<std::string> Tokenizer(const std::string &text)
{
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;

    while (ss >> token)
    {
        tokens.push_back(token);
    }

    return tokens;
}

std::string regex_replace(const std::string &in, const std::string &pattern, const std::string &rewrite)
{
    std::regex re(pattern);
    return std::regex_replace(in, re, rewrite);
}

std::wstring wregex_replace(const std::wstring &in, const std::wstring &pattern, const std::wstring &rewrite)
{
    std::wregex regex(pattern);
    return std::regex_replace(in, regex, rewrite);
}

std::wstring wstrip(const std::wstring &text)
{
    std::wregex pattern(L"(^\\s+)|(\\s+$)");
    return std::regex_replace(text, pattern, L"");
}

std::wstring wlower(const std::wstring &text)
{
    std::wstring result;
    for (wchar_t c : text)
    {
        result += std::towlower(c);
    }
    return result;
}

train_test train_test_split(const ten &x, const ten &y, const float test_size, const size_t rand_state)
{
    ten x_shuffled = shuffle(x, rand_state);
    ten y_shuffled = shuffle(y, rand_state);

    train_test data;
    data.x_train = zeros({static_cast<size_t>(std::floorf(x.shape.front() * (1.0 - test_size))), x.shape.back()});
    data.y_train = zeros({static_cast<size_t>(std::floorf(y.shape.front() * (1.0 - test_size))), y.shape.back()});
    data.x_test = zeros({static_cast<size_t>(std::ceilf(x.shape.front() * test_size)), x.shape.back()});
    data.y_test = zeros({static_cast<size_t>(std::ceilf(y.shape.front() * test_size)), y.shape.back()});

    for (size_t i = 0; i < data.x_train.size; ++i)
        data.x_train[i] = x_shuffled[i];

    for (size_t i = 0; i < data.y_train.size; ++i)
        data.y_train[i] = y_shuffled[i];

    for (size_t i = data.x_train.size; i < x.size; ++i)
        data.x_test[i - data.x_train.size] = x_shuffled[i];

    for (size_t i = data.y_train.size; i < y.size; ++i)
        data.y_test[i - data.y_train.size] = y_shuffled[i];

    return data;
}

std::wstring wjoin(const std::vector<std::wstring> &strings, const std::wstring &separator)
{
    if (strings.empty())
    {
        return L"";
    }

    std::wstring result = strings[0];
    for (size_t i = 1; i < strings.size(); ++i)
    {
        result += separator + strings[i];
    }

    return result;
}