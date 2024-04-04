#include "preproc.h"
#include "arrs.h"
#include "math.hpp"
#include "rand.h"

#include <regex>
#include <sstream>

std::string AddSpaceBetweenPunct(const std::string &text)
{
    std::regex pattern("([.,!?-])");
    std::string s = std::regex_replace(text, pattern, " $1 ");
    s = std::regex_replace(s, std::regex("\\s{2,}"), " ");
    return s;
}

std::vector<std::string> Lemmatizer(const std::vector<std::string> &tokens)
{
    // Define common suffixes to remove
    const std::string suffixes[] = {"s"}; // Add more as needed

    std::vector<std::string> originalForms;

    // Process each word in the vector
    for (const std::string &token : tokens)
    {
        std::string originalForm = token;

        // Iterate through each suffix and check if the word ends with it
        for (const std::string &suffix : suffixes)
        {
            if (originalForm.size() >= suffix.size() &&
                originalForm.substr(originalForm.size() - suffix.size()) == suffix &&
                originalForm.size() > suffix.size())
            {
                // If the word ends with the current suffix (and it's not the whole word itself), remove it
                originalForm = originalForm.substr(0, originalForm.size() - suffix.size());
                break; // Move to the next word
            }
        }

        originalForms.push_back(originalForm);
    }

    return originalForms;
}

Ten MinMaxScaler(Ten &dataset)
{
    auto min_vals = Min(dataset);
    auto max_vals = Max(dataset, 0);
    return (dataset - min_vals) / (max_vals - min_vals);
}

Ten OneHot(const Ten &ten, const size_t depth)
{
    Ten newTensor = zeros({ten.size, depth});

    std::vector<float> indices;

    for (size_t i = 0; i < ten.size; ++i)
    {
        if (i == 0)
            indices.push_back(ten[i]);
        else
            indices.push_back(ten[i] + (i * depth));
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

std::string RemoveHTML(const std::string &text)
{
    std::regex pattern("<[^>]*>");
    return std::regex_replace(text, pattern, " ");
}

std::string RemoveLink(const std::string &text)
{
    std::regex pattern(R"((https?:\/\/|www\.)\S+)");
    return std::regex_replace(text, pattern, "");
}

std::string RemoveNonASCII(const std::string &text)
{
    std::regex pattern("[^\\x00-\\x7f]");
    return std::regex_replace(text, pattern, " ");
}

std::string RemoveNumber(const std::string &text)
{
    std::regex pattern("\\d+");
    return std::regex_replace(text, pattern, "");
}

std::string RemovePunct(const std::string &text)
{
    std::regex pattern("[\"#$%&'()*+/:;<=>@\\[\\\\\\]^_`{|}~]");
    return std::regex_replace(text, pattern, " ");
}

std::string RemovePunct2(const std::string &text)
{
    std::regex regex("[^\\w\\s]");
    std::string result = std::regex_replace(text, regex, " ");

    return result;
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

std::string ToLower(const std::string &text)
{
    std::string result;
    for (char c : text)
    {
        result += std::tolower(c);
    }
    return result;
}

TrainTest TrainTestSplit(const Ten &x, const Ten &y, const float testSize, const size_t randomState)
{
    Ten x_new = shuffle(x, randomState);
    Ten y_new = shuffle(y, randomState);

    TrainTest train_test;
    train_test.trainFeatures =
        zeros({static_cast<size_t>(std::floorf(x.shape.front() * (1.0 - testSize))), x.shape.back()});
    train_test.trainTargets =
        zeros({static_cast<size_t>(std::floorf(y.shape.front() * (1.0 - testSize))), y.shape.back()});
    train_test.testFeatures = zeros({static_cast<size_t>(std::ceilf(x.shape.front() * testSize)), x.shape.back()});
    train_test.testTargets = zeros({static_cast<size_t>(std::ceilf(y.shape.front() * testSize)), y.shape.back()});

    for (size_t i = 0; i < train_test.trainFeatures.size; ++i)
        train_test.trainFeatures[i] = x_new[i];

    for (size_t i = 0; i < train_test.trainTargets.size; ++i)
        train_test.trainTargets[i] = y_new[i];

    for (size_t i = train_test.trainFeatures.size; i < x.size; ++i)
        train_test.testFeatures[i - train_test.trainFeatures.size] = x_new[i];

    for (size_t i = train_test.trainTargets.size; i < y.size; ++i)
        train_test.testTargets[i - train_test.trainTargets.size] = y_new[i];

    return train_test;
}