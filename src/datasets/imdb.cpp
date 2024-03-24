#include "imdb.h"
#include "preprocessing.h"

#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>
#include <windows.h>

std::string RemoveLink(const std::string &text)
{
    std::regex pattern(R"((https?:\/\/|www\.)\S+)");
    return std::regex_replace(text, pattern, "");
}

std::string RemoveHTML(const std::string &text)
{
    std::regex pattern("<[^>]*>");
    return std::regex_replace(text, pattern, " ");
}

std::string AddSpaceBetweenPunct(const std::string &text)
{
    std::regex pattern("([.,!?-])");
    std::string s = std::regex_replace(text, pattern, " $1 ");
    s = std::regex_replace(s, std::regex("\\s{2,}"), " ");
    return s;
}

std::string RemovePunct(const std::string &text)
{
    std::regex pattern("[\"#$%&'()*+/:;<=>@\\[\\\\\\]^_`{|}~]");
    return std::regex_replace(text, pattern, " ");
}

std::string RemoveNumber(const std::string &text)
{
    std::regex pattern("\\d+");
    return std::regex_replace(text, pattern, "");
}

std::string RemoveNonASCII(const std::string &text)
{
    std::regex pattern("[^\\x00-\\x7f]");
    return std::regex_replace(text, pattern, " ");
}

std::string RemoveWhiteSpace(const std::string &text)
{
    std::regex pattern("\\s+");
    return std::regex_replace(text, pattern, " ");
}

std::string RemoveEmoji(const std::string &text)
{
    std::regex pattern("[\xE2\x98\x80-\xE2\x9B\xBF]");
    return std::regex_replace(text, pattern, "");
}

std::string SpellCorrection(const std::string &text)
{
    std::regex pattern("(.)\\1+");
    return std::regex_replace(text, pattern, "$1$1");
}

IMDB LoadIMDB()
{
    std::ifstream file("datasets\\IMDB Dataset.csv");

    if (!file.is_open())
        MessageBox(nullptr, "Failed to open the file", "Error", MB_ICONERROR);

    std::vector<std::string> reviews;
    std::vector<float> sentiments;

    std::string line;
    std::getline(file, line);

    size_t idx = 0;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;

        size_t startPos = 0;
        size_t endPosPositive = line.find(",positive");
        size_t endPosNegative = line.find(",negative");

        size_t endPos;
        if (endPosPositive != std::string::npos)
        {
            endPos = endPosPositive;
            sentiments.push_back(1.0f);
        }
        else if (endPosNegative != std::string::npos)
        {
            endPos = endPosNegative;
            sentiments.push_back(0.0f);
        }

        std::string text = line.substr(startPos, endPos - startPos);
        std::string textNoLink = RemoveLink(text);
        std::string textNoHTML = RemoveHTML(textNoLink);
        std::string textSpaceBetweenPunc = AddSpaceBetweenPunct(textNoHTML);
        std::string textNoPunc = RemovePunct(textSpaceBetweenPunc);
        std::string textNoNumber = RemoveNumber(textNoPunc);
        std::string textNoASCII = RemoveNonASCII(textNoNumber);
        std::string textNoWhiteSpace = RemoveWhiteSpace(textNoASCII);
        std::string textNoEmoji = RemoveEmoji(textNoWhiteSpace);
        std::string textSpellCorrected = SpellCorrection(textNoEmoji);

        auto textTokenized = Tokenizer(textSpellCorrected);

        std::cout << "Text: " << idx + 1 << std::endl;
        std::cout << "++++++++++++++++++++++++++: " << std::endl;
        for (int i = 0; i < textTokenized.size(); ++i)
        {
            std::cout << textTokenized[i] << std::endl;
        }
        std::cout << "--------------------------: " << sentiments[idx] << std::endl << std::endl;
        ++idx;
    }

    return IMDB();
}