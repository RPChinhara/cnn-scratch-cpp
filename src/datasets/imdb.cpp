#include "imdb.h"

#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>
#include <windows.h>

std::string RemoveLink(const std::string &input)
{
    std::regex linkPattern(R"((https?:\/\/|www\.)\S+)");
    std::string output = std::regex_replace(input, linkPattern, "");
    return output;
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

        std::string sentence = line.substr(startPos, endPos - startPos);
        std::string sentenceNoLink = RemoveLink(sentence);

        reviews.push_back(sentenceNoLink);
        std::cout << "Sentence: " << idx + 1 << std::endl;
        std::cout << "++++++++++++++++++++++++++: " << reviews[idx] << std::endl;
        std::cout << "--------------------------: " << sentiments[idx] << std::endl << std::endl;
        ++idx;
    }

    return IMDB();
}