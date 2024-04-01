#include "imdb.h"
#include "preproc.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

IMDB LoadIMDB()
{
    std::ifstream file("datasets\\IMDB Dataset.csv");

    if (!file.is_open())
        std::cerr << "Failed to open the file." << std::endl;

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

        auto tokens = Tokenizer(textSpellCorrected);
        auto tokensNoStopWords = RemoveStopWords(tokens);

        std::cout << "Text: " << idx + 1 << std::endl;
        std::cout << "++++++++++++++++++++++++++: " << std::endl;
        for (int i = 0; i < tokensNoStopWords.size(); ++i)
        {
            std::cout << tokensNoStopWords[i] << std::endl;
        }
        std::cout << "--------------------------: " << sentiments[idx] << std::endl << std::endl;
        ++idx;
    }

    return IMDB();
}