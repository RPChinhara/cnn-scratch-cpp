#include "tripadvisor.h"
#include "preprocessing.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <windows.h>

Tripadvisor LoadTripadvisor()
{
    std::ifstream file("datasets\\tripadvisor_hotel_reviews.csv");

    if (!file.is_open())
        MessageBox(nullptr, "Failed to open the file", "Error", MB_ICONERROR);

    std::vector<std::string> reviews;
    std::vector<float> ratings;

    std::string line;
    std::getline(file, line);

    size_t idx = 0;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;

        size_t startPos = 0;
        size_t endPos1 = line.find(",1");
        size_t endPos2 = line.find(",2");
        size_t endPos3 = line.find(",3");
        size_t endPos4 = line.find(",4");
        size_t endPos5 = line.find(",5");

        size_t endPos;
        if (endPos1 != std::string::npos)
        {
            endPos = endPos1;
            ratings.push_back(1.0f);
        }
        else if (endPos2 != std::string::npos)
        {
            endPos = endPos2;
            ratings.push_back(2.0f);
        }
        else if (endPos3 != std::string::npos)
        {
            endPos = endPos3;
            ratings.push_back(3.0f);
        }
        else if (endPos4 != std::string::npos)
        {
            endPos = endPos4;
            ratings.push_back(4.0f);
        }
        else if (endPos5 != std::string::npos)
        {
            endPos = endPos5;
            ratings.push_back(5.0f);
        }

        std::string text = line.substr(startPos, endPos - startPos);
        std::cout << text << std::endl << std::endl;
        // std::string textNoLink = RemoveLink(text);
        // std::string textNoHTML = RemoveHTML(textNoLink);
        // std::string textSpaceBetweenPunc = AddSpaceBetweenPunct(textNoHTML);
        // std::string textNoPunc = RemovePunct(textSpaceBetweenPunc);
        // std::string textNoNumber = RemoveNumber(textNoPunc);
        // std::string textNoASCII = RemoveNonASCII(textNoNumber);
        // std::string textNoWhiteSpace = RemoveWhiteSpace(textNoASCII);
        // std::string textNoEmoji = RemoveEmoji(textNoWhiteSpace);
        // std::string textSpellCorrected = SpellCorrection(textNoEmoji);

        // auto tokens = Tokenizer(textSpellCorrected);
        // auto tokensNoStopWords = RemoveStopWords(tokens);

        // std::cout << "Text: " << idx + 1 << std::endl;
        // std::cout << "++++++++++++++++++++++++++: " << std::endl;
        // for (int i = 0; i < tokensNoStopWords.size(); ++i)
        // {
        //     std::cout << tokensNoStopWords[i] << std::endl;
        // }
        // std::cout << "--------------------------: " << ratings[idx] << std::endl << std::endl;
        // ++idx;
    }

    return Tripadvisor();
}