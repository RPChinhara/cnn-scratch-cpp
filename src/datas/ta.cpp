#include "ta.h"
#include "preproc.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

Tripadvisor LoadTripadvisor()
{
    std::ifstream file("datas\\tripadvisor_hotel_reviews.csv");

    if (!file.is_open())
        std::cerr << "Failed to open the file." << std::endl;

    std::vector<std::string> reviews;
    std::vector<std::string> ratings;

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
            ratings.push_back("Negative");
        }
        else if (endPos2 != std::string::npos)
        {
            endPos = endPos2;
            ratings.push_back("Negative");
        }
        else if (endPos3 != std::string::npos)
        {
            endPos = endPos3;
            ratings.push_back("Neutral");
        }
        else if (endPos4 != std::string::npos)
        {
            endPos = endPos4;
            ratings.push_back("Positive");
        }
        else if (endPos5 != std::string::npos)
        {
            endPos = endPos5;
            ratings.push_back("Positive");
        }

        std::string text = line.substr(startPos, endPos - startPos);
        std::string textNoEmoji = RemoveEmoji(text);
        std::string textLower = ToLower(textNoEmoji);
        std::string textNoPunc = RemovePunct2(textLower);
        std::string textNoWhiteSpace = RemoveWhiteSpace(textNoPunc);
        auto tokens = Tokenizer(textNoWhiteSpace);

        for (int i = 0; i < tokens.size(); ++i)
        {
            std::cout << tokens[i] << std::endl;
        }
        std::cout << "--------------------------: " << ratings[idx] << std::endl << std::endl;
        ++idx;
    }

    return Tripadvisor();
}