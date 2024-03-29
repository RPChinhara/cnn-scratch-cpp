#include "englishspanish.h"

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdio.h>

EnglishSpanish LoadEnglishSpanish()
{

    std::ifstream file("datasets\\english-spanish.txt");
    if (!file)
    {
        std::cout << "Failed to open the file." << std::endl;
        // return 1;
    }

    std::vector<std::string> targetRaw;
    std::vector<std::string> contextRaw;

    std::string line;
    while (std::getline(file, line))
    {
        size_t pos = line.find("CC-BY");

        if (pos != std::string::npos)
        {
            line.erase(pos);
        }

        pos = line.find_first_of(".!?");

        std::string extractedTargetRaw = line.substr(0, pos + 1);
        targetRaw.push_back(extractedTargetRaw);

        std::string extractedContextRaw = line.substr(pos + 1);
        std::regex regex("\\s*(.*)");
        std::string extractedContextRawNoPrecedingSpaces = std::regex_replace(extractedContextRaw, regex, "$1");
        contextRaw.push_back(extractedContextRawNoPrecedingSpaces);
    }

    file.close();

    EnglishSpanish englishSpanish;
    englishSpanish.targetRaw = targetRaw;
    englishSpanish.contextRaw = contextRaw;

    return englishSpanish;
}