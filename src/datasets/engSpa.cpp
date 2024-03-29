#include "engSpa.h"

#include <fcntl.h>
#include <fstream>
#include <io.h>
#include <iostream>
#include <sstream>
#include <stdio.h>

EngSpa LoadEngSpa()
{

    std::ifstream file("datasets\\eng-spa.txt");
    if (!file)
    {
        std::cerr << "Failed to open the file." << std::endl;
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

        std::cout << line << std::endl;
  
    }

    file.close();

    EngSpa engSpa;

    engSpa.targetRaw = targetRaw;
    engSpa.contextRaw = contextRaw;
}