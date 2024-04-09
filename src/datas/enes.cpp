#include "enes.h"

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdio.h>

EnEs load_en_es()
{
    std::wifstream file("datas/spa.txt");
    if (!file)
        std::wcerr << L"Failed to open the file." << std::endl;

    std::vector<std::wstring> targetRaw;
    std::vector<std::wstring> contextRaw;

    std::wstring line;
    while (std::getline(file, line))
    {
        size_t pos = line.find(L"CC-BY");

        if (pos != std::wstring::npos)
        {
            line.erase(pos);
        }

        pos = line.find_first_of(L".!?");

        std::wstring extractedTargetRaw = line.substr(0, pos + 1);
        targetRaw.push_back(extractedTargetRaw);

        std::wstring extractedContextRaw = line.substr(pos + 1);
        std::wregex regex(L"\\s*(.*)");
        std::wstring extractedContextRawNoPrecedingSpaces = std::regex_replace(extractedContextRaw, regex, L"$1");
        contextRaw.push_back(extractedContextRawNoPrecedingSpaces);
    }

    file.close();

    EnEs eng_spa;
    eng_spa.targetRaw = targetRaw;
    eng_spa.contextRaw = contextRaw;

    return eng_spa;
}