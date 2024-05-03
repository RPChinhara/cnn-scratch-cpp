#include "enes.h"

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdio.h>

en_es load_en_es()
{
    // SetConsoleOutputCP(CP_UTF8);
    // std::locale("es_ES.UTF-8");
    // _setmode(_fileno(stdout), _O_U16TEXT);

    std::locale::global(std::locale("es_ES.UTF-8"));

    std::wifstream file("datas/spa.txt");
    if (!file)
        std::wcerr << L"Failed to open the file." << std::endl;

    std::vector<std::wstring> x;
    std::vector<std::wstring> y;

    std::wstring line;
    while (std::getline(file, line))
    {
        size_t tab_pos = line.find(L"\t");

        std::wstring english_part = line.substr(0, tab_pos);
        std::wstring spanish_part = line.substr(tab_pos + 1);

        size_t cc_by_pos = spanish_part.find(L"CC-BY");

        if (cc_by_pos != std::wstring::npos)
            spanish_part.erase(cc_by_pos);

        x.push_back(spanish_part);
        y.push_back(english_part);
    }

    file.close();

    en_es data;
    data.x = x;
    data.y = y;

    return data;
}