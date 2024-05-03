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
        size_t pos_tab = line.find(L"\t");

        std::wstring english_part = line.substr(0, pos_tab);
        std::wstring spanish_part = line.substr(pos_tab + 1);

        size_t pos_cc_by = spanish_part.find(L"CC-BY");

        if (pos_cc_by != std::wstring::npos)
            spanish_part.erase(pos_cc_by);

        x.push_back(spanish_part);
        y.push_back(english_part);
    }

    file.close();

    en_es data;
    data.x = x;
    data.y = y;

    return data;
}