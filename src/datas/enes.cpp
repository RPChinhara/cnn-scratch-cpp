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
        size_t pos = line.find(L"CC-BY");

        if (pos != std::wstring::npos)
        {
            line.erase(pos);
        }

        pos = line.find_first_of(L".!?");

        std::wstring extracted_y = line.substr(0, pos + 1);
        y.push_back(extracted_y);

        std::wstring extracted_x = line.substr(pos + 1);
        std::wregex regex(L"\\s*(.*)");
        std::wstring extracted_x_no_preceding_spaces = std::regex_replace(extracted_x, regex, L"$1");
        x.push_back(extracted_x_no_preceding_spaces);
    }

    file.close();

    en_es data;
    data.x = x;
    data.y = y;

    return data;
}