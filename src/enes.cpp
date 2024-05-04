#include "enes.h"

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdio.h>

en_es load_en_es()
{
    std::locale::global(std::locale("es_ES.UTF-8"));

    std::wifstream file("datas/spa.txt");

    if (!file)
        std::cerr << "Failed to open the file." << std::endl;

    std::vector<std::wstring> x;
    std::vector<std::wstring> y;

    std::wstring line;
    while (std::getline(file, line))
    {
        size_t tab_pos = line.find(L"\t");

        std::wstring en_part = line.substr(0, tab_pos);
        std::wstring es_part = line.substr(tab_pos + 1);

        x.push_back(es_part);
        y.push_back(en_part);
    }

    file.close();

    en_es data;
    data.x = x;
    data.y = y;

    return data;
}