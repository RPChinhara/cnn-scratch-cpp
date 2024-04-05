#include "datas\enes.h"
#include "mdls\trans.h"

#include <iostream>
#include <windows.h>

int main()
{
    SetConsoleOutputCP(CP_UTF8);

    EnEs en_es = load_en_es();

    for (int i = 0; i < en_es.targetRaw.size(); ++i)
        std::cout << en_es.targetRaw[i] << " " << en_es.contextRaw[i] << std::endl;

    return 0;
}