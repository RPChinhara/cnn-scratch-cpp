#include "datasets\englishspanish.h"
#include "models\transformer.h"

#include <iostream>
#include <windows.h>

int main()
{
    SetConsoleOutputCP(CP_UTF8);

    EnglishSpanish englishSpanish = LoadEnglishSpanish();

    for (int i = 0; i < englishSpanish.targetRaw.size(); ++i)
        std::cout << englishSpanish.targetRaw[i] << " " << englishSpanish.contextRaw[i] << std::endl;

    Transformer transformer = Transformer();

    return 0;
}