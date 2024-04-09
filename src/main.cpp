#include "datas/enes.h"
#include "preproc.h"

#include <fcntl.h>
#include <io.h>
#include <iostream>
#include <locale>
#include <regex>
#include <windows.h>

int main()
{
    // SetConsoleOutputCP(CP_UTF8);
    std::locale::global(std::locale("es_ES.UTF-8"));
    // _setmode(_fileno(stdout), _O_U16TEXT);

    en_es data = load_en_es();

    std::wstring en_es_x;
    std::wstring en_es_y;

    for (int i = 0; i < 20; ++i)
    {
        //     en_es_target = to_lower(en_es.targetRaw[i]);
        //     en_es_context = to_lower(en_es.contextRaw[i]);

        //     en_es_target = regex_replace(en_es_target, "[^ a-z.?!,¿]", "");
        //     en_es_context = regex_replace(en_es_context, "[^ a-z.?!,¿]", "");

        en_es_x = regex_replace_wstring(data.x[i], L"([.?!¡,¿])", L" $1 ");
        en_es_y = regex_replace_wstring(data.y[i], L"([.?!¡,¿])", L" $1 ");

        std::wcout << en_es_y << " " << en_es_x << std::endl;
        //     en_es_target = strip(en_es_target);
        //     en_es_context = strip(en_es_context);

        //     std::vector<std::string> words_target = {"[START]", en_es_target, "[END]"};
        //     std::vector<std::string> words_context = {"[START]", en_es_context, "[END]"};
        //     std::string separator = " ";

        //     // en_es_target = join(words_target, separator);
        //     // en_es_context = join(words_context, separator);
    }

    return 0;
}