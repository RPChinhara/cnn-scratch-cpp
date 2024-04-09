#include "datas/enes.h"
#include "preproc.h"

#include <iostream>
#include <windows.h>

int main()
{
    en_es data = load_en_es();

    std::wstring x;
    std::wstring y;

    for (int i = 0; i < 20; ++i)
    {
        x = to_lower_w(data.x[i]);
        y = to_lower_w(data.y[i]);

        x = regex_replace_wstring(x, L"(¡)", L"");
        y = regex_replace_wstring(y, L"(¡)", L"");

        x = regex_replace_wstring(x, L"([.?!,¿])", L" $1 ");
        y = regex_replace_wstring(y, L"([.?!,¿])", L" $1 ");

        std::wcout << y << " " << x << std::endl;

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