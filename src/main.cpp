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
        x = regex_replace_wstring(data.x[i], L"(á)", L"a");
        x = regex_replace_wstring(data.x[i], L"(é)", L"e");
        x = regex_replace_wstring(data.x[i], L"(í)", L"i");
        x = regex_replace_wstring(data.x[i], L"(ó)", L"o");
        x = regex_replace_wstring(data.x[i], L"(ú)", L"u");

        x = regex_replace_wstring(x, L"(Á)", L"A");
        x = regex_replace_wstring(x, L"(É)", L"E");
        x = regex_replace_wstring(x, L"(Í)", L"I");
        x = regex_replace_wstring(x, L"(Ó)", L"O");
        x = regex_replace_wstring(x, L"(Ú)", L"U");

        x = to_lower_w(x);
        y = to_lower_w(data.y[i]);

        x = regex_replace_wstring(x, L"([^ a-z.?!,¿])", L"");
        y = regex_replace_wstring(y, L"([^ a-z.?!,¿])", L"");

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