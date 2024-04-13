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
        x = regex_replace_wstring(x, L"(é)", L"e");
        x = regex_replace_wstring(x, L"(í)", L"i");
        x = regex_replace_wstring(x, L"(ó)", L"o");
        x = regex_replace_wstring(x, L"(ú)", L"u");

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

        x = strip(x);
        y = strip(y);

        x = join({L"[START]", x, L"[END]"}, L" ");
        y = join({L"[START]", y, L"[END]"}, L" ");

        std::wcout << y << " " << x << std::endl;
    }

    return 0;
}