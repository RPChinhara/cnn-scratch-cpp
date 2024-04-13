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
        x = wregex_replace(data.x[i], L"(á)", L"a");
        x = wregex_replace(x, L"(é)", L"e");
        x = wregex_replace(x, L"(í)", L"i");
        x = wregex_replace(x, L"(ó)", L"o");
        x = wregex_replace(x, L"(ú)", L"u");

        x = wregex_replace(x, L"(Á)", L"A");
        x = wregex_replace(x, L"(É)", L"E");
        x = wregex_replace(x, L"(Í)", L"I");
        x = wregex_replace(x, L"(Ó)", L"O");
        x = wregex_replace(x, L"(Ú)", L"U");

        x = to_lower_w(x);
        y = to_lower_w(data.y[i]);

        x = wregex_replace(x, L"([^ a-z.?!,¿])", L"");
        y = wregex_replace(y, L"([^ a-z.?!,¿])", L"");

        x = wregex_replace(x, L"([.?!,¿])", L" $1 ");
        y = wregex_replace(y, L"([.?!,¿])", L" $1 ");

        x = strip(x);
        y = strip(y);

        x = join({L"[START]", x, L"[END]"}, L" ");
        y = join({L"[START]", y, L"[END]"}, L" ");

        std::wcout << y << " " << x << std::endl;
    }

    return 0;
}