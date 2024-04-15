#include "datas/enes.h"
#include "datas/imdb.h"
#include "preproc.h"

#include <iostream>
#include <windows.h>

int main()
{
    en_es data = load_en_es();

    // std::wstring x;
    // std::wstring y;

    std::vector<std::wstring> x(data.x.size());
    std::vector<std::wstring> y(data.y.size());

    for (int i = 0; i < 10; ++i)
    {
        x[i] = wregex_replace(data.x[i], L"(á)", L"a");
        x[i] = wregex_replace(x[i], L"(é)", L"e");
        x[i] = wregex_replace(x[i], L"(í)", L"i");
        x[i] = wregex_replace(x[i], L"(ó)", L"o");
        x[i] = wregex_replace(x[i], L"(ú)", L"u");

        x[i] = wregex_replace(x[i], L"(Á)", L"A");
        x[i] = wregex_replace(x[i], L"(É)", L"E");
        x[i] = wregex_replace(x[i], L"(Í)", L"I");
        x[i] = wregex_replace(x[i], L"(Ó)", L"O");
        x[i] = wregex_replace(x[i], L"(Ú)", L"U");

        x[i] = wlower(x[i]);
        y[i] = wlower(data.y[i]);

        x[i] = wregex_replace(x[i], L"([^ a-z.?!,¿])", L"");
        y[i] = wregex_replace(y[i], L"([^ a-z.?!,¿])", L"");

        x[i] = wregex_replace(x[i], L"([.?!,¿])", L" $1 ");
        y[i] = wregex_replace(y[i], L"([.?!,¿])", L" $1 ");

        x[i] = wstrip(x[i]);
        y[i] = wstrip(y[i]);

        x[i] = wjoin({L"[START]", x[i], L"[END]"}, L" ");
        y[i] = wjoin({L"[START]", y[i], L"[END]"}, L" ");

        std::wcout << y[i] << " " << x[i] << std::endl;
    }

    // std::vector<std::wstring> sentences = {data.x[0], data.x[1], data.x[1], data.x[2], data.x[0],
    //                                        data.x[0], data.x[1], data.x[0], data.x[0], data.x[2]};

    auto z = text_vectorization({
        y[0],
        y[1],
        y[2],
        y[3],
        y[4],
        y[5],
        y[6],
        y[7],
        y[8],
        y[9],
    });

    for (auto i : z)
        std::cout << i << std::endl;

    return 0;
}