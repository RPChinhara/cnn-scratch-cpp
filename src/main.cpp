#include "enes.h"
#include "imdb.h"
#include "preproc.h"

#include <iostream>

int main()
{
    en_es data = load_en_es();

    std::vector<std::wstring> x(data.x.size());
    std::vector<std::wstring> y(data.y.size());

    for (auto i = 0; i < 10; ++i)
    {
        x[i] = regex_replace(data.x[i], L"(á)", L"a");
        x[i] = regex_replace(x[i], L"(é)", L"e");
        x[i] = regex_replace(x[i], L"(í)", L"i");
        x[i] = regex_replace(x[i], L"(ó)", L"o");
        x[i] = regex_replace(x[i], L"(ú)", L"u");

        x[i] = regex_replace(x[i], L"(Á)", L"A");
        x[i] = regex_replace(x[i], L"(É)", L"E");
        x[i] = regex_replace(x[i], L"(Í)", L"I");
        x[i] = regex_replace(x[i], L"(Ó)", L"O");
        x[i] = regex_replace(x[i], L"(Ú)", L"U");

        x[i] = lower(x[i]);
        y[i] = lower(data.y[i]);

        x[i] = regex_replace(x[i], L"([^ a-z.?!,¿])", L"");
        y[i] = regex_replace(y[i], L"([^ a-z.?!,¿])", L"");

        x[i] = regex_replace(x[i], L"([.?!,¿])", L" $1 ");
        y[i] = regex_replace(y[i], L"([.?!,¿])", L" $1 ");

        x[i] = strip(x[i]);
        y[i] = strip(y[i]);

        x[i] = join({L"[START]", x[i], L"[END]"}, L" ");
        y[i] = join({L"[START]", y[i], L"[END]"}, L" ");

        std::wcout << y[i] << " " << x[i] << std::endl;
    }

    std::vector<std::wstring> foo = {L"size suit of the vocab point bag card no win device egg hell kelvin",
                                     L"adapt the layer to the text"};
    auto z = text_vectorization(foo);

    // auto z = text_vectorization({
    //     y[0],
    //     y[1],
    //     y[2],
    //     y[3],
    //     y[4],
    //     y[5],
    //     y[6],
    //     y[7],
    //     y[8],
    //     y[9],
    // });

    return 0;
}