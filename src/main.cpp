#include "datas.h"
#include "preproc.h"

#include <iostream>
#include <random>

int main()
{
    // en_es data = load_en_es();

    // std::vector<std::wstring> x(data.x.size());
    // std::vector<std::wstring> y(data.y.size());

    // for (auto i = 0; i < x.size(); ++i)
    // {
    //     x[i] = regex_replace(data.x[i], L"á", L"a");
    //     x[i] = regex_replace(x[i], L"é", L"e");
    //     x[i] = regex_replace(x[i], L"í", L"i");
    //     x[i] = regex_replace(x[i], L"ó", L"o");
    //     x[i] = regex_replace(x[i], L"ú", L"u");

    //     x[i] = regex_replace(x[i], L"Á", L"A");
    //     x[i] = regex_replace(x[i], L"É", L"E");
    //     x[i] = regex_replace(x[i], L"Í", L"I");
    //     x[i] = regex_replace(x[i], L"Ó", L"O");
    //     x[i] = regex_replace(x[i], L"Ú", L"U");

    //     x[i] = regex_replace(x[i], L"ñ", L"n");
    //     x[i] = regex_replace(x[i], L"Ñ", L"N");
    //     x[i] = regex_replace(x[i], L"ü", L"u");
    //     x[i] = regex_replace(x[i], L"Ü", L"U");

    //     x[i] = lower(x[i]);
    //     y[i] = lower(data.y[i]);

    //     x[i] = regex_replace(x[i], L"[^ a-z.?!,¿]", L"");
    //     y[i] = regex_replace(y[i], L"[^ a-z.?!,¿]", L"");

    //     x[i] = regex_replace(x[i], L"([.?!,¿])", L" $1 ");
    //     y[i] = regex_replace(y[i], L"([.?!,¿])", L" $1 ");

    //     x[i] = strip(x[i]);
    //     y[i] = strip(y[i]);

    //     x[i] = join({L"[START]", x[i], L"[END]"}, L" ");
    //     y[i] = join({L"[START]", y[i], L"[END]"}, L" ");
    // }

    // auto vec_x = text_vectorization(x, x);
    // auto vec_y = text_vectorization(y, y);

    // std::cout << vec_x << std::endl;
    // std::cout << vec_y << std::endl;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<float> dist(0.0f, 0.05f);
    std::normal_distribution<float> dist2(0.0f, 0.05f);

    std::cout << dist(rng) << std::endl;
    std::cout << dist2(rng) << std::endl;

    return 0;
}