#include "lyrs.h"
#include "preproc.h"

int main()
{
    std::vector<std::wstring> sentences;

    //     x[i] = lower(x[i]);
    // std::wstring sentence1 =
    // std::wstring sentence2 =
    // std::wstring sentence3 =
    // std::wstring sentence4 =
    // std::wstring sentence5 =
    // std::wstring sentence6 =

    sentences.push_back(lower(L"I love machine learning"));
    sentences.push_back(lower(L"I love deep learning"));
    sentences.push_back(lower(L"Machine learning is fascinating"));
    sentences.push_back(lower(L"Deep learning is powerful"));
    sentences.push_back(lower(L"Natural language processing is a part of artificial intelligence"));
    sentences.push_back(lower(L"I enjoy learning new things"));
    // sentences.push_back(L"Standing alone at the top of the stairs, she whips off her cloak to reveal a deep emerald "
    //                     "green dress, sparkly black translucent gloves, soft makeup and loose red waves of hair
    //                     that’s " "parted gently down her shoulder.");

    auto vec_sentences = text_vectorization(sentences, sentences);

    std::cout << vec_sentences << std::endl;

    return 0;
}