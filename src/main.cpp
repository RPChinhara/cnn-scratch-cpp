#include "datas.h"
#include "lyrs.h"

int main()
{
    auto data = load_imdb();

    std::vector<std::string> vocab = {"foo bar", "bar baz", "baz bada boom"};
    std::vector<std::string> x = {"foo qux bar", "qux baz"};

    auto vec_y = text_vectorization(vocab, x, 7);

    std::cout << vec_y << std::endl;

    return 0;
}